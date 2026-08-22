"""Stage-3 scoring: the kernel's tile, the chunk budget, and the control matrix.

Three changes, all of which must leave the scores untouched:

* the fused kernel was handed the whole chunk as its tile, producing exactly one
  unit of parallel work — so every thread but one idled;
* the rows per scoring call are now derived from a memory budget rather than
  fixed at 1024;
* the control matrix is built as one COO instead of n_sets blocks + hstack, and
  the gene-name alias map is cached on the dataset instead of rebuilt per batch.
"""
import inspect

import numpy as np
import pytest
import scipy.sparse as sparse

import piaso
from piaso.tools._normalization import (
    _SCORE_CHUNK_MAX,
    _SCORE_CHUNK_MIN,
    _SCORE_TILE_MAX,
    _SCORE_TILE_MIN,
    _gene_name_alias_map,
    _kernel_tile_rows,
    _score_chunk_rows,
)

MB = 1024 ** 2


# ----------------------------------------------------------- the kernel tile

def test_tile_is_never_the_whole_chunk():
    """The regression itself: one tile means one working thread."""
    for n_rows, n_threads in ((1024, 20), (4096, 20), (1024, 8), (200_000, 20)):
        assert _kernel_tile_rows(n_rows, n_threads) < n_rows


def test_tile_gives_every_thread_several_units_of_work():
    for n_rows, n_threads in ((1024, 20), (4096, 20), (8192, 8)):
        n_tiles = -(-n_rows // _kernel_tile_rows(n_rows, n_threads))
        assert n_tiles >= 2 * n_threads or n_tiles >= n_rows // _SCORE_TILE_MIN


def test_tile_stays_in_the_measured_band():
    for n_rows in (64, 1024, 4096, 65536):
        for n_threads in (1, 2, 8, 20, 64):
            t = _kernel_tile_rows(n_rows, n_threads)
            assert _SCORE_TILE_MIN <= t <= _SCORE_TILE_MAX


def test_tile_handles_degenerate_input():
    assert _kernel_tile_rows(0, 8) == 1
    assert _kernel_tile_rows(-5, 8) == 1
    assert _kernel_tile_rows(10, 0) >= 1


# ------------------------------------------------------- the chunk budget

def test_chunk_rows_shrink_when_cells_are_denser():
    sparse_cells = _score_chunk_rows(500, 100, 256 * MB)
    dense_cells = _score_chunk_rows(5000, 100, 256 * MB)
    assert sparse_cells > dense_cells


def test_chunk_rows_shrink_when_there_are_more_marker_sets():
    few = _score_chunk_rows(2843, 50, 256 * MB)
    many = _score_chunk_rows(2843, 5000, 256 * MB)
    assert few > many


def test_chunk_rows_grow_with_the_budget():
    small = _score_chunk_rows(2843, 910, 64 * MB)
    large = _score_chunk_rows(2843, 910, 1024 * MB)
    assert large > small


def test_chunk_rows_respect_the_clamp():
    assert _score_chunk_rows(1e9, 1_000_000, 1) == _SCORE_CHUNK_MIN
    assert _score_chunk_rows(1, 1, 10 ** 12) == _SCORE_CHUNK_MAX


def test_chunk_rows_honour_the_budget_it_was_given():
    """The derived size must actually fit the budget it was handed."""
    for nnz, n_sets, budget in ((2843, 910, 256 * MB), (1200, 150, 256 * MB),
                                (900, 40, 128 * MB)):
        rows = _score_chunk_rows(nnz, n_sets, budget)
        if _SCORE_CHUNK_MIN < rows < _SCORE_CHUNK_MAX:
            assert rows * (nnz * 12 + n_sets * 8) <= budget


def test_advis_and_a_sparser_dataset_land_where_measured():
    """Sanity-check the two cases quoted in the discussion."""
    assert 5000 < _score_chunk_rows(2843, 910, 256 * MB) < 8000
    assert _score_chunk_rows(1200, 150, 256 * MB) > 15000


# ---------------------------------------------------- plumbed all the way

def test_the_knobs_are_declared_along_the_whole_path():
    from piaso.tools._normalization import _score_streaming_multi
    from piaso.tools._runGDR import (
        _runGDR_multibatch_cytome,
        _runGDRParallel_cytome,
        calculateScoreParallel,
    )

    for fn in (piaso.tl.score, _score_streaming_multi, calculateScoreParallel,
               piaso.tl.runGDR, _runGDRParallel_cytome, _runGDR_multibatch_cytome):
        params = inspect.signature(fn).parameters
        assert "score_chunk_size" in params, fn.__name__
        assert "max_score_chunk_bytes" in params, fn.__name__


def test_stage3_splits_the_budget_across_its_workers():
    """The budget is a total, not per worker — otherwise n workers use n x it."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "piaso" / "tools" / "_runGDR.py").read_text()
    i = src.index("max_score_chunk_bytes=max(")
    assert "_n_score_workers" in src[i:i + 200]


# ------------------------------------------- the control matrix construction

def _blockwise_control(knn_idx, gene_sets_indices, weights_list, n_features,
                       n_ctrl_set, random_seed):
    """The pre-2026-08 construction, kept here as the reference."""
    blocks = []
    for gs_idx in range(len(gene_sets_indices)):
        rs = np.random.RandomState(random_seed)
        gene_idx = gene_sets_indices[gs_idx]
        weights = weights_list[gs_idx]
        gs_knn = knn_idx[gene_idx]
        n_gs = len(gene_idx)
        rand_idx = rs.randint(0, gs_knn.shape[1], size=(n_gs, n_ctrl_set))
        ctrl = gs_knn[np.arange(n_gs)[:, None], rand_idx].T
        blocks.append(sparse.csr_matrix(
            (np.tile(weights, n_ctrl_set),
             (ctrl.ravel(),
              np.repeat(np.arange(n_ctrl_set, dtype=np.int32), n_gs))),
            shape=(n_features, n_ctrl_set)))
    return sparse.hstack(blocks, format="csr")


def _coo_control(knn_idx, gene_sets_indices, weights_list, n_features,
                 n_ctrl_set, random_seed):
    """The new construction, transcribed from _score_streaming_multi."""
    rows, cols, data = [], [], []
    n_sets = len(gene_sets_indices)
    for gs_idx in range(n_sets):
        rs = np.random.RandomState(random_seed)
        gene_idx = gene_sets_indices[gs_idx]
        weights = weights_list[gs_idx]
        gs_knn = knn_idx[gene_idx]
        n_gs = len(gene_idx)
        rand_idx = rs.randint(0, gs_knn.shape[1], size=(n_gs, n_ctrl_set))
        ctrl = gs_knn[np.arange(n_gs)[:, None], rand_idx].T
        rows.append(ctrl.ravel())
        cols.append(np.repeat(np.arange(n_ctrl_set, dtype=np.int64), n_gs)
                    + gs_idx * n_ctrl_set)
        data.append(np.tile(weights, n_ctrl_set))
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_features, n_sets * n_ctrl_set)).tocsr()


@pytest.mark.parametrize("n_sets,n_ctrl_set", [(3, 5), (17, 25), (40, 100)])
def test_single_coo_control_matches_the_block_construction(n_sets, n_ctrl_set):
    rng = np.random.default_rng(0)
    n_features, n_neighbors = 300, 30
    knn_idx = rng.integers(0, n_features, size=(n_features, n_neighbors))
    gene_sets_indices, weights_list = [], []
    for k in range(n_sets):
        n_gs = int(rng.integers(2, 12))
        gene_sets_indices.append(rng.choice(n_features, n_gs, replace=False))
        weights_list.append(rng.random(n_gs))

    ref = _blockwise_control(knn_idx, gene_sets_indices, weights_list,
                             n_features, n_ctrl_set, 1927)
    new = _coo_control(knn_idx, gene_sets_indices, weights_list,
                       n_features, n_ctrl_set, 1927)
    assert ref.shape == new.shape
    assert (ref != new).nnz == 0, "control matrix changed"
    assert np.array_equal(ref.toarray(), new.toarray())


def test_control_construction_handles_duplicate_control_genes():
    """Duplicates must SUM identically in both constructions, not overwrite."""
    n_features, n_ctrl_set = 20, 4
    # every neighbour is the same gene, so every entry collides
    knn_idx = np.zeros((n_features, 6), dtype=int)
    gene_sets_indices = [np.array([1, 2, 3]), np.array([4, 5])]
    weights_list = [np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.25])]
    ref = _blockwise_control(knn_idx, gene_sets_indices, weights_list,
                             n_features, n_ctrl_set, 7)
    new = _coo_control(knn_idx, gene_sets_indices, weights_list,
                       n_features, n_ctrl_set, 7)
    assert np.array_equal(ref.toarray(), new.toarray())
    assert ref[0, 0] == pytest.approx(6.0)   # 1 + 2 + 3, summed not overwritten


def test_the_blockwise_hstack_is_gone():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "piaso" / "tools" / "_normalization.py").read_text()
    assert "all_ctrl_blocks" not in src


# --------------------------------------------------------- the alias cache

class _FakeDS:
    """Minimal stand-in: the cache must key on the dataset object."""

    def __init__(self):
        self.calls = 0


def test_alias_map_is_cached_per_dataset(monkeypatch):
    import piaso.tools._normalization as norm

    calls = {"n": 0}

    def _fake(ds, modality, var_names):
        calls["n"] += 1
        return {n: [i] for i, n in enumerate(var_names)}

    monkeypatch.setattr(norm, "_gene_name_alias_map_uncached", _fake)
    ds = _FakeDS()
    names = np.array(["a", "b", "c"])
    first = norm._gene_name_alias_map(ds, "RNA", names)
    second = norm._gene_name_alias_map(ds, "RNA", names)
    assert first is second
    assert calls["n"] == 1, "alias map rebuilt for the same dataset"

    other = _FakeDS()
    norm._gene_name_alias_map(other, "RNA", names)
    assert calls["n"] == 2, "cache leaked across datasets"


def test_alias_map_falls_back_when_attributes_are_forbidden(monkeypatch):
    import piaso.tools._normalization as norm

    class _NoAttrs:
        __slots__ = ()

    monkeypatch.setattr(norm, "_gene_name_alias_map_uncached",
                        lambda ds, m, v: {"ok": [0]})
    assert norm._gene_name_alias_map(_NoAttrs(), "RNA", np.array(["x"])) == {"ok": [0]}


# --------------------------------------- end to end: the scores must not move

def _rna_cytome(path, n=900, g=120, seed=0):
    import cytome
    import pandas as pd

    rs = np.random.RandomState(seed)
    X = rs.poisson(1.2, (n, g)).astype(np.float32)
    for k in range(3):
        X[k::3, k * 20:(k + 1) * 20] += rs.poisson(6.0, (len(X[k::3]), 20))
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)],
        "grp": [f"c{i % 3}" for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(g),
        "gene_id": [f"ENSG{i:05d}" for i in range(g)],
        "symbol": [f"Gene{i}" for i in range(g)]}))
    ds.add_matrix("RNA_counts", sparse.csr_matrix(X))
    ds.flush()
    return ds


def _sets():
    return {f"s{k}": [f"Gene{j}" for j in range(k * 20, k * 20 + 8)]
            for k in range(3)}


@pytest.mark.parametrize("max_workers", [1, 4])
def test_chunk_size_is_kernel_neutral_when_pass_one_is_fixed(tmp_path, max_workers):
    """With the neighbours pinned, the chunk size changes nothing.

    It is NOT neutral end to end, and this test used to claim it was: pass 1
    accumulates per-feature sums chunk by chunk, so the chunk boundaries set the
    float64 summation order. On ADVIS, 2,599 vs 3,642 rows moves a score by
    8.2e-2. Passing `precomputed_knn` removes pass 1 from the comparison and
    isolates what the chunk size really controls -- the kernel call size.
    """
    ds = _rna_cytome(tmp_path / f"s{max_workers}.cytome")
    try:
        from piaso.tools._normalization import _precompute_stats

        knn = _precompute_stats(ds.iter_chunks(modality="RNA", layer="counts",
                                               batch_size=4096).__next__()[0],
                                30, 40)
        out = {}
        for chunk in (1024, 128, None, 32768):
            m, _, _ = piaso.tl.score(
                ds, gene_list=_sets(), layer="counts", modality="RNA",
                random_seed=1927, max_workers=max_workers, verbosity=0,
                score_chunk_size=chunk, precomputed_knn=knn,
            )
            out[chunk] = np.asarray(m).copy()
        ref = out[1024]
        for chunk, val in out.items():
            assert val.shape == ref.shape
            assert np.array_equal(val, ref), f"chunk={chunk} changed the scores"
    finally:
        ds.close()


def test_chunk_size_is_documented_as_affecting_pass_one():
    """The trap must be written down where the parameter is defined."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "piaso" / "tools" / "_normalization.py").read_text()
    i = src.index("def _score_chunk_rows(")
    assert "summation order" in src[i:i + 2000]


def test_scores_are_bit_identical_across_thread_counts(tmp_path):
    """The tile changed how work is handed out; it must not change the answer."""
    ds = _rna_cytome(tmp_path / "t.cytome")
    try:
        ref = None
        for nt in (1, 2, 8, 20):
            m, _, _ = piaso.tl.score(
                ds, gene_list=_sets(), layer="counts", modality="RNA",
                random_seed=1927, max_workers=nt, verbosity=0,
            )
            m = np.asarray(m)
            if ref is None:
                ref = m
            else:
                assert np.array_equal(m, ref), f"max_workers={nt} changed the scores"
    finally:
        ds.close()


def test_derived_chunk_is_reported_when_verbose(tmp_path, capsys):
    ds = _rna_cytome(tmp_path / "v.cytome")
    try:
        piaso.tl.score(ds, gene_list=_sets(), layer="counts", modality="RNA",
                       random_seed=1927, max_workers=1, verbosity=1)
        assert "Scoring chunk:" in capsys.readouterr().out
    finally:
        ds.close()


def test_the_kernel_is_never_handed_the_chunk_as_its_tile():
    """Source-level guard for the original defect.

    ``fused_matmul_reduce(..., n_sets, n_ctrl_set, TILE, n_threads, pvals)`` —
    passing n_chunk or n_cells in the TILE slot means one tile, one working
    thread, and a stage that looks bandwidth-bound.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "piaso" / "tools" / "_normalization.py").read_text()
    calls = re.findall(r"n_sets, n_ctrl_set,\s*(?:#[^\n]*\n\s*)*"
                       r"(?:#[^\n]*\n\s*)*([^\n,]+),", src)
    assert calls, "could not find the kernel call sites"
    for arg in calls:
        arg = arg.strip()
        assert arg.startswith("_kernel_tile_rows("), f"tile slot holds {arg!r}"
