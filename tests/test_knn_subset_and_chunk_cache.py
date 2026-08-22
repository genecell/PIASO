"""Four stage-3 changes, and which backend each one belongs to.

| change | AnnData | Cytome |
|---|---|---|
| `score_complete`'s kernel tile | yes (only path that calls it) | n/a |
| the tile rule itself | shared helper | shared helper |
| k-NN subset query | yes | yes |
| pass-1 -> pass-2 chunk cache | n/a (matrix is already in memory) | yes |

Every one of them must leave the scores bit-identical.
"""
import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse
from sklearn.neighbors import KDTree

import piaso
from piaso.tools._normalization import (
    _SCORE_TILE_MAX,
    _SCORE_TILE_MIN,
    _gene_list_feature_indices,
    _gene_set_query_rows,
    _kernel_tile_rows,
    _knn_from_mean_var,
    _precompute_stats,
)

MB = 1024 ** 2


# ------------------------------------------------ the tile rule (both backends)

def test_tile_floor_is_16_and_divisor_is_4():
    """The measured grid: a floor of 32 leaves 8 tiles for 20-40 threads."""
    assert (_SCORE_TILE_MIN, _SCORE_TILE_MAX) == (16, 128)
    assert _kernel_tile_rows(256, 20) == 16
    assert _kernel_tile_rows(256, 40) == 16
    assert _kernel_tile_rows(1024, 20) == 16
    assert _kernel_tile_rows(8192, 1) == 128


def test_tile_gives_many_units_of_work_to_many_threads():
    for n_rows, n_threads in ((256, 20), (256, 40), (1024, 20), (4096, 40)):
        n_tiles = -(-n_rows // _kernel_tile_rows(n_rows, n_threads))
        assert n_tiles >= min(n_threads, n_rows // _SCORE_TILE_MIN)


def test_every_kernel_call_site_derives_its_tile():
    """Three call sites now: streaming fused, in-memory fused, score_complete."""
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "piaso" / "tools" / "_normalization.py").read_text()
    assert src.count("_kernel_tile_rows(") >= 4        # 1 def + 3 call sites
    # score_complete used to receive the raw chunk_size
    assert "n_ctrl_set, random_seed, chunk_size," not in src


def test_chunk_size_still_means_memory_in_the_python_fallback():
    """chunk_size has two jobs; only the Rust tile changed.

    The fallback densifies (chunk_size, n_sets*n_ctrl_set); turning that into a
    16-row tile would be a 500x slowdown of the fallback, not a fix.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "piaso" / "tools" / "_normalization.py").read_text()
    assert "for c_start in range(0, n_cells, chunk_size):" in src


# ------------------------------------------- the k-NN subset (both backends)

def _mean_var(seed=0, n=400):
    rng = np.random.default_rng(seed)
    mv = rng.random((n, 2))
    mv[:60] = 0.0                      # a tie group, like zero-count features
    return mv


def test_subset_query_matches_the_full_query_on_the_rows_it_computes():
    mv = _mean_var()
    full = _knn_from_mean_var(mv, 30, 40, query_rows=None)
    rows = np.array([3, 7, 11, 100, 250, 399])
    sub = _knn_from_mean_var(mv, 30, 40, query_rows=rows)
    assert sub.shape == full.shape
    assert np.array_equal(sub[rows], full[rows])


def test_uncomputed_rows_are_zero_and_the_shape_is_full_height():
    """Rust takes the flattened array and indexes it by global feature id."""
    mv = _mean_var()
    rows = np.array([5, 6])
    sub = _knn_from_mean_var(mv, 10, 40, query_rows=rows)
    assert sub.shape == (mv.shape[0], 10)
    untouched = np.setdiff1d(np.arange(mv.shape[0]), rows)
    assert not sub[untouched].any()


def test_self_is_removed_using_the_global_row_id():
    """The subtle one: with a subset, self is `query_rows[i]`, not `i`."""
    mv = _mean_var()
    rows = np.array([120, 300])
    sub = _knn_from_mean_var(mv, 30, 40, query_rows=rows)
    for r in rows:
        assert r not in sub[r], "a feature was its own control neighbour"


def test_subset_matches_a_plain_reference_implementation():
    """Guards the shared helper against the hand-rolled loops it replaced."""
    mv = _mean_var(seed=3, n=300)
    k = 15
    raw = KDTree(mv, leaf_size=40).query(mv, k=k + 1, return_distance=False)
    ref = np.zeros((mv.shape[0], k), dtype=np.int64)
    for i in range(mv.shape[0]):
        ref[i] = np.array([j for j in raw[i] if j != i][:k])
    got = _knn_from_mean_var(mv, k, 40, query_rows=None)
    assert np.array_equal(got, ref)


def test_query_rows_helper_unions_the_sets_and_bails_when_almost_everything():
    assert _gene_set_query_rows([[1, 2], [2, 3]], 100).tolist() == [1, 2, 3]
    assert _gene_set_query_rows([], 100) is None
    assert _gene_set_query_rows([[]], 100) is None
    # covering >75% of features: bookkeeping costs more than it saves
    assert _gene_set_query_rows([list(range(80))], 100) is None


@pytest.mark.parametrize("shape", ["dict", "frame", "list_of_lists", "flat"])
def test_gene_list_indices_handles_every_accepted_shape(shape):
    name_to_idx = {"a": 0, "b": 1, "c": 2}
    payload = {
        "dict": {"s1": ["a", "b"], "s2": ["c", "zzz"]},
        "frame": pd.DataFrame({"s1": ["a", "b"], "s2": ["c", None]}),
        "list_of_lists": [["a", "b"], ["c"]],
        "flat": ["a", "b", "c"],
    }[shape]
    got = _gene_list_feature_indices(payload, name_to_idx)
    assert sorted({i for grp in got for i in grp}) == [0, 1, 2]


def test_alias_map_values_that_are_lists_are_expanded():
    """Cytome's alias map maps a name to ALL its rows (duplicate symbols)."""
    got = _gene_list_feature_indices({"s": ["dup"]}, {"dup": [4, 9]})
    assert sorted(got[0]) == [4, 9]


def test_precompute_stats_accepts_query_rows_on_the_anndata_path():
    assert "query_rows" in inspect.signature(_precompute_stats).parameters
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(1.0, (50, 120)).astype(np.float32))
    full = _precompute_stats(X, 20, 40)
    rows = np.array([1, 40, 119])
    sub = _precompute_stats(X, 20, 40, query_rows=rows)
    assert np.array_equal(sub[rows], full[rows])


# ------------------------------------ end to end: AnnData and cytome unchanged

def _sets(n_genes=120):
    return {f"s{k}": [f"Gene{j}" for j in range(k * 20, k * 20 + 8)]
            for k in range(3)}


def _adata(n=400, g=120, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.poisson(1.2, (n, g)).astype(np.float32)
    for k in range(3):
        X[k::3, k * 20:(k + 1) * 20] += rs.poisson(6.0, (len(X[k::3]), 20))
    a = ad.AnnData(X=sparse.csr_matrix(X))
    a.var_names = [f"Gene{j}" for j in range(g)]
    a.obs["grp"] = pd.Categorical([f"c{i % 3}" for i in range(n)])
    return a


def _cytome(path, n=900, g=120, seed=0):
    import cytome

    rs = np.random.RandomState(seed)
    X = rs.poisson(1.2, (n, g)).astype(np.float32)
    for k in range(3):
        X[k::3, k * 20:(k + 1) * 20] += rs.poisson(6.0, (len(X[k::3]), 20))
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(g),
        "gene_id": [f"ENSG{i:05d}" for i in range(g)],
        "symbol": [f"Gene{i}" for i in range(g)]}))
    ds.add_matrix("RNA_counts", sparse.csr_matrix(X))
    ds.flush()
    return ds


@pytest.mark.parametrize("max_workers", [1, 4, 20])
def test_anndata_scores_are_stable_across_thread_counts(max_workers):
    """Covers score_complete's tile: it is the AnnData fast path."""
    a = _adata()
    m, _, _ = piaso.tl.score(a, gene_list=_sets(), layer=None,
                             random_seed=1927, max_workers=max_workers,
                             verbosity=0)
    ref, _, _ = piaso.tl.score(a, gene_list=_sets(), layer=None,
                               random_seed=1927, max_workers=1, verbosity=0)
    assert np.array_equal(np.asarray(m), np.asarray(ref))


def test_anndata_scores_ignore_chunk_size():
    """chunk_size no longer reaches the tile, so it must not move the answer."""
    a = _adata()
    out = []
    for cs in (10000, 64, 250000):
        m, _, _ = piaso.tl.score(a, gene_list=_sets(), layer=None,
                                 random_seed=1927, max_workers=4,
                                 chunk_size=cs, verbosity=0)
        out.append(np.asarray(m))
    for m in out[1:]:
        assert np.array_equal(m, out[0])


@pytest.mark.parametrize("cache_bytes", [0, 512 * MB])
def test_cytome_scores_are_identical_with_and_without_the_chunk_cache(
        tmp_path, cache_bytes):
    ds = _cytome(tmp_path / f"c{cache_bytes}.cytome")
    try:
        got, _, _ = piaso.tl.score(
            ds, gene_list=_sets(), layer="counts", modality="RNA",
            random_seed=1927, max_workers=4, verbosity=0,
            max_score_batch_cache_bytes=cache_bytes)
        ref, _, _ = piaso.tl.score(
            ds, gene_list=_sets(), layer="counts", modality="RNA",
            random_seed=1927, max_workers=4, verbosity=0,
            max_score_batch_cache_bytes=0)
        assert np.array_equal(np.asarray(got), np.asarray(ref))
    finally:
        ds.close()


def test_cytome_cache_is_reported_and_declines_when_over_budget(tmp_path, capsys):
    ds = _cytome(tmp_path / "rep.cytome")
    try:
        piaso.tl.score(ds, gene_list=_sets(), layer="counts", modality="RNA",
                       random_seed=1927, max_workers=1, verbosity=1,
                       max_score_batch_cache_bytes=1)
        assert "chunk cache: off" in capsys.readouterr().out
        piaso.tl.score(ds, gene_list=_sets(), layer="counts", modality="RNA",
                       random_seed=1927, max_workers=1, verbosity=1,
                       max_score_batch_cache_bytes=512 * MB)
        assert "chunk cache: on" in capsys.readouterr().out
    finally:
        ds.close()


def test_cytome_cell_mask_still_scores_the_right_rows_with_the_cache(tmp_path):
    """The cache holds pre-mask chunks; masking must still happen in pass 2."""
    ds = _cytome(tmp_path / "m.cytome")
    try:
        mask = np.zeros(900, dtype=bool)
        mask[100:400] = True
        got, _, _ = piaso.tl.score(
            ds, gene_list=_sets(), layer="counts", modality="RNA",
            random_seed=1927, max_workers=2, verbosity=0, cell_mask=mask,
            max_score_batch_cache_bytes=512 * MB)
        ref, _, _ = piaso.tl.score(
            ds, gene_list=_sets(), layer="counts", modality="RNA",
            random_seed=1927, max_workers=2, verbosity=0, cell_mask=mask,
            max_score_batch_cache_bytes=0)
        assert np.asarray(got).shape[0] == 300
        assert np.array_equal(np.asarray(got), np.asarray(ref))
    finally:
        ds.close()


def test_precomputed_knn_skips_pass_one_without_tripping_the_cache(tmp_path):
    """precomputed_knn means pass 1 never runs, so the cache stays empty."""
    ds = _cytome(tmp_path / "pk.cytome")
    try:
        ref, _, _ = piaso.tl.score(
            ds, gene_list=_sets(), layer="counts", modality="RNA",
            random_seed=1927, max_workers=1, verbosity=0)
        rng = np.random.default_rng(0)
        knn = rng.integers(0, 120, size=(120, 30))
        got, _, _ = piaso.tl.score(
            ds, gene_list=_sets(), layer="counts", modality="RNA",
            random_seed=1927, max_workers=1, verbosity=0,
            precomputed_knn=knn, max_score_batch_cache_bytes=512 * MB)
        assert np.asarray(got).shape == np.asarray(ref).shape
    finally:
        ds.close()


def test_the_cache_knob_is_cytome_only_and_declared_along_its_path():
    from piaso.tools._normalization import _score_streaming_multi
    from piaso.tools._runGDR import (
        _runGDR_multibatch_cytome,
        _runGDRParallel_cytome,
        calculateScoreParallel,
    )

    for fn in (piaso.tl.score, _score_streaming_multi, calculateScoreParallel,
               piaso.tl.runGDR, _runGDRParallel_cytome, _runGDR_multibatch_cytome):
        assert "max_score_batch_cache_bytes" in inspect.signature(fn).parameters, \
            fn.__name__
    # the AnnData stats function has no second read to cache
    assert "max_score_batch_cache_bytes" not in inspect.signature(
        _precompute_stats).parameters
