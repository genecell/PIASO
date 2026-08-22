"""The pass-1 fixed block: score_chunk_size is now genuinely output-neutral.

Pass 1 accumulates per-feature float64 sums, so whatever block size feeds it
sets their summation order. Historically that was the pass-2 scoring chunk,
which made a tuning knob (score_chunk_size / max_score_chunk_bytes) perturb
scores by up to ~8e-2. Pass 1 now reads at the fixed _PASS1_BLOCK_ROWS and
the pass-1<->2 cache holds blocks at that size (pass 2 glues them back up to
the scoring chunk, which never touched results). These tests pin the
property the decoupling buys: identical bits for any knob setting.
"""
import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

import piaso
from piaso.tools._normalization import _PASS1_BLOCK_ROWS, score

cytome = pytest.importorskip("cytome")

MB = 1024 ** 2


def _sets():
    return {f"s{k}": [f"Gene{j}" for j in range(k * 20, k * 20 + 8)]
            for k in range(3)}


def _cytome(path, n=9000, g=120, seed=0):
    """Bigger than one _PASS1_BLOCK_ROWS block so the block boundary is real."""
    rs = np.random.RandomState(seed)
    X = rs.poisson(1.2, (n, g)).astype(np.float32)
    for k in range(3):
        X[k::3, k * 20:(k + 1) * 20] += rs.poisson(6.0, (len(X[k::3]), 20))
    # Non-integer values: integer-valued floats sum exactly in any order, so
    # they cannot detect a summation-order change. Real normalised layers
    # (the infog layer this path scores in production) are non-integer.
    X *= rs.uniform(0.5, 1.5, size=(n, 1)).astype(np.float32)
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


def _score(ds, **kw):
    m, _, _ = piaso.tl.score(ds, gene_list=_sets(), layer="counts",
                             modality="RNA", random_seed=1927,
                             max_workers=1, verbosity=0, **kw)
    return np.asarray(m)


def test_score_chunk_size_is_output_neutral(tmp_path):
    """The decoupling's contract: any pass-2 chunk size, identical bits."""
    ds = _cytome(tmp_path / "a.cytome")
    try:
        ref = _score(ds)
        for rows in (257, 1024, 5000, 20000):
            got = _score(ds, score_chunk_size=rows)
            assert np.array_equal(got, ref), f"differs at score_chunk_size={rows}"
    finally:
        ds.close()


def test_budget_knob_is_output_neutral(tmp_path):
    ds = _cytome(tmp_path / "b.cytome")
    try:
        ref = _score(ds)
        got = _score(ds, max_score_chunk_bytes=8 * MB)
        assert np.array_equal(got, ref)
    finally:
        ds.close()


def test_cache_on_off_identical_across_the_block_boundary(tmp_path):
    ds = _cytome(tmp_path / "c.cytome")
    try:
        on = _score(ds, max_score_batch_cache_bytes=1024 * MB)
        off = _score(ds, max_score_batch_cache_bytes=0)
        assert np.array_equal(on, off)
    finally:
        ds.close()


def test_pass1_block_is_a_fixed_constant():
    assert _PASS1_BLOCK_ROWS == 4096


def test_cache_default_is_1gb():
    sig = inspect.signature(score)
    assert sig.parameters["max_score_batch_cache_bytes"].default == 1024 * MB
