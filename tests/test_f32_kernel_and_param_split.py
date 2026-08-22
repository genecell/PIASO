"""f32 at the kernel boundary, and `chunk_size` reduced to one meaning.

The kernel forms products and accumulates in f64 either way, so an f32 call is
bit-identical whenever the values round-trip exactly through f32 — which is
always true for a cytome, whose layers are stored float32, and usually true for
AnnData. Float64 data that does NOT round-trip keeps the f64 path and its exact
previous results, so nothing changes for anyone.
"""
import inspect
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

import piaso
from piaso.tools._normalization import (
    _f32_lossless,
    _fused_matmul_reduce_dispatch,
)


# ------------------------------------------------------------ the cast rule

def test_float32_input_passes_through_untouched():
    a = np.array([1.5, 2.25], dtype=np.float32)
    out = _f32_lossless(a)
    assert out is a


def test_exactly_representable_float64_is_accepted():
    a = np.array([1.0, 0.5, -2.25, 0.0], dtype=np.float64)
    out = _f32_lossless(a)
    assert out is not None and out.dtype == np.float32


def test_float64_that_would_lose_precision_is_refused():
    a = np.array([1.0 + 2 ** -40, np.pi], dtype=np.float64)
    assert _f32_lossless(a) is None


def test_other_dtypes_are_refused():
    assert _f32_lossless(np.array([1, 2], dtype=np.int32)) is None


def test_dispatch_prefers_f32_when_both_sides_fit():
    f64, f32 = object(), object()
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0], dtype=np.float64)
    fn, av, bv = _fused_matmul_reduce_dispatch(f64, f32, a, b)
    assert fn is f32 and av.dtype == np.float32 and bv.dtype == np.float32


def test_dispatch_falls_back_when_either_side_does_not_fit():
    f64, f32 = object(), object()
    a = np.array([1.0, 2.0], dtype=np.float32)
    bad = np.array([np.pi], dtype=np.float64)
    fn, av, bv = _fused_matmul_reduce_dispatch(f64, f32, a, bad)
    assert fn is f64 and av.dtype == np.float64 and bv.dtype == np.float64


def test_dispatch_falls_back_when_the_extension_has_no_f32_twin():
    """A stale build must keep working, not raise mid-run."""
    f64 = object()
    a = np.array([1.0], dtype=np.float32)
    fn, av, bv = _fused_matmul_reduce_dispatch(f64, None, a, a)
    assert fn is f64 and av.dtype == np.float64 and bv.dtype == np.float64


# --------------------------------------------------- the kernels agree exactly

def _kernel_inputs(seed=0, n_rows=300, n_cols=200, n_sets=6, n_ctrl=20):
    rng = np.random.default_rng(seed)
    A = sparse.random(n_rows, n_cols, density=0.2, format="csr",
                      random_state=seed,
                      data_rvs=lambda s: rng.integers(1, 9, s).astype(np.float32))
    B = sparse.random(n_cols, n_sets * n_ctrl, density=0.05, format="csr",
                      random_state=seed + 1,
                      data_rvs=lambda s: np.ones(s, dtype=np.float32))
    q = np.zeros(n_rows * n_sets, dtype=np.float64)
    return A, B, q, n_sets, n_ctrl


def test_f32_and_f64_entry_points_agree_bit_for_bit():
    piaso_ext = pytest.importorskip("piaso._piaso")
    f64 = piaso_ext.fused_matmul_reduce
    f32 = getattr(piaso_ext, "fused_matmul_reduce_f32", None)
    if f32 is None:
        pytest.skip("extension predates the f32 entry point")
    A, B, q, n_sets, n_ctrl = _kernel_inputs()
    common = (A.indices.astype(np.int32), A.indptr.astype(np.int32),
              A.shape[0], A.shape[1])
    m64, _ = f64(A.data.astype(np.float64), *common,
                 B.data.astype(np.float64), B.indices.astype(np.int32),
                 B.indptr.astype(np.int32), B.shape[1], q,
                 n_sets, n_ctrl, 32, 4, False)
    m32, _ = f32(A.data.astype(np.float32), *common,
                 B.data.astype(np.float32), B.indices.astype(np.int32),
                 B.indptr.astype(np.int32), B.shape[1], q,
                 n_sets, n_ctrl, 32, 4, False)
    assert np.array_equal(m64, m32)


def test_f32_entry_point_is_registered_in_both_kernels_path():
    piaso_ext = pytest.importorskip("piaso._piaso")
    assert hasattr(piaso_ext, "fused_matmul_reduce")
    assert hasattr(piaso_ext, "fused_matmul_reduce_f32")
    # the internal kernels must survive the rebuild
    for sym in ("score_complete", "scan_motifs_fwd"):
        assert hasattr(piaso_ext, sym), sym


# --------------------------------------------- end to end, both backends

def _sets():
    return {f"s{k}": [f"Gene{j}" for j in range(k * 20, k * 20 + 8)]
            for k in range(3)}


def _adata(n=400, g=120, dtype=np.float32, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.poisson(1.2, (n, g)).astype(dtype)
    a = ad.AnnData(X=sparse.csr_matrix(X))
    a.var_names = [f"Gene{j}" for j in range(g)]
    return a


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_taking_the_f32_path_changes_nothing(dtype, monkeypatch):
    """Isolate the dispatch: same data, f32 entry point vs f64 entry point.

    Not "float32 input scores the same as float64 input" — that is false, and
    was false before this change too: `_precompute_stats` derives per-feature
    means and variances from the matrix, so the two dtypes round differently
    and pick different control neighbours. That is upstream of the kernel.
    """
    import piaso.tools._normalization as norm

    a = _adata(dtype=dtype)
    kw = dict(gene_list=_sets(), layer=None, random_seed=1927,
              max_workers=4, verbosity=0)
    with_f32, _, _ = piaso.tl.score(a, **kw)

    monkeypatch.setattr(norm, "_f32_lossless", lambda arr: None)
    forced_f64, _, _ = piaso.tl.score(a, **kw)

    assert np.array_equal(np.asarray(with_f32), np.asarray(forced_f64))


def test_input_dtype_changing_the_neighbours_is_pre_existing():
    """Pinned so nobody 'fixes' it by rounding inside the kernel dispatch."""
    from piaso.tools._normalization import _precompute_stats

    k32 = _precompute_stats(_adata(dtype=np.float32).X, 30, 40)
    k64 = _precompute_stats(_adata(dtype=np.float64).X, 30, 40)
    assert not np.array_equal(k32, k64)


def test_non_representable_float64_still_scores():
    """The f64 path must still be reachable and correct."""
    a = _adata(dtype=np.float64)
    a.X = a.X.multiply(np.pi).tocsr()          # no longer f32-representable
    m, _, _ = piaso.tl.score(a, gene_list=_sets(), layer=None,
                             random_seed=1927, max_workers=2, verbosity=0)
    assert np.isfinite(np.asarray(m)).all()


# ------------------------------------------------- chunk_size has one meaning

def test_score_declares_the_renamed_parameter_and_keeps_the_alias():
    params = inspect.signature(piaso.tl.score).parameters
    assert "fallback_chunk_size" in params
    assert "chunk_size" in params
    assert params["chunk_size"].default is None


def test_the_alias_warns_and_behaves_identically():
    a = _adata()
    kw = dict(gene_list=_sets(), layer=None, random_seed=1927,
              max_workers=2, verbosity=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old, _, _ = piaso.tl.score(a, chunk_size=5000, **kw)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    new, _, _ = piaso.tl.score(a, fallback_chunk_size=5000, **kw)
    assert np.array_equal(np.asarray(old), np.asarray(new))


def test_the_dead_parameter_is_gone_from_the_streaming_path():
    from piaso.tools._normalization import _score_streaming_multi

    assert "chunk_size" not in inspect.signature(_score_streaming_multi).parameters


def test_the_fallback_block_size_is_documented_with_its_cost():
    doc = piaso.tl.score.__doc__
    assert "fallback_chunk_size : int" in doc
    assert "7.3 GB" in doc          # the allocation it sizes, spelled out
    for knob in ("score_chunk_size :", "max_score_chunk_bytes :",
                 "max_score_batch_cache_bytes :"):
        assert knob in doc, knob


def test_the_cache_budget_guidance_is_in_both_docstrings():
    for doc in (piaso.tl.score.__doc__, piaso.tl.runGDR.__doc__):
        assert "10 s per 100 MB" in doc
