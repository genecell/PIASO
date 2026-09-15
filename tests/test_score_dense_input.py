"""Scoring takes the matrix in whatever container the user has.

Reported from a benchmarking pipeline: `predictCellTypeByMarker` on an AnnData
whose `.X` was a dense ndarray died inside `_precompute_stats` with

    ValueError: setting an array element with a sequence. The requested array
    has an inhomogeneous shape after 1 dimensions.

`.mean(axis=0)` returns `(1, n_genes)` for a sparse matrix or an `np.matrix`
but `(n_genes,)` for an ndarray, so indexing `[0]` turned every per-gene mean
into one scalar and the KD-tree input became ragged. A second failure waited
behind it: the fused Rust kernel calls `.tocsr()`, which an ndarray does not
have. Both are fixed by converting once at the door, so there is one code path
rather than a dense branch nobody exercises.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import anndata as ad

import piaso
from piaso.tools._normalization import _as_scoring_csr, _precompute_stats


GENES = [f"g{i}" for i in range(60)]
SETS = {"A": GENES[:8], "B": GENES[20:32]}


def _counts(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.5, (n, len(GENES))).astype(np.float32)
    X[: n // 2, :8] += rng.poisson(4, (n // 2, 8))
    return X


def _adata(X):
    a = ad.AnnData(X, var=pd.DataFrame(index=GENES))
    a.layers["infog"] = X.copy() if not sp.issparse(X) else X.copy()
    return a


def _rust_available():
    try:
        from piaso._piaso import score_complete  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# the shape bug itself
# ---------------------------------------------------------------------------

def test_precompute_stats_gives_one_row_per_gene_for_every_container():
    X = _counts()
    ref = _precompute_stats(sp.csr_matrix(X), n_nearest_neighbors=5)
    assert ref.shape == (len(GENES), 5)
    for container in (X, np.asmatrix(X), sp.csr_matrix(X), sp.csc_matrix(X)):
        got = _precompute_stats(container, n_nearest_neighbors=5)
        assert got.shape == ref.shape, type(container)


# ---------------------------------------------------------------------------
# end to end: same numbers, whatever the container
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_rust", [False, True])
def test_score_agrees_across_containers(use_rust):
    if use_rust and not _rust_available():
        pytest.skip("piaso._piaso not built")
    X = _counts()
    out = {}
    for name, container in (("dense", X), ("matrix", np.asmatrix(X)),
                            ("csr", sp.csr_matrix(X)), ("csc", sp.csc_matrix(X))):
        s, names, p = piaso.tl.score(
            _adata(container), gene_list=SETS, layer="infog",
            compute_pvalues=True, use_rust=use_rust, n_ctrl_set=20,
            random_seed=7)
        out[name] = (np.asarray(s), list(names), np.asarray(p))
    ref_s, ref_n, ref_p = out["csr"]
    for name, (s, n, p) in out.items():
        assert n == ref_n, name
        assert np.allclose(s, ref_s, rtol=1e-6, atol=1e-8), name
        assert np.allclose(p, ref_p, rtol=1e-6, atol=1e-8), name


def test_predict_cell_type_by_marker_takes_a_dense_matrix():
    """The reported entry point, on the container that used to crash."""
    X = _counts(seed=3, n=240)
    markers = {"typeA": GENES[:8], "typeB": GENES[20:32]}
    res = {}
    for name, container in (("dense", X), ("csr", sp.csr_matrix(X))):
        a = _adata(container)
        piaso.tl.predictCellTypeByMarker(
            a, marker_gene_set=markers, score_layer="infog",
            use_score=True, verbosity=0, key_added="ct",
            smooth_prediction=False)
        res[name] = a.obs["ct"].astype(str).to_numpy()
    assert (res["dense"] == res["csr"]).all()


# ---------------------------------------------------------------------------
# what the door checks
# ---------------------------------------------------------------------------

def test_non_finite_values_are_an_error_not_a_silent_nan():
    X = _counts().astype(np.float64)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        _as_scoring_csr(X, what="adata.X")
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        _as_scoring_csr(sp.csr_matrix(X), what="adata.X")


def test_negative_values_warn_and_still_score():
    """A z-scored matrix is not invalid input — but control genes are matched
    on mean and variance, which say nothing once every gene has mean 0 and
    variance 1, so the user is told which layer to pass instead."""
    X = _counts().astype(np.float64)
    scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    with pytest.warns(UserWarning, match="negative values"):
        out = _as_scoring_csr(scaled, what="adata.X")
    assert sp.issparse(out)
    a = _adata(scaled)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, names, _p = piaso.tl.score(a, gene_list=SETS, layer="infog",
                                      use_rust=False, n_ctrl_set=10)
    assert np.asarray(s).shape == (X.shape[0], 2)


def test_the_conversion_keeps_the_values():
    X = _counts().astype(np.float64)
    assert np.array_equal(_as_scoring_csr(X).toarray(), X)
    assert sp.isspmatrix_csr(_as_scoring_csr(sp.csc_matrix(X)))
