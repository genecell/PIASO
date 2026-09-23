"""Per-gene weights for predictCellTypeByMarker.

A marker list treats every gene the same. When one gene is a textbook marker
and the next is a weak, broadly expressed one, that is throwing away the thing
you know. `marker_gene_weights` passes the weights straight through to
`piaso.tl.score(gene_weights=)`, which already supports them.

They are deliberately *not* normalised, per cell type or otherwise. score()
divides by `median(weights) * n_genes` and builds its control sets from the
same weights, so scaling one cell type's weights by any constant cancels --
the tests below pin that at 1e-12 across a 1000x rescale. Normalising would be
a step that provably changes nothing.

What a rescale cannot survive is a median of zero, which turns the divisor into
zero and every score into NaN without a word. score() rejects that instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def adata():
    import anndata as ad
    rng = np.random.default_rng(0)
    n, g = 400, 200
    genes = [f"G{i}" for i in range(g)]
    X = rng.poisson(0.5, (n, g)).astype(np.float32)
    X[:200, :10] += rng.poisson(6, (200, 10))     # cell type A
    X[200:, 10:20] += rng.poisson(6, (200, 10))   # cell type B
    a = ad.AnnData(X=sp.csr_matrix(X),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=genes))
    a.layers["infog"] = a.X.copy()
    return a


MARKERS = {"A": [f"G{i}" for i in range(10)],
           "B": [f"G{i}" for i in range(10, 20)]}
DECREASING = list(np.arange(10, 0, -1.0))


def _predict(adata, key, weights=None, **kw):
    import piaso
    piaso.tl.predictCellTypeByMarker(
        adata, marker_gene_set=MARKERS, marker_gene_weights=weights,
        smooth_prediction=False, key_added=key, verbosity=0, **kw)
    return adata.obs[key].values, adata.obsm[f"{key}_score"]


def test_weights_are_accepted_and_still_call_the_cell_types(adata):
    calls, _ = _predict(adata, "wt", {"A": DECREASING, "B": DECREASING})
    assert (calls[:200] == "A").mean() == 1.0
    assert (calls[200:] == "B").mean() == 1.0


def test_weights_change_the_scores(adata):
    """If weighting made no difference at all, the argument would be a no-op."""
    _, flat = _predict(adata, "unw")
    _, weighted = _predict(adata, "wt", {"A": DECREASING, "B": DECREASING})
    assert not np.allclose(flat, weighted)


def test_rescaling_one_cell_types_weights_changes_nothing(adata):
    """The reason no normalisation is applied, pinned as a test."""
    base = {"A": DECREASING, "B": DECREASING}
    scaled = {"A": list(np.asarray(DECREASING) * 1000.0), "B": DECREASING}
    calls_a, score_a = _predict(adata, "w1", base)
    calls_b, score_b = _predict(adata, "w2", scaled)
    assert np.max(np.abs(score_a - score_b)) < 1e-12
    assert (calls_a == calls_b).all()


def test_dict_dataframe_and_sequence_forms_agree(adata):
    as_dict = {"A": DECREASING, "B": DECREASING}
    as_frame = pd.DataFrame({"A": DECREASING, "B": DECREASING})
    as_seq = [DECREASING, DECREASING]
    _, s1 = _predict(adata, "f1", as_dict)
    _, s2 = _predict(adata, "f2", as_frame)
    _, s3 = _predict(adata, "f3", as_seq)
    np.testing.assert_allclose(s1, s2)
    np.testing.assert_allclose(s1, s3)


# ── what is refused, and why ─────────────────────────────────────────────────

def test_median_zero_weights_are_refused(adata):
    """Zero-weighting more than half a set makes every score NaN, silently."""
    with pytest.raises(ValueError, match="median"):
        _predict(adata, "bad", {"A": [0.0] * 6 + [1.0] * 4, "B": [1.0] * 10})


def test_non_finite_weights_are_refused(adata):
    with pytest.raises(ValueError, match="non-finite"):
        _predict(adata, "bad", {"A": [np.nan] + [1.0] * 9, "B": [1.0] * 10})


def test_length_mismatch_names_both_numbers(adata):
    with pytest.raises(ValueError, match=r"9 weights for 10 genes"):
        _predict(adata, "bad", {"A": [1.0] * 9, "B": [1.0] * 10})


def test_a_missing_cell_type_is_refused_rather_than_defaulted(adata):
    """Silently weighting the unnamed sets at 1.0 would compare a weighted
    score against an unweighted one."""
    with pytest.raises(ValueError, match="missing"):
        _predict(adata, "bad", {"A": [1.0] * 10})


def test_an_unknown_cell_type_is_refused(adata):
    with pytest.raises(ValueError, match="not in marker_gene_set"):
        _predict(adata, "bad",
                 {"A": [1.0] * 10, "B": [1.0] * 10, "C": [1.0] * 10})


def test_scanpy_scoring_refuses_weights_rather_than_dropping_them(adata):
    with pytest.raises(ValueError, match="only supported by score_method='piaso'"):
        _predict(adata, "bad", {"A": DECREASING, "B": DECREASING},
                 score_method="scanpy")


def test_no_weights_is_unchanged(adata):
    """The default path must not move."""
    import piaso
    a = adata
    piaso.tl.predictCellTypeByMarker(a, marker_gene_set=MARKERS,
                                     smooth_prediction=False, key_added="x",
                                     verbosity=0)
    piaso.tl.predictCellTypeByMarker(a, marker_gene_set=MARKERS,
                                     marker_gene_weights=None,
                                     smooth_prediction=False, key_added="y",
                                     verbosity=0)
    np.testing.assert_array_equal(a.obsm["x_score"], a.obsm["y_score"])
