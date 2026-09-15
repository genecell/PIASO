"""piaso.tl.umap / leiden write, and return nothing.

Before 1.2.4 the AnnData path returned the embedding (or the labels) *and*
wrote them, while the cytome path returned None -- one function with two
contracts, and a 161k-row array echoed into every notebook. The only calls
that still return a value are the ones with nowhere to write: ``data=None``
and, for leiden, ``cell_mask``.

Also covers the GDR batch helper, which exists so that no anndata view (and
therefore no thread-unsafe warning filter) is needed per batch.
"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

import piaso
from piaso.tools._runGDR import _batch_anndata, calculateScoreParallel_multiBatch


def _umap_is_usable() -> bool:
    """umap-learn calls ``check_array(..., force_all_finite=)``, which recent
    scikit-learn removed. Where that pairing is installed every UMAP call
    raises TypeError before reaching our code, so the return-value contract
    cannot be exercised -- skip rather than report our own failure."""
    try:
        import umap
        umap.UMAP(n_neighbors=3, n_components=2, random_state=0).fit_transform(
            np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]))
        return True
    except Exception:
        return False


needs_umap = pytest.mark.skipif(
    not _umap_is_usable(),
    reason="installed umap-learn is incompatible with the installed scikit-learn")


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    # enough genes for the scorer's KDTree over the matched-gene space
    a = ad.AnnData(X=rng.poisson(3.0, size=(60, 90)).astype(np.float32))
    a.var_names = [f"G{i}" for i in range(90)]
    a.obs_names = [f"c{i}" for i in range(60)]
    a.obs["batch"] = pd.Categorical(np.array(["b0", "b1", "b2"] * 20))
    a.layers["infog"] = a.X.copy()
    a.obsm["X_rep"] = rng.standard_normal((60, 5))
    return a


# ------------------------------------------------------- return contract --


def test_leiden_writes_and_returns_none(small):
    piaso.tl.neighbors(small, use_rep="X_rep", n_neighbors=8, random_state=0)
    assert piaso.tl.leiden(small, resolution=1.0, random_state=0,
                           key_added="lab") is None
    assert "lab" in small.obs
    assert len(small.obs["lab"]) == small.n_obs


@needs_umap
def test_umap_writes_and_returns_none(small):
    piaso.tl.neighbors(small, use_rep="X_rep", n_neighbors=8, random_state=0)
    assert piaso.tl.umap(small, key_added="X_u", random_state=0) is None
    assert small.obsm["X_u"].shape == (small.n_obs, 2)


def test_in_memory_leiden_still_returns_because_it_cannot_write():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 6))
    knn = piaso.tl.neighbors(X, n_neighbors=5, random_state=0)
    labels = piaso.tl.leiden(data=None, knn_result=knn, resolution=1.0,
                             random_state=0)
    assert isinstance(labels, np.ndarray) and labels.shape[0] == 40


@needs_umap
def test_in_memory_umap_still_returns_because_it_cannot_write():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 6))
    knn = piaso.tl.neighbors(X, n_neighbors=5, random_state=0)
    emb = piaso.tl.umap(data=None, knn_result=knn, random_state=0)
    assert emb.shape == (40, 2)


def test_leiden_cell_mask_still_returns_labels():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 6))
    knn = piaso.tl.neighbors(X, n_neighbors=5, random_state=0)
    mask = np.ones(40, dtype=bool)
    mask[:15] = False
    labels = piaso.tl.leiden(data=None, knn_result=knn, resolution=1.0,
                             random_state=0, cell_mask=mask)
    assert labels.shape[0] == 25


# ------------------------------------------------------ GDR batch helper --


def test_batch_anndata_matches_a_view_copy_without_making_one(small):
    mask = (small.obs["batch"] == "b1").values
    ref = small[mask].copy()
    got = _batch_anndata(small, mask, None)
    np.testing.assert_allclose(np.asarray(got.X), np.asarray(ref.X))
    assert list(got.obs_names) == list(ref.obs_names)
    assert list(got.var_names) == list(ref.var_names)
    assert got.obs.shape[1] == 0        # no categoricals to rewrite


def test_batch_anndata_carries_the_named_layer(small):
    mask = (small.obs["batch"] == "b0").values
    got = _batch_anndata(small, mask, "infog")
    np.testing.assert_allclose(np.asarray(got.layers["infog"]),
                               np.asarray(small.layers["infog"])[mask])


def test_batch_anndata_raises_on_a_missing_layer(small):
    with pytest.raises(KeyError):
        _batch_anndata(small, np.ones(small.n_obs, dtype=bool), "nope")


class _NeverRaised(Warning):
    """Stand-in for a warning class pandas no longer defines."""


def test_multibatch_scoring_is_warning_free(small):
    """The SettingWithCopyWarning came from anndata's view constructor inside
    the per-batch scoring worker; with no view there is nothing to suppress,
    and nothing for a thread-unsafe filter to leak."""
    import warnings

    # pandas 3 removed SettingWithCopyWarning along with the copy-on-write
    # machinery that raised it. On 3.x there is no such class and nothing can
    # emit it, so the assertion below is vacuously true rather than skippable
    # -- the property still holds, it just cannot fail that way any more.
    SettingWithCopyWarning = getattr(
        pd.errors, "SettingWithCopyWarning", None) or _NeverRaised

    marker = pd.DataFrame({"s0": ["G0", "G1", "G2"], "s1": ["G3", "G4", "G5"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores, barcodes, names = calculateScoreParallel_multiBatch(
            small, batch_key="batch", marker_gene=marker,
            marker_gene_n_groups_indices=[0, 2], score_method="piaso",
            score_layer="infog", max_workers=2, random_seed=0)
    assert not [w for w in caught
                if issubclass(w.category, SettingWithCopyWarning)]
    # every cell scored exactly once, across the three batches
    assert sum(s.shape[0] for s in scores) == small.n_obs
    assert sorted(np.concatenate(barcodes)) == sorted(small.obs_names)
