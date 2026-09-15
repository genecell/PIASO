"""The leiden case-collision sequence, and umap's use_rep contract.

Companion to cytome's test_case_insensitive_columns.py: that file pins the
storage layer; this one pins the two piaso callers whose silences made the
bug invisible — leiden's swallowed set_categories, and umap's use_rep that
either did nothing quietly (graph reuse) or did everything quietly (the
no-stored-kNN fallback).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import cytome


def _ds_with_leiden(path, n=8):
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n),
        "barcode": [f"BC{i}" for i in range(n)],
        "Leiden": [str(i % 2) for i in range(n)],
    }))
    ds.flush()
    return ds


def test_leiden_write_sequence_replaces_stale_categories(tmp_path):
    """The exact user scenario: 'Leiden' exists with stored categories, a new
    clustering is written as 'leiden'. Ends with the FRESH order on the
    canonical column — the pre-fix behaviour left the stale one."""
    ds = _ds_with_leiden(tmp_path / "t.cytome")
    ds.set_categories("Leiden", order=["0", "1"], colors=["#4E79A7", "#E69F00"])

    labels = np.array([str(i % 3) for i in range(8)])
    order = ["0", "1", "2"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the write+persist sequence _leiden.py runs after clustering
        ds.cells["leiden"] = labels
        ds.flush()
        ds.set_categories("leiden", order=order)

    got = ds.get_categories("Leiden")
    assert got["order"] == order, (
        "stale categories survived the overwrite — the plot-time warning is "
        "about to fire two steps from the cause again")
    assert list(ds.cells["leiden"]) == list(labels)


def test_leiden_does_not_swallow_set_categories_failures(monkeypatch, tmp_path):
    """_leiden.py's `except Exception: pass` was load-bearing for the bug; it
    is now a warning that names the failure."""
    from piaso.tools import _leiden as L

    ds = _ds_with_leiden(tmp_path / "t.cytome")

    def boom(*a, **k):
        raise RuntimeError("simulated categories failure")

    monkeypatch.setattr(type(ds), "set_categories", boom)
    knn = {"connectivities": _tiny_graph(8)}
    with pytest.warns(UserWarning, match="category order could not be stored"):
        L.leiden(ds, knn_result=knn, key_added="leiden", resolution=1.0)
    # the labels themselves must still have been written
    assert "leiden" in ds.cells


def _tiny_graph(n):
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    m = sp.random(n, n, density=0.5, random_state=rng, format="csr")
    m = m + m.T
    m.setdiag(0); m.eliminate_zeros()
    return m.tocsr()


# ---------------------------------------------------------------- umap ----


def _adata_with_knn(n=30, key="neighbors"):
    import anndata as ad
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.normal(size=(n, 5)).astype(np.float32))
    a.obsm["X_svd"] = rng.normal(size=(n, 4)).astype(np.float32)
    a.obsm["X_gdr"] = rng.normal(size=(n, 4)).astype(np.float32)
    k = 5
    idx = np.tile(np.arange(k), (n, 1))
    dst = np.abs(rng.normal(size=(n, k))).astype(np.float64)
    a.uns[f"_{key}_knn_indices"] = idx
    a.uns[f"_{key}_knn_dists"] = dst
    a.uns[key] = {"params": {"use_rep": "X_svd"}}
    return a


def test_umap_conflicting_use_rep_errors_on_cytome_path(tmp_path):
    """With a reused graph, use_rep selects data, not topology; a conflict now
    stops you instead of deceiving you."""
    from piaso.tools._umap import umap
    from piaso.tools._neighbors import _neighbors_graph_keys

    ds = _ds_with_leiden(tmp_path / "t.cytome")
    # store a graph + its use_rep metadata under the default neighbors name
    nn_key = _neighbors_graph_keys("neighbors")[2]
    ds.metadata[nn_key] = 5
    ds.metadata[nn_key.replace("n_neighbors", "use_rep")] = "RNA_svd"
    ds.flush()                      # metadata writes are buffered
    with pytest.raises(ValueError, match="was built on 'RNA_svd'"):
        umap(ds, use_rep="ATAC_svd")


def test_umap_no_stored_knn_announces_its_own_neighbors():
    """The one case where use_rep decides the topology is no longer silent."""
    import anndata as ad
    from piaso.tools._umap import umap

    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.normal(size=(30, 5)).astype(np.float32))
    a.obsm["X_svd"] = rng.normal(size=(30, 4)).astype(np.float32)
    with pytest.warns(UserWarning, match="umap-learn will compute its own"):
        with warnings.catch_warnings():
            warnings.filterwarnings("default")
            umap(a, use_rep="X_svd")


def test_umap_reuse_path_still_works_unchanged():
    from piaso.tools._umap import umap

    a = _adata_with_knn()
    out = umap(a, use_rep="X_svd")
    emb = out if out is not None else a.obsm.get("X_umap")
    assert emb is not None and np.asarray(emb).shape == (30, 2)
