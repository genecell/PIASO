"""Unit tests for cell_mask on score / runSVD / neighbors / leiden.

These verify the foundational cell_mask plumbing — each function
must return masked-only output without persisting full-cell-aligned
state to the AnnData/cytome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _make_adata(n_cells=200, n_genes=100, n_clusters=3, seed=0):
    """Small AnnData with cluster structure for testing.

    Default n_cells >= 60 so score()'s default n_nearest_neighbors=30
    doesn't exceed the masked-subset size in tests.
    """
    from anndata import AnnData
    rng = np.random.default_rng(seed)
    cluster_labels = np.repeat(
        np.arange(n_clusters), n_cells // n_clusters
    )
    cluster_labels = np.concatenate([
        cluster_labels,
        np.full(n_cells - len(cluster_labels), n_clusters - 1),
    ])
    rng.shuffle(cluster_labels)
    counts = rng.negative_binomial(1, 0.4, size=(n_cells, n_genes)).astype(np.float32)
    for k in range(n_clusters):
        boost = slice(k * 5, (k + 1) * 5)
        in_k = cluster_labels == k
        counts[np.ix_(in_k, range(boost.start, boost.stop))] += \
            rng.poisson(10, size=(in_k.sum(), 5)).astype(np.float32)
    X = sp.csr_matrix(counts)
    obs = pd.DataFrame({
        "cluster": cluster_labels.astype(str),
    }, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    var["gene_id"] = var.index
    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------
# score() cell_mask
# ---------------------------------------------------------------------

def test_score_cell_mask_returns_n_masked_rows():
    """score() with cell_mask returns score_matrix of shape (n_masked, n_sets)."""
    import piaso
    adata = _make_adata()
    piaso.tl.infog(adata, n_top_genes=15, verbosity=0)
    gene_sets = {"set_a": adata.var_names[:5].tolist(),
                 "set_b": adata.var_names[5:10].tolist()}
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[::2] = True  # every other cell
    n_masked = int(mask.sum())

    score_matrix, names, pvals = piaso.tl.score(
        adata, gene_list=gene_sets, layer="infog",
        cell_mask=mask, compute_pvalues=False,
    )
    assert score_matrix.shape == (n_masked, 2), (
        f"Expected ({n_masked}, 2), got {score_matrix.shape}"
    )


def test_score_cell_mask_equals_subset_then_score():
    """score(adata, cell_mask=mask) == score(adata[mask])."""
    import piaso
    adata = _make_adata(seed=42)
    piaso.tl.infog(adata, n_top_genes=15, verbosity=0)
    gene_sets = {"set_a": adata.var_names[:5].tolist()}
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[3:30] = True

    sm_with_mask, _, _ = piaso.tl.score(
        adata, gene_list=gene_sets, layer="infog",
        cell_mask=mask, compute_pvalues=False, random_seed=0,
    )

    # Manual subset path — slice then score (no mask)
    adata_sub = adata[mask].copy()
    sm_manual, _, _ = piaso.tl.score(
        adata_sub, gene_list=gene_sets, layer="infog",
        compute_pvalues=False, random_seed=0,
    )

    assert sm_with_mask.shape == sm_manual.shape
    # Numerical parity (some KDTree sampling stochasticity allowed)
    np.testing.assert_allclose(sm_with_mask, sm_manual, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------
# runSVD() cell_mask (cytome path; AnnData raises)
# ---------------------------------------------------------------------

def test_runSVD_anndata_cell_mask_raises_NotImplementedError():
    """AnnData runSVD doesn't support cell_mask (use adata[mask] instead)."""
    import piaso
    adata = _make_adata()
    adata.var["highly_variable"] = True
    mask = np.ones(adata.n_obs, dtype=bool)
    mask[:5] = False
    with pytest.raises(NotImplementedError, match="cell_mask"):
        piaso.tl.runSVD(adata, cell_mask=mask, n_components=5)


def test_runSVD_cytome_cell_mask_returns_n_masked_rows(tmp_path):
    """Cytome runSVD with cell_mask returns (n_masked, n_components)
    in-memory and does NOT write to cytome.embeddings."""
    import cytome
    import piaso
    adata = _make_adata(n_cells=60, n_genes=200)
    adata.var["highly_variable"] = True

    p = tmp_path / "rna.cytome"
    ds = cytome.create(p)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(adata.n_obs),
        "barcode": adata.obs_names.astype(str).tolist(),
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(adata.n_vars),
        "gene_id": adata.var_names.astype(str).tolist(),
        "highly_variable": [True] * adata.n_vars,
    }))
    ds.add_matrix("RNA_counts", adata.X)
    ds.flush()
    ds.close()

    ds = cytome.open(p)
    mask = np.zeros(ds.n_cells, dtype=bool)
    mask[10:40] = True
    n_masked = int(mask.sum())

    emb_before = set(ds.list_embeddings())
    # Use the counts matrix directly (no infog layer needed for this test)
    result = piaso.tl.runSVD(
        ds, modality="RNA", n_components=10, n_iter=3,
        cell_mask=mask, streaming=True, random_state=0,
        measurement="counts",
    )
    # runSVD streaming returns (embeddings, S, Vt) tuple
    emb = result[0] if isinstance(result, tuple) else result
    assert emb.shape == (n_masked, 10), (
        f"Expected ({n_masked}, 10), got {emb.shape}"
    )

    # No persisted embedding written
    ds_check = cytome.open(p)
    emb_after = set(ds_check.list_embeddings())
    new_embeddings = emb_after - emb_before
    # The auto_tfidf path writes nothing; no per-variant key was set;
    # cell_mask is set → no write expected
    assert not new_embeddings, (
        f"runSVD(cell_mask=...) should NOT persist to cytome. New: {new_embeddings}"
    )
    ds.close()
    ds_check.close()


# ---------------------------------------------------------------------
# neighbors() cell_mask + ndarray
# ---------------------------------------------------------------------

def test_neighbors_accepts_ndarray_input():
    """neighbors() must accept a raw ndarray (skip cytome/AnnData writes)."""
    import piaso
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 10))
    knn = piaso.tl.neighbors(X, n_neighbors=5, random_state=0)
    assert "knn_indices" in knn
    assert "connectivities" in knn
    assert knn["connectivities"].shape == (50, 50)


def test_neighbors_cell_mask_returns_masked_graph():
    """neighbors() with cell_mask returns (n_masked, n_masked) graph."""
    import piaso
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 10))
    mask = np.zeros(50, dtype=bool)
    mask[10:35] = True  # 25 cells
    knn = piaso.tl.neighbors(X, n_neighbors=5, cell_mask=mask, random_state=0)
    assert knn["connectivities"].shape == (25, 25)
    assert knn["knn_indices"].shape == (25, 5)


# ---------------------------------------------------------------------
# leiden() cell_mask
# ---------------------------------------------------------------------

def test_leiden_cell_mask_returns_masked_labels():
    """leiden() with cell_mask returns labels of length n_masked."""
    import piaso
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    # Use the same mask for neighbors + leiden so the graph is consistent
    mask = np.ones(40, dtype=bool)
    mask[:10] = False  # 30 cells masked-true
    knn = piaso.tl.neighbors(X, n_neighbors=5, cell_mask=mask, random_state=0)
    # knn shape is (30, 30) — already masked. Pass cell_mask=None to leiden
    # because the graph is already in masked space.
    labels = piaso.tl.leiden(
        data=None, knn_result=knn, resolution=1.0, random_state=0,
        cell_mask=None,
    )
    assert labels.shape[0] == 30, f"Expected 30 labels, got {labels.shape[0]}"


def test_leiden_cell_mask_on_full_graph_filters_to_masked():
    """If you pass a full-cell knn_result + cell_mask, leiden subsets
    the adjacency to masked rows × masked cols and returns n_masked
    labels."""
    import piaso
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    # Full-cell KNN
    knn_full = piaso.tl.neighbors(X, n_neighbors=5, random_state=0)
    assert knn_full["connectivities"].shape == (40, 40)

    mask = np.ones(40, dtype=bool)
    mask[:15] = False  # 25 cells masked-true

    labels = piaso.tl.leiden(
        data=None, knn_result=knn_full, resolution=1.0,
        random_state=0, cell_mask=mask,
    )
    assert labels.shape[0] == 25
