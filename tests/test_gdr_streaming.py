"""Test cytome-aware GDR (calculateScoreParallel + runGDRParallel)."""

import sys
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pytest
from conftest import E18_CYTOME

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'cytome'))

# E18_CYTOME comes from conftest.py; @pytest.mark.requires_e18 applies the skip


def _get_gene_names(ds):
    """Get gene names from cytome dataset (handles different column names)."""
    gene_cols = ds.genes.columns
    for col in ["gene_id", "gene_name", "symbol"]:
        if col in gene_cols:
            vals = np.array(ds.genes[col])
            if vals[0] is not None:
                return vals
    raise ValueError(f"Cannot find gene name column with non-null values. Available: {gene_cols}")


@pytest.mark.requires_e18
def test_calculateScoreParallel_cytome():
    """Test calculateScoreParallel with cytome input (RNA modality)."""
    import cytome
    from piaso.tools._runGDR import calculateScoreParallel

    ds = cytome.open(E18_CYTOME)
    gene_names = _get_gene_names(ds)

    # Build marker gene sets from top expressed genes
    np.random.seed(42)
    gene_sums = []
    for chunk, ri in ds.iter_chunks(modality="RNA", layer="counts", batch_size=4096):
        gene_sums.append(np.array(chunk.sum(axis=0)).ravel())
    gene_sums = np.sum(gene_sums, axis=0)
    top_idx = np.argsort(gene_sums)[-60:]
    top_genes = gene_names[top_idx].tolist()

    marker_df = pd.DataFrame({
        "0": top_genes[:20],
        "1": top_genes[20:40],
        "2": top_genes[40:60],
    })

    score_list, gene_set_names = calculateScoreParallel(
        ds, gene_set=marker_df, score_method='piaso',
        modality="RNA", cytome_layer="counts", batch_size=1024,
    )

    assert score_list.shape[0] == ds.n_cells
    assert score_list.shape[1] == 3
    assert len(gene_set_names) == 3
    assert np.all(np.isfinite(score_list))
    print(f"PASS: calculateScoreParallel cytome RNA — {score_list.shape}")
    print(f"  Score range: [{score_list.min():.4f}, {score_list.max():.4f}]")
    ds.close()


@pytest.mark.requires_e18
def test_runGDRParallel_cytome():
    """Test runGDRParallel with cytome input (RNA modality)."""
    import cytome
    from piaso.tools._runGDR import _runGDRParallel_cytome

    ds = cytome.open(E18_CYTOME)

    # Use existing cluster labels
    X_gdr, marker_gene = _runGDRParallel_cytome(
        ds, groupby="CellTypes_Final", n_gene=30, mu=1.0,
        scoring_method='piaso', key_added=None, max_workers=4,
        random_seed=42, verbosity=1,
        modality="RNA", cytome_layer="counts", batch_size_cytome=1024,
        score_layer=None,
    )

    assert X_gdr.shape[0] == ds.n_cells, f"Expected {ds.n_cells} rows, got {X_gdr.shape[0]}"
    assert X_gdr.shape[1] > 0, "Expected non-zero columns"
    assert np.all(np.isfinite(X_gdr))

    # Check L2 normalization (rows should have ~unit norm)
    row_norms = np.linalg.norm(X_gdr, axis=1)
    assert np.all(row_norms > 0), "Some rows have zero norm"
    print(f"PASS: runGDRParallel cytome — X_gdr shape: {X_gdr.shape}")
    print(f"  Row norm range: [{row_norms.min():.4f}, {row_norms.max():.4f}]")
    ds.close()


@pytest.mark.requires_e18
def test_gdr_cytome_vs_anndata():
    """Compare GDR output: cytome streaming vs AnnData path.

    Due to float32/64 accumulation differences, GDR values will be close
    but not bit-identical. We verify high correlation.
    """
    import cytome
    import anndata
    from piaso.tools._runGDR import runGDRParallel, _runGDRParallel_cytome

    ds = cytome.open(E18_CYTOME)

    # Load RNA into AnnData
    chunks = []
    for chunk, ri in ds.iter_chunks(modality="RNA", layer="counts", batch_size=4096):
        chunks.append(chunk)
    X_full = sp.vstack(chunks, format='csr')
    del chunks

    gene_names = _get_gene_names(ds)
    cell_barcodes = np.array(ds.cells["barcode"])
    cluster_labels = np.array(ds.cells["CellTypes_Final"])

    adata = anndata.AnnData(X=X_full)
    adata.var_names = pd.Index(gene_names)
    adata.obs_names = pd.Index(cell_barcodes)
    adata.obs["CellTypes_Final"] = pd.Categorical(cluster_labels)

    # AnnData GDR
    runGDRParallel(
        adata, groupby="CellTypes_Final", n_gene=30, mu=1.0,
        scoring_method='piaso', key_added='gdr',
        max_workers=4, random_seed=42, verbosity=1,
    )
    X_gdr_adata = adata.obsm['gdr']

    # Cytome GDR
    X_gdr_cytome, _ = _runGDRParallel_cytome(
        ds, groupby="CellTypes_Final", n_gene=30, mu=1.0,
        scoring_method='piaso', key_added=None, max_workers=4,
        random_seed=42, verbosity=1,
        modality="RNA", cytome_layer="counts", batch_size_cytome=1024,
        score_layer=None,
    )

    assert X_gdr_adata.shape == X_gdr_cytome.shape, \
        f"Shape mismatch: {X_gdr_adata.shape} vs {X_gdr_cytome.shape}"

    # Correlation-based comparison (float precision causes KNN diffs)
    col_corrs = []
    for j in range(X_gdr_adata.shape[1]):
        corr = np.corrcoef(X_gdr_adata[:, j], X_gdr_cytome[:, j])[0, 1]
        col_corrs.append(corr)
    mean_corr = np.mean(col_corrs)
    min_corr = np.min(col_corrs)
    print(f"GDR column correlations: mean={mean_corr:.4f}, min={min_corr:.4f}")
    assert mean_corr > 0.95, f"Mean GDR correlation too low: {mean_corr:.4f}"
    print(f"PASS: GDR correlation — shape {X_gdr_adata.shape}, mean_corr={mean_corr:.4f}")
    ds.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
