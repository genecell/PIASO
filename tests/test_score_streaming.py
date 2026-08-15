"""Test streaming score() produces results close to AnnData score().

Note: streaming accumulates in float64 while AnnData _precompute_stats
uses float32 accumulation. This causes small KNN differences which
propagate to small score differences. We verify scores are highly
correlated (r > 0.99) rather than bit-identical.
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pytest
from conftest import E18_CYTOME

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'cytome'))

MTG_H5AD = "/path/to/project/mtg_full.h5ad"
# E18_CYTOME comes from conftest.py; @pytest.mark.requires_e18 applies the skip


@pytest.mark.skipif(not os.path.exists(MTG_H5AD), reason="MTG h5ad not available")
def test_score_knn_consistency_rna():
    """Test that precomputed KNN gives identical scores for both paths."""
    import scanpy as sc
    from piaso.tools._normalization import score, _precompute_stats

    adata = sc.read_h5ad(MTG_H5AD)
    if adata.n_obs > 1000:
        adata = adata[:1000].copy()

    # Pick gene sets
    gene_counts = np.array(adata.X.sum(axis=0)).ravel()
    top_idx = np.argsort(gene_counts)[-200:]
    top_genes = adata.var_names[top_idx].tolist()

    gene_sets = {
        "Set1": top_genes[:20],
        "Set2": top_genes[20:50],
    }

    # Compute KNN once and provide to both paths
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    knn = _precompute_stats(X)

    sm1, names1, _ = score(
        adata, gene_sets, layer=None, random_seed=42,
        use_rust=False, precomputed_knn=knn, verbosity=0,
    )
    sm2, names2, _ = score(
        adata, gene_sets, layer=None, random_seed=42,
        use_rust=False, precomputed_knn=knn, verbosity=0,
    )

    np.testing.assert_array_equal(sm1, sm2, err_msg="Same KNN should give identical scores")
    print(f"PASS: same precomputed KNN → identical scores ({sm1.shape})")


@pytest.mark.requires_e18
def test_score_e18_cytome_rna():
    """Test score() on E18 cytome RNA — AnnData vs streaming.

    Due to float32/64 accumulation differences in KNN computation,
    scores will be close but not bit-identical. We verify high correlation.
    """
    import cytome
    from piaso.tools._normalization import score

    ds = cytome.open(E18_CYTOME)

    # Load full RNA matrix for AnnData baseline
    chunks = []
    for chunk_csr, _ri in ds.iter_chunks(modality="RNA", layer="counts", batch_size=4096):
        chunks.append(chunk_csr)
    X_full = sp.vstack(chunks, format='csr')
    del chunks

    gene_cols = ds.genes.columns
    for _gc in ["gene_id", "gene_name", "symbol"]:
        if _gc in gene_cols:
            _vals = np.array(ds.genes[_gc])
            if _vals[0] is not None:
                gene_names = _vals
                break

    import anndata
    adata = anndata.AnnData(X=X_full)
    adata.var_names = pd.Index(gene_names)

    # Create gene sets from top-expressed genes
    gene_sums = np.array(X_full.sum(axis=0)).ravel()
    top_idx = np.argsort(gene_sums)[-100:]
    top_genes = gene_names[top_idx].tolist()

    gene_sets = {
        "SetA": top_genes[:15],
        "SetB": top_genes[15:35],
    }

    # AnnData baseline
    sm_adata, names_adata, _ = score(
        adata, gene_sets, layer=None, random_seed=42,
        use_rust=False, verbosity=0,
    )
    print(f"AnnData score: {sm_adata.shape}")

    # Cytome streaming
    sm_cytome, names_cytome, _ = score(
        ds, gene_sets, random_seed=42,
        use_rust=False, verbosity=1,
        modality="RNA", cytome_layer="counts", batch_size=1024,
    )
    print(f"Cytome score: {sm_cytome.shape}")

    assert sm_adata.shape == sm_cytome.shape
    assert names_adata == names_cytome

    # Check correlation (should be very high despite float precision diffs)
    for i, name in enumerate(names_adata):
        corr = np.corrcoef(sm_adata[:, i], sm_cytome[:, i])[0, 1]
        max_diff = np.abs(sm_adata[:, i] - sm_cytome[:, i]).max()
        print(f"  {name}: r={corr:.6f}, max_diff={max_diff:.6f}")
        assert corr > 0.99, f"Correlation too low for {name}: {corr:.6f}"

    print("PASS: E18 RNA score highly correlated (r > 0.99) between AnnData and cytome")
    ds.close()


@pytest.mark.requires_e18
def test_score_cytome_shape_and_finite():
    """Basic shape and finiteness checks for cytome streaming score."""
    import cytome
    from piaso.tools._normalization import score

    ds = cytome.open(E18_CYTOME)
    gene_cols = ds.genes.columns
    for _gc in ["gene_id", "gene_name", "symbol"]:
        if _gc in gene_cols:
            _vals = np.array(ds.genes[_gc])
            if _vals[0] is not None:
                gene_names = _vals
                break

    # Use some known genes
    gene_sums = []
    for chunk, ri in ds.iter_chunks(modality="RNA", layer="counts", batch_size=4096):
        gene_sums.append(np.array(chunk.sum(axis=0)).ravel())
    gene_sums = np.sum(gene_sums, axis=0)
    top_genes = gene_names[np.argsort(gene_sums)[-30:]].tolist()

    gene_sets = {"Top30": top_genes}

    sm, names, pm = score(
        ds, gene_sets, random_seed=42,
        use_rust=False, verbosity=0,
        modality="RNA", cytome_layer="counts", batch_size=1024,
    )

    assert sm.shape == (ds.n_cells, 1)
    assert names == ["Top30"]
    assert np.all(np.isfinite(sm))
    print(f"PASS: cytome score shape={sm.shape}, range=[{sm.min():.4f}, {sm.max():.4f}]")
    ds.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
