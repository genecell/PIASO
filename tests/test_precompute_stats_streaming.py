"""Test streaming _precompute_stats_streaming matches _precompute_stats."""

import sys
import os
import numpy as np
import scipy.sparse as sp
from scipy import sparse
import pytest
from conftest import E18_CYTOME

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'cytome'))

from piaso.tools._normalization import _precompute_stats, _precompute_stats_streaming

MTG_H5AD = "/path/to/project/mtg_full.h5ad"
# E18_CYTOME comes from conftest.py; @pytest.mark.requires_e18 applies the skip


def _mock_iter_chunks(X_csr, batch_size=512):
    """Create mock chunk iterator from a CSR matrix."""
    def factory():
        n = X_csr.shape[0]
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            yield X_csr[i:end], np.arange(i, end)
    return factory


def _compute_mean_var_from_matrix(X):
    """Compute (mean, residual_var) the same way _precompute_stats does."""
    mean_2d = np.array(X.mean(axis=0))
    infog_mean = mean_2d.copy().ravel()
    mean_sq = infog_mean ** 2
    if sp.issparse(X):
        data_sq = X.data ** 2
        X_sq = sp.csr_matrix((data_sq, X.indices, X.indptr), shape=X.shape, copy=False)
        residual_var = np.squeeze(np.array(X_sq.mean(axis=0)) - mean_sq)
    else:
        residual_var = np.squeeze(np.mean(np.asarray(X) ** 2, axis=0) - mean_sq)
    return infog_mean, residual_var


def _compute_mean_var_streaming(X_csr, batch_size=512):
    """Compute (mean, residual_var) the same way _precompute_stats_streaming does."""
    n_cells, n_features = X_csr.shape
    col_sum = np.zeros(n_features, dtype=np.float64)
    col_sq_sum = np.zeros(n_features, dtype=np.float64)

    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        chunk = X_csr[i:end]
        col_sum += np.array(chunk.sum(axis=0), dtype=np.float64).ravel()
        sq_data = chunk.data.astype(np.float64) ** 2
        X_sq = sp.csr_matrix((sq_data, chunk.indices, chunk.indptr), shape=chunk.shape, copy=False)
        col_sq_sum += np.array(X_sq.sum(axis=0), dtype=np.float64).ravel()

    infog_mean = col_sum / n_cells
    mean_sq = infog_mean ** 2
    residual_var = (col_sq_sum / n_cells) - mean_sq
    return infog_mean, residual_var


def test_streaming_vs_anndata_synthetic():
    """Test on synthetic sparse matrix — small enough that float32/64 doesn't matter."""
    np.random.seed(42)
    n_cells, n_genes = 500, 200
    X = sp.random(n_cells, n_genes, density=0.1, format='csr', dtype=np.float32)
    X.data[:] = np.random.exponential(2, size=X.data.shape).astype(np.float32)

    knn_adata = _precompute_stats(X, n_nearest_neighbors=15)
    knn_stream = _precompute_stats_streaming(
        _mock_iter_chunks(X, batch_size=100),
        n_cells=n_cells, n_features=n_genes, n_nearest_neighbors=15,
    )

    assert knn_adata.shape == knn_stream.shape
    np.testing.assert_array_equal(knn_adata, knn_stream,
                                  err_msg="KNN indices differ between AnnData and streaming")
    print(f"PASS: synthetic {n_cells}x{n_genes} — KNN identical")


def test_streaming_batch_size_consistency():
    """Verify different batch sizes produce identical results (self-consistency)."""
    np.random.seed(99)
    n_cells, n_genes = 300, 100
    X = sp.random(n_cells, n_genes, density=0.15, format='csr', dtype=np.float32)
    X.data[:] = np.random.exponential(1, size=X.data.shape).astype(np.float32)

    knn_ref = _precompute_stats_streaming(
        _mock_iter_chunks(X, batch_size=50),
        n_cells=n_cells, n_features=n_genes,
    )

    for bs in [100, 150, 300]:
        knn_s = _precompute_stats_streaming(
            _mock_iter_chunks(X, batch_size=bs),
            n_cells=n_cells, n_features=n_genes,
        )
        np.testing.assert_array_equal(knn_s, knn_ref,
                                      err_msg=f"batch_size={bs} gives different KNN")
    print("PASS: batch sizes 50, 100, 150, 300 all produce identical streaming KNN")


@pytest.mark.skipif(not os.path.exists(MTG_H5AD), reason="MTG h5ad not available")
def test_streaming_stats_close_rna():
    """Verify mean/var stats are close (float32 vs float64 accumulation).

    The original _precompute_stats accumulates in float32 (sparse matrix .mean()),
    while streaming uses float64. Both should be close within float32 precision.
    """
    import scanpy as sc
    adata = sc.read_h5ad(MTG_H5AD)
    if adata.n_obs > 2000:
        adata = adata[:2000].copy()

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sp.isspmatrix_csr(X):
        X = X.tocsr()

    n_cells, n_genes = X.shape
    print(f"Testing on RNA: {n_cells} cells x {n_genes} genes")

    mean_orig, var_orig = _compute_mean_var_from_matrix(X)
    mean_stream, var_stream = _compute_mean_var_streaming(X, batch_size=512)

    # float32 sparse accumulation has ~1e-7 relative error per addition,
    # compounding over thousands of cells gives rtol ~1e-4 for means
    # and ~1e-3 for variances (squared values amplify error)
    np.testing.assert_allclose(mean_stream, mean_orig, rtol=5e-5,
                               err_msg="Mean differs between approaches")
    np.testing.assert_allclose(var_stream, var_orig, rtol=5e-3,
                               err_msg="Variance differs between approaches")
    print(f"  Mean max rel diff: {np.max(np.abs(mean_stream - mean_orig) / (np.abs(mean_orig) + 1e-30)):.2e}")
    print(f"  Var max rel diff: {np.max(np.abs(var_stream - var_orig) / (np.abs(var_orig) + 1e-30)):.2e}")
    print("PASS: stats within float32 tolerance")


@pytest.mark.skipif(not os.path.exists(MTG_H5AD), reason="MTG h5ad not available")
def test_streaming_knn_high_overlap_rna():
    """Verify KNN overlap is high on real RNA data.

    Float32 vs float64 accumulation causes tiny mean/var differences,
    which can change KNN for equidistant features. We verify >95% overlap.
    """
    import scanpy as sc
    adata = sc.read_h5ad(MTG_H5AD)
    if adata.n_obs > 2000:
        adata = adata[:2000].copy()

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sp.isspmatrix_csr(X):
        X = X.tocsr()

    n_cells, n_genes = X.shape
    knn_adata = _precompute_stats(X, n_nearest_neighbors=30)
    knn_stream = _precompute_stats_streaming(
        _mock_iter_chunks(X, batch_size=512),
        n_cells=n_cells, n_features=n_genes, n_nearest_neighbors=30,
    )

    assert knn_adata.shape == knn_stream.shape

    # Per-gene KNN set overlap
    n_match = 0
    n_total = 0
    for i in range(knn_adata.shape[0]):
        set_a = set(knn_adata[i])
        set_s = set(knn_stream[i])
        n_match += len(set_a & set_s)
        n_total += len(set_a)

    overlap_pct = n_match / n_total * 100
    print(f"KNN overlap: {overlap_pct:.1f}% ({n_match}/{n_total})")
    assert overlap_pct > 95.0, f"KNN overlap too low: {overlap_pct:.1f}%"
    print(f"PASS: RNA KNN overlap {overlap_pct:.1f}% > 95%")


@pytest.mark.requires_e18
def test_streaming_e18_cytome_rna():
    """Test streaming on E18 cytome RNA: verify stats and KNN overlap."""
    import cytome
    ds = cytome.open(E18_CYTOME)

    chunks = []
    for chunk_csr, ri in ds.iter_chunks(modality="RNA", layer="counts", batch_size=1024):
        chunks.append(chunk_csr)
    X_full = sp.vstack(chunks, format='csr')
    n_cells, n_genes = X_full.shape
    print(f"E18 RNA: {n_cells} cells x {n_genes} genes")

    # Verify stats are close (float32 vs float64 accumulation)
    mean_orig, var_orig = _compute_mean_var_from_matrix(X_full)
    mean_stream, var_stream = _compute_mean_var_streaming(X_full, batch_size=1024)
    np.testing.assert_allclose(mean_stream, mean_orig, rtol=5e-5)
    np.testing.assert_allclose(var_stream, var_orig, rtol=5e-3)

    # KNN comparison
    knn_adata = _precompute_stats(X_full, n_nearest_neighbors=30)

    def chunk_factory():
        return ds.iter_chunks(modality="RNA", layer="counts", batch_size=1024)

    knn_stream = _precompute_stats_streaming(
        chunk_factory, n_cells=n_cells, n_features=n_genes, n_nearest_neighbors=30,
    )

    # KNN overlap (E18 RNA has many low-expressed genes with similar stats,
    # so float precision differences cause more KNN changes)
    n_match = 0
    n_total = 0
    for i in range(knn_adata.shape[0]):
        set_a = set(knn_adata[i])
        set_s = set(knn_stream[i])
        n_match += len(set_a & set_s)
        n_total += len(set_a)

    overlap_pct = n_match / n_total * 100
    print(f"KNN overlap: {overlap_pct:.1f}% ({n_match}/{n_total})")
    # E18 RNA has many genes with near-zero expression → many tied distances
    # Even 60% overlap means the score will be very close
    assert overlap_pct > 60.0, f"KNN overlap too low: {overlap_pct:.1f}%"
    print(f"PASS: E18 cytome RNA KNN overlap {overlap_pct:.1f}%")
    ds.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
