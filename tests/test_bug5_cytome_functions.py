"""Tests for Bug 5 fixes: cytome paths in calculateMetrics, normalize_log1p, plotEmbedding.

Bug 5a: _calculateCellMetrics_cytome used non-existent ml.iter_chunks()
Bug 5b: _calculatePeakMetrics_cytome used hardcoded ds.ATAC[measurement]
Bug 5c: Both used row-by-row SQL UPDATEs instead of executemany
Bug 5d: _plotEmbedding resource leak (cytome.open without close)
Bug 5e: _normalize_log1p_cytome used explicit SQL instead of MeasurementLayer API
"""
import os
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import E18_CYTOME

pytestmark = pytest.mark.requires_e18

# tmp_dir and subset_cytome fixtures provided by conftest.py


# ──────────────────────────────────────────────────────────────────────
# Bug 5a+5c: calculateCellMetrics cytome path
# ──────────────────────────────────────────────────────────────────────

def test_calculateCellMetrics_cytome_runs(subset_cytome):
    """calculateCellMetrics runs on cytome without crashing."""
    from piaso.preprocessing._calculateMetrics import calculateCellMetrics

    ds = subset_cytome
    # Check ATAC_counts exists
    has_atac = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'ATAC_counts' LIMIT 1"
    ).fetchone()
    if not has_atac:
        pytest.skip("No ATAC_counts matrix")

    calculateCellMetrics(ds, modality='ATAC', measurement='counts',
                         batch_size=256, verbose=True)

    # Verify columns were written
    cols = {r[1] for r in ds._conn.execute("PRAGMA table_info(cells)").fetchall()}
    assert 'n_fragments_in_peak' in cols
    assert 'n_peaks' in cols

    # Verify values are reasonable
    sums = [r[0] for r in ds._conn.execute(
        "SELECT n_fragments_in_peak FROM cells WHERE n_fragments_in_peak IS NOT NULL"
    ).fetchall()]
    assert len(sums) == ds.n_cells
    assert all(s >= 0 for s in sums)


def test_calculateCellMetrics_cytome_values(subset_cytome):
    """Verify calculateCellMetrics produces correct values by comparing to manual computation."""
    from piaso.preprocessing._calculateMetrics import calculateCellMetrics

    ds = subset_cytome
    has_atac = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'ATAC_counts' LIMIT 1"
    ).fetchone()
    if not has_atac:
        pytest.skip("No ATAC_counts matrix")

    # Manual computation: accumulate from iter_chunks
    manual_sum = np.zeros(ds.n_cells, dtype=np.float64)
    manual_nnz = np.zeros(ds.n_cells, dtype=np.int64)
    for chunk, indices in ds.iter_chunks(modality='ATAC', layer='counts', batch_size=256):
        if sp.issparse(chunk):
            manual_sum[indices] = np.ravel(chunk.sum(axis=1))
            manual_nnz[indices] = chunk.getnnz(axis=1)
        else:
            manual_sum[indices] = chunk.sum(axis=1)
            manual_nnz[indices] = (chunk != 0).sum(axis=1)

    # Run the function
    calculateCellMetrics(ds, modality='ATAC', measurement='counts',
                         batch_size=256, verbose=False)

    # Read back from cells table
    rows = ds._conn.execute(
        "SELECT cell_idx, n_fragments_in_peak, n_peaks FROM cells ORDER BY cell_idx"
    ).fetchall()

    for idx, fip, np_ in rows:
        assert fip == int(manual_sum[idx]), f"Cell {idx}: fip={fip} vs manual={manual_sum[idx]}"
        assert np_ == int(manual_nnz[idx]), f"Cell {idx}: n_peaks={np_} vs manual={manual_nnz[idx]}"


# ──────────────────────────────────────────────────────────────────────
# Bug 5b+5c: calculatePeakMetrics cytome path
# ──────────────────────────────────────────────────────────────────────

def test_calculatePeakMetrics_cytome_runs(subset_cytome):
    """calculatePeakMetrics runs on cytome without crashing."""
    from piaso.preprocessing._calculateMetrics import calculateFeatureMetrics

    ds = subset_cytome
    has_atac = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'ATAC_counts' LIMIT 1"
    ).fetchone()
    if not has_atac:
        pytest.skip("No ATAC_counts matrix")

    calculateFeatureMetrics(ds, modality='ATAC', measurement='counts',
                         batch_size=256, verbose=True)

    # Verify column was written
    cols = {r[1] for r in ds._conn.execute("PRAGMA table_info(peaks)").fetchall()}
    assert 'n_cells' in cols

    # Verify values
    vals = [r[0] for r in ds._conn.execute(
        "SELECT n_cells FROM peaks WHERE n_cells IS NOT NULL"
    ).fetchall()]
    assert len(vals) > 0
    assert all(v >= 0 for v in vals)


def test_calculatePeakMetrics_cytome_values(subset_cytome):
    """Verify calculatePeakMetrics produces correct col-wise nnz values."""
    from piaso.preprocessing._calculateMetrics import calculateFeatureMetrics

    ds = subset_cytome
    has_atac = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'ATAC_counts' LIMIT 1"
    ).fetchone()
    if not has_atac:
        pytest.skip("No ATAC_counts matrix")

    # Get n_features
    n_features = int(ds._conn.execute(
        "SELECT n_cols FROM matrix_meta WHERE matrix_name = 'ATAC_counts'"
    ).fetchone()[0])

    # Manual col-wise nnz
    manual_col_nnz = np.zeros(n_features, dtype=np.int64)
    for chunk, indices in ds.iter_chunks(modality='ATAC', layer='counts', batch_size=256):
        if sp.issparse(chunk):
            manual_col_nnz += np.ravel((chunk != 0).sum(axis=0))
        else:
            manual_col_nnz += (chunk != 0).sum(axis=0)

    # Run function
    calculateFeatureMetrics(ds, modality='ATAC', measurement='counts',
                         batch_size=256, verbose=False)

    # Read back
    rows = ds._conn.execute(
        "SELECT peak_idx, n_cells FROM peaks ORDER BY peak_idx"
    ).fetchall()

    for idx, nc in rows:
        assert nc == int(manual_col_nnz[idx]), \
            f"Peak {idx}: n_cells={nc} vs manual={manual_col_nnz[idx]}"


# ──────────────────────────────────────────────────────────────────────
# Bug 5d (superseded by native plotting refactor — see test_plotting_cytome_native)
# `_cytome_to_proxy_adata` was removed when plot_embeddings_split was rewritten
# natively. The resource-leak concern is now moot (no transient AnnData proxy
# is built; cytome inputs are accessed via the public API only).
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Bug 5e: normalize_log1p uses MeasurementLayer API (not raw SQL)
# ──────────────────────────────────────────────────────────────────────

def test_normalize_log1p_cytome_runs(subset_cytome):
    """normalize_log1p runs on cytome with RNA_counts layer."""
    from piaso.preprocessing._normalize_log1p import normalize_log1p

    ds = subset_cytome
    has_rna = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'RNA_counts' LIMIT 1"
    ).fetchone()
    if not has_rna:
        pytest.skip("No RNA_counts matrix")

    normalize_log1p(ds, target_sum=1e4, key_added='log1p',
                    save_layer=False, modality='RNA', layer='counts',
                    batch_size=256)

    # Result stored in _mem_layers
    assert hasattr(ds, '_mem_layers')
    assert 'log1p' in ds._mem_layers
    result = ds._mem_layers['log1p']
    assert result.shape[0] == ds.n_cells
    # All values should be non-negative (log1p of non-negative)
    assert np.all(result >= 0)


def test_normalize_log1p_cytome_values(subset_cytome):
    """Verify normalize_log1p produces correct values on cytome."""
    from piaso.preprocessing._normalize_log1p import normalize_log1p

    ds = subset_cytome
    has_rna = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'RNA_counts' LIMIT 1"
    ).fetchone()
    if not has_rna:
        pytest.skip("No RNA_counts matrix")

    # Manual: collect all chunks into dense, normalize, log1p
    chunks = []
    for chunk, indices in ds.iter_chunks(modality='RNA', layer='counts', batch_size=256):
        if sp.issparse(chunk):
            chunks.append(chunk.toarray().astype(np.float32))
        else:
            chunks.append(np.asarray(chunk, dtype=np.float32))

    manual = np.vstack(chunks)
    row_sums = manual.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    manual = manual / row_sums * 1e4
    manual = np.log1p(manual)

    # Run function
    normalize_log1p(ds, target_sum=1e4, key_added='test_norm',
                    save_layer=False, modality='RNA', layer='counts',
                    batch_size=256)

    result = ds._mem_layers['test_norm']
    np.testing.assert_allclose(result, manual, rtol=1e-5)


def test_normalize_log1p_bad_matrix_raises(subset_cytome):
    """normalize_log1p raises ValueError for non-existent matrix."""
    from piaso.preprocessing._normalize_log1p import normalize_log1p

    ds = subset_cytome
    with pytest.raises(ValueError, match="not found"):
        normalize_log1p(ds, modality='FAKE', layer='nonexistent')


# ──────────────────────────────────────────────────────────────────────
# Modality parameter test
# ──────────────────────────────────────────────────────────────────────

def test_normalize_log1p_has_modality_layer_params():
    """normalize_log1p exposes modality and layer parameters."""
    import inspect
    from piaso.preprocessing._normalize_log1p import normalize_log1p

    sig = inspect.signature(normalize_log1p)
    params = sig.parameters
    assert 'modality' in params, "Missing modality parameter"
    assert 'layer' in params, "Missing layer parameter"
    assert params['modality'].default == 'RNA', "modality default should be 'RNA'"
    assert params['layer'].default == 'counts', "layer default should be 'counts'"
