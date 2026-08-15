"""Test cytome iter_chunks with col_mask parameter."""

import sys
import os
import numpy as np
import scipy.sparse as sp
import pytest
from conftest import E18_CYTOME

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'cytome'))

# E18_CYTOME comes from conftest.py; @pytest.mark.requires_e18 applies the skip


@pytest.mark.requires_e18
def test_col_mask_basic():
    """Test that col_mask selects the right columns."""
    import cytome
    ds = cytome.open(E18_CYTOME)

    # Get first chunk without mask
    for chunk_full, ri in ds.iter_chunks(modality="RNA", layer="counts", batch_size=100):
        break

    # Get first chunk with mask (first 50 columns)
    col_mask = np.arange(50)
    for chunk_masked, ri2 in ds.iter_chunks(
        modality="RNA", layer="counts", batch_size=100, col_mask=col_mask
    ):
        break

    assert chunk_masked.shape[1] == 50, f"Expected 50 cols, got {chunk_masked.shape[1]}"
    assert chunk_full.shape[0] == chunk_masked.shape[0], "Row count should match"

    # Verify content matches
    full_subset = chunk_full[:, :50]
    if sp.issparse(full_subset):
        full_subset = full_subset.toarray()
    if sp.issparse(chunk_masked):
        chunk_masked_dense = chunk_masked.toarray()
    else:
        chunk_masked_dense = np.asarray(chunk_masked)

    np.testing.assert_array_equal(full_subset, chunk_masked_dense,
                                  err_msg="col_mask content doesn't match")
    print(f"PASS: col_mask basic — {chunk_full.shape} → {chunk_masked.shape}")
    ds.close()


@pytest.mark.requires_e18
def test_col_mask_noncontiguous():
    """Test col_mask with non-contiguous indices."""
    import cytome
    ds = cytome.open(E18_CYTOME)

    col_mask = np.array([0, 5, 10, 100, 500])

    for chunk_full, ri in ds.iter_chunks(modality="RNA", layer="counts", batch_size=100):
        break
    for chunk_masked, ri2 in ds.iter_chunks(
        modality="RNA", layer="counts", batch_size=100, col_mask=col_mask
    ):
        break

    assert chunk_masked.shape[1] == len(col_mask)

    full_subset = chunk_full[:, col_mask]
    if sp.issparse(full_subset):
        full_subset = full_subset.toarray()
    if sp.issparse(chunk_masked):
        chunk_masked_dense = chunk_masked.toarray()
    else:
        chunk_masked_dense = np.asarray(chunk_masked)

    np.testing.assert_array_equal(full_subset, chunk_masked_dense)
    print(f"PASS: non-contiguous col_mask")
    ds.close()


@pytest.mark.requires_e18
def test_col_mask_full_iteration():
    """Test col_mask through full iteration (all chunks)."""
    import cytome
    ds = cytome.open(E18_CYTOME)

    col_mask = np.arange(100)
    total_rows = 0
    for chunk, ri in ds.iter_chunks(
        modality="RNA", layer="counts", batch_size=512, col_mask=col_mask
    ):
        assert chunk.shape[1] == 100, f"Expected 100 cols, got {chunk.shape[1]}"
        total_rows += chunk.shape[0]

    assert total_rows == ds.n_cells, f"Expected {ds.n_cells} total rows, got {total_rows}"
    print(f"PASS: full iteration with col_mask — {total_rows} rows x 100 cols")
    ds.close()


@pytest.mark.requires_e18
def test_col_mask_none_same_as_no_mask():
    """Test that col_mask=None gives same result as no col_mask."""
    import cytome
    ds = cytome.open(E18_CYTOME)

    for chunk1, ri1 in ds.iter_chunks(modality="RNA", layer="counts", batch_size=100):
        break
    for chunk2, ri2 in ds.iter_chunks(
        modality="RNA", layer="counts", batch_size=100, col_mask=None
    ):
        break

    if sp.issparse(chunk1):
        chunk1 = chunk1.toarray()
    if sp.issparse(chunk2):
        chunk2 = chunk2.toarray()

    np.testing.assert_array_equal(chunk1, chunk2)
    print("PASS: col_mask=None same as no mask")
    ds.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
