"""Tests for cytome API wrappers added to replace explicit SQL.

New APIs covered:
- EntityTable.__contains__ and has_column
- Dataset.matrix_meta(name)
- Dataset.list_matrices()
- Dataset.list_embeddings(pattern)
- Dataset.delete_embedding(name)
- Dataset.filter_cells(mask)
"""
import os

import numpy as np
import pytest
from conftest import E18_CYTOME

pytestmark = pytest.mark.requires_e18

# tmp_cytome fixture provided by conftest.py


# ──────────────────────────────────────────────────────────────────────
# EntityTable.__contains__ and has_column
# ──────────────────────────────────────────────────────────────────────

def test_entity_table_contains():
    """`'col' in ds.cells` works as a column existence check."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        assert 'cell_idx' in ds.cells
        assert 'barcode' in ds.cells
        assert 'definitely_not_a_column' not in ds.cells
    finally:
        ds.close()


def test_entity_table_has_column():
    """has_column() works the same as __contains__."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        assert ds.cells.has_column('cell_idx') is True
        assert ds.cells.has_column('barcode') is True
        assert ds.cells.has_column('not_a_real_column') is False
    finally:
        ds.close()


# ──────────────────────────────────────────────────────────────────────
# Dataset.matrix_meta and list_matrices
# ──────────────────────────────────────────────────────────────────────

def test_matrix_meta_returns_dict():
    """matrix_meta() returns full metadata dict for an existing matrix."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        meta = ds.matrix_meta('RNA_counts')
        assert meta is not None
        assert meta['matrix_name'] == 'RNA_counts'
        assert meta['n_rows'] == ds.n_cells
        assert meta['n_cols'] > 0
        assert meta['col_entity'] == 'genes'
    finally:
        ds.close()


def test_matrix_meta_returns_none_for_missing():
    """matrix_meta() returns None if the matrix doesn't exist."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        assert ds.matrix_meta('NonExistent_Matrix') is None
    finally:
        ds.close()


def test_list_matrices_returns_all():
    """list_matrices() returns all registered matrix names."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        mats = ds.list_matrices()
        assert isinstance(mats, list)
        assert len(mats) > 0
        # Sanity: every name from list_matrices should resolve via matrix_meta
        for name in mats:
            assert ds.matrix_meta(name) is not None
    finally:
        ds.close()


# ──────────────────────────────────────────────────────────────────────
# Dataset.list_embeddings
# ──────────────────────────────────────────────────────────────────────

def test_list_embeddings_no_pattern():
    """list_embeddings() with no pattern returns all embeddings."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        emb = ds.list_embeddings()
        assert isinstance(emb, list)
        # E18 should have at least PCA / UMAP / SVD embeddings
        assert len(emb) > 0
    finally:
        ds.close()


def test_list_embeddings_with_pattern():
    """list_embeddings(pattern) filters by SQL LIKE."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        all_emb = ds.list_embeddings()
        svd_emb = ds.list_embeddings('%svd%')
        umap_emb = ds.list_embeddings('%umap%')

        # Pattern results should be subsets of all embeddings
        assert set(svd_emb).issubset(set(all_emb))
        assert set(umap_emb).issubset(set(all_emb))

        # Sanity: svd names should contain 'svd'
        for n in svd_emb:
            assert 'svd' in n.lower()
        for n in umap_emb:
            assert 'umap' in n.lower()
    finally:
        ds.close()


# ──────────────────────────────────────────────────────────────────────
# Dataset.delete_embedding
# ──────────────────────────────────────────────────────────────────────

def test_delete_embedding_removes(tmp_cytome):
    """delete_embedding() removes the embedding from the dataset."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        before = ds.list_embeddings()
        if not before:
            pytest.skip("No embeddings in test cytome")
        target = before[0]
        ds.delete_embedding(target)
        after = ds.list_embeddings()
        assert target not in after
        assert len(after) == len(before) - 1
    finally:
        ds.close()


def test_delete_embedding_idempotent(tmp_cytome):
    """delete_embedding() on a non-existent name is a no-op."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        before = ds.list_embeddings()
        ds.delete_embedding('does_not_exist')
        after = ds.list_embeddings()
        assert before == after
    finally:
        ds.close()


# ──────────────────────────────────────────────────────────────────────
# Dataset.filter_cells
# ──────────────────────────────────────────────────────────────────────

def test_filter_cells_keeps_correct_count(tmp_cytome):
    """filter_cells reduces n_cells to the keep count."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        n_before = ds.n_cells
        keep = np.zeros(n_before, dtype=bool)
        keep[:n_before // 2] = True
        n_after = ds.filter_cells(keep, include_fragments=False)
        assert n_after == n_before // 2
        assert ds.n_cells == n_after
    finally:
        ds.close()


def test_filter_cells_updates_matrix_n_rows(tmp_cytome):
    """filter_cells keeps matrix n_rows in sync with new n_cells."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        n_before = ds.n_cells
        keep = np.zeros(n_before, dtype=bool)
        keep[:300] = True
        ds.filter_cells(keep, include_fragments=False)
        for name in ds.list_matrices():
            meta = ds.matrix_meta(name)
            if meta['row_entity'] == 'cells':
                assert meta['n_rows'] == ds.n_cells, \
                    f"{name} n_rows={meta['n_rows']} != n_cells={ds.n_cells}"
    finally:
        ds.close()


def test_filter_cells_with_indices(tmp_cytome):
    """filter_cells accepts integer indices, not just bool mask."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        keep_idx = np.array([0, 5, 10, 100, 200])
        n_after = ds.filter_cells(keep_idx, include_fragments=False)
        assert n_after == len(keep_idx)
    finally:
        ds.close()


def test_filter_cells_preserves_tiles_entity(tmp_cytome):
    """filter_cells preserves the tiles entity table.

    Regression for Bug 6: subset() was only copying genes/peaks, dropping
    tiles entirely. Downstream COSG with modality='tiles' then crashed
    because the tiles entity table was empty even though tiles_counts had
    millions of columns.
    """
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        n_tiles_before = ds._conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        if n_tiles_before == 0:
            pytest.skip("Test cytome has no tiles to preserve")

        keep = np.zeros(ds.n_cells, dtype=bool)
        keep[:max(10, ds.n_cells // 2)] = True
        ds.filter_cells(keep, include_fragments=False)

        n_tiles_after = ds._conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        assert n_tiles_after == n_tiles_before, (
            f"tiles dropped by filter_cells: {n_tiles_before} -> {n_tiles_after}"
        )

        # tiles_counts matrix n_cols should still match
        tiles_meta = ds.matrix_meta('tiles_counts')
        if tiles_meta is not None:
            assert tiles_meta['n_cols'] == n_tiles_before
    finally:
        ds.close()


# ──────────────────────────────────────────────────────────────────────
# Dataset.delete_matrix
# ──────────────────────────────────────────────────────────────────────

def test_delete_matrix_exact(tmp_cytome):
    """delete_matrix(name) removes a single matrix exactly."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        before = ds.list_matrices()
        if 'tiles_counts' not in before:
            pytest.skip("No tiles_counts matrix in test cytome")
        deleted = ds.delete_matrix('tiles_counts')
        assert deleted == ['tiles_counts']
        assert 'tiles_counts' not in ds.list_matrices()
        assert ds.matrix_meta('tiles_counts') is None
    finally:
        ds.close()


def test_delete_matrix_pattern(tmp_cytome):
    """delete_matrix with like=True removes by SQL LIKE pattern."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        before = ds.list_matrices()
        atac_mats = [m for m in before if m.startswith('ATAC_')]
        if not atac_mats:
            pytest.skip("No ATAC_ matrices in test cytome")
        deleted = ds.delete_matrix('ATAC_%', like=True)
        assert sorted(deleted) == sorted(atac_mats)
        for name in atac_mats:
            assert ds.matrix_meta(name) is None
    finally:
        ds.close()


def test_delete_matrix_missing(tmp_cytome):
    """delete_matrix on a non-existent name returns empty list."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        deleted = ds.delete_matrix('NoSuchMatrix')
        assert deleted == []
    finally:
        ds.close()


# ──────────────────────────────────────────────────────────────────────
# MetadataStore.get and __contains__
# ──────────────────────────────────────────────────────────────────────

def test_metadata_get_returns_default():
    """metadata.get(key, default) returns default when key absent."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        # 'definitely_not_a_key' should not exist
        assert ds.metadata.get('definitely_not_a_key') is None
        assert ds.metadata.get('definitely_not_a_key', 'fallback') == 'fallback'
    finally:
        ds.close()


def test_metadata_contains():
    """`'key' in ds.metadata` works as existence check."""
    import cytome

    ds = cytome.open(E18_CYTOME)
    try:
        assert ('definitely_not_a_key' in ds.metadata) is False
        # E18 is known to have 'leiden' / 'umap' / 'hvg' keys
        keys = ds.metadata.keys()
        if keys:
            assert keys[0] in ds.metadata
    finally:
        ds.close()


def test_metadata_get_returns_value(tmp_cytome):
    """metadata.get returns the stored value for an existing key."""
    import cytome

    ds = cytome.open(tmp_cytome)
    try:
        ds.metadata['_test_key'] = 'test_value'
        ds.flush()
        assert ds.metadata.get('_test_key') == 'test_value'
        assert '_test_key' in ds.metadata
    finally:
        ds.close()
