"""The importer's per-feature and per-cell columns, and the guard on them.

The value of these columns is that a consumer can skip a pass over a matrix
that may be tens of gigabytes. The risk is that the columns outlive the matrix
they describe, in which case the consumer gets a wrong answer with no error.
So most of what is tested here is the refusal: every field of the guard is
falsified in turn and the reader must decline.
"""
import json
import sqlite3

import numpy as np
import pytest
import scipy.sparse as sp

from piaso.utils._importer_stats import read_feature_stats, read_cell_stats


class _FakeDS:
    """The two attributes the reader touches."""

    def __init__(self, conn, table="tiles"):
        self._conn = conn
        self._table = table


@pytest.fixture
def ds(monkeypatch):
    """A 6-cell x 4-tile matrix with its statistics written as columns."""
    X = sp.csr_matrix(np.array([
        [0, 3, 0, 1],
        [2, 0, 0, 0],
        [0, 1, 5, 0],
        [0, 0, 0, 4],
        [7, 2, 0, 0],
        [0, 0, 1, 1],
    ], dtype=np.float64))
    n_rows, n_cols = X.shape
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE matrix_meta (matrix_name TEXT PRIMARY KEY, n_rows INT,
                                  n_cols INT, n_nonzero INT);
        CREATE TABLE _metadata (key TEXT PRIMARY KEY, value TEXT, value_type TEXT);
        CREATE TABLE tiles (tile_idx INTEGER PRIMARY KEY, n_cells_accessible INT,
                            tile_total_counts INT);
        CREATE TABLE cells (cell_idx INTEGER PRIMARY KEY, n_tiles_accessible INT,
                            tile_total_counts INT);
    """)
    conn.execute("INSERT INTO matrix_meta VALUES ('tiles_counts', ?, ?, ?)",
                 (n_rows, n_cols, X.nnz))
    guard = {"matrix": "tiles_counts", "n_rows": n_rows, "n_cols": n_cols,
             "n_nonzero": int(X.nnz), "counting": "insertions"}
    conn.execute("INSERT INTO _metadata VALUES ('importer_stats:tiles_counts', ?, 'scalar')",
                 (json.dumps(guard),))
    conn.execute("INSERT INTO _metadata VALUES ('counting:tiles_counts', '\"insertions\"', 'scalar')")
    for j in range(n_cols):
        col = X[:, j]
        conn.execute("INSERT INTO tiles VALUES (?, ?, ?)",
                     (j, int(col.nnz), int(col.sum())))
    for i in range(n_rows):
        row = X[i]
        conn.execute("INSERT INTO cells VALUES (?, ?, ?)",
                     (i, int(row.nnz), int(row.sum())))
    conn.commit()
    monkeypatch.setattr(
        "piaso.utils._importer_stats._entity_for",
        lambda _ds, _mod: ("tiles", "tile_idx"),
    )
    # This matrix is tiny; without lowering the gate the reader would
    # (correctly) say "just stream it" and none of the guard logic would run.
    monkeypatch.setattr("piaso.utils._importer_stats._MIN_NNZ_PER_COLUMN", 0)
    return _FakeDS(conn), X


def test_columns_reproduce_the_matrix_exactly(ds):
    d, X = ds
    n_per_feature, feature_sum = read_feature_stats(d, "tiles")
    np.testing.assert_array_equal(n_per_feature, (X > 0).sum(axis=0).A1)
    np.testing.assert_array_equal(feature_sum, X.sum(axis=0).A1)
    n_per_cell, cell_sum = read_cell_stats(d, "tiles")
    np.testing.assert_array_equal(n_per_cell, (X > 0).sum(axis=1).A1)
    np.testing.assert_array_equal(cell_sum, X.sum(axis=1).A1)


@pytest.mark.parametrize("field,value", [
    ("matrix", "something_else"),
    ("n_rows", 99),
    ("n_cols", 99),
    ("n_nonzero", 99),
    ("counting", "overlaps"),
])
def test_a_stale_guard_field_makes_the_reader_decline(ds, field, value):
    d, _ = ds
    guard = json.loads(d._conn.execute(
        "SELECT value FROM _metadata WHERE key='importer_stats:tiles_counts'"
    ).fetchone()[0])
    guard[field] = value
    d._conn.execute("UPDATE _metadata SET value=? WHERE key='importer_stats:tiles_counts'",
                    (json.dumps(guard),))
    assert read_feature_stats(d, "tiles") is None
    assert read_cell_stats(d, "tiles") is None


def test_no_guard_means_no_shortcut(ds):
    d, _ = ds
    d._conn.execute("DELETE FROM _metadata WHERE key='importer_stats:tiles_counts'")
    assert read_feature_stats(d, "tiles") is None


def test_a_null_is_not_read_as_a_zero(ds):
    """A half-written column must be refused, not silently under-counted."""
    d, _ = ds
    d._conn.execute("UPDATE tiles SET tile_total_counts = NULL WHERE tile_idx = 2")
    assert read_feature_stats(d, "tiles") is None


def test_a_missing_column_is_not_an_error(ds):
    d, _ = ds
    d._conn.execute("ALTER TABLE tiles DROP COLUMN n_cells_accessible")
    assert read_feature_stats(d, "tiles") is None


def test_row_count_mismatch_is_refused(ds):
    """Cells dropped from the table after import must not shift the arrays."""
    d, _ = ds
    d._conn.execute("DELETE FROM cells WHERE cell_idx = 5")
    assert read_cell_stats(d, "tiles") is None


def test_a_sparse_matrix_is_left_to_the_stream(ds, monkeypatch):
    """Below the break-even density, streaming is cheaper and is chosen."""
    d, _ = ds
    monkeypatch.setattr("piaso.utils._importer_stats._MIN_NNZ_PER_COLUMN", 1000)
    assert read_feature_stats(d, "tiles") is None
