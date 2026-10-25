"""Per-feature and per-cell summaries the importer already computed.

``selectPeaks`` needs two vectors -- non-zeros per feature and total counts
per feature -- and the TF-IDF setup needs those plus the same two per cell.
Every one of them is a sum over the matrix, and the importer had every entry
in hand while writing it, so it records them as entity columns. Reading four
columns instead of streaming the matrix is the difference between seconds and
a full pass over 34 GB, and it is the same arithmetic either way.

The risk in a cached statistic is that it outlives what it describes. So the
importer writes a guard beside the columns naming the matrix, its shape, its
non-zero count and its counting convention, and nothing here is used unless
all five still match. A cytome whose matrix was rebuilt, re-imported under the other
convention, or filtered down to fewer cells fails the check and the caller
streams, which is what it would have done anyway. Being wrong here is silent
and would propagate into feature selection, so the guard is deliberately
stricter than it needs to be.

The guard names a matrix, but the columns are shared: a store holds one
``n_cells_accessible`` per feature whatever number of matrices it holds over
those features. So the reader also checks that the columns hold the guarded
matrix's sums: each stored entry adds one to exactly one feature's count and to
exactly one cell's, so either count column sums to the matrix's number of
non-zeros, and either total column to the matrix's total count, which the guard
records. Columns another matrix wrote (a matrix quantified beside the main one,
by a writer that predates the side-matrix role) fail that check even when the
guard's five fields still match.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np

__all__ = ["read_feature_stats", "read_cell_stats", "columns_owner"]

_GUARD_PREFIX = "importer_stats:"
_COUNTING_PREFIX = "counting:"

# Reading the columns is not unconditionally cheaper. The table scan costs one
# row per feature; the matrix pass costs one entry per non-zero. Measured on
# this box over a 4,854-cell x 5.45M-tile matrix: 0.84 us per table row against
# 0.039 us per non-zero, so the two cost the same at about 21 non-zeros per
# column, and below that streaming wins.
#
# The gate sits well under that, because the measurement was warm-cache and the
# comparison is not symmetric when it is not: the matrix is always the larger
# object on disk -- 330 MB against a few MB of columns here, 34 GB on a real
# tile matrix -- so a cold read swings hard toward the columns, and mean
# non-zeros per column grows with the cell count, which is exactly when the
# matrix pass becomes unaffordable.
_MIN_NNZ_PER_COLUMN = 8


def _singular(entity: str) -> str:
    return entity[:-1] if entity.endswith("s") else entity


def _entity_for(ds, modality: str) -> Optional[Tuple[str, str]]:
    """(entity table, its primary key column) for a modality, or None."""
    try:
        from cytome import modality_feature_table_info
        table, idx_col, _ = modality_feature_table_info(ds, modality)
        return table, idx_col
    except Exception:
        return None


def _matrix_meta(conn, matrix_name):
    return conn.execute(
        "SELECT n_rows, n_cols, n_nonzero FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()


def _scalar(conn, key):
    row = conn.execute(
        "SELECT value FROM _metadata WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _guard(conn, matrix_name):
    """The matrix's guard if all five fields still match the matrix, else None."""
    guard = _scalar(conn, _GUARD_PREFIX + matrix_name)
    if not isinstance(guard, dict):
        return None
    meta = _matrix_meta(conn, matrix_name)
    if meta is None:
        return None
    counting = _scalar(conn, _COUNTING_PREFIX + matrix_name)
    # n_nonzero is the content fingerprint: the other four fields catch every
    # rebuild that changes the shape or the convention, this one catches a
    # matrix rewritten in place with both unchanged.
    ok = (
        guard.get("matrix") == matrix_name
        and guard.get("n_rows") == meta[0]
        and guard.get("n_cols") == meta[1]
        and guard.get("n_nonzero") == meta[2]
        and guard.get("counting") == counting
    )
    return guard if ok else None


def _guard_ok(conn, matrix_name) -> bool:
    return _guard(conn, matrix_name) is not None


def _holds_the_matrix(nnz_column, total_column, guard) -> bool:
    """Whether per-feature (or per-cell) columns sum to the guarded matrix's
    number of non-zeros and, when the guard records it, to its total count, as
    the matrix's own sums would. Guards written before the total was recorded
    (and the ones a filtered store's recomputation writes) are checked on the
    non-zeros alone."""
    if int(np.asarray(nnz_column, dtype=np.int64).sum()) != int(guard["n_nonzero"]):
        return False
    total = guard.get("total")
    if total is None:
        return True
    # a writer may have stored the totals as float32; a matrix that is not the
    # guarded one is off by far more than that rounding
    got = float(np.asarray(total_column, dtype=np.float64).sum())
    return abs(got - float(total)) <= 1e-6 * max(abs(float(total)), 1.0)


def _read_columns(conn, table, idx_col, columns, n_expected):
    """Every column as an array ordered by index, or None if any is unusable.

    The rows arrive in batches straight into arrays of the final size: a tile
    table has millions of rows, and a list of every row's tuple cost more
    memory (about 120 bytes a row) than the arrays it was converted into, the
    one thing reading columns instead of the matrix should not cost.

    A NULL anywhere means the column was only partly written, which would read
    back as a zero and quietly understate a feature. ``np.array`` turns a
    ``None`` into a NaN rather than raising, so the check is explicit -- and
    it catches a stored NaN too, which is no more trustworthy.
    """
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if not set(columns).issubset(have):
        return None
    sel = ", ".join(columns)
    cur = conn.execute(f"SELECT {sel} FROM {table} ORDER BY {idx_col}")
    out = np.empty((n_expected, len(columns)), dtype=np.float64)
    pos = 0
    while True:
        rows = cur.fetchmany(65536)
        if not rows:
            break
        try:
            block = np.array(rows, dtype=np.float64)
        except (TypeError, ValueError):
            return None      # a NULL or a non-number in one of the columns
        if block.ndim != 2 or block.shape[1] != len(columns) or pos + len(block) > n_expected:
            return None
        out[pos:pos + len(block)] = block
        pos += len(block)
    if pos != n_expected or not np.isfinite(out).all():
        return None          # a NULL or a NaN in one of the columns
    return [np.ascontiguousarray(out[:, j]) for j in range(len(columns))]


def read_feature_stats(ds, modality: str, layer: str = "counts"):
    """``(n_cells_per_feature, feature_total_counts)`` or ``None``.

    ``None`` means the caller should compute them by streaming -- either
    because nothing trustworthy is stored, or because for this matrix the
    stream is the cheaper of the two.
    """
    matrix_name = f"{modality}_{layer}"
    conn = ds._conn
    guard = _guard(conn, matrix_name)
    if guard is None:
        return None
    _shape = conn.execute(
        "SELECT n_nonzero, n_cols FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    if _shape is None or not _shape[1]:
        return None
    if (_shape[0] or 0) < _MIN_NNZ_PER_COLUMN * _shape[1]:
        return None
    ent = _entity_for(ds, modality)
    if ent is None:
        return None
    table, idx_col = ent
    meta = _matrix_meta(conn, matrix_name)
    cols = _read_columns(
        conn, table, idx_col,
        ["n_cells_accessible", f"{_singular(table)}_total_counts"],
        meta[1],
    )
    if cols is None or not _holds_the_matrix(cols[0], cols[1], guard):
        return None
    return cols[0].astype(np.int64), cols[1]


def read_cell_stats(ds, modality: str, layer: str = "counts"):
    """``(n_features_per_cell, cell_total_counts)`` or ``None``."""
    matrix_name = f"{modality}_{layer}"
    conn = ds._conn
    guard = _guard(conn, matrix_name)
    if guard is None:
        return None
    ent = _entity_for(ds, modality)
    if ent is None:
        return None
    table, _ = ent
    meta = _matrix_meta(conn, matrix_name)
    cols = _read_columns(
        conn, "cells", "cell_idx",
        [f"n_{table}_accessible", f"{_singular(table)}_total_counts"],
        meta[0],
    )
    if cols is None or not _holds_the_matrix(cols[0], cols[1], guard):
        return None
    return cols[0].astype(np.int64), cols[1]


def columns_owner(ds, table: str) -> Optional[str]:
    """The matrix whose sums a feature table's statistics columns hold, or None.

    That is a matrix over ``table`` whose guard holds and whose non-zero count
    the table's ``n_cells_accessible`` sums to. A writer of those columns asks
    before writing another matrix's sums into them: the guard would go on
    vouching for its own matrix.
    """
    conn = ds._conn
    try:
        names = [r[0] for r in conn.execute(
            "SELECT matrix_name FROM matrix_meta WHERE col_entity = ?", (table,))]
        nnz, tot = conn.execute(
            f"SELECT SUM(n_cells_accessible), SUM({_singular(table)}_total_counts) FROM {table}"
        ).fetchone()
    except Exception:
        return None
    if nnz is None or tot is None:
        return None
    for name in names:
        guard = _guard(conn, name)
        if guard is not None and _holds_the_matrix([nnz], [tot], guard):
            return name
    return None
