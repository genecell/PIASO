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
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np

__all__ = ["read_feature_stats", "read_cell_stats"]

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


def _guard_ok(conn, matrix_name) -> bool:
    guard = _scalar(conn, _GUARD_PREFIX + matrix_name)
    if not isinstance(guard, dict):
        return False
    meta = _matrix_meta(conn, matrix_name)
    if meta is None:
        return False
    counting = _scalar(conn, _COUNTING_PREFIX + matrix_name)
    # n_nonzero is the content fingerprint: the other four fields catch every
    # rebuild that changes the shape or the convention, this one catches a
    # matrix rewritten in place with both unchanged.
    return (
        guard.get("matrix") == matrix_name
        and guard.get("n_rows") == meta[0]
        and guard.get("n_cols") == meta[1]
        and guard.get("n_nonzero") == meta[2]
        and guard.get("counting") == counting
    )


def _read_columns(conn, table, idx_col, columns, n_expected):
    """Every column as an array ordered by index, or None if any is unusable.

    The conversion is one ``np.array`` over the whole result rather than a
    comprehension per column: a tile table has millions of rows, and at that
    size the per-row Python work, not the query, is what the caller would be
    paying to avoid a matrix pass.

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
    cur.arraysize = 65536
    rows = cur.fetchall()
    if len(rows) != n_expected:
        return None
    try:
        arr = np.array(rows, dtype=np.float64)
    except TypeError:
        return None          # a NULL in one of the columns
    if arr.ndim != 2 or arr.shape[1] != len(columns):
        return None
    if not np.isfinite(arr).all():
        return None          # a NULL or a NaN in one of the columns
    return [np.ascontiguousarray(arr[:, j]) for j in range(len(columns))]


def read_feature_stats(ds, modality: str, layer: str = "counts"):
    """``(n_cells_per_feature, feature_total_counts)`` or ``None``.

    ``None`` means the caller should compute them by streaming -- either
    because nothing trustworthy is stored, or because for this matrix the
    stream is the cheaper of the two.
    """
    matrix_name = f"{modality}_{layer}"
    conn = ds._conn
    if not _guard_ok(conn, matrix_name):
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
    if cols is None:
        return None
    return cols[0].astype(np.int64), cols[1]


def read_cell_stats(ds, modality: str, layer: str = "counts"):
    """``(n_features_per_cell, cell_total_counts)`` or ``None``."""
    matrix_name = f"{modality}_{layer}"
    conn = ds._conn
    if not _guard_ok(conn, matrix_name):
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
    if cols is None:
        return None
    return cols[0].astype(np.int64), cols[1]
