"""Content digests for results cached in a store's metadata: one invalidation rule for derived caches.

A cache is trusted only while the inputs it was computed from are unchanged. Its entry records a digest of those
inputs (the labels, the number of cells, the identity of each matrix it read, the code's version and parameters),
and a reader recomputes the digest and treats a mismatch as a miss. A key built from parameter names alone survives
a relabel, a subset (cytome's ``subset`` copies the metadata table) and a re-quantification, and then hands back a
result computed from other data (code audit of 2026-10-27, part 3; review I1, improvement 1).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


def matrix_identity(ds, matrix_name: str):
    """A matrix's identity from ``matrix_meta``: its name, shape, non-zero count, dtype and creation time (every
    writer stamps ``created_at``, so a re-quantified or rewritten matrix differs). ``None`` when the store has no
    such matrix."""
    try:
        row = ds._conn.execute(
            "SELECT n_rows, n_cols, n_nonzero, dtype, created_at FROM matrix_meta WHERE matrix_name = ?",
            (matrix_name,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {"matrix": matrix_name, "n_rows": int(row[0]), "n_cols": int(row[1]), "n_nonzero": int(row[2]),
            "dtype": str(row[3]), "created_at": None if row[4] is None else str(row[4])}


def cell_column(ds, column: str) -> list:
    """A column of the cells table in ``cell_idx`` order (the caller checks that the column exists)."""
    return [r[0] for r in ds._conn.execute(f'SELECT "{column}" FROM cells ORDER BY cell_idx')]


def labels_digest(values) -> str:
    """Digest of a label column in cell order. A missing label (None or NaN) has a token of its own, so labelling a
    cell, or removing its label, changes the digest."""
    s = pd.Series(list(values), dtype=object)
    missing = s.isna().to_numpy()
    text = s.astype(str).to_numpy(dtype=object)
    text[missing] = "\x00NA"
    h = hashlib.blake2b(digest_size=16)
    h.update(str(len(text)).encode())
    h.update("\x1f".join(text.tolist()).encode("utf-8"))
    return h.hexdigest()


def digest(inputs) -> str:
    """Digest of a JSON-serialisable description of a cache's inputs (key order does not matter)."""
    return hashlib.blake2b(json.dumps(inputs, sort_keys=True, default=_json_default).encode("utf-8"),
                           digest_size=16).hexdigest()


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
