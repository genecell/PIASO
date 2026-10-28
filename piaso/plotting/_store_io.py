"""Read-only access to a store for the plots.

A plot only reads, so it opens a store with ``mode="r"``: the file is never
changed (a downloaded store keeps its checksum) and a read-only file opens
without a warning. ``piaso.utils._cytome_compat.read_cells_columns`` opens
read-write, which every grouped plot reached through it (audit of 2026-10-27,
review I2, finding 18).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..utils._cytome_compat import open_cytome

__all__ = ["read_cells_columns", "read_cells_column", "get_metadata",
           "require_cells_column"]


def require_cells_column(source, column, *, caller: str, param: str = "groupby") -> str:
    """``column`` as the store spells it, or a ValueError that names it.

    Run before any SQL sees the name: SQLite reads a double-quoted name that
    matches no column as a string literal, so ``SELECT "no_such_col" FROM
    cells`` returns the text 'no_such_col' for every cell, and a plot grouped
    by it drew one group, named after the column, that held every cell.
    """
    with open_cytome(source, mode="r") as ds:
        columns = sorted(ds.cells.columns)
        actual = None if column is None else ds.cells.resolve_column(str(column))
        where = getattr(ds, "path", None)
    if actual is not None:
        return actual
    store = f" of {where}" if where else ""
    if column is None:
        raise ValueError(f"{caller}: {param} is required with a store; name a column "
                         f"of the cells table{store}. Its columns: {columns}")
    raise ValueError(f"{caller}: {param}={column!r} is not a column of the cells "
                     f"table{store}. Its columns: {columns}")


def read_cells_columns(source, columns: Iterable[str]) -> pd.DataFrame:
    """Columns of ``ds.cells`` as a DataFrame; raises ValueError on a missing one."""
    cols = list(columns)
    with open_cytome(source, mode="r") as ds:
        missing = [c for c in cols if c not in ds.cells]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in cytome cells table. "
                f"Available: {sorted(ds.cells.columns)}"
            )
        return pd.DataFrame({c: np.asarray(ds.cells[c]) for c in cols})


def read_cells_column(source, column: str) -> np.ndarray:
    """One ``cells`` column as a numpy array."""
    return read_cells_columns(source, [column])[column].values


def get_metadata(source, key: str, default=None):
    """A value from ``ds.metadata``, or ``default``."""
    with open_cytome(source, mode="r") as ds:
        return ds.metadata.get(key, default)
