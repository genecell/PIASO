"""Shared cytome compatibility helpers for piaso.

Centralises the duplicated `_is_cytome_input` checks and provides small
accessors that read from cytome via the public `ds.cells[col]` /
`ds.matrix_meta` API instead of raw SQL.

Two resolvers are exported for code that needs the underlying file path
(subprocess invocations) or SQLite connection (direct-SQL utilities):

- `resolve_cytome_path(source)` — return the .cytome filesystem path,
  whether `source` is already a path or a `Dataset` instance.
- `resolve_cytome_conn(source)` — return `(conn, owns_it)` so a caller
  can do raw SQL without leaking a connection.

These two are the surface that lets STR_ONLY-via-subprocess and
SQLite-bound functions accept either a `Dataset` or a string path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Tuple

import numpy as np
import pandas as pd


def _looks_like_cytome_dataset_class(obj) -> bool:
    """Pure type test (no liveness probe). Used internally by
    :func:`is_cytome_input` and :func:`_is_cytome_dataset_obj` so that
    those public predicates can also assert the dataset is still open.
    """
    if isinstance(obj, str):
        return False
    cls_name = type(obj).__name__
    module = type(obj).__module__ or ""
    return "cytome" in module.lower() and "dataset" in cls_name.lower()


def _assert_cytome_open(obj) -> None:
    """If ``obj`` is a cytome Dataset that has been ``close()``d, raise an
    actionable :class:`RuntimeError` pointing the user at how to recover.

    Called from :func:`is_cytome_input` and :func:`_is_cytome_dataset_obj`
    so that every PIASO function which branches on "is this a cytome?"
    surfaces the closed state immediately, rather than producing a
    confusing ``sqlite3.ProgrammingError: Cannot operate on a closed
    database`` deep in a downstream SQL call.

    Closed cytomes are NOT auto-reopened: ``ds.close()`` is explicit user
    intent. The safe-recovery paths (NFS-aware reopen, lock checks) are
    best left to the caller.
    """
    if not _looks_like_cytome_dataset_class(obj):
        return
    # Prefer the dataset's own diagnostic when present (cytome >= 0.9.x);
    # fall back to a probe + raise for older cytomes that lack _check_open.
    check = getattr(obj, "_check_open", None)
    if callable(check):
        check()
        return
    is_closed = getattr(obj, "is_closed", None)
    if is_closed is True:
        path = getattr(obj, "path", "<unknown>")
        raise RuntimeError(
            f"cytome Dataset is closed (path: {path}). "
            f"Re-open with cytome.open(...) and retry, or pass the path "
            f"string directly to the function."
        )


def is_cytome_input(obj) -> bool:
    """Return True if ``obj`` is a cytome ``Dataset`` or a path to a .cytome file.

    If ``obj`` is a *closed* cytome Dataset, raises ``RuntimeError`` with
    an actionable message instead of silently returning True. Path strings
    are not probed (no I/O).
    """
    if isinstance(obj, str):
        return obj.endswith('.cytome') or obj.endswith('.db')
    if _looks_like_cytome_dataset_class(obj):
        _assert_cytome_open(obj)
        return True
    return False


def _is_cytome_dataset_obj(obj) -> bool:
    """True iff ``obj`` is a cytome Dataset instance (not a string).

    If ``obj`` is a *closed* cytome Dataset, raises ``RuntimeError`` with
    an actionable message instead of silently returning True.
    """
    if isinstance(obj, str):
        return False
    if _looks_like_cytome_dataset_class(obj):
        _assert_cytome_open(obj)
        return True
    return False


@contextmanager
def open_cytome(source) -> Iterator:
    """Context manager that yields a cytome Dataset.

    If ``source`` is a string path, opens the dataset and closes on exit.
    If ``source`` is already a Dataset, yields it unchanged and does not close.
    """
    if isinstance(source, str):
        import cytome
        ds = cytome.open(source)
        try:
            yield ds
        finally:
            ds.close()
    else:
        yield source


def open_cytome_sync(path):
    """Open a cytome file from a string path; the caller owns the close.

    Bare-function counterpart to ``open_cytome`` (the context manager).
    Use this when a function needs to hold a Dataset across multiple
    statements but the existing call structure makes a ``with`` block
    inconvenient. Raises ``ImportError`` with a clean message if the
    ``cytome`` package isn't installed.

    Parameters
    ----------
    path : str
        Filesystem path to a ``.cytome``/``.db`` file.

    Returns
    -------
    cytome.Dataset
        Open dataset. Caller is responsible for ``ds.close()``.
    """
    try:
        import cytome
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'cytome' package is required to open a cytome file. "
            "Install with: pip install cytome"
        ) from exc
    return cytome.open(path)


def resolve_cytome_path(source) -> str:
    """Return the filesystem path to a cytome regardless of input shape.

    Accepts a ``Dataset`` instance (returns ``str(source.path)``) or a
    string path (returns it unchanged). Raises ``TypeError`` for other
    inputs.

    Designed for functions that shell out to a subprocess (e.g. PICCO,
    ``cytome-import-fragments``, ``cytome-quantify-peaks``) which
    require an on-disk path.
    """
    if isinstance(source, str):
        return source
    if _is_cytome_dataset_obj(source):
        return str(source.path)
    raise TypeError(
        f"Expected str path or cytome.Dataset, got {type(source).__name__}"
    )


def resolve_cytome_conn(source) -> Tuple["sqlite3.Connection", bool]:  # noqa: F821
    """Return ``(conn, owns_it)`` for a cytome source.

    If ``source`` is a Dataset, returns its already-open ``_conn`` and
    ``owns_it=False`` (caller must NOT close it). If ``source`` is a
    string path, opens a new ``sqlite3`` connection and returns
    ``owns_it=True``.

    Designed for functions that need direct SQL access (e.g.
    ``generateBigWigFromCytome`` reading from ``fragment_chunks``).
    """
    import sqlite3
    if isinstance(source, str):
        return sqlite3.connect(source), True
    if _is_cytome_dataset_obj(source):
        return source._conn, False
    raise TypeError(
        f"Expected str path or cytome.Dataset, got {type(source).__name__}"
    )


def read_cells_columns(source, columns: Iterable[str]) -> pd.DataFrame:
    """Read one or more columns from ``ds.cells`` as a DataFrame.

    Uses the public ``ds.cells['col']`` API (no raw SQL). Validates that
    every requested column exists; raises ``ValueError`` otherwise.

    Parameters
    ----------
    source
        cytome Dataset or path to a .cytome file.
    columns
        Iterable of column names to fetch.
    """
    cols = list(columns)
    with open_cytome(source) as ds:
        missing = [c for c in cols if c not in ds.cells]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in cytome cells table. "
                f"Available: {sorted(ds.cells.columns)}"
            )
        return pd.DataFrame({c: np.asarray(ds.cells[c]) for c in cols})


def read_cells_column(source, column: str) -> np.ndarray:
    """Read a single ``cells`` column as a numpy array."""
    return read_cells_columns(source, [column])[column].values


def has_cells_column(source, column: str) -> bool:
    """Return True if ``column`` exists in ``ds.cells``."""
    with open_cytome(source) as ds:
        return column in ds.cells


def get_metadata(source, key: str, default=None):
    """Fetch a value from ``ds.metadata`` with a default fallback."""
    with open_cytome(source) as ds:
        return ds.metadata.get(key, default)


def resolve_cytome_alias(cytome, old_value, old_name, func_name):
    """Soft-deprecation shim for the renamed cytome parameter.

    The public path/Dataset parameter is now uniformly ``cytome`` across
    piaso (picco, importFragments, quantifyPeakActivity, processFragment).
    The previous names (``cytome_path`` / ``output_cytome``) still work but
    emit a ``DeprecationWarning`` and map onto ``cytome`` (``cytome`` wins if
    both are given). Returns the resolved value.
    """
    import warnings

    if old_value is not None:
        warnings.warn(
            f"{func_name}(): `{old_name}=` is deprecated; use `cytome=` "
            f"(accepts a path or an open cytome.Dataset). `{old_name}=` still "
            f"works for now but will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        if cytome is None:
            cytome = old_value
    return cytome
