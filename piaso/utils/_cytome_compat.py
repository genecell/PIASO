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

Why several helpers rather than one
-----------------------------------
They are not variations on a theme; they return different things, because
callers need different things:

===========================  =====================================  ==============
helper                       question it answers                    returns
===========================  =====================================  ==============
``is_cytome_input``          take the cytome branch at all?         bool
``_is_cytome_dataset_obj``   is it already open (do not reopen)?    bool
``open_cytome``              give me a Dataset, manage its life     context manager
``open_cytome_sync``         give me a Dataset, I will close it     Dataset
``resolve_cytome_path``      I need a file path (subprocess)        str
``resolve_cytome_conn``      I need raw SQL                         (conn, owns_it)
===========================  =====================================  ==============

Folding them into one function with a mode argument would move the branch to
every call site and lose the return type, which is the part that stops
mistakes.

What matters is that this module is the *only* place any of it is decided.
Four modules once carried their own ``_is_cytome`` -- three identical
``isinstance`` checks and one duck-type on ``iter_chunks`` -- and none noticed
a closed dataset, so that failed later as a sqlite error several frames from
the cause. ``tests/test_cytome_helpers_are_shared.py`` fails if a private one
reappears.

Every path/Dataset fork goes through ``_is_path_like``, so ``str``,
``pathlib.Path`` and ``Dataset`` mean the same thing everywhere.
"""

from __future__ import annotations

import os
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


def _is_path_like(obj) -> bool:
    """True for a filesystem path in any of the shapes callers actually pass.

    ``isinstance(obj, str)`` was the test everywhere, so ``pathlib.Path`` fell
    through to the Dataset branch and was used as though it were one -- failing
    later, somewhere else, with a message about the wrong thing.
    """
    return isinstance(obj, (str, os.PathLike))


def is_cytome_input(obj) -> bool:
    """Return True if ``obj`` is a cytome ``Dataset`` or a path to a .cytome file.

    If ``obj`` is a *closed* cytome Dataset, raises ``RuntimeError`` with
    an actionable message instead of silently returning True. Path strings
    are not probed (no I/O).
    """
    if _is_path_like(obj):
        name = os.fspath(obj)
        return name.endswith('.cytome') or name.endswith('.db')
    if _looks_like_cytome_dataset_class(obj):
        _assert_cytome_open(obj)
        return True
    return False


def _is_cytome_dataset_obj(obj) -> bool:
    """True iff ``obj`` is a cytome Dataset instance (not a string).

    If ``obj`` is a *closed* cytome Dataset, raises ``RuntimeError`` with
    an actionable message instead of silently returning True.
    """
    if _is_path_like(obj):
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
    if _is_path_like(source):
        import cytome
        ds = cytome.open(os.fspath(source))
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
    if _is_cytome_dataset_obj(path):
        # Returning it would be worse than failing: the contract here is
        # "caller owns the close", and the caller would then close a Dataset
        # it did not open. open_cytome() is the one that knows the difference.
        raise TypeError(
            "open_cytome_sync() takes a path; it was given an open Dataset, "
            "and its caller closes what it returns. Use open_cytome() as a "
            "context manager, which closes only what it opened.")
    try:
        import cytome
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'cytome' package is required to open a cytome file. "
            "Install with: pip install cytome"
        ) from exc
    return cytome.open(os.fspath(path))


def resolve_cytome_path(source) -> str:
    """Return the filesystem path to a cytome regardless of input shape.

    Accepts a ``Dataset`` instance (returns ``str(source.path)``) or a
    string path (returns it unchanged). Raises ``TypeError`` for other
    inputs.

    Designed for functions that shell out to a subprocess (e.g. PICCO,
    ``cytome-import-fragments``, ``cytome-quantify-peaks``) which
    require an on-disk path.
    """
    if _is_path_like(source):
        return os.fspath(source)
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
    if _is_path_like(source):
        return sqlite3.connect(os.fspath(source)), True
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


def write_cells_column(ds, name: str, values, *, overwrite: bool = False,
                       what: str = "column") -> str:
    """Write a per-cell column without landing on somebody else's.

    SQLite column names are case-insensitive, and cytome resolves a new
    spelling to an existing column with only a warning. So a QC step that
    writes ``nucleosome_signal`` into a cytome that already carries Signac's
    ``Nucleosome_signal`` would silently replace it. Two rules:

    * the **same spelling** overwrites, silently -- re-running a QC step
      replaces its own column, which is what re-running means;
    * a **different spelling** that resolves to an existing column is
      refused unless ``overwrite=True``, naming both spellings.

    Returns the name actually written.
    """
    import numpy as np
    resolved = check_cells_column_free(ds, name, overwrite=overwrite, what=what)
    ds.cells[name] = np.asarray(values)
    # cytome buffers a new column until flush; flush now so the schema the
    # next check reads already contains this one.
    ds.flush()
    return resolved


def check_cells_column_free(ds, name: str, *, overwrite: bool = False,
                            what: str = "column") -> str:
    """The check behind :func:`write_cells_column`, without the write.

    Returns the column the name will land on. Call it for every column a
    function intends to write *before* writing the first, so a refusal
    leaves the cytome untouched rather than half-written.
    """
    # The schema itself, not a cached column list: a column written a moment
    # ago must already count.
    existing = [r[1] for r in ds._conn.execute("PRAGMA table_info(cells)")]
    same = [c for c in existing if c == name]
    variants = [c for c in existing if c != name and c.lower() == str(name).lower()]
    if variants and not same and not overwrite:
        raise ValueError(
            f"{what} {name!r} would overwrite the existing column {variants[0]!r}: "
            f"SQLite column names are case-insensitive, so the two are one "
            f"column. Pass overwrite=True to replace it, or choose another "
            f"name (key_added=...).")
    return variants[0] if (variants and not same) else name


def require_cells_column(source, column: str, argument: str = "groupby") -> None:
    """Raise unless ``column`` exists in ``ds.cells``.

    This exists because of a SQLite compatibility misfeature, and it is worth
    stating plainly: **a double-quoted token that does not resolve to a column
    is treated as a string literal, not an error.** So

        SELECT cell_idx, "Leiden" FROM cells        -- no Leiden column

    returns the string ``'Leiden'`` for every row instead of failing. A typo in
    a ``groupby`` then produces one group named after the typo, and everything
    downstream is internally consistent -- a confident wrong answer rather than
    a traceback. That is exactly how a COSG run reported ``1 groups`` and a
    plausible marker table for a column that did not exist.

    Any function that interpolates a user-supplied name into a quoted
    identifier should call this first.
    """
    with open_cytome(source) as ds:
        if column in ds.cells:
            return
        available = sorted(ds.cells.columns)
    close = [c for c in available if c.lower() == str(column).lower()]
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise KeyError(
        f"{argument}={column!r} is not a column of the cytome cells table."
        f"{hint} Available: {available}"
    )


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


def require_row_major(ds, modality: str, layer: str, what: str,
                      alternative: str = "") -> None:
    """Raise an actionable error when ``what`` needs rows and there are none.

    For operations that genuinely require row chunks -- ones that write a
    row-major layer, or transform cell by cell -- and for which assembling
    rows from columns would mean reading the whole matrix once per row block.
    The generic refusal from ``iter_rows`` names the layout but not the way
    out; this names both, and fires before any work is done rather than
    partway through a stream.
    """
    if column_major_layer(ds, modality, layer) is None:
        return
    tail = f" {alternative}" if alternative else ""
    raise ValueError(
        f"{what} needs row chunks, and '{modality}_{layer}' is stored "
        f"column-major (the fragment importer and the peak quantifier write "
        f"peaks and tiles that way by default), so it has none. Building rows "
        f"from columns would read the whole matrix once per row block.{tail}"
    )


def column_major_layer(ds, modality: str, layer: str):
    """The layer as a ``MeasurementLayer`` when it is stored column-major, else ``None``.

    A matrix written with ``--layout csc`` has CSC chunks and no row chunks;
    ``iter_chunks`` refuses it, and a consumer that knows the layout walks it
    by columns instead.
    """
    # Takes a path or an open Dataset: callers hold one or the other, and a
    # helper that only accepted a Dataset turned every guarded call site into
    # an AttributeError on the string path.
    conn, opened = resolve_cytome_conn(ds)
    try:
        row = conn.execute(
            "SELECT COALESCE(n_chunks, 0), COALESCE(has_csc, 0) FROM matrix_meta "
            "WHERE matrix_name = ?", (f"{modality}_{layer}",),
        ).fetchone()
        if row is None or row[0] != 0 or not row[1]:
            return None
    except Exception:
        return None
    finally:
        if opened:
            conn.close()
    from cytome.core.measurement import MeasurementLayer
    if opened:
        # A path caller gets a layer on its own connection: the one used for
        # the probe above is closed, and a MeasurementLayer needs a live one.
        import sqlite3
        live = sqlite3.connect(f"file:{resolve_cytome_path(ds)}?mode=ro", uri=True)
        return MeasurementLayer(live, f"{modality}_{layer}")
    return MeasurementLayer(ds._conn, f"{modality}_{layer}")
