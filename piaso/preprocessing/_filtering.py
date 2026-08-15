"""Cell and feature filtering utilities.

Provides ``filter_cells`` and ``filter_features`` as scanpy-free
replacements for ``sc.pp.filter_cells`` and ``sc.pp.filter_genes``.

``filter_cells`` is generalised: in addition to the count / feature
thresholds it accepts a ``mask`` kwarg with several shapes
(bool array, integer indices, pandas Series, query string, dict,
or callable). The mask sources AND-compose with the threshold
kwargs to give a single ``keep`` mask.

For cytome inputs, simple query strings and dict masks are pushed
down to SQL ``WHERE`` clauses — only the matching ``cell_idx`` rows
are materialised, keeping peak RAM bounded for million- and
hundred-million-cell datasets.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# backend detection
# ---------------------------------------------------------------------------

def _is_cytome(data) -> bool:
    try:
        from cytome.core.dataset import CytomeDataset
    except ImportError:
        return False
    return isinstance(data, CytomeDataset)


# ---------------------------------------------------------------------------
# SQL pushdown: pandas-query subset → SQL WHERE
# ---------------------------------------------------------------------------

class _QueryTooComplex(Exception):
    """Sentinel raised when a query string can't be translated to SQL.

    The caller falls back to the streaming pandas-eval path.
    """


_CMP_OP_TO_SQL = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _query_to_sql_where(query: str, valid_cols: set[str]) -> tuple[str, list]:
    """Translate a subset of pandas-query syntax to ``(where_sql, params)``.

    Supported nodes:

    - boolean ops: ``and`` / ``or`` / ``not`` (and the ``& | ~`` aliases)
    - comparisons: ``== != < <= > >=`` (single-comparator only)
    - method calls: ``col.isin([a, b, ...])``, ``col.between(a, b)``
    - membership: ``col in [a, b, ...]``  (incl. ``not in``)
    - literals: int / float / str / bool / None

    Raises :class:`_QueryTooComplex` for anything outside this subset;
    callers fall back to the streaming pandas path.
    """
    tree = ast.parse(query, mode="eval").body
    params: list[Any] = []

    def emit(node: ast.AST) -> str:
        if isinstance(node, ast.BoolOp):
            join = " AND " if isinstance(node.op, ast.And) else " OR "
            return "(" + join.join(emit(v) for v in node.values) + ")"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return f"(NOT {emit(node.operand)})"
            if isinstance(node.op, ast.Invert):
                return f"(NOT {emit(node.operand)})"
            raise _QueryTooComplex(f"unary op: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            # pandas-query parser treats `a & b` and `a | b` as BinOp on
            # bool-ish nodes; map them to AND/OR so users can write either.
            if isinstance(node.op, ast.BitAnd):
                return f"({emit(node.left)} AND {emit(node.right)})"
            if isinstance(node.op, ast.BitOr):
                return f"({emit(node.left)} OR {emit(node.right)})"
            raise _QueryTooComplex(f"binop: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            # We only handle single comparators (a < b), not chained
            # (a < b < c) — translating chains needs auxiliary parens.
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise _QueryTooComplex("chained comparison")
            op = node.ops[0]
            left = emit(node.left)
            right_node = node.comparators[0]
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(right_node, (ast.List, ast.Tuple, ast.Set)):
                    raise _QueryTooComplex("`in` rhs must be a literal list/tuple/set")
                values = [_const(v) for v in right_node.elts]
                placeholders = ", ".join(["?"] * len(values))
                params.extend(values)
                op_sql = "IN" if isinstance(op, ast.In) else "NOT IN"
                return f"({left} {op_sql} ({placeholders}))"
            op_sql = _CMP_OP_TO_SQL.get(type(op))
            if op_sql is None:
                raise _QueryTooComplex(f"comparator: {type(op).__name__}")
            return f"({left} {op_sql} {emit(right_node)})"
        if isinstance(node, ast.Call):
            # Support: col.isin([...]), col.between(lo, hi)
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                col = func.value.id
                if col not in valid_cols:
                    raise _QueryTooComplex(f"unknown column: {col!r}")
                if func.attr == "isin":
                    if len(node.args) != 1 or not isinstance(
                        node.args[0], (ast.List, ast.Tuple, ast.Set)
                    ):
                        raise _QueryTooComplex("isin arg must be a literal list")
                    values = [_const(v) for v in node.args[0].elts]
                    placeholders = ", ".join(["?"] * len(values))
                    params.extend(values)
                    return f"([{col}] IN ({placeholders}))"
                if func.attr == "between":
                    if len(node.args) != 2:
                        raise _QueryTooComplex("between takes 2 args")
                    lo = _const(node.args[0])
                    hi = _const(node.args[1])
                    params.extend([lo, hi])
                    return f"([{col}] >= ? AND [{col}] <= ?)"
            raise _QueryTooComplex(f"call: {ast.dump(node)}")
        if isinstance(node, ast.Name):
            if node.id == "True":
                return "1"
            if node.id == "False":
                return "0"
            if node.id not in valid_cols:
                raise _QueryTooComplex(f"unknown column: {node.id!r}")
            return f"[{node.id}]"
        if isinstance(node, ast.Constant):
            params.append(node.value)
            return "?"
        if isinstance(node, ast.Attribute):
            raise _QueryTooComplex(f"attribute access: {ast.dump(node)}")
        raise _QueryTooComplex(f"unsupported: {ast.dump(node)}")

    def _const(n: ast.AST) -> Any:
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(
            n.operand, ast.Constant
        ):
            return -n.operand.value
        raise _QueryTooComplex(f"non-literal in collection: {ast.dump(n)}")

    return emit(tree), params


def _as_range(val):
    """If ``val`` is a numeric range 2-tuple, return ``(lo, hi)``; else ``None``.

    A range tuple is a 2-tuple where each bound is numeric **or** ``None``, with
    at least one numeric bound. ``None`` means an open-ended bound:

    - ``(lo, hi)``   → ``col >= lo AND col <= hi``
    - ``(lo, None)`` → ``col >= lo``
    - ``(None, hi)`` → ``col <= hi``

    Only tuples match — a 2-element **list** ``[lo, hi]`` is left for ``isin``
    (the documented escape hatch).
    """
    if not (isinstance(val, tuple) and len(val) == 2):
        return None
    lo, hi = val

    def _num_or_none(x):
        return x is None or isinstance(x, (int, float, np.integer, np.floating))

    if _num_or_none(lo) and _num_or_none(hi) and not (lo is None and hi is None):
        return lo, hi
    return None


def _dict_to_sql_where(mask_dict: dict, valid_cols: set[str]) -> tuple[str, list]:
    """Translate a dict-shaped mask spec to a SQL WHERE clause.

    Per-key value semantics:

    - scalar             → equality
    - list / set         → ``IN (...)``
    - numeric 2-tuple    → range; open-ended via ``None``:
      ``(lo, hi)`` → ``col >= lo AND col <= hi``; ``(lo, None)`` → ``col >= lo``;
      ``(None, hi)`` → ``col <= hi``. (Use a 2-element list ``[lo, hi]`` to force
      ``IN`` instead.)
    """
    if not mask_dict:
        return "1", []
    clauses: list[str] = []
    params: list[Any] = []
    for col, val in mask_dict.items():
        if col not in valid_cols:
            raise KeyError(
                f"mask dict: column {col!r} not found in cells. "
                f"Available: {sorted(valid_cols)}"
            )
        rng = _as_range(val)
        if rng is not None:
            lo, hi = rng
            if lo is not None and hi is not None:
                clauses.append(f"([{col}] >= ? AND [{col}] <= ?)")
                params.extend([lo, hi])
            elif lo is not None:
                clauses.append(f"([{col}] >= ?)")
                params.append(lo)
            else:
                clauses.append(f"([{col}] <= ?)")
                params.append(hi)
        elif isinstance(val, (list, set, np.ndarray)):
            values = list(val)
            placeholders = ", ".join(["?"] * len(values))
            clauses.append(f"([{col}] IN ({placeholders}))")
            params.extend(values)
        else:
            clauses.append(f"([{col}] = ?)")
            params.append(val)
    return " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# mask shape resolution (AnnData path)
# ---------------------------------------------------------------------------

def _resolve_mask_anndata(
    mask, obs_df: pd.DataFrame, n_obs: int
) -> tuple[np.ndarray | None, str | None]:
    """Resolve any of the supported ``mask`` shapes to a bool array.

    Returns ``(bool_array, description_string)`` or ``(None, None)``
    when ``mask is None``.
    """
    if mask is None:
        return None, None

    if callable(mask) and not isinstance(mask, (np.ndarray, pd.Series)):
        result = mask(obs_df)
        return _coerce_bool(result, n_obs), "callable"

    if isinstance(mask, dict):
        m = np.ones(n_obs, dtype=bool)
        for col, val in mask.items():
            if col not in obs_df.columns:
                raise KeyError(
                    f"mask dict: column {col!r} not found in obs. "
                    f"Available: {list(obs_df.columns)}"
                )
            col_vals = obs_df[col].values
            rng = _as_range(val)
            if rng is not None:
                lo, hi = rng
                if lo is not None:
                    m &= (col_vals >= lo)
                if hi is not None:
                    m &= (col_vals <= hi)
            elif isinstance(val, (list, set, np.ndarray)):
                m &= np.isin(col_vals, list(val))
            else:
                m &= col_vals == val
        return m, f"dict({list(mask.keys())})"

    if isinstance(mask, str):
        m = obs_df.eval(mask)
        if hasattr(m, "values"):
            m = m.values
        return _coerce_bool(m, n_obs), f"query: {mask!r}"

    return _coerce_bool(mask, n_obs), "array"


def _coerce_bool(value, n_obs: int) -> np.ndarray:
    """Normalise array-like to a bool array of length ``n_obs``.

    Accepts bool arrays/series (must match length) or integer indices
    (converted to a bool array).
    """
    arr = np.asarray(value)
    if arr.dtype == bool:
        if arr.shape[0] != n_obs:
            raise ValueError(
                f"mask length {arr.shape[0]} != n_cells {n_obs}"
            )
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        m = np.zeros(n_obs, dtype=bool)
        m[arr] = True
        return m
    raise TypeError(
        f"mask: unsupported dtype {arr.dtype!r}. "
        f"Accepts bool array, integer indices, str query, dict, or callable."
    )


# ---------------------------------------------------------------------------
# mask shape resolution (cytome path — RAM efficient)
# ---------------------------------------------------------------------------

def _resolve_mask_cytome(
    mask, ds, n_cells: int, *, callable_chunk_size: int = 200_000
) -> tuple[np.ndarray | None, str | None]:
    """Cytome-aware mask resolution.

    Optimised so simple query / dict masks push down to SQL — only
    the matching ``cell_idx`` values are materialised. Complex
    queries fall back to streaming pandas eval; callables materialise
    the cells table in chunks.
    """
    if mask is None:
        return None, None

    # Array-like / Series → coerce directly without touching the DB.
    if isinstance(mask, (np.ndarray, list, pd.Series)) and not isinstance(
        mask, dict
    ):
        return _coerce_bool(mask, n_cells), "array"

    # Determine cell column names (no full materialisation).
    valid_cols = {r[1] for r in ds._conn.execute(
        "PRAGMA table_info(cells)"
    ).fetchall()}

    if isinstance(mask, dict):
        where_sql, params = _dict_to_sql_where(mask, valid_cols)
        return _sql_select_mask(ds, where_sql, params, n_cells), \
            f"dict({list(mask.keys())})"

    if isinstance(mask, str):
        try:
            where_sql, params = _query_to_sql_where(mask, valid_cols)
        except _QueryTooComplex as exc:
            # Fall back: streaming pandas eval, chunked over the cells table.
            return _streaming_pandas_eval(
                ds, mask, n_cells, chunk_size=callable_chunk_size,
            ), f"query (streaming pandas, reason: {exc}): {mask!r}"
        return _sql_select_mask(ds, where_sql, params, n_cells), \
            f"query (SQL): {mask!r}"

    if callable(mask):
        return _streaming_callable_mask(
            ds, mask, n_cells, chunk_size=callable_chunk_size,
        ), "callable"

    raise TypeError(
        f"mask: unsupported type {type(mask).__name__}. "
        f"Accepts bool array, integer indices, str query, dict, or callable."
    )


def _sql_select_mask(ds, where_sql: str, params: list, n_cells: int) -> np.ndarray:
    """Run ``SELECT cell_idx FROM cells WHERE {where_sql}`` and build a bool mask.

    RAM is O(n_matched), not O(n_cells × n_cols).
    """
    idx = np.fromiter(
        (r[0] for r in ds._conn.execute(
            f"SELECT cell_idx FROM cells WHERE {where_sql}", params
        )),
        dtype=np.int64,
    )
    m = np.zeros(n_cells, dtype=bool)
    if idx.size:
        m[idx] = True
    return m


def _stream_cells_chunks(ds, chunk_size: int):
    """Yield ``(cell_idx_array, df_chunk)`` slices of the cells table."""
    cols = [r[1] for r in ds._conn.execute(
        "PRAGMA table_info(cells)"
    ).fetchall()]
    col_sql = ", ".join(f"[{c}]" for c in cols)
    offset = 0
    while True:
        rows = ds._conn.execute(
            f"SELECT {col_sql} FROM cells "
            f"ORDER BY cell_idx LIMIT ? OFFSET ?",
            (chunk_size, offset),
        ).fetchall()
        if not rows:
            return
        df = pd.DataFrame(rows, columns=cols)
        chunk_idx = df["cell_idx"].values.astype(np.int64)
        yield chunk_idx, df
        offset += len(rows)


def _streaming_pandas_eval(
    ds, query: str, n_cells: int, *, chunk_size: int = 200_000,
) -> np.ndarray:
    """Streaming fallback for query strings that don't translate to SQL.

    Reads the cells table in ``chunk_size`` slices via SQL
    ``LIMIT/OFFSET``, evaluates the pandas query on each chunk, and
    fills a bool mask. Peak RAM ≈ ``chunk_size × n_cell_cols``.
    """
    mask = np.zeros(n_cells, dtype=bool)
    for chunk_idx, df in _stream_cells_chunks(ds, chunk_size):
        result = df.eval(query)
        if hasattr(result, "values"):
            result = result.values
        mask[chunk_idx] = np.asarray(result, dtype=bool)
    return mask


def _streaming_callable_mask(
    ds, fn: Callable, n_cells: int, *, chunk_size: int = 200_000,
) -> np.ndarray:
    """Streaming callable: chunked materialisation of cells, ``fn(chunk)``."""
    mask = np.zeros(n_cells, dtype=bool)
    for chunk_idx, df in _stream_cells_chunks(ds, chunk_size):
        result = fn(df)
        if hasattr(result, "values"):
            result = result.values
        mask[chunk_idx] = np.asarray(result, dtype=bool)
    return mask


# ---------------------------------------------------------------------------
# QC-threshold mask (computed from the matrix)
# ---------------------------------------------------------------------------

def _qc_threshold_mask_anndata(
    adata, min_counts, max_counts, min_features, max_features,
) -> tuple[np.ndarray | None, list[str]]:
    """AnnData QC mask from the matrix. Returns (mask|None, source_descs)."""
    if all(v is None for v in (min_counts, max_counts, min_features, max_features)):
        return None, []
    X = adata.X
    keep = np.ones(adata.n_obs, dtype=bool)
    descs: list[str] = []
    if min_counts is not None or max_counts is not None:
        counts = np.asarray(X.sum(axis=1)).ravel()
        if min_counts is not None:
            keep &= counts >= min_counts
            descs.append(f"min_counts={min_counts}")
        if max_counts is not None:
            keep &= counts <= max_counts
            descs.append(f"max_counts={max_counts}")
    if min_features is not None or max_features is not None:
        if sparse.issparse(X):
            n_feat = np.asarray((X != 0).sum(axis=1)).ravel()
        else:
            n_feat = np.asarray((X != 0).sum(axis=1)).ravel()
        if min_features is not None:
            keep &= n_feat >= min_features
            descs.append(f"min_features={min_features}")
        if max_features is not None:
            keep &= n_feat <= max_features
            descs.append(f"max_features={max_features}")
    return keep, descs


def _qc_threshold_mask_cytome(
    ds, min_counts, max_counts, min_features, max_features,
    modality: str, batch_size: int,
) -> tuple[np.ndarray | None, list[str]]:
    """Cytome QC mask: one streaming pass over ``{modality}_counts``."""
    if all(v is None for v in (min_counts, max_counts, min_features, max_features)):
        return None, []
    n_cells = ds.n_cells
    counts = np.zeros(n_cells, dtype=np.float64)
    n_feat = np.zeros(n_cells, dtype=np.int64)
    # ds.iter_chunks yields (chunk_csr, cell_indices_array) tuples.
    for chunk, cell_idx in ds.iter_chunks(
        modality=modality, layer="counts", batch_size=batch_size,
    ):
        counts[cell_idx] = np.asarray(chunk.sum(axis=1)).ravel()
        n_feat[cell_idx] = np.asarray((chunk != 0).sum(axis=1)).ravel()
    keep = np.ones(n_cells, dtype=bool)
    descs: list[str] = []
    if min_counts is not None:
        keep &= counts >= min_counts
        descs.append(f"min_counts={min_counts}")
    if max_counts is not None:
        keep &= counts <= max_counts
        descs.append(f"max_counts={max_counts}")
    if min_features is not None:
        keep &= n_feat >= min_features
        descs.append(f"min_features={min_features}")
    if max_features is not None:
        keep &= n_feat <= max_features
        descs.append(f"max_features={max_features}")
    return keep, descs


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _write_provenance(
    ds, *, n_before: int, n_after: int, mask_sources: list[str],
    modality: str, inplace: bool, output: str | None,
):
    """Write filter_cells provenance directly via SQL on ``ds._conn``.

    Writing via raw SQL (rather than ``ds.metadata[key] = value``)
    keeps the call site backend-agnostic: ``ds._conn`` is always the
    live connection, regardless of whether the caller has
    materialised the cached ``MetadataStore`` accessor yet. This
    matters for the in-place ``filter_cells`` path, where the
    metadata write happens after ``ds.filter_cells`` reopens the
    connection.

    Requires cytome ≥ 0.2.1 — earlier versions did not refresh
    cached accessors after ``filter_cells`` reopened ``self._conn``,
    so any later ``ds.metadata.get(...)`` raised ProgrammingError.
    The cytome 0.2.1 fix invalidates ``_metadata_obj`` automatically
    in ``filter_cells``; we no longer need the manual reset here.
    """
    import json
    import sys
    from datetime import datetime, timezone

    try:
        import piaso
        version = getattr(piaso, "__version__", "unknown")
    except Exception:
        version = "unknown"

    params = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "piaso_version": version,
        "n_before": int(n_before),
        "n_after": int(n_after),
        "mask_sources": list(mask_sources),
        "modality": modality,
        "inplace": bool(inplace),
        "output": str(output) if output else None,
    }

    conn = ds._conn
    if conn.in_transaction:
        conn.commit()

    conn.execute(
        "INSERT OR REPLACE INTO _metadata(key, value, value_type) "
        "VALUES (?, ?, ?)",
        ("piaso_filter_cells_params", json.dumps(params), "json"),
    )
    conn.execute(
        "INSERT INTO _provenance("
        "  timestamp, operation, package_name, package_version,"
        "  python_version, function_name, parameters,"
        "  input_objects, output_objects"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            params["timestamp"],
            "filter_cells",
            "piaso",
            version,
            sys.version,
            "piaso.pp.filter_cells",
            json.dumps(params),
            json.dumps([]),
            json.dumps(["cells"]),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def filter_cells(
    data,
    min_counts=None,
    max_counts=None,
    min_features=None,
    max_features=None,
    *,
    mask=None,
    inplace: bool = True,
    output: str | Path | None = None,
    overwrite: bool = False,
    modality: str = "RNA",
    batch_size: int = 2048,
    include_fragments: bool = True,
    include_embeddings: bool = True,
    verbose: int = 1,
):
    """Filter cells. Polymorphic on AnnData / cytome inputs.

    The keep-mask is built from up to two independent sources, then
    intersected:

    1. **QC thresholds** computed from the counts matrix
       (``min_counts``, ``max_counts``, ``min_features``,
       ``max_features``). Backward-compatible with the prior
       AnnData-only signature.
    2. **General mask** via the ``mask`` kwarg. Six shapes accepted:

       - boolean ``np.ndarray`` / ``pd.Series`` of length ``n_cells``
       - integer ``np.ndarray`` of cell indices to keep
       - string pandas-query expression
         (e.g. ``"n_counts > 1000 and cluster.isin(['T','B'])"``)
       - dict mapping column → scalar / list / numeric 2-tuple range.
         Ranges may be open-ended via ``None``: ``(lo, hi)`` → ``lo ≤ col ≤ hi``,
         ``(lo, None)`` → ``col ≥ lo``, ``(None, hi)`` → ``col ≤ hi``. This is the
         clean way to express ATAC cell-QC after ``calculateCellMetrics`` /
         the importer have written the columns, e.g.::

             filter_cells(ds, modality="ATAC",
                 mask={"n_fragments": (1000, None),
                       "tss_score":   (3.0,  None),
                       "frip":        (0.2,  None)})

       - callable ``fn(obs_df) -> bool series``

    Cytome path is RAM-efficient: dict masks and simple query
    strings push down to SQL ``WHERE`` and only matching
    ``cell_idx`` values are materialised. Complex queries / callables
    fall back to chunked streaming over the cells table.

    Parameters
    ----------
    data
        AnnData or ``cytome.Dataset``.
    min_counts, max_counts, min_features, max_features
        Per-cell QC thresholds. Computed from the counts matrix.
    mask
        General-purpose mask (see above for accepted shapes).
    inplace
        If True, apply the filter in place. For cytome this calls
        ``ds.filter_cells(...)`` which atomically replaces the file.
        For AnnData this calls ``adata._inplace_subset_obs(...)``.
    output
        Cytome-only. When set with ``inplace=False``, write a filtered
        copy to this path via ``ds.subset(mask, output=...)`` instead of
        modifying the original. AnnData ignores this with a warning.
    overwrite
        When ``output`` exists, replace it (default False raises).
    modality
        Cytome modality used to compute QC thresholds. Default 'RNA'.
    batch_size
        Streaming batch size for QC threshold pass on cytome.
    include_fragments, include_embeddings
        Forwarded to ``ds.filter_cells`` / ``ds.subset``.
    verbose
        0 silent, 1 (default) prints summary, 2 prints mask sources.

    Returns
    -------
    AnnData backend:
      - ``inplace=True``  → ``None`` (modifies in place)
      - ``inplace=False`` → bool mask of length ``n_cells``

    Cytome backend:
      - ``inplace=True,  output=None`` → ``int`` (n_kept)
      - ``inplace=True,  output=path`` → TypeError (contradictory)
      - ``inplace=False, output=None`` → bool mask of length ``n_cells``
      - ``inplace=False, output=path`` → open ``cytome.Dataset`` at ``path``

    Raises
    ------
    TypeError
        ``inplace=True`` and ``output`` both given.
    ValueError
        No cells survive the filter, or a mask has the wrong length.
    KeyError
        ``mask`` dict references a missing column.
    FileExistsError
        ``output`` exists and ``overwrite=False``.
    """
    if _is_cytome(data):
        return _filter_cells_cytome(
            data,
            min_counts=min_counts, max_counts=max_counts,
            min_features=min_features, max_features=max_features,
            mask=mask, inplace=inplace, output=output, overwrite=overwrite,
            modality=modality, batch_size=batch_size,
            include_fragments=include_fragments,
            include_embeddings=include_embeddings,
            verbose=verbose,
        )
    return _filter_cells_anndata(
        data,
        min_counts=min_counts, max_counts=max_counts,
        min_features=min_features, max_features=max_features,
        mask=mask, inplace=inplace, output=output, verbose=verbose,
    )


# Alias for callers who prefer "subset_cells" — filter and subset are the
# same operation under the hood.
subset_cells = filter_cells


# ---------------------------------------------------------------------------
# AnnData backend
# ---------------------------------------------------------------------------

def _filter_cells_anndata(
    adata, *, min_counts, max_counts, min_features, max_features,
    mask, inplace, output, verbose,
):
    if output is not None:
        warnings.warn(
            "filter_cells: `output=` is cytome-only and is ignored for AnnData. "
            "To save a filtered AnnData, call `adata.write_h5ad(path)` after "
            "the call.",
            UserWarning, stacklevel=3,
        )

    n_obs = adata.n_obs
    keep = np.ones(n_obs, dtype=bool)
    sources: list[str] = []

    qc_mask, qc_descs = _qc_threshold_mask_anndata(
        adata, min_counts, max_counts, min_features, max_features,
    )
    if qc_mask is not None:
        keep &= qc_mask
        sources.append("thresholds: " + ", ".join(qc_descs))

    user_mask, user_desc = _resolve_mask_anndata(mask, adata.obs, n_obs)
    if user_mask is not None:
        keep &= user_mask
        sources.append(user_desc)

    n_kept = int(keep.sum())
    if verbose:
        print(f"filter_cells (AnnData): {n_kept}/{n_obs} cells pass "
              f"(sources: {sources or ['(no filters)']})")

    if not inplace:
        return keep

    if n_kept == 0:
        # AnnData lets you subset to an empty .obs without raising —
        # preserve that (scanpy users rely on it). Warn so the no-op
        # downstream cause is visible.
        warnings.warn(
            "filter_cells: zero cells pass the filter; subsetting AnnData "
            "to 0 cells. Most downstream calls will fail on an empty "
            "AnnData — verify your thresholds / mask.",
            UserWarning, stacklevel=3,
        )
    adata._inplace_subset_obs(keep)
    return None


# ---------------------------------------------------------------------------
# Cytome backend
# ---------------------------------------------------------------------------

def _filter_cells_cytome(
    ds, *, min_counts, max_counts, min_features, max_features,
    mask, inplace, output, overwrite,
    modality, batch_size, include_fragments, include_embeddings, verbose,
):
    if inplace and output is not None:
        raise TypeError(
            "filter_cells: `inplace=True` and `output=<path>` are "
            "contradictory. For an in-place atomic replace use "
            "`inplace=True, output=None`; to write a subset to a new "
            "file use `inplace=False, output=<path>`."
        )

    n_cells = ds.n_cells
    keep = np.ones(n_cells, dtype=bool)
    sources: list[str] = []

    qc_mask, qc_descs = _qc_threshold_mask_cytome(
        ds, min_counts, max_counts, min_features, max_features,
        modality=modality, batch_size=batch_size,
    )
    if qc_mask is not None:
        keep &= qc_mask
        sources.append(f"thresholds[{modality}]: " + ", ".join(qc_descs))

    user_mask, user_desc = _resolve_mask_cytome(mask, ds, n_cells)
    if user_mask is not None:
        keep &= user_mask
        sources.append(user_desc)

    n_kept = int(keep.sum())
    if verbose:
        print(f"filter_cells (cytome): {n_kept}/{n_cells} cells pass "
              f"(sources: {sources or ['(no filters)']})")
    if verbose >= 2:
        for s in sources:
            print(f"  · {s}")

    # mask-only mode
    if not inplace and output is None:
        return keep

    if n_kept == 0:
        raise ValueError(
            f"filter_cells: zero cells pass the filter (started with "
            f"{n_cells}). Sources: {sources}. Loosen thresholds or "
            f"review the mask before applying."
        )

    if not inplace:
        # write subset to a new file
        out_path = Path(output)
        if out_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{out_path} exists; pass overwrite=True to replace, "
                    f"or pick a different output path."
                )
            out_path.unlink()
        new_ds = ds.subset(
            keep, output=out_path,
            include_fragments=include_fragments,
            include_embeddings=include_embeddings,
        )
        # Provenance on the new dataset (the source is read-only here).
        _write_provenance(
            new_ds, n_before=n_cells, n_after=n_kept,
            mask_sources=sources, modality=modality,
            inplace=False, output=str(out_path),
        )
        return new_ds

    # inplace atomic in-place replace
    n_after = int(ds.filter_cells(
        keep,
        include_fragments=include_fragments,
        include_embeddings=include_embeddings,
    ))
    _write_provenance(
        ds, n_before=n_cells, n_after=n_after,
        mask_sources=sources, modality=modality,
        inplace=True, output=None,
    )
    return n_after


# ---------------------------------------------------------------------------
# filter_features — unchanged (AnnData only for now)
# ---------------------------------------------------------------------------

def filter_features(adata, min_cells=None, max_cells=None,
                    min_counts=None, max_counts=None, inplace=True):
    """Filter features (genes/peaks) based on cell or count thresholds.

    AnnData only. Cytome equivalent lives in ``piaso.pp.selectPeaks`` /
    ``piaso.tl.infog`` (for highly-variable features).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    min_cells : int, optional
        Minimum number of cells expressing the feature (non-zero).
    max_cells : int, optional
        Maximum number of cells expressing the feature.
    min_counts : int, optional
        Minimum total counts per feature.
    max_counts : int, optional
        Maximum total counts per feature.
    inplace : bool
        If True, subset adata in place. If False, return boolean mask.

    Returns
    -------
    If ``inplace=True``: modifies ``adata`` in place and returns None.
    If ``inplace=False``: returns boolean array of shape ``(n_vars,)``.
    """
    X = adata.X
    keep = np.ones(adata.n_vars, dtype=bool)

    if min_cells is not None or max_cells is not None:
        n_cells = np.asarray((X != 0).sum(axis=0)).ravel()
        if min_cells is not None:
            keep &= n_cells >= min_cells
        if max_cells is not None:
            keep &= n_cells <= max_cells

    if min_counts is not None or max_counts is not None:
        counts = np.asarray(X.sum(axis=0)).ravel()
        if min_counts is not None:
            keep &= counts >= min_counts
        if max_counts is not None:
            keep &= counts <= max_counts

    if inplace:
        adata._inplace_subset_var(keep)
        return None
    return keep
