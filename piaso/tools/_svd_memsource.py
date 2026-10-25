"""An in-memory matrix as a chunk source for the SVD pass engine.

The engine (``_piaso.svd_pass_source``) multiplies chunks that any object
hands it, through three calls -- the protocol of cytome's ``ChunkSource``:
``meta()``, ``chunks(layout)`` and ``fetch(layout, idx, next_idx)``. A store
answers with its compressed blobs. :class:`MatrixChunkSource` answers with
slices of a matrix in memory, as uncompressed bytes, so an AnnData's matrix
goes through the same decoding, the same transform and the same order of
additions as a store's.

**The result does not depend on the chunking or the layout.** For every
output element the engine adds the terms in chunk order, then in stored
order within a chunk. On canonical chunks (indices ascending, no duplicates)
that is ascending column order for ``X @ B`` and ascending row order for
``X^T @ Y`` whichever way the matrix is cut, so a matrix in memory and the
same matrix in a store give the same products to the bit.

:func:`matrix_python_ops` gives the same products from Python, one chunk at a
time, for when the extension is missing, ``use_engine=False`` is asked for, or
the transform is one only Python computes (INFOG with explicit parameters).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse

__all__ = ["MatrixChunkSource", "matrix_python_ops", "engine_ready", "tfidf_stats_matrix",
           "assert_raw_counts_matrix"]

# The value types the engine decodes (svd_pass.rs, decode_chunk).
_ENGINE_DTYPES = frozenset({"float32", "float64", "int32", "int64", "int16", "uint16",
                            "uint32", "int8", "uint8"})

# Entries per chunk: enough chunks for every thread to take several, few
# enough that each is worth a call, and small enough that the copies a
# thread decodes stay small (a million entries is 8 MB as float32 values and
# int32 indices). The products do not depend on it.
_MIN_CHUNK_NNZ = 1 << 16
_MAX_CHUNK_NNZ = 1 << 20

# Bucketed entries one engine batch may hold (16 bytes each), against the
# store path's 512 MB: a batch's chunks are also held as bytes until the
# batch is done, so a smaller batch keeps fewer copies in flight. Speed only.
BATCH_BYTES = 128 << 20


def _engine_dtype(dt) -> np.dtype:
    """The type a value travels as: the stored type when the engine reads
    it (native byte order), float32 for booleans and half floats, float64
    for anything else."""
    dt = np.dtype(dt)
    if dt.name in _ENGINE_DTYPES:
        return np.dtype(dt.name)
    if dt == np.bool_ or (np.issubdtype(dt, np.floating) and dt.itemsize < 4):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def engine_ready(X):
    """``X`` as the engine can take it: a canonical CSR or CSC matrix
    (sorted indices, no duplicates), or a 2-D ndarray read row block by row
    block. Other sparse formats become CSR. A matrix that is not canonical is
    copied, never changed in place: it belongs to the caller."""
    if sparse.issparse(X):
        if X.format not in ("csr", "csc"):
            X = X.tocsr()
        if not X.has_canonical_format:
            # Duplicates would be transformed one by one (log1p(a) + log1p(b)
            # is not log1p(a + b)); unsorted indices only change the order.
            X = X.copy()
            X.sum_duplicates()
        return X
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"runSVD: expected a 2-D matrix, got shape {X.shape}")
    return X


def _bounds(indptr, n_outer, nnz, n_threads):
    """Chunk boundaries over the outer dimension, balanced by entries."""
    target = int(np.clip(nnz // max(8 * n_threads, 1), _MIN_CHUNK_NNZ, _MAX_CHUNK_NNZ))
    cuts = np.searchsorted(indptr, np.arange(target, nnz, target, dtype=np.int64), side="right") - 1
    b = np.unique(np.concatenate(([0], cuts, [n_outer]))).astype(np.int64)
    return b[(b >= 0) & (b <= n_outer)]


class MatrixChunkSource:
    """An in-memory matrix, handed to the SVD engine chunk by chunk.

    Parameters
    ----------
    X
        A canonical CSR or CSC matrix or a 2-D ndarray (see
        :func:`engine_ready`). CSR and dense input are cut into row chunks,
        CSC into column chunks; nothing is converted or copied up front.
    n_threads
        The engine's threads, which sets how finely the matrix is cut.
    rows_per_chunk
        Row-chunk height for dense input, whose entries are not counted
        until a block is read.
    cache
        Keep each chunk's bytes after the first pass, so later passes copy
        nothing. They hold one copy of the matrix; if that would exceed
        ``cache_budget_bytes`` (default: half the memory available to this
        process) the cache is dropped with a warning and every pass copies
        each chunk as it goes, as without the option.

    Each fetch copies the chunk once into ``bytes``; the engine decodes it on
    its own threads. ``bytes_read`` counts what was handed over.
    """

    matrix_name = "in-memory matrix"

    def __init__(self, X, n_threads: int = 1, rows_per_chunk: int = 1024, cache: bool = False,
                 cache_budget_bytes: int | None = None):
        self.X = X
        self.n_rows, self.n_cols = (int(s) for s in X.shape)
        self.dtype = _engine_dtype(X.dtype)
        self.dense = not sparse.issparse(X)
        self.layout = "csc" if (not self.dense and X.format == "csc") else "csr"
        if self.dense:
            step = max(1, int(rows_per_chunk))
            self._b = np.unique(np.r_[np.arange(0, self.n_rows, step), self.n_rows]).astype(np.int64)
            self._nnz = np.array([np.count_nonzero(X[s:e]) for s, e in zip(self._b[:-1], self._b[1:])],
                                 dtype=np.int64)
        else:
            n_outer = self.n_rows if self.layout == "csr" else self.n_cols
            indptr = np.asarray(X.indptr, dtype=np.int64)
            self._b = _bounds(indptr, n_outer, int(indptr[-1]), int(n_threads))
            self._nnz = np.diff(indptr[self._b])
        self.cache = bool(cache)
        if cache_budget_bytes is None and self.cache:
            try:
                from cytome.io.chunk_source import available_memory_bytes
                avail = available_memory_bytes()
            except Exception:  # pragma: no cover - cytome without the helper
                avail = None
            cache_budget_bytes = avail // 2 if avail else None
        self.cache_budget_bytes = cache_budget_bytes
        self._cache: dict = {}
        self.cached_bytes = 0
        self.cache_declined = None
        self.bytes_read = 0

    # -- protocol ---------------------------------------------------------
    def meta(self):
        """``(n_rows, n_cols, dtype, has_row_chunks, has_column_chunks)``."""
        return (self.n_rows, self.n_cols, self.dtype.name,
                self.layout == "csr", self.layout == "csc")

    def chunks(self, layout: str):
        """``(chunk_idx, start, end, n_nonzero)`` as int64 arrays."""
        if layout != self.layout:
            raise ValueError(f"{self.matrix_name} is cut into {self.layout} chunks, not {layout}")
        n = len(self._b) - 1
        return (np.arange(n, dtype=np.int64), np.ascontiguousarray(self._b[:-1]),
                np.ascontiguousarray(self._b[1:]), np.ascontiguousarray(self._nnz))

    def fetch(self, layout: str, idx, next_idx=None):
        """The chunks ``idx`` as ``(data, indices, indptr, 'none')`` tuples.
        ``next_idx`` is ignored: there is nothing to read ahead."""
        out = []
        for i in idx:
            i = int(i)
            blobs = self._cache.get(i)
            if blobs is None:
                blobs = self._blobs(i)
                nbytes = len(blobs[0]) + len(blobs[1]) + len(blobs[2])
                self.bytes_read += nbytes
                if self.cache:
                    if (self.cache_budget_bytes is not None
                            and self.cached_bytes + nbytes > self.cache_budget_bytes):
                        self.cache = False
                        self.cache_declined = (self.cached_bytes + nbytes, self.cache_budget_bytes)
                        self._cache.clear()
                        self.cached_bytes = 0
                        warnings.warn(
                            f"runSVD: keeping the matrix's chunks between passes would pass the "
                            f"budget of {self.cache_budget_bytes / 1e9:.1f} GB; each pass copies "
                            f"every chunk as it goes instead.", RuntimeWarning, stacklevel=2)
                    else:
                        self._cache[i] = blobs
                        self.cached_bytes += nbytes
            out.append(blobs)
        return out

    def close(self) -> None:
        self._cache.clear()
        self.cached_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- slicing ----------------------------------------------------------
    def _blobs(self, i):
        s, e = int(self._b[i]), int(self._b[i + 1])
        if self.dense:
            part = sparse.csr_matrix(self.X[s:e])
            data, ind, ptr = part.data, part.indices, part.indptr
        else:
            a, b = int(self.X.indptr[s]), int(self.X.indptr[e])
            data, ind = self.X.data[a:b], self.X.indices[a:b]
            ptr = np.asarray(self.X.indptr[s:e + 1], dtype=np.int64) - a
        return (np.asarray(data).astype(self.dtype, copy=False).tobytes(),
                np.asarray(ind).astype(np.int32, copy=False).tobytes(),
                np.asarray(ptr).astype(np.int32, copy=False).tobytes(),
                "none")


def matrix_python_ops(X, col_mask=None, row_keep=None, tfidf=None, chunk_norm=None,
                      rows_per_chunk: int = 1024):
    """The engine's two products from Python, with its semantics: rows taken
    from ``row_keep`` in that order, the transform on full rows (TF-IDF from
    ``tfidf``, or ``chunk_norm(chunk, rows)``), then the columns of
    ``col_mask``. One chunk of ``rows_per_chunk`` rows in memory at a time."""
    from ._rsvd_core import FactoryOps

    n_rows, n_cols = X.shape
    cols = None if col_mask is None else np.flatnonzero(np.asarray(col_mask, dtype=bool))
    keep = None if row_keep is None else np.asarray(row_keep, dtype=np.int64)
    n_out = n_rows if keep is None else len(keep)
    n_sel = n_cols if cols is None else len(cols)
    if tfidf is not None:
        from ._runTFIDF import _normalize_chunk_tfidf
        depth, idf = tfidf["cell_depth"], np.asarray(tfidf["idf"], dtype=np.float32)
        scale = tfidf.get("scale_factor", 1e4)
    step = max(1, int(rows_per_chunk))

    def factory():
        for s in range(0, n_out, step):
            e = min(s + step, n_out)
            rows = np.arange(s, e) if keep is None else keep[s:e]
            chunk = X[s:e] if keep is None else X[rows]
            if not sparse.issparse(chunk):
                chunk = sparse.csr_matrix(chunk)
            if tfidf is not None:
                chunk = _normalize_chunk_tfidf(chunk.tocsr(), depth[rows], idf, scale)
            if chunk_norm is not None:
                chunk = chunk_norm(chunk.tocsr(), rows)
            if cols is not None:
                chunk = chunk.tocsr()[:, cols]
            yield chunk, np.arange(s, e)

    return FactoryOps(factory, n_out, n_sel)


def tfidf_stats_matrix(X, scale_factor: float = 1e4):
    """TF-IDF statistics of an in-memory count matrix, computed as
    ``compute_tfidf_stats`` computes them on a store: each cell's depth over
    every feature (float64), and ``idf = log1p(n_cells / max(column sum, 1))``
    in float32."""
    if sparse.issparse(X):
        depth = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()
        col = np.asarray(X.sum(axis=0, dtype=np.float64)).ravel()
    else:
        A = np.asarray(X)
        depth = A.sum(axis=1, dtype=np.float64)
        col = A.sum(axis=0, dtype=np.float64)
    idf = np.log1p(X.shape[0] / np.maximum(col, 1.0)).astype(np.float32)
    return {"cell_depth": depth, "idf": idf, "scale_factor": scale_factor}


def assert_raw_counts_matrix(X, where: str, what: str = "auto_tfidf", n_values: int = 1_000_000):
    """Refuse TF-IDF on values that are not counts, as ``assert_raw_counts``
    does on a store: negative or fractional values mean the matrix is already
    normalised. Reads up to ``n_values`` of the first stored values."""
    if sparse.issparse(X):
        # any values answer the question; the first stored ones cost nothing
        vals = X.data[:n_values]
    else:
        vals = np.asarray(X).ravel()[:n_values]
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if (vals < 0).any():
        raise ValueError(
            f"{what} needs raw counts, but {where} holds negative values. It looks already "
            f"normalised (z-scored or centred). Point layer= at the raw count matrix.")
    if not np.allclose(vals, np.round(vals), rtol=0, atol=1e-6):
        raise ValueError(
            f"{what} needs raw counts, but {where} holds fractional values, so it has already "
            f"been normalised (TF-IDF, log1p or similar). Applying it again gives a "
            f"plausible-looking and wrong result. Point layer= at the raw count matrix.")
