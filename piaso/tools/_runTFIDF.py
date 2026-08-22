"""
TF-IDF normalization for ATAC-seq peak/tile matrices.

Supports both in-memory (AnnData) and streaming (Cytome) backends.

Streaming mode uses 2 passes:
    Pass 1: Accumulate cell_depth (row sums) and peak_depth (column sums).
    Pass 2: Apply TF-IDF normalization per chunk and write via ChunkedLayerWriter.

RAM: O(n_cells + n_features) regardless of matrix size.
"""

from typing import Union, Optional
from ..settings import _resolve_layer_dtype
import numpy as np
from scipy import sparse
from anndata import AnnData

from ._normalization import _safe_n_cells

from ..utils._cytome_compat import is_cytome_input as _is_cytome_source
from ..utils._cytome_compat import open_cytome_sync as _open_cytome
from ._compat import resolve_data_arg as _resolve_data_arg, _UNSET


# ===================================================================
#  Public API — dispatcher
# ===================================================================

def run_TFIDF(
    data=_UNSET,
    layer: Optional[str] = None,
    scale_factor: float = 1e4,
    streaming: bool = False,
    batch_size: int = 1024,
    output_layer: str = "tfidf",
    measurement: Optional[str] = None,
    modality: str = "ATAC",
    inplace: bool = False,
    # ---- deprecated aliases (back-compat) ----
    source=_UNSET,
    adata=_UNSET,
):
    """
    Compute TF-IDF normalization for peak count data.

    Parameters
    ----------
    source : AnnData or CytomeDataset or str
        Input data. AnnData for in-memory, Cytome for streaming.
    layer : str, optional
        Input layer to read counts from.
        - AnnData: ``adata.layers[layer]`` (or ``adata.X`` if ``None``).
        - Cytome: takes precedence over ``measurement`` if both are set.
    scale_factor : float, default 1e4
        Scaling factor for TF values before log1p.
    streaming : bool, default False
        Force streaming mode even for AnnData input.
    batch_size : int, default 1024
        Cells per chunk in streaming mode.
    output_layer : str, default 'tfidf'
        Name for the output. AnnData: writes ``adata.layers[output_layer]``.
        Cytome: creates a measurement matrix named ``{modality}_{output_layer}``.
        Pass ``output_layer=None`` (AnnData only) to skip the layer write and
        only mutate ``adata.X`` (the legacy in-place behaviour — requires
        ``inplace=True``).
    measurement : str, optional
        Input Cytome measurement name (default: 'counts').
    modality : str, default 'ATAC'
        Modality prefix for Cytome layer names (e.g., 'ATAC', 'tiles').
    inplace : bool, default False
        AnnData only. If ``True``, also overwrite ``adata.X`` with the TF-IDF
        result. The default (``False``) writes only to ``adata.layers[output_layer]``
        and leaves ``adata.X`` untouched. Useful when downstream calls
        (e.g. ``infog_svd(layer=None)``) expect TF-IDF on ``.X``.

    Returns
    -------
    None
        Modifies source in-place. AnnData: writes ``adata.layers[output_layer]``
        (and optionally ``adata.X`` when ``inplace=True``). Cytome: materialises
        ``{modality}_{output_layer}``.
    """
    source = _resolve_data_arg(data, 'run_TFIDF', source=source, adata=adata)
    is_cytome = _is_cytome_source(source)
    if is_cytome or streaming:
        return _runTFIDF_streaming(
            source, batch_size, scale_factor, layer, output_layer, measurement,
            modality,
        )
    else:
        return _runTFIDF_original(
            source, layer, scale_factor, output_layer=output_layer,
            inplace=inplace,
        )


# ===================================================================
#  Original in-memory backend
# ===================================================================

def _runTFIDF_original(
    adata: AnnData,
    layer: Optional[str] = None,
    scale_factor: float = 1e4,
    output_layer: Optional[str] = "tfidf",
    inplace: bool = False,
):
    """Original in-memory TF-IDF.

    By default writes the TF-IDF result to ``adata.layers[output_layer]`` and
    leaves ``adata.X`` untouched. Pass ``inplace=True`` to ALSO overwrite
    ``adata.X``. Pass ``output_layer=None`` together with ``inplace=True`` for
    the legacy "mutate ``adata.X`` only, no layer write" behaviour.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("Expected an AnnData object.")

    if output_layer is None and not inplace:
        raise ValueError(
            "run_TFIDF: output_layer is None and inplace is False — nowhere "
            "to write the result. Either pass output_layer='tfidf' "
            "(writes adata.layers['tfidf']) or inplace=True (overwrites adata.X)."
        )

    # Read input from .layers[layer] if specified, else .X. Crucially do NOT
    # mutate adata.X to point at the input layer (the previous version did
    # `adata.X = adata.layers[layer]` which had a surprising side-effect even
    # before the TF-IDF step ran).
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"run_TFIDF: input layer '{layer}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}."
            )
        X_in = adata.layers[layer]
    else:
        X_in = adata.X

    if not sparse.issparse(X_in):
        X_in = sparse.csr_matrix(X_in)

    # TF: normalize by cell depth (row sums)
    n_peaks = np.asarray(X_in.sum(axis=1)).reshape(-1)
    # Guard against zero-sum cells before constructing the diagonal scaler
    safe_n = np.where(n_peaks == 0, 1.0, n_peaks)
    inv = sparse.dia_matrix((1.0 / safe_n, 0), shape=(safe_n.size, safe_n.size))
    tf = inv @ X_in

    # Scale + log1p
    if scale_factor != 1 and scale_factor != 0:
        tf = tf * scale_factor
    tf.data = np.log1p(tf.data)

    # IDF: inverse document frequency
    col_sums = np.asarray(X_in.sum(axis=0)).reshape(-1)
    idf = np.log1p(adata.shape[0] / np.maximum(col_sums, 1.0))

    idf = sparse.dia_matrix((idf, 0), shape=(idf.size, idf.size))
    tf_idf = tf.dot(idf)

    # Clean up NaN/inf values
    np.nan_to_num(tf_idf.data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    tf_idf.eliminate_zeros()

    if output_layer is not None:
        adata.layers[output_layer] = tf_idf
    if inplace:
        adata.X = tf_idf


# ===================================================================
#  TF-IDF stats-only (for fused TF-IDF + SVD)
# ===================================================================

def compute_tfidf_stats(
    source,
    measurement: str = "counts",
    batch_size: int = 1024,
    scale_factor: float = 1e4,
    modality: str = "ATAC",
    write_to_metadata: bool = True,
):
    """Compute TF-IDF statistics without materializing the full matrix.

    Pass 1 only: accumulates cell_depth and peak_depth, computes idf.
    The returned arrays can be passed to runSVD via tfidf_params for
    inline TF-IDF application during SVD — eliminating the persistent
    TF-IDF layer entirely.

    Parameters
    ----------
    source : CytomeDataset or str
        Cytome dataset or path.
    measurement : str
        Input measurement name (default 'counts').
    batch_size : int
        Cells per chunk.
    scale_factor : float
        TF-IDF scale factor.
    modality : str
        Modality prefix (default 'ATAC').

    Returns
    -------
    dict
        {'cell_depth': ndarray, 'idf': ndarray, 'scale_factor': float}
    """
    if isinstance(source, str):
        ds = _open_cytome(source)
    else:
        ds = source

    _layer = measurement

    # Get matrix dimensions
    meta_row = ds._conn.execute(
        "SELECT n_rows, n_cols FROM matrix_meta WHERE matrix_name = ?",
        (f"{modality}_{_layer}",)
    ).fetchone()
    if meta_row is None:
        raise ValueError(
            f"Measurement '{modality}_{_layer}' not found in Cytome dataset."
        )
    _meta_n_rows, n_features = meta_row
    n_alloc, n_true_cells = _safe_n_cells(ds, modality, _layer)

    # Pass 1: accumulate row sums and column sums
    cell_depth = np.zeros(n_alloc, dtype=np.float64)
    peak_depth = np.zeros(n_features, dtype=np.float64)

    for chunk, indices in ds.iter_chunks(
        modality=modality, layer=_layer, batch_size=batch_size
    ):
        cell_depth[indices] = np.array(chunk.sum(axis=1)).ravel()
        peak_depth += np.array(chunk.sum(axis=0)).ravel()

    # Compute IDF (use true cell count for the IDF formula)
    idf = np.log1p(n_true_cells / np.maximum(peak_depth, 1.0)).astype(np.float32)

    params = {
        "cell_depth": cell_depth,
        "idf": idf,
        "scale_factor": scale_factor,
    }
    # Cache to per-modality metadata so plotting on-the-fly TF-IDF can
    # reuse without recomputing. The legacy un-prefixed alias is also
    # written for any consumer not yet on the new key.
    if write_to_metadata:
        ds.metadata[f"{modality}_tfidf_params"] = params
        # NOTE: the modality-blind un-prefixed 'tfidf_params' legacy alias is no
        # longer written — it clobbered across modalities (an ATAC payload then a
        # tiles payload overwriting each other), the write-side of a cross-modality
        # leak. Readers use the prefixed '{modality}_tfidf_params' and fall back to
        # a residual legacy key only under a feature-count guard.
        # Write per-feature IDF as a column on the modality's feature table. MUST
        # use ds.features(modality), NOT getattr(ds, entity_name): for 'tiles'/'GA'
        # the modality name collides with / differs from the table name, so getattr
        # resolves to a Modality (no __setitem__). ds.features() returns the
        # writable EntityTable uniformly across modalities.
        try:
            ds.features(modality)['tfidf_idf'] = idf.astype(np.float32)
        except Exception as e:
            # Best-effort write; don't fail the whole stats computation
            # if the modality isn't in the registry or the entity is
            # read-only. Metadata stays the source of truth.
            import warnings as _warnings
            _warnings.warn(
                f"compute_tfidf_stats: could not write tfidf_idf to "
                f"var entity for modality={modality!r}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
        ds.flush()
    return params


def _load_or_compute_tfidf_stats(
    ds,
    modality: str = "ATAC",
    layer: str = "counts",
    batch_size: int = 1024,
    scale_factor: float = 1e4,
    force_recompute: bool = False,
    verbosity: int = 0,
) -> dict:
    """Return TF-IDF params (cell_depth, idf, scale_factor), using the
    cached value in ``ds.metadata['{modality}_tfidf_params']`` when
    present, otherwise computing via one streaming pass and caching.

    Mirrors how runGDR / COSG resolve the INFOG params dict
    (``ds.metadata['{modality}_infog_params']``).
    """
    key = f"{modality}_tfidf_params"
    if not force_recompute and key in ds.metadata:
        if verbosity > 0:
            print(f"  Using cached TF-IDF stats from ds.metadata[{key!r}]")
        return ds.metadata[key]
    if verbosity > 0:
        print(f"  No cached TF-IDF stats at {key!r}; computing (one pass)")
    return compute_tfidf_stats(
        ds, measurement=layer, batch_size=batch_size,
        scale_factor=scale_factor, modality=modality,
        write_to_metadata=True,
    )


# ===================================================================
#  Pure chunk function
# ===================================================================

def _normalize_chunk_tfidf(chunk, cell_depth_chunk, idf, scale_factor):
    """
    Apply TF-IDF to one sparse chunk. Pure function.

    Parameters
    ----------
    chunk : csr_matrix
        (batch_size, n_features) sparse matrix.
    cell_depth_chunk : ndarray
        Row sums for this chunk's cells.
    idf : ndarray
        Pre-computed IDF vector (n_features,).
    scale_factor : float
        Scaling factor.

    Returns
    -------
    csr_matrix
        TF-IDF normalized chunk.
    """
    # In-place TF-IDF: single copy instead of 3-4 intermediate CSR matrices.
    # This dramatically reduces peak RSS (43 GB -> ~5 GB on MC).
    safe_depth = np.maximum(cell_depth_chunk, 1.0)
    # The in-place ops below all write float results back into chunk.data, so
    # an integer-dtype chunk raises UFuncOutputCastingError on the very first
    # divide. Raw count matrices are commonly int (a Matrix Market import
    # keeps int64), so cast on the copy we already make rather than making the
    # caller pre-convert.
    chunk = (chunk.astype(np.float32) if not np.issubdtype(chunk.dtype, np.floating)
             else chunk.copy())

    # TF: in-place divide by cell depth (row-wise)
    row_nnz = np.diff(chunk.indptr)
    np.divide(chunk.data,
              np.repeat(safe_depth, row_nnz),
              out=chunk.data)

    # Scale + log1p in-place
    if scale_factor != 1 and scale_factor != 0:
        chunk.data *= scale_factor
    np.log1p(chunk.data, out=chunk.data)

    # IDF: in-place column-wise multiplication
    chunk.data *= idf[chunk.indices]

    # Clean up NaN/inf
    np.nan_to_num(chunk.data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    chunk.eliminate_zeros()

    return chunk


# ===================================================================
#  Streaming backend (Cytome)
# ===================================================================

def _runTFIDF_streaming(
    source,
    batch_size: int = 1024,
    scale_factor: float = 1e4,
    layer: Optional[str] = None,
    output_layer: str = "tfidf",
    measurement: Optional[str] = None,
    modality: str = "ATAC",
    dtype: Optional[str] = None,
):
    """
    2-pass streaming TF-IDF via Cytome. RAM: O(n_cells + n_features).

    Pass 1: Accumulate cell_depth (row sums) and peak_depth (column sums).
    Pass 2: Normalize per chunk and write via ChunkedLayerWriter.
    """
    if isinstance(source, str):
        ds = _open_cytome(source)
    else:
        ds = source

    _layer = measurement or layer or "counts"

    # Determine col_entity from modality
    col_entity = "peaks" if modality == "ATAC" else modality

    # Get matrix dimensions
    meta_row = ds._conn.execute(
        "SELECT n_rows, n_cols FROM matrix_meta WHERE matrix_name = ?",
        (f"{modality}_{_layer}",)
    ).fetchone()
    if meta_row is None:
        raise ValueError(
            f"Measurement '{modality}_{_layer}' not found in Cytome dataset."
        )
    _meta_n_rows, n_features = meta_row
    n_alloc, n_true_cells = _safe_n_cells(ds, modality, _layer)

    # --- Pass 1: Accumulate global statistics ---
    cell_depth = np.zeros(n_alloc, dtype=np.float64)    # row sums (safe size)
    peak_depth = np.zeros(n_features, dtype=np.float64)  # column sums

    for chunk, indices in ds.iter_chunks(
        modality=modality, layer=_layer, batch_size=batch_size
    ):
        cell_depth[indices] = np.array(chunk.sum(axis=1)).ravel()
        peak_depth += np.array(chunk.sum(axis=0)).ravel()

    # Compute IDF (use true cell count, not inflated n_alloc)
    idf = np.log1p(n_true_cells / np.maximum(peak_depth, 1.0))

    # --- Pass 2: Normalize and write ---
    out_name = f"{modality}_{output_layer}"
    writer = ds.create_layer_writer(
        layer_name=out_name,
        n_rows=n_true_cells,
        n_cols=n_features,
        dtype=_resolve_layer_dtype(dtype),
        compression="zstd",
        col_entity=col_entity,
        overwrite=True,
    )

    for chunk, indices in ds.iter_chunks(
        modality=modality, layer=_layer, batch_size=batch_size
    ):
        tfidf_chunk = _normalize_chunk_tfidf(
            chunk, cell_depth[indices], idf, scale_factor
        )
        writer.write_chunk(tfidf_chunk, row_offset=int(indices[0]))

    writer.finalize()
    ds.flush()
