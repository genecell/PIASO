"""Library-size normalization followed by log1p transform.

Supports both AnnData and cytome.Dataset as input.
"""

import numpy as np
from scipy import sparse as sp


from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome_dataset


def _normalize_chunk_log1p(chunk, cell_depth_chunk, scale_factor):
    """Per-chunk log1p normalization: ``log1p(chunk / cell_depth * scale_factor)``.

    Mirrors :func:`_normalize_chunk_infog` / :func:`_normalize_chunk_tfidf`
    in preserving sparse format when the input is sparse — important
    because COSG cytome streaming chains the result into sparse-matmul
    (``chunk.T @ Lam_chunk``) and ``chunk.data ** 2`` accumulators.

    Sparse path uses the same row-loop pattern as
    :func:`_normalize_log1p_anndata` (above): walk indptr per row,
    scale ``data[start:end]`` in-place, then log1p the data array
    (sparse-safe because ``log1p(0) == 0``). Dense path keeps the
    one-shot vectorised form.

    Parameters
    ----------
    chunk : scipy.sparse.csr_matrix / spmatrix / np.ndarray
        (n_cells_chunk, n_features) raw counts.
    cell_depth_chunk : np.ndarray (n_cells_chunk,)
        Per-cell total counts for THIS chunk's rows. Zero entries are
        replaced with 1.0 to avoid divide-by-zero.
    scale_factor : float
        Target sum (e.g. 1e4) — every cell's row is scaled to this total
        before the log.

    Returns
    -------
    Same type as input: csr_matrix if sparse, ndarray if dense.
    """
    cd = np.asarray(cell_depth_chunk, dtype=np.float64)
    cd = np.where(cd == 0, 1.0, cd)
    scale = float(scale_factor)

    if sp.issparse(chunk):
        out = chunk.copy().astype(np.float32)
        if not sp.isspmatrix_csr(out):
            out = out.tocsr()
        # Row-scale in place via the indptr (same pattern as
        # _normalize_log1p_anndata above)
        for i in range(out.shape[0]):
            start, end = out.indptr[i], out.indptr[i + 1]
            if end > start:
                out.data[start:end] = out.data[start:end] / cd[i] * scale
        # log1p in place on the data array — log1p(0) == 0, so sparse
        # structure is preserved.
        np.log1p(out.data, out=out.data)
        return out

    # Dense path
    dense = np.asarray(chunk, dtype=np.float32)
    depth = cd.reshape(-1, 1)
    dense = dense / depth * scale
    return np.log1p(dense, out=dense)


def normalize_log1p(data, target_sum=1e4, key_added='log1p', save_layer=False,
                    modality='RNA', layer='counts', batch_size=1024):
    """Library-size normalization followed by log1p transform.

    For each cell, divides by total counts, scales by ``target_sum``,
    then applies ``log1p``. Result is stored in ``data.layers[key_added]``.

    Parameters
    ----------
    data : AnnData or cytome.Dataset
        Input data. For AnnData, reads raw counts from ``.X``.
        For cytome, streams from the specified modality/layer on disk.
    target_sum : float
        Target sum for per-cell normalization. Default: 1e4.
    key_added : str
        Layer name to store the result. Default: ``'log1p'``.
    save_layer : bool
        For cytome only. If True, writes the normalized layer to the
        cytome file. If False (default), only stores in-memory on the
        AnnData representation. Ignored for AnnData input.
    modality : str
        Cytome modality (default ``'RNA'``). Ignored for AnnData.
    layer : str
        Cytome layer within the modality (default ``'counts'``). Ignored for AnnData.
    batch_size : int
        Chunk size for streaming (cytome mode). Default: 1024.

    Returns
    -------
    None
        Modifies ``data`` in place.
    """
    if _is_cytome_dataset(data):
        _normalize_log1p_cytome(data, target_sum, key_added, save_layer,
                                modality, layer, batch_size)
    else:
        _normalize_log1p_anndata(data, target_sum, key_added)


def _normalize_log1p_anndata(adata, target_sum, key_added):
    """AnnData path: normalize .X and store in layers[key_added]."""
    X = adata.X
    if sp.issparse(X):
        X = X.copy().astype(np.float32)
        # Per-cell library size
        row_sums = np.array(X.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        # Normalize: divide each row by its sum, multiply by target_sum
        # For CSR: scale data in-place row by row
        if sp.isspmatrix_csr(X):
            for i in range(X.shape[0]):
                start, end = X.indptr[i], X.indptr[i + 1]
                X.data[start:end] = X.data[start:end] / row_sums[i] * target_sum
        else:
            X = X.tocsr()
            for i in range(X.shape[0]):
                start, end = X.indptr[i], X.indptr[i + 1]
                X.data[start:end] = X.data[start:end] / row_sums[i] * target_sum
        # log1p in-place on nonzero entries
        np.log1p(X.data, out=X.data)
    else:
        X = np.array(X, dtype=np.float32)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X = X / row_sums * target_sum
        X = np.log1p(X)
    adata.layers[key_added] = X


def _normalize_log1p_cytome(data, target_sum, key_added, save_layer,
                            modality, layer, batch_size):
    """Cytome path: streaming normalize and optionally persist."""
    from cytome.core.measurement import MeasurementLayer

    # Resolve matrix dimensions via the MeasurementLayer API
    matrix_name = f"{modality}_{layer}"
    try:
        ml = MeasurementLayer(data._conn, matrix_name)
        n_cols = ml.shape[1]
    except KeyError:
        raise ValueError(
            f"Matrix '{matrix_name}' not found in cytome. "
            f"Check available layers for the '{modality}' modality."
        )

    if save_layer:
        writer_name = f"{modality}_{key_added}"
        writer = data.create_layer_writer(
            writer_name, n_rows=data.n_cells, n_cols=n_cols, dtype='float32',
        )

    n_cells = int(data.n_cells)
    cell_depth = np.zeros(n_cells, dtype=np.float64)
    chunks = []
    for chunk_csr, row_indices in data.iter_chunks(
        modality=modality, layer=layer, batch_size=batch_size
    ):
        if sp.issparse(chunk_csr):
            chunk = chunk_csr.toarray().astype(np.float32)
        else:
            chunk = np.asarray(chunk_csr, dtype=np.float32)
        row_sums = chunk.sum(axis=1)
        cell_depth[row_indices] = row_sums.astype(np.float64)
        row_sums_2d = row_sums.reshape(-1, 1).astype(np.float32)
        row_sums_2d = np.where(row_sums_2d == 0, 1.0, row_sums_2d)
        chunk = chunk / row_sums_2d * target_sum
        chunk = np.log1p(chunk)
        if save_layer:
            # ChunkedLayerWriter needs the chunk's global starting row. iter_chunks
            # yields contiguous row_indices, so the first is the offset.
            writer.write_chunk(sp.csr_matrix(chunk), int(row_indices[0]))
        else:
            chunks.append(chunk)

    # Cache the params so plotting on-the-fly can reproduce per-feature.
    # Per-modality keyed; future GA / ATAC log1p calls land in their own slot.
    data.metadata[f"{modality}_log1p_params"] = {
        "cell_depth": cell_depth,
        "scale_factor": float(target_sum),
    }

    if save_layer:
        writer.finalize()
        data.flush()
        print(f"Normalized layer written to cytome: {writer_name}")
    else:
        data.flush()
        result = np.vstack(chunks)
        if not hasattr(data, '_mem_layers'):
            data._mem_layers = {}
        data._mem_layers[key_added] = result
