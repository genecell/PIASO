"""Nearest neighbor graph construction using pynndescent.

Supports both AnnData and cytome.Dataset as input.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome_dataset


def _decompress_blob(blob, compression):
    """Decompress a data blob."""
    if compression in (None, 'none', ''):
        return blob
    if compression == 'zstd':
        import zstandard
        return zstandard.ZstdDecompressor().decompress(blob)
    if compression == 'lz4':
        import lz4.block
        return lz4.block.decompress(blob)
    if compression == 'zlib':
        import zlib
        return zlib.decompress(blob)
    return blob


def _load_embedding_from_cytome(data, key, modality="RNA"):
    """Load embedding from Cytome dense_chunks table.

    Tries exact name first, then common prefixed variants
    (e.g. X_svd -> ATAC_svd, RNA_svd).
    """
    from ._embedding_names import resolve_embedding_name
    name = resolve_embedding_name(data, key, modality)
    meta_row = data._conn.execute(
        "SELECT n_cols, dtype FROM embedding_meta WHERE array_name = ?",
        (name,),
    ).fetchone()

    n_cols = meta_row[0]
    dtype = np.dtype(meta_row[1])

    rows = data._conn.execute(
        "SELECT data_blob, compression FROM dense_chunks "
        "WHERE array_name = ? ORDER BY chunk_idx",
        (name,),
    ).fetchall()
    parts = []
    for blob, compression in rows:
        raw = _decompress_blob(blob, compression)
        parts.append(np.frombuffer(raw, dtype=dtype))
    return np.concatenate(parts).reshape(-1, n_cols).astype(np.float32)


def _neighbors_graph_keys(key_added, modality=None):
    """Resolve the (connectivities, distances, n_neighbors) storage keys for a
    given ``key_added``. Mirrors the naming in :func:`neighbors` so that
    ``umap`` / ``leiden`` can locate the graphs a prior ``neighbors`` call wrote.

    ``key_added`` is a **prefix**: ``None`` (default) writes the un-prefixed
    ``connectivities`` / ``distances``; ``'SVD'`` writes
    ``SVD_connectivities`` / ``SVD_distances``. The legacy sentinel
    ``'neighbors'`` is kept as a back-compat alias for "no prefix" (so existing
    cytomes' graph names are unchanged).
    """
    # NOTE: the un-prefixed default is a tested contract
    # (test_round12_neighbors_key_added). It is also a multimodality hazard:
    # an RNA graph and an ATAC graph in one cytome both land on
    # 'connectivities', so running neighbors on the second silently overwrites
    # the first, and `key_added` is the only way to avoid it with nothing
    # telling the user so. Changing the default is a migration, not a patch,
    # so it is left alone here; `modality` is accepted and used for READS so a
    # future prefixed write is already readable.
    if key_added is None or key_added == 'neighbors':
        if modality:
            return (f'{modality}_connectivities',
                    f'{modality}_distances',
                    f'{modality}_n_neighbors')
        return 'connectivities', 'distances', 'n_neighbors'
    return (f'{key_added}_connectivities',
            f'{key_added}_distances',
            f'{key_added}_n_neighbors')


def _neighbors_graph_keys_for_read(key_added, modality=None):
    """Every (conn, dist, n_neighbors) triple a prior run might have written.

    New writes are modality-prefixed; files written before that are not, so a
    reader has to accept both or every existing cytome stops working.
    """
    out = [_neighbors_graph_keys(key_added, modality)]
    plain = _neighbors_graph_keys(key_added, None)
    if plain not in out:
        out.append(plain)
    return out


def reconstruct_knn_from_cytome(ds, neighbors_key='neighbors', modality='RNA'):
    """Rebuild ``(knn_indices, knn_dists)`` from the ``distances`` graph a prior
    ``neighbors()`` call persisted to the cytome.

    The ``distances`` graph is an ``(n_cells, n_cells)`` sparse matrix holding
    exactly the kNN distances — column indices are neighbor cell ids, values are
    distances. We reconstruct the dense ``(n_cells, n_neighbors)`` arrays UMAP's
    ``precomputed_knn`` expects: per row, the self-neighbor is re-inserted at
    distance 0 (pynndescent always makes a cell its own first neighbor; the
    explicit 0 may not survive graph serialisation), then neighbors are sorted
    by increasing distance.

    This avoids persisting separate dense kNN arrays — the graph machinery
    already holds everything needed.
    """
    _conn_key = dist_key = nn_key = None
    for _c, _d, _n in _neighbors_graph_keys_for_read(neighbors_key, modality):
        if ds.metadata.get(_n) is not None:
            _conn_key, dist_key, nn_key = _c, _d, _n
            break
    if nn_key is None:
        _conn_key, dist_key, nn_key = _neighbors_graph_keys(neighbors_key)

    n_neighbors = ds.metadata.get(nn_key)
    if n_neighbors is None:
        raise KeyError(
            f"No kNN graph found on the cytome for neighbors_key="
            f"'{neighbors_key}' (missing metadata '{nn_key}'). Run "
            f"piaso.tl.neighbors(ds, ...) first."
        )
    n_neighbors = int(n_neighbors)

    D = ds.graphs[dist_key].to_sparse().tocsr()
    n = D.shape[0]
    counts = np.diff(D.indptr)

    # Fast path: every row has exactly n_neighbors stored entries (self-zero
    # survived). Reshape + per-row argsort by distance, vectorised.
    if n_neighbors > 0 and np.all(counts == n_neighbors):
        idx = D.indices.reshape(n, n_neighbors)
        dst = D.data.reshape(n, n_neighbors).astype(np.float64)
        order = np.argsort(dst, axis=1, kind='stable')
        knn_indices = np.take_along_axis(idx, order, axis=1).astype(np.int64)
        knn_dists = np.take_along_axis(dst, order, axis=1)
        return knn_indices, knn_dists

    # Ragged fallback: re-insert the self-neighbor per row, sort, pad/truncate.
    knn_indices = np.empty((n, n_neighbors), dtype=np.int64)
    knn_dists = np.empty((n, n_neighbors), dtype=np.float64)
    for i in range(n):
        sl = slice(D.indptr[i], D.indptr[i + 1])
        cols = D.indices[sl]
        vals = D.data[sl].astype(np.float64)
        # Drop any stored self entry (explicit 0 or otherwise); we add it back.
        keep = cols != i
        cols = cols[keep]
        vals = vals[keep]
        order = np.argsort(vals, kind='stable')[: n_neighbors - 1]
        row_idx = np.concatenate(([i], cols[order]))
        row_dst = np.concatenate(([0.0], vals[order]))
        if row_idx.shape[0] < n_neighbors:
            pad = n_neighbors - row_idx.shape[0]
            row_idx = np.concatenate((row_idx, np.full(pad, i, dtype=row_idx.dtype)))
            row_dst = np.concatenate((row_dst, np.zeros(pad, dtype=row_dst.dtype)))
        knn_indices[i] = row_idx
        knn_dists[i] = row_dst
    return knn_indices, knn_dists


def neighbors(
    data,
    use_rep='X_svd',
    n_neighbors=15,
    modality='RNA',
    metric='euclidean',
    random_state=42,
    key_added=None,
    cell_mask=None,
):
    """Build kNN graph and compute fuzzy simplicial set connectivities.

    Uses pynndescent for approximate nearest neighbor search and
    umap's fuzzy_simplicial_set for UMAP-compatible connectivities.

    Parameters
    ----------
    data : AnnData or cytome.Dataset
        If AnnData: reads from obsm, stores in obsp/uns.
        If cytome.Dataset: reads from embeddings, stores graphs in cytome.
    use_rep : str
        Embedding name. For AnnData: key in ``obsm``. For cytome: embedding name.
    n_neighbors : int
        Number of nearest neighbors.
    metric : str
        Distance metric for pynndescent.
    random_state : int
        Random seed for reproducibility.
    key_added : str, optional
        **Prefix** for the stored graph names. ``None`` (default) writes the
        un-prefixed ``connectivities`` / ``distances`` (+ ``n_neighbors``
        metadata); ``'SVD'`` writes ``SVD_connectivities`` / ``SVD_distances``.
        The legacy value ``'neighbors'`` is a back-compat alias for "no prefix".
        Pass the same string as ``neighbors_key`` to ``leiden`` / ``umap``.

    Returns
    -------
    dict or None
        For AnnData and the in-memory ndarray / ``cell_mask`` paths, returns a
        dict with 'knn_indices', 'knn_dists', 'connectivities', 'distances'.
        For a **cytome.Dataset** (no ``cell_mask``) returns ``None`` — the graph
        is persisted on the cytome (``connectivities`` / ``distances`` graphs +
        an ``n_neighbors`` metadata entry), and ``piaso.tl.umap`` /
        ``piaso.tl.leiden`` read it back from there. The function is
        self-contained: no value passing required.
    """
    # A cytome path is as valid an input as it is for infog / runSVD / runGDR.
    # Without this a pipeline written entirely with a path worked for three
    # calls and then raised "'str' object has no attribute 'obsm'" on the
    # fourth, which reads like the caller passed the wrong type.
    if isinstance(data, str):
        from ._normalization import _open_cytome
        data = _open_cytome(data)

    from pynndescent import NNDescent
    from umap.umap_ import fuzzy_simplicial_set

    # Accept an in-memory ndarray as ``data`` for masked / per-batch
    # use (e.g. runGDR multi-batch path). When ndarray, no cytome / obsm
    # writeback is attempted — caller gets the knn_result dict only.
    _is_array_input = isinstance(data, np.ndarray)
    if _is_array_input:
        X = data
    elif _is_cytome_dataset(data):
        X = _load_embedding_from_cytome(data, use_rep, modality)
        # Truncate if embedding has more rows than cells (stale metadata)
        true_n = data.n_cells
        if X.shape[0] > true_n:
            X = X[:true_n]
    else:
        X = data.obsm[use_rep]

    # cell_mask: build KNN on masked subset only. Output knn_result has
    # n_masked rows (graph is a (n_masked, n_masked) adjacency). Used by
    # multi-batch GDR's per-batch auto-cluster.
    if cell_mask is not None:
        cell_mask_arr = np.asarray(cell_mask).astype(bool)
        if cell_mask_arr.shape[0] != X.shape[0]:
            raise ValueError(
                f"cell_mask length ({cell_mask_arr.shape[0]}) must match "
                f"embedding rows ({X.shape[0]})."
            )
        X = X[cell_mask_arr]

    n_cells = X.shape[0]

    # kNN search
    index = NNDescent(X, n_neighbors=n_neighbors, metric=metric, random_state=random_state)
    knn_indices, knn_dists = index.neighbor_graph

    # Build sparse distance matrix (n_cells x n_cells)
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = knn_indices.ravel()
    dists = knn_dists.ravel()
    distances = csr_matrix(
        coo_matrix((dists, (rows, cols)), shape=(n_cells, n_cells))
    )

    # Fuzzy simplicial set (UMAP connectivities)
    connectivities, _sigmas, _rhos = fuzzy_simplicial_set(
        X,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    connectivities = connectivities.tocsr()

    result = {
        'knn_indices': knn_indices,
        'knn_dists': knn_dists,
        'connectivities': connectivities,
        'distances': distances,
    }

    # Resolve per-variant key names so sweep_cluster jobs writing to a
    # SHARED cytome don't race on hardcoded graph names. Round 12
    # follow-up (2026-05-27): the previous hardcoded 'connectivities'
    # and 'distances' caused concurrent sweep_cluster jobs to overwrite
    # each other's KNN graphs, contaminating downstream Leiden ARI.
    conn_key, dist_key, nn_key = _neighbors_graph_keys(key_added)

    # Store results.
    # When data is a raw ndarray, or cell_mask was set (in which case the
    # graph dimensions are n_masked != n_total and can't be written into
    # a cells-aligned storage), skip the writeback and return knn_result
    # for the caller to consume (the one self-containment exception — the
    # transient per-batch GDR path).
    if _is_array_input or cell_mask is not None:
        return result
    elif _is_cytome_dataset(data):
        data.add_graph(conn_key, connectivities, axis='obs', entity_table='cells')
        data.add_graph(dist_key, distances, axis='obs', entity_table='cells')
        # Persist n_neighbors so umap() can reconstruct knn_indices/knn_dists
        # from the distances graph — no separate dense kNN arrays needed. Also
        # persist use_rep so umap() loads the SAME representation the graph was
        # built on (and can warn on mismatch) — symmetry with the AnnData path
        # which reads use_rep from uns['neighbors']['params'].
        data.metadata[nn_key] = int(n_neighbors)
        data.metadata[nn_key.replace('n_neighbors', 'use_rep')] = str(use_rep)
        data.flush()
        # Strict / self-contained: results live on the cytome, nothing returned.
        return None
    else:
        # AnnData path (scanpy-compatible format). uns keys can't be None;
        # the un-prefixed default maps to the conventional 'neighbors' uns slot
        # (umap looks for '_neighbors_knn_indices').
        uns_key = key_added if key_added is not None else 'neighbors'

        data.obsp[dist_key] = distances
        data.obsp[conn_key] = connectivities
        data.uns[uns_key] = {
            'connectivities_key': conn_key,
            'distances_key': dist_key,
            'params': {
                'n_neighbors': n_neighbors,
                'method': 'umap',
                'random_state': random_state,
                'metric': metric,
                'use_rep': use_rep,
            },
        }
        data.uns[f'_{uns_key}_knn_indices'] = knn_indices
        data.uns[f'_{uns_key}_knn_dists'] = knn_dists

    return result
