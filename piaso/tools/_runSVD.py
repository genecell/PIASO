from ._normalization import infog, _is_cytome_dataset, _open_cytome, _normalize_chunk_infog, _safe_n_cells
from ..settings import _resolve_layer_dtype
from ._compat import resolve_data_arg as _resolve_data_arg, _UNSET

### run SVD
import os
import warnings
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from anndata import AnnData
import numpy as np
from scipy import sparse


# Round 12 (2026-05-26):
# - Single canonical column-name kwarg ``selected_feature_col_name`` for
#   the boolean column that marks "use these features for SVD". The
#   default is ``'highly_variable'`` (matches piaso.tl.infog and scanpy's
#   highly_variable_genes output) so RNA / GA users with the conventional
#   column name just work without passing the kwarg.
# - For ATAC and tiles, the historical default column name was
#   ``'selected'`` (written by piaso.preprocessing.selectPeaks). To
#   avoid silently breaking existing pipelines, the resolver below
#   issues a DeprecationWarning and falls back to ``'selected'`` when
#   the user passes the default AND the new name is absent AND the
#   legacy name IS present. Pass ``selected_feature_col_name='selected'``
#   to silence the warning.
_DEFAULT_SELECTED_COL = 'highly_variable'
_LEGACY_ATAC_TILES_COL = 'selected'
_LEGACY_FALLBACK_MODALITIES = {'ATAC', 'tiles'}


def _resolve_selected_col_anndata(adata, modality, user_value):
    """Pick the column to read from adata.var for the HVG/selected mask.

    Returns the column name (str). Raises if nothing usable is present.
    """
    if user_value != _DEFAULT_SELECTED_COL:
        if user_value not in adata.var.columns:
            raise KeyError(
                f"runSVD: selected_feature_col_name={user_value!r} not found "
                f"in adata.var.columns. Available: {list(adata.var.columns)}"
            )
        return user_value
    # User passed the default. Try canonical first.
    if _DEFAULT_SELECTED_COL in adata.var.columns:
        return _DEFAULT_SELECTED_COL
    # ATAC / tiles backward-compat fallback.
    if modality in _LEGACY_FALLBACK_MODALITIES \
            and _LEGACY_ATAC_TILES_COL in adata.var.columns:
        warnings.warn(
            f"runSVD(modality={modality!r}): falling back to legacy "
            f"adata.var[{_LEGACY_ATAC_TILES_COL!r}] column. The canonical "
            f"name is {_DEFAULT_SELECTED_COL!r} — rename your column or "
            f"pass selected_feature_col_name={_LEGACY_ATAC_TILES_COL!r} "
            f"explicitly to silence this warning. Future versions may "
            f"drop the fallback.",
            DeprecationWarning,
            stacklevel=3,
        )
        return _LEGACY_ATAC_TILES_COL
    raise KeyError(
        f"runSVD: neither {_DEFAULT_SELECTED_COL!r} nor "
        f"{_LEGACY_ATAC_TILES_COL!r} found in adata.var. Set "
        f"selected_feature_col_name= to the column you want."
    )


def _resolve_selected_col_cytome(ds, modality, user_value):
    """Pick the column to read from the modality's var-entity table for
    the HVG/selected mask. Returns ``(entity_name, column_name)``.

    Reads the entity schema via SQL rather than ``getattr(ds, name)``
    because the latter resolves to a ``Modality`` (not an ``EntityTable``)
    for tiles where the modality and entity share a name.
    """
    from cytome import modality_var_entity
    var_entity, _ = modality_var_entity(modality)
    cols = [r[1] for r in ds._conn.execute(
        f"PRAGMA table_info({var_entity})"
    ).fetchall()]
    if user_value != _DEFAULT_SELECTED_COL:
        if user_value not in cols:
            raise KeyError(
                f"runSVD: selected_feature_col_name={user_value!r} not found "
                f"in {var_entity} columns. Available: {cols}"
            )
        return var_entity, user_value
    if _DEFAULT_SELECTED_COL in cols:
        return var_entity, _DEFAULT_SELECTED_COL
    if modality in _LEGACY_FALLBACK_MODALITIES \
            and _LEGACY_ATAC_TILES_COL in cols:
        warnings.warn(
            f"runSVD(modality={modality!r}): falling back to legacy "
            f"{var_entity}.{_LEGACY_ATAC_TILES_COL!r} column. The "
            f"canonical name is {_DEFAULT_SELECTED_COL!r} — rerun "
            f"selectPeaks/infog with key_added={_DEFAULT_SELECTED_COL!r} "
            f"or pass selected_feature_col_name={_LEGACY_ATAC_TILES_COL!r} "
            f"explicitly to silence this warning. Future versions may "
            f"drop the fallback.",
            DeprecationWarning,
            stacklevel=3,
        )
        return var_entity, _LEGACY_ATAC_TILES_COL
    raise KeyError(
        f"runSVD: neither {_DEFAULT_SELECTED_COL!r} nor "
        f"{_LEGACY_ATAC_TILES_COL!r} found in {var_entity}. Run "
        f"selectPeaks or infog first, or pass selected_feature_col_name= "
        f"to the column you want."
    )


def _read_var_entity_column(ds, var_entity, col):
    """Read a single column from a var-entity table by name, returning a
    numpy array ordered by the table's INTEGER PRIMARY KEY (peak_idx /
    gene_idx / tile_idx). Uses SQL so it works uniformly for peaks /
    genes / GA_genes / tiles even when the entity name collides with a
    modality name (tiles)."""
    rows = ds._conn.execute(
        f"SELECT {col} FROM {var_entity} ORDER BY rowid"
    ).fetchall()
    return np.asarray([r[0] for r in rows])


# ============================================================================
# Original runSVD — COMPLETELY UNCHANGED body
# ============================================================================
def _runSVD_original(
    adata: AnnData,
    use_highly_variable: bool = True,
    n_components: int = 50,
    random_state: Optional[int] = 10,
    scale_data: bool = False,
    n_iter: int = 7,
    key_added: str = 'X_svd',
    layer: Optional[str] = None,
    verbosity: int = 0,
    selected_feature_col_name: str = _DEFAULT_SELECTED_COL,
    modality: str = "RNA",
):
    """
    Performs Truncated Singular Value Decomposition (SVD) on the specified gene expression matrix (adata.X or a specified layer)
    within an AnnData object and stores the resulting low-dimensional representation in `adata.obsm`.

    Parameters
    ----------
    adata : AnnData
        An AnnData object.
    use_highly_variable : bool, optional, default=True
        If True, the decomposition is performed only on highly variable genes/features.
    n_components : int, optional, default=50
        The number of principal components to retain.
    random_state : int, optional, default=10
        A random seed to ensure reproducibility.
    scale_data : bool, optional, default=False
        If True, standardizes the input data before performing SVD.
    n_iter : int, optional, default=7
        Number of iterations for randomized SVD solver. The default is larger than the default in randomized_svd to handle sparse matrices that may have large slowly decaying spectrum. Also larger than the `n_iter` default value (5) in the TruncatedSVD function.
    key_added : str, optional, default='X_svd'
        The key under which the resulting cell embeddings are stored in `adata.obsm`.
    layer : str, optional, default=None
        Specifies which layer of `adata` to use for the transformation. If None, `adata.X` is used.
    verbosity : int, optional, default=0
        Controls the verbosity of logging messages.

    Returns
    -------
    None
        The function modifies `adata` in place, storing the cell embeddings in `adata.obsm[key_added]`.

    Example
    -------
    >>> import piaso
    >>> piaso.tl.runSVD(adata, use_highly_variable=True, n_components=50, random_state=42,
    ...        scale_data=False, key_added='X_svd', layer=None)
    >>>
    >>> # Access the transformed data
    >>> adata.obsm['X_svd']
    """

    if layer and layer not in adata.layers:
        raise ValueError(f"{layer} not found in adata.layers.")

    if use_highly_variable:
        hv_col = _resolve_selected_col_anndata(adata, modality, selected_feature_col_name)
        # Coerce rather than index with the raw column: a highly_variable that
        # round-tripped through h5ad as `category` raises "Unknown indexer"
        # here, and reads as ALL TRUE wherever .astype(bool) is used instead.
        from ..utils._bool_mask import as_bool_mask
        _hv = as_bool_mask(adata.var[hv_col].values, name=f"var.{hv_col}")
        if layer:
            expr = adata[:, _hv].layers[layer]
        else:
            expr = adata[:, _hv].X
    else:
        if layer:
            expr = adata.layers[layer]
        else:
            expr = adata.X


    if scale_data:
        expr = StandardScaler(with_mean=False).fit_transform(expr)

    transformer = TruncatedSVD(n_components=n_components, random_state=random_state, n_iter = n_iter)
    adata.obsm[key_added] = transformer.fit_transform(expr)

    if verbosity>0:
        print(f'The cell embeddings were saved as `{key_added}` in adata.obsm.')


def _engine_threads(n_threads):
    """The threads the engine will use: the caller's number, or every core
    this process is allowed (its CPU affinity -- on a cluster, the job's
    allocation, not the node's)."""
    if n_threads and int(n_threads) > 0:
        return int(n_threads)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _cached_factory(factory):
    """A chunk factory that reads the store once and replays from memory."""
    cache = []

    def cached():
        if not cache:
            for chunk, idx in factory():
                cache.append((chunk, idx))
        return iter(cache)
    return cached


# ============================================================================
# Streaming randomized SVD (HMT algorithm)
# ============================================================================
def _streaming_rsvd(chunk_iter_factory, n_cells, n_features, n_components, n_iter,
                    random_state, oversampling=10, verbosity=0, cache_chunks=False,
                    tol=None, method="power"):
    """
    Streaming randomized SVD using Halko-Martinsson-Tropp algorithm.

    Parameters
    ----------
    chunk_iter_factory : callable
        Returns a fresh iterator yielding (csr_chunk, row_indices) each call.
    n_cells, n_features : int
        Matrix dimensions.
    n_components : int
        Number of SVD components.
    n_iter : int
        Number of power iterations.
    random_state : int or None
        Random seed.
    oversampling : int
        Extra sketch dimensions beyond n_components.
    cache_chunks : bool
        If True, cache all chunks in memory on first pass. Subsequent passes
        read from RAM instead of re-reading from disk. Trades ~2-4 GB RAM for
        ~8x fewer I/O passes. Recommended when RAM is available.

    Returns
    -------
    embeddings : ndarray (n_cells, n_components)
        U * S — the cell embeddings.
    S : ndarray (n_components,)
        Singular values.
    Vt : ndarray (n_components, n_features)
        Right singular vectors.
    """
    k = n_components + oversampling
    rng = np.random.RandomState(random_state)

    total_passes = 3 + 2 * n_iter
    pass_num = [0]

    def _log_pass(desc):
        pass_num[0] += 1
        if verbosity > 0:
            print(f"  SVD pass {pass_num[0]}/{total_passes}: {desc}")

    if cache_chunks:
        import gc
        if verbosity > 0:
            print("  Caching all SVD chunks in memory...")
        cached = []
        for _ci, _item in enumerate(chunk_iter_factory()):
            cached.append(_item)
            if (_ci + 1) % 50 == 0:
                gc.collect()
        if len(cached) % 50 != 0:
            gc.collect()
        total_bytes = sum(
            (c.data.nbytes + c.indices.nbytes + c.indptr.nbytes
             if sparse.issparse(c) else c.nbytes)
            for c, _ in cached
        )
        if verbosity > 0:
            print(f"  Cached {len(cached)} chunks, ~{total_bytes / 1e9:.1f} GB")
        _original_factory = chunk_iter_factory
        def chunk_iter_factory():
            return iter(cached)

    from ._rsvd_core import FactoryOps, rsvd_core
    embeddings, S, Vt, _info = rsvd_core(
        FactoryOps(chunk_iter_factory, n_cells, n_features),
        n_components=n_components, n_iter=n_iter, random_state=random_state,
        oversampling=oversampling, tol=tol, method=method, verbosity=verbosity,
    )
    return embeddings, S, Vt


def _mask_to_indices(cell_mask, n_rows):
    """Normalise a cell mask to sorted row indices within ``n_rows``."""
    arr = np.asarray(cell_mask)
    if arr.dtype == bool:
        if arr.shape[0] < n_rows:
            padded = np.zeros(n_rows, dtype=bool)
            padded[:arr.shape[0]] = arr
            arr = padded
        elif arr.shape[0] > n_rows:
            arr = arr[:n_rows]
        return np.flatnonzero(arr).astype(np.int64)
    return np.sort(arr[arr < n_rows].astype(np.int64))


def _assert_enough_features(n_features, n_components, oversampling, column):
    """Refuse an SVD that cannot mean anything, before it costs passes.

    Zero selected features factorises a ``cells x 0`` matrix: every pass runs,
    nothing objects, and the embedding is noise. There is already a guard for
    the opposite mistake -- ``as_bool_mask`` warns when EVERY entry is
    selected, the signature of a text-typed boolean -- and none for this,
    which is worse, because all-true at least computes something.
    """
    import warnings as _w

    if n_features == 0:
        raise ValueError(
            f"runSVD: no features are selected, so there is nothing to "
            f"factorise ({column!r} is all False). Either the selection "
            f"really is empty -- check the thresholds you passed to "
            f"selectPeaks / selectFeatures -- or the column was written but "
            f"not flushed, in which case call ds.flush() before runSVD, "
            f"because a cytome buffers a column write until then."
        )
    need = int(n_components) + int(oversampling)
    if n_features < need:
        _w.warn(
            f"runSVD: {n_features} selected features is fewer than "
            f"n_components + oversampling ({n_components} + {oversampling} = "
            f"{need}), so the requested rank cannot be reached and the "
            f"trailing components will be arbitrary. Select more features or "
            f"lower n_components.",
            UserWarning, stacklevel=3,
        )


def _get_svd_chunk_iterator(source, hvg_indices, batch_size, layer, measurement,
                            modality="RNA", compute_on_fly=True, cell_mask=None,
                            infog_params=None):
    """Get chunk iterator factory for SVD — reads HVG columns only.

    ``cell_mask`` is pushed down into ``iter_chunks`` on the cytome path so
    on-disk chunks holding no selected cell are never fetched. Masking above
    this function instead meant a per-batch GDR decompressed the whole matrix
    once per batch per SVD pass. Returns a 4th element, ``mask_applied``,
    telling the caller the rows are already filtered (it still has to renumber
    them into the compressed output space).
    """
    if isinstance(source, str) or _is_cytome_dataset(source):
        ds = _open_cytome(source) if isinstance(source, str) else source
        _layer = measurement or layer or "infog"

        # Check if compact HVG measurement exists
        _has_compact = False
        if measurement:
            try:
                meta = ds._conn.execute(
                    "SELECT 1 FROM matrix_meta WHERE matrix_name = ?",
                    (f"{modality}_{measurement}",)
                ).fetchone()
                _has_compact = meta is not None
            except Exception:
                pass

        if _has_compact:
            n_alloc, _n_true = _safe_n_cells(ds, modality, measurement)
            meta = ds._conn.execute(
                "SELECT n_cols FROM matrix_meta WHERE matrix_name = ?",
                (f"{modality}_{measurement}",)
            ).fetchone()
            n_cols_total = meta[0]
            n_features = len(hvg_indices) if hvg_indices is not None else n_cols_total
            _keep = None if cell_mask is None else _mask_to_indices(cell_mask, n_alloc)
            def factory():
                for chunk, idx in ds.iter_chunks(modality=modality, layer=measurement,
                                                 cell_mask=_keep, batch_size=batch_size):
                    if hvg_indices is not None:
                        yield chunk[:, hvg_indices], idx
                    else:
                        yield chunk, idx
            # A materialised measurement is read as it is: the identity
            # transform, which the Rust pass can take over.
            factory.read_layer = measurement
            factory.identity = True
            n_out = n_alloc if _keep is None else len(_keep)
            return n_out, n_features, factory, _keep is not None
        else:
            # A normalization PIASO knows how to recompute does not need a
            # materialised layer: infog() records its parameters and writes no
            # matrix by default, which made infog() + runSVD(layer='infog')
            # fail with "Matrix not found: RNA_infog" several minutes in.
            _read_layer, _chunk_norm = _layer, None
            if infog_params is not None:
                # Explicit per-batch parameters: normalise raw counts with the
                # statistics of THIS batch rather than whatever whole-dataset
                # INFOG the file happens to hold.
                from ._normalization import _normalize_chunk_infog
                _ip = infog_params
                _read_layer = "counts"

                def _chunk_norm(chunk, indices, _ip=_ip):
                    return _normalize_chunk_infog(
                        chunk, _ip["cell_depth"][indices], _ip["inv_gene_depth"],
                        _ip["scale"], _ip["counts_sum"], _ip.get("threshold"),
                    )
            elif _layer in ("infog",):
                _read_layer, _chunk_norm = _infog_chunk_normalizer(
                    ds, modality, _layer, compute_on_fly)

            _layer_for_check = _read_layer if _read_layer != "infog" else "counts"
            n_alloc, _n_true = _safe_n_cells(ds, modality, _layer_for_check)
            n_features = len(hvg_indices) if hvg_indices is not None else ds.n_genes
            _keep = None if cell_mask is None else _mask_to_indices(cell_mask, n_alloc)
            def factory():
                # idx stays global, so the normalizer's per-cell lookups
                # (cell_depth[idx]) remain correct under a pushed-down mask.
                for chunk, idx in ds.iter_chunks(modality=modality,
                                                 layer=_read_layer,
                                                 cell_mask=_keep,
                                                 batch_size=batch_size):
                    if _chunk_norm is not None:
                        chunk = _chunk_norm(chunk, idx)
                    if hvg_indices is not None:
                        yield chunk[:, hvg_indices], idx
                    else:
                        yield chunk, idx
            # What the factory does to a stored chunk, for the caller that
            # can skip the factory: the Rust pass reads a layer as it is, so
            # it can stand in whenever the per-chunk transform is the identity.
            factory.read_layer = _read_layer
            factory.identity = _chunk_norm is None
            n_out = n_alloc if _keep is None else len(_keep)
            return n_out, n_features, factory, _keep is not None
    else:
        # AnnData
        X = source.layers[layer] if layer else source.X
        if hvg_indices is not None:
            X = X[:, hvg_indices]
        n_cells, n_features = X.shape
        def factory():
            for i in range(0, n_cells, batch_size):
                end = min(i + batch_size, n_cells)
                chunk = X[i:end]
                if not sparse.issparse(chunk):
                    chunk = sparse.csr_matrix(chunk)
                yield chunk, np.arange(i, end)
        # AnnData is already in memory: no I/O to save by masking here, so the
        # caller filters as before.
        return n_cells, n_features, factory, False


def _runSVD_streaming(
    source,
    use_highly_variable: bool = True,
    n_components: int = 50,
    n_iter: int = 7,
    random_state: Optional[int] = 10,
    scale_data: bool = False,
    oversampling: int = 10,
    batch_size: int = 1024,
    key_added: str = 'X_svd',
    layer: Optional[str] = None,
    measurement: Optional[str] = None,
    modality: str = "RNA",
    verbosity: int = 0,
    cache_chunks: bool = False,
    tfidf_params: Optional[dict] = None,
    infog_params: Optional[dict] = None,
    hvg_indices_override=None,
    compute_on_fly: bool = True,
    tol: Optional[float] = None,
    method: str = "auto",
    n_threads: int = 0,
    use_engine: bool = True,
    selected_feature_col_name: str = _DEFAULT_SELECTED_COL,
    auto_tfidf: bool = False,
    _auto_tfidf_inferred: bool = False,
    cell_mask=None,
    return_svd: bool = False,
):
    """Streaming randomized SVD implementation."""
    is_adata = isinstance(source, AnnData)
    is_cytome = isinstance(source, str) or _is_cytome_dataset(source)

    if scale_data:
        raise NotImplementedError("scale_data=True not yet supported in streaming SVD mode.")

    # None reaches here only from `method='auto'` with no explicit count; the
    # other solvers need a number.
    if n_iter is None and method != "auto":
        n_iter = 7

    # Auto-compute TF-IDF stats if requested and not already provided.
    # Mirrors how the workflow used to do it externally in select_tfidf_svd.py.
    if (auto_tfidf and tfidf_params is None and is_cytome
            and modality in _LEGACY_FALLBACK_MODALITIES):
        from ._runTFIDF import _load_or_compute_tfidf_stats, assert_raw_counts
        _ds_for_tfidf = _open_cytome(source) if isinstance(source, str) else source
        # One chunk, against the passes about to follow. Both the engine and
        # the Python path apply log1p(v/depth*scale)*idf to whatever this
        # layer holds, so an already-normalised layer silently normalises
        # twice. Asked for explicitly, that is an error; switched on by the
        # default, it only means this layer is not what the default assumed,
        # and the SVD runs on the values as stored, as it did before.
        try:
            assert_raw_counts(_ds_for_tfidf, modality, measurement or "counts",
                              what="auto_tfidf")
            _apply_tfidf = True
        except ValueError as _e:
            if not _auto_tfidf_inferred:
                raise
            warnings.warn(
                f"runSVD: TF-IDF is applied by default to {modality} counts, but "
                f"{modality}_{measurement or 'counts'} does not look like raw "
                f"counts ({_e}). Factorising the values as stored; pass "
                f"auto_tfidf=False to silence this.", UserWarning, stacklevel=3)
            _apply_tfidf = False
        if _apply_tfidf:
            if verbosity > 0:
                print(f"auto_tfidf=True: loading or computing TF-IDF stats for "
                      f"modality={modality!r}")
            tfidf_params = _load_or_compute_tfidf_stats(
                _ds_for_tfidf, modality=modality,
                layer=measurement or "counts",
                batch_size=batch_size, verbosity=verbosity,
            )
            tfidf_params = dict(tfidf_params)  # shallow copy so the cached dict isn't mutated
            # The selected features become the column mask -- unless the
            # caller asked for every feature.
            if use_highly_variable:
                _entity_name, _col = _resolve_selected_col_cytome(
                    _ds_for_tfidf, modality, selected_feature_col_name,
                )
                from ..utils._bool_mask import as_bool_mask
                tfidf_params["col_mask"] = as_bool_mask(
                    _read_var_entity_column(_ds_for_tfidf, _entity_name, _col),
                    name=f"{_entity_name}.{_col}",
                )
            else:
                tfidf_params.pop("col_mask", None)
    elif auto_tfidf and not is_cytome:
        raise ValueError(
            "auto_tfidf=True is only supported on a cytome source "
            "(string path or cytome.Dataset). Pre-compute TF-IDF "
            "for in-memory AnnData via piaso.tl.run_TFIDF."
        )

    # When tfidf_params has col_mask, skip HVG lookup (TF-IDF handles column slicing)
    _tfidf_col_mask = tfidf_params.get("col_mask", None) if tfidf_params else None

    # Determine HVG/selected-peak indices
    hvg_indices = None
    if _tfidf_col_mask is not None:
        # Fused TF-IDF mode: col_mask handles column slicing inside the wrapper
        pass
    elif use_highly_variable:
        if is_adata:
            hv_col = _resolve_selected_col_anndata(
                source, modality, selected_feature_col_name,
            )
            # Same coercion as the cytome sites: an h5ad whose highly_variable
            # was written as object or category round-trips as `category`, and
            # .astype(bool) on that is ALL TRUE -- every gene selected, silently.
            from ..utils._bool_mask import as_bool_mask
            hvg_indices = np.where(
                as_bool_mask(source.var[hv_col].values, name=f"var.{hv_col}")
            )[0]
        elif is_cytome:
            ds = _open_cytome(source) if isinstance(source, str) else source
            entity_name, col = _resolve_selected_col_cytome(
                ds, modality, selected_feature_col_name,
            )
            from ..utils._bool_mask import as_bool_mask
            hv_col = _read_var_entity_column(ds, entity_name, col)
            hvg_indices = np.where(
                as_bool_mask(hv_col, name=f"{entity_name}.{col}",
                             source=str(getattr(ds, "path", "")) or None)
            )[0]

    if hvg_indices_override is not None:
        hvg_indices = np.asarray(hvg_indices_override)
    n_cells, n_features, chunk_factory, _mask_pushed_down = _get_svd_chunk_iterator(
        source, hvg_indices, batch_size, layer, measurement,
        modality=modality, compute_on_fly=compute_on_fly, cell_mask=cell_mask,
        infog_params=infog_params,
    )

    # cell_mask: filter chunk rows before SVD ingests them. The SVD sees
    # only masked rows; output shape is (n_masked, n_components). Used by
    # multi-batch GDR's per-batch auto-cluster path (no cytome write).
    if cell_mask is not None:
        _orig_factory_for_mask = chunk_factory

        if _mask_pushed_down:
            # Rows were filtered at the storage layer; n_cells is already the
            # masked count. Only the renumbering into compressed space is left.
            def chunk_factory():
                out_offset = 0
                for chunk, _indices in _orig_factory_for_mask():
                    n_in_chunk = chunk.shape[0]
                    compressed_idx = np.arange(out_offset, out_offset + n_in_chunk)
                    out_offset += n_in_chunk
                    yield chunk, compressed_idx
        else:
            cell_mask_arr = np.asarray(cell_mask).astype(bool)
            # Cytome may over-allocate cells; pad mask to n_cells if needed
            if cell_mask_arr.shape[0] < n_cells:
                mask_full = np.zeros(n_cells, dtype=bool)
                mask_full[:cell_mask_arr.shape[0]] = cell_mask_arr
                cell_mask_arr = mask_full
            elif cell_mask_arr.shape[0] > n_cells:
                cell_mask_arr = cell_mask_arr[:n_cells]
            n_masked_cells = int(cell_mask_arr.sum())

            def chunk_factory():
                out_offset = 0
                for chunk, indices in _orig_factory_for_mask():
                    idx = np.asarray(indices)
                    keep = cell_mask_arr[idx]
                    if not keep.any():
                        continue
                    if not keep.all():
                        chunk = chunk[keep]
                    n_in_chunk = chunk.shape[0]
                    # Re-index rows into compressed (masked) space
                    compressed_idx = np.arange(out_offset, out_offset + n_in_chunk)
                    out_offset += n_in_chunk
                    yield chunk, compressed_idx

            n_cells = n_masked_cells

    # Wrap chunk_factory with inline TF-IDF if tfidf_params provided
    if tfidf_params is not None:
        from ._runTFIDF import _normalize_chunk_tfidf
        _td_cell_depth = tfidf_params["cell_depth"]
        _td_idf = tfidf_params["idf"]
        _td_scale = tfidf_params.get("scale_factor", 1e4)
        _original_factory = chunk_factory

        # Recompute n_features if col_mask is applied
        if _tfidf_col_mask is not None:
            n_features = int(_tfidf_col_mask.sum())

        def chunk_factory():
            for chunk, indices in _original_factory():
                tfidf_chunk = _normalize_chunk_tfidf(
                    chunk, _td_cell_depth[indices], _td_idf, _td_scale
                )
                if _tfidf_col_mask is not None:
                    tfidf_chunk = tfidf_chunk[:, _tfidf_col_mask] if sparse.issparse(tfidf_chunk) else tfidf_chunk[:, _tfidf_col_mask]
                yield tfidf_chunk, indices

        if verbosity > 0:
            print(f"  Fused TF-IDF+SVD: inline TF-IDF on {n_features} features")

    # Every path lands here with its final feature count, so one guard covers
    # the selected-column path, explicit hvg indices and the unfiltered case.
    _assert_enough_features(n_features, n_components, oversampling,
                            selected_feature_col_name)

    if verbosity > 0:
        # Say the shape here and let the solver say what it will do. Printing
        # "{n_iter} power iterations, {3 + 2 * n_iter} passes" before the
        # solver is resolved announced numbers that never happened whenever
        # method='auto' chose Lanczos, where the tolerance decides the rounds
        # and n_iter is only a ceiling.
        print(f"Streaming SVD: {n_cells} cells x {n_features} features, "
              f"{n_components} components")
        if method == "power":
            print(f"  Solver: power iteration"
                  + (f" (tol={tol}, at most {n_iter} rounds = "
                     f"{2 + 2 * n_iter} passes)" if tol is not None
                     else f" ({n_iter} rounds = {2 + 2 * n_iter} passes)"))

    # The Rust pass engine covers every cytome SVD whose per-chunk transform
    # it can do itself: TF-IDF on the fly (the peak and tile matrices), or
    # none at all -- raw counts, or a materialised layer such as INFOG read
    # as it is, with the selected features as a column mask. Only a
    # transform computed in Python per chunk (INFOG on the fly, explicit
    # per-batch parameters) and AnnData keep the chunk-factory path.
    _engine_ops = None
    _identity = tfidf_params is None and bool(getattr(chunk_factory, "identity", False))
    if use_engine and is_cytome and (
            (tfidf_params is not None and hvg_indices is None) or _identity):
        try:
            from .. import _piaso as _ext
            _has_engine = hasattr(_ext, "svd_pass")
        except Exception:
            _has_engine = False
        if _has_engine:
            from ._rsvd_core import EngineOps
            _ds_e = _open_cytome(source) if isinstance(source, str) else source
            if _identity:
                _mname = f"{modality}_{chunk_factory.read_layer}"
            else:
                _mname = f"{modality}_{measurement or 'counts'}"
            _nr, _nc = _ds_e._conn.execute(
                "SELECT n_rows, n_cols FROM matrix_meta WHERE matrix_name = ?", (_mname,)
            ).fetchone()
            _path = str(getattr(_ds_e, "path", source))
            _keep = None if cell_mask is None else _mask_to_indices(cell_mask, int(_nr))
            if _identity:
                _col_mask = None
                if hvg_indices is not None:
                    _col_mask = np.zeros(int(_nc), dtype=bool)
                    _col_mask[np.asarray(hvg_indices, dtype=np.int64)] = True
                _tf = None
            else:
                _col_mask = _tfidf_col_mask
                _tf = {"cell_depth": tfidf_params["cell_depth"], "idf": tfidf_params["idf"],
                       "scale_factor": tfidf_params.get("scale_factor", 1e4)}
            # Everything this call writes before the passes (the TF-IDF
            # statistics) is in the store by now, so the log can be
            # checkpointed and the passes can read without file locks.
            from ._rsvd_core import snapshot_store
            _snap, _snap_reason = snapshot_store(_ds_e)
            _engine_ops = EngineOps(
                _path, _mname, int(_nr), int(_nc),
                col_mask=_col_mask, row_keep=_keep, tfidf=_tf,
                n_threads=n_threads, snapshot=_snap,
            )
            if verbosity > 0:
                _lay = getattr(_engine_ops, "layout", "auto")
                print(f"  SVD engine: Rust pass over {_mname} "
                      f"({_engine_ops.n_rows} rows x {_engine_ops.n_cols} selected columns, "
                      f"{'column' if _lay == 'csc' else 'row'}-major store, "
                      f"{'TF-IDF on the fly' if _tf is not None else 'values as stored'})")
                print("  SVD engine: reading a snapshot of the store, no file locks"
                      if _snap else
                      f"  SVD engine: reading with file locks -- {_snap_reason}")
                print(f"  SVD engine: {_engine_threads(n_threads)} threads")
                # A row-major-only peak or tile matrix pays ~8x on the
                # transpose product: it scatters into a features x block buffer
                # instead of a cells x block one. Say so once, with the fix.
                if (_lay == "csr" and not getattr(_engine_ops, "has_csc", True)
                        and _engine_ops.n_rows < _engine_ops.n_cols):
                    print("  Note: this matrix is stored row-major only. With more "
                          "features than cells the SVD is several times faster on a "
                          "column-major store; re-quantify with "
                          "quantifyPeakActivity(..., layout='csc') (the default for "
                          "new stores) to get it.")
            if cache_chunks:
                # The engine reads, normalises, masks and multiplies in one
                # fused sweep per pass, so there is no decoded chunk to keep.
                # Saying nothing would let a caller believe the matrix is in
                # memory while every pass goes back to disk -- which is the
                # wrong trade on slow or networked storage, where the cached
                # Python path can win despite the slower kernel.
                warnings.warn(
                    "runSVD(cache_chunks=True) belongs to the Python chunk "
                    "reader; the Rust engine streams each pass from the store "
                    "and ignores it. Nothing is lost on local disk, where the "
                    "operating system keeps the store in memory between "
                    "passes. Pass use_engine=False only if you need the "
                    "in-memory cache (e.g. on slow networked storage with a "
                    "tight memory limit), at the cost of a slower kernel.",
                    UserWarning, stacklevel=3,
                )
    if _engine_ops is not None:
        import time as _time
        from ._rsvd_core import rsvd_core
        _t_solver = _time.perf_counter()
        embeddings, S, Vt, _svd_info = rsvd_core(
            _engine_ops, n_components=n_components, n_iter=n_iter,
            random_state=random_state, oversampling=oversampling,
            tol=tol, method=method, verbosity=verbosity,
        )
        if verbosity > 0:
            _wall = _time.perf_counter() - _t_solver
            print(f"  Solver time: {_wall:.1f} s, of which {_engine_ops.seconds:.1f} s in "
                  f"{_engine_ops.passes} passes over the store and "
                  f"{max(_wall - _engine_ops.seconds, 0.0):.1f} s in numpy between them")
        n_cells = _engine_ops.n_rows
    else:
        # Guarded HERE, not where the factory was built: the factory is a
        # generator, so building it over a column-major matrix costs nothing
        # and raises nothing, and the Rust engine above reads columns happily.
        # Only this fallback actually iterates rows.
        if is_cytome:
            from ..utils._cytome_compat import require_row_major
            require_row_major(
                _open_cytome(source) if isinstance(source, str) else source,
                modality, measurement or "counts",
                "runSVD's Python chunk reader",
                "Leave use_engine=True (the default) -- the Rust pass engine "
                "reads either layout -- or re-import with --layout csr.")
        if method in ("auto", "krylov"):
            # The chunk reader is an operator like any other; the Lanczos
            # solver takes it through the same core as the engine. Without
            # this, method='auto' on the fallback reached the power loop
            # with no iteration count and failed before the first pass.
            from ._rsvd_core import rsvd_core, FactoryOps
            _fac = _cached_factory(chunk_factory) if cache_chunks else chunk_factory
            embeddings, S, Vt, _svd_info = rsvd_core(
                FactoryOps(_fac, n_cells, n_features),
                n_components=n_components, n_iter=n_iter,
                random_state=random_state, oversampling=oversampling,
                tol=tol, method=method, verbosity=verbosity,
            )
        else:
            embeddings, S, Vt = _streaming_rsvd(
                chunk_factory, n_cells, n_features,
                n_components=n_components, n_iter=n_iter,
                random_state=random_state, oversampling=oversampling,
                verbosity=verbosity, cache_chunks=cache_chunks,
                tol=tol, method=method,
            )

    # Truncate to true cell count if defensive over-allocation was used
    if is_cytome and cell_mask is None:
        ds = _open_cytome(source) if isinstance(source, str) else source
        true_n = ds.n_cells
        if embeddings.shape[0] > true_n:
            embeddings = embeddings[:true_n]

    # Write results.
    # When cell_mask is set, embeddings shape is (n_masked, n_components)
    # — NOT (n_cells, n_components) — so writing to a full-cell-aligned
    # storage (obsm or cytome.embeddings) is unsafe. Skip the write and
    # let the caller handle the in-memory result.
    if cell_mask is None:
        if is_adata:
            source.obsm[key_added] = embeddings
        elif is_cytome:
            from ._embedding_names import storage_name
            ds.add_embedding(
                storage_name(key_added, modality), embeddings,
                dtype=_resolve_layer_dtype(None),
                provenance={"modality": modality, "function": "piaso.tl.runSVD",
                            "layer": layer, "key_added": key_added},
            )
            ds.flush()

    if verbosity > 0:
        print(f"Streaming SVD complete. Embeddings shape: {embeddings.shape}")

    # Return semantics: the cytome write path (is_cytome, cell_mask is None)
    # persists the embedding to the cytome -> return None by default (matches
    # runGDR's write-to-cytome convention). Keep the in-memory (emb, S, Vt)
    # tuple for: AnnData input (in-memory callers chain on it), cell_mask runs
    # (no cytome write — required by runGDR), or an explicit return_svd=True.
    if is_adata or cell_mask is not None or return_svd:
        return embeddings, S, Vt
    return None


# ============================================================================
# Public runSVD dispatcher
# ============================================================================
def runSVD(
    data=_UNSET,
    use_highly_variable: bool = True,
    n_components: int = 50,
    random_state: Optional[int] = 10,
    scale_data: bool = False,
    n_iter=_UNSET,
    key_added: str = 'X_svd',
    layer: Optional[str] = None,
    verbosity: int = 0,
    # NEW parameters:
    streaming: bool = False,
    batch_size: int = 1024,
    measurement: Optional[str] = None,
    oversampling: int = 10,
    modality: str = "RNA",
    cache_chunks: bool = False,
    tfidf_params: Optional[dict] = None,
    infog_params: Optional[dict] = None,
    hvg_indices_override=None,
    selected_feature_col_name: str = _DEFAULT_SELECTED_COL,
    tol: Optional[float] = None,
    method: str = "auto",
    n_threads: int = 0,
    use_engine: bool = True,
    auto_tfidf: Optional[bool] = None,
    cell_mask=None,
    return_svd: bool = False,
    compute_on_fly: bool = True,
    # ---- deprecated aliases (back-compat) ----
    source=_UNSET,
    adata=_UNSET,
):
    """
    Truncated SVD dimensionality reduction.

    Supports three modes:
    - ``runSVD(adata)`` — standard in-memory.
    - ``runSVD(adata, streaming=True)`` — streaming from an in-memory AnnData.
    - ``runSVD("path.cytome")`` / ``runSVD(ds)`` — streaming from a cytome
      (self-contained: writes the embedding to the cytome and returns None).

    Parameters
    ----------
    data : AnnData, cytome.Dataset, or str
        Input. An AnnData (in-memory), an open cytome ``Dataset``, or a path to
        a ``.cytome`` file (the latter two stream from disk).
    use_highly_variable : bool, default True
        Restrict SVD to features flagged in ``selected_feature_col_name``.
    n_components : int, default 50
        Number of singular components (SVD dimensions).
    random_state : int or None, default 10
        Random seed for the randomized SVD solver.
    scale_data : bool, default False
        Z-score the (selected) features before SVD.
    n_iter : int, optional
        The number of rounds. For ``'power'`` and ``'krylov'`` without
        ``tol`` it is exact (7 when left unset); with ``tol``, and for ``'auto'``,
        it is the ceiling and the tolerance decides when to stop below it
        (``'auto'`` left unset uses its own ceiling of 30). An explicit
        value is honoured by every solver.
    key_added : str, default 'X_svd'
        Name for the embedding (AnnData ``obsm[key_added]`` / cytome embedding;
        on a cytome it is stored as ``{modality}_{key_added without 'X_'}``).
    layer : str, optional
        AnnData layer to read instead of ``.X`` (in-memory path).
    verbosity : int, default 0
        Verbosity level.
    streaming : bool, default False
        Force the chunked streaming path for an in-memory AnnData. A cytome
        always streams; this has no effect on one.
    batch_size : int, default 1024
        Rows per chunk for the Python chunk reader (AnnData, or a cytome with
        ``use_engine=False``). The Rust engine reads the store's own chunks
        and ignores it.
    measurement : str, optional
        Cytome measurement/layer to read (e.g. ``'counts'``, ``'infog'``,
        ``'tfidf'``). Defaults to the modality's standard layer.
    oversampling : int, default 10
        Extra components sampled by the randomized SVD for accuracy
        (solver uses ``n_components + oversampling``).
    modality : str, default 'RNA'
        Cytome modality (``'RNA'``, ``'ATAC'``, ``'GA'``, ``'tiles'``) — routes
        the var-entity / matrix lookups via the modality registry.
    cache_chunks : bool, default False
        Python chunk reader only: keep the decoded chunks in memory after the
        first pass. The Rust engine, which a cytome uses by default, streams
        every pass from the store and ignores this flag; on local disk the
        operating system already keeps the store in memory between passes.
    tfidf_params : dict, optional
        Apply TF-IDF inline during chunk iteration. Keys: ``'cell_depth'``
        (ndarray), ``'idf'`` (ndarray), ``'scale_factor'`` (float); optional
        ``'col_mask'`` (bool ndarray) to slice to selected peaks. Avoids a
        persistent TF-IDF layer.
    selected_feature_col_name : str, default 'highly_variable'
        Boolean column in the var entity (``adata.var`` / ``ds.genes`` /
        ``ds.peaks`` / ``ds.GA_genes`` / ``ds.tiles``) marking the SVD features.
        For ATAC/tiles, the legacy ``'selected'`` column is auto-detected with a
        ``DeprecationWarning`` when the default is left untouched.
    infog_params : dict, optional
        The INFOG counterpart of ``tfidf_params``: apply INFOG normalisation
        inline during chunk iteration instead of materialising a layer.
    hvg_indices_override : ndarray, optional
        Explicit column indices to run SVD on, bypassing
        ``selected_feature_col_name`` entirely. Used by callers that have
        already selected features by another route.
    auto_tfidf : bool, optional
        Cytome only. Apply TF-IDF on the fly: load the statistics from
        ``ds.metadata['{modality}_tfidf_params']`` (or compute and cache them
        in one pass), with the column mask from ``selected_feature_col_name``.
        Left unset it is on for ``modality='ATAC'`` or ``'tiles'`` reading raw
        counts with no ``tfidf_params`` -- an SVD of raw peak or tile counts is
        never the intended analysis -- and off otherwise. Pass ``False`` to
        factorise raw counts anyway.
    method : {'auto', 'power', 'krylov'}, default 'auto'
        The randomized solver. ``'auto'`` stops when the leading singular
        values have converged (``tol=1e-3``), using block Lanczos when its
        basis fits in memory and power iteration otherwise. ``'power'`` is
        subspace iteration and ``'krylov'`` block Lanczos regardless of
        memory; **both run exactly** ``n_iter`` **rounds** unless ``tol`` is
        set, which is the way to benchmark the effect of the round count.
        A round is two passes over the store, plus one pass for the sketch
        and one for the final projection: ``2 + 2 * n_iter`` in all. For
        in-memory AnnData the solver is scikit-learn's and this only sets
        the default ``n_iter``.
    tol : float, optional
        Stop when the leading singular values change by less than this in
        relative terms, rather than always running ``n_iter`` rounds. Guarded
        by Aitken extrapolation of the remaining change, so a slowly
        converging run is not mistaken for a converged one. ``None`` (the
        default) runs the full ``n_iter``. With a tolerance set, ``n_iter``
        becomes the cap rather than the count.
    n_threads : int, default 0
        Threads for the Rust SVD pass engine. ``0`` uses every core this
        process is allowed -- on a cluster, the cores the job was allocated
        (e.g. Slurm ``--cpus-per-task``), not the node's. More threads than
        allocated cores only makes them take turns.
    use_engine : bool, default True
        Use the Rust pass engine for cytome input, which reads the matrix and
        applies TF-IDF in one fused pass. Set False to force the Python path,
        which is the one to compare against when a result looks wrong.
    compute_on_fly : bool, default True
        Apply the normalisation inline while streaming rather than requiring a
        materialised layer on disk.
    cell_mask : ndarray, optional
        Boolean mask / sorted indices to run SVD on a cell subset (streaming /
        cytome paths only; returns the embedding for the masked cells).
    return_svd : bool, default False
        Cytome input only: by default the embedding is written to the cytome and
        ``None`` is returned. Pass ``return_svd=True`` to also get the in-memory
        ``(embeddings, S, Vt)`` tuple back.

    Returns
    -------
    AnnData, tuple, or None
        AnnData input: returns the AnnData with ``obsm[key_added]`` set (or the
        SVD tuple for the streaming-array path). Cytome input: writes the
        embedding to the cytome and returns **None** (self-contained); returns
        the ``(emb, S, Vt)`` tuple when ``cell_mask`` is set or
        ``return_svd=True``.

    Other Parameters
    ----------------
    source, adata
        Deprecated aliases for ``data``, kept so existing notebooks keep
        running; either emits a ``FutureWarning``. They show in the signature
        as ``<unset>``, which means "not supplied" -- passing one along with
        ``data`` is an error. Use ``data``.

    Notes
    -----
    On a cytome the cost is passes over the matrix, so the solver choice is
    the lever. Power iteration with ``n_iter=k`` costs ``2k + 3`` passes;
    ``method='auto'`` (the default) stops on a tolerance instead and typically
    converges in 8-12 rounds on ATAC peak matrices. The engine uses every
    allocated core; ``PIASO_SVD_PROFILE=1`` in the environment prints, per
    pass, how many threads ran and how the time split between reading the
    store and computing.

    Every pass re-reads the store. On a network file system that costs
    minutes where a node-local disk costs seconds -- the same twenty-pass
    SVD took 5 min 13 s from a network mount and 29 s from the node's own
    disk -- so on a cluster copy the cytome to local scratch (``$TMPDIR``)
    first and open the copy. The engine reads without file locks, which
    removes the worst of it, but not the data transfer itself.
    """
    source = _resolve_data_arg(data, 'runSVD', source=source, adata=adata)
    # `n_iter` means different things to the two solvers: a round count for
    # power iteration, a ceiling for a tolerance-driven one. Left unset it is
    # 7 rounds for power and "let the tolerance decide" for auto; given a
    # value it is honoured by both. Before this, method='auto' silently
    # replaced an explicit n_iter with 30.
    if n_iter is _UNSET:
        n_iter = None if method == "auto" else 7
    is_cytome = isinstance(source, str) or _is_cytome_dataset(source)
    _tfidf_inferred = auto_tfidf is None
    if auto_tfidf is None:
        # On by default exactly where TF-IDF is the only sensible input: raw
        # peak or tile counts on a cytome with no statistics supplied.
        auto_tfidf = bool(
            is_cytome and tfidf_params is None
            and modality in _LEGACY_FALLBACK_MODALITIES
            and (measurement or "counts") == "counts"
            and layer in (None, "counts")
        )

    if is_cytome or streaming:
        return _runSVD_streaming(
            source, use_highly_variable=use_highly_variable,
            n_components=n_components, n_iter=n_iter,
            random_state=random_state, scale_data=scale_data,
            oversampling=oversampling, batch_size=batch_size,
            key_added=key_added, layer=layer,
            measurement=measurement, modality=modality,
            verbosity=verbosity, cache_chunks=cache_chunks,
            tol=tol, method=method, n_threads=n_threads, use_engine=use_engine,
            tfidf_params=tfidf_params,
            infog_params=infog_params,
            hvg_indices_override=hvg_indices_override,
            selected_feature_col_name=selected_feature_col_name,
            auto_tfidf=auto_tfidf,
            _auto_tfidf_inferred=_tfidf_inferred,
            cell_mask=cell_mask,
            return_svd=return_svd,
            compute_on_fly=compute_on_fly,
        )
    else:
        if auto_tfidf:
            raise ValueError(
                "auto_tfidf=True is cytome-only; got an in-memory AnnData. "
                "Either pass a cytome path / Dataset, or call "
                "piaso.tl.run_TFIDF on the AnnData first."
            )
        if cell_mask is not None:
            raise NotImplementedError(
                "cell_mask= is only supported by runSVD's streaming "
                "(cytome) and in-memory streaming paths. For in-memory "
                "AnnData, slice the AnnData first: runSVD(adata[mask])."
            )
        return _runSVD_original(
            source, use_highly_variable=use_highly_variable,
            n_components=n_components, random_state=random_state,
            scale_data=scale_data, n_iter=7 if n_iter is None else n_iter,
            key_added=key_added, layer=layer, verbosity=verbosity,
            selected_feature_col_name=selected_feature_col_name,
            modality=modality,
        )


# ============================================================================
# Original runSVDLazy — COMPLETELY UNCHANGED body
# ============================================================================
import pandas as pd
from typing import Iterable, Union

def _runSVDLazy_original(
    adata,
    copy: bool = False,
    n_components: int = 50,
    use_highly_variable: bool = True,
    n_top_genes: int = 3000,
    verbosity: int = 0,
    batch_key: Optional[str] = None,
    random_state: Optional[int] = 1927,
    scale_data: bool = False,
    n_iter: int = 7,
    infog_trim: bool = True,
    key_added: str = 'X_svd',
    layer: Optional[str] = 'infog',
    infog_layer: Optional[str] = None,
    allow_non_integer: bool = False,
):
    """
    Performs Truncated Singular Value Decomposition (SVD) in a "lazy" mode, based on the `piaso.tl.runSVD` function.

    Compared to `piaso.tl.runSVD`, this function includes the step of highly variable gene section. If `layer` is set to `infog`,
    both the highly variable genes and normalized gene expression values were taken from the INFOG normalization outputs.

    This function performs on the specified gene expression matrix (adata.X or a specified layer) within an AnnData object
    and stores the resulting low-dimensional representation in `adata.obsm`.

    Parameters
    ----------
    adata : AnnData
        An AnnData object.
    copy : bool, optional, default=False
        If True, returns a copy of `adata` with the computed embeddings instead of modifying in place.
    n_components : int, optional, default=50
        The number of singular value decomposition (SVD) components to retain.
    use_highly_variable : bool, optional, default=True
        If True, uses only highly variable genes for the decomposition.
    n_top_genes : int, optional, default=3000
        The number of top highly variable genes to retain before performing SVD.
    verbosity : int, optional, default=0
        Controls the verbosity of logging messages.
    batch_key : str, optional, default=None
        Specifies the key in `adata.obs` containing batch labels for highly variable gene selection.
    random_state : int, optional, default=1927
        A random seed to ensure reproducibility.
    scale_data : bool, optional, default=False
        If True, standardizes the input data before performing SVD.
    n_iter : int, optional, default=7
        Number of iterations for randomized SVD solver. The default is larger than the default in randomized_svd to handle sparse matrices that may have large slowly decaying spectrum. Also larger than the `n_iter` default value (5) in the TruncatedSVD function.
    infog_trim : bool, optional, default=True
        Used for the `trim` parameter in `piaso.tl.infog` function, effective only when `layer` set to `infog`.
    key_added : str, optional, default='X_svd'
        The key under which the resulting cell embeddings are stored in `adata.obsm`.
    layer : str, optional, default='infog'
        ``'infog'`` (the default) normalises with INFOG first, reading the raw
        counts from ``infog_layer`` (``None``: ``adata.X``), and selects genes
        as INFOG does. Any other value names a layer to factorise as it is,
        with the object's ``highly_variable`` column if it has one and every
        gene otherwise; ``None`` means ``adata.X`` as it is.
    infog_layer : str, optional, default=None
        The layer holding raw counts for INFOG (``None``: ``adata.X``). Used
        only when ``layer='infog'``.

    Returns
    -------
    If `copy` is True, returns a modified AnnData object. Otherwise, modifies `adata` in place.

    Example
    -------
    >>> import piaso
    >>> adata = piaso.tl.runSVDLazy(
    ...     adata, n_components=50, n_top_genes=3000,
    ...     use_highly_variable=True, key_added="X_svd",
    ...     layer=None
    ... )
    >>>
    >>> # Access the cell embedding
    >>> adata.obsm['X_svd']
    """

    adata = adata.copy() if copy else adata

    if layer=='infog':
        ### Run INFOG normalization
        infog(
            adata,
            copy=False,
            layer=infog_layer,
            n_top_genes=n_top_genes,
            key_added='infog',
            trim=infog_trim,
            verbosity=verbosity,
            allow_non_integer=allow_non_integer,
        )

    elif use_highly_variable and 'highly_variable' not in adata.var.columns:
        # Another layer, no selection on the object: use every gene rather
        # than reach for scanpy, which this package does not require.
        if verbosity > 0:
            print(f"infog_svd: layer={layer!r} and no 'highly_variable' column in var; "
                  f"using every gene. Run infog or selectFeatures first to select.")
        use_highly_variable = False


    ### Use the runSVD function
    _runSVD_original(
        adata,
        use_highly_variable=use_highly_variable,
        n_components=n_components,
        random_state=random_state,
        scale_data=scale_data,
        n_iter=n_iter,
        key_added=key_added,
        layer=layer,
        verbosity=verbosity,
    )

    ### Return the result
    return adata if copy else None


# ============================================================================
# Streaming runSVDLazy — INFOG → HVG compact → SVD
# ============================================================================
def _infog_chunk_normalizer(ds, modality, layer, compute_on_fly, verbosity=0):
    """Return a per-chunk normalizer for ``layer``, or None to read it directly.

    Precedence, matching COSG's ``_resolve_layer_to_read``:

    1. a materialised ``{modality}_{layer}`` matrix, read as-is (storage beats
       recomputation, and it is what a reproducibility check wants);
    2. the stored parameters, applied per chunk from raw counts;
    3. otherwise raise, naming both routes.

    Returns ``(read_layer, normalizer_or_None)``.
    """
    from ._normalization import _normalize_chunk_infog
    from ._normalize_resolve import ensure_infog_params

    matrices = {r[0] for r in
                ds._conn.execute("SELECT matrix_name FROM matrix_meta")}
    materialised = f"{modality}_{layer}" in matrices
    params = ds.metadata.get(f"{modality}_{layer}_params")

    if materialised:
        # Both present: the layer was written from some run of infog(), the
        # params are from the latest one. If they disagree the layer is stale,
        # and silently preferring it would normalise with parameters the user
        # did not ask for.
        if params is not None:
            n_cells_params = len(params.get("cell_depth", ()))
            if n_cells_params and n_cells_params != ds.n_cells:
                warnings.warn(
                    f"{modality}_{layer} is materialised but the stored "
                    f"{layer} params describe {n_cells_params} cells and this "
                    f"cytome has {ds.n_cells}. Reading the materialised layer; "
                    f"pass compute_on_fly=True after re-running infog() if "
                    f"that is not what you want.", stacklevel=3)
            # NB: do not compare n_cols to n_top_genes. INFOG normalises every
            # gene, so the layer has n_genes columns while n_top_genes counts
            # highly variable genes; they are unrelated numbers.
        return layer, None

    if not compute_on_fly:
        raise KeyError(
            f"no {modality}_{layer} matrix in this cytome, and "
            f"compute_on_fly=False. Either run "
            f"piaso.tl.infog(ds, save_layer=True) to materialise it, or pass "
            f"compute_on_fly=True to normalise from the stored parameters.")

    if params is None:
        raise KeyError(
            f"no {modality}_{layer} matrix and no {modality}_{layer}_params "
            f"in this cytome. Run piaso.tl.infog(ds) first (it records the "
            f"parameters), or piaso.tl.infog(ds, save_layer=True) to write the "
            f"layer as well.")

    if verbosity > 0:
        print(f"  {layer}: no materialised layer; normalising on the fly from "
              f"{modality}_counts and the stored parameters")
    cd = np.asarray(params["cell_depth"], dtype=np.float64)
    ig = np.asarray(params["inv_gene_depth"], dtype=np.float64)
    scale = float(params["scale"])
    counts_sum = float(params["counts_sum"])
    thr = params.get("threshold")
    return "counts", (lambda chunk, indices:
                      _normalize_chunk_infog(chunk, cd[indices], ig, scale,
                                             counts_sum, thr))


def _write_hvg_compact_measurement(source, hvg_indices, infog_params=None,
                                    source_layer='counts',
                                    target_measurement='infog_hvg',
                                    batch_size=1024, dtype=None):
    """Write a compact HVG-only measurement to cytome (one pass, chunked write).

    If infog_params is provided, reads from RAW measurement and normalizes
    on-the-fly (V5 lazy INFOG). Otherwise falls back to reading from
    source_layer directly (V4 behavior).
    """
    ds = _open_cytome(source) if isinstance(source, str) else source

    n_hvg = len(hvg_indices)
    matrix_name = f"RNA_{target_measurement}"
    _n_alloc, n_true = _safe_n_cells(ds, "RNA", source_layer)
    writer = ds.create_layer_writer(
        matrix_name, n_rows=n_true, n_cols=n_hvg,
        dtype=_resolve_layer_dtype(dtype),
    )

    for chunk, indices in ds.iter_chunks(modality="RNA", layer=source_layer, batch_size=batch_size):
        if infog_params is not None:
            # V5: normalize on-the-fly from raw counts
            chunk = _normalize_chunk_infog(
                chunk,
                infog_params['cell_depth'][indices],
                infog_params['inv_gene_depth'],
                infog_params['scale'],
                infog_params['counts_sum'],
                infog_params['threshold'],
            )
        hvg_chunk = chunk[:, hvg_indices]
        writer.write_chunk(hvg_chunk, row_offset=int(indices[0]))

    writer.finalize()
    ds._conn.commit()


def _runSVDLazy_streaming(
    source,
    copy: bool = False,
    n_components: int = 50,
    use_highly_variable: bool = True,
    n_top_genes: int = 3000,
    verbosity: int = 0,
    batch_key: Optional[str] = None,
    random_state: Optional[int] = 1927,
    scale_data: bool = False,
    n_iter: int = 7,
    infog_trim: bool = True,
    key_added: str = 'X_svd',
    layer: Optional[str] = 'infog',
    infog_layer: Optional[str] = None,
    batch_size: int = 1024,
    allow_non_integer: bool = False,
):
    """
    Streaming runSVDLazy: INFOG → HVG compact → SVD.

    When source is cytome:
        Total passes: 2 (INFOG) + 1 (write HVG compact) + 17 (SVD) = 20
        Peak RAM: ~110 MB at 185k cells

    When source is AnnData (streaming=True):
        Total passes: 2 (INFOG) + 17 (SVD on HVG subset) = 19
        Peak RAM: ~110 MB + adata size
    """
    import time

    is_adata = isinstance(source, AnnData)
    is_cytome = isinstance(source, str) or _is_cytome_dataset(source)

    if is_adata:
        adata = source.copy() if copy else source

    t0 = time.time()

    # Step 1: INFOG (streaming)
    if verbosity > 0:
        print("Step 1: INFOG normalization (streaming, 2 passes)...")
    infog_result = infog(
        source if is_cytome else adata,
        n_top_genes=n_top_genes, trim=infog_trim,
        batch_size=batch_size, key_added='infog',
        layer=infog_layer, verbosity=verbosity, streaming=True,
        allow_non_integer=allow_non_integer,
    )

    t_infog = time.time() - t0

    # Get HVG info
    if is_adata:
        hvg_indices = np.where(adata.var['highly_variable'].values)[0]
    else:
        hvg_indices = infog_result['hvg_indices']

    # Step 2: For cytome — write compact HVG measurement (1 pass)
    # V5: reads RAW counts and normalizes on-the-fly (no INFOG layer on disk)
    t1 = time.time()
    if is_cytome:
        if verbosity > 0:
            print(f"Step 2: Writing compact HVG measurement ({len(hvg_indices)} genes, 1 pass)...")
        _write_hvg_compact_measurement(
            source, hvg_indices,
            infog_params=infog_result.get('infog_params'),
            source_layer='counts', target_measurement='infog_hvg',
            batch_size=batch_size,
        )
    t_hvg_compact = time.time() - t1

    # Step 3: SVD (streaming)
    t2 = time.time()
    total_passes = 3 + 2 * n_iter
    if verbosity > 0:
        print(f"Step 3: Streaming SVD ({n_components} components, {n_iter} iterations, {total_passes} passes)...")

    measurement = 'infog_hvg' if is_cytome else None
    svd_layer = 'infog'

    # When using the compact HVG measurement (cytome), the data is already
    # subset to HVG columns — don't apply HVG indices again (double-subsetting).
    _svd_use_hvg = use_highly_variable if not is_cytome else False

    embeddings, S, Vt = _runSVD_streaming(
        source if is_cytome else adata,
        n_components=n_components, n_iter=n_iter,
        random_state=random_state, scale_data=scale_data,
        batch_size=batch_size, use_highly_variable=_svd_use_hvg,
        layer=svd_layer, measurement=measurement,
        key_added=key_added, verbosity=verbosity,
        return_svd=True,   # this caller unpacks the tuple
    )
    t_svd = time.time() - t2

    t_total = time.time() - t0
    if verbosity > 0:
        print(f"\nStreaming runSVDLazy complete in {t_total:.1f}s "
              f"(INFOG: {t_infog:.1f}s, HVG compact: {t_hvg_compact:.1f}s, SVD: {t_svd:.1f}s)")

    if is_adata:
        return adata if copy else None
    return embeddings


# ============================================================================
# Public infog_svd dispatcher (renamed from runSVDLazy)
# ============================================================================
def infog_svd(
    source,
    copy: bool = False,
    n_components: int = 50,
    use_highly_variable: bool = True,
    n_top_genes: int = 3000,
    verbosity: int = 0,
    batch_key: Optional[str] = None,
    random_state: Optional[int] = 1927,
    scale_data: bool = False,
    n_iter: int = 7,
    infog_trim: bool = True,
    key_added: str = 'X_svd',
    layer: Optional[str] = 'infog',
    infog_layer: Optional[str] = None,
    # NEW parameters:
    streaming: bool = False,
    batch_size: int = 1024,
    allow_non_integer: bool = False,
):
    """
    INFOG normalization → HVG selection → SVD in one call.

    Performs INFOG normalization, selects highly variable genes, and runs
    truncated SVD for dimensionality reduction in a single function call.

    Supports three modes:
    - infog_svd(adata) — standard in-memory (existing behavior)
    - infog_svd(adata, streaming=True) — streaming from in-memory AnnData
    - infog_svd("path.cytome") — streaming from on-disk cytome dataset
    - infog_svd(cytome_dataset) — streaming from already-opened cytome object
    allow_non_integer : bool, default False
        By default INFOG refuses input whose values are not integers, because
        its dispersion model is defined on raw UMI counts and silently returns
        meaningless numbers on normalized, log-transformed or scaled data. Set
        True to run anyway -- appropriate for Smart-seq2 TPM/FPKM, imputed or
        already-corrected matrices. Ignored when ``layer``/``infog_layer`` is
        given, since naming a layer already answers the question.

    Parameters
    ----------
    source : AnnData, CytomeDataset or str
        The data: an AnnData, an open cytome, or a path to one.
    copy : bool, default False
        AnnData: return a copy with the embedding instead of editing in place.
    n_components : int, default 50
        SVD dimensions.
    use_highly_variable : bool, default True
        Restrict the SVD to the selected genes.
    n_top_genes : int, default 3000
        Genes INFOG selects.
    verbosity : int, default 0
        ``1`` reports each step.
    batch_key : str, optional
        Kept for API compatibility; the selection no longer depends on it.
    random_state : int, default 1927
        Seed for the randomised SVD.
    scale_data : bool, default False
        Scale features before the SVD (AnnData path only).
    n_iter : int, default 7
        Rounds of the randomised solver.
    infog_trim : bool, default True
        Passed to INFOG as ``trim``.
    key_added : str, default 'X_svd'
        Name of the embedding.
    layer : str, default 'infog'
        ``'infog'`` (the default) normalises with INFOG first, reading the raw
        counts from ``infog_layer`` (``None``: ``adata.X``), and selects genes
        as INFOG does. Any other value names a layer to factorise as it is,
        with the object's ``highly_variable`` column if it has one and every
        gene otherwise; ``None`` means ``adata.X`` as it is.
    infog_layer : str, optional
        The layer holding raw counts for INFOG (``None``: ``adata.X``). Used
        only when ``layer='infog'``.
    streaming : bool, default False
        AnnData: use the chunked streaming path; a cytome always streams.
    batch_size : int, default 1024
        Rows per chunk on the streaming path.
    allow_non_integer : bool, default False
        Let INFOG accept a matrix that does not hold integer counts.
    """
    is_cytome = isinstance(source, str) or _is_cytome_dataset(source)

    if is_cytome or streaming:
        return _runSVDLazy_streaming(
            source, copy=copy, n_components=n_components,
            use_highly_variable=use_highly_variable,
            n_top_genes=n_top_genes, verbosity=verbosity,
            batch_key=batch_key, random_state=random_state,
            scale_data=scale_data, n_iter=n_iter,
            infog_trim=infog_trim, key_added=key_added,
            layer=layer, infog_layer=infog_layer,
            batch_size=batch_size, allow_non_integer=allow_non_integer,
        )
    else:
        return _runSVDLazy_original(
            source, copy=copy, n_components=n_components,
            use_highly_variable=use_highly_variable,
            n_top_genes=n_top_genes, verbosity=verbosity,
            batch_key=batch_key, random_state=random_state,
            scale_data=scale_data, n_iter=n_iter,
            infog_trim=infog_trim, key_added=key_added,
            layer=layer, infog_layer=infog_layer,
            allow_non_integer=allow_non_integer,
        )


import warnings

def runSVDLazy(*args, **kwargs):
    """Deprecated alias for :func:`infog_svd`. Use ``piaso.tl.infog_svd()`` instead."""
    warnings.warn(
        "runSVDLazy is deprecated, use piaso.tl.infog_svd() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return infog_svd(*args, **kwargs)
