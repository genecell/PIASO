import warnings

import pandas as pd
import numpy as np

from typing import Iterable, Union, Optional

from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome_dataset
from ..utils._cytome_compat import open_cytome_sync as _open_cytome
from ._compat import resolve_data_arg as _resolve_data_arg, _UNSET


def _safe_n_cells(ds, modality, layer):
    """Return safe row count for pre-allocating arrays indexed by iter_chunks.

    Compares ``ds.n_cells`` (cells table count) with ``matrix_meta.n_rows``.
    If they differ — e.g. after a SQL DELETE without matrix rebuild — warns
    and returns the larger value so that ``array[indices] = ...`` cannot
    produce an IndexError.

    Returns ``(n_alloc, n_cells)`` where *n_alloc* is the safe allocation
    size and *n_cells* is the true cell count (for output truncation).
    """
    n_cells = ds.n_cells
    matrix_name = f"{modality}_{layer}"
    meta = ds._conn.execute(
        "SELECT n_rows FROM matrix_meta WHERE matrix_name = ?",
        (matrix_name,),
    ).fetchone()
    if meta is not None:
        n_rows = int(meta[0])
        if n_rows != n_cells:
            import warnings
            warnings.warn(
                f"Cytome consistency: cells table has {n_cells} rows but "
                f"matrix '{matrix_name}' has {n_rows} rows. "
                f"Run ds.repair() to fix. "
                f"Allocating {max(n_cells, n_rows)} rows defensively.",
                stacklevel=3,
            )
            return max(n_cells, n_rows), n_cells
    return n_cells, n_cells


def _resolve_gene_names(ds, modality="RNA"):
    """Resolve the display gene-name array for a cytome.

    Round 25: delegate to the SINGLE SOURCE OF TRUTH
    (``cytome.utils.modality.modality_feature_table_info``) so the names returned
    here match the names COSG emits (populated symbol-first, else ``gene_id``).
    Before this, a hard-coded ``gene_id``-first order returned Ensembl ids while
    COSG (post-Round-24) returned symbols — the two never matched and runGDR/score
    raised "No valid gene sets found" on cellranger cytomes (gene_id != symbol).
    """
    try:
        from cytome.utils.modality import modality_feature_table_info
        feat_tbl, idx_col, name_col = modality_feature_table_info(ds, modality)
        rows = ds._conn.execute(
            f'SELECT "{name_col}" FROM {feat_tbl} ORDER BY {idx_col}'
        ).fetchall()
        vals = np.array([r[0] for r in rows], dtype=object)
        if len(vals) and vals[0] is not None:
            return vals
    except Exception:
        pass
    # Legacy fallback (RNA genes table only).
    gene_cols = ds.genes.columns
    for col in ["gene_id", "gene_name", "symbol"]:
        if col in gene_cols:
            vals = np.array(ds.genes[col])
            if vals[0] is not None:
                return vals
    raise ValueError(f"Cannot find gene name column with non-null values. Available: {gene_cols}")


def _gene_name_alias_map(ds, modality, var_names):
    """Build ``name -> [row indices]`` including aliases from every candidate id/name
    column, so a gene set in EITHER vocabulary resolves (COSG markers are symbols;
    user-supplied lists may be Ensembl ids). A name that maps to **several** rows
    (duplicate gene symbols — common in cellranger output where ``symbol`` is not
    de-duplicated) resolves to **all** of them (union scoring); a one-time warning is
    emitted so the ambiguity is visible (Round 26 — keep raw symbols + union-resolve).
    """
    name_to_idx: dict = {}
    for idx, name in enumerate(var_names):
        name_to_idx.setdefault(name, []).append(idx)
    dups = sorted({n for n, v in name_to_idx.items() if len(v) > 1 and n})
    if dups:
        import warnings as _w
        _w.warn(
            f"score/runGDR: {len(dups)} gene symbol(s) are duplicated in modality "
            f"'{modality}' (e.g. {dups[:5]}); each is scored over ALL its features. "
            f"Use the unique id column to disambiguate.", UserWarning, stacklevel=3)
    try:
        from cytome.utils.modality import _registry_entry
        _, feat_tbl, idx_col, candidate_cols = _registry_entry(modality)
        feat_cols = [c[1] for c in ds._conn.execute(
            f"PRAGMA table_info({feat_tbl})").fetchall()]
        for c in (col for col in candidate_cols if col in feat_cols):
            for ridx, val in ds._conn.execute(
                f'SELECT {idx_col}, "{c}" FROM {feat_tbl} ORDER BY {idx_col}'
            ).fetchall():
                if val is not None and val != "" and val not in name_to_idx:
                    name_to_idx[val] = [ridx]
    except Exception:
        pass
    return name_to_idx




# ============================================================================
# Original infog — COMPLETELY UNCHANGED body from v1.1.0
# ============================================================================
def _infog_original(
    adata,
    copy: bool = False,
    inplace: bool = False,
    n_top_genes: int = 3000,
    key_added: str = 'infog',
    key_added_highly_variable_gene: str = 'highly_variable',
    trim: bool = True,
    verbosity: int = 1,
    layer: Optional[str] = None
):
    """
    Performs INFOG normalization of single-cell RNA sequencing data based on "biological information".

    This function outputs the selected highly variable genes and normalized gene expression values based on the raw UMI counts.

    Parameters
    ----------
    adata : AnnData
        An AnnData object.
    copy : bool, optional, default=False
        If True, returns a new AnnData object with the normalized data instead of modifying `adata` in place.
    inplace : bool, optional, default=False
        If True, the normalized data is stored in `adata.X` rather than in `adata.layers[key_added]`.
    n_top_genes : int, optional, default=3000
        The number of top highly variable genes to select.
    key_added : str, optional, default='infog'
        The key under which the normalized gene expression matrix is stored in `adata.layers`.
    key_added_highly_variable_gene : str, optional, default='highly_variable'
        The key under which the selection of highly variable genes is stored in `adata.var`.
    trim : bool, optional, default=True
        If True, trim the normalized gene expression values.
    verbosity : int, optional, default=1
        Controls the level of logging and output messages.
    layer : str, optional, default=None
        Specifies which layer of `adata` to use for INFOG normalization. If None, `adata.X` is used. Note: the raw UMIs counts should be used.

    Returns
    -------
    If `copy` is True, returns a modified AnnData object with the normalized expression matrix. 
    Otherwise, modifies `adata` in place.
    The normalized gene expression values will be saved in `adata.X` if `inplace` is True, or in `adata.layers`
    with the key `key_added` by default if `inplace` is False.

    Example
    -------
    >>> import piaso
    >>> adata = piaso.tl.infog(
    ...     adata, n_top_genes=3000, key_added="infog",
    ...     trim=True, layer="raw"
    ... )
    >>> 
    >>> # Access the normalized data
    >>> adata.layers['infog']
    >>> # Access the highly variable genes
    >>> adata.var['highly_variable']
    """
    
    if layer and layer not in adata.layers:
        raise ValueError(f"{layer} not found in adata.layers.")
    
    adata = adata.copy() if copy else adata
    
    ### To get the gene expression matrix
    counts = adata.layers[layer] if layer else adata.X
    
    ### Ensure counts is in csr sparse format
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)

    ### Raise an error if any negative values are found in counts
    if counts.data.size > 0 and counts.data.min() < 0:
        raise ValueError("Input counts contain negative values, which is not allowed.")

    ### Compute cell and gene depths
    cell_depth = np.array(counts.sum(axis=1)).ravel()
    gene_depth = np.array(counts.sum(axis=0)).ravel()
    
    
    
    counts_sum = counts.sum()
    scale = np.median(cell_depth)
    
    
    ### should use this one, especially for downsampling experiment, only this one works, the sequencing baises are corrected, partially because only this transformation is linear
    ### Instead of using sparse.diags, use element-wise multiplication with broadcasting.
    # Fused INFOG formula — eliminates intermediate sparse matrices for lower RAM.
    # Derivation: result[i,j] = sqrt(counts[i,j] * (scale/cd[i]) * counts[i,j] * (cs/cd[i]) * (1/gd[j]))
    #           = counts[i,j] * sqrt(scale * cs / gd[j]) / cd[i]
    # This fuses 3 sparse operations into 1 copy + in-place modify (peak ~2x vs ~4x).
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_gene_depth = 1.0 / gene_depth
    inv_gene_depth[~np.isfinite(inv_gene_depth)] = 0.0

    # Precompute per-gene and per-cell factors (1D arrays, negligible memory)
    per_gene = np.sqrt(scale * counts_sum * inv_gene_depth)   # (n_genes,)
    per_cell = 1.0 / cell_depth                               # (n_cells,)
    per_cell[~np.isfinite(per_cell)] = 0.0

    # Single sparse copy, then in-place scaling via CSR structure
    normalized = counts.copy().tocsr()
    # Vectorized in-place: scale each nonzero by per_cell[row] * per_gene[col]
    row_factor = np.repeat(per_cell, np.diff(normalized.indptr))
    normalized.data = normalized.data.astype(np.float64)
    normalized.data *= row_factor * per_gene[normalized.indices]


    
    if trim:
        threshold = np.sqrt(counts.shape[0])
        normalized.data[normalized.data > threshold] = threshold
        
    

    ### Calculate the variance BEFORE storing (avoids in-place squaring corrupting the layer)
    mean = np.array(normalized.mean(axis=0)).ravel()
    # Compute mean-of-squares without modifying normalized in-place
    sq_data = normalized.data ** 2
    sq = sparse.csr_matrix((sq_data, normalized.indices, normalized.indptr),
                           shape=normalized.shape)
    mean_sq = np.array(sq.mean(axis=0)).ravel()
    del sq, sq_data

    residual_var_orig_b = mean_sq - mean**2

    # Save the normalized data (no extra copy — normalized is already standalone)
    if inplace:
        adata.X = normalized
    else:
        adata.layers[key_added] = normalized
    adata.var[key_added + '_var'] = residual_var_orig_b
    
    
    ### Feature selection    
    pos_gene=_select_top_n(adata.var[key_added+'_var'],n_top_genes)
    tmp=np.repeat(False,adata.n_vars)
    tmp[pos_gene]=True
    ### Change 'highly_variable_'+key_added to 'highly_variable', let's use it by default
    adata.var[key_added_highly_variable_gene]=tmp
    
    if verbosity > 0:
        if inplace:
            print(f'The normalized data is saved in `adata.X`.')
        else:
            print(f'The normalized data is saved as `{key_added}` in `adata.layers`.')
        print(f'The highly variable genes are saved as `{key_added_highly_variable_gene}` in `adata.var`.')
        print('Finished INFOG normalization.')
         
    ### Return the result
    return adata if copy else None


# ============================================================================
# Per-modality params cache — read with legacy fall-through
# ============================================================================
def _read_modality_params(ds, key_base: str, modality: str):
    """Read per-modality cached params from ``ds.metadata``.

    Tries ``f'{modality}_{key_base}'`` first (the new namespaced key),
    falls back to the unprefixed legacy key (e.g. ``infog_params``) with
    a one-time DeprecationWarning. Returns ``None`` if neither key
    exists.

    The two-key write at the producer side (``piaso.tl.infog`` etc.)
    keeps the legacy alias populated for backward compat; this helper
    is the canonical reader.
    """
    new_key = f"{modality}_{key_base}"
    new_val = ds.metadata.get(new_key)
    if new_val is not None:
        return new_val
    legacy = ds.metadata.get(key_base)
    if legacy is not None:
        import warnings as _warnings
        _warnings.warn(
            f"cytome metadata uses legacy key '{key_base}' for {modality} — "
            f"using as '{new_key}'. To refresh, recompute with "
            f"piaso.tl.infog(ds, modality='{modality}') or analogous tool.",
            DeprecationWarning, stacklevel=2,
        )
        return legacy
    return None


# ============================================================================
# INFOG chunk normalization helper (pure function)
# ============================================================================
def _normalize_chunk_infog(chunk, cell_depth_chunk, inv_gene_depth, scale,
                           counts_sum, threshold):
    """Apply INFOG normalization to a sparse chunk. Pure function, no side effects.

    Parameters
    ----------
    chunk : scipy.sparse.csr_matrix (batch_size x n_genes)
    cell_depth_chunk : np.ndarray (batch_size,) — per-cell total counts for these rows
    inv_gene_depth : np.ndarray (n_genes,) — 1/gene_depth (0 where gene_depth==0)
    scale : float — median(cell_depth)
    counts_sum : float — total counts
    threshold : float or None — sqrt(n_cells) for trimming

    Returns
    -------
    normalized : csr_matrix — same shape, same sparsity pattern
    """
    cd = cell_depth_chunk[:, None]
    normalized = chunk.multiply(scale / cd)
    info_factor = chunk.multiply(counts_sum / cd).multiply(inv_gene_depth)
    normalized = normalized.multiply(info_factor)
    normalized.data = np.sqrt(normalized.data)
    if threshold is not None:
        normalized.data[normalized.data > threshold] = threshold
    return normalized


# ============================================================================
# Streaming INFOG — works with cytome on-disk and AnnData in-memory chunks
# ============================================================================
def _get_infog_chunk_iterator(source, batch_size, layer=None, modality="RNA"):
    """Return (n_cells, n_genes, chunk_iterator_factory).

    chunk_iterator_factory() yields (csr_chunk, row_indices) pairs.

    ``n_genes`` is read from the matrix_meta row for ``{modality}_{layer}``
    so non-RNA modalities (GA / ATAC / tiles) get the correct feature
    count (cytome's ``ds.n_genes`` is hardcoded to the RNA gene table).
    """
    def _n_features_for_matrix(_ds, _modality, _layer):
        matrix_name = f"{_modality}_{_layer}"
        row = _ds._conn.execute(
            "SELECT n_cols FROM matrix_meta WHERE matrix_name = ?",
            (matrix_name,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"piaso.tl.infog: matrix {matrix_name!r} not found in cytome. "
                f"Available matrices: "
                f"{[r[0] for r in _ds._conn.execute('SELECT matrix_name FROM matrix_meta').fetchall()]}"
            )
        return int(row[0])

    if isinstance(source, str):
        ds = _open_cytome(source)
        _layer = layer or "counts"
        n_alloc, _n_true = _safe_n_cells(ds, modality, _layer)
        n_genes = _n_features_for_matrix(ds, modality, _layer)
        def factory():
            return ds.iter_chunks(modality=modality, layer=_layer, batch_size=batch_size)
        return n_alloc, n_genes, factory, ds
    elif _is_cytome_dataset(source):
        _layer = layer or "counts"
        n_alloc, _n_true = _safe_n_cells(source, modality, _layer)
        n_genes = _n_features_for_matrix(source, modality, _layer)
        def factory():
            return source.iter_chunks(modality=modality, layer=_layer, batch_size=batch_size)
        return n_alloc, n_genes, factory, source
    else:
        # AnnData
        from anndata import AnnData
        X = source.layers[layer] if layer else source.X
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        n_cells, n_genes = X.shape
        def factory():
            for i in range(0, n_cells, batch_size):
                end = min(i + batch_size, n_cells)
                chunk = X[i:end]
                if not sparse.issparse(chunk):
                    chunk = sparse.csr_matrix(chunk)
                yield chunk, np.arange(i, end)
        return n_cells, n_genes, factory, None


def _infog_streaming(
    source,
    n_top_genes: int = 3000,
    trim: bool = True,
    batch_size: int = 1024,
    key_added: str = 'infog',
    key_added_highly_variable_gene: str = 'highly_variable',
    layer: Optional[str] = None,
    verbosity: int = 0,
    copy: bool = False,
    inplace: bool = False,
    save_layer: bool = False,
    modality: str = "RNA",
):
    """
    Streaming INFOG normalization. Works with cytome (on-disk) and AnnData (in-memory chunks).

    Two-pass algorithm:
      Pass 1: Compute cell_depth, gene_depth, counts_sum, scale
      Pass 2: Normalize, accumulate variance, write output

    Returns
    -------
    dict with keys: 'gene_var', 'hvg_mask', 'hvg_indices', 'scale', 'counts_sum',
                    'cell_depth', 'n_cells', 'n_genes'
    Side effects: writes normalized data to source (adata.layers or cytome layer).
    """
    from anndata import AnnData

    is_adata = isinstance(source, AnnData)
    is_cytome = isinstance(source, str) or _is_cytome_dataset(source)

    if is_adata:
        adata = source.copy() if copy else source

    n_cells, n_genes, chunk_factory, ds_obj = _get_infog_chunk_iterator(
        source, batch_size, layer=layer, modality=modality,
    )

    # ---- Pass 1: Collect global statistics ----
    cell_depth = np.zeros(n_cells, dtype=np.float64)
    gene_depth = np.zeros(n_genes, dtype=np.float64)
    counts_sum = 0.0

    for chunk, indices in chunk_factory():
        cell_depth[indices] = np.array(chunk.sum(axis=1)).ravel()
        gene_depth += np.array(chunk.sum(axis=0)).ravel()
        counts_sum += chunk.sum()

    scale = np.median(cell_depth)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_gene_depth = 1.0 / gene_depth
    inv_gene_depth[~np.isfinite(inv_gene_depth)] = 0.0

    threshold = np.sqrt(n_cells) if trim else None

    # ---- Pass 2: Normalize on-the-fly, accumulate variance ----
    gene_sum = np.zeros(n_genes, dtype=np.float64)
    gene_sum_sq = np.zeros(n_genes, dtype=np.float64)

    if is_adata:
        output_chunks = []

    if is_cytome:
        ds = ds_obj if ds_obj is not None else _open_cytome(source)

    for chunk, indices in chunk_factory():
        normalized = _normalize_chunk_infog(
            chunk, cell_depth[indices], inv_gene_depth,
            scale, counts_sum, threshold,
        )

        # Accumulate for per-gene variance
        gene_sum += np.array(normalized.sum(axis=0)).ravel()
        sq = normalized.copy()
        sq.data **= 2
        gene_sum_sq += np.array(sq.sum(axis=0)).ravel()

        # For AnnData: collect output chunks (in-memory, fast)
        if is_adata:
            output_chunks.append(normalized)
        # For cytome: NO write — INFOG layer is NOT materialized on disk

    # Compute variance and HVG
    gene_mean = gene_sum / n_cells
    gene_var = gene_sum_sq / n_cells - gene_mean ** 2

    # Select top n HVG by variance
    hvg_indices = np.argsort(gene_var)[::-1][:n_top_genes]
    hvg_indices = np.sort(hvg_indices)  # sort for consistent ordering
    hvg_mask = np.zeros(n_genes, dtype=bool)
    hvg_mask[hvg_indices] = True

    # Write output
    if is_adata:
        full_normalized = sparse.vstack(output_chunks, format='csr')
        if inplace:
            adata.X = full_normalized
        else:
            adata.layers[key_added] = full_normalized
        adata.var[key_added + '_var'] = gene_var
        adata.var[key_added_highly_variable_gene] = hvg_mask

        if verbosity > 0:
            if inplace:
                print(f'The normalized data is saved in `adata.X`.')
            else:
                print(f'The normalized data is saved as `{key_added}` in `adata.layers`.')
            print(f'The highly variable genes are saved as `{key_added_highly_variable_gene}` in `adata.var`.')
            print('Finished INFOG normalization (streaming).')

    elif is_cytome:
        # Write HVG boolean + variance to the correct feature table for the
        # modality (RNA→genes, GA→GA_genes, ATAC→peaks, tiles→tiles). Use
        # ds.features(modality), NOT getattr(ds, name): for 'tiles'/'GA' the
        # modality name collides with / differs from the table name, so attribute
        # access returns a Modality (no __setitem__). ds.features() returns the
        # writable EntityTable uniformly.
        _entity = ds.features(modality)
        _entity[key_added_highly_variable_gene] = hvg_mask
        _entity[key_added + '_var'] = gene_var.astype(np.float32)

        # Save INFOG normalization parameters to metadata (for on-the-fly
        # reconstruction). Namespaced by modality so e.g. an RNA-infog run and a
        # GA-infog run can coexist on the same cytome. NOTE: the un-prefixed legacy
        # 'infog_params' alias is no longer written (it was RNA-only but is
        # modality-blind on read); readers use '{modality}_infog_params' and fall
        # back to a residual legacy key only under a feature-count guard.
        _infog_params_payload = {
            'cell_depth': cell_depth,
            'inv_gene_depth': inv_gene_depth,
            'scale': float(scale),
            'counts_sum': float(counts_sum),
            'threshold': float(threshold) if threshold is not None else None,
            'n_top_genes': int(n_top_genes),
        }
        ds.metadata[f'{modality}_infog_params'] = _infog_params_payload

        # Optionally write full INFOG layer to cytome (non-lazy mode)
        if save_layer:
            matrix_name = f"{modality}_{key_added}"
            writer = ds.create_layer_writer(
                matrix_name, n_rows=n_cells, n_cols=n_genes, dtype=np.float64,
            )
            for chunk, indices in chunk_factory():
                normed = _normalize_chunk_infog(
                    chunk, cell_depth[indices], inv_gene_depth,
                    scale, counts_sum, threshold,
                )
                writer.write_chunk(normed, row_offset=int(indices[0]))
            writer.finalize()

        ds.flush()

        if verbosity > 0:
            if save_layer:
                print(f'Full INFOG layer written to cytome as `{key_added}`.')
            else:
                print(f'INFOG params saved to cytome metadata (lazy, no full layer written).')
            print(f'The highly variable genes are saved as `{key_added_highly_variable_gene}` in gene metadata.')
            print('Finished INFOG normalization (streaming, cytome).')

    # Build infog_params dict for downstream use
    infog_params = {
        'cell_depth': cell_depth,
        'inv_gene_depth': inv_gene_depth,
        'scale': scale,
        'counts_sum': counts_sum,
        'threshold': threshold,
    }

    result = {
        'gene_var': gene_var,
        'hvg_mask': hvg_mask,
        'hvg_indices': hvg_indices,
        'scale': scale,
        'counts_sum': counts_sum,
        'cell_depth': cell_depth,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'infog_params': infog_params,
    }

    if is_adata:
        return adata if copy else result
    return result


# ============================================================================
# Public dispatcher
# ============================================================================
def infog(
    data=_UNSET,
    copy: bool = False,
    inplace: bool = False,
    n_top_genes: int = 3000,
    key_added: str = 'infog',
    key_added_highly_variable_gene: str = 'highly_variable',
    trim: bool = True,
    verbosity: int = 1,
    layer: Optional[str] = None,
    # NEW parameters for streaming:
    streaming: bool = False,
    batch_size: int = 1024,
    save_layer: bool = False,
    modality: str = "RNA",
    return_info: bool = False,
    # ---- deprecated aliases (back-compat) ----
    source=_UNSET,
    adata=_UNSET,
):
    """
    INFOG normalization of single-cell RNA sequencing data.

    Supports three modes:
    - infog(adata) — standard in-memory (existing behavior, unchanged)
    - infog(adata, streaming=True) — streaming from in-memory AnnData chunks
    - infog("path.cytome") — streaming from on-disk cytome dataset
    - infog(cytome_dataset) — streaming from already-opened cytome object

    Parameters
    ----------
    source : AnnData, CytomeDataset, or str
        AnnData object, cytome Dataset object, or path to .cytome file.
    modality : str, default 'RNA'
        Cytome modality to read. Only meaningful when ``source`` is a cytome
        path or Dataset; ignored for AnnData inputs. Use 'GA' to compute
        INFOG on gene-activity matrices (e.g. after
        ``piaso.tl.inferGeneActivity``). The math is modality-agnostic;
        passing 'ATAC' / 'tiles' is technically supported but biologically
        unusual.
    streaming : bool, default=False
        If True and source is AnnData, use streaming mode. Ignored if source is str
        or CytomeDataset (these always use streaming).
    batch_size : int, default=1024
        Number of cells per chunk in streaming mode. Ignored in standard mode.
    save_layer : bool, default=False
        If True and source is cytome, write the full INFOG-normalized matrix to the
        cytome file (layer ``key_added``). Default (False) is lazy mode: only
        normalization parameters are saved, and normalization is applied on-the-fly
        during downstream operations like SVD.
    [all other parameters unchanged from original infog()]
    """
    data = _resolve_data_arg(data, 'infog', source=source, adata=adata)
    is_cytome = isinstance(data, str) or _is_cytome_dataset(data)

    if is_cytome or streaming:
        result = _infog_streaming(
            data, n_top_genes=n_top_genes, trim=trim,
            batch_size=batch_size, key_added=key_added,
            key_added_highly_variable_gene=key_added_highly_variable_gene,
            layer=layer, verbosity=verbosity, copy=copy, inplace=inplace,
            save_layer=save_layer, modality=modality,
        )
        # AnnData + copy=True returns the new normalized AnnData; otherwise the
        # results live in ds.metadata / ds.genes (cytome) or adata (in-place),
        # so return None by default (consistent with importFragments/selectPeaks/
        # runSVD). Pass return_info=True to get the params dict back.
        if copy and not is_cytome:
            return result
        return result if return_info else None
    else:
        out = _infog_original(
            data, copy=copy, inplace=inplace,
            n_top_genes=n_top_genes, key_added=key_added,
            key_added_highly_variable_gene=key_added_highly_variable_gene,
            trim=trim, verbosity=verbosity, layer=layer,
        )
        # _infog_original already returns adata-if-copy-else-None; honor return_info
        # only when it produced a result dict (it doesn't today, so this is a no-op).
        return out


### Refer to Scanpy for _select_top_n function
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores.iloc[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices

### Refer to scDRS for _get_p_from_empi_null function
def _get_p_from_empi_null(
    v_t,
    v_t_null
):
    """
    Compute p-value from empirical null
    For score T and a set of null score T_1,...T_N, the p-value is

        p= [1 + \Sigma_{i=1}^N 1_{ (T_i \geq T) }] / (1+N)

    If T, T_1, ..., T_N are i.i.d. variables following a null distritbuion,
    then p is super-uniform.

    The naive algorithm is N^2. Here we provide an O(N log N) algorithm to
    compute the p-value for each of the N elements in v_t

    Args
    ----
    v_t : np.ndarray
        Observed score of shape (M,).
    v_t_null : np.ndarray
        Null scores of shape (N,).
        
    Returns
    -------
    v_p: : np.ndarray
        P-value for each element in v_t of shape (M,).
    """

    v_t = np.array(v_t)
    v_t_null = np.array(v_t_null)

    v_t_null = np.sort(v_t_null)
    v_pos = np.searchsorted(v_t_null, v_t, side="left")
    v_p = (v_t_null.shape[0] - v_pos + 1) / (v_t_null.shape[0] + 1)
    return v_p


import numpy as np
from sklearn.neighbors import KDTree
from scipy import sparse


def _precompute_stats(cellxgene, n_nearest_neighbors=30, leaf_size=40):
    """Compute KDTree-based KNN indices for control gene sampling.

    Uses CSR sharing trick for variance computation (reuses indices/indptr,
    only allocates data**2).

    Parameters
    ----------
    cellxgene : sparse matrix or ndarray
        Gene expression matrix (n_cells, n_genes).
    n_nearest_neighbors : int
        Number of nearest neighbors per gene in (mean, var) space.
    leaf_size : int
        KDTree leaf size.

    Returns
    -------
    knn_idx : ndarray (n_genes, n_nearest_neighbors)
        KNN indices for each gene (self-loops removed).
    """
    mean_2d = np.array(cellxgene.mean(axis=0), dtype=np.float64)  # (1, n_genes)
    infog_mean = mean_2d.copy()[0]  # (n_genes,)
    mean_sq = mean_2d ** 2

    if sparse.issparse(cellxgene):
        data_sq = cellxgene.data.astype(np.float64) ** 2
        X_sq = sparse.csr_matrix((data_sq, cellxgene.indices, cellxgene.indptr),
                                  shape=cellxgene.shape, copy=False)
        residual_var = np.squeeze(np.array(X_sq.mean(axis=0), dtype=np.float64) - mean_sq)
        del X_sq, data_sq
    else:
        residual_var = np.squeeze(np.mean(np.asarray(cellxgene, dtype=np.float64) ** 2, axis=0) - mean_sq)

    mean_var = np.array([infog_mean, residual_var]).T
    kdt = KDTree(mean_var, leaf_size=leaf_size, metric='euclidean')
    knn_idx = kdt.query(mean_var, k=n_nearest_neighbors + 1, return_distance=False)

    # Remove self-loops (vectorized — avoids Python list comprehension)
    n_genes_knn = knn_idx.shape[0]
    self_col = np.arange(n_genes_knn)
    self_mask = (knn_idx == self_col[:, None])
    # Find which column holds the self-match (first True per row; if none, use last col)
    has_self = self_mask.any(axis=1)
    first_self_col = np.argmax(self_mask, axis=1)  # 0 if no self → handled below
    # Build shifted index: for rows with self-match, skip that column
    cols = np.arange(knn_idx.shape[1])
    shift = (cols[None, :] >= first_self_col[:, None]) & has_self[:, None]
    gather = cols[None, :] + shift.astype(np.intp)
    # Clamp to valid range
    np.clip(gather, 0, knn_idx.shape[1] - 1, out=gather)
    knn_idx = np.take_along_axis(knn_idx, gather, axis=1)[:, :n_nearest_neighbors].astype(np.int64)

    return knn_idx


def _precompute_stats_streaming(
    iter_chunks_fn,
    n_cells: int,
    n_features: int,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 40,
    cell_mask=None,
):
    """Streaming version of _precompute_stats: one pass through cytome chunks.

    Accumulates per-feature sum and sum-of-squares across row-chunks,
    then builds KDTree on (mean, residual_variance) space.
    Memory: O(n_features) — the full matrix is never loaded.

    Parameters
    ----------
    iter_chunks_fn : callable
        Factory returning iterator yielding ``(csr_chunk, row_indices)``.
    n_cells : int
        Total number of cells.
    n_features : int
        Number of features (genes/peaks).
    n_nearest_neighbors, leaf_size : int
        Same as ``_precompute_stats``.
    cell_mask : np.ndarray or None
        Boolean mask of length ``n_cells``. When set, accumulates stats
        only on masked rows and the per-feature denominator becomes
        the number of True entries. Used by multi-batch GDR so each
        batch's control-set sampling reflects that batch's local
        per-feature distribution.

    Returns
    -------
    knn_idx : ndarray (n_features, n_nearest_neighbors)
        KNN indices for each feature (self-loops removed).
    """
    col_sum = np.zeros(n_features, dtype=np.float64)
    col_sq_sum = np.zeros(n_features, dtype=np.float64)

    if cell_mask is not None:
        cell_mask = np.asarray(cell_mask).astype(bool)
        n_effective = int(cell_mask.sum())
    else:
        n_effective = n_cells

    for chunk_csr, row_indices in iter_chunks_fn():
        if cell_mask is not None:
            chunk_keep = cell_mask[row_indices]
            if not chunk_keep.any():
                continue
            if not chunk_keep.all():
                chunk_csr = chunk_csr[chunk_keep]
        # Accumulate per-column sum
        col_sum += np.array(chunk_csr.sum(axis=0), dtype=np.float64).ravel()
        # CSR sharing trick: reuse indices/indptr, only square data
        sq_data = chunk_csr.data.astype(np.float64) ** 2
        X_sq = sparse.csr_matrix(
            (sq_data, chunk_csr.indices, chunk_csr.indptr),
            shape=chunk_csr.shape, copy=False,
        )
        col_sq_sum += np.array(X_sq.sum(axis=0), dtype=np.float64).ravel()

    infog_mean = col_sum / max(n_effective, 1)
    mean_sq = infog_mean ** 2
    residual_var = (col_sq_sum / max(n_effective, 1)) - mean_sq

    mean_var = np.array([infog_mean, residual_var]).T
    kdt = KDTree(mean_var, leaf_size=leaf_size, metric='euclidean')
    knn_idx = kdt.query(mean_var, k=n_nearest_neighbors + 1, return_distance=False)

    # Remove self-loops (same vectorized code as _precompute_stats)
    n_genes_knn = knn_idx.shape[0]
    self_col = np.arange(n_genes_knn)
    self_mask = (knn_idx == self_col[:, None])
    has_self = self_mask.any(axis=1)
    first_self_col = np.argmax(self_mask, axis=1)
    cols = np.arange(knn_idx.shape[1])
    shift = (cols[None, :] >= first_self_col[:, None]) & has_self[:, None]
    gather = cols[None, :] + shift.astype(np.intp)
    np.clip(gather, 0, knn_idx.shape[1] - 1, out=gather)
    knn_idx = np.take_along_axis(knn_idx, gather, axis=1)[:, :n_nearest_neighbors].astype(np.int64)

    return knn_idx


def _score_streaming_multi(
    ds,
    gene_list,
    gene_weights=None,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 40,
    modality: str = "ATAC",
    layer: str = "counts",
    random_seed: int = 1927,
    n_ctrl_set: int = 100,
    compute_pvalues: bool = False,
    chunk_size: int = 10000,
    max_workers: int = 1,
    use_rust: bool = True,
    precomputed_knn=None,
    batch_size: int = 1024,
    verbosity: int = 0,
    cell_mask=None,
):
    """Two-pass streaming score() for cytome datasets.

    Pass 1: ``_precompute_stats_streaming`` → knn_idx
    Between passes: build ``big_ctrl`` (no I/O)
    Pass 2: fused query_scores + ctrl matmul per chunk

    Memory: ~388 MB for ADVIS-scale (200K cells, 700K peaks).

    SINGLE-PROCESS: this cytome path does NOT fork/spawn any worker processes —
    it streams chunks in one Python process and offloads the matmul/reduce to the
    Rust ``fused_matmul_reduce`` (or a numpy fallback). ``max_workers`` here is the
    **Rust thread (rayon) count**, NOT a Python process pool. The spawn-context
    ``ProcessPoolExecutor`` in ``calculateScoreParallel`` is the AnnData/``scanpy``
    scoring path only (which raises for cytome). So callers like
    ``inferGeneActivity`` / ``inferTFActivity`` on a cytome never spawn processes
    and are nbconvert/fork-safe regardless of ``max_workers``.

    Parameters
    ----------
    ds : cytome Dataset
        Open cytome dataset.
    gene_list : dict, DataFrame, or list of lists
        Gene set(s) to score.
    modality, layer : str
        Cytome modality and layer to read.
    batch_size : int
        Cytome chunk size for streaming.
    Other parameters same as ``score()``.

    Returns
    -------
    score_matrix, gene_set_names, pval_matrix : same as ``score()`` multi-set.
    """
    np.random.seed(random_seed)

    # Get dimensions (defensive: use safe allocation size for index safety)
    matrix_name = f"{modality}_{layer}"
    from cytome.core.measurement import MeasurementLayer
    ml = MeasurementLayer(ds._conn, matrix_name)
    n_alloc, n_true_cells = _safe_n_cells(ds, modality, layer)
    n_cells = n_alloc
    n_features = ml.shape[1]

    # Feature names for gene set matching
    cursor = ds._conn.execute(
        "SELECT col_entity FROM matrix_meta WHERE matrix_name=?", (matrix_name,)
    )
    row = cursor.fetchone()
    entity_table = row[0] if row else "genes"
    if entity_table == "peaks":
        var_names = np.array(ds.peaks["peak_id"])
        var_names_to_idx = {name: [idx] for idx, name in enumerate(var_names)}
    elif entity_table == "tiles":
        # ds.tiles is a Modality (no __getitem__); the tiles var TABLE is
        # ds.features('tiles').
        var_names = np.array(ds.features("tiles")["tile_id"])
        var_names_to_idx = {name: [idx] for idx, name in enumerate(var_names)}
    else:
        # Display names via the single source of truth (symbol-first), PLUS a
        # vocabulary-robust lookup that also indexes gene_id etc. — so COSG marker
        # symbols AND any user-supplied Ensembl list both resolve (Round 25 fix).
        var_names = _resolve_gene_names(ds, modality)
        var_names_to_idx = _gene_name_alias_map(ds, modality, var_names)

    if verbosity > 0:
        print(f"  Streaming score: {n_cells} cells × {n_features} features")

    # Validate and normalize cell_mask
    if cell_mask is not None:
        cell_mask_arr = np.asarray(cell_mask).astype(bool)
        if cell_mask_arr.shape[0] != n_true_cells:
            raise ValueError(
                f"cell_mask length ({cell_mask_arr.shape[0]}) must match "
                f"cytome n_cells ({n_true_cells})."
            )
        # Expand to n_alloc-length (defensive over-allocation may pad chunks)
        if n_alloc > n_true_cells:
            mask_full = np.zeros(n_alloc, dtype=bool)
            mask_full[:n_true_cells] = cell_mask_arr
            cell_mask_arr = mask_full
    else:
        cell_mask_arr = None

    # --- Parse + validate gene sets FIRST (fail fast with a pointed error before
    # the expensive KNN pass, so a wrong `modality` doesn't surface as an obscure
    # KDTree error). ---
    if isinstance(gene_list, pd.DataFrame):
        parsed = {}
        for col in gene_list.columns:
            genes = gene_list[col].dropna().tolist()
            if genes:
                parsed[col] = genes
    elif isinstance(gene_list, dict):
        parsed = gene_list
    else:  # list of lists
        parsed = {f"GeneSet_{i}": gs for i, gs in enumerate(gene_list)}

    gene_set_names = []
    gene_sets_indices = []
    weights_list = []

    for gs_idx, (name, genes) in enumerate(parsed.items()):
        # name -> [indices]: a duplicated symbol resolves to ALL its features
        # (union scoring). Track per-resolved-index weights in parallel.
        valid_idx = []
        valid_w = []
        w_vec = gene_weights[gs_idx] if gene_weights is not None else None
        for gi, g in enumerate(genes):
            idxs = var_names_to_idx.get(g)
            if not idxs:
                continue
            valid_idx.extend(idxs)
            valid_w.extend([(w_vec[gi] if w_vec is not None else 1.0)] * len(idxs))
        if not valid_idx:
            continue
        gene_set_names.append(name)
        gene_sets_indices.append(valid_idx)
        weights_list.append(np.array(valid_w, dtype=float))

    n_sets = len(gene_set_names)
    if n_sets == 0:
        _ex = list(var_names[:5])
        raise ValueError(
            f"No valid gene sets: none of the supplied gene names matched features "
            f"in modality '{modality}' (matrix '{modality}_{layer}', {n_features} "
            f"features named like {_ex}). The most common cause is the wrong "
            f"`modality` — for gene-set scoring on RNA pass `modality='RNA'` "
            f"(the default is 'RNA'; ATAC/tiles hold peaks, not gene symbols). Also "
            f"check the gene names match the cytome's feature names (symbols vs Ensembl)."
        )

    # --- Pass 1: streaming stats -> KNN ---
    if precomputed_knn is not None:
        knn_idx = precomputed_knn
    else:
        def chunk_factory():
            return ds.iter_chunks(modality=modality, layer=layer, batch_size=batch_size)
        if verbosity > 0:
            print(f"  Pass 1: Computing per-feature stats (streaming)...")
        knn_idx = _precompute_stats_streaming(
            chunk_factory, n_cells, n_features, n_nearest_neighbors, leaf_size,
            cell_mask=cell_mask_arr,
        )

    # --- Build control structures (no I/O) ---
    all_ctrl_blocks = []
    scaling_factors = np.zeros(n_sets, dtype=np.float64)

    for gs_idx in range(n_sets):
        np.random.seed(random_seed)
        gene_idx = gene_sets_indices[gs_idx]
        weights = weights_list[gs_idx]

        gs_knn_idx = knn_idx[gene_idx]
        n_gs = len(gene_idx)
        n_neighbors = gs_knn_idx.shape[1]
        rand_idx = np.random.randint(0, n_neighbors, size=(n_gs, n_ctrl_set))
        ctrl_sampled = gs_knn_idx[np.arange(n_gs)[:, None], rand_idx].T

        rows_all = ctrl_sampled.ravel()
        cols_all = np.repeat(np.arange(n_ctrl_set, dtype=np.int32), n_gs)
        data_all = np.tile(weights, n_ctrl_set)
        ctrl_block = sparse.csr_matrix(
            (data_all, (rows_all, cols_all)),
            shape=(n_features, n_ctrl_set)
        )
        all_ctrl_blocks.append(ctrl_block)
        scaling_factors[gs_idx] = np.median(weights) * n_gs

    big_ctrl_csr = sparse.hstack(all_ctrl_blocks, format='csr')

    # --- Try Rust fused_matmul_reduce ---
    _rust_fmr = None
    if use_rust:
        try:
            from piaso._piaso import fused_matmul_reduce
            _rust_fmr = fused_matmul_reduce
        except ImportError:
            pass

    # --- Pass 2: streaming query scores + control matmul ---
    if verbosity > 0:
        print(f"  Pass 2: Streaming matmul ({n_sets} gene sets)...")
    query_scores = np.zeros((n_cells, n_sets), dtype=np.float64)
    ctrl_means = np.zeros((n_cells, n_sets), dtype=np.float64)
    pval_matrix = np.zeros((n_cells, n_sets), dtype=np.float64) if compute_pvalues else None

    # Pre-convert big_ctrl for Rust
    bc_data = big_ctrl_csr.data.astype(np.float64, copy=False)
    bc_indices = big_ctrl_csr.indices.astype(np.int32, copy=False)
    bc_indptr = big_ctrl_csr.indptr.astype(np.int32, copy=False)
    bc_n_cols = big_ctrl_csr.shape[1]

    for chunk_csr, row_indices in ds.iter_chunks(
        modality=modality, layer=layer, batch_size=batch_size
    ):
        if not sparse.isspmatrix_csr(chunk_csr):
            chunk_csr = chunk_csr.tocsr()
        ri = np.asarray(row_indices)

        if cell_mask_arr is not None:
            chunk_keep = cell_mask_arr[ri]
            if not chunk_keep.any():
                continue
            if not chunk_keep.all():
                chunk_csr = chunk_csr[chunk_keep]
                ri = ri[chunk_keep]

        n_chunk = chunk_csr.shape[0]

        # Query scores: column-subset weighted sum
        for gs_idx in range(n_sets):
            gene_idx = gene_sets_indices[gs_idx]
            weights = weights_list[gs_idx]
            sub = chunk_csr[:, gene_idx]
            query_scores[ri, gs_idx] = np.ravel(sub.multiply(weights).sum(axis=1))

        # Control matmul
        if _rust_fmr is not None:
            m_flat, p_flat = _rust_fmr(
                chunk_csr.data.astype(np.float64, copy=False),
                chunk_csr.indices.astype(np.int32, copy=False),
                chunk_csr.indptr.astype(np.int32, copy=False),
                n_chunk, n_features,
                bc_data, bc_indices, bc_indptr, bc_n_cols,
                query_scores[ri].ravel().astype(np.float64, copy=False),
                n_sets, n_ctrl_set, n_chunk, max(1, max_workers), compute_pvalues,
            )
            ctrl_means[ri] = m_flat.reshape(n_chunk, n_sets)
            if compute_pvalues and p_flat is not None:
                pval_matrix[ri] = p_flat.reshape(n_chunk, n_sets)
        else:
            # Python fallback: chunked dense
            chunk_result = chunk_csr @ big_ctrl_csr
            if sparse.issparse(chunk_result):
                chunk_dense = chunk_result.toarray()
            else:
                chunk_dense = np.asarray(chunk_result)
            chunk_3d = chunk_dense.reshape(n_chunk, n_sets, n_ctrl_set)
            ctrl_means[ri] = chunk_3d.mean(axis=2)
            if compute_pvalues:
                q_chunk = query_scores[ri]
                n_greater = np.sum(chunk_3d >= q_chunk[:, :, None], axis=2)
                pval_matrix[ri] = (n_greater + 1) / (n_ctrl_set + 1)

    # Background-subtracted scores
    score_matrix = (query_scores / scaling_factors[None, :]) - (ctrl_means / scaling_factors[None, :])

    # Truncate to true cell count if defensive over-allocation was used
    if n_alloc > n_true_cells:
        score_matrix = score_matrix[:n_true_cells]
        if pval_matrix is not None:
            pval_matrix = pval_matrix[:n_true_cells]

    # If cell_mask was set, the output rows correspond to MASKED cells only.
    # Filter to (n_masked, n_sets) so downstream multi-batch GDR can drop
    # the per-batch scores into the correct slots.
    if cell_mask is not None:
        cell_mask_short = np.asarray(cell_mask).astype(bool)[:n_true_cells]
        score_matrix = score_matrix[cell_mask_short]
        if pval_matrix is not None:
            pval_matrix = pval_matrix[cell_mask_short]

    return score_matrix, gene_set_names, pval_matrix


def _write_cytome_score(ds, key_added, score_val, pval, compute_pvalues, pvalue_to, cell_mask, verbosity):
    """Write a single-gene-set cytome score (+ p-values) to ``ds.cells`` and/or ``ds.metadata``.

    Mirrors the AnnData path (``adata.obs[key_added]`` = score, ``adata.uns[key_added]`` = the table):
    here ``ds.cells[key_added]`` = score, plus (when ``compute_pvalues``) the plottable columns
    ``{key_added}_pval`` (Monte-Carlo per-cell), ``{key_added}_nlog10pval``, ``{key_added}_FDR`` —
    and, for ``pvalue_to in ('metadata','both')``, the full per-cell table in ``ds.metadata[key_added]``.
    """
    from statsmodels.stats.multitest import multipletests
    n_cells = getattr(ds, "n_cells", None) or ds.cells.shape[0]

    def _scatter(vals):
        if cell_mask is None or len(vals) == n_cells:
            return np.asarray(vals, dtype=np.float32)
        full = np.full(n_cells, np.nan, dtype=np.float32)
        full[np.asarray(cell_mask).astype(bool)] = np.asarray(vals, dtype=np.float32)
        return full

    cols = {key_added: _scatter(score_val)}
    if compute_pvalues and pval is not None:
        pv = np.asarray(pval, dtype=np.float64)
        fdr = multipletests(pv, method="fdr_bh")[1]
        cols[f"{key_added}_pval"] = _scatter(pv)               # Monte-Carlo per-cell empirical p
        cols[f"{key_added}_nlog10pval"] = _scatter(-np.log10(np.clip(pv, 1e-300, None)))
        cols[f"{key_added}_FDR"] = _scatter(fdr)
    if pvalue_to in ("cells", "both"):
        for name, arr in cols.items():
            ds.cells[name] = arr
    if pvalue_to in ("metadata", "both"):
        ds.metadata[key_added] = {k: np.asarray(v, dtype=np.float32).tolist() for k, v in cols.items()}
    if hasattr(ds, "flush"):
        ds.flush()                              # entity/metadata writes are enqueued — persist them
    if verbosity > 0:
        where = "ds.cells" + ("" if pvalue_to == "cells" else " + ds.metadata")
        print(f"Finished. The score is saved as ds.cells['{key_added}'] "
              f"({'and p-values ' if compute_pvalues else ''}in {where}).")


#### Gene Set Scoring Method
def score(
    data=_UNSET,
    gene_list=_UNSET,
    gene_weights=None,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 40,
    layer=_UNSET,                              # default → 'infog'; None → adata.X; canonical (beats cytome_layer)
    random_seed: int = 1927,
    n_ctrl_set: int = 100,
    key_added: str = None,
    compute_pvalues: bool = False,
    chunk_size: int = 10000,
    max_workers: int = 1,
    use_rust: bool = True,
    precomputed_knn: np.ndarray = None,
    verbosity: int = 0,
    verbose: int = None,                       # deprecated alias for `verbosity` (matches runGDR/infog)
    # Cytome streaming parameters
    modality: str = "RNA",
    batch_size: int = 1024,
    cell_mask=None,
    pvalue_to: str = "both",                   # cytome single-set: 'cells' | 'metadata' | 'both'
    # ---- deprecated aliases (back-compat) ----
    adata=_UNSET,
):
    """
    Compute gene-set enrichment scores for one or more gene sets, on **AnnData or a Cytome dataset**.

    Backend / modality
    ------------------
    Pass either an ``AnnData`` (in-memory) or a Cytome ``Dataset`` (streamed from disk). For Cytome,
    ``modality`` (e.g. ``'RNA'`` / ``'ATAC'`` / ``'GA'``) and ``cytome_layer`` select the matrix; for
    AnnData, ``layer`` selects ``adata.layers[layer]`` (default ``'infog'``).

    Output
    ------
    Single gene set with ``key_added`` set:
      - **AnnData:** score → ``adata.obs[key_added]``; the full per-cell p-value table (``score``,
        ``pval_mc`` [Monte-Carlo, per cell vs its own control sets], ``pval`` [pooled empirical], plus
        ``*_FDR`` / ``nlog10_*``) → ``adata.uns[key_added]``.
      - **Cytome:** score → ``ds.cells[key_added]``; the plottable columns ``{key_added}_pval``
        (Monte-Carlo), ``{key_added}_nlog10pval``, ``{key_added}_FDR`` → ``ds.cells`` and/or the full
        table → ``ds.metadata[key_added]``, controlled by ``pvalue_to`` (``'cells'`` / ``'metadata'`` /
        ``'both'``). Always also returns ``(score, names, pval)``.

    Supports two modes based on the type of ``gene_list``:

    **Single gene set** (list of str): Computes scores and full p-value suite
    (Monte Carlo, pooled empirical, FDR) for one gene set. Results are stored
    in ``adata.obs`` and ``adata.uns``. Returns None.

    **Multiple gene sets** (dict, DataFrame, or list of lists): Scores all gene
    sets in one batched pass using a single hstack'd sparse matmul. Optionally
    uses the Rust ``piaso_score`` backend for 16.8x faster matmul with 200x less
    RAM per thread. Returns (score_matrix, gene_set_names, pval_matrix).

    Parameters
    ----------
    adata : AnnData
        The AnnData object for the gene expression matrix.

    gene_list : list of str, dict, DataFrame, or list of lists
        A list of gene names (single gene set), or a dict / DataFrame / list of
        lists mapping gene set names to gene lists (multiple gene sets).

    gene_weights : array-like or list of arrays, optional
        For single gene set: a list of weights matching ``gene_list``.
        For multiple gene sets: a list of weight arrays, one per gene set.
        If None, all genes are weighted equally. Default is None.

    n_nearest_neighbors : int, optional
        Number of nearest neighbors for control gene sampling. Default is 30.

    leaf_size : int, optional
        KDTree leaf size. Default is 40.

    layer : str, optional
        Layer in ``adata.layers`` to use. Default is 'infog'.

    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.

    n_ctrl_set : int, optional
        Number of control gene sets. Default is 100.

    key_added : str, optional
        Key for storing results in adata (single-set mode only).
        Default is None ('INFOG_score').

    compute_pvalues : bool, optional
        Compute Monte Carlo p-values in multi-set mode. Single-set mode
        always computes full p-values. Default is False.

    chunk_size : int, optional
        Cell chunk size for Python dense matmul fallback. Default is 10000.

    max_workers : int, optional
        Thread count for Rust backend (1 = single-threaded). Default is 1.

    use_rust : bool, optional
        Try Rust fused matmul-reduce backend if available. Default is True.

    precomputed_knn : ndarray, optional
        Pre-computed KNN indices from ``_precompute_stats()``. If provided,
        skips the KDTree construction and KNN search. Useful when calling
        ``score()`` multiple times on the same expression matrix.

    verbosity : int, optional
        Level of verbosity. Default is 0.

    Returns
    -------
    Single-set mode: None. Modifies ``adata`` in-place.
    Multi-set mode: (score_matrix, gene_set_names, pval_matrix).

    Example
    -------
    >>> import piaso
    >>> # Single gene set
    >>> piaso.tl.score(adata, ['Gene1', 'Gene2', 'Gene3'], key_added='my_score')
    >>>
    >>> # Multiple gene sets (batched, with optional Rust acceleration)
    >>> scores, names, pvals = piaso.tl.score(
    ...     adata, {'SetA': ['Gene1', 'Gene2'], 'SetB': ['Gene3', 'Gene4']},
    ...     compute_pvalues=True, max_workers=8
    ... )
    """
    adata = _resolve_data_arg(data, 'score', adata=adata)
    if gene_list is _UNSET:
        raise TypeError("score() missing required argument: 'gene_list'")
    if verbose is not None:                    # deprecated alias → verbosity (matches runGDR/infog)
        import warnings as _w
        _w.warn("piaso.tl.score: `verbose` is deprecated; use `verbosity`.", FutureWarning, stacklevel=2)
        verbosity = verbose
    # `layer` is the CANONICAL layer param for BOTH backends (matches runGDR).
    # The `cytome_layer=` alias was removed in 1.2.0 — it never shipped, and
    # letting a deprecated alias override the canonical param was the source of
    # the `KeyError: 'counts'` bug (regulonActivity forwarded a stale
    # cytome_layer='counts' that clobbered layer='infog').
    if layer is _UNSET:
        layer = 'infog'        # documented default (both backends). Explicit None = the base/raw matrix:
                               # AnnData → adata.X; cytome → {modality}_counts (mapped just below).
    # --- Cytome streaming dispatch ---
    if _is_cytome_dataset(adata):
        if layer is None:      # a cytome has no `.X`; None means the raw matrix = {modality}_counts
            layer = "counts"   # (mirrors `_get_infog_chunk_iterator`'s `layer or 'counts'`)
        _multi = isinstance(gene_list, (dict, pd.DataFrame)) or (
            isinstance(gene_list, list) and len(gene_list) > 0
            and isinstance(gene_list[0], (list, np.ndarray))
        )
        if _multi:
            return _score_streaming_multi(
                adata, gene_list, gene_weights=gene_weights,
                n_nearest_neighbors=n_nearest_neighbors, leaf_size=leaf_size,
                modality=modality, layer=layer,
                random_seed=random_seed, n_ctrl_set=n_ctrl_set,
                compute_pvalues=compute_pvalues, chunk_size=chunk_size,
                max_workers=max_workers, use_rust=use_rust,
                precomputed_knn=precomputed_knn,
                batch_size=batch_size, verbosity=verbosity,
                cell_mask=cell_mask,
            )
        else:
            # Single gene set via cytome: wrap in dict, call multi, unwrap
            wrapped = {"_single": gene_list}
            wrapped_weights = [gene_weights] if gene_weights is not None else None
            sm, _names, pm = _score_streaming_multi(
                adata, wrapped, gene_weights=wrapped_weights,
                n_nearest_neighbors=n_nearest_neighbors, leaf_size=leaf_size,
                modality=modality, layer=layer,
                random_seed=random_seed, n_ctrl_set=n_ctrl_set,
                compute_pvalues=compute_pvalues, chunk_size=chunk_size,
                max_workers=max_workers, use_rust=use_rust,
                precomputed_knn=precomputed_knn,
                batch_size=batch_size, verbosity=verbosity,
                cell_mask=cell_mask,
            )
            score_val = sm[:, 0]
            pval = pm[:, 0] if pm is not None else None
            if key_added is not None:
                _write_cytome_score(adata, key_added, score_val, pval, compute_pvalues,
                                    pvalue_to, cell_mask, verbosity)
            return score_val, _names, pval

    np.random.seed(random_seed)

    # cell_mask on AnnData: subset adata, then run the existing logic on the
    # subset. Output rows correspond to masked cells only — matches the
    # cytome path's contract.
    _original_adata = None
    if cell_mask is not None:
        cell_mask_arr = np.asarray(cell_mask).astype(bool)
        if cell_mask_arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"cell_mask length ({cell_mask_arr.shape[0]}) must match "
                f"adata.n_obs ({adata.n_obs})."
            )
        _original_adata = adata
        adata = adata[cell_mask_arr].copy()

    # --- Determine mode: single gene set vs multiple ---
    _multi = isinstance(gene_list, (dict, pd.DataFrame)) or (
        isinstance(gene_list, list) and len(gene_list) > 0
        and isinstance(gene_list[0], (list, np.ndarray))
    )

    # --- Expression matrix and shared stats ---
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"piaso.tl.score: layer '{layer}' not found in adata.layers "
            f"(available: {list(adata.layers)}). Create it (e.g. piaso.tl.score/normalize to make "
            f"'infog'), pass an existing layer, or pass layer=None to use adata.X.")
    cellxgene = adata.layers[layer] if layer is not None else adata.X
    n_cells, n_genes = cellxgene.shape
    if precomputed_knn is not None:
        knn_idx = precomputed_knn
    else:
        knn_idx = _precompute_stats(cellxgene, n_nearest_neighbors, leaf_size)
    var_names_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}

    # --- Try Rust backend (multi-set only) ---
    _rust_fmr = None
    _rust_sc = None
    if use_rust and _multi:
        try:
            from piaso._piaso import fused_matmul_reduce, score_complete
            _rust_fmr = fused_matmul_reduce
            _rust_sc = score_complete
        except ImportError:
            try:
                from piaso._piaso import fused_matmul_reduce
                _rust_fmr = fused_matmul_reduce
            except ImportError:
                pass
            # use_rust=True was requested but the compiled extension
            # (piaso._piaso) is missing/stale, so we silently dropped to the
            # pure-Python path (correct, but ~16× slower). Warn so this isn't
            # invisible — mirrors quantifyPeakActivity's stale-.so guard, but a
            # warning (not an error) since the Python path is fully valid.
            # Python de-dupes identical warnings per call site, so loops won't
            # spam. Pass use_rust=False to silence intentionally.
            warnings.warn(
                "score(use_rust=True): the compiled Rust backend "
                "(piaso._piaso.score_complete) is unavailable — falling back to "
                "the pure-Python path (~16x slower, identical results). Rebuild "
                "the extension (maturin develop / pip install with a Rust "
                "toolchain) for the fast path, or pass use_rust=False to silence.",
                RuntimeWarning,
                stacklevel=2,
            )
            if verbosity > 0:
                print("Rust backend not available. Using Python backend. "
                      "Reinstall with: pip install piaso-tools (requires Rust toolchain for source builds)")

    if _multi:
        # ================================================================
        # MULTI GENE SET PATH — batched matmul with optional Rust
        # ================================================================

        # Parse gene_list into {name: [gene_names]}
        if isinstance(gene_list, pd.DataFrame):
            parsed = {}
            for col in gene_list.columns:
                genes = gene_list[col].dropna().tolist()
                if genes:
                    parsed[col] = genes
        elif isinstance(gene_list, dict):
            parsed = gene_list
        else:  # list of lists
            parsed = {f"GeneSet_{i}": gs for i, gs in enumerate(gene_list)}

        # Map gene names to indices, filter invalid
        gene_set_names = []
        gene_sets_indices = []
        weights_list = []

        for gs_idx, (name, genes) in enumerate(parsed.items()):
            valid_idx = [var_names_to_idx[g] for g in genes if g in var_names_to_idx]
            if not valid_idx:
                continue
            gene_set_names.append(name)
            gene_sets_indices.append(valid_idx)
            if gene_weights is not None:
                w = gene_weights[gs_idx]
                valid_mask = [g in var_names_to_idx for g in genes]
                weights_list.append(np.array([w_i for w_i, m in zip(w, valid_mask) if m], dtype=float))
            else:
                weights_list.append(np.ones(len(valid_idx), dtype=float))

        n_sets = len(gene_set_names)
        if n_sets == 0:
            raise ValueError("No valid gene sets found.")

        # --- Fused Rust path: control sampling + matmul + reduce in one call ---
        if _rust_sc is not None:
            cellxgene_csr = cellxgene.tocsr() if not sparse.isspmatrix_csr(cellxgene) else cellxgene

            # Pack gene sets and weights into flat arrays with offsets (CSR-style)
            gs_flat = []
            gs_offsets = [0]
            w_flat = []
            w_offsets = [0]
            for gs_idx in range(n_sets):
                gs_flat.extend(gene_sets_indices[gs_idx])
                gs_offsets.append(len(gs_flat))
                w_flat.extend(weights_list[gs_idx].tolist())
                w_offsets.append(len(w_flat))

            scores_flat, query_flat, sf_flat, pval_flat = _rust_sc(
                cellxgene_csr.data.astype(np.float64, copy=False),
                cellxgene_csr.indices.astype(np.int32, copy=False),
                cellxgene_csr.indptr.astype(np.int32, copy=False),
                n_cells, n_genes,
                knn_idx.ravel().astype(np.int64, copy=False),
                knn_idx.shape[1],
                np.array(gs_flat, dtype=np.int32),
                np.array(gs_offsets, dtype=np.int32),
                np.array(w_flat, dtype=np.float64),
                np.array(w_offsets, dtype=np.int32),
                n_ctrl_set, random_seed, chunk_size,
                max(1, max_workers), compute_pvalues,
            )
            score_matrix = scores_flat.reshape(n_cells, n_sets)
            pval_matrix = pval_flat.reshape(n_cells, n_sets) if pval_flat is not None else None
            return score_matrix, gene_set_names, pval_matrix

        # --- Python fallback: build control blocks + matmul ---
        all_ctrl_blocks = []
        query_scores = np.zeros((n_cells, n_sets), dtype=np.float64)
        scaling_factors = np.zeros(n_sets, dtype=np.float64)

        for gs_idx in range(n_sets):
            # Reset seed per gene set for reproducibility
            np.random.seed(random_seed)

            gene_idx = gene_sets_indices[gs_idx]
            weights = weights_list[gs_idx]

            # KNN indices for this gene set's genes
            gs_knn_idx = knn_idx[gene_idx]

            # Sample control genes (vectorized)
            n_gs = len(gene_idx)
            n_neighbors = gs_knn_idx.shape[1]
            rand_idx = np.random.randint(0, n_neighbors, size=(n_gs, n_ctrl_set))
            ctrl_sampled = gs_knn_idx[np.arange(n_gs)[:, None], rand_idx].T

            # Build sparse block (vectorized)
            rows_all = ctrl_sampled.ravel()
            cols_all = np.repeat(np.arange(n_ctrl_set, dtype=np.int32), n_gs)
            data_all = np.tile(weights, n_ctrl_set)
            ctrl_block = sparse.csr_matrix(
                (data_all, (rows_all, cols_all)),
                shape=(n_genes, n_ctrl_set)
            )
            all_ctrl_blocks.append(ctrl_block)

            # Query scores
            subset = cellxgene[:, gene_idx]
            query_scores[:, gs_idx] = np.ravel(subset.multiply(weights).sum(axis=1))
            scaling_factors[gs_idx] = np.median(weights) * n_gs

        # Batched matrix multiply
        if _rust_fmr is not None:
            big_ctrl_csr = sparse.hstack(all_ctrl_blocks, format='csr')
            cellxgene_csr = cellxgene.tocsr() if not sparse.isspmatrix_csr(cellxgene) else cellxgene
            means_flat, pval_flat = _rust_fmr(
                cellxgene_csr.data.astype(np.float64, copy=False),
                cellxgene_csr.indices.astype(np.int32, copy=False),
                cellxgene_csr.indptr.astype(np.int32, copy=False),
                n_cells, cellxgene_csr.shape[1],
                big_ctrl_csr.data.astype(np.float64, copy=False),
                big_ctrl_csr.indices.astype(np.int32, copy=False),
                big_ctrl_csr.indptr.astype(np.int32, copy=False),
                big_ctrl_csr.shape[1],
                query_scores.ravel().astype(np.float64, copy=False),
                n_sets, n_ctrl_set, chunk_size, max(1, max_workers), compute_pvalues,
            )
            ctrl_means = means_flat.reshape(n_cells, n_sets)
            pval_matrix = pval_flat.reshape(n_cells, n_sets) if pval_flat is not None else None
        else:
            # Python fallback: chunked dense conversion
            big_ctrl = sparse.hstack(all_ctrl_blocks, format='csc')
            ctrl_means = np.zeros((n_cells, n_sets), dtype=np.float64)
            pval_matrix = np.zeros((n_cells, n_sets), dtype=np.float64) if compute_pvalues else None

            for c_start in range(0, n_cells, chunk_size):
                c_end = min(c_start + chunk_size, n_cells)

                chunk = cellxgene[c_start:c_end] @ big_ctrl
                if sparse.issparse(chunk):
                    chunk_dense = chunk.toarray()
                else:
                    chunk_dense = np.asarray(chunk)

                # Reshape to (chunk_size, n_sets, n_ctrl_set)
                chunk_3d = chunk_dense.reshape(c_end - c_start, n_sets, n_ctrl_set)
                ctrl_means[c_start:c_end] = chunk_3d.mean(axis=2)

                if compute_pvalues:
                    q_chunk = query_scores[c_start:c_end]
                    n_greater = np.sum(chunk_3d >= q_chunk[:, :, None], axis=2)
                    pval_matrix[c_start:c_end] = (n_greater + 1) / (n_ctrl_set + 1)

        # Background-subtracted scores
        score_matrix = (query_scores / scaling_factors[None, :]) - (ctrl_means / scaling_factors[None, :])
        return score_matrix, gene_set_names, pval_matrix

    else:
        # ================================================================
        # SINGLE GENE SET PATH — full p-value suite, stores in adata
        # ================================================================

        # Ensure gene_weights is correctly initialized
        if gene_weights is None:
            gene_weights = np.ones(len(gene_list), dtype=float)
        elif len(gene_weights) != len(gene_list):
            raise ValueError(f"Length mismatch: the input gene_weights ({len(gene_weights)}) and gene_list ({len(gene_list)}) must be the same.")

        # Only keep genes in adata.var_names
        valid_genes = set(adata.var_names)
        filtered_genes = []
        filtered_weights = []
        filtered_out_genes = []
        for gene, weight in zip(gene_list, gene_weights):
            if gene in valid_genes:
                filtered_genes.append(gene)
                filtered_weights.append(weight)
            else:
                filtered_out_genes.append(gene)

        gene_list = filtered_genes
        gene_weights = np.array(filtered_weights, dtype=float)

        n_filtered_genes = len(filtered_out_genes)
        if verbosity > 0 and n_filtered_genes > 0:
            print(f"Note: {n_filtered_genes} genes were not found in adata.var_names and are excluded from scoring: {', '.join(filtered_out_genes[:10])} {'...' if n_filtered_genes > 10 else ''}")

        cellxgene_subset = adata[:, gene_list].layers[layer] if layer is not None else adata[:, gene_list].X

        # Map gene names to indices for KNN lookup
        gene_idx = [var_names_to_idx[g] for g in gene_list]
        gene_list_knn_idx = knn_idx[gene_idx]

        # Vectorized control gene sampling
        n_genes_gs = len(gene_idx)
        n_neighbors = gene_list_knn_idx.shape[1]
        rand_idx = np.random.randint(0, n_neighbors, size=(n_genes_gs, n_ctrl_set))
        ctrl_sampled = gene_list_knn_idx[np.arange(n_genes_gs)[:, None], rand_idx].T

        # Build sparse control weight matrix (vectorized)
        rows = ctrl_sampled.ravel()
        cols = np.repeat(np.arange(n_ctrl_set, dtype=np.int32), n_genes_gs)
        data = np.tile(gene_weights, n_ctrl_set)
        ctrl_gene_weight = sparse.csr_matrix(
            (data, (rows, cols)), shape=(adata.n_vars, n_ctrl_set)
        )

        cellxgene_ctrl = cellxgene @ ctrl_gene_weight

        # Query scores
        cellxgene_query = np.ravel(cellxgene_subset.multiply(gene_weights).sum(axis=1))

        # --- P-values ---
        from statsmodels.stats.multitest import multipletests

        # Monte Carlo p-values
        n_greater = np.sum(cellxgene_ctrl >= cellxgene_query[:, None], axis=1)
        p_value_monte_carlo = np.ravel((n_greater + 1) / (n_ctrl_set + 1))
        nlog10_p_value_monte_carlo = -np.log10(p_value_monte_carlo)
        pooled_p_monte_carlo_FDR = multipletests(p_value_monte_carlo, method="fdr_bh")[1]
        nlog10_pooled_p_monte_carlo_FDR = -np.log10(pooled_p_monte_carlo_FDR)

        # Pooled empirical p-values
        pooled_p = _get_p_from_empi_null(cellxgene_query, cellxgene_ctrl.toarray().flatten())
        nlog10_pooled_p = -np.log10(pooled_p)
        pooled_p_FDR = multipletests(pooled_p, method="fdr_bh")[1]
        nlog10_pooled_p_FDR = -np.log10(pooled_p_FDR)

        # Background and final score
        BG = np.ravel(cellxgene_ctrl.mean(axis=1))
        scaling_factor = np.median(gene_weights) * len(gene_list)
        cellxgene_query = cellxgene_query / scaling_factor
        BG = BG / scaling_factor
        score_val = cellxgene_query - BG

        score_pval_res = {
            "score": score_val,
            "score_query": cellxgene_query,
            "score_ctrl_average": BG,
            "pval_mc": p_value_monte_carlo,
            "nlog10_pval_mc": nlog10_p_value_monte_carlo,
            "pval_mc_FDR": pooled_p_monte_carlo_FDR,
            "nlog10_pval_mc_FDR": nlog10_pooled_p_monte_carlo_FDR,
            "pval": pooled_p,
            "nlog10_pval": nlog10_pooled_p,
            "pval_FDR": pooled_p_FDR,
            "nlog10_pval_FDR": nlog10_pooled_p_FDR
        }

        df_score_pval_res = pd.DataFrame(index=adata.obs.index, data=score_pval_res, dtype=np.float32)

        if key_added is None:
            adata.obs['INFOG_score'] = score_val
            adata.uns['INFOG_score'] = df_score_pval_res
            if verbosity > 0:
                print(f"Finished. The scores are saved in adata.obs['INFOG_score'] and the scores, P values are saved in adata.uns['INFOG_score'].")
        else:
            adata.obs[key_added] = score_val
            adata.uns[key_added] = df_score_pval_res
            if verbosity > 0:
                print(f"Finished. The scores are saved in adata.obs['{key_added}'] and the scores, P values are saved in adata.uns['{key_added}'].")