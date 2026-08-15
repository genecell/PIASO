from typing import Optional

from ._runSVD import infog_svd
from ._normalization import score
from ._neighbors import neighbors as _piaso_neighbors
from ._leiden import leiden as _piaso_leiden

### Run GDR
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import cosg

import warnings

import os
import multiprocessing


# Sentinel for "user did not pass this argument" — distinguishes from
# explicit None. Used in runGDR to mirror score_layer to layer (and
# score_cytome_layer to cytome_layer) when the caller leaves them unset.
_LAYER_DEFAULT = object()

from ._compat import resolve_data_arg as _resolve_data_arg, _UNSET


def _sc_score_genes(adata, gene_list, score_name='score', random_state=0, **kwargs):
    """Lazy-import wrapper for scanpy's score_genes (optional dependency)."""
    import scanpy as sc
    sc.settings.verbosity = 0
    sc.tl.score_genes(adata, gene_list, score_name=score_name, random_state=random_state, **kwargs)


def _determine_parallelism(n_batches, max_workers):
    """Auto-determine (n_concurrent, threads_per_batch) for ThreadPoolExecutor.

    Strategy: prioritize inter-batch concurrency.
    - For many small batches: maximize concurrent batches, 1T each
    - For few large batches: moderate concurrency, more threads per batch
    """
    n_cores = max_workers if max_workers else (os.cpu_count() or 1)
    n_concurrent = min(n_batches, n_cores)
    threads_per_batch = max(1, n_cores // n_concurrent)
    return n_concurrent, threads_per_batch

def runGDR(
    data=_UNSET,
    batch_key: str = None,
    groupby: str = None,
    n_gene: int = 20,
    mu: float = 10.0,
    layer: str = 'infog',
    score_layer=_LAYER_DEFAULT,
    infog_layer: Optional[str] = None,
    use_highly_variable: bool = True,
    n_highly_variable_genes: int = 5000,
    n_svd_dims: int = 50,
    n_svd_iter: int = 7,
    resolution: float = 1.0,
    scoring_method: str = None,
    key_added: str = None,
    max_workers: int = 8,
    calculate_score_multiBatch: bool = True,
    n_concurrent_batches: int = None,
    verbosity: int = 0,
    random_seed: int = 1927,
    # Cytome streaming parameters
    modality: str = "RNA",
    batch_size_cytome: int = 1024,
    # Cytome on-disk persistence (cytome path only)
    write_to_cytome: bool = True,
    cytome_marker_gene_key: str = "runGDR_marker_genes",
    save_reference: bool = True,
    # ---- deprecated aliases (back-compat; see _compat / docstring) ----
    adata=_UNSET,
):
    """
    Run GDR (marker Gene-guided dimensionality reduction) on single-cell data.

    GDR performs dimensionality reduction guided by marker genes to better preserve
    biological signals. When ``max_workers > 1`` (the default), multi-batch processing
    uses parallel COSG marker identification and parallel gene-set scoring for faster
    execution. Set ``max_workers=1`` for sequential processing (useful for debugging
    or memory-constrained environments).

    Parameters
    -----------
    adata : AnnData, cytome Dataset, or str
        Annotated data matrix. Also accepts a cytome Dataset or a path to a ``.cytome`` file.
    batch_key : str, optional
        Key in `adata.obs` representing batch information. Defaults to None. If provided, marker gene identifications will be performed for each batch separately.
    groupby : str, optional
        Key in `adata.obs` to specify which cell group information to use. Defaults to None. If none, de novo clustering will be performed.
    n_gene : int, optional
        Number of genes, parameter used in COSG. Defaults to 30.
    mu : float, optional
        Gene expression specificity parameter, used in COSG. Defaults to 1.0.
    layer : str, optional
        Layer in ``adata.layers`` used for COSG marker identification. Defaults to
        ``'infog'`` — PIASO's recommended normalization for marker calling. Run
        ``piaso.tl.infog(adata)`` first to materialise this layer. Pass
        ``layer=None`` to fall back to ``adata.X`` (requires scanpy for HVG selection).
    score_layer : str, optional
        Layer in ``adata.layers`` used for gene-set scoring. Defaults to ``'infog'``
        (matches the recommended ``layer`` default — both COSG and score read the
        same INFOG-normalised matrix). Pass ``None`` to score on ``adata.X``
        directly. **Important:** for equivalence with the cytome path, AnnData
        and cytome MUST score on the same data; ``score_layer='infog'`` matches
        the cytome default ``score_cytome_layer='infog'``.
    infog_layer : str, optional
        Source layer for ``piaso.tl.infog`` when INFOG is auto-computed (only
        when ``groupby=None`` and ``layer='infog'`` triggers de novo clustering).
        ``None`` (default) → ``adata.X`` is used as the raw-counts source. If
        your ``adata.X`` is normalized, point ``infog_layer`` at the layer that
        holds raw UMI counts (e.g. ``infog_layer='counts'``).
    use_highly_variable : bool, optional
        Whether to use only highly variable genes when rerunning the dimensionality reduction. Defaults to True. Only effective when `groupby=None`.
    n_highly_variable_genes : int, optional
        Number of highly variable genes to use when `use_highly_variable` is True. Defaults to 5000. Only effective when `groupby=None`.
    n_svd_dims : int, optional
        Number of dimensions to use for SVD. Defaults to 50. Only effective when `groupby=None`.
    n_svd_iter : int, optional, default=7
        Number of iterations for randomized SVD solver. The default is larger than the default in randomized_svd to handle sparse matrices that may have large slowly decaying spectrum. Also larger than the `n_iter` default value (5) in the TruncatedSVD function.
    resolution : float, optional
        Resolution parameter for de novo clustering. Defaults to 1.0. Only effective when `groupby=None`.
    scoring_method : str, optional
        Specifies the gene set scoring method used to compute gene scores.
    key_added : str, optional
        Key under which the GDR dimensionality reduction results will be stored in `adata.obsm`. If None, results will be saved to `adata.obsm[X_gdr]`.
    max_workers : int, optional
        Maximum number of workers for parallel computation. When > 1, multi-batch
        COSG and scoring run in parallel. Defaults to 8.
    calculate_score_multiBatch : bool, optional
        .. deprecated::
            This parameter will be removed in a future version.
            Use ``max_workers=1`` for sequential processing instead.
        Whether to calculate gene scores across multiple batches in parallel.
        Defaults to True.
    n_concurrent_batches : int, optional
        Number of batches to process concurrently. If None, auto-determined from
        ``max_workers``. Default is None.
    verbosity : int, optional
        Verbosity level of the function. Higher values provide more detailed logs. Defaults to 0.
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.
    modality : str, optional
        Modality for cytome datasets. Defaults to ``'ATAC'``.
    layer : str, optional
        Layer used for COSG marker identification, on **both** backends.
        Defaults to ``'infog'`` (INFOG is the recommended normalization). For a
        cytome, run ``piaso.tl.infog(ds, save_layer=True)`` first to materialise
        the ``{modality}_infog`` matrix; pass ``layer='counts'`` for raw counts.
    score_layer : str, optional
        Layer used for gene-set scoring, on both backends. Defaults to
        ``'infog'`` and mirrors ``layer`` when left unset — keep the two equal
        for AnnData/cytome equivalence.
    batch_size_cytome : int, optional
        Batch size for cytome streaming. Defaults to 1024.
    write_to_cytome : bool, default True
        Cytome path only. If ``True``, the X_gdr embedding is persisted via
        ``ds.add_embedding('X_gdr', ...)`` and marker genes via
        ``ds.metadata[cytome_marker_gene_key]``. If ``False``, the function
        returns ``(X_gdr, marker_gene)`` without writing.
    cytome_marker_gene_key : str, default 'runGDR_marker_genes'
        Cytome path only. Metadata key under which the marker-gene table is
        stored when ``write_to_cytome=True``.

    Returns
    -------
    None or (X_gdr, marker_gene) tuple
        - AnnData path: writes to ``adata.obsm[key_added]`` and returns ``None``.
        - Cytome path with ``write_to_cytome=True`` (default): writes
          ``ds.embeddings['X_gdr']`` + marker genes to ``ds.metadata`` and
          returns ``None``.
        - Cytome path with ``write_to_cytome=False``: returns
          ``(X_gdr, marker_gene)`` for the caller to handle.

    Examples
    --------
    >>> import anndata
    >>> import piaso
    >>>
    >>> adata = anndata.read_h5ad("example.h5ad")
    >>> piaso.tl.infog(adata)  # compute INFOG normalization first
    >>> piaso.tl.runGDR(
    ...     adata,
    ...     batch_key="batch",
    ...     groupby="CellTypes",
    ...     n_gene=30,
    ...     max_workers=8,
    ...     verbosity=0
    ... )
    >>> print(adata.obsm["X_gdr"])
    """

    # Resolve the polymorphic first argument (AnnData / cytome / path); `adata=` is a
    # deprecated alias for `data`. Body below keeps using the name `adata`.
    adata = _resolve_data_arg(data, 'runGDR', adata=adata)

    # Warn if calculate_score_multiBatch is explicitly set to False
    if not calculate_score_multiBatch:
        warnings.warn(
            "calculate_score_multiBatch=False is deprecated. "
            "Use max_workers=1 for sequential processing instead.",
            DeprecationWarning, stacklevel=2,
        )

    # Resolve layer-mirror default: if the caller didn't pass score_layer
    # explicitly, mirror `layer` (works for layer='infog' → score_layer='infog',
    # layer=None → score_layer=None, layer='custom' → score_layer='custom').
    # This ensures AnnData and cytome paths score on the same data by default —
    # a common footgun before this change.
    if score_layer is _LAYER_DEFAULT:
        score_layer = layer
    # Internal cytome dispatch still uses the cytome_layer/score_cytome_layer
    # variable names; they now mirror the unified params.
    cytome_layer = layer
    score_cytome_layer = score_layer

    # Cytome dispatch
    from ._normalization import _is_cytome_dataset
    if _is_cytome_dataset(adata) or isinstance(adata, str):
        # D-5: fail loudly when scanpy method is requested on cytome.
        # The cytome streaming score path uses the piaso method
        # (KDTree control-set sampling); scanpy.tl.score_genes
        # requires a full in-memory AnnData and isn't implemented
        # for streaming.
        if scoring_method == 'scanpy':
            raise NotImplementedError(
                "runGDR(cytome, scoring_method='scanpy') is not supported. "
                "The cytome streaming path uses piaso's KDTree-based "
                "scoring; scanpy.tl.score_genes requires a full in-memory "
                "AnnData. Either: (a) pass scoring_method='piaso' (the "
                "default), or (b) export the cytome to AnnData first via "
                "ds.to_anndata(modality='RNA') and call runGDR on the AnnData."
            )
        return _runGDRParallel_cytome(
            adata, groupby=groupby, n_gene=n_gene, mu=mu,
            scoring_method=scoring_method or 'piaso',
            key_added=key_added, max_workers=max_workers,
            random_seed=random_seed, verbosity=verbosity,
            modality=modality, cytome_layer=cytome_layer,
            score_cytome_layer=score_cytome_layer,
            batch_size_cytome=batch_size_cytome,
            score_layer=score_layer,
            write_to_cytome=write_to_cytome,
            cytome_marker_gene_key=cytome_marker_gene_key,
            save_reference=save_reference,
            # Auto-cluster + multi-batch knobs forwarded for D-1 / D-4
            batch_key=batch_key,
            n_svd_dims=n_svd_dims,
            n_svd_iter=n_svd_iter,
            resolution=resolution,
        )

    ### Check the scoring method, improve this part of codes later
    if scoring_method is not None:
        valid_methods = {"scanpy", "piaso"}
        if scoring_method not in valid_methods:
            raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be one of {', '.join(valid_methods)}.")
    else:
        scoring_method = 'piaso'  # Use PIASO's scoring method as default

    # Check if key exists in adata.obs
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.columns.")
    
    if groupby is not None and groupby not in adata.obs.columns:
        raise ValueError(f"Group key '{groupby}' not found in adata.obs.columns.")
    
    # Validate layer existence
    if layer is not None and layer not in adata.layers:
        if layer == 'infog':
            raise ValueError(
                "INFOG layer not found in adata.layers. "
                "Please run piaso.tl.infog(adata) first to compute INFOG normalization, "
                "or pass layer=None to skip INFOG (requires scanpy for HVG selection)."
            )
        raise ValueError(
            f"Layer '{layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}."
        )
    
    if score_layer is not None and score_layer not in adata.layers:
        if score_layer == 'infog':
            raise ValueError(
                "score_layer='infog' was requested but adata.layers['infog'] "
                "is missing. Run piaso.tl.infog(adata) first to compute INFOG, "
                "or pass score_layer=None to score on adata.X (note: AnnData "
                "and cytome runGDR must score on the same data — for "
                "equivalence with cytome_layer='infog', use score_layer='infog')."
            )
        raise ValueError(
            f"score_layer='{score_layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}."
        )

    if infog_layer is not None and infog_layer not in adata.layers:
        raise ValueError(
            f"infog_layer='{infog_layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}. "
            f"INFOG requires raw UMI counts — point infog_layer at whichever "
            f"layer holds them. If your raw counts are in adata.X, pass "
            f"infog_layer=None (the default) instead."
        )
    
    # Remove empty log1p entry in adata.uns if it exists
    ### add this to avoid the errors in pp.highly_variable_genes and _highly_variable_genes_single_batch in scanpy
    if 'log1p' in adata.uns and (not adata.uns['log1p'] or adata.uns['log1p'] == {}):
        del adata.uns['log1p']
        if verbosity > 1:
            print("Removed empty log1p entry from adata.uns")
    
 
    try:
        # Initialize collection for marker gene scores
        score_list_collection_collection=[]

        if batch_key is None:
            nbatches=1
        else:
            batch_list=np.unique(adata.obs[batch_key])
            nbatches=len(batch_list)

        if nbatches==1:
            ### Calculate the clustering labels if there is no specified clustering labels to use
            if groupby is None:
                if verbosity > 0:
                    print("No groupby provided, performing de novo clustering")

                # Run SVD
                if verbosity > 0:
                    print(f"Running SVD with {n_svd_dims} dimensions and {n_highly_variable_genes} highly variable genes")


                ### Run SVD in a lazy mode
                infog_svd(
                    adata,
                    copy=False,
                    n_components=n_svd_dims,
                    n_top_genes=n_highly_variable_genes,
                    use_highly_variable=use_highly_variable,
                    verbosity=0,
                    batch_key=None,
                    scale_data=False,
                    n_iter=n_svd_iter,
                    layer=layer,
                    infog_layer=infog_layer,
                    infog_trim=True,
                    key_added='X_svd_TMP_GDR',
                    random_state=random_seed
                )
                # Run clustering
                if verbosity > 0:
                    print("Computing clustering")

                _piaso_neighbors(
                    adata,
                    use_rep='X_svd_TMP_GDR',
                    n_neighbors=15,
                    random_state=random_seed,
                    key_added='neighbors_TMP_GDR',
                )
                # Self-contained: leiden reads the graph from obsp via adjacency_key.
                _piaso_leiden(
                    adata,
                    resolution=resolution,
                    key_added='gdr_local_TMP_GDR',
                    adjacency_key='neighbors_TMP_GDR_connectivities',
                    random_state=random_seed,
                )
                groupby = 'gdr_local_TMP_GDR'

            if verbosity>0:
                print(f"Identified {len(np.unique(adata.obs[groupby]))} clusters.'")

            # Run marker gene identification with COSG
            if verbosity > 0:
                print(f"Identifying marker genes using COSG (mu={mu})")

            ### Run marker gene identification
            cosg_params = {
                'key_added': 'cosg_TMP_GDR',
                'mu': mu,
                'expressed_pct': 0.1,
                'remove_lowly_expressed': True,
                'n_genes_user': n_gene,
                'groupby': groupby
            }

            if layer is not None:
                cosg_params['use_raw'] = False
                cosg_params['layer'] = layer

            cosg.cosg(adata, **cosg_params)

            marker_gene=pd.DataFrame(adata.uns['cosg_TMP_GDR']['names'])


            # Calculate scores
            if verbosity > 0:
                print(f"Calculating gene scores using '{scoring_method}' method")


            ### Calculate scores
            score_list_collection=[]

            if scoring_method == 'piaso':
                from ._normalization import score as _score
                score_list, _, _ = _score(
                    adata, gene_list=marker_gene,
                    layer=score_layer, random_seed=random_seed,
                    compute_pvalues=False,
                )
            elif scoring_method == 'scanpy':
                score_list = []
                adata_tmp = adata.copy()
                if score_layer is not None:
                    adata_tmp.X = adata_tmp.layers[score_layer]
                for i in marker_gene.columns:
                    marker_gene_i = marker_gene[i].values
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        _sc_score_genes(adata_tmp, marker_gene_i, score_name='markerGeneFeatureScore_i', random_state=random_seed)
                    score_list.append(adata_tmp.obs['markerGeneFeatureScore_i'].values.copy())
                score_list = np.vstack(score_list).T
            else:
                raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be either 'scanpy' or 'piaso'.")
            ### Normalization
            score_list=normalize(score_list,norm='l2',axis=0)
            score_list=normalize(score_list,norm='l2',axis=1) ## Adding this is important
            score_list_collection.append(score_list)


            score_list_collection=np.vstack(score_list_collection)
            score_list_collection_collection.append(score_list_collection)

            marker_gene_scores=np.hstack(score_list_collection_collection)


            ### Make sure the order are matched to the adata
            marker_gene_scores=pd.DataFrame(marker_gene_scores)
            marker_gene_scores.index=adata.obs_names
            # marker_gene_scores.index=np.hstack([adata_list[0].obs_names.values, adata_list[1].obs_names.values])
            marker_gene_scores=marker_gene_scores.loc[adata.obs_names]

        ### Have multiple batches
        else:
            _use_parallel = max_workers > 1

            if _use_parallel:
                ### Parallel path: use runCOSGParallel + calculateScoreParallel_multiBatch
                marker_gene, batch_n_groups = runCOSGParallel(
                    adata,
                    batch_key=batch_key,
                    groupby=groupby,
                    n_gene=n_gene,
                    mu=mu,
                    use_highly_variable=use_highly_variable,
                    n_highly_variable_genes=n_highly_variable_genes,
                    layer=layer,
                    infog_layer=infog_layer,
                    n_svd_dims=n_svd_dims,
                    n_svd_iter=n_svd_iter,
                    resolution=resolution,
                    verbosity=verbosity,
                    return_gene_names=True,
                    max_workers=max_workers,
                    random_seed=random_seed,
                )

                batch_n_groups_indices = np.cumsum([0] + batch_n_groups)

                if calculate_score_multiBatch:
                    score_list_collection, cellbarcode_info, gene_set_names_collection = calculateScoreParallel_multiBatch(
                        adata,
                        batch_key=batch_key,
                        marker_gene=marker_gene,
                        marker_gene_n_groups_indices=batch_n_groups_indices,
                        score_layer=score_layer,
                        max_workers=max_workers,
                        n_concurrent_batches=n_concurrent_batches,
                        score_method=scoring_method,
                        random_seed=random_seed,
                    )
                else:
                    from tqdm import tqdm
                    score_list_collection = []
                    cellbarcode_info = list()
                    for batch_u in tqdm(batch_list, desc="Calculating cell embeddings", unit="batch"):
                        cellbarcode_info.append(adata.obs_names[adata.obs[batch_key] == batch_u].values)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            adata_batch_u = adata[adata.obs[batch_key] == batch_u].copy()
                        score_list, gene_set_names = calculateScoreParallel(
                            adata_batch_u, gene_set=marker_gene,
                            score_method=scoring_method, score_layer=score_layer,
                            max_workers=max_workers, random_seed=random_seed,
                        )
                        score_list = normalize(score_list, norm='l2', axis=0)
                        for start, end in zip(batch_n_groups_indices[:-1], batch_n_groups_indices[1:]):
                            score_list[:, start:end] = normalize(score_list[:, start:end], norm='l2', axis=1)
                        score_list_collection.append(score_list)

                score_list_collection = np.vstack(score_list_collection)
                marker_gene_scores = pd.DataFrame(score_list_collection)
                marker_gene_scores.index = np.hstack(cellbarcode_info)
                marker_gene_scores = marker_gene_scores.loc[adata.obs_names]

            else:
                ### Sequential path (max_workers=1): process batches one by one
                cellbarcode_info = list()
                for batch_idx, batch_i in enumerate(batch_list):
                    if verbosity > 0:
                        print(f"Processing batch {batch_idx+1}/{nbatches}: '{batch_i}'")

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        adata_i = adata[adata.obs[batch_key] == batch_i].copy()

                    if groupby is None:
                        infog_svd(
                            adata_i, copy=False, n_components=n_svd_dims,
                            n_top_genes=n_highly_variable_genes,
                            use_highly_variable=use_highly_variable, verbosity=0,
                            batch_key=None, scale_data=False, n_iter=n_svd_iter,
                            layer=layer, infog_layer=infog_layer, infog_trim=True,
                            key_added='X_svd', random_state=random_seed,
                        )
                        _piaso_neighbors(adata_i, use_rep='X_svd', n_neighbors=15, random_state=random_seed)
                        _piaso_leiden(adata_i, resolution=resolution, key_added='gdr_local', random_state=random_seed)
                        groupby_i = 'gdr_local'
                    else:
                        groupby_i = groupby

                    if verbosity > 0:
                        print(f"Identified {len(np.unique(adata_i.obs[groupby_i]))} clusters in batch '{batch_i}'")

                    cellbarcode_info.append(adata_i.obs_names.values)

                    cosg_params = {
                        'key_added': 'cosg', 'mu': mu, 'expressed_pct': 0.1,
                        'remove_lowly_expressed': True, 'n_genes_user': n_gene,
                        'groupby': groupby_i,
                    }
                    if layer is not None:
                        cosg_params['use_raw'] = False
                        cosg_params['layer'] = layer
                    cosg.cosg(adata_i, **cosg_params)
                    marker_gene = pd.DataFrame(adata_i.uns['cosg']['names'])

                    score_list_collection = []
                    for batch_u in batch_list:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            adata_u = adata[adata.obs[batch_key] == batch_u].copy()
                        if scoring_method == 'piaso':
                            from ._normalization import score as _score
                            score_list, _, _ = _score(
                                adata_u, gene_list=marker_gene,
                                layer=score_layer, random_seed=random_seed,
                                compute_pvalues=False,
                            )
                        elif scoring_method == 'scanpy':
                            score_list = []
                            if score_layer is not None:
                                adata_u.X = adata_u.layers[score_layer]
                            for i in marker_gene.columns:
                                marker_gene_i = marker_gene[i].values
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=FutureWarning)
                                    _sc_score_genes(adata_u, marker_gene_i, score_name='markerGeneFeatureScore_i', random_state=random_seed)
                                score_list.append(adata_u.obs['markerGeneFeatureScore_i'].values.copy())
                            score_list = np.vstack(score_list).T
                        else:
                            raise ValueError(f"Invalid scoring_method: '{scoring_method}'.")
                        score_list = normalize(score_list, norm='l2', axis=0)
                        score_list = normalize(score_list, norm='l2', axis=1)
                        score_list_collection.append(score_list)

                    score_list_collection = np.vstack(score_list_collection)
                    score_list_collection_collection.append(score_list_collection)

                marker_gene_scores = np.hstack(score_list_collection_collection)
                marker_gene_scores = pd.DataFrame(marker_gene_scores)
                marker_gene_scores.index = np.hstack(cellbarcode_info)
                marker_gene_scores = marker_gene_scores.loc[adata.obs_names]

        ### Set the low-dimensional representations
        if key_added is not None:
            output_key = key_added
        else:
            output_key = 'X_gdr'

        adata.obsm[output_key] = marker_gene_scores.values

        # Store metadata about the GDR run
        adata.uns['gdr'] = {
            'params': {
                'n_gene': n_gene,
                'mu': mu,
                'layer': layer,
                'score_layer': score_layer,
                'infog_layer': infog_layer,
                'scoring_method': scoring_method,
                'random_seed': random_seed
            }
        }



        # Record the frozen-reference recipe so `piaso.tl.projectGDR` can map new cells into this
        # space. Only the cheap parts are stored here (marker sets, block structure, layers, seed);
        # the column norms and control-gene KNN are completed on first projectGDR call and cached
        # back, so runGDR costs nothing extra.
        if save_reference:
            try:
                from ._projectGDR import _make_reference_recipe
                _blk = batch_n_groups_indices if (
                    nbatches > 1 and 'batch_n_groups_indices' in dir()) else None
                adata.uns['gdr_reference'] = _make_reference_recipe(
                    marker_gene,
                    block_indices=_blk,
                    layer=score_layer if score_layer is not None else layer,
                    groupby=(None if groupby == 'gdr_local_TMP_GDR' else groupby),
                    batch_key=batch_key,
                    random_seed=random_seed,
                    denovo_labels=('gdr_local' if groupby == 'gdr_local_TMP_GDR' else None),
                    modality=modality,
                )
                if verbosity > 0:
                    print("Reference state saved to adata.uns['gdr_reference'] "
                          "(use piaso.tl.projectGDR to map new cells into this space)")
            except Exception as _e:
                warnings.warn(f"runGDR: could not record gdr_reference ({_e})", RuntimeWarning)

        # Clean up intermediate data if batch_key is None and we performed de novo clustering
        if nbatches == 1 and groupby == 'gdr_local_TMP_GDR':
            # Remove intermediate SVD result
            if 'X_svd_TMP_GDR' in adata.obsm:
                del adata.obsm['X_svd_TMP_GDR']
                if verbosity > 1:
                    print("Removed temporary X_svd_TMP_GDR from adata.obsm")


            # Remove temporary neighbors data
            if 'neighbors_TMP_GDR' in adata.uns:
                del adata.uns['neighbors_TMP_GDR']
                if verbosity > 1:
                    print("Removed temporary neighbors_TMP_GDR data from adata.uns")


            # Keep the de-novo cluster labels: the GDR dimensions ARE these clusters, so without
            # them a projected coordinate has no interpretable name. Renamed to drop the TMP suffix.
            if 'gdr_local_TMP_GDR' in adata.obs.columns:
                adata.obs['gdr_local'] = adata.obs['gdr_local_TMP_GDR'].values
                del adata.obs['gdr_local_TMP_GDR']
                if verbosity > 1:
                    print("Kept de-novo cluster labels as adata.obs['gdr_local']")

        # Clean up the COSG results if batch_key is None            
        if nbatches == 1:           
            # Remove intermediate COSG result
            if 'cosg_TMP_GDR' in adata.uns:
                del adata.uns['cosg_TMP_GDR']
                if verbosity > 1:
                    print("Removed temporary COSG_TMP_GDR results from adata.uns")

        print(f"GDR embeddings saved to adata.obsm['{output_key}']")
    except Exception as e:
        raise e # Re-raise the error after cleanup

########################################
###### Codes for running GDR in Parallel
########################################
from multiprocessing import shared_memory
from scipy.sparse import csr_matrix
import numpy as np

def _setup_shared_memory_sparse(csr_matrix):
    """
    Set up shared memory for a sparse CSR matrix.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        The input sparse matrix.

    Returns
    -------
    dict
        A dictionary containing shared memory objects, shapes, dtypes, and metadata
        required for reconstructing the matrix in worker processes.
    """
    # Extract components
    data, indices, indptr = csr_matrix.data, csr_matrix.indices, csr_matrix.indptr

    # Create shared memory for each component
    shm_data = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)
    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)

    # Copy data directly into shared memory
    np.copyto(np.ndarray(data.shape, dtype=data.dtype, buffer=shm_data.buf), data)
    np.copyto(np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf), indices)
    np.copyto(np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf), indptr)

    # Return shared memory objects and metadata
    return {
        "shm_data": shm_data,
        "shm_indices": shm_indices,
        "shm_indptr": shm_indptr,
        "shapes": {
            "data_shape": data.shape,
            "indices_shape": indices.shape,
            "indptr_shape": indptr.shape,
            "matrix_shape": csr_matrix.shape
        },
        "dtypes": {
            "data_dtype": data.dtype,
            "indices_dtype": indices.dtype,
            "indptr_dtype": indptr.dtype
        }
    }



def _setup_shared_memory_dense(matrix):
    """
    Set up shared memory for a dense matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The input dense matrix.

    Returns
    -------
    dict
        A dictionary containing the shared memory object, shape, and dtype of the matrix.
    """
    shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
    shared_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
    np.copyto(shared_matrix, matrix)

    return {"shm": shm, "shape": matrix.shape, "dtype": matrix.dtype}


from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.sparse import isspmatrix_csr

def _process_gene_sets(gene_set, var_names):
    """
    Process gene sets to filter valid genes and map them to indices in var_names.

    Parameters
    ----------
    gene_set : dict, list of lists, or pandas.DataFrame
        A collection of gene sets, where each gene set can be:
            - A dictionary: Keys are gene set names, values are lists of genes.
            - A list of lists: Each sublist contains genes in a gene set.
            - A pandas.DataFrame: Each column represents a gene set, and column names are gene set names.
    var_names : pd.Index
        The gene names in `adata.var.index`.

    Returns
    -------
    dict
        A dictionary where keys are gene set names (or indices for lists) and values are lists of indices.
    """
    valid_gene_sets = {}
    var_names_dict = {gene: idx for idx, gene in enumerate(var_names)}

    if isinstance(gene_set, dict):
        for name, genes in gene_set.items():
            valid_indices = [var_names_dict[gene] for gene in genes if gene in var_names_dict]
            if valid_indices:
                valid_gene_sets[name] = valid_indices
    elif isinstance(gene_set, list):
        for idx, genes in enumerate(gene_set):
            valid_indices = [var_names_dict[gene] for gene in genes if gene in var_names_dict]
            if valid_indices:
                valid_gene_sets[f"GeneSet_{idx}"] = valid_indices
    elif isinstance(gene_set, pd.DataFrame):
        for col in gene_set.columns:
            valid_indices = [var_names_dict[gene] for gene in gene_set[col].dropna() if gene in var_names_dict]
            if valid_indices:
                valid_gene_sets[col] = valid_indices
    else:
        raise ValueError("gene_set must be a dictionary, list of lists, or pandas.DataFrame.")

    return valid_gene_sets


def _calculate_gene_set_score_shared(gene_indices, metadata, score_name, is_sparse, score_method, random_seed):
    """
    Worker function to calculate gene set score using shared memory.

    Parameters
    ----------
    gene_indices : list of int
        The adata var's cooresponding indices of the genes in the gene set to score.
    metadata : dict
        Metadata containing shared memory names, shapes, and dtypes.
    score_name : str
        The score name used when scoring the gene set among cells.
    is_sparse : bool
        Whether the input matrix is sparse.
    score_method : {'scanpy', 'piaso'}, optional
        The method used for gene set scoring. Must be either 'scanpy' (default) or 'piaso'.
        - 'scanpy': Uses the Scanpy's built-in gene set scoring method.
        - 'piaso': Uses the PIASO's gene set scoring method, which is more robust to sequencing depth variations.
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.
        
    Returns
    -------
    tuple or np.ndarray
        If score_method is 'piaso', returns a tuple of (scores, p_values).
        If score_method is 'scanpy', returns only scores for the gene set.
    """
    
    # Force single-threaded mode inside the worker, to prevent OpenMP from trying to create threads inside a process
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    import numpy as np
    np.random.seed(random_seed) # Explicitly seed the worker
    
    
    import warnings
    
    # Back up the current verbosity level
    
    
    try:
        # print(is_sparse)
        ### The gene_set parameter should be placed at the first position, because we are using the partial function
        if is_sparse:
  
            from multiprocessing import shared_memory
            import numpy as np
            from scipy.sparse import csr_matrix

            # Access shared memory
            shm_data = shared_memory.SharedMemory(name=metadata["shm_data"].name, create=False)
            shm_indices = shared_memory.SharedMemory(name=metadata["shm_indices"].name, create=False)
            shm_indptr = shared_memory.SharedMemory(name=metadata["shm_indptr"].name, create=False)


            # Reconstruct arrays
            data = np.ndarray(metadata["shapes"]["data_shape"], dtype=metadata["dtypes"]["data_dtype"], buffer=shm_data.buf)
            indices = np.ndarray(metadata["shapes"]["indices_shape"], dtype=metadata["dtypes"]["indices_dtype"], buffer=shm_indices.buf)
            indptr = np.ndarray(metadata["shapes"]["indptr_shape"], dtype=metadata["dtypes"]["indptr_dtype"], buffer=shm_indptr.buf)


            X = csr_matrix((data, indices, indptr), shape=metadata["shapes"]["matrix_shape"])


        else:
            from multiprocessing import shared_memory
            import numpy as np
            shm = shared_memory.SharedMemory(name=metadata["shm"].name, create=False)
            X = np.ndarray(metadata["shape"], dtype=metadata["dtype"], buffer=shm.buf)

        from anndata import AnnData
        adata_tmp = AnnData(X=X)
        # warnings.filterwarnings("ignore", category=FutureWarning)
        # # # Suppress FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if score_method=='scanpy':
                _sc_score_genes(adata_tmp, adata_tmp.var.index[gene_indices].tolist(), score_name=score_name, random_state=random_seed)
                
                
                return adata_tmp.obs[score_name].values.copy()
                
            elif score_method=='piaso':
                ## Set layer to None, because the scoring layer is already constructed as the adata.X
                score(adata_tmp, gene_list=adata_tmp.var.index[gene_indices].tolist(), key_added=score_name, layer=None, random_seed=random_seed)
                
                                # Get both scores and -log10(p-values)
                scores = adata_tmp.obs[score_name].values.copy()
                nlog10_pvals = None
                if score_name in adata_tmp.uns and 'nlog10_pval' in adata_tmp.uns[score_name]:
                    nlog10_pvals = adata_tmp.uns[score_name]['nlog10_pval'].copy()
                
                return scores, nlog10_pvals
                
            else:
                raise ValueError(f"Invalid score_method: '{score_method}'. Must be either 'scanpy' or 'piaso'.")

    
    finally:
        # Clean up shared memory
        if is_sparse:
            metadata["shm_data"].close()
            metadata["shm_indices"].close()
            metadata["shm_indptr"].close()
        else:
            metadata["shm"].close()
            
        # Restore the original verbosity level


def _safe_calculate_gene_set_score_shared(gene_indices, metadata, score_name, is_sparse):
    try:
        return _calculate_gene_set_score_shared(gene_indices, metadata, score_name, is_sparse)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Worker encountered an error: {e}")



# from scipy.sparse import isspmatrix_csr
from scipy.sparse import issparse
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from typing import Literal, Union, Optional
from tqdm import tqdm

def calculateScoreParallel(
    adata,
    gene_set: Union[dict, list, pd.DataFrame],
    score_method: Literal["scanpy", "piaso"] = "piaso",
    random_seed: int = 1927,
    score_layer: Optional[str] = None,
    max_workers: Optional[int] = None,
    return_pvals: bool = False,
    precomputed_knn: np.ndarray = None,
    verbosity: int = 0,
    # Cytome streaming parameters
    modality: str = "RNA",
    cytome_layer: str = "counts",
    batch_size: int = 1024,
    cell_mask=None,
):
    """
    Compute gene set scores in parallel using shared memory for efficiency.

    This function processes multiple gene sets in parallel, computing enrichment scores
    for each gene set across all cells in the AnnData object. When using the 'piaso'
    scoring method, it uses a vectorized batched approach (score() with multi-set mode)
    that precomputes gene-level statistics once and scores all gene sets in a single pass,
    which is significantly faster and more memory-efficient than scoring each set independently.
    For the 'scanpy' method, it uses shared memory to pass the expression matrix
    to worker processes.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing gene expression data.
    gene_set : dict, list of lists, or pandas.DataFrame
        A collection of gene sets to score. Supported formats:
            - dict: Keys are gene set names, values are lists of gene names.
            - list of lists: Each sublist contains gene names for one gene set.
              Gene sets will be named "GeneSet_0", "GeneSet_1", etc.
            - pandas.DataFrame: Each column represents a gene set, with column names
              as gene set names and gene names as values.
    score_method : {'scanpy', 'piaso'}, default 'piaso'
        The method used for gene set scoring.
        - 'scanpy': Uses Scanpy's built-in gene set scoring method.
        - 'piaso': Uses the PIASO's gene set scoring method, which is more robust to sequencing
          depth variations and provides p-values.
    random_seed : int, default 1927
        Random seed for reproducibility.
    score_layer : str or None, default None
        Layer of the AnnData object to use. If None, `adata.X` is used.
    max_workers : int or None, default None
        Number of parallel worker processes to use. If None, defaults to the number
        of CPU cores available. Only used when score_method='scanpy'.
    return_pvals : bool, default False
        Whether to return -log10(p-values) when using 'piaso' method. Only applicable
        when score_method='piaso'. If True, returns a third array containing p-values.
    verbosity : int, default 0
        Level of verbosity for progress reporting.
        - 0: Silent (no progress bar)
        - >0: Show progress bar during parallel computation


    Returns
    -------
    score_matrix : np.ndarray
        A 2D array of shape (n_cells, n_gene_sets) where each column contains
        the scores for one gene set across all cells.
    gene_set_names : list of str
        The names of the gene sets, in the same order as columns in score_matrix.
    nlog10_pval_matrix : np.ndarray, optional
        Only returned when score_method='piaso' and return_pvals=True.
        A 2D array of shape (n_cells, n_gene_sets) containing -log10(p-values)
        for each gene set score. Returns None if p-values are not available.

    Examples
    --------
    >>> import anndata
    >>> import numpy as np
    >>> import piaso
    >>>
    >>> # Load example data
    >>> adata = anndata.read_h5ad('pbmc3k.h5ad')
    >>>
    >>> # Define gene sets
    >>> gene_sets = {
    ...     'T_cell_markers': ['CD3D', 'CD3E', 'CD8A'],
    ...     'B_cell_markers': ['CD79A', 'CD79B', 'MS4A1']
    ... }
    >>>
    >>> # Compute scores using Scanpy method
    >>> scores, names = piaso.tl.calculateScoreParallel(
    ...     adata,
    ...     gene_set=gene_sets,
    ...     score_method='piaso',
    ...     verbosity=1
    ... )
    >>>
    >>> # Add scores to AnnData object
    >>> for i, name in enumerate(names):
    ...     adata.obs[f'{name}_score'] = scores[:, i]
    """

    # Validate score_method
    if score_method not in {"scanpy", "piaso"}:
        raise ValueError(f"Invalid score_method: '{score_method}'. Must be either 'scanpy' or 'piaso'.")

    # --- Vectorized batched scoring path for piaso method ---
    # Precomputes mean/var/KDTree once and scores all gene sets in a single
    # batched matrix multiply. ~11-14x faster and ~100x less RAM than the
    # per-set ProcessPoolExecutor path.
    if score_method == 'piaso':
        from ._normalization import score as _score, _is_cytome_dataset
        # Round 26: score() now has a unified `layer` for both backends — use the
        # cytome measurement for a cytome, the AnnData layer otherwise (no more
        # deprecated cytome_layer= which would warn on every internal call).
        _score_layer = cytome_layer if _is_cytome_dataset(adata) else score_layer
        score_matrix, gene_set_names, pval_matrix = _score(
            adata, gene_list=gene_set,
            compute_pvalues=return_pvals,
            layer=_score_layer, random_seed=random_seed,
            n_ctrl_set=100, max_workers=max_workers if max_workers is not None else 1,
            precomputed_knn=precomputed_knn,
            verbosity=verbosity,
            modality=modality, batch_size=batch_size,
            cell_mask=cell_mask,
        )
        if return_pvals:
            # Convert raw p-values to -log10 format to match original score() behavior
            nlog10_pval_matrix = -np.log10(pval_matrix) if pval_matrix is not None else None
            return score_matrix, gene_set_names, nlog10_pval_matrix
        return score_matrix, gene_set_names

    # --- Original multi-process path (used for scanpy method) ---
    # Scanpy method requires AnnData
    from ._normalization import _is_cytome_dataset
    if _is_cytome_dataset(adata):
        raise ValueError("scanpy scoring method requires AnnData input, not cytome dataset. Use score_method='piaso'.")
    # Determine the input matrix
    if score_layer is not None:
        data = adata.layers[score_layer]
    else:
        data = adata.X

    # Determine matrix type and set up shared memory
    if issparse(data):
        if not isinstance(data, csr_matrix):
            raise ValueError("For the gene expression matrix, if you want to use sparse matrix, the format must be in CSR format.")
        shm_metadata = _setup_shared_memory_sparse(data)
        is_sparse = True
    else:
        shm_metadata = _setup_shared_memory_dense(data)
        is_sparse = False
        
        
     
    # Preprocess gene sets to map to indices, and only keep the genes in the adata.var.index
    valid_gene_sets = _process_gene_sets(gene_set, adata.var.index)
    
    if not valid_gene_sets:
        raise ValueError(
            "No valid gene sets found. Ensure that gene names in gene_set match "
            "gene names in adata.var.index."
        )
    
    # ### need to add list() to valid_gene_sets.values(), otherwise, the dict's value is dict_value object, which is an iterable view, not a standard list.
    # valid_gene_sets_indices=list(valid_gene_sets.values())
    
    
    # Extract names and indices together to ensure alignment
    gene_set_names, valid_gene_sets_indices = zip(*valid_gene_sets.items()) if valid_gene_sets else ([], [])
    gene_set_names = list(gene_set_names)
    valid_gene_sets_indices = list(valid_gene_sets_indices)

    # Prepare partial function
    partial_func = partial(
        _calculate_gene_set_score_shared, ### also could uses _safe_calculate_gene_set_score_shared
        metadata=shm_metadata,
        score_name="geneSetScore_i", ### this is actually redundant, we don't need this
        is_sparse=is_sparse,
        score_method=score_method, ### Specify which gene set scoring method to use
        random_seed=random_seed ### Set the random seed for reproducibility
    )
    
    try:
        # Use spawn context
        ctx = multiprocessing.get_context('spawn')
        

        if verbosity > 0:
            total_gene_sets = len(valid_gene_sets_indices)
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                results = list(tqdm(
                    executor.map(partial_func, valid_gene_sets_indices),
                    total=total_gene_sets,
                    desc="Scoring gene sets",
                    unit="set"
                ))
        else:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                results = list(executor.map(partial_func, valid_gene_sets_indices))

    finally:
        # Robust Memory Cleanup
        if is_sparse:
            shm_metadata["shm_data"].close()
            shm_metadata["shm_data"].unlink()
            shm_metadata["shm_indices"].close()
            shm_metadata["shm_indices"].unlink()
            shm_metadata["shm_indptr"].close()
            shm_metadata["shm_indptr"].unlink()
        else:
            shm_metadata["shm"].close()
            shm_metadata["shm"].unlink()
    
    
    # Process results — only scanpy method reaches here (piaso uses fast path above)
    score_matrix = np.vstack(results).T
    return score_matrix, gene_set_names

### Calculate gene set score for different batches, separately, but in parallel
def _calculateScoreParallel_single_batch(batch_key, shared_data, batch_i, marker_gene, marker_gene_n_groups_indices, max_workers, score_method, random_seed):
    """
    Process a single batch to calculate scores, different marker gene sets will be calculated in parallel with `calculateScoreParallel` function. Note: max_workers here refers to INNER workers passed from the parent.
    """
    # Force single threading for linear algebra to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"

    # Reconstruct matrix from shared memory
    if 'shm_indices' in shared_data:
        data = np.ndarray(shared_data['shapes']['data_shape'], dtype=shared_data['dtypes']['data_dtype'], buffer=shared_data['shm_data'].buf)
        indices = np.ndarray(shared_data['shapes']['indices_shape'], dtype=shared_data['dtypes']['indices_dtype'], buffer=shared_data['shm_indices'].buf)
        indptr = np.ndarray(shared_data['shapes']['indptr_shape'], dtype=shared_data['dtypes']['indptr_dtype'], buffer=shared_data['shm_indptr'].buf)
        matrix = csr_matrix((data, indices, indptr), shape=shared_data['shapes']['matrix_shape'])
    else:
        matrix = np.ndarray(shared_data['shapes']['matrix_shape'], dtype=shared_data['dtypes']['data_dtype'], buffer=shared_data['shm_data'].buf)

    from anndata import AnnData
    batch_mask = shared_data['obs'][batch_key] == batch_i
    adata = AnnData(matrix[batch_mask.to_numpy()])
    adata.obs = shared_data['obs'][batch_mask.to_numpy()].copy()
    adata.var_names = shared_data["var_names"].copy()

    # Compute gene set scores, in parallel for different gene sets
    # Use the passed max_workers
    score_list, gene_set_names = calculateScoreParallel(
        adata,
        gene_set=marker_gene,
        score_method=score_method,
        score_layer=None, ## As the score layer already used in setting up the shared memory
        max_workers=max_workers, 
        random_seed=random_seed
    )


    score_list = normalize(score_list, norm="l2", axis=0)
    for start, end in zip(marker_gene_n_groups_indices[:-1], marker_gene_n_groups_indices[1:]):
        score_list[:, start:end] = normalize(score_list[:, start:end], norm="l2", axis=1)
        
    cell_barcodes = adata.obs_names.values
    ### Return batch_i at the start of the tuple
    return batch_i, score_list, cell_barcodes, gene_set_names


from typing import Literal
from concurrent.futures import ThreadPoolExecutor

def calculateScoreParallel_multiBatch(
    adata,
    batch_key: str,
    marker_gene: pd.DataFrame,
    marker_gene_n_groups_indices: list,
    score_method: Literal["scanpy", "piaso"],
    score_layer: str = None,
    max_workers: int = 8,
    n_concurrent_batches: int = None,
    random_seed: int = 1927
):
    """
    Calculate gene set scores for each adata batch in parallel using shared memory. Different marker gene sets will be calculated in parallel as well.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str
        The key in `adata.obs` used to identify batches.
    marker_gene : DataFrame
        The marker gene DataFrame.
    marker_gene_n_groups_indices : list
        Indices specifying the marker gene set group boundaries, used for score normalization within each marker gene set group.
    max_workers : int
        Maximum number of parallel workers to use (total threads).
    score_layer : str
        The layer of `adata` to use for scoring.
    score_method : {'scanpy', 'piaso'}, optional
        The method used for gene set scoring. Must be either 'scanpy' (default) or 'piaso'.
        - 'scanpy': Uses the Scanpy's built-in gene set scoring method.
        - 'piaso': Uses the PIASO's gene set scoring method, which is more robust to sequencing depth variations.
    n_concurrent_batches : int, optional
        Number of batches to process concurrently via ThreadPoolExecutor.
        If None, auto-determined based on max_workers and number of batches.
        Only used when score_method='piaso'. Default is None.
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.

    Returns
    -------
    tuple
        - list: A list of normalized score arrays for each batch.
        - list: A list of cell barcodes for each batch.
        - list: A list of gene set names.

    Examples
    --------
    >>> import anndata
    >>> import piaso
    >>> adata = anndata.read_h5ad('example_data.h5ad')
    >>> score_list, cellbarcode_info, gene_set_names = piaso.tl.calculateScoreParallel_multiBatch(
    ...     adata=adata,
    ...     batch_key='batch',
    ...     marker_gene=marker_gene,
    ...     marker_gene_n_groups_indices=marker_gene_n_groups_indices,
    ...     score_layer='piaso',
    ...     max_workers=8
    ... )
    >>> print(score_list)
    >>> print(cellbarcode_info)
    """

    batch_order_map = {batch: i for i, batch in enumerate(np.unique(adata.obs[batch_key]))}
    batch_list = list(batch_order_map.keys())

    score_list_collection = [None] * len(batch_list)
    cellbarcode_info = [None] * len(batch_list)
    gene_set_names_collection = [None] * len(batch_list)

    # --- Fast parallel path for piaso method ---
    # Uses ThreadPoolExecutor for inter-batch parallelism. Both sklearn KDTree
    # and Rust score_complete release the GIL, enabling true parallel execution.
    if score_method == 'piaso':
        if n_concurrent_batches is not None:
            n_concurrent = min(n_concurrent_batches, len(batch_list))
            threads_per_batch = max(1, (max_workers or os.cpu_count() or 1) // n_concurrent)
        else:
            n_concurrent, threads_per_batch = _determine_parallelism(len(batch_list), max_workers)

        def _score_one_batch(batch_idx_and_name):
            idx, batch_i = batch_idx_and_name
            batch_mask = adata.obs[batch_key] == batch_i
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                adata_batch = adata[batch_mask].copy()

            score_list, gene_set_names = calculateScoreParallel(
                adata_batch,
                gene_set=marker_gene,
                score_method='piaso',
                score_layer=score_layer,
                max_workers=threads_per_batch,
                random_seed=random_seed,
            )

            score_list = normalize(score_list, norm="l2", axis=0)
            for start, end in zip(marker_gene_n_groups_indices[:-1], marker_gene_n_groups_indices[1:]):
                score_list[:, start:end] = normalize(score_list[:, start:end], norm="l2", axis=1)

            return idx, score_list, adata_batch.obs_names.values, gene_set_names

        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = [executor.submit(_score_one_batch, (i, b)) for i, b in enumerate(batch_list)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating cell embeddings/scores", unit="batch"):
                idx, scores, barcodes, names = future.result()
                score_list_collection[idx] = scores
                cellbarcode_info[idx] = barcodes
                gene_set_names_collection[idx] = names

        return score_list_collection, cellbarcode_info, gene_set_names_collection

    # --- Original ProcessPoolExecutor path (scanpy method) ---
    # Extract gene expression data
    if score_layer is not None:
        gene_expression_data = adata.layers[score_layer]
    else:
        gene_expression_data = adata.X

    # Set up shared memory
    if issparse(gene_expression_data):
        if not isinstance(gene_expression_data, csr_matrix):
            raise ValueError("Sparse matrix must be in CSR format.")
        shared_data = _setup_shared_memory_sparse(gene_expression_data)
    else:
        shared_data = _setup_shared_memory_dense(gene_expression_data)

    shared_data["obs"] = adata.obs[[batch_key]].copy()
    shared_data["var_names"] = adata.var_names.copy()

    # --- Efficiency Calculation ---
    num_batches = len(batch_list)
    actual_outer_workers = min(num_batches, max_workers)

    # Calculate remaining cores for the Inner Loop
    total_cores = multiprocessing.cpu_count()
    # Ensure at least 1 worker
    inner_workers = max(1, total_cores // max(1, actual_outer_workers))
    # Cap inner workers to reasonable limit (e.g., 4) to avoid overhead if not needed
    inner_workers = min(inner_workers, 4)
    # -----------------------------------

    try:
        # Use spawn context
        ctx = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(max_workers=actual_outer_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    _calculateScoreParallel_single_batch,
                    batch_key,
                    shared_data,
                    batch_i,
                    marker_gene,
                    marker_gene_n_groups_indices,
                    inner_workers, # Pass the calculated inner workers
                    score_method,
                    random_seed
                ) for batch_i in batch_list
            ]

            # Collect raw results
            raw_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating cell embeddings/scores", unit="batch"):
                raw_results.append(future.result())

        # Sort results
        raw_results.sort(key=lambda x: batch_order_map[x[0]])

        # Unpack safely in the correct order (indexed assignment, not append)
        for batch_name, score_list, cell_barcodes, gene_names in raw_results:
            idx = batch_order_map[batch_name]
            score_list_collection[idx] = score_list
            cellbarcode_info[idx] = cell_barcodes
            gene_set_names_collection[idx] = gene_names

    finally:
        # Clean up shared memory — key is "shm_data" for sparse, "shm" for dense
        if 'shm_data' in shared_data:
            shared_data['shm_data'].close()
            shared_data['shm_data'].unlink()
        elif 'shm' in shared_data:
            shared_data['shm'].close()
            shared_data['shm'].unlink()
        if 'shm_indices' in shared_data:
            shared_data['shm_indices'].close()
            shared_data['shm_indices'].unlink()
            shared_data['shm_indptr'].close()
            shared_data['shm_indptr'].unlink()

    return score_list_collection, cellbarcode_info, gene_set_names_collection




#### Function to process the runCOSGParallel in each individual batches, and the shared memory will be used
import os
import sys
import logging

def _runCOSGParallel_single_batch(
    batch_key, shared_data, batch_i, groupby, n_svd_dims, n_svd_iter, n_highly_variable_genes, verbosity, resolution, mu, n_gene, use_highly_variable, layer, random_seed):
    """
    Process a single batch using shared memory and perform clustering and marker gene identification.

    Parameters
    ----------
    batch_key : str
        The key to identify batches in the data.
    shared_data : dict
        Dictionary containing shared memory and metadata to reconstruct the matrix.
    batch_i : str or int
        The batch identifier to process.
    groupby : str or None
        The key to group observations for clustering. If None, clustering will be performed.
    n_svd_dims : int
        Number of SVD components to calculate.
    n_svd_iter : int, optional, default=7
        Number of iterations for randomized SVD solver. The default is larger than the default in randomized_svd to handle sparse matrices that may have large slowly decaying spectrum. Also larger than the `n_iter` default value (5) in the TruncatedSVD function.
    n_highly_variable_genes : int
        Number of highly variable genes to use.
    verbosity : int
        Verbosity level.
    resolution : float
        Resolution parameter for clustering.
    mu : float
        Parameter for cosg.
    n_gene : int
        Number of genes to use in cosg.
    use_highly_variable : bool
        Whether to use highly variable genes for SVD.
    layer : str
        Layer in `adata.layers` to use for the analysis. Defaults to None, which uses `adata.X`.
    random_seed : int
        Random seed for reproducibility. Default is 1927.
    

    Returns
    -------
    DataFrame
        Marker gene DataFrame with batch-specific suffix.
    """
    
    import warnings
    
    
    # Reconstruct matrix from shared memory
    if 'shm_indices' in shared_data:
        data = np.ndarray(
            shared_data['shapes']['data_shape'],
            dtype=shared_data['dtypes']['data_dtype'],
            buffer=shared_data['shm_data'].buf
        )
        indices = np.ndarray(
            shared_data['shapes']['indices_shape'],
            dtype=shared_data['dtypes']['indices_dtype'],
            buffer=shared_data['shm_indices'].buf
        )
        indptr = np.ndarray(
            shared_data['shapes']['indptr_shape'],
            dtype=shared_data['dtypes']['indptr_dtype'],
            buffer=shared_data['shm_indptr'].buf
        )
        matrix = csr_matrix((data, indices, indptr), shape=shared_data['shapes']['matrix_shape'])
    else:
        matrix = np.ndarray(
            shared_data['shapes']['matrix_shape'],
            dtype=shared_data['dtypes']['data_dtype'],
            buffer=shared_data['shm_data'].buf
        )

    # adata = AnnData(matrix)
    # adata.obs = shared_data['obs'].copy()
    # ### No need to create a adata_i, because adata here is rebuilt from the matrix
    # # Filter the AnnData object for the current batch
    # adata = adata[adata.obs[batch_key] == batch_i].copy()
    

    ### Only directly select slices
    from anndata import AnnData
    batch_mask = shared_data['obs'][batch_key] == batch_i
    adata = AnnData(matrix[batch_mask.to_numpy()])
    adata.obs = shared_data['obs'][batch_mask.to_numpy()].copy()
    
    try:
        # Extract marker gene signatures
        if groupby is None:
            # Run SVD lazily
            if layer=='infog':
                ### in this case, adata.X will be the raw UMI counts, the infog_layer will be transferred as adata.X in this function input
                infog_svd(
                    adata,
                    copy=False,
                    n_components=n_svd_dims,
                    n_top_genes=n_highly_variable_genes,
                    use_highly_variable=use_highly_variable,
                    verbosity=verbosity,
                    batch_key=None,
                    scale_data=False,
                    n_iter=n_svd_iter,
                    layer='infog', ### Use INFOG normalization
                    infog_layer=None, ### By default, adata.X will be used for INFOG normalization
                    infog_trim=True,
                    key_added='X_svd',
                    random_state=random_seed
                )
            else:
                infog_svd(
                    adata,
                    copy=False,
                    n_components=n_svd_dims,
                    n_top_genes=n_highly_variable_genes,
                    use_highly_variable=use_highly_variable,
                    verbosity=verbosity,
                    batch_key=None,
                    scale_data=False,
                    n_iter=n_svd_iter,
                    layer=None,
                    infog_layer=None,
                    infog_trim=True,
                    key_added='X_svd',
                    random_state=random_seed
                )



            # Run clustering
            _piaso_neighbors(adata, use_rep='X_svd', n_neighbors=15, random_state=random_seed)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                _piaso_leiden(adata, resolution=resolution, key_added='gdr_local', random_state=random_seed)
            groupby_i = 'gdr_local'

        else:
            groupby_i = groupby

        if verbosity > 0:
            print(f'Processing batch {batch_i} with {len(np.unique(adata.obs[groupby_i]))} clusters.')

        
        # Run marker gene identification
        ### Because only one layer is transferred, so just use the adata.X
        cosg.cosg(
            adata,
            key_added='cosg',
            mu=mu,
            expressed_pct=0.1,
            remove_lowly_expressed=True,
            n_genes_user=n_gene,
            groupby=groupby_i
        )

        marker_gene = pd.DataFrame(adata.uns['cosg']['names'])
        marker_gene = marker_gene.add_suffix(f'@{batch_i}')

        # return marker_gene
        return batch_i, marker_gene
    except Exception as e:
        raise e

### To record the progress
from concurrent.futures import as_completed
from tqdm import tqdm
import warnings
from scipy.sparse import issparse

def runCOSGParallel(
    adata,
    batch_key: str,
    groupby: str = None,
    layer: str = None,
    infog_layer:str=None,
    n_svd_dims: int = 50,
    n_svd_iter: int = 7,
    n_highly_variable_genes: int = 5000,
    verbosity: int = 0,
    resolution: float = 1.0,
    mu: float = 1.0,
    n_gene: int = 30,
    use_highly_variable: bool = True,
    return_gene_names: bool = False,
    max_workers: int = 8,
    random_seed: int=1927
):
    """
    Run COSG on batches in parallel using shared memory and multiprocessing.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str
        The key in `adata.obs` used to identify batches.
    groupby : str, optional (default: None)
        The key in `adata.obs` used to group observations for clustering. If None, clustering will be performed.
    n_svd_dims : int, optional (default: 50)
        Number of SVD components to compute.
    n_svd_iter : int, optional, default=7
        Number of iterations for randomized SVD solver. The default is larger than the default in randomized_svd to handle sparse matrices that may have large slowly decaying spectrum. Also larger than the `n_iter` default value (5) in the TruncatedSVD function.
    n_highly_variable_genes : int, optional (default: 5000)
        Number of highly variable genes to use for SVD.
    verbosity : int, optional (default: 0)
        Level of verbosity for logging information.
    resolution : float, optional (default: 1.0)
        Resolution parameter for clustering.
    layer : str, optional (default: None)
        Layer of the `adata` object to use for COSG.
    infog_layer : str, optional (default: None)
        If specified, the INFOG normalization will be calculated using this layer of `adata.layers`, which is expected to contain the UMI count matrix. Defaults to None.
    mu : float, optional (default: 1.0)
        COSG parameter to control regularization.
    n_gene : int, optional (default: 30)
        Number of marker genes to compute for each cluster.
    use_highly_variable : bool, optional (default: True)
        Whether to use highly variable genes for SVD.
    return_gene_names : bool, optional (default: False)
        Whether to return gene names instead of indices in the marker gene DataFrame.
    max_workers : int, optional (default: 8)
        Maximum number of parallel workers to use. If None, defaults to the number of available CPU cores.
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.

    Returns
    -------
    DataFrame
        Combined marker gene DataFrame with batch-specific suffixes.

    Examples
    --------
    >>> import anndata
    >>> import piaso
    >>> adata = anndata.read_h5ad('example_data.h5ad')
    >>> marker_genes = piaso.tl.runCOSGParallel(
    ...     adata=adata,
    ...     batch_key='batch',
    ...     groupby=None,
    ...     n_svd_dims=50,
    ...     n_highly_variable_genes=5000,
    ...     verbosity=1,
    ...     resolution=1.0,
    ...     layer='log1p',
    ...     mu=1.0,
    ...     n_gene=30,
    ...     use_highly_variable=True,
    ...     return_gene_names=True,
    ...     max_workers=4
    ... )
    >>> print(marker_genes.head())
    """
    # Generate batch list
    batch_list = np.unique(adata.obs[batch_key])

    # Determine the input matrix
    if layer is None:
        gene_expression_data = adata.X
    elif layer == 'infog':
        if infog_layer is None:
            warnings.warn("Please set 'infog_layer'. Using adata.X.")
            gene_expression_data = adata.X
        else:
            gene_expression_data = adata.layers[infog_layer]
    else:
        gene_expression_data = adata.layers[layer]

    if issparse(gene_expression_data):
        if not isinstance(gene_expression_data, csr_matrix):
            raise ValueError("Sparse matrix must be in CSR format.")
        shared_data = _setup_shared_memory_sparse(gene_expression_data)
    else:
        shared_data = _setup_shared_memory_dense(gene_expression_data)
        
    shared_data['obs'] = adata.obs[[batch_key] + ([groupby] if groupby else [])].copy()

    
    # Create a map to enforce the exact order of batch_list
    # This handles cases where batches aren't alphabetical (e.g., ["Day1", "Day10", "Day2"])
    batch_order_map = {batch: i for i, batch in enumerate(batch_list)}
    
    marker_genes = []
    batch_n_groups = []

    try:
        # Use spawn context
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = []
            for batch_i in batch_list:
                futures.append(
                    executor.submit(
                        _runCOSGParallel_single_batch, batch_key, shared_data, batch_i, groupby,
                        n_svd_dims, n_svd_iter, n_highly_variable_genes, verbosity, resolution,
                        mu, n_gene, use_highly_variable, layer, random_seed
                    )
                )
            # Collect results into a list first
            raw_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating marker genes", unit="batch"):
                raw_results.append(future.result()) # Tuple of (batch_i, data)
                # marker_gene = future.result()
                # marker_genes.append(marker_gene)
                # batch_n_groups.append(marker_gene.shape[1])
                
        # Sort results based on the original batch_list order
        raw_results.sort(key=lambda x: batch_order_map[x[0]])

        # Unpack in the correct order
        for _, marker_gene in raw_results:
            marker_genes.append(marker_gene)
            batch_n_groups.append(marker_gene.shape[1])
    
    finally:
        # Robust Cleanup
        shared_data['shm_data'].close()
        shared_data['shm_data'].unlink()
        if 'shm_indices' in shared_data:
            shared_data['shm_indices'].close()
            shared_data['shm_indices'].unlink()
            shared_data['shm_indptr'].close()
            shared_data['shm_indptr'].unlink()

    # Merge and Format
    marker_genes = pd.concat(marker_genes, axis=1)
    marker_genes = marker_genes.astype(int)
    index_to_name_mapping = {i: name for i, name in enumerate(adata.var.index)}

    if return_gene_names:
        try:
            # New Pandas
            marker_genes = marker_genes.map(lambda idx: index_to_name_mapping.get(idx, idx))
        except AttributeError:
            # Old Pandas
            marker_genes = marker_genes.applymap(lambda idx: index_to_name_mapping.get(idx, idx))


    return marker_genes, batch_n_groups



def _runGDRParallel_cytome(
    ds, groupby, n_gene, mu, scoring_method, key_added, max_workers,
    random_seed, verbosity, modality, cytome_layer, batch_size_cytome,
    score_layer, score_cytome_layer=None, expressed_pct: float = 0.1,
    write_to_cytome: bool = True,
    cytome_marker_gene_key: str = "runGDR_marker_genes",
    save_reference: bool = True,
    # Auto-cluster knobs (used when groupby is None — mirrors the
    # AnnData path in runGDR which runs SVD + neighbors + Leiden inline)
    batch_key=None,
    n_svd_dims: int = 50,
    n_svd_iter: int = 7,
    resolution: float = 1.0,
):
    """Cytome streaming path for runGDRParallel.

    Steps:
    1. Read cluster labels from cytome
    2. Run COSG via streaming cytome backend (bounded memory)
    3. Score marker gene sets via streaming score()
    4. L2 normalize → X_gdr
    5. Persist (default) via ds.add_embedding + ds.metadata, or return tuple
    """
    from cytome.core.measurement import MeasurementLayer
    from ._normalization import _open_cytome

    # Accept either a cytome.Dataset or a path string (the public dispatch
    # advertises both via `isinstance(adata, str)`). Open here so downstream
    # code can rely on `ds._conn`.
    _opened_here = False
    if isinstance(ds, str):
        ds = _open_cytome(ds)
        _opened_here = True

    # ----- Pre-flight: validate that the requested cytome layers exist -----
    # The score step reads {modality}_{score_cytome_layer or cytome_layer}
    # via iter_chunks, so that matrix MUST be materialised on disk. COSG may
    # accept compute_on_fly when params are cached in metadata, but score
    # has no such on-the-fly path. Fail fast with an informative error so
    # users know exactly which `piaso.tl.infog(...)` call to make.
    _scoring_layer = score_cytome_layer if score_cytome_layer is not None else cytome_layer
    _required_matrices = {f"{modality}_{cytome_layer}", f"{modality}_{_scoring_layer}"}
    _present = {
        row[0] for row in ds._conn.execute(
            "SELECT matrix_name FROM matrix_meta"
        ).fetchall()
    }
    _missing = _required_matrices - _present
    if _missing:
        # If 'infog' is missing, give the precise piaso command to materialise it.
        hint_lines = []
        for name in sorted(_missing):
            if name.endswith("_infog"):
                hint_lines.append(
                    f"  - {name}: run `piaso.tl.infog(ds, save_layer=True, modality='{modality}')`"
                )
            elif name.endswith("_tfidf"):
                hint_lines.append(
                    f"  - {name}: run `piaso.tl.run_TFIDF(ds, modality='{modality}', "
                    f"output_layer='tfidf')` (writes the matrix to disk in cytome streaming mode)"
                )
            else:
                hint_lines.append(f"  - {name}: ensure the layer is materialised in the cytome")
        if _opened_here:
            ds.close()
        raise ValueError(
            "runGDR (cytome): required matrix layers not found in cytome:\n"
            + "\n".join(hint_lines)
            + f"\nAvailable matrices: {sorted(_present)}.\n"
            f"Tip: cytome runGDR scoring reads from `{modality}_{_scoring_layer}` "
            f"via iter_chunks — that matrix must be on disk. To match the AnnData "
            f"workflow's defaults, run `piaso.tl.infog(ds, save_layer=True, "
            f"modality='{modality}')` once before runGDR."
        )

    # n_features is needed for the verbose log; read it from
    # {modality}_counts (always present), not {modality}_{cytome_layer}
    # — the latter may not be materialized when compute_on_fly=True does
    # the work per-chunk inside COSG.
    counts_name = f"{modality}_counts"
    ml = MeasurementLayer(ds._conn, counts_name)
    n_cells = ds.n_cells
    n_features = ml.shape[1]

    # Multi-batch dispatch — D-4 (option a). When batch_key is set the
    # cytome path runs per-batch COSG via cell_mask + cross-batch
    # scoring with per-batch-marker-block L2 normalization, mirroring
    # the AnnData multi-batch path.
    if batch_key is not None:
        return _runGDR_multibatch_cytome(
            ds, batch_key=batch_key, groupby=groupby, n_gene=n_gene, mu=mu,
            scoring_method=scoring_method, key_added=key_added,
            max_workers=max_workers, random_seed=random_seed,
            verbosity=verbosity, modality=modality,
            cytome_layer=cytome_layer, batch_size_cytome=batch_size_cytome,
            score_layer=score_layer, score_cytome_layer=score_cytome_layer,
            write_to_cytome=write_to_cytome,
            cytome_marker_gene_key=cytome_marker_gene_key,
            save_reference=save_reference,
            expressed_pct=expressed_pct,
            n_svd_dims=n_svd_dims, n_svd_iter=n_svd_iter,
            resolution=resolution,
            _opened_here=_opened_here,
        )

    # Step 1: cluster labels — single-batch path.
    # D-1: when groupby is None, run de novo SVD + Leiden inline,
    # mirroring the AnnData runGDR single-batch path.
    if groupby is None:
        if verbosity > 0:
            print(
                "  No groupby provided; running de novo SVD + Leiden "
                "on cytome to derive cluster labels"
            )
        # piaso functions exposed via the local imports already used by
        # this file
        import piaso as _piaso_mod
        _piaso_mod.tl.runSVD(
            ds, modality=modality, n_components=n_svd_dims,
            n_iter=n_svd_iter, random_state=random_seed,
            key_added='X_svd_TMP_GDR', verbosity=0,
            streaming=True, measurement=cytome_layer,
        )
        knn_tmp = _piaso_mod.tl.neighbors(
            ds, use_rep=f'X_svd_TMP_GDR',
            n_neighbors=15, random_state=random_seed,
            key_added='neighbors_TMP_GDR',
        )
        # neighbors() on a cytome writes the graph to disk and returns None, so
        # knn_tmp is None — leiden must be told which stored graph to read via
        # neighbors_key (else it defaults to 'connectivities' and KeyErrors).
        _piaso_mod.tl.leiden(
            ds, resolution=resolution,
            key_added='gdr_local_TMP_GDR',
            random_state=random_seed,
            neighbors_key='neighbors_TMP_GDR',
            knn_result=knn_tmp, n_iterations=10,
        )
        groupby = 'gdr_local_TMP_GDR'

    cluster_labels = np.array(ds.cells[groupby])
    n_clusters = len(np.unique(cluster_labels))
    if verbosity > 0:
        print(f"  runGDRParallel (cytome): {n_cells} cells, {n_clusters} clusters")

    # Step 2: COSG via streaming cytome backend (no full matrix in memory)
    if verbosity > 0:
        print(f"  Running COSG on {modality} ({n_features} features) via streaming...")

    from cosg._cytome_streaming import run_cosg_cytome

    # Bug fix: pass cytome_layer + compute_on_fly to COSG so the marker
    # identification respects the user's requested layer (e.g. 'infog'
    # for RNA mirroring the AnnData layer='infog' default). Pre-fix the
    # cytome path silently ran COSG on raw counts while scoring used the
    # requested layer — silently produced different markers than the
    # AnnData-equivalent call.
    # Round 9 (2026-05-23): run_cosg_cytome_cpu was renamed to
    # run_cosg_cytome (function rename) and its cytome_layer kwarg was
    # renamed to layer (kwarg rename). We keep PIASO's own local variable
    # name `cytome_layer` (per Y-B in 2026-05-23 audit) and just map at
    # the call site.
    cosg_result = run_cosg_cytome(
        cytome_path=str(ds.path),
        groupby=groupby,
        # Round 7 (2026-05-22): n_top_genes -> n_genes_user (matches AnnData
        # cosg.cosg). Default output_format='ndarray' returns the (names,
        # scores) shape we consume below.
        n_genes_user=n_gene,
        mu=mu,
        remove_lowly_expressed=True,
        expressed_pct=expressed_pct,
        modality=modality,
        batch_size=batch_size_cytome,
        verbose=(verbosity > 0),
        feature_batching="auto",
        layer=cytome_layer,
        compute_on_fly=True,
        use_cached_stats=True,
    )
    marker_gene = pd.DataFrame(
        cosg_result['names'],
        columns=[str(g) for g in cosg_result['groups_order']],
    )

    # Step 3: Score via streaming (this is the key: no full matrix in memory)
    # _scoring_layer was resolved during pre-flight above.
    if verbosity > 0:
        print(
            f"  Scoring {len(marker_gene.columns)} gene sets "
            f"(streaming on {modality}_{_scoring_layer})..."
        )

    score_list, gene_set_names = calculateScoreParallel(
        ds,
        gene_set=marker_gene,
        score_method=scoring_method,
        score_layer=score_layer,
        max_workers=max_workers,
        random_seed=random_seed,
        modality=modality,
        cytome_layer=_scoring_layer,
        batch_size=batch_size_cytome,
    )

    # Robustness: calculateScoreParallel (via _process_gene_sets) DROPS gene sets
    # with no mappable genes — e.g. a tiny/rare cluster whose COSG markers were all
    # filtered out. When that happens `score_list` (→ X_gdr) has FEWER columns than
    # `marker_gene`, and a downstream `IGS @ X_gdr.T` matmul in inferGeneActivity
    # fails (e.g. 34 vs 35 clusters). Realign marker_gene to the surviving, scored
    # gene sets (in score_list column order) so X_gdr dims == marker_gene columns
    # for both the tuple return and the cytome-metadata write below.
    gsn = [str(n) for n in gene_set_names]
    if list(marker_gene.columns) != gsn:
        dropped = [c for c in marker_gene.columns if c not in set(gsn)]
        if dropped and verbosity > 0:
            print(f"  Dropped {len(dropped)} empty-gene-set cluster(s) from GDR "
                  f"(no mappable markers): {dropped}")
        marker_gene = marker_gene.loc[:, gsn]

    # Step 4: L2 normalize
    score_list = normalize(score_list, norm='l2', axis=0)
    score_list = normalize(score_list, norm='l2', axis=1)

    # Step 5: Persist or return.
    # Default (write_to_cytome=True): write X_gdr via add_embedding +
    # marker_gene as a metadata dict-of-lists; caller gets None back, mirroring
    # the AnnData path's in-place behaviour.
    if write_to_cytome:
        emb_name = key_added if key_added is not None else "X_gdr"
        ds.add_embedding(emb_name, np.asarray(score_list, dtype=np.float32))
        # Frozen-reference recipe for piaso.tl.projectGDR (cytome path). Mirrors the AnnData
        # branch: only the cheap recipe is written here; projectGDR completes and caches the rest.
        if save_reference:
            # _set_state encodes the DataFrame/ndarray members for the JSON-backed cytome
            # metadata; a raw assignment silently failed to serialise (round-9 bug).
            from ._projectGDR import _make_reference_recipe, _set_state, GDR_REFERENCE_KEY
            _denovo = groupby == 'gdr_local_TMP_GDR'
            _set_state(ds, _make_reference_recipe(
                marker_gene, block_indices=None, layer=_scoring_layer,
                groupby=(None if _denovo else groupby),
                batch_key=None, random_seed=random_seed,
                denovo_labels=('gdr_local' if _denovo else None),
                modality=modality))
            if verbosity > 0:
                print(f"Reference state saved to ds.metadata['{GDR_REFERENCE_KEY}'] "
                      f"(use piaso.tl.projectGDR to map new cells into this space)")

        ds.metadata[cytome_marker_gene_key] = {
            str(col): [str(v) for v in marker_gene[col].tolist()]
            for col in marker_gene.columns
        }
        ds.flush()
        if verbosity > 0:
            print(
                f"  Wrote runGDR results to cytome: embeddings['{emb_name}'] "
                f"({score_list.shape}) + metadata['{cytome_marker_gene_key}']"
            )
        if _opened_here:
            ds.close()
        return None
    else:
        if _opened_here:
            ds.close()
        return score_list, marker_gene


def _runGDR_multibatch_cytome(
    ds, batch_key, groupby, n_gene, mu,
    scoring_method, key_added, max_workers,
    random_seed, verbosity, modality, cytome_layer, batch_size_cytome,
    score_layer, score_cytome_layer=None,
    write_to_cytome: bool = True,
    cytome_marker_gene_key: str = "runGDR_marker_genes",
    save_reference: bool = True,
    # NOTE: this was referenced in the body but never declared (added by 96a9b53 when
    # cosg_expressed_pct was threaded through), so every multi-batch cytome runGDR raised
    # NameError. Declared here and forwarded from _runGDRParallel_cytome.
    expressed_pct: float = 0.1,
    n_svd_dims: int = 50,
    n_svd_iter: int = 7,
    resolution: float = 1.0,
    _opened_here: bool = False,
):
    """Multi-batch cytome streaming GDR — mirrors AnnData multi-batch path.

    Per-batch COSG via cell_mask, per-batch scoring with batch-local
    KDTree statistics, per-batch-marker-block L2 normalization,
    output reordered by cell_idx.

    When ``groupby is None``: runs per-batch SVD + Leiden inline
    (D-4 / Phase 3), using cell_mask on runSVD / neighbors / leiden
    so each batch gets its own LOCAL labels. Writes them to a single
    temp ``gdr_local_TMP_GDR`` column at masked positions.
    """
    from sklearn.preprocessing import normalize
    from cosg._cytome_streaming import run_cosg_cytome
    import piaso as _piaso_mod

    batch_labels = np.asarray(ds.cells[batch_key])
    batches = np.unique(batch_labels)
    n_cells = ds.n_cells
    if verbosity > 0:
        print(f"  Multi-batch GDR (cytome): {len(batches)} batches")

    _scoring_layer = (
        score_cytome_layer if score_cytome_layer is not None else cytome_layer
    )

    # Step 1: cluster labels — either from groupby column or per-batch de novo
    if groupby is None:
        if verbosity > 0:
            print(
                "  No groupby; running per-batch SVD + Leiden via cell_mask"
            )
        gdr_local_col = np.full(n_cells, "", dtype=object)
        for batch in batches:
            cell_mask = batch_labels == batch
            if verbosity > 0:
                print(f"    batch '{batch}': {int(cell_mask.sum())} cells")
            # SVD on masked cells (returns in-memory ndarray, no cytome write)
            svd_emb, _S, _Vt = _piaso_mod.tl.runSVD(
                ds, modality=modality, n_components=n_svd_dims,
                n_iter=n_svd_iter, random_state=random_seed,
                key_added='X_svd_TMP_GDR_BATCH',
                verbosity=0, streaming=True,
                measurement=cytome_layer,
                cell_mask=cell_mask,
            )
            # KNN on the in-memory SVD subset (passes ndarray, not cytome)
            knn_batch = _piaso_mod.tl.neighbors(
                svd_emb, n_neighbors=15,
                random_state=random_seed,
            )
            # Leiden on the in-memory knn_result (no cytome write)
            labels_batch = _piaso_mod.tl.leiden(
                ds, knn_result=knn_batch,
                resolution=resolution, random_state=random_seed,
                key_added='gdr_local_TMP_GDR_BATCH',
                n_iterations=10,
                cell_mask=np.ones(svd_emb.shape[0], dtype=bool),
            )
            gdr_local_col[cell_mask] = labels_batch.astype(str)
        ds.cells['gdr_local_TMP_GDR'] = gdr_local_col
        ds.flush()
        groupby = 'gdr_local_TMP_GDR'

    cluster_labels = np.asarray(ds.cells[groupby])

    # Step 2: per-batch COSG via cell_mask
    if verbosity > 0:
        print(f"  Per-batch COSG (cell_mask) across {len(batches)} batches")
    marker_gene_per_batch = []
    batch_n_groups = []
    for batch in batches:
        cell_mask = batch_labels == batch
        cosg_result = run_cosg_cytome(
            cytome_path=str(ds.path),
            groupby=groupby,
            cell_mask=cell_mask,
            n_genes_user=n_gene,
            mu=mu,
            remove_lowly_expressed=True,
            expressed_pct=expressed_pct,
            modality=modality,
            batch_size=batch_size_cytome,
            verbose=(verbosity > 0),
            feature_batching="auto",
            layer=cytome_layer,
            compute_on_fly=True,
            use_cached_stats=True,
        )
        mg_i = pd.DataFrame(
            cosg_result['names'],
            columns=[f"{batch}_{g}" for g in cosg_result['groups_order']],
        )
        marker_gene_per_batch.append(mg_i)
        batch_n_groups.append(len(cosg_result['groups_order']))

    marker_gene = pd.concat(marker_gene_per_batch, axis=1)
    batch_n_groups_indices = np.cumsum([0] + batch_n_groups)
    n_total_markers = int(batch_n_groups_indices[-1])

    # Step 3: per-batch scoring via cell_mask. Each batch's cells are
    # scored against the FULL combined marker matrix using a KDTree
    # built on that batch's cells only — mirrors the AnnData parallel
    # multi-batch path (per-batch KDTree, full-marker scoring).
    if verbosity > 0:
        print(
            f"  Per-batch scoring on {n_total_markers} markers "
            f"(streaming on {modality}_{_scoring_layer})"
        )
    score_full = np.zeros((n_cells, n_total_markers), dtype=np.float32)
    for batch_idx, batch in enumerate(batches):
        cell_mask = batch_labels == batch
        if verbosity > 0:
            print(
                f"    scoring batch {batch_idx+1}/{len(batches)}: '{batch}'"
            )
        score_list, _gene_set_names = calculateScoreParallel(
            ds,
            gene_set=marker_gene,
            score_method=scoring_method or 'piaso',
            score_layer=score_layer,
            max_workers=max_workers,
            random_seed=random_seed,
            modality=modality,
            cytome_layer=_scoring_layer,
            batch_size=batch_size_cytome,
            cell_mask=cell_mask,
        )
        # L2 normalize within this batch's cells: axis=0 over all
        # markers, then axis=1 within each batch-marker block.
        score_list = normalize(score_list, norm='l2', axis=0)
        for start, end in zip(
            batch_n_groups_indices[:-1], batch_n_groups_indices[1:]
        ):
            if end > start:
                score_list[:, start:end] = normalize(
                    score_list[:, start:end], norm='l2', axis=1,
                )
        score_full[cell_mask] = score_list.astype(np.float32)

    # Step 4: persist or return (same convention as single-batch path).
    if write_to_cytome:
        emb_name = key_added if key_added is not None else "X_gdr"
        ds.add_embedding(emb_name, score_full)
        # Frozen-reference recipe for projectGDR. The multi-batch path normalises axis=1 per
        # marker block, so the block boundaries must travel with the recipe — projectGDR
        # reproduces the same per-block row normalisation from `block_indices`.
        if save_reference:
            from ._projectGDR import _make_reference_recipe, _set_state, GDR_REFERENCE_KEY
            _denovo = groupby == 'gdr_local_TMP_GDR'
            _set_state(ds, _make_reference_recipe(
                marker_gene, block_indices=batch_n_groups_indices,
                layer=_scoring_layer,
                groupby=(None if _denovo else groupby),
                batch_key=batch_key, random_seed=random_seed,
                denovo_labels=('gdr_local' if _denovo else None),
                modality=modality))
            if verbosity > 0:
                print(f"Reference state saved to ds.metadata['{GDR_REFERENCE_KEY}'] "
                      f"(use piaso.tl.projectGDR to map new cells into this space)")

        ds.metadata[cytome_marker_gene_key] = {
            str(col): [str(v) for v in marker_gene[col].tolist()]
            for col in marker_gene.columns
        }
        ds.flush()
        if verbosity > 0:
            print(
                f"  Wrote multi-batch runGDR results: "
                f"embeddings['{emb_name}'] {score_full.shape} + "
                f"metadata['{cytome_marker_gene_key}']"
            )
        if _opened_here:
            ds.close()
        return None
    else:
        if _opened_here:
            ds.close()
        return score_full, marker_gene


def runGDRParallel(
    data=_UNSET,
    batch_key:str=None,
    groupby:str=None,
    n_gene:int=20,
    mu:float=10.0,
    expressed_pct:float=0.1,
    layer:str='infog',
    score_layer=_LAYER_DEFAULT,
    infog_layer:str=None,
    use_highly_variable:bool=True,
    n_highly_variable_genes:int=5000,
    n_svd_dims:int=50,
    n_svd_iter:int=7,
    resolution:float=1.0,
    scoring_method:str=None,
    key_added:str=None,
    max_workers:int=8,
    calculate_score_multiBatch:bool=True,
    n_concurrent_batches:int=None,
    verbosity: int=0,
    random_seed:int=1927,
    # Cytome streaming parameters
    modality: str = "RNA",
    batch_size_cytome: int = 1024,
    # ---- deprecated aliases (back-compat) ----
    adata=_UNSET,
):
    """
    .. deprecated::
        ``runGDRParallel`` is deprecated. Use :func:`runGDR` instead —
        parallel execution is now the default (``max_workers=8``).

    Run GDR (marker Gene-guided dimensionality reduction) in parallel using multi-cores and shared memory.

    Parameters
    -----------
    adata : AnnData
        Annotated data matrix.

    batch_key : str, optional
        Key in `adata.obs` representing batch information. Defaults to None. If specified, different batches will be processed separately and in parallel, otherwise, the input data will be processed as one batch.

    groupby : str, optional
        Key in `adata.obs` to specify which cell group information to use. Defaults to None. If none, de novo clustering will be performed.

    n_gene : int, optional
        Number of genes, parameter used in COSG. Defaults to 30.

    mu : float, optional
        Gene expression specificity parameter, used in COSG. Defaults to 1.0.

    layer : str, optional
        Layer in ``adata.layers`` to use for the analysis. Defaults to ``'infog'``,
        which uses PIASO's INFOG normalization (requires ``piaso.tl.infog(adata)`` first).
        Pass ``layer=None`` to use ``adata.X`` directly (requires scanpy for HVG selection).

    score_layer : str, optional
        If specified, the gene scoring will be calculated using this layer of `adata.layers`. Defaults to None.

    infog_layer : str, optional
        If specified, the INFOG normalization will be calculated using this layer of `adata.layers`, which is expected to contain the UMI count matrix. Defaults to None.

    use_highly_variable : bool, optional
        Whether to use only highly variable genes when rerunning the dimensionality reduction. Defaults to True. Only effective when `groupby=None`.

    n_highly_variable_genes : int, optional
        Number of highly variable genes to use when `use_highly_variable` is True. Defaults to 5000. Only effective when `groupby=None`.

    n_svd_dims : int, optional
        Number of dimensions to use for SVD. Defaults to 50. Only effective when `groupby=None`.

    n_svd_iter : int, optional, default=7
        Number of iterations for randomized SVD solver. The default is larger than the default in randomized_svd to handle sparse matrices that may have large slowly decaying spectrum. Also larger than the `n_iter` default value (5) in the TruncatedSVD function.

    resolution : float, optional
        Resolution parameter for de novo clustering. Defaults to 1.0. Only effective when `groupby=None`.

    scoring_method : str, optional
        Specifies the gene set scoring method used to compute gene scores. If set to None, use PIASO's scoring method as default.

    key_added : str, optional
        Key under which the GDR dimensionality reduction results will be stored in `adata.obsm`. If None, results will be saved to `adata.obsm[X_gdr]`.

    max_workers : int, optional
        Maximum number of workers to use for parallel computation. Defaults to 8.

    calculate_score_multiBatch : bool, optional
        .. deprecated:: Use ``max_workers=1`` for sequential processing.
        Whether to calculate gene scores across multiple adata batches (if `batch_key` is specified). Defaults to True.

    n_concurrent_batches : int, optional
        Number of batches to process concurrently when ``calculate_score_multiBatch=True`` and
        ``scoring_method='piaso'``. Uses ThreadPoolExecutor for inter-batch parallelism (both
        sklearn KDTree and Rust score_complete release the GIL). If None, auto-determined from
        ``max_workers`` and the number of batches. Default is None.

    verbosity : int, optional
        Verbosity level of the function. Higher values provide more detailed logs. Defaults to 0.
        
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.

    Returns
    -------
    None
        The function modifies `adata` in place by adding GDR dimensionality reduction result to `adata.obsm[key_added]`.

    Examples
    --------
    >>> import anndata
    >>> import piaso
    >>>
    >>> adata = anndata.read_h5ad("example.h5ad")
    >>> piaso.tl.infog(adata)  # compute INFOG normalization first
    >>> piaso.tl.runGDRParallel(
    ...     adata,
    ...     batch_key="batch",
    ...     groupby="CellTypes",
    ...     n_gene=30,
    ...     max_workers=8,
    ...     verbosity=0
    ... )
    >>> print(adata.obsm["X_gdr"])
    """
    warnings.warn(
        "runGDRParallel is deprecated. Use runGDR() instead — "
        "parallel execution is now the default (max_workers=8).",
        DeprecationWarning, stacklevel=2,
    )
    adata = _resolve_data_arg(data, 'runGDRParallel', adata=adata)
    # Deprecated cytome-specific layer aliases → unified params.
    # Resolve layer-mirror default so the deprecated alias matches runGDR.
    if score_layer is _LAYER_DEFAULT:
        score_layer = layer
    cytome_layer = layer
    score_cytome_layer = score_layer
    from ._normalization import _is_cytome_dataset
    if _is_cytome_dataset(adata):
        if scoring_method == 'scanpy':
            raise NotImplementedError(
                "runGDRParallel(cytome, scoring_method='scanpy') is not "
                "supported. Pass scoring_method='piaso' (default) or "
                "export to AnnData via ds.to_anndata() first."
            )
        return _runGDRParallel_cytome(
            adata, groupby=groupby, n_gene=n_gene, mu=mu,
            scoring_method=scoring_method or 'piaso',
            key_added=key_added, max_workers=max_workers,
            random_seed=random_seed, verbosity=verbosity,
            modality=modality, cytome_layer=cytome_layer,
            score_cytome_layer=score_cytome_layer,
            batch_size_cytome=batch_size_cytome,
            score_layer=score_layer,
            batch_key=batch_key,
            n_svd_dims=n_svd_dims, n_svd_iter=n_svd_iter,
            resolution=resolution,
        )

    ### Validate layer existence
    if layer is not None and layer not in adata.layers:
        if layer == 'infog':
            raise ValueError(
                "INFOG layer not found in adata.layers. "
                "Please run piaso.tl.infog(adata) first to compute INFOG normalization, "
                "or pass layer=None to skip INFOG (requires scanpy for HVG selection)."
            )
        raise ValueError(
            f"Layer '{layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}."
        )

    ### Check the scoring method, improve this part of codes later
    if scoring_method is not None:
        # Validate scoring_method
        if scoring_method not in {"scanpy", "piaso"}:
            raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be either 'scanpy' or 'piaso'.")
    else:
        ### Use piaso's scoring method as the default
        scoring_method='piaso'
    
    if batch_key is None:
        nbatches=1
    else:
        batch_list=np.unique(adata.obs[batch_key])
        nbatches=len(batch_list)
    
    if nbatches==1:
        ### Calculate the clustering labels if there is no specified clustering labels to use
        if groupby is None:
            ### Run SVD in a lazy mode
            infog_svd(
                adata,
                copy=False,
                n_components=n_svd_dims,
                n_top_genes=n_highly_variable_genes,
                use_highly_variable=use_highly_variable,
                verbosity=verbosity,
                batch_key=None,
                scale_data=False,
                n_iter=n_svd_iter,
                layer=layer,
                infog_layer=infog_layer,
                infog_trim=True,
                key_added='X_svd',
                random_state=random_seed
            )
            ### Because the verbosity will be reset in the above function
            _knn_result = _piaso_neighbors(adata, use_rep='X_svd', n_neighbors=15, random_state=random_seed)
            _piaso_leiden(adata, resolution=resolution, key_added='gdr_local', knn_result=_knn_result, random_state=random_seed)
            groupby='gdr_local'
            
        if verbosity>0:
            print('Number of clusters: ',len(np.unique(adata.obs[groupby])))
            
        ### Run marker gene identification
        if layer is not None:
            cosg.cosg(adata,
                key_added='cosg',
                use_raw=False,
                layer=layer,
                mu=mu,
                expressed_pct=expressed_pct,
                remove_lowly_expressed=True,
                n_genes_user=n_gene,
                groupby=groupby
                     )
        else:
            cosg.cosg(adata,
                key_added='cosg',
                mu=mu,
                expressed_pct=expressed_pct,
                remove_lowly_expressed=True,
                n_genes_user=n_gene,
                groupby=groupby
                     )
        
        marker_gene=pd.DataFrame(adata.uns['cosg']['names'])

    
        ### Calculate scores
        score_list_collection=[]
        score_list=[]
        
        
        #### Using single-CPU version
        # ### use a copy of the adata for the scoring
        # adata_tmp=adata.copy()
        # ### Set the layer used for scoring
        # if score_layer is not None:
        #     adata_tmp.X=adata_tmp.layers[score_layer]   
        # for i in marker_gene.columns:
        #     marker_gene_i=marker_gene[i].values
        #     _sc_score_genes(adata_tmp,
        #                   marker_gene_i,
        #                   score_name='markerGeneFeatureScore_i')
        #     ### Need to add the .copy()
        #     score_list.append(adata_tmp.obs['markerGeneFeatureScore_i'].values.copy())
        # score_list=np.vstack(score_list).T
        
        ### Use parallel computing of the gene set scores
        score_list, gene_set_names=calculateScoreParallel(
            adata,
            gene_set=marker_gene,
            score_method=scoring_method,
            score_layer=score_layer,
            max_workers=max_workers,
            random_seed=random_seed
        )
        
        ### Normalization
        score_list=normalize(score_list,norm='l2',axis=0)
        score_list=normalize(score_list,norm='l2',axis=1) ## Adding this is important
        score_list_collection.append(score_list)


        score_list_collection=np.vstack(score_list_collection)
        # score_list_collection_collection.append(score_list_collection)
        # marker_gene_scores=np.hstack(score_list_collection_collection)
        marker_gene_scores=score_list_collection
        marker_gene_scores=pd.DataFrame(marker_gene_scores)
        marker_gene_scores.index=adata.obs_names
        # marker_gene_scores.index=np.hstack([adata_list[0].obs_names.values, adata_list[1].obs_names.values])
        marker_gene_scores=marker_gene_scores.loc[adata.obs_names]
                
    ### Have multiple batches    
    else:
        
        
        
        ### Calculate the marker genes for each batch in parallel, the returned marker_genes data frame contains the marker gene lists combined from each individual batch
        marker_gene, batch_n_groups=runCOSGParallel(
            adata,
            batch_key=batch_key,
            groupby=groupby,
            n_gene=n_gene,
            mu=mu,
            use_highly_variable=use_highly_variable,
            n_highly_variable_genes=n_highly_variable_genes,
            layer=layer,
            infog_layer=infog_layer,
            n_svd_dims=n_svd_dims,
            n_svd_iter=n_svd_iter,
            resolution=resolution,
            verbosity=verbosity,
            return_gene_names=True, ### Return the names, making it easier to be compatible with the calculateScoreParallel function
            max_workers=max_workers,
            random_seed=random_seed

        )
        
    
        
        
        
        ### Create the indices for each batch's corresponding marker gene sets' indices in the merged marker gene dataframe
        batch_n_groups_indices = np.cumsum([0] + batch_n_groups)
        
        
        if calculate_score_multiBatch:
            #### Calculate marker gene sets for different batches separately, but in parallel
            score_list_collection, cellbarcode_info, gene_set_names_collection=calculateScoreParallel_multiBatch(
                adata,
                batch_key=batch_key,
                marker_gene=marker_gene,
                marker_gene_n_groups_indices=batch_n_groups_indices,
                score_layer=score_layer,
                max_workers=max_workers,
                n_concurrent_batches=n_concurrent_batches,
                score_method=scoring_method,
                random_seed=random_seed
            )
            
        else:
            ##################Score gene sets in different batches, seperately
            #############
            ### Calculate scores
            score_list_collection=[]
            ### Store the cell barcodes info
            cellbarcode_info=list()

            ### Scoring the geneset among different batches
            # for batch_u in batch_list:
            for batch_u in tqdm(batch_list, desc="Calculating cell embeddings", unit="batch"):
                ### Store the cell barcodes for each batch
                cellbarcode_info.append(adata.obs_names[adata.obs[batch_key]==batch_u].values)

                ### Use parallel computing of the gene set scores
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    adata_batch_u = adata[adata.obs[batch_key]==batch_u].copy()
                score_list, gene_set_names=calculateScoreParallel(
                    adata_batch_u,
                    gene_set=marker_gene,
                    score_method=scoring_method,
                    score_layer=score_layer,
                    max_workers=max_workers,
                    random_seed=random_seed
                )

                ### Normalization, within each batch
                score_list=normalize(score_list,norm='l2',axis=0)
                # score_list=normalize(score_list,norm='l2',axis=1) ## Adding this is important
                # Block-wise L2-normalization, maybe could be better implemented with higher efficiency
                # comparing different groups' marker gene scores
                for start, end in zip(batch_n_groups_indices[:-1], batch_n_groups_indices[1:]):
                    score_list[:, start:end] = normalize(score_list[:, start:end], norm='l2', axis=1) ## Adding this is important

                score_list_collection.append(score_list)

            ################## End ####################
        
    

        score_list_collection=np.vstack(score_list_collection)
        
        # score_list_collection_collection.append(score_list_collection)
        # marker_gene_scores=np.hstack(score_list_collection_collection)
        marker_gene_scores=score_list_collection
        marker_gene_scores=pd.DataFrame(marker_gene_scores)
        marker_gene_scores.index=np.hstack(cellbarcode_info)
        marker_gene_scores=marker_gene_scores.loc[adata.obs_names]
    
    
    
    
    # Store metadata about the GDR run
    adata.uns['gdr'] = {
        'params': {
            'n_gene': n_gene,
            'mu': mu,
            'layer': layer,
            'score_layer': score_layer,
            'infog_layer': infog_layer,
            'scoring_method': scoring_method,
            'random_seed': random_seed
        }
    }
    
    
    ### Set the low-dimensional representations for the integrated dataset
    if key_added is not None:
        adata.obsm[key_added]=marker_gene_scores.values
        print('The cell embeddings calculated by GDR were saved as `'+key_added+'` in adata.obsm.')
    else:
        adata.obsm['X_gdr']=marker_gene_scores.values
        print('The cell embeddings calculated by GDR were saved as `X_gdr` in adata.obsm.')


