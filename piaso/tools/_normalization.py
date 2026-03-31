import scanpy as sc
import pandas as pd
import numpy as np

from typing import Iterable, Union, Optional

### Normalization based on information
def infog(
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
        counts=sparse.csr_matrix(counts)    
    
        #### Raise an error if any negative values are found in counts. Only check when the input is not in a sparse format
        if counts.data.size > 0 and counts.data.min() < 0:
            raise ValueError("Input counts contain negative values, which is not allowed.")
    
    ### Compute cell and gene depths
    cell_depth = np.array(counts.sum(axis=1)).ravel()
    gene_depth = np.array(counts.sum(axis=0)).ravel()
    
    
    
    counts_sum = counts.sum()
    scale = np.median(cell_depth)
    
    
    ### should use this one, especially for downsampling experiment, only this one works, the sequencing baises are corrected, partially because only this transformation is linear
    ### Instead of using sparse.diags, use element-wise multiplication with broadcasting.
    normalized = counts.multiply(scale / cell_depth[:, None])    
    
    # Compute info_factor: first, scale rows by counts_sum/cell_depth, then columns by 1/gene_depth. 
    # Avoid division by zero: for gene_depth==0, set reciprocal to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_gene_depth = 1.0 / gene_depth
    inv_gene_depth[~np.isfinite(inv_gene_depth)] = 0.0
    info_factor = counts.multiply(counts_sum / cell_depth[:, None]).multiply(inv_gene_depth)


    
    # Element-wise multiplication and square root.
    ### Previously, here I created another name, but it's not good for the memory usage, so here I kept using normalized
    normalized = normalized.multiply(info_factor)
    # Apply element-wise square root
    normalized.data = np.sqrt(normalized.data)
    
    # normalized2=normalized.multiply(info_factor).sqrt()
    
    if trim:
        threshold = np.sqrt(counts.shape[0])
        normalized.data[normalized.data >  threshold] =  threshold
        
    

    # Save the normalized data according to the inplace flag
    if inplace:
        adata.X = normalized.copy()
    else:
        adata.layers[key_added] = normalized.copy()
    
    ### Calculate the variance
    # Compute mean of normalized2 (from the unmodified copy stored in adata)
    mean = np.array(normalized.mean(axis=0)).ravel()
    
    # Fast in-place squaring for mean-of-squares calculation
    normalized.data **= 2
    mean_sq = np.array(normalized.mean(axis=0)).ravel()
    
    residual_var_orig_b = mean_sq - mean**2
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
    mean_2d = np.array(cellxgene.mean(axis=0))  # (1, n_genes)
    infog_mean = mean_2d.copy()[0]  # (n_genes,)
    mean_sq = mean_2d ** 2

    if sparse.issparse(cellxgene):
        data_sq = cellxgene.data ** 2
        X_sq = sparse.csr_matrix((data_sq, cellxgene.indices, cellxgene.indptr),
                                  shape=cellxgene.shape, copy=False)
        residual_var = np.squeeze(np.array(X_sq.mean(axis=0)) - mean_sq)
        del X_sq, data_sq
    else:
        residual_var = np.squeeze(np.mean(np.asarray(cellxgene) ** 2, axis=0) - mean_sq)

    mean_var = np.array([infog_mean, residual_var]).T
    kdt = KDTree(mean_var, leaf_size=leaf_size, metric='euclidean')
    knn_idx = kdt.query(mean_var, k=n_nearest_neighbors + 1, return_distance=False)

    # Remove self-loops
    mask = knn_idx != np.arange(knn_idx.shape[0])[:, None]
    knn_idx = np.array([
        knn_idx[i, mask[i]][:n_nearest_neighbors]
        for i in range(knn_idx.shape[0])
    ], dtype=np.int64)

    return knn_idx


#### Gene Set Scoring Method
def score(
    adata,
    gene_list,
    gene_weights=None,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 40,
    layer: str = 'infog',
    random_seed: int = 1927,
    n_ctrl_set: int = 100,
    key_added: str = None,
    compute_pvalues: bool = False,
    chunk_size: int = 10000,
    max_workers: int = 1,
    use_rust: bool = True,
    verbosity: int = 0,
):
    """
    Compute gene expression enrichment scores for one or more gene sets.

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
    np.random.seed(random_seed)

    # --- Determine mode: single gene set vs multiple ---
    _multi = isinstance(gene_list, (dict, pd.DataFrame)) or (
        isinstance(gene_list, list) and len(gene_list) > 0
        and isinstance(gene_list[0], (list, np.ndarray))
    )

    # --- Expression matrix and shared stats ---
    cellxgene = adata.layers[layer] if layer is not None else adata.X
    n_cells, n_genes = cellxgene.shape
    knn_idx = _precompute_stats(cellxgene, n_nearest_neighbors, leaf_size)
    var_names_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}

    # --- Try Rust backend (multi-set only) ---
    _rust_fmr = None
    if use_rust and _multi:
        try:
            from piaso._piaso_score import fused_matmul_reduce
            _rust_fmr = fused_matmul_reduce
        except ImportError:
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

        # Build control blocks and query scores
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