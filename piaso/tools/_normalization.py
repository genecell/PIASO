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

#### Gene Set Scoring Method
def score(
    adata,
    gene_list,
    gene_weights=None,
    n_nearest_neighbors: int=30,
    leaf_size: int=40,
    layer: str='infog',
    random_seed: int=1927,
    n_ctrl_set:int=100,
    key_added:str=None,
    verbosity: int=0
):
    """
    For a given gene set, compute gene expression enrichment scores and P values for all the cells.

    Parameters
    ----------
    adata : AnnData
        The AnnData object for the gene expression matrix.
    
    gene_list : list of str
        A list of gene names for which the score will be computed.
    
    gene_weights : list of floats, optional
        A list of weights corresponding to the genes in `gene_list`. The length of 
        `gene_weights` must match the length of `gene_list`. If None, all genes in 
        `gene_list` are weighted equally. Default is None.
    
    n_nearest_neighbors : int, optional
        Number of nearest neighbors to consider for randomly selecting control gene sets based on the similarity of genes' mean and variance among all cells.
        Default is 30.
    
    leaf_size : int, optional
        Leaf size for the KD-tree or Ball-tree used in nearest neighbor calculations. Default is 40.
    
    layer : str, optional
        The name of the layer in `adata.layers` to use for gene expression values. Default is 'infog'.
    
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.
    
    n_ctrl_set : int, optional
        Number of control gene sets to be used for calculating P values. Default is 100.
    
    key_added : str, optional
        If provided, the computed scores will be stored in `adata.obs[key_added]`. The scores and P values will be stored in `adata.uns[key_added]` as well.
        Default is None, and the `INFOG_score` will be used as the key.
    verbosity : int, optional (default: 0)
        Level of verbosity for logging information.
        
    Returns
    -------
    None
        Modifies the `adata` object in-place, see `key_added`.
    """
    ### Set the random seed
    # Set the random seed for reproducibility
    np.random.seed(random_seed) 
    
    # Ensure gene_weights is correctly initialized
    if gene_weights is None:
        gene_weights = np.ones(len(gene_list), dtype=float) 
    elif len(gene_weights) != len(gene_list):
        raise ValueError(f"Length mismatch: the input gene_weights ({len(gene_weights)}) and gene_list ({len(gene_list)}) must be the same.")

    ### Only keep the ones in adata.var_names
    valid_genes = set(adata.var_names)  # Convert to set for fast lookup
    
    # Initialize lists for valid and invalid genes
    filtered_genes = []
    filtered_weights = []
    filtered_out_genes = []
    # Single pass loop to separate valid and invalid genes
    for gene, weight in zip(gene_list, gene_weights):
        if gene in valid_genes:
            filtered_genes.append(gene)
            filtered_weights.append(weight)
        else:
            filtered_out_genes.append(gene)
            
    ### Reset the lists
    gene_list=filtered_genes
    gene_weights=filtered_weights
    
    
    ### Check the number of filtered out genes
    n_filtered_genes=len(filtered_out_genes)
    if verbosity>0 and n_filtered_genes>0:
        print(f"Note: {n_filtered_genes} genes were not found in adata.var_names and are excluded from scoring: {', '.join(filtered_out_genes[:10])} {'...' if n_filtered_genes > 10 else ''}")



    ### Calculate the variance
    # Determine the input matrix
    if layer is not None:
        cellxgene = adata.layers[layer] ### I think .copy() is not needed here
        ### For the query gene set
        cellxgene_subset=adata[:, gene_list].layers[layer]
    else:
        cellxgene = adata.X
        ### For the query gene set
        cellxgene_subset= adata[:, gene_list].X
    
    
    ### Calculate the mean and variance
    # c=cellxgene.copy()
    mean=np.array(cellxgene.mean(axis=0))
    infog_mean=mean.copy()[0]
    mean **=2
    ### Instead of c.data **= 2, I used cellxgene.multiply(cellxgene), which is the same result
    # c.data **= 2
    # residual_var_orig_b=np.squeeze(np.array(c.mean(axis=0))-mean) 
    residual_var_orig_b = np.squeeze(np.array(cellxgene.multiply(cellxgene).mean(axis=0)) - mean)
    
    mean_var=np.array([infog_mean, residual_var_orig_b]).T
    ### Construct a kNN graph for the genes based on gene means and gene variances 
    kdt = KDTree(mean_var, leaf_size=leaf_size, metric='euclidean')
    mean_var_knn_idx=kdt.query(mean_var, k=n_nearest_neighbors+1, return_distance=False)
    ### Remove the self node
    mask=mean_var_knn_idx != np.arange(mean_var.shape[0])[:,None]
    ### Use the mask to remove the self node
    mean_var_knn_idx=np.vstack(np.array([mean_var_knn_idx[i, mask[i]][: n_nearest_neighbors] for i in range(mean_var_knn_idx.shape[0])], dtype=np.int64))

    ### Only select for the query gene set
    mean_var_knn_idx_df=pd.DataFrame(mean_var_knn_idx)
    mean_var_knn_idx_df.index=adata.var_names.values
    gene_list_knn_idx=mean_var_knn_idx_df.loc[gene_list].values

    ### Create a matrix to hold the gene weights for randomly pickedup control genes

    # # Randomly select indices from each row
    # # This generates a matrix of shape (T, N) where each column contains random indices for the corresponding row
    # random_indices = np.random.randint(mean_var_knn_idx.shape[1], size=(n_ctrl_set, mean_var_knn_idx.shape[0]))
    n_genes=gene_list_knn_idx.shape[0]
    # Initialize an array to hold the sampled values
    n_ctrl_set_idx = np.empty((n_ctrl_set, n_genes), dtype=gene_list_knn_idx.dtype)
    ### Sampling genes with similar mean and variance 
    for n in range(n_genes):
        n_ctrl_set_idx[:,n] = np.random.choice(gene_list_knn_idx[n], size=n_ctrl_set, replace=True)


    ### Create a sparse matrix, rows are genes, columns are control gene set
    rows = []
    cols = []
    data = []
    for ctrl_i, ctrl_gene_idx in enumerate(n_ctrl_set_idx):
        rows.append(ctrl_gene_idx)
        cols.append(np.repeat(ctrl_i,len(gene_list)))
        data.append(gene_weights)

    ctrl_gene_weight = sparse.csr_matrix((np.ravel(data), (np.ravel(rows), np.ravel(cols))), shape=(adata.n_vars, n_ctrl_set))

    #### Apply L1-normalization as we need to calculate the mean value
    #### But it's not equal to L1-normalization, because the weight has it's own scale
    # ctrl_gene_weight=normalize(ctrl_gene_weight,norm='l1', axis=0)

    cellxgene_ctrl=cellxgene @ ctrl_gene_weight
    
    
    ### Need to do element-wise multiplication to add the gene weights:
    ### The following one is not correct, because the gene orders will be changed:
    ### cellxgene_query=np.ravel(cellxgene[:,np.isin(adata.var_names,gene_list)].multiply(gene_weights).mean(axis=1))
    # cellxgene_query=np.ravel(adata[:, gene_list].layers[layer].multiply(gene_weights).mean(axis=1))
    ## Use sum here, because the ctrl multiplication equals to sum
    ### cellxgene_subset is the cellxgene matrice with the input gene kept
    cellxgene_query=np.ravel(cellxgene_subset.multiply(np.array(gene_weights)).sum(axis=1))
    
    # Get p-values
    from statsmodels.stats.multitest import multipletests
    ### Should use >=, because a[i-1] < v <= a[i] is for left in numpy.searchsorted
    ### Refer to https://numpy.org/doc/2.1/reference/generated/numpy.searchsorted.html
    n_greater=np.sum(cellxgene_ctrl>= cellxgene_query[:, None], axis=1)
    p_value_monte_carlo = np.ravel( (n_greater+1) / (n_ctrl_set+1))
    nlog10_p_value_monte_carlo = -np.log10(p_value_monte_carlo)
    pooled_p_monte_carlo_FDR=multipletests(p_value_monte_carlo, method="fdr_bh")[1]
    nlog10_pooled_p_monte_carlo_FDR=-np.log10(pooled_p_monte_carlo_FDR)
    
    
    ### Caculate pool_valu
    pooled_p = _get_p_from_empi_null(cellxgene_query, cellxgene_ctrl.toarray().flatten())
    nlog10_pooled_p = -np.log10(pooled_p)
    
    pooled_p_FDR=multipletests(pooled_p, method="fdr_bh")[1]
    nlog10_pooled_p_FDR=-np.log10(pooled_p_FDR)
    

    BG=np.ravel(cellxgene_ctrl.mean(axis=1))
    
    ### Normalize the score by the number of genes and the gene weights
    scaling_factor=np.median(gene_weights)*len(gene_list)
    cellxgene_query=cellxgene_query/scaling_factor
    BG=BG/scaling_factor
    
    score= cellxgene_query - BG
    ### Use division
    # score= cellxgene_query/(BG + 1e-10) ## adding epsilon to avoid division by zero
    
    score_pval_res = {
                "score": score,
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
        adata.obs['INFOG_score']=score
        adata.uns['INFOG_score']=df_score_pval_res
        if verbosity>0:
            print(f"Finished. The scores are saved in adata.obs['INFOG_score'] and the scores, P values are saved in adata.uns['INFOG_score'].")
        
    else:
        adata.obs[key_added]=score
        adata.uns[key_added]=df_score_pval_res
        if verbosity>0:
            print(f"Finished. The scores are saved in adata.obs['{key_added}'] and the scores, P values are saved in adata.uns['{key_added}'].")
        
    