import scanpy as sc
import pandas as pd
import numpy as np

from typing import Iterable, Union, Optional

### Normalization based on information
def infog(
    adata,
    copy:bool=False,
    n_top_genes:int=1000,
    key_added:str='infog',
    random_state:int =10,
    trim:bool=True,
    verbosity:int=1,
    layer: Optional[str] = None
):
    if layer and layer not in adata.layers:
        raise ValueError(f"{layer} not found in adata.layers.")
    
    adata = adata.copy() if copy else adata
    
    ### To get the gene expression matrix
    if layer:
        counts = adata.layers[layer]
    else:
        counts = adata.X
    
    ### Convert to csr sparse matrix:
    if not sparse.issparse(counts):
        counts=sparse.csr_matrix(counts)    
    

    cell_depth=counts.sum(axis=1).A
    gene_depth=counts.sum(axis=0).A   
    counts_sum  = np.sum(counts)
    scale = np.median(cell_depth.ravel())
    ### should use this one, especially for downsampling experiment, only this one works, the sequencing baises are corrected, partially because only this transformation is linear
    normalized =  sparse.diags(scale/cell_depth.ravel()) @ counts 
    
    info_factor=sparse.diags(counts_sum/cell_depth.ravel()) @ counts @ sparse.diags(1/gene_depth.ravel())
    normalized2=normalized.multiply(info_factor).sqrt()
    if trim:
        threshold = np.sqrt(counts.shape[0])
        normalized2.data[normalized2.data >  threshold] =  threshold

    adata.layers[key_added]=normalized2
    
    ### Calculate the variance
    c = normalized2.copy()
    mean=np.array(c.mean(axis=0))
    mean **=2
    c.data **= 2
    residual_var_orig_b=np.squeeze(np.array(c.mean(axis=0))-mean) 
#     del c
    adata.var[key_added+'_var']=residual_var_orig_b
    
    ### Feature selection    
    pos_gene=_select_top_n(adata.var[key_added+'_var'],n_top_genes)
    tmp=np.repeat(False,adata.n_vars)
    tmp[pos_gene]=True
    adata.var['highly_variable_'+key_added]=tmp
    if verbosity>0:
        print(f'The normalized data is saved as `{key_added}` in `adata.layers`.')
        print(f'The highly variable genes are saved as `highly_variable_{key_added}` in `adata.obs`.')
        print(f'Finished INFOG normalization.')
         
    ### Return the result
    return adata if copy else None


### Refer to Scanpy for _select_top_n function
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices

### Refer to scDRS for _get_p_from_empi_null function
def _get_p_from_empi_null(v_t, v_t_null):
    """Compute p-value from empirical null
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
    
    
    if gene_weights is None:
        gene_weights=np.repeat(1.0, len(gene_list))
    
    ### Calculate the mean and variance

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
    pooled_p = _get_p_from_empi_null(cellxgene_query, cellxgene_ctrl.A.flatten())
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
        
    