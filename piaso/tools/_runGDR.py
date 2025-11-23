from ._runSVD import runSVDLazy
from ._normalization import score

### Run GDR
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import cosg

import warnings

import multiprocessing

def runGDR(
    adata,
    batch_key: str = None,
    groupby: str = None,
    n_gene: int = 30,
    mu: float = 1.0,
    layer: str = None,
    score_layer: str = None,
    infog_layer: str = None,
    use_highly_variable: bool = True,
    n_highly_variable_genes: int = 5000,
    n_svd_dims: int = 50,
    n_svd_iter: int = 7,
    resolution: float = 1.0,
    scoring_method: str = None,
    key_added: str = None,
    verbosity: int = 0,
    random_seed: int = 1927
):
    """
    Run GDR (marker Gene-guided dimensionality reduction) on single-cell data.
    
    GDR performs dimensionality reduction guided by marker genes to better preserve biological signals. 

    Parameters
    -----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str, optional
        Key in `adata.obs` representing batch information. Defaults to None. If provided, marker gene identifications will be performed for each batch separately.
    groupby : str, optional
        Key in `adata.obs` to specify which cell group information to use. Defaults to None. If none, de novo clustering will be performed.
    n_gene : int, optional
        Number of genes, parameter used in COSG. Defaults to 30.
    mu : float, optional
        Gene expression specificity parameter, used in COSG. Defaults to 1.0.  
    layer : str, optional
        Layer in `adata.layers` to use for the analysis. Defaults to None, which uses `adata.X`.
    score_layer : str, optional
        If specified, the gene scoring will be calculated using this layer of `adata.layers`. Defaults to None.    
    infog_layer : str, optional
        If specified, INFOG normalization will be applied using this layer, which should
        contain the raw UMI count matrix. Defaults to None.
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
    >>> import scanpy as sc
    >>> import piaso
    >>> 
    >>> adata = sc.read_h5ad("example.h5ad")
    >>> piaso.tl.runGDR(
    ...     adata,
    ...     batch_key="batch",
    ...     groupby="CellTypes",
    ...     n_gene=30,
    ...     verbosity=0
    ... )
    >>> print(adata.obsm["X_gdr"])
    """
    
    ### Check the scoring method, improve this part of codes later
    if scoring_method is not None:
        valid_methods = {"scanpy", "piaso"}
        if scoring_method not in valid_methods:
            raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be one of {', '.join(valid_methods)}.")
    else:
        scoring_method = 'scanpy'  # Use scanpy's scoring method as default
        
    # Check if key exists in adata.obs
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.columns.")
    
    if groupby is not None and groupby not in adata.obs.columns:
        raise ValueError(f"Group key '{groupby}' not found in adata.obs.columns.")
    
    # Check for layer existence if specified
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    
    if score_layer is not None and score_layer not in adata.layers:
        raise ValueError(f"Score layer '{score_layer}' not found in adata.layers.")
    
    if infog_layer is not None and infog_layer not in adata.layers:
        raise ValueError(f"INFOG layer '{infog_layer}' not found in adata.layers.")
    
    # Remove empty log1p entry in adata.uns if it exists
    ### add this to avoid the errors in pp.highly_variable_genes and _highly_variable_genes_single_batch in scanpy
    if 'log1p' in adata.uns and (not adata.uns['log1p'] or adata.uns['log1p'] == {}):
        del adata.uns['log1p']
        if verbosity > 1:
            print("Removed empty log1p entry from adata.uns")
    
 
    # Set scanpy verbosity
    original_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0  # Suppress scanpy messages
    
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
                runSVDLazy(
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
                ### Because the verbosity will be reset in the above function
                sc.settings.verbosity=0

                # Run clustering
                if verbosity > 0:
                    print("Computing clustering")

                ### Run clustering
                sc.pp.neighbors(
                    adata,
                    use_rep='X_svd_TMP_GDR',
                    n_neighbors=15,
                    random_state=random_seed,
                    knn=True,
                    method="umap",
                    key_added='neighbors_TMP_GDR'
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    sc.tl.leiden(adata, resolution=resolution, key_added='gdr_local', neighbors_key='neighbors_TMP_GDR', random_state=random_seed) ## Leiden also used a random_state parameter
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
            score_list=[]

            ### use a copy of the adata for the scoring
            adata_tmp=adata.copy()
            ### Set the layer used for scoring
            if score_layer is not None:
                adata_tmp.X=adata_tmp.layers[score_layer]
            for i in marker_gene.columns:
                marker_gene_i=marker_gene[i].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    if scoring_method=='scanpy':
                        sc.tl.score_genes(adata_tmp, marker_gene_i, score_name='markerGeneFeatureScore_i', random_state=random_seed)
                    elif scoring_method=='piaso':
                        ## Set layer to None, because the scoring layer is already constructed as the adata.X
                        score(adata_tmp, gene_list=marker_gene_i.tolist(), key_added='markerGeneFeatureScore_i', layer=None, random_seed=random_seed)
                    else:
                        raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be either 'scanpy' or 'piaso'.")


                ### Need to add the .copy()
                score_list.append(adata_tmp.obs['markerGeneFeatureScore_i'].values.copy())

            score_list=np.vstack(score_list).T
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
            ### Store the cell barcodes info
            cellbarcode_info=list()
            for batch_idx, batch_i in enumerate(batch_list):
                if verbosity > 0:
                    print(f"Processing batch {batch_idx+1}/{nbatches}: '{batch_i}'")

                adata_i=adata[adata.obs[batch_key]==batch_i].copy()

                ### Extract marker gene signatures
                ### Calculate clustering labels if no clustering info was specified
                if groupby is None:
                    if verbosity > 0:
                        print(f"Running SVD for batch '{batch_i}'")

                    ### Run SVD in a lazy mode
                    runSVDLazy(
                        adata_i,
                        copy=False,
                        n_components=n_svd_dims,
                        n_top_genes=n_highly_variable_genes,
                        use_highly_variable=use_highly_variable,
                        verbosity=0,
                        batch_key=None, ### Need to set as None, because the SVD is calculated in each batch separately
                        scale_data=False,
                        n_iter=n_svd_iter,
                        layer=layer,
                        infog_layer=infog_layer,
                        infog_trim=True,
                        key_added='X_svd',
                        random_state=random_seed
                    )
                    ### Because the verbosity will be reset in the above function, the good way is to remember the previous state of verbosity
                    sc.settings.verbosity=0


                    if verbosity > 0:
                        print(f"Computing clustering for batch '{batch_i}'")

                    ### Run clustering
                    sc.pp.neighbors(adata_i,
                        use_rep='X_svd',
                        n_neighbors=15,random_state=random_seed,knn=True,
                        method="umap")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)                
                        sc.tl.leiden(adata_i,resolution=resolution,key_added='gdr_local')
                    groupby_i='gdr_local'    
                else:
                    groupby_i=groupby    

                if verbosity>0:
                    print(f"Identified {len(np.unique(adata_i.obs[groupby_i]))} clusters in batch '{batch_i}'")


                cellbarcode_info.append(adata_i.obs_names.values)

                # Run marker gene identification with COSG
                if verbosity > 0:
                    print(f"Identifying marker genes for batch '{batch_i}'")


                ### Run marker gene identification
                cosg_params = {
                    'key_added': 'cosg',
                    'mu': mu,
                    'expressed_pct': 0.1,
                    'remove_lowly_expressed': True,
                    'n_genes_user': n_gene,
                    'groupby': groupby_i
                }

                if layer is not None:
                    cosg_params['use_raw'] = False
                    cosg_params['layer'] = layer

                cosg.cosg(adata_i, **cosg_params)

                marker_gene=pd.DataFrame(adata_i.uns['cosg']['names'])


                ### Calculate scores
                score_list_collection=[]


                ### Scoring the geneset
                for batch_u in batch_list:
                    adata_u=adata[adata.obs[batch_key]==batch_u].copy()

                    score_list=[]

                    # adata_u.X=adata_u.layers['log1p'] ## do not use log1p, sometimes raw counts is better here
                    if score_layer is not None:
                        adata_u.X=adata_u.layers[score_layer]

                    for i in marker_gene.columns:
                        marker_gene_i=marker_gene[i].values


                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            if scoring_method=='scanpy':
                                sc.tl.score_genes(adata_u, marker_gene_i, score_name='markerGeneFeatureScore_i', random_state=random_seed)
                            elif scoring_method=='piaso':
                                ## Set layer to None, because the scoring layer is already constructed as the adata.X
                                score(adata_u,
                                      gene_list=marker_gene_i.tolist(), key_added='markerGeneFeatureScore_i', layer=None, random_seed=random_seed)
                            else:
                                raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be either 'scanpy' or 'piaso'.")

                        ### Need to add the .copy()
                        score_list.append(adata_u.obs['markerGeneFeatureScore_i'].values.copy())

                    score_list=np.vstack(score_list).T
                    ### Normalization
                    score_list=normalize(score_list,norm='l2',axis=0)
                    score_list=normalize(score_list,norm='l2',axis=1) ## Adding this is important

                    score_list_collection.append(score_list)
                score_list_collection=np.vstack(score_list_collection)
                score_list_collection_collection.append(score_list_collection)

            marker_gene_scores=np.hstack(score_list_collection_collection)

            ### Make sure the order are matched to the adata
            marker_gene_scores=pd.DataFrame(marker_gene_scores)
            marker_gene_scores.index=np.hstack(cellbarcode_info)
            marker_gene_scores=marker_gene_scores.loc[adata.obs_names]

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


            # Keep the cluster information for reference by default, but rename it to remove TMP suffix
            if 'gdr_local_TMP_GDR' in adata.obs.columns:
                del adata.obs['gdr_local_TMP_GDR']
                if verbosity > 1:
                    print("Removed temporary cell labels gdr_local_TMP_GDR from adata.obs")

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
    finally:
        # This always runs, even if error occurs
        sc.settings.verbosity = original_verbosity
    
    


        
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
    original_verbosity = sc.settings.verbosity
    
    
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
            shm = shared_memory.SharedMemory(name=metadata["shm"].name, create=False)
            X = np.ndarray(metadata["shape"], dtype=metadata["dtype"], buffer=shm.buf)          

        
        adata_tmp = sc.AnnData(X=X)
        
        
        sc.settings.verbosity = 0
        # ### Globally supress warning
        # warnings.filterwarnings("ignore", category=FutureWarning)
        # # # Suppress FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if score_method=='scanpy':
                sc.tl.score_genes(adata_tmp, adata_tmp.var.index[gene_indices].tolist(), score_name=score_name, random_state=random_seed)
                
                
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
        sc.settings.verbosity = original_verbosity


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
import scanpy as sc
from typing import Literal
from tqdm import tqdm

def calculateScoreParallel(
    adata,
    gene_set,
    score_method: Literal["scanpy", "piaso"],
    random_seed: int = 1927,
    score_layer = None,
    max_workers = None,
    return_pvals: bool = False,
    verbosity: int = 0,
):
    """
    Compute gene set scores in parallel using shared memory for efficiency.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object.
    gene_set : dict, list of lists, or pandas.DataFrame
        A collection of gene sets, where each gene set is either:
            - A dictionary: Keys are gene set names, values are lists of genes.
            - A list of lists: Each sublist contains genes in a gene set.
            - A pandas.DataFrame: Each column represents a gene set, and column names are gene set names.
    score_method : {'scanpy', 'piaso'}, optional
        The method used for gene set scoring. Must be either 'scanpy' (default) or 'piaso'.
        - 'scanpy': Uses the Scanpy's built-in gene set scoring method.
        - 'piaso': Uses the PIASO's gene set scoring method, which is more robust to sequencing depth variations.
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.
    score_layer : str or None, optional
        Layer of the AnnData object to use. If None, `adata.X` is used.
    max_workers : int or None, optional
        Number of parallel workers to use. Defaults to the number of CPUs.
    return_pvals : bool, optional
        Whether to return -log10 p-values when using 'piaso' method. Default is False.
    verbosity : int, optional
        Level of verbosity. Default is 0.

        
    Returns
    -------
    tuple
        If score_method is 'scanpy':
            - np.ndarray: A 2D array where each column contains the scores for a gene set.
            - list: The names of the gene sets.
        If score_method is 'piaso' and return_pvals is True:
            - np.ndarray: A 2D array where each column contains the scores for a gene set.
            - list: The names of the gene sets.
            - np.ndarray: A 2D array where each column contains the -log10(p-values) for a gene set.
        If score_method is 'piaso' and return_pvals is False:
            - np.ndarray: A 2D array where each column contains the scores for a gene set.
            - list: The names of the gene sets.
    """
    
    # Validate score_method
    if score_method not in {"scanpy", "piaso"}:
        raise ValueError(f"Invalid score_method: '{score_method}'. Must be either 'scanpy' or 'piaso'.")

    
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
    ### need to add list() to valid_gene_sets.values(), otherwise, the dict's value is dict_value object, which is an iterable view, not a standard list.
    valid_gene_sets_indices=list(valid_gene_sets.values())


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
    
    
    # Process results based on score_method
    gene_set_names=list(valid_gene_sets.keys())
    
    if score_method == 'piaso':
        # For piaso, results should be tuples of (scores, nlog10_pvals)
        first_result = results[0]
        is_tuple_result = isinstance(first_result, tuple) and len(first_result) == 2
        
        if is_tuple_result:
            # Results are in tuple format (score, pvals)
            scores, nlog10_pvals = zip(*results)
            
            # Only process p-values if needed and if they exist
            nlog10_pval_matrix = None
            if return_pvals and any(p is not None for p in nlog10_pvals[:1]):
                nlog10_pval_matrix = np.vstack(nlog10_pvals).T
        else:
            # Not tuple format - direct scores
            scores = results
            nlog10_pval_matrix = None
            
        score_matrix = np.vstack(scores).T
        
        if return_pvals:
            return score_matrix, gene_set_names, nlog10_pval_matrix
        else:
            return score_matrix, gene_set_names
    else:
        # For scanpy, results only contain scores
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

    batch_mask = shared_data['obs'][batch_key] == batch_i
    adata = sc.AnnData(matrix[batch_mask.to_numpy()])
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
def calculateScoreParallel_multiBatch(
    adata: sc.AnnData,
    batch_key: str,
    marker_gene: pd.DataFrame,
    marker_gene_n_groups_indices: list,
    score_method: Literal["scanpy", "piaso"],
    score_layer: str = None,
    max_workers: int = 8,
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
        Maximum number of parallel workers to use.
    score_layer : str
        The layer of `adata` to use for scoring.
    score_method : {'scanpy', 'piaso'}, optional
        The method used for gene set scoring. Must be either 'scanpy' (default) or 'piaso'.
        - 'scanpy': Uses the Scanpy's built-in gene set scoring method.
        - 'piaso': Uses the PIASO's gene set scoring method, which is more robust to sequencing depth variations.
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
    >>> import scanpy as sc
    >>> from piaso
    >>> adata = sc.read_h5ad('example_data.h5ad')
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

    # Process batches in parallel
    # batch_list = np.unique(adata.obs[batch_key])
    
    # Create Map
    batch_order_map = {batch: i for i, batch in enumerate(np.unique(adata.obs[batch_key]))}
    
    score_list_collection = []
    cellbarcode_info = []
    gene_set_names_collection = []

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
                # score_list, cell_barcodes, gene_set_names = future.result()
                # score_list_collection.append(score_list)
                # cellbarcode_info.append(cell_barcodes)
                
        # Sort results
        raw_results.sort(key=lambda x: batch_order_map[x[0]])
        

        # Unpack safely in the correct order
        for _, score_list, cell_barcodes, gene_names in raw_results:
            score_list_collection.append(score_list)
            cellbarcode_info.append(cell_barcodes)
            gene_set_names_collection.append(gene_names)

    finally:
        # Clean up shared memory
        shared_data['shm_data'].close()
        shared_data['shm_data'].unlink()
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

    # adata = sc.AnnData(matrix)
    # adata.obs = shared_data['obs'].copy()
    # ### No need to create a adata_i, because adata here is rebuilt from the matrix
    # # Filter the AnnData object for the current batch
    # adata = adata[adata.obs[batch_key] == batch_i].copy()
    

    ### Only directly select slices
    batch_mask = shared_data['obs'][batch_key] == batch_i
    adata = sc.AnnData(matrix[batch_mask.to_numpy()])
    adata.obs = shared_data['obs'][batch_mask.to_numpy()].copy()
    
    
    # Temporarily suppress Scanpy verbosity
    original_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0  # Suppress messages
    
    try:
        # Extract marker gene signatures
        if groupby is None:
            # Run SVD lazily
            if layer=='infog':
                ### in this case, adata.X will be the raw UMI counts, the infog_layer will be transferred as adata.X in this function input
                runSVDLazy(
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
                runSVDLazy(
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
            sc.pp.neighbors(adata, use_rep='X_svd', n_neighbors=15, random_state=random_seed, knn=True, method="umap")
            # sc.pp.neighbors(adata, use_rep='X_svd')
            

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                sc.tl.leiden(adata, resolution=resolution, key_added='gdr_local', random_state=random_seed)
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

    finally:
        # Restore original verbosity
        sc.settings.verbosity = original_verbosity

    # return marker_gene
    return batch_i, marker_gene

### To record the progress
from concurrent.futures import as_completed
from tqdm import tqdm
import warnings
from scipy.sparse import issparse

def runCOSGParallel(
    adata: sc.AnnData,
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
    >>> import scanpy as sc
    >>> import piaso
    >>> adata = sc.read_h5ad('example_data.h5ad')
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



def runGDRParallel(
    adata,
    batch_key:str=None,
    groupby:str=None,
    n_gene:int=30,
    mu:float=1.0,
    layer:str=None,
    score_layer:str=None,
    infog_layer:str=None,
    use_highly_variable:bool=True,
    n_highly_variable_genes:int=5000,
    n_svd_dims:int=50,
    n_svd_iter:int=7,
    resolution:float=1.0,
    scoring_method:str=None,
    key_added:str=None,
    max_workers:int=8,
    calculate_score_multiBatch:bool=False,
    verbosity: int=0,
    random_seed:int=1927
):
    """
    Run GDR (marker Gene-guided dimensionality reduction) in parallel using multi-cores and shared memeory.

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
        Layer in `adata.layers` to use for the analysis. Defaults to None, which uses `adata.X`.

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
        Whether to calculate gene scores across multiple adata batches (if `batch_key` is specified). Defaults to False.

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
    >>> import scanpy as sc
    >>> import piaso
    >>> 
    >>> adata = sc.read_h5ad("example.h5ad")
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


    sc.settings.verbosity=0
    
    ### Check the scoring method, improve this part of codes later
    if scoring_method is not None:
        # Validate scoring_method
        if scoring_method not in {"scanpy", "piaso"}:
            raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be either 'scanpy' or 'piaso'.")
    else:
        ### Use piaso's scoring method as the default
        scoring_method='piaso'
    
#     ### To store the scores
#     score_list_collection_collection=[]
    
    if batch_key is None:
        nbatches=1
    else:
        batch_list=np.unique(adata.obs[batch_key])
        nbatches=len(batch_list)
    
    if nbatches==1:
        ### Calculate the clustering labels if there is no specified clustering labels to use
        if groupby is None:
            ### Run SVD in a lazy mode
            runSVDLazy(
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
            sc.settings.verbosity=0
            
            ### Run clustering
            sc.pp.neighbors(
                adata,
                use_rep='X_svd',
                n_neighbors=15,
                random_state=random_seed,
                knn=True,
                method="umap")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)                            
                sc.tl.leiden(adata, resolution=resolution, key_added='gdr_local')
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
                expressed_pct=0.1,
                remove_lowly_expressed=True,
                n_genes_user=n_gene,
                groupby=groupby
                     )
        else:
            cosg.cosg(adata,
                key_added='cosg',
                mu=mu,
                expressed_pct=0.1,
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
        #     sc.tl.score_genes(adata_tmp,
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
        sc.settings.verbosity=3

        ### Make sure the order are matched to the adata
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
                score_list, gene_set_names=calculateScoreParallel(
                    adata[adata.obs[batch_key]==batch_u],
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
        
        sc.settings.verbosity=3
        
        ### Make sure the order are matched to the adata
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


