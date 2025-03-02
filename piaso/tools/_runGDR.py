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


def runGDR(
    adata,
    batch_key:str=None,
    groupby:str=None,
    n_gene:int=30,
    mu:float=1.0,
    use_highly_variable:bool=True,
    n_highly_variable_genes:int=5000,
    layer:str=None,
    score_layer:str=None,
    infog_layer:str=None,
    n_svd_dims:int=50,
    resolution:float=1.0,
    scoring_method:str=None,
    key_added:str=None,
    verbosity: int=0,
    random_seed:int=1927
):
    
    ### Check the scoring method, improve this part of codes later
    if scoring_method is not None:
        # Validate scoring_method
        if scoring_method not in {"scanpy", "piaso"}:
            raise ValueError(f"Invalid scoring_method: '{scoring_method}'. Must be either 'scanpy' or 'piaso'.")
    else:
        ### Use scanpy's scoring method as the default
        scoring_method='scanpy'
    

    sc.settings.verbosity=0
    score_list_collection_collection=[]
    
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
                layer=layer,
                infog_layer=infog_layer,
                infog_trim=True,
                key_added='X_svd',
                random_state=random_seed
            )
            ### Because the verbosity will be reset in the above function
            sc.settings.verbosity=0
            
            ### Run clustering
            sc.pp.neighbors(adata,
                use_rep='X_svd',
                n_neighbors=15,random_state=10,knn=True,
                method="umap")
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

        sc.settings.verbosity=3

        ### Make sure the order are matched to the adata
        marker_gene_scores=pd.DataFrame(marker_gene_scores)
        marker_gene_scores.index=adata.obs_names
        # marker_gene_scores.index=np.hstack([adata_list[0].obs_names.values, adata_list[1].obs_names.values])
        marker_gene_scores=marker_gene_scores.loc[adata.obs_names]
                
    ### Have multiple batches    
    else:
        ### Store the cell barcodes info
        cellbarcode_info=list()
        for batch_i in batch_list:
            adata_i=adata[adata.obs[batch_key]==batch_i].copy()
        
            ### Extract marker gene signatures
            ### Calculate clustering labels if no clustering info was specified
            if groupby is None:
                ### Run SVD in a lazy mode
                runSVDLazy(
                    adata_i,
                    copy=False,
                    n_components=n_svd_dims,
                    n_top_genes=n_highly_variable_genes,
                    use_highly_variable=use_highly_variable,
                    verbosity=verbosity,
                    batch_key=None, ### Need to set as None, because the SVD is calculated in each batch separately
                    scale_data=False,
                    layer=layer,
                    infog_layer=infog_layer,
                    infog_trim=True,
                    key_added='X_svd',
                    random_state=random_seed
                )
                ### Because the verbosity will be reset in the above function, the good way is to remember the previous state of verbosity
                sc.settings.verbosity=0
                
                ### Run clustering
                sc.pp.neighbors(adata_i,
                    use_rep='X_svd',
                    n_neighbors=15,random_state=10,knn=True,
                    method="umap")
                sc.tl.leiden(adata_i,resolution=resolution,key_added='gdr_local')
                groupby_i='gdr_local'    
            else:
                groupby_i=groupby    
            
            if verbosity>0:
                print('Processing the batch ', batch_i ,' which contains ',len(np.unique(adata_i.obs[groupby_i])), ' clusters.')
            cellbarcode_info.append(adata_i.obs_names.values)
            ### Run marker gene identification
            if layer is not None:
                cosg.cosg(adata_i,
                    key_added='cosg',
                    use_raw=False,
                    layer=layer,
                    mu=mu,
                    expressed_pct=0.1,
                    remove_lowly_expressed=True,
                    n_genes_user=n_gene,
                    groupby=groupby_i
                         )
            else:
                cosg.cosg(adata_i,
                    key_added='cosg',
                    mu=mu,
                    expressed_pct=0.1,
                    remove_lowly_expressed=True,
                    n_genes_user=n_gene,
                    groupby=groupby_i
                         )

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
        sc.settings.verbosity=3
        
        ### Make sure the order are matched to the adata
        marker_gene_scores=pd.DataFrame(marker_gene_scores)
        marker_gene_scores.index=np.hstack(cellbarcode_info)
        marker_gene_scores=marker_gene_scores.loc[adata.obs_names]
    
    ### Set the low-dimensional representations for the integrated dataset
    if key_added is not None:
        adata.obsm[key_added]=marker_gene_scores.values
        print(f'The cell embeddings calculated by GDR were saved as `{key_added}` in adata.obsm.')
    else:
        adata.obsm['X_gdr']=marker_gene_scores.values
        print(f'The cell embeddings calculated by GDR were saved as `X_gdr` in adata.obsm.')

        
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
    np.ndarray
        Scores for the gene set.
    """
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

        
        adata_tmp = sc.AnnData(X=X, dtype=X.dtype)
        
        # print(adata_tmp.X.dtype)
        # print(adata_tmp.X.data[:10])
        # print(adata_tmp.var.index[gene_indices].tolist())
        
        sc.settings.verbosity = 0
        # ### Globally supress warning
        # warnings.filterwarnings("ignore", category=FutureWarning)
        # # # Suppress FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if score_method=='scanpy':
                sc.tl.score_genes(adata_tmp, adata_tmp.var.index[gene_indices].tolist(), score_name=score_name, random_state=random_seed)
            elif score_method=='piaso':
                ## Set layer to None, because the scoring layer is already constructed as the adata.X
                score(adata_tmp, gene_list=adata_tmp.var.index[gene_indices].tolist(), key_added=score_name, layer=None, random_seed=random_seed)
            else:
                raise ValueError(f"Invalid score_method: '{score_method}'. Must be either 'scanpy' or 'piaso'.")

        scores=adata_tmp.obs[score_name].values.copy()
 
        return scores
    
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

def calculateScoreParallel(
    adata,
    gene_set,
    score_method: Literal["scanpy", "piaso"],
    random_seed:int=1927,
    score_layer=None,
    max_workers=None,
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
        
        
    Returns
    -------
    tuple
        - np.ndarray: A 2D array where each column contains the scores for a gene set.
            If `gene_set` is a dictionary, columns correspond to the keys of the dictionary.
            If `gene_set` is a list of lists, columns correspond to the index of the sublists.
            If `gene_set` is pandas.DataFrame, columns correspond to the index of the columns.
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
    
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        score_list = list(executor.map(partial_func, valid_gene_sets_indices))

    # Clean up shared memory
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
    
    # Convert score_list to a 2D array
    score_matrix = np.vstack(score_list).T
    gene_set_names=list(valid_gene_sets.keys())
    return score_matrix, gene_set_names




### Calculate gene set score for different batches, separately, but in parallel
def _calculateScoreParallel_single_batch(batch_key,  shared_data, batch_i, marker_gene, marker_gene_n_groups_indices, max_workers, score_method, random_seed):
    """
    Process a single batch to calculate scores, different marker gene sets will be calculated in parallel with `calculateScoreParallel` function.
    """
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
    # #### Need to reset the gene names, because this will be used in gene set scoring, need to know the genes
    # adata.var_names=shared_data["var_names"].copy()
    # ### No need to create a adata_i, because adata here is rebuilt from the matrix
    # # Filter the AnnData object for the current batch
    # adata = adata[adata.obs[batch_key] == batch_i].copy()
    
    ### Only directly select slices
    batch_mask = shared_data['obs'][batch_key] == batch_i
    adata = sc.AnnData(matrix[batch_mask])
    adata.obs = shared_data['obs'][batch_mask].copy()
    #### Need to reset the gene names, because this will be used in gene set scoring, need to know the genes
    adata.var_names=shared_data["var_names"].copy()
    
    
    # Compute gene set scores, in parallel for different gene sets
    score_list, gene_set_names = calculateScoreParallel(
        adata,
        gene_set=marker_gene,
        score_method=score_method,
        score_layer=None, ## As the score layer already used in setting up the shared memory
        max_workers=max_workers,
        random_seed=random_seed
    )
    # print(gene_set_names)
    # print(marker_gene.columns)
    # print('Gene set order not changed:', np.array_equal(gene_set_names, marker_gene.columns))

    # Normalize scores
    # Block-wise L2-normalization, maybe could be better implemented with higher efficiency
    # comparing different groups' marker gene scores
    score_list = normalize(score_list, norm="l2", axis=0)
    for start, end in zip(marker_gene_n_groups_indices[:-1], marker_gene_n_groups_indices[1:]):
        score_list[:, start:end] = normalize(score_list[:, start:end], norm="l2", axis=1)

        
    # Retrieve cell barcodes
    cell_barcodes = adata.obs_names.values
    
    return score_list, cell_barcodes, gene_set_names

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
    batch_list = np.unique(adata.obs[batch_key])
    score_list_collection = []
    cellbarcode_info = []
    gene_set_names = None

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _calculateScoreParallel_single_batch,
                    batch_key,
                    shared_data,
                    batch_i,
                    marker_gene,
                    marker_gene_n_groups_indices,
                    max_workers,
                    score_method,
                    random_seed
                ) for batch_i in batch_list
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating cell embeddings/scores", unit="batch"):
                score_list, cell_barcodes, gene_set_names = future.result()
                score_list_collection.append(score_list)
                cellbarcode_info.append(cell_barcodes)

    finally:
        # Clean up shared memory
        shared_data['shm_data'].close()
        shared_data['shm_data'].unlink()
        if 'shm_indices' in shared_data:
            shared_data['shm_indices'].close()
            shared_data['shm_indices'].unlink()
            shared_data['shm_indptr'].close()
            shared_data['shm_indptr'].unlink()

    return score_list_collection, cellbarcode_info, gene_set_names



#### Function to process the runCOSGParallel in each individual batches, and the shared memory will be used
import os
import sys
import logging

def _runCOSGParallel_single_batch(
    batch_key, shared_data, batch_i, groupby, n_svd_dims, n_highly_variable_genes, verbosity, resolution, mu, n_gene, use_highly_variable, layer, random_seed):
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
    adata = sc.AnnData(matrix[batch_mask])
    adata.obs = shared_data['obs'][batch_mask].copy()
    
    
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
                    layer=None,
                    infog_layer=None,
                    infog_trim=True,
                    key_added='X_svd',
                    random_state=random_seed
                )
            ### Because in runSVDLazy, it will reset the sc.settings.verbosity, so we need to set the verbosity again here
            sc.settings.verbosity = 0  # Suppress messages

            # Run clustering
            sc.pp.neighbors(adata, use_rep='X_svd', n_neighbors=15, random_state=10, knn=True, method="umap")
            sc.tl.leiden(adata, resolution=resolution, key_added='gdr_local')
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

    return marker_gene

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
            warnings.warn(
                "Please set 'infog_layer' to the UMI counts matrix. "
                "If adata.X already is the UMI counts matrix, you can safely ignore this message."
            )
            gene_expression_data = adata.X
        else:
            ### Set it to the infog_layer, which should contain the UMI counts matrix, and will be used for the SVDlazy function
            gene_expression_data = adata.layers[infog_layer]
    else:
        gene_expression_data = adata.layers[layer]
    

    
    # Determine matrix type and set up shared memory
    if issparse(gene_expression_data):
        if not isinstance(gene_expression_data, csr_matrix):
            raise ValueError("For the gene expression matrix, if you want to use sparse matrix, the format must be in CSR format.")
        shared_data = _setup_shared_memory_sparse(gene_expression_data)
    else:
        shared_data = _setup_shared_memory_dense(gene_expression_data)
        
        
        
    # Update shared_data['obs'] to handle groupby=None
    shared_data['obs'] = adata.obs[[batch_key] + ([groupby] if groupby else [])].copy()

    # Use ProcessPoolExecutor for parallel processing
    marker_genes = []
    batch_n_groups = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_i in batch_list:
            futures.append(
                executor.submit(
                    _runCOSGParallel_single_batch, batch_key, shared_data, batch_i, groupby,
                    n_svd_dims, n_highly_variable_genes, verbosity, resolution,
                    mu, n_gene, use_highly_variable, layer, random_seed
                )
            )
            
        # futures = {
        #     executor.submit(
        #         process_batch, batch_key, shared_data, batch_i, groupby,
        #         n_svd_dims, n_highly_variable_genes, verbosity, resolution,
        #         mu, n_gene, use_highly_variable
        #     ): batch_i for batch_i in batch_list
        # }
        
        # for future in futures:
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating marker genes in batches", unit="batch"):
            marker_gene = future.result()
            marker_genes.append(marker_gene)
            ### Return the number of groups for each batch
            batch_n_groups.append(marker_gene.shape[1])
    
    # Merge all marker gene DataFrames into one
    marker_genes = pd.concat(marker_genes, axis=1)

    # Convert gene indices to int
    marker_genes = marker_genes.astype(int)

    # Build a mapping dictionary for gene index to names
    index_to_name_mapping = {i: name for i, name in enumerate(adata.var.index)}

    # Convert gene indices to names if required
    if return_gene_names:
        marker_genes = marker_genes.applymap(lambda idx: index_to_name_mapping.get(idx, idx))

    # Clean up shared memory
    shared_data['shm_data'].close()
    shared_data['shm_data'].unlink()
    if 'shm_indices' in shared_data:
        shared_data['shm_indices'].close()
        shared_data['shm_indices'].unlink()
        shared_data['shm_indptr'].close()
        shared_data['shm_indptr'].unlink()
    
    
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

    resolution : float, optional
        Resolution parameter for de novo clustering. Defaults to 1.0. Only effective when `groupby=None`.

    scoring_method : str, optional
        Specifies the gene set scoring method used to compute gene scores.

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
        ### Use scanpy's scoring method as the default
        scoring_method='scanpy'
    
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
                layer=layer,
                infog_layer=infog_layer,
                infog_trim=True,
                key_added='X_svd',
                random_state=random_seed
            )
            ### Because the verbosity will be reset in the above function
            sc.settings.verbosity=0
            
            ### Run clustering
            sc.pp.neighbors(adata,
                use_rep='X_svd',
                n_neighbors=15,random_state=10,knn=True,
                method="umap")
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
            score_list_collection, cellbarcode_info, geneSetNames=calculateScoreParallel_multiBatch(
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
    
    ### Set the low-dimensional representations for the integrated dataset
    if key_added is not None:
        adata.obsm[key_added]=marker_gene_scores.values
        print('The cell embeddings calculated by GDR were saved as `'+key_added+'` in adata.obsm.')
    else:
        adata.obsm['X_gdr']=marker_gene_scores.values
        print('The cell embeddings calculated by GDR were saved as `X_gdr` in adata.obsm.')


### Predict cell types using GDR
### Firstly packaged as a function on April 10, 2024
def predictCellTypeByGDR(
    adata,
    adata_ref,
    layer:str='log1p',
    layer_reference:str='log1p',
    reference_groupby:str='CellTypes',
    query_groupby:str='Leiden',
    mu:float=10.0,
    n_genes:int=15,
    return_integration:bool=False,
    use_highly_variable:bool=True,
    n_highly_variable_genes:int=5000,
    n_svd_dims:int=50, ### SVD
    resolution:float=1.0,
    scoring_method:str=None,
    key_added:str=None,
    verbosity: int=0

):
    """
    Predicts cell types in a query dataset (`adata`) using the GDR dimensionality reduction method based on a reference dataset (`adata_ref`).
    To use GDR for dimensionality reduction, please refer to `piaso.tl.runGDR` or `piaso.tl.runGDRParallel`.

    Parameters
    ----------
    adata : AnnData
        The query single-cell AnnData object for which cell types are to be predicted.

    adata_ref : AnnData
        The reference single-cell AnnData object with known cell type annotations.

    layer : str, optional (default: 'log1p')
        The layer in `adata` to use for gene expression data. If `None`, uses the `.X` matrix.

    layer_reference : str, optional (default: 'log1p')
        The layer in `adata_ref` to use for reference gene expression data. If `None`, uses the `.X` matrix.

    reference_groupby : str, optional (default: 'CellTypes')
        The column in `adata_ref.obs` used to define reference cell type groupings.

    query_groupby : str, optional (default: 'Leiden')
        The column in `adata.obs` used to for GDR dimensionality reduction, such as clusters identified using Leiden or Louvain algorithms.

    mu : float, optional (default: 10.0)
        A regularization parameter for controlling the gene expression specificity, used in COSG (marker gene identification) and GDR.

    n_genes : int, optional (default: 15)
        The number of top specific genes per group, used in COSG and GDR.

    return_integration : bool, optional (default: False)
        If `True`, the function will return the integrated low-dimensional cell embeddings of the query dataset and reference dataset.

    use_highly_variable : bool, optional (default: True)
        Whether to use highly variable genes, used in GDR.

    n_highly_variable_genes : int, optional (default: 5000)
        The number of highly variable genes to select, if `use_highly_variable` is `True`, used in GDR.

    n_svd_dims : int, optional (default: 50)
        The number of dimensions to retain during SVD, used in GDR.

    resolution : float, optional (default: 1.0)
        Resolution parameter for clustering, used in GDR.

    scoring_method : str, optional (default: None)
        The method used for gene set scoring, used in GDR.

    key_added : str, optional (default: None)
        A key to add the predicted cell types or integration results to `adata.obs`. If `None`, `CellTypes_gdr` will be used.

    verbosity : int, optional (default: 0)
        The level of logging output. Higher values produce more detailed logs for debugging and monitoring progress.

    Returns
    -------
    None or AnnData
        If `return_integration` is `True`, returns an AnnData object of merged reference and query datasets with integrated
        cell embeddings and predicted cell types. Otherwise, updates `adata` in place with the predicted cell types.

    Example
    -------
    >>> import scanpy as sc
    >>> # Load query dataset
    >>> adata = sc.read_h5ad("query_data.h5ad")
    >>> 
    >>> # Load reference dataset with known cell type annotations
    >>> adata_ref = sc.read_h5ad("reference_data.h5ad")
    >>> 
    >>> # Predict cell types for the query dataset
    >>> piaso.tl.predictCellTypeByGDR(
    >>>     adata=adata,
    >>>     adata_ref=adata_ref,
    >>>     layer='log1p',
    >>>     layer_reference='log1p',
    >>>     reference_groupby='CellTypes',
    >>>     query_groupby='Leiden',
    >>>     mu=10.0,
    >>>     n_genes=20,
    >>>     return_integration=False,
    >>>     use_highly_variable=True,
    >>>     n_highly_variable_genes=3000,
    >>>     n_svd_dims=50,
    >>>     resolution=0.8,
    >>>     key_added='CellTypes_gdr',
    >>>     verbosity=0
    >>> )
    >>> 
    >>> # Access the predicted cell types in the query dataset
    >>> print(adata.obs['CellTypes_gdr'])
    """
    
    sc.settings.verbosity=verbosity
    
    if layer_reference is not None:
        adata_ref.X=adata_ref.layers[layer_reference]
    shared_genes=np.intersect1d(adata_ref.var_names, adata.var_names)
    ### Make a copy of the reference anndata
    adata_ref=adata_ref[:,shared_genes].copy()
    
    adata_ref.obs['gdr_by']=adata_ref.obs[reference_groupby]
    adata.obs['gdr_by']=adata.obs[query_groupby]
    
    
    if layer is not None:
        adata.X=adata.layers[layer]
        
    adata_combine=sc.AnnData.concatenate(adata_ref, adata[:,adata_ref.var_names])
    ### Needs to be categorical variable
    adata_combine.obs['gdr_by']=adata_combine.obs['gdr_by'].astype('category')
    
    ### Already pre-determined
    # if layer is not None:
        # adata_combine.X=adata_combine.layers[layer]
    
    print("Running GDR for the query dataset and the reference dataset:")

    runGDR(
        adata_combine,
        batch_key='batch',
        n_gene=n_genes,
        groupby='gdr_by',
        mu=mu,
        use_highly_variable=use_highly_variable,
        n_highly_variable_genes=n_highly_variable_genes,
        layer=None, ### Already included or converted in the adata.X
        score_layer=None, ### Already included or converted in the adata.X
        n_svd_dims=n_svd_dims,
        resolution=resolution,
        scoring_method=scoring_method,
        verbosity=verbosity
    )
    
    ### Add a SVD layer
    from sklearn.decomposition import TruncatedSVD
    transformer = TruncatedSVD(n_components=adata_combine.obsm['X_gdr'].shape[1], random_state=10)
    adata_combine.obsm['X_gdr_svd'] = transformer.fit_transform(adata_combine.obsm['X_gdr'])

    
    ### Run Harmony
    import scanpy.external as sce
    sce.pp.harmony_integrate(adata_combine,
                                 key='batch',
                                 basis='X_gdr_svd',
                                 adjusted_basis='X_gdr_harmony')
    
    if return_integration:
        sc.pp.neighbors(adata_combine,
                    use_rep='X_gdr_harmony',
                   n_neighbors=15,random_state=10,knn=True,
                    method="umap")
        sc.tl.umap(adata_combine)

    from sklearn import svm
    print("Predicting cell types:")
    clf = svm.SVC(kernel='rbf', 
                  class_weight='balanced',
                  # decision_function_shape='ovo'
                 )

    from sklearn import preprocessing
    clf.fit(adata_combine[adata_combine.obs['batch']=='0'].obsm['X_gdr_harmony'],
            adata_combine[adata_combine.obs['batch']=='0'].obs[reference_groupby])
    
    if key_added is not None:
        adata.obs[key_added]=clf.predict(adata_combine[adata_combine.obs['batch']=='1'].obsm['X_gdr_harmony'])
        print("All finished. The predicted cell types are saved as `"+key_added+"` in adata.obs.")
    else:
        adata.obs['CellTypes_gdr']=clf.predict(adata_combine[adata_combine.obs['batch']=='1'].obsm['X_gdr_harmony'])
        print("All finished. The predicted cell types are saved as `CellTypes_gdr` in adata.obs.")
    
    
    sc.settings.verbosity=3
    
    if return_integration:
        return(adata_combine)