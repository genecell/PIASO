from ._runSVD import runSVDLazy

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
    layer:str='log1p',
    score_layer:str='log1p',
    n_svd_dims:int=50,
    resolution:float=1.0,
    scoring_method:str=None,
    key_added:str=None,
    verbosity: int=0,         
):

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
                trim_value=None,
                key_added='X_svd'
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
                sc.tl.score_genes(adata_tmp,
                              marker_gene_i,
                              score_name='markerGeneFeatureScore_i')
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
                    trim_value=None,
                    key_added='X_svd'
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
                        sc.tl.score_genes(adata_u,
                                      marker_gene_i,
                                      score_name='markerGeneFeatureScore_i')
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
        print('The cell embeddings calculated by GDR were saved as `'+key_added+'` in adata.obsm.')
    else:
        adata.obsm['X_gdr']=marker_gene_scores.values
        print('The cell embeddings calculated by GDR were saved as `X_gdr` in adata.obsm.')

        
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


def _calculate_gene_set_score_shared(gene_indices, metadata, score_name, is_sparse):
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


        sc.settings.verbosity = 0
        # ### Globally supress warning
        # warnings.filterwarnings("ignore", category=FutureWarning)
        # # # Suppress FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sc.tl.score_genes(adata_tmp, adata_tmp.var.index[gene_indices].tolist(), score_name=score_name)

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

        
def calculateScoreParallel(
    adata,
    gene_set,
    score_layer=None,
    max_workers=None):
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
    from scipy.sparse import isspmatrix_csr
    from multiprocessing import shared_memory
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    import numpy as np
    import scanpy as sc
    
    
    # Determine the input matrix
    if score_layer is not None:
        data = adata.layers[score_layer]
    else:
        data = adata.X

    # Set up shared memory
    if isspmatrix_csr(data):
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
        is_sparse=is_sparse
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



def runGDRParallel(
    adata,
    batch_key:str=None,
    groupby:str=None,
    n_gene:int=30,
    mu:float=1.0,
    use_highly_variable:bool=True,
    n_highly_variable_genes:int=5000,
    layer:str='log1p',
    score_layer:str='log1p',
    n_svd_dims:int=50,
    resolution:float=1.0,
    scoring_method:str=None,
    key_added:str=None,
    max_workers:int=8,
    verbosity: int=0,         
):

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
                trim_value=None,
                key_added='X_svd'
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
            score_layer=score_layer,
            max_workers=max_workers
        )
        
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
                    trim_value=None,
                    key_added='X_svd'
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
            
            ### Scoring the geneset among different batches
            for batch_u in batch_list:
                adata_u=adata[adata.obs[batch_key]==batch_u].copy()
                
                ### Single-CPU version of GDR
                # score_list=[]
                # # adata_u.X=adata_u.layers['log1p'] ## do not use log1p, sometimes raw counts is better here
                # if score_layer is not None:
                #     adata_u.X=adata_u.layers[score_layer]
                # for i in marker_gene.columns:
                #     marker_gene_i=marker_gene[i].values
                #     sc.tl.score_genes(adata_u,
                #                   marker_gene_i,
                #                   score_name='markerGeneFeatureScore_i')
                #     ### Need to add the .copy()
                #     score_list.append(adata_u.obs['markerGeneFeatureScore_i'].values.copy())
                # score_list=np.vstack(score_list).T
                
                ### Use parallel computing of the gene set scores
                score_list, gene_set_names=calculateScoreParallel(
                    adata_u,
                    gene_set=marker_gene,
                    score_layer=score_layer,
                    max_workers=max_workers
                )
                
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
    transformer = TruncatedSVD(n_components=adata_combine.obsm['X_gdr'].shape[1], random_state=20)
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