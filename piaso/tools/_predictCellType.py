from ._runGDR import runGDR, calculateScoreParallel

import numpy as np
import pandas as pd
from scipy import stats
import scanpy as sc
from sklearn.decomposition import TruncatedSVD

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


   
    
import numpy as np
import pandas as pd
from scipy import stats

def smoothCellTypePrediction(
    adata, 
    groupby: str, 
    use_rep: str = 'X_pca', 
    k_nearest_neighbors: int = 5, 
    return_confidence: bool = False,
    inplace: bool = True,
    use_existing_adjacency_graph: bool = True,
    use_faiss: bool = False,
    key_added: str = None,
    verbosity: int = 1,
    n_jobs: int = -1
):
    """
    Smooth cell type predictions using k-nearest neighbors in a low-dimensional embedding.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell data
    groupby : str
        Key in adata.obs containing the cell type predictions to smooth
    use_rep : str, default='X_pca'
        Key in adata.obsm containing the low-dimensional embedding to use for finding neighbors
    k_nearest_neighbors : int, default=5
        Number of neighbors to consider (including the cell itself)
    return_confidence : bool, default=False
        Whether to return confidence scores (proportion of neighbors with the majority label)
    inplace : bool, default=True
        Whether to modify adata inplace or return a copy
    use_existing_adjacency_graph : bool, default=True
        Whether to use existing neighborhood graph (adata.obsp['connectivities']) if available
    use_faiss : bool, default=False
        Whether to use FAISS for faster neighbor search (requires faiss package)
    key_added : str or None, default=None
        If provided, use this key as the output key in adata.obs instead of '{groupby}_smoothed'
    verbosity : int, default=1
        Level of verbosity (0=no output, 1=basic info, 2=detailed info)
    n_jobs : int, default=-1
        Number of jobs for parallel processing. -1 means using all processors.
        
    Returns
    -------
    If inplace=True:
        None, but adds 'groupby_smoothed' (or key_added) to adata.obs
        If return_confidence=True, also adds 'groupby_confidence' (or key_added_confidence) to adata.obs
    If inplace=False:
        Copy of adata with added columns
        
    Examples
    --------
    >>> import scanpy as sc
    >>> import piaso
    >>> 
    >>> # Basic usage
    >>> piaso.tl.smoothCellTypePrediction(
    ...     adata, 
    ...     groupby='CellTypes_pred',
    ...     use_rep='X_pca',
    ...     key_added='CellTypes_pred_smoothed'
    ... )
    >>> 
    >>> # With confidence scores
    >>> piaso.tl.smoothCellTypePrediction(
    ...     adata, 
    ...     groupby='CellTypes_pred',
    ...     k_nearest_neighbors=15,
    ...     return_confidence=True,
    ...     key_added='CellTypes_pred_smoothed'
    ... )
    """
    # Input validation
    if groupby not in adata.obs:
        raise ValueError(f"Group key '{groupby}' not found in adata.obs")
    
    if verbosity > 0:
        print(f"Smoothing cell type predictions from '{groupby}' using {k_nearest_neighbors}-nearest neighbors")
    
    # Get original predictions
    original_predictions = adata.obs[groupby]
    
    # Create result
    adata = adata.copy() if not inplace else adata
    
    # Check if we can use existing neighborhood graph
    has_neighbors = 'neighbors' in adata.uns and 'connectivities' in adata.obsp and use_existing_adjacency_graph
    
    if verbosity > 1:
        if has_neighbors:
            print("Using existing neighborhood graph from adata.obsp['connectivities']")
        else:
            print(f"Computing new neighborhood graph using embedding from adata.obsm['{use_rep}']")
            if use_faiss:
                print("Using FAISS for accelerated nearest neighbor search")
            else:
                print(f"Using scikit-learn NearestNeighbors with n_jobs={n_jobs}")
    
    # Initialize arrays for storing results
    smoothed_predictions = np.empty(len(adata), dtype=object)
    if return_confidence:
        confidence_scores = np.empty(len(adata))
    
    # Track cells with insufficient neighbors
    cells_with_insufficient_neighbors = 0
    
    if has_neighbors:
        # Use existing connectivity matrix from adata (CSR format)
        # Check if the connectivity matrix is in CSR format
        from scipy.sparse import csr_matrix
        if not isinstance(adata.obsp['connectivities'], csr_matrix):
            raise ValueError(
                f"Expected adata.obsp['connectivities'] to be a CSR matrix, but got {type(adata.obsp['connectivities']).__name__}. "
                f"Please convert it using `.tocsr()` method."
            )
        conn_matrix = adata.obsp['connectivities']
        
        if verbosity > 1:
            print(f"Processing {len(adata)} cells using CSR connectivity matrix...")
        
        # Process each cell
        for i in range(conn_matrix.shape[0]):
            # Direct access to CSR data structures - memory efficient
            row_start, row_end = conn_matrix.indptr[i], conn_matrix.indptr[i+1]
            data = conn_matrix.data[row_start:row_end]
            col_indices = conn_matrix.indices[row_start:row_end]
            
            # Check if we have enough connections
            if len(data) < k_nearest_neighbors:
                cells_with_insufficient_neighbors += 1
                # Use all available connections
                top_k = col_indices
            else:
                # Get top k connections by strength
                top_k_idx = np.argsort(data)[::-1][:k_nearest_neighbors]
                top_k = col_indices[top_k_idx]
            
            # Get neighbor types for this cell
            neighbor_types = original_predictions.iloc[top_k] if hasattr(original_predictions, 'iloc') else original_predictions[top_k]
            
            # Get most common type
            try:
                most_common = stats.mode(neighbor_types, keepdims=False)
                majority_type = most_common.mode
            except TypeError:
                most_common = stats.mode(neighbor_types)
                majority_type = most_common[0][0]
            
            # Store results directly
            smoothed_predictions[i] = majority_type
            if return_confidence:
                confidence_scores[i] = np.mean(neighbor_types == majority_type)
            
            # Print progress periodically
            if verbosity > 1 and i > 0 and i % 10000 == 0:
                print(f"  Processed {i}/{conn_matrix.shape[0]} cells...")
        
    else:
        # We need to compute neighbors
        if use_rep not in adata.obsm:
            raise ValueError(f"Embedding key '{use_rep}' not found in adata.obsm")
        
        # Get embedding
        X_embed = adata.obsm[use_rep]
        
        if verbosity > 1:
            print(f"Embedding shape: {X_embed.shape}")
        
        if use_faiss and k_nearest_neighbors < len(adata):
            try:
                import faiss
                # Use FAISS for faster neighbor search
                n_features = X_embed.shape[1]
                
                # Check if already float32 before conversion to avoid unnecessary memory allocation
                if X_embed.dtype == np.float32:
                    X_embed_float32 = X_embed
                else:
                    X_embed_float32 = X_embed.astype('float32')
                
                # Build index
                index = faiss.IndexFlatL2(n_features)
                index.add(X_embed_float32)
                
                # Search for nearest neighbors
                distances, indices = index.search(X_embed_float32, k_nearest_neighbors)
                
                if verbosity > 1:
                    print(f"Processing {len(adata)} cells to determine majority labels...")
                
                # Process each cell
                for i in range(len(adata)):
                    neighbor_indices_i = indices[i]
                    neighbor_types = original_predictions.iloc[neighbor_indices_i] if hasattr(original_predictions, 'iloc') else original_predictions[neighbor_indices_i]
                    
                    # Get most common type
                    try:
                        most_common = stats.mode(neighbor_types, keepdims=False)
                        majority_type = most_common.mode
                    except TypeError:
                        most_common = stats.mode(neighbor_types)
                        majority_type = most_common[0][0]
                    
                    # Store results
                    smoothed_predictions[i] = majority_type
                    if return_confidence:
                        confidence_scores[i] = np.mean(neighbor_types == majority_type)
                    
                    # Print progress periodically
                    if verbosity > 1 and i > 0 and i % 10000 == 0:
                        print(f"  Processed {i}/{len(adata)} cells...")
                
            except ImportError:
                print("FAISS not available, falling back to scikit-learn NearestNeighbors")
                if verbosity > 1:
                    print("To use FAISS (much faster for large datasets), install it with: pip install faiss-cpu")
                use_faiss = False
        
        if not use_faiss:
            # Use scikit-learn NearestNeighbors with parallel processing
            from sklearn.neighbors import NearestNeighbors
            
            nn = NearestNeighbors(n_neighbors=k_nearest_neighbors, n_jobs=n_jobs)
            nn.fit(X_embed)
            _, indices = nn.kneighbors(X_embed)
            
            if verbosity > 1:
                print(f"Processing {len(adata)} cells to determine majority labels...")
            
            # Process each cell
            for i in range(len(adata)):
                neighbor_indices_i = indices[i]
                neighbor_types = original_predictions.iloc[neighbor_indices_i] if hasattr(original_predictions, 'iloc') else original_predictions[neighbor_indices_i]
                
                # Get most common type
                try:
                    most_common = stats.mode(neighbor_types, keepdims=False)
                    majority_type = most_common.mode
                except TypeError:
                    most_common = stats.mode(neighbor_types)
                    majority_type = most_common[0][0]
                
                # Store results
                smoothed_predictions[i] = majority_type
                if return_confidence:
                    confidence_scores[i] = np.mean(neighbor_types == majority_type)
                
                # Print progress periodically
                if verbosity > 1 and i > 0 and i % 10000 == 0:
                    print(f"  Processed {i}/{len(adata)} cells...")
    
    # Add smoothed predictions to the result
    smoothed_key = key_added if key_added is not None else f"{groupby}_smoothed"
    
    # Convert array to pandas categorical 
    if hasattr(original_predictions, 'cat'):
        adata.obs[smoothed_key] = pd.Categorical(
            smoothed_predictions,
            categories=original_predictions.cat.categories
        )
    else:
        adata.obs[smoothed_key] = smoothed_predictions
    
    # Add confidence scores if requested
    if return_confidence:
        confidence_key = f"{smoothed_key}_confidence" if key_added is None else f"{key_added}_confidence"
        adata.obs[confidence_key] = confidence_scores
    
    # Report results
    if verbosity > 0:
        print(f"Smoothed predictions stored in adata.obs['{smoothed_key}']")
        if return_confidence:
            print(f"Confidence scores stored in adata.obs['{confidence_key}']")
        
        # Calculate how many labels changed
        changed = (adata.obs[smoothed_key] != original_predictions).sum()
        change_percent = (changed / len(adata)) * 100
        print(f"Modified {changed} cell labels ({change_percent:.2f}% of total)")
        
        # Report cells with insufficient neighbors
        if cells_with_insufficient_neighbors > 0:
            insufficient_percent = (cells_with_insufficient_neighbors / len(adata)) * 100
            print(f"Warning: {cells_with_insufficient_neighbors} cells ({insufficient_percent:.2f}%) had fewer than {k_nearest_neighbors} neighbors in the connectivity matrix")
        
        if verbosity > 1:
            # Show distribution of confidence scores
            if return_confidence:
                confidence_mean = np.mean(confidence_scores)
                confidence_median = np.median(confidence_scores)
                print(f"Confidence scores: mean={confidence_mean:.3f}, median={confidence_median:.3f}")
            
            # Show before/after distribution of labels
            orig_counts = original_predictions.value_counts()
            new_counts = adata.obs[smoothed_key].value_counts()
            print("\nLabel distribution before and after smoothing:")
            for cat in sorted(set(list(orig_counts.index) + list(new_counts.index))):
                orig_count = orig_counts.get(cat, 0)
                new_count = new_counts.get(cat, 0)
                diff = new_count - orig_count
                diff_sign = "+" if diff > 0 else ""
                print(f"  {cat}: {orig_count} → {new_count} ({diff_sign}{diff})")
    
    if not inplace:
        return adata
    
    
    
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Literal, List, Dict, Union, Optional, Tuple
from scipy.sparse import issparse
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

def predictCellTypeByMarker(
    adata,
    marker_gene_set: Union[List, Dict, pd.DataFrame],
    score_method: Literal["scanpy", "piaso"] = "piaso",
    score_layer: Optional[str] = "infog",
    use_score: bool = True,
    max_workers: Optional[int] = None,
    smooth_prediction: bool = True,
    use_rep: str = 'X_gdr',
    k_nearest_neighbors: int = 7,
    return_confidence: bool = True,
    use_existing_adjacency_graph: bool = False,
    use_faiss: bool = False,
    key_added: str = "CellTypes_predicted",
    extract_cell_type: bool = False,
    delimiter_cell_type: str = '-',
    inplace: bool = True,
    random_seed: int = 1927,
    verbosity: int = 1,
    n_jobs: int = -1
):
    """
    Predict cell types using marker genes and optionally smooth predictions.
    
    This function performs cell type prediction using marker genes in two steps:
    1. Calculate gene set scores for marker genes
    2. Optionally smooth the predictions using k-nearest neighbors
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell data
    marker_gene_set : list, dict, or pandas.DataFrame
        Collection of marker genes for different cell types (pre-filtered/prepared)
    score_method : {'scanpy', 'piaso'}, default='piaso'
        Method to use for scoring marker genes
    score_layer : str or None, default='infog'
        Layer of the AnnData object to use for scoring
    use_score : bool, default=True
        Whether to use scores (True) or p-values (False) for cell type prediction
    max_workers : int or None, default=None
        Number of parallel workers for score calculation
    smooth_prediction : bool, default=True
        Whether to smooth predictions using k-nearest neighbors
    use_rep : str, default='X_gdr'
        Key in adata.obsm containing the low-dimensional embedding to use for neighbor search
    k_nearest_neighbors : int, default=7
        Number of neighbors to consider for smoothing
    return_confidence : bool, default=True
        Whether to return confidence scores for smoothed predictions
    use_existing_adjacency_graph : bool, default=False
        Whether to use existing neighborhood graph if available
    use_faiss : bool, default=False
        Whether to use FAISS for faster neighbor search
    key_added : str, default='CellTypes_predicted'
        Key to use for storing cell type predictions in adata.obs
    extract_cell_type : bool, default=False
        Whether to extract cell type name by removing suffix after delimiter
    delimiter_cell_type : str, default='-'
        Delimiter to use when extracting cell type names (only used if extract_cell_type=True)
    inplace : bool, default=True
        Whether to modify adata in place or return a copy
    random_seed : int, default=1927
        Random seed for reproducibility
    verbosity : int, default=1
        Level of verbosity (0=quiet, 1=basic info, 2=detailed)
    n_jobs : int, default=-1
        Number of jobs for parallel processing during smoothing
        
    Returns
    -------
    If inplace=False:
        AnnData: Copy of adata with cell type predictions added
    If inplace=True:
        None, but adata is modified in place
        
    Examples
    --------
    >>> import scanpy as sc
    >>> import piaso
    >>> 
    >>> # Basic usage
    >>> piaso.tl.predictCellTypeByMarker(
    ...     adata,
    ...     marker_gene_set=cosgMarkerDB,
    ...     score_method='piaso',
    ...     use_score=False,
    ...     smooth_prediction=True,
    ...     inplace=True
    ... )
    """
    # Make a copy if not modifying in place
    if not inplace:
        adata = adata.copy()
    
    if verbosity > 0:
        print(f"Calculating gene set scores using {score_method} method...")
        
    
    # Calculate gene set scores
    if score_method == 'piaso':
        marker_results = calculateScoreParallel(
            adata,
            gene_set=marker_gene_set,
            score_method=score_method,
            score_layer=score_layer,
            max_workers=max_workers,
            return_pvals=True,
            random_seed=random_seed,
            verbosity=verbosity
        )
        
        # Extract results
        marker_gene_score = marker_results[0]
        gene_set_names = marker_results[1]
        marker_gene_nlog10pvals = marker_results[2]
    else:
        # For scanpy method
        marker_results = calculateScoreParallel(
            adata,
            gene_set=marker_gene_set,
            score_method=score_method,
            score_layer=score_layer,
            max_workers=max_workers,
            random_seed=random_seed,
            verbosity=verbosity
        )
        
        # Extract results
        marker_gene_score = marker_results[0]
        gene_set_names = marker_results[1]
        
    # Predict cell types based on scores or p-values
    if use_score:
        if verbosity > 0:
            print("Predicting cell types based on marker gene scores...")
            
        # Store scores in AnnData
        score_key = f"{key_added}_score"
        adata.obsm[score_key] = marker_gene_score
        
        # Convert to DataFrame and assign column names
        score_df = pd.DataFrame(marker_gene_score)
        score_df.columns = gene_set_names
        
        # Predict cell types based on maximum score
        adata.obs[key_added] = score_df.idxmax(axis=1).values
        
        # Extract cell type name if requested
        if extract_cell_type:
            if verbosity > 1:
                print(f"Extracting cell type names using delimiter: '{delimiter_cell_type}'")
            adata.obs[key_added] = [np.str_.rsplit(i, delimiter_cell_type, 1)[0] for i in adata.obs[key_added].values]
        
        # Store maximum score value
        adata.obs[f"{key_added}_score"] = score_df.max(axis=1).values
        
    else:
        if verbosity > 0:
            print("Predicting cell types based on marker gene p-values...")
            
        # Store scores in AnnData (even when using p-values for prediction)
        score_key = f"{key_added}_score"
        adata.obsm[score_key] = marker_gene_score
        
        # Store p-values in AnnData
        pval_key = f"{key_added}_nlog10pvals"
        adata.obsm[pval_key] = marker_gene_nlog10pvals
        
        # Convert to DataFrame and assign column names
        pval_df = pd.DataFrame(marker_gene_nlog10pvals)
        pval_df.columns = gene_set_names
        
        # Predict cell types based on maximum -log10(p-value)
        adata.obs[key_added] = pval_df.idxmax(axis=1).values
        
        # Extract cell type name if requested
        if extract_cell_type:
            if verbosity > 1:
                print(f"Extracting cell type names using delimiter: '{delimiter_cell_type}'")
            adata.obs[key_added] = [np.str_.rsplit(i, delimiter_cell_type, 1)[0] for i in adata.obs[key_added].values]
        
        # Store maximum -log10(p-value)
        adata.obs[f"{key_added}_nlog10pvals"] = pval_df.max(axis=1).values
    
    # Smooth predictions if requested
    if smooth_prediction:
        if verbosity > 0:
            print("Smoothing cell type predictions...")
        
        # Store original predictions for smoothing
        smoothing_input_key = f"{key_added}_raw"
        adata.obs[smoothing_input_key] = adata.obs[key_added].copy()
        
        # Set key for smoothed predictions
        smoothed_key = f"{key_added}_smoothed"
        
        # Smooth predictions
        smoothCellTypePrediction(
            adata,
            groupby=smoothing_input_key,
            key_added=smoothed_key,
            use_rep=use_rep,
            k_nearest_neighbors=k_nearest_neighbors,
            return_confidence=return_confidence,
            inplace=True,
            use_existing_adjacency_graph=use_existing_adjacency_graph,
            use_faiss=use_faiss,
            verbosity=verbosity,
            n_jobs=n_jobs
        )
        
        # Update main prediction key with smoothed predictions
        if verbosity > 1:
            print(f"Updating {key_added} with smoothed predictions...")
        
        # If confidence scores are calculated during smoothing
        if return_confidence:
            smoothed_confidence_key = f"{smoothed_key}_confidence"
            adata.obs[f"{key_added}_confidence_smoothed"] = adata.obs[smoothed_confidence_key].copy()
        
        # Replace original predictions with smoothed ones
        adata.obs[key_added] = adata.obs[smoothed_key].copy()
    
        
    if verbosity > 0:
        print(f"Cell type prediction completed. Results saved to:")
        print(f"  - adata.obs['{key_added}']: predicted cell types")
        
        if use_score:
            print(f"  - adata.obsm['{key_added}_score']: full score matrix")
            print(f"  - adata.obs['{key_added}_score']: maximum scores")
        else:
            print(f"  - adata.obsm['{key_added}_score']: full score matrix")
            print(f"  - adata.obsm['{key_added}_nlog10pvals']: full -log10(p-value) matrix")
            print(f"  - adata.obs['{key_added}_nlog10pvals']: maximum -log10(p-values)")
            
        if smooth_prediction:
            print(f"  - adata.obs['{key_added}_raw']: original unsmoothed predictions")
            if return_confidence:
                print(f"  - adata.obs['{key_added}_confidence_smoothed']: smoothing confidence scores")

    
    if not inplace:
        return adata
    return None

