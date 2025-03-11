from ._runGDR import runGDR

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



def smoothCellTypePredictions(
    adata: "AnnData", 
    groupby: str, 
    use_rep: str = 'X_pca', 
    k_nearest_neighbors: int = 5, 
    return_confidence: bool = False,
    inplace: bool = True,
    use_existing_adjacency_graph: bool = True,
    use_faiss: bool = False,
    key_added: str = None,
    verbosity: int = 1
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
    >>> piaso.tl.smoothCellTypePredictions(
    ...     adata, 
    ...     groupby='CellTypes_pred',
    ...     use_rep='X_pca',
    ...     key_added='CellTypes_pred_smoothed'
    ... )
    >>> 
    >>> # With confidence scores
    >>> piaso.tl.smoothCellTypePredictions(
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
                print("Using scikit-learn NearestNeighbors")
    
    if has_neighbors:
        # Use existing connectivity matrix from adata
        conn_matrix = adata.obsp['connectivities']
        
        # For each cell, find top k neighbors from connectivity matrix
        neighbor_indices = []
        for i in range(conn_matrix.shape[0]):
            # Get row, convert to dense format if sparse
            row = conn_matrix[i].toarray().flatten() if hasattr(conn_matrix[i], 'toarray') else conn_matrix[i]
            # Get indices of top k neighbors (including self)
            top_k = np.argsort(row)[::-1][:k_nearest_neighbors]
            neighbor_indices.append(top_k)
        
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
                n_samples, n_features = X_embed.shape
                
                # Convert to float32 as required by FAISS
                X_embed_float32 = X_embed.astype('float32')
                
                # Build index
                index = faiss.IndexFlatL2(n_features)
                index.add(X_embed_float32)
                
                # Search for nearest neighbors
                distances, indices = index.search(X_embed_float32, k_nearest_neighbors)
                neighbor_indices = indices
                
            except ImportError:
                print("FAISS not available, falling back to scikit-learn NearestNeighbors")
                if verbosity > 1:
                    print("To use FAISS (much faster for large datasets), install it with: pip install faiss-cpu")
                use_faiss = False
        
        if not use_faiss:
            # Use scikit-learn NearestNeighbors
            from sklearn.neighbors import NearestNeighbors
            
            nn = NearestNeighbors(n_neighbors=k_nearest_neighbors)
            nn.fit(X_embed)
            _, indices = nn.kneighbors(X_embed)
            neighbor_indices = indices
    
    # For each cell, get the most common cell type among its neighbors
    smoothed_predictions = []
    confidence_scores = []
    
    if verbosity > 1:
        print(f"Processing {len(adata)} cells to determine majority labels...")
    
    for i in range(len(adata)):
        # Get neighbors for this cell
        if has_neighbors or use_faiss:
            neighbor_indices_i = neighbor_indices[i]
        else:
            neighbor_indices_i = indices[i]
            
        neighbor_types = original_predictions.iloc[neighbor_indices_i] if hasattr(original_predictions, 'iloc') else original_predictions[neighbor_indices_i]
        
        # Get most common type and its count
        try:
            # Try newer SciPy version (1.9.0+) with keepdims parameter
            most_common = stats.mode(neighbor_types, keepdims=False)
            majority_type = most_common.mode
        except TypeError:
            # Fall back to older SciPy version
            most_common = stats.mode(neighbor_types)
            majority_type = most_common[0][0]
        
        # Calculate confidence (proportion of neighbors with majority label)
        if return_confidence:
            confidence = np.mean(neighbor_types == majority_type)
            confidence_scores.append(confidence)
        
        smoothed_predictions.append(majority_type)
    
    # Add smoothed predictions to the result
    smoothed_key = key_added if key_added is not None else f"{groupby}_smoothed"
    adata.obs[smoothed_key] = pd.Categorical(
        smoothed_predictions,
        categories=original_predictions.cat.categories if hasattr(original_predictions, 'cat') else None
    )
    
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