from ._normalization import infog

### run SVD
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from anndata import AnnData

def runSVD(
    adata: AnnData, 
    use_highly_variable: bool = True, 
    n_components: int = 50, 
    random_state: Optional[int] = 10, 
    scale_data: bool = False,
    key_added: str = 'X_svd',
    layer: Optional[str] = None,
    verbosity: int = 0
):
    """
    Performs Truncated Singular Value Decomposition (SVD) on the specified gene expression matrix (adata.X or a specified layer)
    within an AnnData object and stores the resulting low-dimensional representation in `adata.obsm`.

    Parameters
    ----------
    adata : AnnData
        An AnnData object.
    use_highly_variable : bool, optional, default=True
        If True, the decomposition is performed only on highly variable genes/features.
    n_components : int, optional, default=50
        The number of principal components to retain.
    random_state : int, optional, default=10
        A random seed to ensure reproducibility.
    scale_data : bool, optional, default=False
        If True, standardizes the input data before performing SVD.
    key_added : str, optional, default='X_svd'
        The key under which the resulting cell embeddings are stored in `adata.obsm`.
    layer : str, optional, default=None
        Specifies which layer of `adata` to use for the transformation. If None, `adata.X` is used.
    verbosity : int, optional, default=0
        Controls the verbosity of logging messages.

    Returns
    -------
    None
        The function modifies `adata` in place, storing the cell embeddings in `adata.obsm[key_added]`.

    Example
    -------
    >>> import piaso
    >>> piaso.tl.runSVD(adata, use_highly_variable=True, n_components=50, random_state=42, 
    ...        scale_data=False, key_added='X_svd', layer=None)
    >>> 
    >>> # Access the transformed data
    >>> adata.obsm['X_svd']
    """
    
    if layer and layer not in adata.layers:
        raise ValueError(f"{layer} not found in adata.layers.")
        
    if use_highly_variable:
        if 'highly_variable' not in adata.var.columns:
            raise ValueError("adata.var should have 'highly_variable' column to use highly variable features.")
        if layer:
            expr = adata[:, adata.var['highly_variable']].layers[layer]
        else:
            expr = adata[:, adata.var['highly_variable']].X
    else:
        if layer:
            expr = adata.layers[layer]
        else:
            expr = adata.X
        
    
    if scale_data:
        expr = StandardScaler(with_mean=False).fit_transform(expr)
    
    transformer = TruncatedSVD(n_components=n_components, random_state=random_state)
    adata.obsm[key_added] = transformer.fit_transform(expr)
    
    if verbosity>0:
        print(f'The cell embeddings were saved as `{key_added}` in adata.obsm.')
    
##### Run SVD in a lazy mode
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Iterable, Union, Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

### Run SVD in lazy mode
def runSVDLazy(
    adata,
    copy: bool = False,
    n_components: int = 50,
    use_highly_variable: bool = True,
    n_top_genes: int = 3000,
    verbosity: int = 0,
    batch_key: Optional[str] = None,
    random_state: Optional[int] = 1927, 
    scale_data: bool = False,
    infog_trim: bool = True,
    key_added: str = 'X_svd',
    layer: Optional[str] = None,
    infog_layer: Optional[str] = None

):
    """
    Performs Truncated Singular Value Decomposition (SVD) in a "lazy" mode, based on the `piaso.tl.runSVD` function.
    
    Compared to `piaso.tl.runSVD`, this function includes the step of highly variable gene section. If `layer` is set to `infog`,
    both the highly variable genes and normalized gene expression values were taken from the INFOG normalization outputs.

    This function performs on the specified gene expression matrix (adata.X or a specified layer) within an AnnData object
    and stores the resulting low-dimensional representation in `adata.obsm`.

    Parameters
    ----------
    adata : AnnData
        An AnnData object.
    copy : bool, optional, default=False
        If True, returns a copy of `adata` with the computed embeddings instead of modifying in place.
    n_components : int, optional, default=50
        The number of singular value decomposition (SVD) components to retain.
    use_highly_variable : bool, optional, default=True
        If True, uses only highly variable genes for the decomposition.
    n_top_genes : int, optional, default=3000
        The number of top highly variable genes to retain before performing SVD.
    verbosity : int, optional, default=0
        Controls the verbosity of logging messages.
    batch_key : str, optional, default=None
        Specifies the key in `adata.obs` containing batch labels for `batch_key` used in the `sc.pp.highly_variable_genes` function.
    random_state : int, optional, default=1927
        A random seed to ensure reproducibility.
    scale_data : bool, optional, default=False
        If True, standardizes the input data before performing SVD.
    infog_trim : bool, optional, default=True
        Used for the `trim` parameter in `piaso.tl.infog` function, effective only when `layer` set to `infog`.
    key_added : str, optional, default='X_svd'
        The key under which the resulting cell embeddings are stored in `adata.obsm`.
    layer : str, optional, default=None
        Specifies which layer of `adata` to use for the transformation. If None, `adata.X` is used.
    infog_layer : str, optional, default=None
        Used for the `layer` parameter in `piaso.tl.infog` function, effective only when `layer` set to `infog`.

    Returns
    -------
    If `copy` is True, returns a modified AnnData object. Otherwise, modifies `adata` in place.

    Example
    -------
    >>> import piaso
    >>> adata = piaso.tl.runSVDLazy(
    ...     adata, n_components=50, n_top_genes=3000,
    ...     use_highly_variable=True, key_added="X_svd", 
    ...     layer=None
    ... )
    >>> 
    >>> # Access the cell embedding
    >>> adata.obsm['X_svd']
    """
    
    adata = adata.copy() if copy else adata

    # Back up the current verbosity level
    original_verbosity = sc.settings.verbosity
    
    sc.settings.verbosity=verbosity
    
    if layer=='infog':
        ### Run INFOG normalization
        infog(
            adata,
            copy=False,
            layer=infog_layer,
            n_top_genes=n_top_genes,
            key_added='infog',
            trim=infog_trim,
            verbosity=verbosity
        )
        ### No need to reset now, as now by default, the key is `highly_variable`
        # ### Set the highly variable genes selected by INFOG
        # adata.var['highly_variable']=adata.var['highly_variable_infog']
        
    else:
        sc.pp.highly_variable_genes(adata,
                                n_top_genes=n_top_genes,
                                batch_key=batch_key
                               )
        
        
    ### Use the runSVD function
    runSVD(
        adata,
        use_highly_variable=use_highly_variable, 
        n_components=n_components, 
        random_state=random_state, 
        scale_data=scale_data,
        key_added=key_added,
        layer=layer,
        verbosity=verbosity
    )
    
    sc.settings.verbosity=original_verbosity

    ### Return the result
    return adata if copy else None
