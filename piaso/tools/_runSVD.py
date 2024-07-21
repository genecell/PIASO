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
    layer: Optional[str] = None
) -> None:
    """
    Run Truncated Singular Value Decomposition (SVD) on the AnnData object and stores the result in adata.obsm[key_added].
    
    Parameters:
    adata (AnnData): The annotated data matrix.
    use_highly_variable (bool): Whether to use highly variable genes/features only. Default is True.
    n_components (int): Desired dimensionality of output data. Must be strictly less than the number of features. Default is 50.
    random_state (int, optional): Random seed for reproducibility. Default is 10.
    scale_data (bool): Whether to scale the data using StandardScaler. Default is False.
    key_added (str): Key in adata.obsm to store the result. Default is 'X_svd'.
    layer (str, optional): Specify the layer to use. If None, use adata.X. Default is None.
    
    Usage:
    ```python
    runSVD(adata, use_highly_variable=True, n_components=50, random_state=10, scale_data=False, key_added='X_svd', layer='raw')
    ```
    This will run SVD on the specified layer 'raw' of the adata object, and will store the result in adata.obsm['X_svd'].
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


    
##### Run SVD in a lazy mode
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Iterable, Union, Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

### Run pca in lazy mode
def runSVDLazy(
    adata,
    copy:bool=False,
    n_components:int=50,
    n_top_genes: int=3000,
    use_highly_variable: bool=True,
    verbosity: int=0,
    batch_key: str=None,
    trim_value:float=None,
    key_added: str=None
):
    
    adata = adata.copy() if copy else adata
    sc.settings.verbosity=verbosity
    sc.pp.highly_variable_genes(adata,
                            n_top_genes=n_top_genes,
                            batch_key=batch_key
                           )
    expr = adata[:, adata.var['highly_variable']].X if use_highly_variable else adata.X
    expr=StandardScaler(with_mean=False).fit_transform(expr)
    if trim_value is not None:
        expr[expr > trim_value] = trim_value
    transformer = TruncatedSVD(n_components=n_components, random_state=10)

    if key_added is None:
        adata.obsm['X_pca']= transformer.fit_transform(expr)
    else:
        adata.obsm[key_added]= transformer.fit_transform(expr)
    
    sc.settings.verbosity=3

    ### Return the result
    return adata if copy else None
