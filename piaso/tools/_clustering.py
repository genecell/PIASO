from ._runSVD import runSVDLazy
from ._runGDR import runGDR



###### Some basics
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Iterable, Union, Optional

### Refer to: https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/_compat.py
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass
######



### Run leiden at another scale, i.e., locally
from typing import Sequence
def leiden_local(
    adata,
    clustering_type: str='each',
    groupby: str='Leiden',
    groups: Optional[Sequence[str]] = None,
    resolution: float = 0.25,
    batch_key: Optional[Sequence[str]] = None,
    key_added: str =  'Leiden_local',
    dr_method: str = 'X_pca',
    gdr_resolution: float=1.0,
    copy: bool = False,
):
    adata = adata.copy() if copy else adata
    
    existing_groups = adata.obs[groupby].astype('U').copy()
    
    if groups is None:
        groups=np.unique(adata.obs[groupby])
    
    if clustering_type=='each':
        
        for group in groups:
            
            group_idx=adata.obs[groupby].isin([group])
            adata_i=adata[group_idx].copy()
            ## Run DR
            sc.pp.filter_genes(adata_i, min_cells=1)
            sqrt_ncells=int(np.sqrt(adata_i.shape[0]))
            n_components = 30 if sqrt_ncells > 30 else sqrt_ncells
            if dr_method=='X_pca':
                runSVDLazy(adata_i, n_components = n_components)
                use_rep='X_pca'
                
            elif dr_method=='X_gdr':
                runSVDLazy(adata_i, n_components = n_components)
                sc.tl.leiden(adata_i, resolution=gdr_resolution, key_added='leiden_tmp')
                
                runGDR(adata_i,
                            batch_key=None,
                            groupby='leiden_tmp'
                )
                use_rep='X_gdr'
                
            elif dr_method=='X_pca_harmony':
                runSVDLazy(adata_i, n_components = n_components)
                sc.external.pp.harmony_integrate(adata_i,
                    key=batch_key,
                    basis = 'X_pca',
                    adjusted_basis = 'X_pca_harmony',
                                                )
                
                use_rep='X_pca_harmony'
            
                
                
            sc.pp.neighbors(adata_i,
                use_rep=use_rep,
                n_neighbors=15,random_state=10,knn=True,
                method="umap")
            
            sc.tl.leiden(adata_i, resolution=resolution, key_added='leiden_local')
            ## Add prefix
            max_len = len(max(adata_i.obs['leiden_local'], key=len)) 
            ### Make sure the numbers having the same length
            local_leiden=list(map(lambda x: group + '-'+ x.zfill(max_len), adata_i.obs['leiden_local']))
            ## Rename the files
            existing_groups[group_idx]=local_leiden
            
    elif clustering_type=='all':
        
        if len(groups)>1:
        
            # print('')
            group_idx=adata.obs[groupby].isin(groups)
            adata_i=adata[group_idx].copy()
            ## Run DR
            sc.pp.filter_genes(adata_i, min_cells=1)
            sqrt_ncells=int(np.sqrt(adata_i.shape[0]))
            n_components = 30 if sqrt_ncells > 30 else sqrt_ncells
            
            if dr_method=='X_pca':
                runSVDLazy(adata_i, n_components = n_components)
                use_rep='X_pca'
                
            elif dr_method=='X_gdr':
                runSVDLazy(adata_i, n_components = n_components)
                sc.tl.leiden(adata_i, resolution=gdr_resolution, key_added='leiden_tmp')
                
                runGDR(adata_i,
                            batch_key=None,
                            groupby='leiden_tmp'
                )
                use_rep='X_gdr'
                
            elif dr_method=='X_pca_harmony':
                runSVDLazy(adata_i, n_components = n_components)
                sc.external.pp.harmony_integrate(adata_i,
                    key=batch_key,
                    basis = 'X_pca',
                    adjusted_basis = 'X_pca_harmony',
                                                )
                use_rep='X_pca_harmony'
            
            sc.pp.neighbors(adata_i,
                use_rep=use_rep,
                n_neighbors=15,random_state=10,knn=True,
                method="umap")
                
            sc.tl.leiden(adata_i, resolution=resolution, key_added='leiden_local')
            ## Add prefix
            max_len = len(max(adata_i.obs['leiden_local'], key=len)) 
            ### Make sure the numbers having the same length
            local_leiden=list(map(lambda x: 'M' + '-'+ x.zfill(max_len), adata_i.obs['leiden_local']))
            ## Rename the files
            existing_groups[group_idx]=local_leiden
        else:
            ### Only provide one element
            group=groups
            group_idx=adata.obs[groupby].isin([group])
            adata_i=adata[group_idx].copy()
            ## Run DR
            sc.pp.filter_genes(adata_i, min_cells=1)
            sqrt_ncells=int(np.sqrt(adata_i.shape[0]))
            n_components = 30 if sqrt_ncells > 30 else sqrt_ncells
            
            if dr_method=='X_pca':
                runSVDLazy(adata_i, n_components = n_components)
                use_rep='X_pca'
                
            elif dr_method=='X_gdr':
                runSVDLazy(adata_i, n_components = n_components)
                sc.tl.leiden(adata_i, resolution=gdr_resolution, key_added='leiden_tmp')
                
                runGDR(adata_i,
                            batch_key=None,
                            groupby='leiden_tmp'
                )
                use_rep='X_gdr'
                
            elif dr_method=='X_pca_harmony':
                runSVDLazy(adata_i, n_components = n_components)
                sc.external.pp.harmony_integrate(adata_i,
                    key=batch_key,
                    basis = 'X_pca',
                    adjusted_basis = 'X_pca_harmony',
                                                )
                use_rep='X_pca_harmony'
                
            sc.pp.neighbors(adata_i,
                use_rep=use_rep,
                n_neighbors=15,random_state=10,knn=True,
                method="umap")
            
            sc.tl.leiden(adata_i, resolution=resolution, key_added='leiden_local')
            ## Add prefix
            max_len = len(max(adata_i.obs['leiden_local'], key=len)) 
            ### Make sure the numbers having the same length
            local_leiden=list(map(lambda x: group + '-'+ x.zfill(max_len), adata_i.obs['leiden_local']))
            ## Rename the files
            existing_groups[group_idx]=local_leiden
            
        
        
    adata.obs[key_added]=existing_groups
    adata.obs[key_added]=adata.obs[key_added].astype('category')
    
    return adata if copy else None
    

