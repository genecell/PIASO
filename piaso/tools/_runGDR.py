from ._runSVD import runSVDLazy

### Run GDR
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import cosg


def runGDR(adata,
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