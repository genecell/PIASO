from ._runSVD import infog_svd, runSVD
from ._runGDR import runGDR
from ._neighbors import neighbors as _piaso_neighbors
from ._leiden import leiden as _piaso_leiden
from .external import runHarmony as _piaso_harmony


###### Some basics
from anndata import AnnData
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



from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome
from ..utils._cytome_compat import open_cytome_sync as _open_cytome


### Run leiden at another scale, i.e., locally
from typing import Sequence


def _leiden_local_cytome_streaming_svd(
    ds,
    groupby: str,
    groups,
    resolution: float,
    key_added: str,
    modality: str = "ATAC",
    measurement: str = "counts",
    max_nnz_percentile: float = 20.0,
    n_components: int = 30,
    n_iter: int = 7,
    n_neighbors: int = 15,
    batch_size: int = 1024,
    random_state: int = 10,
    min_cells: int = 20,
    verbose: bool = True,
):
    """Streaming, RAM-bounded `leiden_local` for an ATAC/tiles cytome.

    For each coarse group, **subset the cytome** to that group's cells (the
    O1 streaming subset — peak RAM ≈ one chunk, never the whole matrix), then
    run the cytome-native chain **in that group's own low-dim space**:
    ``selectPeaks(max_cumulative_nnz_percentile) → TF-IDF → randomized SVD →
    neighbors → leiden`` (all streaming). Sub-labels are prefixed ``<group>-<k>``
    and stitched into ``ds.cells[key_added]``. No full-AnnData materialisation,
    so this scales to large ATAC cytomes (where `to_anndata` would OOM).
    """
    import os
    import tempfile
    import warnings
    from ..preprocessing._selectPeaks import selectPeaks
    from ._runTFIDF import compute_tfidf_stats

    coarse = np.asarray(ds.cells[groupby]).astype("U")
    all_bc = np.asarray(ds.cells["barcode"], dtype=object)
    if groups is None:
        groups = np.unique(coarse)
    # object dtype — a fixed-width "<U1" (from a single-char coarse label) would
    # silently TRUNCATE the prefixed sub-labels ("A-3" → "A").
    result = coarse.astype(object)

    tmpdir = tempfile.mkdtemp(prefix="leiden_local_")
    try:
        for gi, g in enumerate(groups):
            gmask = (coarse == str(g))
            n_g = int(gmask.sum())
            if n_g == 0:
                continue
            # Too few cells to re-embed → keep the whole coarse group as one
            # sub-cluster (can't run SVD/Leiden meaningfully).
            if n_g < max(min_cells, 3):
                result[gmask] = [f"{g}-0"] * n_g
                if verbose:
                    print(f"  [leiden_local] group {g}: {n_g} cells (<{min_cells}) → single sub-cluster")
                continue

            sub_path = os.path.join(tmpdir, f"g{gi}.cytome")
            sub = ds.subset(gmask, output=sub_path,
                            include_fragments=False, include_embeddings=False)
            try:
                # 1) per-group peak selection (20% cumulative-nnz, like the pipeline)
                selectPeaks(sub, streaming=True,
                            max_cumulative_nnz_percentile=max_nnz_percentile,
                            measurement=measurement, modality=modality,
                            key_added="highly_variable", verbose=False)
                feat_tbl = "peaks" if modality == "ATAC" else modality
                selected = np.asarray(_read_selected_mask(sub, feat_tbl)).astype(bool)
                n_sel = int(selected.sum())
                # adaptive n_components: < n_cells and < n_selected
                nc = int(min(n_components, n_g - 1, max(n_sel - 1, 1)))
                if nc < 2 or n_sel < 2:
                    result[gmask] = [f"{g}-0"] * n_g
                    if verbose:
                        print(f"  [leiden_local] group {g}: n_sel={n_sel}, nc={nc} → single sub-cluster")
                    continue

                # 2) TF-IDF stats + 3) streaming SVD on selected peaks
                tfidf_stats = compute_tfidf_stats(sub, measurement=measurement, modality=modality)
                tfidf_stats["col_mask"] = selected
                runSVD(sub, modality=modality, measurement=measurement, streaming=True,
                       tfidf_params=tfidf_stats, key_added="X_svd",
                       n_components=nc, n_iter=n_iter, random_state=random_state,
                       batch_size=batch_size, verbosity=0)

                # 4) neighbors + 5) leiden in the group's own space
                _piaso_neighbors(sub, use_rep="X_svd",
                                 n_neighbors=min(n_neighbors, n_g - 1),
                                 random_state=random_state, key_added="neighbors")
                # Private, case-collision-proof key: the subset inherits the
                # parent's cell columns, and cortex RNA annotations carry a
                # 'Leiden_local' column. SQLite columns are CASE-INSENSITIVE, so
                # writing 'leiden_local' would silently land in the existing
                # 'Leiden_local' and the exact-case read-back would KeyError
                # (→ every group collapsed to a single sub-cluster on mc/hc_adol).
                _sub_key = "_piaso_ll_subleiden"
                _piaso_leiden(sub, resolution=resolution, key_added=_sub_key,
                              random_state=random_state, n_iterations=10,
                              adjacency_key="connectivities")

                sub_labels = np.asarray(sub.cells[_sub_key]).astype("U")
                sub_bc = np.asarray(sub.cells["barcode"], dtype=object)
                lab_map = {b: f"{g}-{lab}" for b, lab in zip(sub_bc, sub_labels)}
                result[gmask] = [lab_map[b] for b in all_bc[gmask]]
                if verbose:
                    print(f"  [leiden_local] group {g}: {n_g} cells, {n_sel} peaks → "
                          f"{len(set(sub_labels))} sub-clusters")
            except Exception as exc:  # robust: one bad group shouldn't kill the run
                warnings.warn(
                    f"leiden_local: group {g} failed ({type(exc).__name__}: {exc}); "
                    f"keeping it as a single sub-cluster.", RuntimeWarning, stacklevel=2)
                result[gmask] = [f"{g}-0"] * n_g
            finally:
                try:
                    sub.close()
                except Exception:
                    pass
                for suf in ("", "-wal", "-shm"):
                    fp = sub_path + suf
                    if os.path.exists(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    ds.cells[key_added] = result
    ds.flush()
    return None


def _read_selected_mask(ds, feat_tbl):
    """Read the boolean 'highly_variable' column from a feature entity table."""
    ent = getattr(ds, feat_tbl)
    return np.asarray(ent["highly_variable"])


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
    modality: Optional[str] = None,
    cytome_layer: str = "counts",
    max_nnz_percentile: float = 20.0,
    n_components: int = 30,
    n_iter: int = 7,
    n_neighbors: int = 15,
    batch_size: int = 1024,
    random_state: int = 10,
):
    """
    Perform Leiden clustering locally, i.e., on selected group(s), on an **AnnData object
    or a cytome.Dataset** (or a cytome path string).
    This function enables flexible clustering within specified groups, supports batch effect handling, and stores results back in the object.

    Both inputs are supported:

    - **AnnData**: in-memory clustering; labels are written to ``adata.obs[key_added]``.
    - **cytome.Dataset / path str**: labels are written to ``ds.cells[key_added]``. With
      ``dr_method='X_svd'`` a **RAM-bounded streaming** path is used — each coarse group is
      subset to its own cytome and clustered in its own low-dim space (per-group
      selectPeaks → TF-IDF → randomized SVD → neighbors → Leiden), so it scales to large
      ATAC/tiles cytomes where materialising a full AnnData would OOM. This is the path the
      snakemake peak-calling workflow uses (``picco_preliminary_method: leiden_local``).

    Parameters
    ----------
    adata : AnnData, cytome.Dataset, or str
        AnnData object, an open ``cytome.Dataset``, or a path to a ``.cytome`` file.
    clustering_type : str, optional (default: 'each')
        Specifies the clustering approach:
        - 'each': Perform clustering independently within each group.
        - 'all': Perform clustering across all selected groups.
    groupby : str, optional (default: 'Leiden')
        The key in `adata.obs` specifying the cell labels to be used for selecting groups.
    groups : Sequence[str], optional (default: None)
        A list of specific group(s) to be clustered. If None, all groups in the `groupby` category will be used.
    resolution : float, optional (default: 0.25)
        Resolution parameter for the Leiden algorithm, controlling clustering granularity. 
        Higher values result in more clusters.
    batch_key : Sequence[str], optional (default: None)
        Key in `adata.obs` specifying batch labels. If provided, it handles batch effects during clustering. If None, batch effects are ignored.
    key_added : str, optional (default: 'Leiden_local')
        The name of the key under which the local Leiden clustering results will be stored in `adata.obs`.
    dr_method : str, optional (default: 'X_pca')
        Dimensionality reduction method to be used for local clustering.
        Allowed values are: 'X_pca', 'X_gdr', 'X_pca_harmony', 'X_svd_full', 'X_svd_full_harmony'.
    gdr_resolution : float, optional (default: 1.0)
        Resolution parameter for the GDR dimensionality reduction method if 'dr_method' is set to 'X_gdr'.
    copy : bool, optional (default: False)
        If False, the operation is performed in-place. If True, a copy of the `adata` object is returned with the clustering results added.
        
    Returns
    -------
    AnnData or None
        - If `copy=True`: Returns a new AnnData object with clustering results added to `adata.obs[key_added]`.
        - If `copy=False`: Modifies the input `adata` object in-place by adding clustering results to `adata.obs[key_added]`.

    Example
    -------
    >>> # Example usage
    >>> leiden_local(
    ...     adata,
    ...     clustering_type='each',
    ...     groupby='Leiden',
    ...     groups=['0', '1'],
    ...     resolution=0.2,
    ...     batch_key=None,
    ...     key_added='Leiden_local',
    ...     dr_method='X_pca',
    ...     copy=False
    ... )
    """
    
    # --- Streaming cytome path (dr_method='X_svd', ATAC/tiles) ---
    # selectPeaks(max_nnz_percentile) → TF-IDF → rSVD → neighbors → leiden,
    # per coarse group, on a streaming per-group subset cytome (no full-AnnData
    # materialisation — scales to large ATAC where to_anndata would OOM). Only
    # 'each' makes sense here (per-group own low-dim space).
    if dr_method == 'X_svd' and (isinstance(adata, str) or _is_cytome(adata)):
        if clustering_type != 'each':
            raise ValueError("dr_method='X_svd' (streaming cytome) supports "
                             "clustering_type='each' only.")
        _ds = _open_cytome(adata) if isinstance(adata, str) else adata
        return _leiden_local_cytome_streaming_svd(
            _ds, groupby=groupby, groups=groups, resolution=resolution,
            key_added=key_added,
            modality=modality if modality is not None else "ATAC",
            measurement=cytome_layer,
            max_nnz_percentile=max_nnz_percentile, n_components=n_components,
            n_iter=n_iter, n_neighbors=n_neighbors, batch_size=batch_size,
            random_state=random_state,
        )

    # --- Cytome dispatch (other dr_methods): convert to AnnData, run, write back ---
    _cytome_ds = None
    if isinstance(adata, str) or _is_cytome(adata):
        _cytome_ds = _open_cytome(adata) if isinstance(adata, str) else adata
        _modality_for_dispatch = modality if modality is not None else "RNA"
        adata = _cytome_ds.to_anndata(
            modality=_modality_for_dispatch, layer=cytome_layer,
        )

    # Validate clustering_type
    if clustering_type not in {'each', 'all'}:
        raise ValueError(
            f"Invalid value for clustering_type: '{clustering_type}'. "
            "Allowed values are 'each' or 'all'."
        )

    # Validate dimensionality reduction approaches
    if dr_method not in {'X_pca', 'X_gdr', 'X_pca_harmony', 'X_svd_full', 'X_svd_full_harmony'}:
        raise ValueError(
            f"Invalid value for dr_method: '{dr_method}'. "
            "Allowed values are: 'X_pca', 'X_gdr', 'X_pca_harmony', 'X_svd_full', "
            "'X_svd_full_harmony', or 'X_svd' (streaming cytome ATAC/tiles only)."
        )
    
    
    adata = adata.copy() if copy else adata
    
    existing_groups = adata.obs[groupby].astype('U').copy()
    
    if groups is None:
        groups=np.unique(adata.obs[groupby])
    
    if clustering_type=='each':
        
        for group in groups:
            
            group_idx=adata.obs[groupby].isin([group])
            adata_i=adata[group_idx].copy()
            ## Run DR
            adata_i = adata_i[:, np.asarray(adata_i.X.getnnz(axis=0)).ravel() >= 1].copy()
            sqrt_ncells=int(np.sqrt(adata_i.shape[0]))
            n_components = 30 if sqrt_ncells > 30 else sqrt_ncells
            if dr_method=='X_pca':
                infog_svd(adata_i, n_components = n_components)
                use_rep='X_pca'
                
            elif dr_method=='X_svd_full':
                runSVD(
                    adata_i, 
                    use_highly_variable=False, 
                    n_components=n_components, 
                    random_state = 10, 
                    scale_data = False,
                    key_added = 'X_svd_full'
                )
                use_rep='X_svd_full'
                
            elif dr_method=='X_gdr':
                infog_svd(adata_i, n_components = n_components)
                _knn_result_i = _piaso_neighbors(adata_i, use_rep='X_pca', n_neighbors=15, random_state=10)
                _piaso_leiden(adata_i, resolution=gdr_resolution, key_added='leiden_tmp', knn_result=_knn_result_i, random_state=10)
                
                runGDR(adata_i,
                            batch_key=None,
                            groupby='leiden_tmp'
                )
                use_rep='X_gdr'
                
            elif dr_method=='X_pca_harmony':
                infog_svd(adata_i, n_components = n_components)
                _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_pca', key_added='X_pca_harmony')
                use_rep='X_pca_harmony'
                
            elif dr_method=='X_svd_full_harmony':
                runSVD(
                    adata_i, 
                    use_highly_variable=False, 
                    n_components=n_components, 
                    random_state = 10, 
                    scale_data = False,
                    key_added = 'X_svd_full'
                )
                _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_svd_full', key_added='X_svd_full_harmony')
                use_rep='X_svd_full_harmony'
            
                
                
            _knn_result_i = _piaso_neighbors(adata_i, use_rep=use_rep, n_neighbors=15, random_state=10)
            
            _piaso_leiden(adata_i, resolution=resolution, key_added='leiden_local', knn_result=_knn_result_i, random_state=10)
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
            adata_i = adata_i[:, np.asarray(adata_i.X.getnnz(axis=0)).ravel() >= 1].copy()
            sqrt_ncells=int(np.sqrt(adata_i.shape[0]))
            n_components = 30 if sqrt_ncells > 30 else sqrt_ncells
            
            if dr_method=='X_pca':
                infog_svd(adata_i, n_components = n_components)
                use_rep='X_pca'
            
            elif dr_method=='X_svd_full':
                runSVD(
                    adata_i, 
                    use_highly_variable=False, 
                    n_components=n_components, 
                    random_state = 10, 
                    scale_data = False,
                    key_added = 'X_svd_full'
                )
                use_rep='X_svd_full'
                
            elif dr_method=='X_gdr':
                infog_svd(adata_i, n_components = n_components)
                _knn_result_i = _piaso_neighbors(adata_i, use_rep='X_pca', n_neighbors=15, random_state=10)
                _piaso_leiden(adata_i, resolution=gdr_resolution, key_added='leiden_tmp', knn_result=_knn_result_i, random_state=10)
                
                runGDR(adata_i,
                            batch_key=None,
                            groupby='leiden_tmp'
                )
                use_rep='X_gdr'
                
            elif dr_method=='X_pca_harmony':
                infog_svd(adata_i, n_components = n_components)
                _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_pca', key_added='X_pca_harmony')
                use_rep='X_pca_harmony'
                
            elif dr_method=='X_svd_full_harmony':
                runSVD(
                    adata_i, 
                    use_highly_variable=False, 
                    n_components=n_components, 
                    random_state = 10, 
                    scale_data = False,
                    key_added = 'X_svd_full'
                )
                _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_svd_full', key_added='X_svd_full_harmony')
                use_rep='X_svd_full_harmony'
            
            _knn_result_i = _piaso_neighbors(adata_i, use_rep=use_rep, n_neighbors=15, random_state=10)
                
            _piaso_leiden(adata_i, resolution=resolution, key_added='leiden_local', knn_result=_knn_result_i, random_state=10)
            ## Add prefix
            max_len = len(max(adata_i.obs['leiden_local'], key=len)) 
            ### Make sure the numbers having the same length
            local_leiden=list(map(lambda x: 'M' + '-'+ x.zfill(max_len), adata_i.obs['leiden_local']))
            ## Rename the files
            existing_groups[group_idx]=local_leiden
        else:
            ### Only provide one element
            group=groups[0]
            group_idx=adata.obs[groupby].isin([group])
            adata_i=adata[group_idx].copy()
            ## Run DR
            adata_i = adata_i[:, np.asarray(adata_i.X.getnnz(axis=0)).ravel() >= 1].copy()
            sqrt_ncells=int(np.sqrt(adata_i.shape[0]))
            n_components = 30 if sqrt_ncells > 30 else sqrt_ncells
            
            if dr_method=='X_pca':
                infog_svd(adata_i, n_components = n_components)
                use_rep='X_pca'
                
            elif dr_method=='X_svd_full':
                runSVD(
                    adata_i, 
                    use_highly_variable=False, 
                    n_components=n_components, 
                    random_state = 10, 
                    scale_data = False,
                    key_added = 'X_svd_full'
                )
                use_rep='X_svd_full'
                
            elif dr_method=='X_gdr':
                infog_svd(adata_i, n_components = n_components)
                _knn_result_i = _piaso_neighbors(adata_i, use_rep='X_pca', n_neighbors=15, random_state=10)
                _piaso_leiden(adata_i, resolution=gdr_resolution, key_added='leiden_tmp', knn_result=_knn_result_i, random_state=10)
                
                runGDR(adata_i,
                            batch_key=None,
                            groupby='leiden_tmp'
                )
                use_rep='X_gdr'
                
            elif dr_method=='X_pca_harmony':
                infog_svd(adata_i, n_components = n_components)
                _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_pca', key_added='X_pca_harmony')
                use_rep='X_pca_harmony'
                
            elif dr_method=='X_svd_full_harmony':
                runSVD(
                    adata_i, 
                    use_highly_variable=False, 
                    n_components=n_components, 
                    random_state = 10, 
                    scale_data = False,
                    key_added = 'X_svd_full'
                )
                _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_svd_full', key_added='X_svd_full_harmony')
                use_rep='X_svd_full_harmony'
                
            _knn_result_i = _piaso_neighbors(adata_i, use_rep=use_rep, n_neighbors=15, random_state=10)
            
            _piaso_leiden(adata_i, resolution=resolution, key_added='leiden_local', knn_result=_knn_result_i, random_state=10)
            ## Add prefix
            max_len = len(max(adata_i.obs['leiden_local'], key=len)) 
            ### Make sure the numbers having the same length
            local_leiden=list(map(lambda x: group + '-'+ x.zfill(max_len), adata_i.obs['leiden_local']))
            ## Rename the files
            existing_groups[group_idx]=local_leiden
            
        
        
    adata.obs[key_added]=existing_groups
    adata.obs[key_added]=adata.obs[key_added].astype('category')

    # --- Write results back to cytome ---
    if _cytome_ds is not None:
        _cytome_ds.cells[key_added] = adata.obs[key_added].values.astype(str)
        _cytome_ds.flush()
        return None

    return adata if copy else None
    

