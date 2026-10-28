# import faulthandler
# faulthandler.enable()

# import os
# # Force thread count to 1 to prevent C-library collisions
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'



import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import itertools 
import warnings 
import typing 


# Import COSG if available
try:
    import cosg
except ImportError:
    cosg = None # Set to None if not available
    

    

# Helper Function for Jaccard Index
def calculate_jaccard(set1: set, set2: set) -> float:
    """
    Calculates Jaccard index between two sets.

    Parameters
    ----------
    set1
        First set of items.
    set2
        Second set of items.

    Returns
    -------
    float
        Jaccard index (between 0.0 and 1.0). Returns 0.0 if union is 0.
    """
    # Ensure inputs are sets
    set1 = set(set1)
    set2 = set(set2)
    # Handle potential None values if COSG failed catastrophically (although initialization should prevent this)
    if set1 is None or set2 is None:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    # Jaccard index is 0 if union is 0 (both sets empty)
    return intersection / union if union > 0 else 0.0 

def _version_tuple(v: str):
    """``'1.16.3'`` -> ``(1, 16, 3)``; non-numeric parts end the tuple."""
    out = []
    for part in str(v).split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def _installed_version(dist: str):
    """The installed version of ``dist``, or None. (pkg_resources, used before,
    is deprecated and gone from new setuptools, where its ImportError read as
    "BBKNN is required".)"""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:                       # pragma: no cover
        return None
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _markers_from_names(names_df, labels, n_markers):
    """``{label: set of the first n_markers marker names}`` from a COSG names table."""
    out = {}
    for lab in labels:
        if lab in names_df.columns:
            out[lab] = set(names_df[lab].dropna().astype(str).tolist()[:n_markers])
        else:
            out[lab] = set()
    return out


def _cosg_store_markers(store, labels_key, mask, n_markers, cosg_args):
    """COSG markers of each group of ``labels_key`` among the cells of ``mask``, read
    from a store's own expression (``run_cosg_cytome``, streaming)."""
    from cosg import run_cosg_cytome
    ds, modality, layer = store["ds"], store["modality"], store["layer"]
    res = run_cosg_cytome(
        cytome_path=str(ds.path), groupby=labels_key, cell_mask=np.asarray(mask, dtype=bool),
        n_genes_user=n_markers, mu=cosg_args["mu"], expressed_pct=cosg_args["expressed_pct"],
        remove_lowly_expressed=cosg_args["remove_lowly_expressed"], modality=modality,
        layer=layer, verbose=False)
    groups = [str(g) for g in res["groups_order"]]
    return pd.DataFrame(np.asarray(res["names"], dtype=object), columns=groups)


# Helper Function for Building the Pruned Graph
def _build_marker_filtered_graph_dual(
    adata: AnnData,
    batch_key: str,
    use_rep: str,
    neighbors_within_batch: int,
    # --- Clustering Params ---
    leiden_resolution: float,
    leiden_n_neighbors: int,
    cluster_key_added: str, # Key to store final batch-cluster labels
    random_state: typing.Union[int, None], # Seed for Leiden
    # --- COSG/Marker Params ---
    n_markers: int,
    marker_overlap_threshold: float, # Threshold used for both local and global
    use_global_markers: bool, # Control whether to use global markers
    cosg_layer: typing.Union[str, None], # Layer for COSG input
    cosg_mu: float, # Changed to float
    cosg_expressed_pct: float,
    cosg_remove_lowly_expressed: bool,
    # --- BBKNN specific ---
    bbknn_trim: typing.Union[int, None],
    bbknn_computation: str = 'annoy', 
    bbknn_metric: str = 'euclidean',
    verbosity: int = 0, # Control internal verbosity of this helper
    cosg_store: typing.Union[dict, None] = None, # a store's own expression for COSG
    ) -> typing.Tuple[ # Return graph, compatibility dict, local markers, global markers
        typing.Union[sp.csr_matrix, None], 
        typing.Union[dict, None],
        typing.Union[dict, None],
        typing.Union[dict, None] 
    ]: 
    """
    Internal helper function for stitchSpace. Builds and prunes a KNN graph of cells.

    Steps:
    1. Build initial BBKNN graph
    2. Cluster within batches using Leiden & store labels 
       in `adata.obs[cluster_key_added]`
    3. Run COSG within batches to find local markers.
    4. Optionally run COSG globally on batch-cluster labels to find global markers.
    5. Compare markers (local & optionally global) to create a combined compatibility lookup 
       (dictionary mapping pairs of batch-cluster labels to boolean compatibility).
    6. Prune the initial graph efficiently using vectorized lookups based on the compatibility.
    7. Return the pruned connectivity matrix, compatibility lookup dictionary, 
       local marker dictionary, and global marker dictionary.

    Parameters are documented in the main `stitchSpace` function.
    """
 
    n_cells = adata.n_obs
    delimiter = "@" # Use '@' as the delimiter
    
    
    
    # BBKNN, and its version. Every failure here raises: it used to print a
    # message and return None, which is also what success returns.
    try:
        import bbknn
        import bbknn.matrix
    except ImportError as e:
        raise ImportError("piaso.tl.stitchSpace needs BBKNN: pip install bbknn") from e
    bbknn_version = _installed_version("bbknn") or getattr(bbknn, "__version__", "0")
    is_old_version = _version_tuple(bbknn_version) <= (1, 5, 1)
    if verbosity > 0:
        print(f"BBKNN version {bbknn_version} ({'<= 1.5.1' if is_old_version else '> 1.5.1'} parameters)")

    # Step 1: Build Initial BBKNN Graph using bbknn.matrix.bbknn
    if verbosity > 0: print(f"Building initial graph via BBKNN (neighbors_within_batch={neighbors_within_batch})...")
    cell_embeddings = adata.obsm[use_rep]
    batch_list = adata.obs[batch_key].values
    base_params = {
        'neighbors_within_batch': neighbors_within_batch,
        'trim': bbknn_trim,
        'metric': bbknn_metric,
        'n_pcs': cell_embeddings.shape[1],  # Use all available dimensions
    }
    computation = bbknn_computation or 'annoy'
    if computation == 'annoy':
        # Annoy >= 1.17 makes BBKNN segfault (a crash, not an error to catch).
        annoy_version = _installed_version("annoy")
        if annoy_version is not None and _version_tuple(annoy_version) >= (1, 17, 0):
            raise ImportError(
                f"stitchSpace: annoy {annoy_version} is installed, and annoy >= 1.17.0 makes BBKNN "
                f"crash (segmentation fault). Install annoy 1.16.3 (pip install annoy==1.16.3, then "
                f"restart the kernel), or pass filter_bbknn_computation='pynndescent' or 'faiss'.")
    if is_old_version:
        # BBKNN <= 1.5.1 names the computation with flags instead
        params_to_use = {**base_params, 'approx': computation in ('annoy', 'faiss'),
                         'annoy_n_trees': 10, 'use_annoy': computation == 'annoy',
                         'use_faiss': computation == 'faiss'}
    else:
        params_to_use = {**base_params, 'computation': computation, 'annoy_n_trees': 10}
    params_to_use = {k: v for k, v in params_to_use.items() if v is not None}
    if verbosity > 0:
        print(f"Calling bbknn.matrix.bbknn with parameters: {params_to_use}")
    try:
        distances, connectivities, parameters = bbknn.matrix.bbknn(
            cell_embeddings, batch_list, **params_to_use)
    except Exception as e:
        raise RuntimeError(f"stitchSpace: BBKNN failed to build the graph ({type(e).__name__}: {e})") from e
    if connectivities is None:
        raise RuntimeError("stitchSpace: BBKNN returned no connectivities")
    if verbosity > 0: print("Initial graph built using bbknn.matrix.bbknn successfully.")
        
    

    # Step 2: Clustering within Batches (with verbosity control)
    if verbosity > 0: print(f"Clustering within batches (Leiden res={leiden_resolution}, k={leiden_n_neighbors})...")

    # Initialize cluster column with placeholder
    adata.obs[cluster_key_added] = "N/A" 
    adata.obs[batch_key] = adata.obs[batch_key].astype('category') 

    from ._neighbors import neighbors as _piaso_neighbors
    from ._leiden import leiden as _piaso_leiden

    try:
        # Use tqdm only if available and verbosity is enabled
        iterator = adata.obs[batch_key].cat.categories
        if verbosity > 0:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="Clustering per batch")
            except ImportError:
                if verbosity > 0: print("tqdm not available, not showing progress bar")

        # Process each batch
        for batch in iterator:
            batch_mask = adata.obs[batch_key] == batch
            n_cells_in_batch = np.sum(batch_mask)

            if n_cells_in_batch == 0:
                if verbosity > 1: print(f"Skipping empty batch: {batch}")
                continue

            # Handle small batches
            if n_cells_in_batch < leiden_n_neighbors:
                if verbosity > 0: print(f"Batch {batch}: Assigning single cluster (only {n_cells_in_batch} cells, fewer than k={leiden_n_neighbors})")
                adata.obs.loc[batch_mask, cluster_key_added] = f"{batch}{delimiter}0"
                continue

            # Process normal-sized batches
            try:
                # Create a copy of the batch data to work with
                adata_batch = adata[batch_mask].copy()

                # Compute neighbors (self-contained: leiden reads obsp via adjacency_key)
                _piaso_neighbors(
                    adata_batch,
                    use_rep=use_rep,
                    n_neighbors=leiden_n_neighbors,
                    random_state=random_state,
                    key_added='cluster_graph',
                )

                # Run Leiden clustering
                _piaso_leiden(
                    adata_batch,
                    resolution=leiden_resolution,
                    key_added='leiden_labels',
                    adjacency_key='cluster_graph_connectivities',
                    random_state=random_state,
                )

                # Create combined labels and transfer back to main object
                batch_labels = [f"{batch}{delimiter}{cluster}" for cluster in adata_batch.obs['leiden_labels']]
                adata.obs.loc[batch_mask, cluster_key_added] = batch_labels

                if verbosity > 1:
                    n_clusters = len(set(adata_batch.obs['leiden_labels']))
                    print(f"Batch {batch}: Found {n_clusters} clusters from {n_cells_in_batch} cells")

            except Exception as e:
                # Raised, no longer turned into an 'NA' batch: such a batch lost
                # every edge to the other batches, silently.
                raise RuntimeError(f"stitchSpace: clustering batch {batch!r} failed "
                                   f"({type(e).__name__}: {e})") from e

            finally:
                # Clean up temporary objects and references
                if 'adata_batch' in locals():
                    # Remove the temporary neighbors data to free memory
                    if 'cluster_graph' in adata_batch.uns: 
                        del adata_batch.uns['cluster_graph']
                    for key in ['cluster_graph_connectivities', 'cluster_graph_distances']:
                        if key in adata_batch.obsp:
                            del adata_batch.obsp[key]
                    del adata_batch

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"stitchSpace: the within-batch clustering failed ({type(e).__name__}: {e})") from e

    # Convert result to categorical
    adata.obs[cluster_key_added] = adata.obs[cluster_key_added].astype('category')

    if verbosity > 0: 
        n_clusters = len(adata.obs[cluster_key_added].cat.categories)
        print(f"Within-batch clustering finished. Found {n_clusters} clusters total.")
        print(f"Labels stored in adata.obs['{cluster_key_added}']")
        
        


    # Step 3: Run COSG within each Batch (Local Markers)
    if verbosity > 0: print(f"Identifying top {n_markers} LOCAL markers per batch-cluster using COSG...")
    local_marker_genes = {} 
    all_valid_batch_clusters = set(adata.obs[cluster_key_added][~adata.obs[cluster_key_added].astype(str).str.endswith(f"{delimiter}NA")].astype(str).unique())
    if verbosity > 0: print(f"debug: Found {len(all_valid_batch_clusters)} valid batch-clusters for local COSG.") 
    for bc_label in all_valid_batch_clusters: local_marker_genes[bc_label] = set()

    # Prepare COSG arguments from explicit parameters
    current_cosg_args = {
        'mu': cosg_mu, 
        'expressed_pct': cosg_expressed_pct, 
        'remove_lowly_expressed': cosg_remove_lowly_expressed
    }
    
    # Use tqdm only if available and verbosity is enabled
    iterator = adata.obs[batch_key].cat.categories
    if verbosity > 0:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Running Local COSG per batch")
        except ImportError:
            if verbosity > 0: print("tqdm not available, not showing progress bar")

    if cosg_store is not None:
        # The batch-cluster labels go to the store's cells table, which COSG
        # reads them from (they are also this function's documented output).
        cosg_store["ds"].cells[cluster_key_added] = adata.obs[cluster_key_added].astype(str).values
        cosg_store["ds"].flush()

    # Process each batch
    for batch in iterator:
        batch_mask = (adata.obs[batch_key] == batch)
        if not np.any(batch_mask): continue
        labels_b = adata.obs[cluster_key_added][batch_mask].astype(str)
        valid_clusters_in_batch = sorted({c for c in labels_b if not c.endswith(f"{delimiter}NA")})
        if len(valid_clusters_in_batch) <= 1: continue

        # Errors raise: they used to become a warning and empty marker sets,
        # which pruned every edge of the batch's clusters to the other batches.
        try:
            if cosg_store is not None:
                names_df = _cosg_store_markers(cosg_store, cluster_key_added, batch_mask.values,
                                               n_markers, current_cosg_args)
            else:
                adata_batch = adata[batch_mask].copy()
                adata_batch.obs['groupby_col'] = adata_batch.obs[cluster_key_added].astype('category')
                args = dict(current_cosg_args)
                if cosg_layer is not None:
                    if cosg_layer in adata_batch.layers:
                        args['layer'] = cosg_layer
                    else:
                        warnings.warn(f"Layer '{cosg_layer}' not found in batch {batch}. Using adata.X for COSG.")
                cosg.cosg(adata_batch,
                          key_added='cosg_temp',
                          groupby='groupby_col',
                          n_genes_user=n_markers,
                          use_raw=False,
                          **args)
                names_df = pd.DataFrame(adata_batch.uns['cosg_temp']['names'])
                del adata_batch
        except Exception as e:
            raise RuntimeError(f"stitchSpace: COSG failed on batch {batch!r} "
                               f"({type(e).__name__}: {e})") from e
        local_marker_genes.update(_markers_from_names(names_df, valid_clusters_in_batch, n_markers))

    if verbosity > 0: print("Local marker identification finished.")

    # Step 4: Run COSG Globally (Global Markers) - Conditional
    global_marker_genes = {}
    for bc_label in all_valid_batch_clusters: global_marker_genes[bc_label] = set()

    if use_global_markers:
        if verbosity > 0: print(f"Identifying top {n_markers} GLOBAL markers per batch-cluster using COSG...")
        valid_mask = ~adata.obs[cluster_key_added].astype(str).str.endswith(f"{delimiter}NA")

        if not np.any(valid_mask): 
            print("Skipping Global COSG: No valid batch-clusters found.")
        elif adata.obs[cluster_key_added][valid_mask].nunique() <= 1:
            print("Skipping Global COSG: Not enough distinct valid batch-clusters found.")
        else:
            try:
                if cosg_store is not None:
                    names_df = _cosg_store_markers(cosg_store, cluster_key_added, valid_mask.values,
                                                   n_markers, current_cosg_args)
                else:
                    adata_valid_clusters = adata[valid_mask].copy()
                    adata_valid_clusters.obs['groupby_col'] = adata_valid_clusters.obs[cluster_key_added].astype('category')
                    global_cosg_args = dict(current_cosg_args)
                    if cosg_layer is not None:
                        if cosg_layer in adata_valid_clusters.layers:
                            global_cosg_args['layer'] = cosg_layer
                        else:
                            warnings.warn(f"Layer '{cosg_layer}' not found in full data. Using adata.X for Global COSG.")
                    cosg.cosg(
                        adata_valid_clusters, 
                        key_added='cosg_global_temp', 
                        groupby='groupby_col', 
                        n_genes_user=n_markers, 
                        use_raw=False, 
                        **global_cosg_args
                    )
                    names_df = pd.DataFrame(adata_valid_clusters.uns['cosg_global_temp']['names'])
                    del adata_valid_clusters
            except Exception as e:
                raise RuntimeError(f"stitchSpace: the global COSG failed ({type(e).__name__}: {e})") from e
            global_marker_genes.update(_markers_from_names(
                names_df, [c for c in names_df.columns if c in global_marker_genes], n_markers))

        # Ensure all batch-clusters have an entry (even if empty)
        for bc_label in all_valid_batch_clusters:
            if bc_label not in global_marker_genes:
                global_marker_genes[bc_label] = set()

        if verbosity > 0: 
            print("Global marker identification finished.")
    else:
        if verbosity > 0: 
            print("Skipping Global marker identification as 'use_global_markers' is False.")

    # Step 5: Build Combined Compatibility Lookup
    if verbosity > 0: print("Comparing marker lists and building combined compatibility lookup...")
    batch_cluster_pairs = list(itertools.combinations(all_valid_batch_clusters, 2))
    allow_connection_combined = {} 
    n_compatible_pairs = 0 
    
    # Use the robust batch extraction map
    bc_to_batch_map = {}
    for bc_label in all_valid_batch_clusters:
        first_cell_index = adata.obs[adata.obs[cluster_key_added] == bc_label].index[0]
        original_batch = adata.obs[batch_key].loc[first_cell_index]
        bc_to_batch_map[bc_label] = str(original_batch) 

    for bc1, bc2 in batch_cluster_pairs: 
        bc1_str, bc2_str = str(bc1), str(bc2) 
        batch1 = bc_to_batch_map.get(bc1_str)
        batch2 = bc_to_batch_map.get(bc2_str)
        if batch1 is not None and batch2 is not None and batch1 != batch2:
            local_markers1 = local_marker_genes.get(bc1_str, set())
            local_markers2 = local_marker_genes.get(bc2_str, set())
            overlap_local = calculate_jaccard(local_markers1, local_markers2)
            allow_local = overlap_local >= marker_overlap_threshold
            if use_global_markers:
                global_markers1 = global_marker_genes.get(bc1_str, set())
                global_markers2 = global_marker_genes.get(bc2_str, set())
                overlap_global = calculate_jaccard(global_markers1, global_markers2)
                allow_global = overlap_global >= marker_overlap_threshold
                is_compatible = allow_local and allow_global
            else: is_compatible = allow_local 
            allow_connection_combined[(bc1_str, bc2_str)] = is_compatible
            allow_connection_combined[(bc2_str, bc1_str)] = is_compatible
            if is_compatible: n_compatible_pairs += 1 
    if verbosity > 0:
        print(f"Combined compatibility lookup built (use_global={use_global_markers}). Threshold = {marker_overlap_threshold}")
        print(f"debug: Number of compatible inter-batch cluster pairs: {n_compatible_pairs}") 

    # Step 6: Filter Cell-Cell Graph using Vectorized Lookup
    if verbosity > 0: print("Pruning graph using vectorized lookup ...")
    
    # 1. Map Batch-Cluster Labels to Integers
    # Sorted list ensures deterministic index mapping
    unique_bc_labels = sorted(list(all_valid_batch_clusters))
    bc_to_int = {label: i for i, label in enumerate(unique_bc_labels)}
    n_unique_bc = len(unique_bc_labels)
    
    if n_unique_bc == 0:
        # Fallback if no clusters exist
        return sp.csr_matrix(connectivities.shape), allow_connection_combined, local_marker_genes, global_marker_genes

    # 2. Build Dense Boolean Compatibility Matrix (Small: N_clusters x N_clusters)
    compatibility_matrix = np.zeros((n_unique_bc, n_unique_bc), dtype=bool)
    
    # Fill matrix from the dictionary we built in Step 5
    # We only iterate over compatible pairs found, which is fast
    for (bc1, bc2), is_allowed in allow_connection_combined.items():
        if is_allowed and bc1 in bc_to_int and bc2 in bc_to_int:
            idx1, idx2 = bc_to_int[bc1], bc_to_int[bc2]
            compatibility_matrix[idx1, idx2] = True
    
    # 3. Create Cell-to-Cluster Integer Mapping Array
    # Initialize with -1 (cells with no valid cluster)
    cell_bc_indices = np.full(n_cells, -1, dtype=int)
    
    # Get the current labels from adata
    obs_bc_labels = adata.obs[cluster_key_added].astype(str).values
    
    # Vectorized mapping: 
    # Ideally we use pandas codes, but since we have custom strings, we map carefully.
    # This loop runs once per cluster (e.g. 50 times), not per cell (50k times).
    for label, idx in bc_to_int.items():
        # boolean mask for cells belonging to this cluster
        mask = (obs_bc_labels == label)
        cell_bc_indices[mask] = idx

    # 4. Process Graph in Coordinate Format (COO)
    # COO format allows direct access to all row (source) and col (target) indices
    coo = connectivities.tocoo()
    
    # Get cluster indices for every edge's source and target
    src_bc_idxs = cell_bc_indices[coo.row]
    tgt_bc_idxs = cell_bc_indices[coo.col]
    
    # 5. Identify Valid Edges (Marker Compatible)
    # First, ignore edges where either cell has no valid cluster (-1)
    valid_cluster_mask = (src_bc_idxs != -1) & (tgt_bc_idxs != -1)
    
    # Initialize keep mask
    marker_keep_mask = np.zeros(coo.nnz, dtype=bool)
    
    # Perform the vectorized lookup in the compatibility matrix
    # This is the magic step: numpy handles the indexing for millions of edges instantly
    if np.any(valid_cluster_mask):
        marker_keep_mask[valid_cluster_mask] = compatibility_matrix[
            src_bc_idxs[valid_cluster_mask], 
            tgt_bc_idxs[valid_cluster_mask]
        ]
        
    # 6. Identify Intra-Batch Edges (Always Keep)
    # Get integer codes for batches to speed up comparison
    batch_codes = adata.obs[batch_key].astype('category').cat.codes.values
    intra_batch_mask = (batch_codes[coo.row] == batch_codes[coo.col])
    
    # 7. Combine Logic: Keep if (Intra-Batch) OR (Inter-Batch AND Compatible)
    final_edge_mask = intra_batch_mask | marker_keep_mask
    
    # 8. Rebuild CSR Matrix
    # We only pass the data and indices that passed the mask
    connectivities_pruned = sp.csr_matrix(
        (coo.data[final_edge_mask], (coo.row[final_edge_mask], coo.col[final_edge_mask])),
        shape=connectivities.shape
    )
    
    if verbosity > 0:
        n_before = connectivities.nnz
        n_after = connectivities_pruned.nnz
        print(f"Graph pruned: {n_before} -> {n_after} edges (Removed {n_before - n_after}).")

    return connectivities_pruned, allow_connection_combined, local_marker_genes, global_marker_genes
    
    


# --- Main Function Combining Filtering and Single-Step Correction ---
def stitchSpace(
    adata: AnnData,
    batch_key: str,
    use_rep: str = 'X_pca',
    key_added: str = 'X_stitch', 
    # --- Graph Filtering Params ---
    filter_cluster_key_added: typing.Union[str, None] = None, 
    filter_pruned_graph_key: typing.Union[str, None] = None, 
    filter_use_global_markers: bool = False, 
    filter_leiden_resolution: float = 0.5,
    filter_leiden_n_neighbors: int = 15,
    filter_n_markers: int = 50,
    filter_marker_overlap_threshold: float = 0.1,
    # --- COSG specific params ---
    filter_cosg_layer: typing.Union[str, None] = None, # Layer for COSG
    filter_cosg_mu: float = 100.0, # Default 100.0 (float)
    filter_cosg_expressed_pct: float = 0.1,
    filter_cosg_remove_lowly_expressed: bool = True,
    # --- BBKNN specific params ---
    filter_bbknn_neighbors_within_batch: int = 3, 
    filter_bbknn_trim: typing.Union[int, None] = None,
    filter_bbknn_computation: str = 'annoy',
    filter_bbknn_metric: str = 'euclidean',
    random_state: typing.Union[int, None] = 1927, # Seed for clustering, default 1927
    # --- Correction Params ---
    correction_smooth_within_batch: bool = True,
    correction_use_mutual_sqrt_weights: bool = False, 
    copy: bool = False,
    verbosity: int = 0, # Control overall verbosity (0=minimal, 1+=more info)
    modality: str = 'RNA',
) -> typing.Union[AnnData, None]:
    """
    Performs a batch correction using a BBKNN graph that has been pruned based on marker gene overlap between batch-specific 
    clusters. Overlap check uses local markers and optionally global markers (controlled by `filter_use_global_markers`).

    Clusters are identified internally using Leiden and stored in `adata.obs[filter_cluster_key_added]`.
    Markers identified by COSG.
    Intermediate results like the compatibility 'hypergraph', marker gene lists, and the pruned graph 
    structure are stored in `adata.uns` and `adata.obsp`/`adata.uns`.

    The correction moves each cell towards the average position of its neighbors in the pruned graph.

    Parameters
    ----------
    adata
        Annotated data matrix. Needs expression data for COSG (in .X or specified layer).
    batch_key
        Key in `adata.obs` for batch information.
    use_rep
        Representation in `adata.obsm` for BBKNN, clustering, and correction (e.g., 'X_pca').
    key_added
        Base key for storing results. Corrected embedding will be in 
        `adata.obsm[key_added]`. Intermediate results stored in `adata.uns`.
    filter_cluster_key_added
        Key in `adata.obs` where generated batch-cluster labels will be stored. 
        If None, a default key is generated (e.g., f"{batch_key}@leiden@res{res}").
    filter_pruned_graph_key
        Base key for storing the pruned graph structure in `adata.obsp` and `adata.uns`.
        If None, defaults to "pruned_markers". Connectivities/distances will be stored
        as `{filter_pruned_graph_key}_connectivities`/`_distances`.
    filter_use_global_markers
        If True, run global COSG and require BOTH local AND global marker overlap 
        for inter-batch cluster compatibility. If False (default), only local overlap is used.
    filter_leiden_resolution
        Resolution parameter for internal within-batch Leiden clustering.
    filter_leiden_n_neighbors
        KNN parameter for internal within-batch Leiden clustering's graph.
    filter_n_markers
        Number of top COSG markers to compare between clusters.
    filter_marker_overlap_threshold
        Minimum Jaccard index for marker overlap to consider clusters compatible.
    filter_cosg_layer
        Layer in `adata.layers` to use for COSG marker identification. If None (default), uses `adata.X`.
    filter_cosg_mu
        `mu` parameter for COSG (default: 100.0). Higher values increase sparsity.
    filter_cosg_expressed_pct
        `expressed_pct` parameter for COSG (default: 0.1). Minimum expression pct for a gene.
    filter_cosg_remove_lowly_expressed
        `remove_lowly_expressed` parameter for COSG (default: True). Filter lowly expressed genes.
    filter_bbknn_neighbors_within_batch
        `neighbors_within_batch` parameter for the initial `bbknn.bbknn` call.
    filter_bbknn_trim
        Optional `trim` parameter passed to the initial `bbknn.bbknn` call.
    filter_bbknn_computation
        BBKNN's neighbour search: ``'annoy'`` (default), ``'pynndescent'``, ``'faiss'`` or
        ``'cKDTree'``. It was always ``'annoy'`` before, whatever was asked.
    filter_bbknn_metric
        Distance for BBKNN's neighbour search.
    random_state
        Seed for the random number generator used in Leiden clustering for reproducibility.
        Default: 1927.
    correction_smooth_within_batch
        If True, smooth the correction vector within batches using the pruned graph structure.
    correction_use_mutual_sqrt_weights
        If True, applies symmetrization and sqrt weighting to the *pruned* graph 
        before the correction step.
    copy
        If True, return a modified copy of `adata`. Otherwise, modify `adata` inplace.
    verbosity
        Level of detail to print: 0 (minimal), 1 or higher (more progress messages 
        and intermediate storage locations). Default: 0. Controls BBKNN logging level.
    modality
        Cytome only: the modality whose embedding ``use_rep`` names and whose expression COSG
        reads (``filter_cosg_layer`` names a layer of it; ``None`` reads what ``cosg`` reads
        by default, ``layer='auto'``). The batch-cluster labels are written to the store's
        cells table and the corrected embedding as ``{modality}_{key_added without 'X_'}``.

    A failure of BBKNN, of a batch's clustering or of COSG raises: it used to print a
    message and leave the graph without that batch's links, or return ``None``, which is
    also what success returns.

    Returns
    -------
    AnnData or None
        If `copy=True`, returns the modified AnnData object. 
        Otherwise, modifies the input `adata` object inplace and returns None.
        Adds/updates:
        - `adata.obsm[key_added]`: The corrected embedding.
        - `adata.obs[filter_cluster_key_added]`: Generated batch-cluster labels (using '@' delimiter).
        - `adata.uns[f'{key_added}_hypergraph_compatibility']`: Compatibility dict.
        - `adata.uns[f'{key_added}_local_markers']`: Local marker dict.
        - `adata.uns[f'{key_added}_global_markers']`: Global marker dict.
        - `adata.obsp[f'{filter_pruned_graph_key}_connectivities']`: Pruned graph connectivities.
        - `adata.obsp[f'{filter_pruned_graph_key}_distances']`: Pruned graph dummy distances.
        - `adata.uns[filter_pruned_graph_key]`: Neighbors dictionary for pruned graph.
        - `adata.uns[f'{key_added}_params']`: Dictionary of parameters used.
        
    Example
    -------
    >>> import anndata
    >>> import piaso
    >>> adata = anndata.read_h5ad('pbmc68k.h5ad')
    >>> # Simulate batches (replace with actual batch info)
    >>> adata.obs['batch'] = ['A' if i % 2 == 0 else 'B' for i in range(adata.n_obs)]
    >>> # Assume normalized data is in adata.layers['log1p']
    >>> adata.layers['log1p'] = adata.X.copy()
    >>> # Run correction using log1p layer for COSG, increased verbosity
    >>> piaso.tl.stitchSpace(
    ...     adata,
    ...     batch_key='batch',
    ...     use_rep='X_pca',
    ...     key_added='X_stitch_corrected',
    ...     filter_cluster_key_added='batch@cluster_stitch',
    ...     filter_cosg_layer='log1p',
    ...     random_state=1927,
    ...     verbosity=1
    ... )
    >>> # Visualize results
    >>> piaso.tl.neighbors(adata, use_rep='X_stitch_corrected')
    >>> piaso.tl.umap(adata)
    >>> piaso.pl.plotEmbedding(adata, color='batch')
    >>> piaso.pl.plotEmbedding(adata, color='batch@cluster_stitch')
    """
        
    # --- Cytome dispatch ---
    # A store's embedding and batch column go into a lightweight AnnData for
    # BBKNN, the clustering and the correction; COSG reads the store's own
    # expression (run_cosg_cytome, per batch). Before, the embedding was looked
    # up as use_rep minus 'X_' (a name PIASO never writes), COSG ran on a dummy
    # one-column matrix and failed into empty marker sets (every edge between
    # batches pruned: no integration), and the write-back assigned into an
    # accessor that does not support it.
    from ._normalization import _is_cytome_dataset, _open_cytome
    _is_cytome = _is_cytome_dataset(adata) or isinstance(adata, str)
    _cytome_ds = None
    _cosg_store = None
    if _is_cytome:
        import anndata
        from ._neighbors import _load_embedding_from_cytome
        _cytome_ds = _open_cytome(adata) if isinstance(adata, str) else adata
        from ..utils._cytome_compat import require_writable as _require_writable
        _require_writable(_cytome_ds, "piaso.tl.stitchSpace")
        X_emb_arr = np.asarray(_load_embedding_from_cytome(_cytome_ds, use_rep, modality))
        cols = {r[1] for r in _cytome_ds._conn.execute("PRAGMA table_info(cells)").fetchall()}
        if batch_key not in cols:
            raise ValueError(f"batch_key '{batch_key}' not found in cytome cells table. Available: {sorted(cols)}")
        batch_vals = np.asarray(_cytome_ds.cells[batch_key])
        n_cells = X_emb_arr.shape[0]
        if len(batch_vals) != n_cells:
            raise ValueError(f"stitchSpace: the embedding has {n_cells} rows and the cells table "
                             f"{len(batch_vals)}")
        adata = anndata.AnnData(
            X=sp.csr_matrix((n_cells, 1), dtype=np.float32),   # placeholder: COSG reads the store
            obs=pd.DataFrame({batch_key: pd.Series(batch_vals).astype(str).values}),
        )
        adata.obsm[use_rep] = X_emb_arr
        _cosg_store = {"ds": _cytome_ds, "modality": modality,
                       "layer": filter_cosg_layer if filter_cosg_layer is not None else "auto"}
        copy = False  # operate in-place on the temporary AnnData

    # --- Input Validation ---
    if filter_n_markers <= 0:
        raise ValueError("filter_n_markers must be greater than 0")
    if filter_marker_overlap_threshold < 0 or filter_marker_overlap_threshold > 1:
        raise ValueError("filter_marker_overlap_threshold must be between 0 and 1")
    if filter_bbknn_neighbors_within_batch <= 0:
        raise ValueError("filter_bbknn_neighbors_within_batch must be greater than 0")

    X_emb = adata.obsm[use_rep]
    # Check for NaNs
    if np.isnan(X_emb).any():
        raise ValueError(f"Your representation '{use_rep}' contains NaN values. This causes BBKNN to segfault.")
    # Check for Infinite values
    if np.isinf(X_emb).any():
        raise ValueError(f"Your representation '{use_rep}' contains Infinite values. This causes BBKNN to segfault.")
    if verbosity > 0: print(f"Data integrity check passed for {use_rep}.")
        
    adata_original = adata # Keep reference to original if copy=False
    if copy:
        adata = adata.copy()
        
    # Determine the key for storing cluster labels
    if filter_cluster_key_added is None:
        # Create a default key if none provided
        filter_cluster_key_added = f"{batch_key}@leiden@res{filter_leiden_resolution}" # Use @ delimiter
        if verbosity > 0: print(f"Cluster labels will be stored in adata.obs['{filter_cluster_key_added}']")
         
    # Determine the key for storing the pruned graph structure
    if filter_pruned_graph_key is None:
        filter_pruned_graph_key = "pruned_markers" # Default base name
        if verbosity > 0: print(f"Pruned graph structure will be stored using base key '{filter_pruned_graph_key}'")


    # Stage 1: Build the marker-filtered graph
    if verbosity > 0: print(f"Stage 1: Building Marker-Filtered Graph (Use Global Markers: {filter_use_global_markers})", flush=True)
    # Pass the adata object to be modified with cluster labels
    pruned_connectivities, hypergraph_compatibility, local_markers, global_markers = _build_marker_filtered_graph_dual(
        adata=adata, 
        batch_key=batch_key,
        use_rep=use_rep,
        neighbors_within_batch=filter_bbknn_neighbors_within_batch,
        leiden_resolution=filter_leiden_resolution,
        leiden_n_neighbors=filter_leiden_n_neighbors,
        cluster_key_added=filter_cluster_key_added, # Pass the key
        random_state=random_state, # Pass seed
        n_markers=filter_n_markers,
        marker_overlap_threshold=filter_marker_overlap_threshold,
        use_global_markers=filter_use_global_markers, # Pass control flag
        # Pass explicit COSG args instead of kwargs dict
        cosg_layer=filter_cosg_layer,
        cosg_mu=filter_cosg_mu,
        cosg_expressed_pct=filter_cosg_expressed_pct,
        cosg_remove_lowly_expressed=filter_cosg_remove_lowly_expressed,
        bbknn_trim=filter_bbknn_trim,
        bbknn_computation=filter_bbknn_computation,
        bbknn_metric=filter_bbknn_metric,
        verbosity=verbosity, # Pass main verbosity level down
        cosg_store=_cosg_store,
    )
        
    # Store intermediate results
    adata.uns[f'{key_added}_hypergraph_compatibility'] = hypergraph_compatibility
    adata.uns[f'{key_added}_local_markers'] = local_markers
    adata.uns[f'{key_added}_global_markers'] = global_markers # Will be empty if use_global_markers=False
    
    # Store pruned graph structure for UMAP/downstream use
    conn_key = f"{filter_pruned_graph_key}_connectivities"
    dist_key = f"{filter_pruned_graph_key}_distances"
    adata.obsp[conn_key] = pruned_connectivities
    
    # Updated Dummy Distance Calculation
    connectivities_nonneg = pruned_connectivities.copy()
    if hasattr(connectivities_nonneg, 'data'):
         connectivities_nonneg.data = np.maximum(0, connectivities_nonneg.data)
    dist_coo = connectivities_nonneg.tocoo()
    non_zero_mask = dist_coo.data > 1e-6 
    dist_data = np.ones_like(dist_coo.data) 
    dist_data[non_zero_mask] = 1.0 / dist_coo.data[non_zero_mask] 
    dummy_distances = sp.coo_matrix(
        (dist_data, (dist_coo.row, dist_coo.col)),
        shape=connectivities_nonneg.shape
    ).tocsr()
    dummy_distances.eliminate_zeros()

    adata.obsp[dist_key] = dummy_distances
    
    # Store info in uns[neighbors_key] like sc.pp.neighbors does
    adata.uns[filter_pruned_graph_key] = {
        'connectivities_key': conn_key,
        'distances_key': dist_key,
        'params': { # Store relevant parameters used to create this graph
            'method': 'marker_filtered_bbknn',
            'filter_use_global_markers': filter_use_global_markers, 
            'filter_leiden_resolution': filter_leiden_resolution,
            'filter_leiden_n_neighbors': filter_leiden_n_neighbors,
            'filter_n_markers': filter_n_markers,
            'filter_marker_overlap_threshold': filter_marker_overlap_threshold,
            'filter_cosg_layer': filter_cosg_layer, # Store COSG layer used
            'filter_cosg_mu': filter_cosg_mu,
            'filter_cosg_expressed_pct': filter_cosg_expressed_pct,
            'filter_cosg_remove_lowly_expressed': filter_cosg_remove_lowly_expressed,
            'filter_bbknn_neighbors_within_batch': filter_bbknn_neighbors_within_batch,
            'filter_bbknn_trim': filter_bbknn_trim,
            'random_state': random_state 
            # Add more relevant params as needed
        }
    }

    # Stage 2: Perform Single-Step Correction using the pruned graph
    if verbosity > 0: print("\nStage 2: Single-Step Correction using Filtered Graph")

    Z_orig = adata.obsm[use_rep].copy()
    if sp.issparse(Z_orig):
         Z_orig = Z_orig.toarray()

    n_cells = adata.n_obs
    batch_labels_numeric = adata.obs[batch_key].astype('category').cat.codes.values
    n_batches = len(adata.obs[batch_key].astype('category').cat.categories)

    # Use the pruned connectivities directly from obsp
    connectivities_iter = adata.obsp[conn_key].copy() 

    # Optional Graph Modification (on pruned graph)
    if correction_use_mutual_sqrt_weights:
        if verbosity > 0: print("Applying mutual neighbor selection and sqrt weighting to pruned graph...")
        connectivities_iter = connectivities_iter.multiply(connectivities_iter.T)
        if hasattr(connectivities_iter, 'data'):
            if connectivities_iter.data.size > 0: 
                connectivities_iter.data = np.sqrt(connectivities_iter.data)
                connectivities_iter.data = np.nan_to_num(connectivities_iter.data)
        connectivities_iter.eliminate_zeros()
        if verbosity > 0: print("Pruned graph modified.")

    # Prepare Pruned Graph for Correction (Row-normalize)
    adj = normalize(connectivities_iter, norm='l1', axis=1)

    # Prepare within-batch smoothing matrix (based on pruned graph)
    adj_within = None
    if correction_smooth_within_batch:
        if verbosity > 0: print("Preparing within-batch smoothing matrix from pruned graph (Masking Approach)...")
        # 1. Create a mask for intra-batch edges
        batch_labels_arr = adata.obs[batch_key].astype(str).values
        # Use the connectivities that might have been modified (mutual/sqrt)
        connectivities_smooth_base = connectivities_iter.tocoo() 
        
        row_batches = batch_labels_arr[connectivities_smooth_base.row]
        col_batches = batch_labels_arr[connectivities_smooth_base.col]
        intra_batch_mask = (row_batches == col_batches)
        
        # 2. Create the intra-batch graph by applying the mask
        adj_within_unnormalized = sp.coo_matrix(
            (connectivities_smooth_base.data[intra_batch_mask], 
             (connectivities_smooth_base.row[intra_batch_mask], connectivities_smooth_base.col[intra_batch_mask])),
            shape=connectivities_iter.shape
        ).tocsr()
        
        # 3. Symmetrize (optional but recommended for smoothing) and Normalize
        adj_within_unnormalized = adj_within_unnormalized + adj_within_unnormalized.T
        adj_within = normalize(adj_within_unnormalized, norm='l1', axis=1)
        adj_within.eliminate_zeros() # Clean up after normalization
        
        if adj_within is not None and adj_within.nnz > 0: 
             if verbosity > 0: print("Within-batch smoothing matrix prepared.")
        else:
             warnings.warn("Within-batch smoothing matrix is empty. Skipping smoothing.")
             adj_within = None # Ensure it's None if empty

    # Single-Step Correction Calculation
    if verbosity > 0: print(f"Running single-step correction...")
    # Use Z_orig for calculation, store result in Z_corrected
    Z_current = Z_orig 

    # Calculate Target Centroid (T) based on pruned graph
    T = adj @ Z_current 

    # Calculate Correction Vector (V)
    V = T - Z_current 

    # Smooth Correction Vector (V) within batches (optional)
    if correction_smooth_within_batch and adj_within is not None:
        # Check dimensions and non-empty status before applying
        if V.shape[0] == adj_within.shape[1] and adj_within.nnz > 0: 
             if verbosity > 0: print("Applying within-batch smoothing to correction vector...")
             V_smoothed = adj_within @ V 
             V = V_smoothed # Use smoothed vector
        else: 
             # Provide more specific warning if smoothing is skipped
             if V.shape[0] != adj_within.shape[1]:
                  warnings.warn(f"Dimension mismatch smoother ({adj_within.shape[1]}) vs vector ({V.shape[0]}). Skipping smoothing.")
             elif adj_within.nnz == 0: # Check explicitly for empty smoother
                  warnings.warn("Smoother matrix is empty (nnz=0). Skipping smoothing.")

    # Update Embeddings with alpha=1.0
    Z_corrected = Z_current + V 
            
    # print("Correction finished.")

    # Store Final Results
    adata.obsm[key_added] = Z_corrected # Store the final corrected embedding
    # Store parameters, including the key where cluster labels were saved
    adata.uns[f'{key_added}_params'] = {
        'use_rep': use_rep,
        'batch_key': batch_key,
        'method': 'StitchSpace', # Renamed method
        'filter_params': {
             'cluster_key_added': filter_cluster_key_added, 
             'pruned_graph_key': filter_pruned_graph_key, 
             'use_global_markers': filter_use_global_markers, 
             'leiden_resolution': filter_leiden_resolution,
             'leiden_n_neighbors': filter_leiden_n_neighbors,
             'random_state': random_state, # Store seed
             'n_markers': filter_n_markers,
             'marker_overlap_threshold': filter_marker_overlap_threshold,
             # Store explicit COSG parameters
             'cosg_layer': filter_cosg_layer, 
             'cosg_mu': filter_cosg_mu,
             'cosg_expressed_pct': filter_cosg_expressed_pct,
             'cosg_remove_lowly_expressed': filter_cosg_remove_lowly_expressed,
             'bbknn_neighbors_within_batch': filter_bbknn_neighbors_within_batch,
             'bbknn_trim': filter_bbknn_trim,
             # 'bbknn_verbose': filter_bbknn_verbose # Removed
        },
        'correction_params':{
             # Removed alpha and max_iter as they are fixed
             'smooth_within_batch': correction_smooth_within_batch,
             'use_mutual_sqrt_weights': correction_use_mutual_sqrt_weights
        },
        'key_added': key_added,
        'verbosity': verbosity # Store verbosity level used
    }

    print(f"The corrected embeddings stored in adata.obsm['{key_added}']")
    # Control final summary prints with verbosity flag
    if verbosity > 0:
        print(f"Batch-cluster labels stored in adata.obs['{filter_cluster_key_added}']")
        print(f"Hypergraph compatibility stored in adata.uns['{key_added}_hypergraph_compatibility']")
        print(f"Local markers stored in adata.uns['{key_added}_local_markers']")
        print(f"Global markers stored in adata.uns['{key_added}_global_markers']")
        print(f"Pruned graph structure stored using base key '{filter_pruned_graph_key}'")


    # Return the modified adata if copy=False, otherwise return the copy
    # Write corrected embedding back to cytome if applicable
    if _is_cytome and _cytome_ds is not None and key_added in adata.obsm:
        from ._embedding_names import storage_name
        from ..settings import _resolve_layer_dtype
        emb_out_key = storage_name(key_added, modality)
        _cytome_ds.add_embedding(
            emb_out_key, np.asarray(adata.obsm[key_added]),
            dtype=_resolve_layer_dtype(None),
            provenance={"modality": modality, "function": "piaso.tl.stitchSpace",
                        "use_rep": use_rep, "batch_key": batch_key, "key_added": key_added,
                        "cluster_key": filter_cluster_key_added})
        _cytome_ds.flush()
        if verbosity > 0:
            print(f"Corrected embedding written to cytome: embeddings['{emb_out_key}']")
        return None

    return adata if copy else None 
