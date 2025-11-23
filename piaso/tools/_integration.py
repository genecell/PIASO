# import faulthandler
# faulthandler.enable()

# import os
# # Force thread count to 1 to prevent C-library collisions
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'



import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
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
    
# # Import and version check for BBKNN
# try:
#     import bbknn
#     import bbknn.matrix
#     import pkg_resources

#     # Get BBKNN version
#     bbknn_version = pkg_resources.get_distribution("bbknn").version
#     is_old_version = pkg_resources.parse_version(bbknn_version) <= pkg_resources.parse_version('1.5.1')


#     print(f"Successfully imported BBKNN version {bbknn_version}")
#     print(f"Using parameters for version {'<= 1.5.1' if is_old_version else '> 1.5.1'}")

# except ImportError as e:
#     print(f"Error importing BBKNN: {e}")
#     print("BBKNN is required for this function. Please install it with 'pip install bbknn'.")

    

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

# Helper Function for Building the Pruned Graph
def _build_marker_filtered_graph_dual( 
    adata: sc.AnnData,
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
    verbosity: int = 0 # Control internal verbosity of this helper
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
    
    
    
    # Import and version check for BBKNN
    try:
        import bbknn
        import bbknn.matrix
        import pkg_resources

        # Get BBKNN version
        bbknn_version = pkg_resources.get_distribution("bbknn").version
        is_old_version = pkg_resources.parse_version(bbknn_version) <= pkg_resources.parse_version('1.5.1')

        if verbosity > 0: 
            print(f"Successfully imported BBKNN version {bbknn_version}")
            print(f"Using parameters for version {'<= 1.5.1' if is_old_version else '> 1.5.1'}")

    except ImportError as e:
        print(f"Error importing BBKNN: {e}")
        print("BBKNN is required for this function. Please install it with 'pip install bbknn'.")
        return None, None, None, None
    
    
    
    # Step 1: Build Initial BBKNN Graph using bbknn.matrix.bbknn with version checking
    if verbosity > 0: print(f"Building initial graph via BBKNN (neighbors_within_batch={neighbors_within_batch})...")

    original_verbosity_bbknn = sc.settings.verbosity
    sc.settings.verbosity = 3 if verbosity > 0 else 1
    

    try:
        # Check BBKNN version
        import pkg_resources
        bbknn_version = pkg_resources.get_distribution("bbknn").version
        is_old_version = pkg_resources.parse_version(bbknn_version) <= pkg_resources.parse_version('1.5.1')
        if verbosity > 0: print(f"Detected BBKNN version: {bbknn_version} (is_old_version={is_old_version})")

        # Extract cell embeddings and batch information
        cell_embeddings = adata.obsm[use_rep]
        batch_list = adata.obs[batch_key].values

        # Base parameters for all versions
        base_params = {
            'neighbors_within_batch': neighbors_within_batch,
            'trim': bbknn_trim,
            'metric': bbknn_metric,
            'n_pcs': cell_embeddings.shape[1],  # Use all available dimensions
        }

        # Version-specific parameter handling
        if is_old_version:
            # For versions <= 1.5.1, use the older parameter set
            params_to_use = {
                **base_params,
                'approx': True,
                'annoy_n_trees': 10,
                'use_annoy': True,
                'use_faiss': False if 'use_faiss' in locals() and use_faiss is not None else True
            }

            if verbosity > 0: 
                print(f"Using parameters for version <= 1.5.1: approx=True, use_annoy=True")
        else:
            computation_param = 'annoy'  # Default
            # For versions > 1.5.1 (e.g., 1.6.0+), use 'computation' parameter
            if hasattr(locals(), 'bbknn_computation') and bbknn_computation is not None:
                computation_param = bbknn_computation
            
            if computation_param == 'annoy':
                try:
                    import pkg_resources
                    annoy_version = pkg_resources.get_distribution("annoy").version

                    # Check if version is >= 1.17.0
                    if pkg_resources.parse_version(annoy_version) >= pkg_resources.parse_version('1.17.0'):
                        error_msg = (
                            f"\n\nCRITICAL ERROR: Incompatible Annoy version detected ({annoy_version}).\n"
                            "Annoy versions >= 1.17.0 are known to cause Segmentation Faults (Core Dumps) with BBKNN.\n"
                            "--------------------------------------------------------\n"
                            "PLEASE RUN THE FOLLOWING COMMAND TO FIX THIS:\n"
                            "pip install annoy==1.16.3\n"
                            "--------------------------------------------------------\n"
                            "(Remember to restart your kernel after installing!)\n"
                        )
                        raise ImportError(error_msg)

                except pkg_resources.DistributionNotFound:
                    # If annoy isn't installed, BBKNN will raise its own error later, so we can skip
                    pass
                

            params_to_use = {
                **base_params,
                'computation': computation_param,
                'annoy_n_trees': 10
            }

            if verbosity > 0: 
                print(f"Using parameters for version > 1.5.1: computation={computation_param}")

        # Remove any None values
        params_to_use = {k: v for k, v in params_to_use.items() if v is not None}

        # Call bbknn.matrix.bbknn with the appropriate parameters
        if verbosity > 0: 
            print(f"Calling bbknn.matrix.bbknn with parameters: {params_to_use}")

        distances, connectivities, parameters = bbknn.matrix.bbknn(
            cell_embeddings,
            batch_list,
            **params_to_use
        )

        # Check if results were generated correctly
        if connectivities is None:
            raise ValueError("BBKNN did not return valid connectivities.")

        if verbosity > 0: print("Initial graph built using bbknn.matrix.bbknn successfully.")

    except Exception as e:
        print(f"Error during BBKNN graph construction: {e}")
        sc.settings.verbosity = original_verbosity_bbknn
        return None, None, None, None
    finally:
        sc.settings.verbosity = original_verbosity_bbknn
        
    

    # Step 2: Clustering within Batches (with verbosity control)
    if verbosity > 0: print(f"Clustering within batches (Leiden res={leiden_resolution}, k={leiden_n_neighbors})...")

    # Initialize cluster column with placeholder
    adata.obs[cluster_key_added] = "N/A" 
    adata.obs[batch_key] = adata.obs[batch_key].astype('category') 

    # Control verbosity for Leiden
    original_verbosity_leiden = sc.settings.verbosity
    sc.settings.verbosity = 0  # Always silence Leiden

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

                # Compute neighbors
                sc.pp.neighbors(
                    adata_batch, 
                    n_neighbors=leiden_n_neighbors, 
                    use_rep=use_rep, 
                    key_added='cluster_graph',
                    random_state=random_state
                )

                # Run Leiden clustering
                sc.tl.leiden(
                    adata_batch, 
                    resolution=leiden_resolution, 
                    key_added='leiden_labels', 
                    neighbors_key='cluster_graph', 
                    random_state=random_state
                ) 

                # Create combined labels and transfer back to main object
                batch_labels = [f"{batch}{delimiter}{cluster}" for cluster in adata_batch.obs['leiden_labels']]
                adata.obs.loc[batch_mask, cluster_key_added] = batch_labels

                if verbosity > 1:
                    n_clusters = len(set(adata_batch.obs['leiden_labels']))
                    print(f"Batch {batch}: Found {n_clusters} clusters from {n_cells_in_batch} cells")

            except Exception as e:
                print(f"Error clustering batch {batch}: {str(e)}")
                print(f"Assigning '{batch}{delimiter}NA' to all cells in this batch")
                adata.obs.loc[batch_mask, cluster_key_added] = f"{batch}{delimiter}NA"

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

    except Exception as e:
        print(f"Error during batch clustering process: {str(e)}")

    finally:
        # Restore original verbosity
        sc.settings.verbosity = original_verbosity_leiden

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

    # Process each batch
    for batch in iterator:
        batch_mask = (adata.obs[batch_key] == batch)
        if not np.any(batch_mask): continue 
        adata_batch = adata[batch_mask].copy() 
        if cluster_key_added not in adata_batch.obs: continue
         
        # Use the combined label for grouping
        adata_batch.obs['groupby_col'] = adata_batch.obs[cluster_key_added].astype('category')
        valid_clusters_in_batch = [str(c) for c in adata_batch.obs['groupby_col'].cat.categories if not str(c).endswith(f"{delimiter}NA")]
        if len(valid_clusters_in_batch) <= 1: continue 
         
        

        try:
            # Check if layer exists and update COSG args appropriately
            if cosg_layer is not None:
                if cosg_layer in adata_batch.layers:
                    current_cosg_args['layer'] = cosg_layer
                else:
                    warnings.warn(f"Layer '{cosg_layer}' not found in batch {batch}. Using adata.X for COSG.")
                    # Don't add layer param - COSG will default to using adata.X

            # Run COSG directly on adata_batch with appropriate parameters
            cosg.cosg(adata_batch, 
                      key_added='cosg_temp', 
                      groupby='groupby_col', 
                      n_genes_user=n_markers,
                      use_raw=False, 
                      **current_cosg_args)

            # Process results
            if 'cosg_temp' in adata_batch.uns and 'names' in adata_batch.uns['cosg_temp']:
                cosg_names_df = pd.DataFrame(adata_batch.uns['cosg_temp']['names'])

                # Process in a more vectorized way
                for bc_label in valid_clusters_in_batch:
                    if bc_label in cosg_names_df.columns:
                        markers = cosg_names_df[bc_label].dropna().tolist()[:n_markers]
                        local_marker_genes[bc_label] = set(markers)
                    else:
                        local_marker_genes[bc_label] = set()

            del adata_batch  # Delete original copy when done

        except Exception as e:
            warnings.warn(f"Error running Local COSG for batch {batch}: {e}")
            if 'adata_batch' in locals(): 
                del adata_batch

            # Initialize empty marker sets for this batch
            for bc_label in valid_clusters_in_batch:
                if bc_label not in local_marker_genes:
                    local_marker_genes[bc_label] = set()

    if verbosity > 0: print("Local marker identification finished.")

    # Step 4: Run COSG Globally (Global Markers) - Conditional
    global_marker_genes = {}
    for bc_label in all_valid_batch_clusters: global_marker_genes[bc_label] = set()

    if use_global_markers:
        if verbosity > 0: print(f"Identifying top {n_markers} GLOBAL markers per batch-cluster using COSG...")
        valid_mask = ~adata.obs[cluster_key_added].astype(str).str.endswith(f"{delimiter}NA")

        if not np.any(valid_mask): 
            print("Skipping Global COSG: No valid batch-clusters found.")
        else:
            adata_valid_clusters = adata[valid_mask].copy() 

            if len(adata_valid_clusters.obs[cluster_key_added].unique()) > 1:
                try:
                    adata_valid_clusters.obs['groupby_col'] = adata_valid_clusters.obs[cluster_key_added].astype('category')

                    # Update COSG args based on layer availability
                    global_cosg_args = current_cosg_args.copy()
                    if cosg_layer is not None:
                        if cosg_layer in adata_valid_clusters.layers:
                            global_cosg_args['layer'] = cosg_layer
                        else:
                            warnings.warn(f"Layer '{cosg_layer}' not found in full data. Using adata.X for Global COSG.")
                            # If layer not present, don't add to args - default to X
                            if 'layer' in global_cosg_args:
                                del global_cosg_args['layer']

                    # Run COSG directly on filtered AnnData
                    cosg.cosg(
                        adata_valid_clusters, 
                        key_added='cosg_global_temp', 
                        groupby='groupby_col', 
                        n_genes_user=n_markers, 
                        use_raw=False, 
                        **global_cosg_args
                    )

                    # Process results efficiently
                    if ('cosg_global_temp' in adata_valid_clusters.uns and 
                        'names' in adata_valid_clusters.uns['cosg_global_temp']):

                        cosg_global_names_df = pd.DataFrame(adata_valid_clusters.uns['cosg_global_temp']['names'])

                        # Extract markers for each batch-cluster in one pass
                        for bc_label in cosg_global_names_df.columns:
                            if bc_label in global_marker_genes:
                                markers = cosg_global_names_df[bc_label].dropna().tolist()
                                global_marker_genes[bc_label] = set(markers[:n_markers])



                    del adata_valid_clusters

                except Exception as e:
                    warnings.warn(f"Error running Global COSG: {e}. Global markers might be missing.")
                    if 'adata_valid_clusters' in locals(): 
                        del adata_valid_clusters
            else: 
                print("Skipping Global COSG: Not enough distinct valid batch-clusters found.")
                del adata_valid_clusters

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
    
    
#     # Step 6: Filter Cell-Cell Graph using Vectorized Lookup
#     if verbosity > 0: print("Pruning the initial graph using vectorized lookup...")
#     n_edges_before = connectivities.nnz

#     # Convert labels to arrays once
#     batch_labels_arr = adata.obs[batch_key].astype(str).values
#     bc_labels_arr = adata.obs[cluster_key_added].astype(str).values

#     # Create mapping for batch-cluster labels to indices
#     unique_bc_labels = sorted(list(all_valid_batch_clusters))
#     bc_to_int = {label: i for i, label in enumerate(unique_bc_labels)}
#     n_unique_bc = len(unique_bc_labels)

#     if n_unique_bc == 0:
#         warnings.warn("No valid batch-clusters found for building compatibility matrix.")
#         connectivities_pruned = sp.csr_matrix(connectivities.shape, dtype=connectivities.dtype)
#     else:
#         # Build compatibility matrix more efficiently
#         compatibility_matrix = np.zeros((n_unique_bc, n_unique_bc), dtype=bool)

#         # Batch process compatible pairs
#         compatible_pairs = [(bc1, bc2) for (bc1, bc2), is_compatible 
#                             in allow_connection_combined.items() 
#                             if is_compatible]

#         for bc1, bc2 in compatible_pairs:
#             idx1 = bc_to_int.get(str(bc1))
#             idx2 = bc_to_int.get(str(bc2))
#             if idx1 is not None and idx2 is not None:
#                 compatibility_matrix[idx1, idx2] = True

#         # Ensure symmetry in one vectorized operation
#         compatibility_matrix = compatibility_matrix | compatibility_matrix.T

#         # Process in CSR format for memory efficiency
#         connectivities_csr = connectivities.tocsr()
#         invalid_bc_index = -1

#         # Prepare arrays for building new CSR matrix
#         data = []
#         indices = []
#         indptr = [0]

#         # Fast batch lookup arrays
#         batch_lookup = {batch: i for i, batch in enumerate(np.unique(batch_labels_arr))}
#         batch_indices = np.array([batch_lookup[b] for b in batch_labels_arr])

#         # Pre-compute batch-cluster indices for all cells
#         all_bc_indices = np.array([bc_to_int.get(str(label), invalid_bc_index) 
#                                   for label in bc_labels_arr])

#         # Process row by row to avoid materializing huge arrays
#         for i in range(connectivities_csr.shape[0]):
#             row_start, row_end = connectivities_csr.indptr[i], connectivities_csr.indptr[i+1]
#             if row_start == row_end:  # Empty row
#                 indptr.append(len(indices))
#                 continue

#             # Get batch and cluster info for source cell
#             source_batch_idx = batch_indices[i]
#             source_bc_idx = all_bc_indices[i]

#             # Get target cells and values
#             target_indices = connectivities_csr.indices[row_start:row_end]
#             target_data = connectivities_csr.data[row_start:row_end]

#             # Get batch indices for targets
#             target_batch_indices = batch_indices[target_indices]

#             # Keep intra-batch connections
#             intra_batch_mask = target_batch_indices == source_batch_idx

#             # Check inter-batch connections
#             if not intra_batch_mask.all() and source_bc_idx != invalid_bc_index:
#                 inter_batch_mask = ~intra_batch_mask
#                 inter_batch_targets = target_indices[inter_batch_mask]
#                 inter_batch_data = target_data[inter_batch_mask]

#                 # Get cluster indices for inter-batch targets
#                 target_bc_indices = all_bc_indices[inter_batch_targets]

#                 # Check which edges to keep
#                 valid_targets = target_bc_indices != invalid_bc_index
#                 if np.any(valid_targets):
#                     keep_inter = np.zeros(len(inter_batch_targets), dtype=bool)
#                     valid_indices = np.where(valid_targets)[0]

#                     # Vectorized lookup in compatibility matrix
#                     src_idx_array = np.full(len(valid_indices), source_bc_idx)
#                     tgt_idx_array = target_bc_indices[valid_indices]
#                     keep_inter[valid_indices] = compatibility_matrix[src_idx_array, tgt_idx_array]

#                     # Add the valid inter-batch connections
#                     valid_inter_indices = inter_batch_targets[keep_inter]
#                     valid_inter_data = inter_batch_data[keep_inter]

#                     indices.extend(valid_inter_indices)
#                     data.extend(valid_inter_data)

#             # Add all intra-batch connections
#             if np.any(intra_batch_mask):
#                 intra_indices = target_indices[intra_batch_mask]
#                 intra_data = target_data[intra_batch_mask]
#                 indices.extend(intra_indices)
#                 data.extend(intra_data)

#             indptr.append(len(indices))

#         # Create pruned CSR matrix directly
#         connectivities_pruned = sp.csr_matrix(
#             (data, indices, indptr),
#             shape=connectivities.shape
#         )
#         connectivities_pruned.eliminate_zeros()

#     # Diagnostic Info
#     n_edges_after = connectivities_pruned.nnz
#     if verbosity > 0:
#         pruned_count = n_edges_before - n_edges_after
#         pruned_percent = (pruned_count / n_edges_before * 100) if n_edges_before > 0 else 0
#         print(f"Pruned {pruned_count} edges ({pruned_percent:.1f}%) based on marker compatibility.")
        
    
#     if verbosity > 0: print("Graph pruning finished.")
#     # Return all results needed by the main function
#     return connectivities_pruned, allow_connection_combined, local_marker_genes, global_marker_genes 


# --- Main Function Combining Filtering and Single-Step Correction ---
def stitchSpace( 
    adata: sc.AnnData,
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
    random_state: typing.Union[int, None] = 1927, # Seed for clustering, default 1927
    # --- Correction Params ---
    correction_smooth_within_batch: bool = True,
    correction_use_mutual_sqrt_weights: bool = False, 
    copy: bool = False,
    verbosity: int = 0 # Control overall verbosity (0=minimal, 1+=more info)
) -> typing.Union[sc.AnnData, None]:
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
    >>> import scanpy as sc
    >>> import stitchSpaceModule # Assuming the code is saved as stitchSpaceModule.py
    >>> adata = sc.datasets.pbmc68k_reduced() 
    >>> # Simulate batches (replace with actual batch info)
    >>> adata.obs['batch'] = ['A' if i % 2 == 0 else 'B' for i in range(adata.n_obs)]
    >>> # Assume normalized data is in adata.layers['log1p']
    >>> adata.layers['log1p'] = adata.X.copy() 
    >>> # Precompute PCA if not done
    >>> sc.tl.pca(adata) 
    >>> # Run correction using log1p layer for COSG, increased verbosity
    >>> piaso.tl.stitchSpace(
    ...     adata, 
    ...     batch_key='batch', 
    ...     use_rep='X_pca', 
    ...     key_added='X_stitch_corrected',
    ...     filter_cluster_key_added='batch@cluster_stitch', # Optional: specify key
    ...     filter_cosg_layer='log1p', # Specify layer for COSG
    ...     random_state=1927, 
    ...     verbosity=1 
    ... )
    >>> # Visualize results
    >>> sc.pp.neighbors(adata, use_rep='X_stitch_corrected') # Compute neighbors on corrected embedding
    >>> sc.tl.umap(adata)
    >>> sc.pl.umap(adata, color=['batch', 'batch@cluster_stitch']) 
    >>> # Visualize UMAP based on the pruned graph itself
    >>> sc.tl.umap(adata, neighbors_key='pruned_markers') # Use default key or specify if changed
    >>> sc.pl.umap(adata, color=['batch', 'batch@cluster_stitch'], title="UMAP on Pruned Graph")
    """
        
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
        verbosity=verbosity # Pass main verbosity level down
        # Pass other bbknn params if needed, e.g.:
        # bbknn_computation='annoy', bbknn_metric='euclidean'
    )

    if pruned_connectivities is None:
        print("Graph building/filtering failed. Aborting correction.")
        # Attempt cleanup of cluster key if it exists and copy=False
        if not copy and filter_cluster_key_added in adata_original.obs:
            # Check if the key actually exists before deleting
            if filter_cluster_key_added in adata_original.obs:
                 del adata_original.obs[filter_cluster_key_added]
        return None
        
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
    return adata if copy else None 
