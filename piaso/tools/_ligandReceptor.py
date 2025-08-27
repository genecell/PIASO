import numpy as np
import pandas as pd
import anndata
from scipy import sparse
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
from statsmodels.stats.multitest import multipletests
from typing import List, Optional

def _calculate_interaction_scores(
    specificity_matrix: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    ligand_col: str,
    receptor_col: str,
    sender_cell_types: List[str],
    receiver_cell_types: List[str],
    annotation_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Internal helper to efficiently calculate ligand-receptor interaction scores.
    Assumes lr_pairs have already been validated.
    """
    # --- 1. Select relevant columns ---
    lr_cols_to_keep = [ligand_col, receptor_col]
    if annotation_col and annotation_col in lr_pairs.columns:
        lr_cols_to_keep.append(annotation_col)
    valid_lr_pairs = lr_pairs[lr_cols_to_keep]

    # --- 2. Prepare Ligand and Receptor Data ---
    ligands = valid_lr_pairs[ligand_col].unique()
    receptors = valid_lr_pairs[receptor_col].unique()

    ligand_spec = specificity_matrix.loc[ligands, sender_cell_types].reset_index().rename(columns={'index': ligand_col})
    ligand_long = ligand_spec.melt(id_vars=ligand_col, var_name='sender', value_name='ligand_score')

    receptor_spec = specificity_matrix.loc[receptors, receiver_cell_types].reset_index().rename(columns={'index': receptor_col})
    receptor_long = receptor_spec.melt(id_vars=receptor_col, var_name='receiver', value_name='receptor_score')

    # --- 3. Merge and Calculate Score ---
    merged_df = pd.merge(valid_lr_pairs, ligand_long, on=ligand_col)
    merged_df = pd.merge(merged_df, receptor_long, on=receptor_col)
    merged_df['interaction_score'] = merged_df['ligand_score'] * merged_df['receptor_score']

    # --- 4. Clean Up ---
    final_cols = [ligand_col, receptor_col]
    if annotation_col and annotation_col in lr_pairs.columns:
        final_cols.append(annotation_col)
    final_cols.extend(['sender', 'receiver', 'interaction_score'])
    
    result_df = merged_df[final_cols].rename(columns={ligand_col: 'ligand', receptor_col: 'receptor'})
    
    return result_df

def _prepare_background_genes(
    adata: anndata.AnnData,
    layer: str = None,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 30
) -> pd.DataFrame:
    """
    Pre-calculates a nearest-neighbor graph for all genes based on their
    mean expression and variance.
    """
    print("Preparing background gene set by calculating mean and variance for all genes...")
    if layer is not None and layer in adata.layers:
        expression_matrix = adata.layers[layer]
    else:
        expression_matrix = adata.X

    if not sparse.isspmatrix_csr(expression_matrix):
        expression_matrix = expression_matrix.tocsr()

    mean_expression = np.array(expression_matrix.mean(axis=0)).flatten()
    mean_squared_expression = np.array(expression_matrix.power(2).mean(axis=0)).flatten()
    variance_expression = mean_squared_expression - (mean_expression ** 2)
    mean_variance_data = np.array([mean_expression, variance_expression]).T

    print(f"Building KDTree to find {n_nearest_neighbors} nearest neighbors for each gene...")
    kdt = KDTree(mean_variance_data, leaf_size=leaf_size, metric='euclidean')
    _, indices = kdt.query(mean_variance_data, k=n_nearest_neighbors + 1)
    
    neighbor_indices = indices[:, 1:]
    knn_df = pd.DataFrame(neighbor_indices, index=adata.var_names)
    print("Finished preparing background gene set.")
    return knn_df

### The SCALAR function for the cell type-specific ligand-receptor anlayis
def runSCALAR(
    adata: anndata.AnnData,
    specificity_matrix: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    ligand_col: str = 'ligand',
    receptor_col: str = 'receptor',
    annotation_col: Optional[str] = None,
    sender_cell_types: Optional[List[str]] = None,
    receiver_cell_types: Optional[List[str]] = None,
    n_permutations: int = 1000,
    n_nearest_neighbors: int = 30,
    layer: str = None,
    random_seed: int = 42,
    rank_by_score: bool = True,
    chunk_size: int = 50000,
    prefilter_fdr: bool = True,
    prefilter_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Calculates ligand-receptor interaction scores, computes permutation-based p-values
    using a vectorized approach, and corrects for multiple testing using FDR for each
    cell type-cell type pair independently.

    Args:
        adata: AnnData object with gene expression data.
        specificity_matrix: DataFrame with genes as rows, cell types as columns,
                            and specificity scores as values.
        lr_pairs: DataFrame listing interacting gene pairs.
        ligand_col: Column name for ligands in lr_pairs.
        receptor_col: Column name for receptors in lr_pairs.
        annotation_col: Optional column in lr_pairs to carry over.
        sender_cell_types: List of cell types to use as senders. If None, all are used.
        receiver_cell_types: List of cell types to use as receivers. If None, all are used.
        n_permutations: Number of permutations for the null distribution.
        n_nearest_neighbors: Number of control genes to sample from.
        layer: Layer in adata to use for expression.
        random_seed: Seed for reproducibility.
        rank_by_score: If True, sorts the final output by interaction_score.
        chunk_size: The number of interactions to process in each vectorized chunk
                    to manage memory usage.
        prefilter_fdr: If True, interactions with scores <= prefilter_threshold are
                       excluded from FDR calculation within each group and assigned an
                       FDR of 1.0.
        prefilter_threshold: The score threshold used for pre-filtering before FDR calculation.

    Returns:
        A pandas DataFrame with interaction scores, p-values, and FDR-corrected p-values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # --- Step 1: Validate inputs and filter LR pairs ---
    print("--- Step 1: Validating inputs and filtering LR pairs ---")
    genes_in_adata = set(adata.var_names)
    genes_in_spec = set(specificity_matrix.index)
    common_genes = genes_in_adata.intersection(genes_in_spec)

    if not common_genes:
        print("Error: No common genes found between AnnData object and specificity matrix.")
        return pd.DataFrame()

    original_pair_count = len(lr_pairs)
    lr_pairs_filtered = lr_pairs[
        lr_pairs[ligand_col].isin(common_genes) &
        lr_pairs[receptor_col].isin(common_genes)
    ].copy()

    if len(lr_pairs_filtered) < original_pair_count:
        print(f"Filtered out {original_pair_count - len(lr_pairs_filtered)} LR pairs that were not found in both the AnnData object and the specificity matrix.")

    if lr_pairs_filtered.empty:
        print("Warning: No valid ligand-receptor pairs found after filtering. Returning an empty DataFrame.")
        return pd.DataFrame()

    # --- Step 2: Calculate all observed interaction scores ---
    print("\n--- Step 2: Calculating all observed interaction scores ---")
    all_cell_types = specificity_matrix.columns.tolist()
    senders = all_cell_types if sender_cell_types is None else [ct for ct in sender_cell_types if ct in all_cell_types]
    receivers = all_cell_types if receiver_cell_types is None else [ct for ct in receiver_cell_types if ct in all_cell_types]

    interaction_scores_df = _calculate_interaction_scores(
        specificity_matrix=specificity_matrix,
        lr_pairs=lr_pairs_filtered, # Use the filtered dataframe
        ligand_col=ligand_col,
        receptor_col=receptor_col,
        sender_cell_types=senders,
        receiver_cell_types=receivers,
        annotation_col=annotation_col
    )

    if interaction_scores_df.empty:
        print("Warning: No valid interactions found. Returning an empty DataFrame.")
        return pd.DataFrame()
    print(f"Found {len(interaction_scores_df)} potential interactions to test.")

    # --- 3. Prepare background gene set for permutation testing ---
    print("\n--- Step 3: Preparing background gene set for permutation testing ---")
    background_genes_df = _prepare_background_genes(
        adata, layer=layer, n_nearest_neighbors=n_nearest_neighbors
    )
    all_genes_in_adata = adata.var_names

    # --- 4. Calculate p-values for each interaction (Vectorized) ---
    print(f"\n--- Step 4: Calculating p-values via {n_permutations} permutations (vectorized) ---")
    
    p_values_list = []
    # Group by sender-receiver pairs to vectorize calculations within each group
    grouped_interactions = interaction_scores_df.groupby(['sender', 'receiver'])

    for (sender, receiver), group_df in tqdm(grouped_interactions, total=len(grouped_interactions), desc="Processing cell type pairs"):
        
        # Process each group in chunks to manage memory
        for i in range(0, len(group_df), chunk_size):
            chunk_df = group_df.iloc[i:i + chunk_size]
            n_in_chunk = len(chunk_df)

            # Get data for the current chunk
            ligands = chunk_df['ligand'].values
            receptors = chunk_df['receptor'].values
            observed_scores = chunk_df['interaction_score'].values
            
            # Vectorized sampling of control genes for the entire chunk
            ligand_neighbor_indices = background_genes_df.loc[ligands].values
            receptor_neighbor_indices = background_genes_df.loc[receptors].values
            
            # Create random indices to sample from neighbor lists for each interaction
            rand_ligand_indices = np.random.randint(0, ligand_neighbor_indices.shape[1], size=(n_in_chunk, n_permutations))
            rand_receptor_indices = np.random.randint(0, receptor_neighbor_indices.shape[1], size=(n_in_chunk, n_permutations))

            # Use advanced indexing to get the final control gene indices
            row_idx = np.arange(n_in_chunk)[:, np.newaxis]
            ligand_control_indices = ligand_neighbor_indices[row_idx, rand_ligand_indices]
            receptor_control_indices = receptor_neighbor_indices[row_idx, rand_receptor_indices]
            
            # Vectorized score lookup for the null distribution
            sender_scores_all = specificity_matrix[sender].reindex(all_genes_in_adata).values
            receiver_scores_all = specificity_matrix[receiver].reindex(all_genes_in_adata).values
            
            null_ligand_scores = sender_scores_all[ligand_control_indices]
            null_receptor_scores = receiver_scores_all[receptor_control_indices]
            
            null_distribution = null_ligand_scores * null_receptor_scores
            
            # Vectorized p-value calculation by comparing observed scores to the null matrix
            is_greater = null_distribution >= observed_scores[:, np.newaxis]
            p_values_chunk = (np.sum(is_greater, axis=1) + 1) / (n_permutations + 1)
            
            p_values_list.append(pd.Series(p_values_chunk, index=chunk_df.index))

    # Concatenate all p-value series and add to the main dataframe
    interaction_scores_df['p_value'] = pd.concat(p_values_list)
    
    # --- 5. FDR Correction per Cell-Type Pair ---
    print("\n--- Step 5: Applying FDR (Benjamini/Hochberg) correction per cell-type pair ---")
    
    # Initialize the new column with NaNs
    interaction_scores_df['p_value_fdr'] = np.nan

    # Iterate through each group to apply FDR and assign back correctly
    for _, group_df in tqdm(interaction_scores_df.groupby(['sender', 'receiver']), desc="FDR Correction"):
        
        p_values = group_df['p_value']
        if p_values.empty or p_values.isnull().all():
            continue
            
            
        # Round p-values to 6 decimal places to improve stability of FDR calculation
        p_values = p_values.round(6)

        if prefilter_fdr:
            # Create a mask for interactions that pass the threshold
            filter_mask = group_df['interaction_score'] > prefilter_threshold
            p_values_to_test = p_values[filter_mask]

            if not p_values_to_test.empty:
                # Apply FDR correction only to the filtered set
                _, p_values_corrected, _, _ = multipletests(p_values_to_test, method='fdr_bh')
                interaction_scores_df.loc[p_values_to_test.index, 'p_value_fdr'] = p_values_corrected
            
            # For interactions that were filtered out, set their FDR to 1.0
            interaction_scores_df.loc[p_values[~filter_mask].index, 'p_value_fdr'] = 1.0
        
        else: # Original behavior if pre-filtering is off
            _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            interaction_scores_df.loc[group_df.index, 'p_value_fdr'] = p_values_corrected

    # Clip FDR p-values to be at most 1.0 to handle potential floating point inaccuracies.
    interaction_scores_df['p_value_fdr'] = interaction_scores_df['p_value_fdr'].clip(upper=1.0)

    # Calculate the -log10 of the FDR p-value
    interaction_scores_df['nlog10_p_value_fdr'] = -np.log10(interaction_scores_df['p_value_fdr'] + 1e-10)
    
    # Correct any small negative values resulting from p-values of 1.0
    interaction_scores_df['nlog10_p_value_fdr'] = interaction_scores_df['nlog10_p_value_fdr'].clip(lower=0)


    # --- 6. Final Sorting ---
    if rank_by_score:
        interaction_scores_df = interaction_scores_df.sort_values(by='interaction_score', ascending=False)
    else:
        # Sort by cell types first, then by score or p-value if desired
        sort_order = ['sender', 'receiver', 'interaction_score']
        interaction_scores_df = interaction_scores_df.sort_values(by=sort_order, ascending=[True, True, False])

    print("\nAnalysis complete.")
    return interaction_scores_df.reset_index(drop=True)