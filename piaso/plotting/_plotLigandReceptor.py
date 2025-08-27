import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

def plotLigandReceptorInteraction(
    interactions_df: pd.DataFrame,
    specificity_df: pd.DataFrame,
    cell_type_pairs: list,
    top_n: int = 50,
    y_max: int = 10,
    cell_type_sep: str = '@',
    ligand_receptor_sep: str = '-->',
    heatmap_height_ratio: float = 1.5,
    heatmap_cmap: str = 'Purples',
    heatmap_cmap_ligand: str = None,
    heatmap_cmap_receptor: str = None,
    shared_legend: bool = False,
    heatmap_vmax: float = None,
    save_path: str = None,
    fig_width: int = 24,
    fig_height_per_pair: int = 9,
    col_interaction_score: str = 'interaction_score',
    col_ligand_receptor_pair: str = 'ligandXreceptor',
    col_cell_type_pair: str = 'CellTypeXCellType',
    col_annotation: str = 'annotation',
    col_ligand: str = 'ligand',
    col_receptor: str = 'receptor',
    vertical_layout: bool = False,
    color_labels_by_annotation: bool = False,
    barplot_palette: str = 'Paired',
    sort_by_category: bool = False,
    category_agg_method: str = 'sum',
    preserve_input_order: bool = False
):
    """
    Generates plots with a bar plot of top interactions and a heatmap showing 
    ligand and receptor specificity, with an option for vertical orientation.

    Args:
        interactions_df (pd.DataFrame): DataFrame with interaction scores.
        specificity_df (pd.DataFrame): DataFrame with gene specificity scores.
        cell_type_pairs (list): A list of 'CellTypeXCellType' strings to plot.
        top_n (int): The number of top interactions to display.
        y_max (int): The maximum y-axis value for the bar plot.
        cell_type_sep (str): The separator for sender/receiver cell types.
        ligand_receptor_sep (str): The separator for ligand/receptor genes.
        heatmap_height_ratio (float): The height/width ratio of the heatmap relative to the bar plot.
        heatmap_cmap (str): The colormap for the specificity heatmap (used when ligand/receptor cmaps not specified).
        heatmap_cmap_ligand (str): The colormap for ligand specificity. If None, uses heatmap_cmap.
        heatmap_cmap_receptor (str): The colormap for receptor specificity. If None, uses heatmap_cmap.
        shared_legend (bool): If True, a single legend/colorbar is shown for all plots.
        heatmap_vmax (float): The maximum value for the heatmap color scale. 
        save_path (str, optional): Path to save the figure (e.g., 'plot.pdf').
        fig_width (int): For horizontal layout, the total figure width. For vertical, this controls the total figure HEIGHT.
        fig_height_per_pair (int): For horizontal layout, height per subplot. For vertical, this controls the WIDTH of each subplot group.
        col_interaction_score (str): Column name for interaction scores.
        col_ligand_receptor_pair (str): Column name for ligand-receptor pair strings.
        col_cell_type_pair (str): Column name for cell type pair strings.
        col_annotation (str): Column name for pathway/annotation data.
        col_ligand (str): Column name for ligand after splitting.
        col_receptor (str): Column name for receptor after splitting.
        vertical_layout (bool): If True, plots are arranged horizontally (rotated 90 degrees).
        color_labels_by_annotation (bool): If True, color ligand-receptor labels by their annotation category.
        barplot_palette (str or list): Color palette for bar plots. Can be a seaborn palette name (e.g., 'Paired', 'Set1') 
                                      or a list of hex colors (e.g., ['#F198CC', '#D6DAB9', '#BC938B']).
        sort_by_category (bool): If True, sort interactions by category first, then by interaction score within category.
        category_agg_method (str): Method to aggregate interaction scores by category ('sum' or 'mean') when sort_by_category=True.
        preserve_input_order (bool): If True, preserve the original order from interactions_df without any sorting.
    
    Examples:
        # Preserve original input order
        plotLigandReceptorInteraction(
            interactions_df=specific_interactions,
            specificity_df=cosg_scores,
            cell_type_pairs=['L5 NP@SST-Chrna2'],
            preserve_input_order=True,  # Use original DataFrame order
            vertical_layout=False
        )
        
        # Horizontal layout with category sorting
        plotLigandReceptorInteraction(
            interactions_df=specific_interactions,
            specificity_df=cosg_scores,
            cell_type_pairs=['L5 NP@SST-Chrna2', 'L5 PT@SST-Chrna2'],
            ligand_receptor_sep='-->',
            top_n=50,
            y_max=10,
            heatmap_cmap_ligand='Blues',
            heatmap_cmap_receptor='Reds',
            shared_legend=True,
            vertical_layout=False,
            sort_by_category=True,
            category_agg_method='sum',
            color_labels_by_annotation=True
        )
        
        # Vertical layout with custom hex colors
        plotLigandReceptorInteraction(
            interactions_df=specific_interactions_cellchat,
            specificity_df=cosg_scores,
            cell_type_pairs=['L5 NP@SST-Chrna2'],
            ligand_receptor_sep='-->',
            top_n=50,
            y_max=10,
            heatmap_cmap_ligand='Purples',
            heatmap_cmap_receptor='Reds',
            shared_legend=True,
            vertical_layout=True,
            barplot_palette=['#F198CC', '#D6DAB9', '#BC938B', '#93DCFC', '#F4DBCD', '#bcf60c'],
            sort_by_category=True,
            category_agg_method='mean'
        )
        
    Raises:
        ValueError: If required columns are missing from DataFrames or if data is inconsistent.
        KeyError: If specified cell type pairs are not found in the data.
    """
    
    # ERROR HANDLING
    # Check if DataFrames are empty
    if interactions_df.empty:
        raise ValueError("interactions_df is empty")
    if specificity_df.empty:
        raise ValueError("specificity_df is empty")
    
    # Check required columns in interactions_df
    required_interaction_cols = [col_interaction_score, col_ligand_receptor_pair, col_cell_type_pair, col_annotation]
    missing_cols = [col for col in required_interaction_cols if col not in interactions_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in interactions_df: {missing_cols}")
    
    # Check if cell_type_pairs is valid
    if not cell_type_pairs or not isinstance(cell_type_pairs, list):
        raise ValueError("cell_type_pairs must be a non-empty list")
    
    # Check if cell type pairs exist in data
    available_pairs = set(interactions_df[col_cell_type_pair].unique())
    missing_pairs = [pair for pair in cell_type_pairs if pair not in available_pairs]
    if missing_pairs:
        raise KeyError(f"Cell type pairs not found in data: {missing_pairs}. Available pairs: {sorted(available_pairs)}")
    
    # Check cell_type_sep format
    for pair in cell_type_pairs:
        if cell_type_sep not in pair:
            raise ValueError(f"Cell type pair '{pair}' does not contain separator '{cell_type_sep}'")
        parts = pair.split(cell_type_sep)
        if len(parts) != 2:
            raise ValueError(f"Cell type pair '{pair}' should have exactly one separator '{cell_type_sep}'")
    
    # Check ligand_receptor_sep format in data
    sample_interactions = interactions_df[col_ligand_receptor_pair].head()
    invalid_pairs = [pair for pair in sample_interactions if ligand_receptor_sep not in pair]
    if invalid_pairs:
        raise ValueError(f"Some ligand-receptor pairs don't contain separator '{ligand_receptor_sep}': {invalid_pairs[:3]}...")
    
    # Check numeric parameters
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if y_max <= 0:
        raise ValueError("y_max must be positive")
    if fig_width <= 0 or fig_height_per_pair <= 0:
        raise ValueError("Figure dimensions must be positive")
    if heatmap_height_ratio <= 0:
        raise ValueError("heatmap_height_ratio must be positive")
    
    # Check if specificity_df has the required cell types
    all_cell_types = set()
    for pair in cell_type_pairs:
        sender, receiver = pair.split(cell_type_sep)
        all_cell_types.update([sender, receiver])
    
    missing_cell_types = [ct for ct in all_cell_types if ct not in specificity_df.columns]
    if missing_cell_types:
        raise KeyError(f"Cell types not found in specificity_df columns: {missing_cell_types}")
    
    # Check if barplot_palette is valid
    if isinstance(barplot_palette, list):
        # Check if it's a list of valid hex colors
        for i, color in enumerate(barplot_palette):
            if not isinstance(color, str) or not color.startswith('#') or len(color) != 7:
                raise ValueError(f"Invalid hex color at index {i}: '{color}'. Expected format: '#RRGGBB'")
    elif not isinstance(barplot_palette, str):
        raise ValueError("barplot_palette must be either a string (seaborn palette name) or a list of hex colors")
    
    # Check category sorting parameters
    if not isinstance(sort_by_category, bool):
        raise ValueError("sort_by_category must be a boolean")
    if category_agg_method not in ['sum', 'mean']:
        raise ValueError("category_agg_method must be either 'sum' or 'mean'")
    if not isinstance(preserve_input_order, bool):
        raise ValueError("preserve_input_order must be a boolean")
    
    # Check parameter compatibility
    if preserve_input_order and sort_by_category:
        raise ValueError("preserve_input_order and sort_by_category cannot both be True")
    
    # Warn if no interactions will be plotted
    total_valid_interactions = 0
    for pair in cell_type_pairs:
        pair_interactions = interactions_df[interactions_df[col_cell_type_pair] == pair]
        if not pair_interactions.empty:
            total_valid_interactions += len(pair_interactions)
    
    if total_valid_interactions == 0:
        raise ValueError("No valid interactions found for the specified cell type pairs")
    n_pairs = len(cell_type_pairs)
    
    # Set up colormaps
    cmap_ligand = heatmap_cmap_ligand if heatmap_cmap_ligand else heatmap_cmap
    cmap_receptor = heatmap_cmap_receptor if heatmap_cmap_receptor else heatmap_cmap
    different_cmaps = (heatmap_cmap_ligand is not None and heatmap_cmap_receptor is not None and 
                       heatmap_cmap_ligand != heatmap_cmap_receptor)
    
    if vertical_layout:
        # For vertical layout, width scales with n_pairs, height is fixed.
        # Adjust height ratios to give more space to plots and less to legend
        fig = plt.figure(figsize=(fig_height_per_pair * n_pairs, fig_width))
        main_gs = gridspec.GridSpec(2, n_pairs, figure=fig, wspace=0.4, height_ratios=[12, 1])
    else:
        # For horizontal layout, height scales with n_pairs, width is fixed.
        fig = plt.figure(figsize=(fig_width, fig_height_per_pair * n_pairs))
        main_gs = gridspec.GridSpec(n_pairs, 2, figure=fig, hspace=0.8, width_ratios=[15, 3])

    all_annotations = sorted(list(interactions_df[col_annotation].unique()))
    
    # CHANGE: Set up color palette based on barplot_palette parameter
    if isinstance(barplot_palette, list):
        # Custom hex colors provided
        if len(barplot_palette) < len(all_annotations):
            # Extend the palette by repeating colors if not enough provided
            extended_palette = (barplot_palette * ((len(all_annotations) // len(barplot_palette)) + 1))[:len(all_annotations)]
            color_palette = dict(zip(all_annotations, extended_palette))
        else:
            color_palette = dict(zip(all_annotations, barplot_palette[:len(all_annotations)]))
    else:
        # Seaborn palette name provided
        try:
            palette_colors = sns.color_palette(barplot_palette, len(all_annotations))
            color_palette = dict(zip(all_annotations, palette_colors))
        except ValueError:
            # Fallback to 'Paired' if invalid palette name
            print(f"Warning: Invalid palette '{barplot_palette}', using 'Paired' instead")
            palette_colors = sns.color_palette('Paired', len(all_annotations))
            color_palette = dict(zip(all_annotations, palette_colors))
    
    global_vmax = float('-inf')
    if shared_legend and heatmap_vmax is None:
        for pair in cell_type_pairs:
            sender, receiver = pair.split(cell_type_sep)
            top_interactions = interactions_df[interactions_df[col_cell_type_pair] == pair].nlargest(top_n, col_interaction_score)
            if top_interactions.empty: continue
            df_copy = top_interactions.copy()
            df_copy[[col_ligand, col_receptor]] = df_copy[col_ligand_receptor_pair].str.split(ligand_receptor_sep, expand=True)
            mask = (df_copy[col_ligand].isin(specificity_df.index)) & (df_copy[col_receptor].isin(specificity_df.index))
            for _, row in df_copy[mask].iterrows():
                l_score = specificity_df.loc[row[col_ligand], sender]
                r_score = specificity_df.loc[row[col_receptor], receiver]
                global_vmax = max(global_vmax, l_score, r_score)

    legend_handles, legend_labels, mappable = None, None, None
    mappable_ligand, mappable_receptor = None, None

    for idx, cell_type_pair in enumerate(cell_type_pairs):
        sender_cell_type, receiver_cell_type = cell_type_pair.split(cell_type_sep)
        
        # CHANGE: Handle different sorting options
        if preserve_input_order:
            # Use original order from DataFrame, just filter by cell type pair and take top_n
            pair_interactions = interactions_df[interactions_df[col_cell_type_pair] == cell_type_pair]
            top_interactions = pair_interactions.head(top_n)  # Take first top_n in original order
        else:
            # Default behavior: sort by interaction score first
            top_interactions = interactions_df[interactions_df[col_cell_type_pair] == cell_type_pair].nlargest(top_n, col_interaction_score)
        
        if top_interactions.empty: continue
        
        filtered_interactions = top_interactions.copy()
        filtered_interactions[[col_ligand, col_receptor]] = filtered_interactions[col_ligand_receptor_pair].str.split(ligand_receptor_sep, expand=True)
        valid_mask = (filtered_interactions[col_ligand].isin(specificity_df.index)) & (filtered_interactions[col_receptor].isin(specificity_df.index))
        filtered_interactions = filtered_interactions[valid_mask]
        if filtered_interactions.empty: continue
        
        # CHANGE: Apply category sorting only if requested and not preserving input order
        if sort_by_category and not preserve_input_order:
            # Calculate category aggregated scores
            category_scores = filtered_interactions.groupby(col_annotation)[col_interaction_score].agg(category_agg_method).sort_values(ascending=False)
            
            # Create a custom sort key for each interaction
            category_rank = {cat: i for i, cat in enumerate(category_scores.index)}
            filtered_interactions['category_rank'] = filtered_interactions[col_annotation].map(category_rank)
            
            # Sort by category rank first, then by interaction score within category (descending)
            filtered_interactions = filtered_interactions.sort_values(
                ['category_rank', col_interaction_score], 
                ascending=[True, False]
            ).reset_index(drop=True)
            
            # Clean up the temporary column
            filtered_interactions = filtered_interactions.drop('category_rank', axis=1)
        # If preserve_input_order=True, no additional sorting is applied
        
        if vertical_layout:
            filtered_interactions = filtered_interactions.iloc[::-1].reset_index(drop=True)

        heatmap_scores, valid_ligand_receptors = [], []
        for _, row in filtered_interactions.iterrows():
            ligand, receptor = row[col_ligand], row[col_receptor]
            ligand_score = specificity_df.loc[ligand, sender_cell_type]
            receptor_score = specificity_df.loc[receptor, receiver_cell_type]
            heatmap_scores.append([ligand_score, receptor_score])
            valid_ligand_receptors.append(row[col_ligand_receptor_pair])
        
        heatmap_data = pd.DataFrame(np.array(heatmap_scores).T, index=['Ligand', 'Receptor'], columns=valid_ligand_receptors)
        N = len(filtered_interactions)
        vmax_actual = heatmap_vmax if heatmap_vmax is not None else (global_vmax if shared_legend else heatmap_data.max().max())

        # --- Plotting Area Setup ---
        bar_colors = filtered_interactions[col_annotation].map(color_palette)
        
        if vertical_layout:
            # Create subgridspec for better control
            plot_gs = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=main_gs[0, idx], 
                                                       wspace=0, width_ratios=[4, 0.8, 5.2, 0, 0, 0, 0, 0, 0, 0])
            
            # Use the first three columns for our plots
            ax_labels = fig.add_subplot(plot_gs[0])
            ax_hm = fig.add_subplot(plot_gs[1])
            ax_bar = fig.add_subplot(plot_gs[2])
            
            # Column 1: Labels - right aligned to be close to heatmap
            ax_labels.set_ylim(-0.5, N - 0.5)
            ax_labels.set_xlim(0, 1)
            
            # Plot invisible points to establish y-positions
            ax_labels.scatter([0.95] * N, np.arange(N), alpha=0)
            
            # Add text labels aligned to the right
            for i, label in enumerate(valid_ligand_receptors):
                label_color = bar_colors.iloc[i] if color_labels_by_annotation else 'black'
                ax_labels.text(0.98, i, label, ha='right', va='center', fontsize=10, color=label_color)
            
            ax_labels.tick_params(axis='both', which='both', bottom=False, labelbottom=False, 
                                  left=False, labelleft=False, right=False, labelright=False,
                                  top=False, labeltop=False)
            for spine in ax_labels.spines.values():
                spine.set_visible(False)
            ax_labels.grid(False)  # Ensure no grid
            
            # Add rotated title on the left side - CHANGE: Update to "From X to Y" format
            title_text = f'From {sender_cell_type} to {receiver_cell_type}'
            ax_labels.text(-0.15, 0.5, title_text, 
                           transform=ax_labels.transAxes, rotation=90, va='center', ha='center', 
                           fontsize=14, weight='normal')

            # Column 2: Heatmap
            if different_cmaps:
                ligand_data = heatmap_data.iloc[0:1, :].values.reshape(1, -1)
                receptor_data = heatmap_data.iloc[1:2, :].values.reshape(1, -1)
                
                img_ligand = ax_hm.imshow(ligand_data.T, cmap=cmap_ligand, aspect='auto', 
                                          interpolation='nearest', extent=[-0.5, 0.5, N - 0.5, -0.5], 
                                          vmin=0, vmax=vmax_actual)
                img_receptor = ax_hm.imshow(receptor_data.T, cmap=cmap_receptor, aspect='auto', 
                                            interpolation='nearest', extent=[0.5, 1.5, N - 0.5, -0.5], 
                                            vmin=0, vmax=vmax_actual)
                img = img_ligand  # For compatibility
                if mappable_ligand is None: mappable_ligand = img_ligand
                if mappable_receptor is None: mappable_receptor = img_receptor
            else:
                img = ax_hm.imshow(heatmap_data.T, cmap=heatmap_cmap, aspect='auto', interpolation='nearest',
                                   extent=[-0.5, 1.5, N - 0.5, -0.5], vmin=0, vmax=vmax_actual)
            
            # FIX: Adjusted tick positions to prevent overlap of "Ligand" and "Receptor"
            ax_hm.set_xticks([-0.2, 1.0])
            ax_hm.set_xticklabels(['Ligand', 'Receptor'], rotation=45, ha='right', fontsize=10)
            ax_hm.xaxis.set_label_position('bottom')
            ax_hm.xaxis.tick_bottom()
            
            # Remove y-axis ticks and labels
            ax_hm.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
            ax_hm.tick_params(axis='x', length=2, pad=2)
            ax_hm.set_ylim(-0.5, N - 0.5)
            ax_hm.set_xlim(-0.5, 1.5)
            
            ax_hm.grid(False)
            
            for spine in ax_hm.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.5)
            
            # Column 3: Bar Plot
            ax_bar.barh(y=np.arange(N), width=filtered_interactions[col_interaction_score],
                        color=bar_colors, height=1.0, edgecolor='black', linewidth=0.5)
            ax_bar.set_xlim(0, y_max)
            ax_bar.set_ylim(-0.5, N - 0.5)
            ax_bar.set_xlabel('Interaction Score', fontsize=14)
            ax_bar.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
            ax_bar.tick_params(axis='x', rotation=0, labelsize=10)
            ax_bar.spines['left'].set_visible(False)
            ax_bar.spines['right'].set_visible(True)
            ax_bar.spines['top'].set_visible(True)
            ax_bar.spines['bottom'].set_visible(True)
            
            ax_bar.grid(False)
            for i, patch in enumerate(ax_bar.patches):
                if patch.get_width() > y_max:
                    y_pos = patch.get_y() + patch.get_height() / 2
                    ax_bar.text(y_max * 0.95, y_pos, '→', ha='right', va='center', fontsize=12, fontweight='bold', color='red')
        else:
            # Original Horizontal Layout - FIXED all spacing and alignment issues
            # CHANGE: Remove all spacing between barplot and heatmap
            plot_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[idx, 0], 
                                                       height_ratios=[8, heatmap_height_ratio], hspace=0)
            ax_bar = fig.add_subplot(plot_gs[0])
            ax_hm = fig.add_subplot(plot_gs[1])
            
            # Remove bottom spine of bar plot and top spine of heatmap to make them seamless
            # CHANGE: Keep the bottom spine visible for barplot border
            ax_hm.spines['top'].set_visible(False)
            
            ax_bar.bar(x=np.arange(N), height=filtered_interactions[col_interaction_score],
                       color=bar_colors, width=1.0, edgecolor='black', linewidth=0.5)
            ax_bar.set_ylim(0, y_max)
            ax_bar.set_xlim(-0.5, N - 0.5)
            ax_bar.set_ylabel('Interaction Score', fontsize=14)
            ax_bar.set_xlabel('')
            ax_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
            # CHANGE: Update title format to "From X to Y" only
            title_text = f'From {sender_cell_type} to {receiver_cell_type}'
            ax_bar.set_title(title_text, fontsize=18, pad=15)
            for patch in ax_bar.patches:
                if patch.get_height() > y_max:
                    x_pos = patch.get_x() + patch.get_width() / 2
                    ax_bar.text(x_pos, y_max * 0.95, '↑', ha='center', va='top', fontsize=12, fontweight='bold', color='red')
            
            # Heatmap with different colormaps if specified - FIXED row order, heights and positioning
            if different_cmaps:
                ligand_data = heatmap_data.iloc[0:1, :].values
                receptor_data = heatmap_data.iloc[1:2, :].values
                
                # FIXED: Correct row order (Ligand on top, Receptor on bottom) and equal heights
                img_ligand = ax_hm.imshow(ligand_data, cmap=cmap_ligand, aspect='auto', 
                                          interpolation='nearest', extent=[-0.5, N - 0.5, 2.0, 1.0], 
                                          vmin=0, vmax=vmax_actual)
                img_receptor = ax_hm.imshow(receptor_data, cmap=cmap_receptor, aspect='auto', 
                                            interpolation='nearest', extent=[-0.5, N - 0.5, 1.0, 0.0], 
                                            vmin=0, vmax=vmax_actual)
                img = img_ligand  # For compatibility
                if mappable_ligand is None: mappable_ligand = img_ligand
                if mappable_receptor is None: mappable_receptor = img_receptor
                
                # Set axis limits to match the extents
                ax_hm.set_ylim(0.0, 2.0)
            else:
                img = ax_hm.imshow(heatmap_data, cmap=heatmap_cmap, aspect='auto', interpolation='nearest',
                                   extent=[-0.5, N - 0.5, 2.0, 0.0], vmin=0, vmax=vmax_actual)
                ax_hm.set_ylim(0.0, 2.0)
            
            ax_hm.set_ylabel('')
            ax_hm.set_xlabel('')
            
            # FIXED: Position y-ticks at center of each row but remove labels (we'll add them manually)
            if different_cmaps:
                ax_hm.set_yticks([1.5, 0.5])  # Center of ligand row (top) and receptor row (bottom)
                ax_hm.set_yticklabels([])  # Remove default labels, we'll position them manually
            else:
                ax_hm.set_yticks([1.5, 0.5])  # Center of each row
                ax_hm.set_yticklabels([])  # Remove default labels, we'll position them manually
            
            # CHANGE: Manually position "Ligand" and "Receptor" labels with more downward spacing to avoid overlap
            # Move "Ligand" further down to avoid overlap with "0" tick label
            ax_hm.text(-0.7, 1.5 - 1.0, 'Ligand', rotation=45, va='center', ha='right', fontsize=12, 
                       color='black', clip_on=False)
            # Move "Receptor" even further down to create clear separation
            ax_hm.text(-0.7, 0.5 - 1.3, 'Receptor', rotation=45, va='center', ha='right', fontsize=12, 
                       color='black', clip_on=False)
            
            ax_hm.set_xticks(np.arange(N))
            
            # FIXED: Position labels with small gap below heatmap and shift right by half block width
            ax_hm.set_xticklabels([])  # Clear default labels first
            # CHANGE: Move labels down slightly from heatmap
            y_pos = -0.3
            for i, label in enumerate(valid_ligand_receptors):
                label_color = bar_colors.iloc[i] if color_labels_by_annotation else 'black'
                # Shift right by 0.5 (half block width) from tick centers
                ax_hm.text(i + 0.5, y_pos, label, ha='right', va='top', rotation=45, fontsize=10, 
                           color=label_color, clip_on=False)

            ax_hm.grid(False)
            
            # Only show necessary spines
            for spine_name, spine in ax_hm.spines.items():
                if spine_name in ['left', 'right', 'bottom']:
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(0.5)
                else:
                    spine.set_visible(False)
        
        current_handles = [mpatches.Patch(color=c, label=l) for l, c in color_palette.items() if l in filtered_interactions[col_annotation].unique()]
        current_labels = [h.get_label() for h in current_handles]
        if mappable is None: mappable = img
        if mappable_ligand is None and different_cmaps: 
            mappable_ligand = img_ligand if 'img_ligand' in locals() else None
        if mappable_receptor is None and different_cmaps: 
            mappable_receptor = img_receptor if 'img_receptor' in locals() else None
        if legend_handles is None: legend_handles, legend_labels = current_handles, current_labels
    
    # --- Legend and Colorbar Area ---
    if vertical_layout and not shared_legend:
        pass 
    elif shared_legend and (legend_handles or mappable):
        if vertical_layout:
            # CHANGE: Use left-right layout instead of top-bottom to prevent overlap
            # Left: heatmap colorbars, Right: annotation legend
            # CHANGE: Reduce right section ratio from 2 to 1.5 for better balance
            bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[1, :], 
                                                         width_ratios=[1, 1.5], wspace=0.3)
            
            # Left section: Heatmap colorbars (stacked vertically if needed)
            if mappable or mappable_ligand:
                cbar_ax_container = fig.add_subplot(bottom_gs[0])
                cbar_ax_container.axis('off')
                
                if different_cmaps and mappable_ligand and mappable_receptor:
                    # CHANGE: Further increase spacing between colorbars
                    cbar_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bottom_gs[0], hspace=1.2)
                    
                    # Calculate appropriate colorbar width based on number of cell type pairs
                    # Limit width to prevent them from becoming too wide with multiple pairs
                    max_cbar_width = min(0.8, max(0.4, 1.0 / n_pairs))  # Width decreases with more pairs
                    cbar_x_start = (1.0 - max_cbar_width) / 2  # Center the colorbar
                    
                    # Ligand colorbar (top)
                    cbar_ax_ligand_container = fig.add_subplot(cbar_gs[0])
                    cax_l = cbar_ax_ligand_container.inset_axes([cbar_x_start, 0.3, max_cbar_width, 0.4])
                    cbar_ligand = fig.colorbar(mappable_ligand, cax=cax_l, orientation='horizontal')
                    cbar_ligand.set_label('Ligand Specificity', size=9, labelpad=4)
                    vmax_to_use = global_vmax if global_vmax > 0 else vmax_actual
                    cbar_ligand.set_ticks([0, vmax_to_use])
                    cbar_ligand.set_ticklabels([f'0', f'{vmax_to_use:.1f}'])
                    cbar_ligand.ax.tick_params(labelsize=8)
                    cbar_ligand.ax.xaxis.set_label_position('top')
                    cbar_ax_ligand_container.axis('off')
                    
                    # Receptor colorbar (bottom)
                    cbar_ax_receptor_container = fig.add_subplot(cbar_gs[1])
                    cax_r = cbar_ax_receptor_container.inset_axes([cbar_x_start, 0.3, max_cbar_width, 0.4])
                    cbar_receptor = fig.colorbar(mappable_receptor, cax=cax_r, orientation='horizontal')
                    cbar_receptor.set_label('Receptor Specificity', size=9, labelpad=4)
                    cbar_receptor.set_ticks([0, vmax_to_use])
                    cbar_receptor.set_ticklabels([f'0', f'{vmax_to_use:.1f}'])
                    cbar_receptor.ax.tick_params(labelsize=8)
                    cbar_receptor.ax.xaxis.set_label_position('top')
                    cbar_ax_receptor_container.axis('off')
                else:
                    # Single colorbar (centered vertically) - limit width here too
                    max_cbar_width = min(0.8, max(0.4, 1.2 / n_pairs))  # Slightly wider for single colorbar
                    cbar_x_start = (1.0 - max_cbar_width) / 2
                    
                    cbar_ax_inner = cbar_ax_container.inset_axes([cbar_x_start, 0.4, max_cbar_width, 0.2])
                    cbar = fig.colorbar(mappable if mappable else mappable_ligand, cax=cbar_ax_inner, orientation='horizontal')
                    cbar.set_label('Specificity Score', size=10, labelpad=4)
                    vmax_to_use = global_vmax if global_vmax > 0 else vmax_actual
                    cbar.set_ticks([0, vmax_to_use])
                    cbar.set_ticklabels([f'0.00', f'{vmax_to_use:.2f}'])
                    cbar.ax.tick_params(labelsize=8)
                    cbar.ax.xaxis.set_label_position('top')
            
            # Right section: Annotation legend (gets more space)
            if all_annotations:
                legend_ax = fig.add_subplot(bottom_gs[1])
                legend_ax.axis('off')
                
                fresh_handles = [mpatches.Patch(color=color_palette[label], label=label) for label in all_annotations]
                n_annotations = len(all_annotations)
                total_width = fig_height_per_pair * n_pairs  # Total available width
                
                # Increase font sizes to match heatmap legend font sizes
                if n_annotations <= 3:
                    ncol = 1
                    font_size = 10
                    title_font_size = 11
                elif n_annotations <= 6:
                    ncol = 2
                    font_size = 9
                    title_font_size = 10
                elif n_annotations <= 12:
                    ncol = 3
                    font_size = 8
                    title_font_size = 9
                else:
                    ncol = 4
                    font_size = 7
                    title_font_size = 8
                
                # Position legend on the left side of the right section
                legend = legend_ax.legend(handles=fresh_handles, title='Annotation', 
                                          loc='center left', ncol=ncol, 
                                          frameon=False, fontsize=font_size, title_fontsize=title_font_size,
                                          columnspacing=0.6, handletextpad=0.3)
                # Fix title alignment
                legend.get_title().set_horizontalalignment('left')
        else:
            # FIXED: Horizontal layout colorbar positioning with adaptive layout to prevent overlap
            # CHANGE: Calculate space needed for legend based on number of annotations
            n_annotations = len(all_annotations)
            legend_needs_space = n_annotations > 6  # If many annotations, need more space
            
            if legend_needs_space:
                # Use 3 rows: legend, colorbars, extra space
                right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[0, 1], 
                                                           height_ratios=[1.5, 0.8, 0.2], hspace=0.4)
            else:
                # Standard 2 rows: legend, colorbars
                right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 1], 
                                                           height_ratios=[2, 1], hspace=0.6)
            
            if legend_handles:
                legend_ax = fig.add_subplot(right_gs[0])
                fresh_handles = [mpatches.Patch(color=color_palette[label], label=label) for label in all_annotations]
                
                # CHANGE: Adaptive legend layout based on number of annotations
                if n_annotations <= 4:
                    ncol = 1
                elif n_annotations <= 8:
                    ncol = 2  
                else:
                    ncol = 2  # Force 2 columns even for many items, use smaller font
                    
                font_size = 8 if n_annotations > 8 else 10  # Smaller font for many items
                
                legend_ax.legend(handles=fresh_handles, title='Annotation', loc='center left', 
                                frameon=False, ncol=ncol, fontsize=font_size, title_fontsize=font_size+1,
                                columnspacing=0.5, handletextpad=0.3)
                legend_ax.axis('off')
                
            if mappable or (mappable_ligand and mappable_receptor):
                colorbar_row = 1 if not legend_needs_space else 1  # Same row index for both cases
                
                if different_cmaps and mappable_ligand and mappable_receptor:
                    # FIXED: Stack colorbars vertically and put labels on top
                    cbar_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_gs[colorbar_row], hspace=1.5)
                    
                    # Ligand colorbar
                    cbar_ax_ligand = fig.add_subplot(cbar_gs[0])
                    cbar_ligand = fig.colorbar(mappable_ligand, cax=cbar_ax_ligand, orientation='horizontal')
                    # CHANGE: Put label on top
                    cbar_ligand.set_label('Ligand Specificity', size=10)
                    cbar_ligand.ax.xaxis.set_label_position('top')
                    cbar_ligand.set_ticks([0, global_vmax])
                    cbar_ligand.set_ticklabels([f'0', f'{global_vmax:.1f}'])
                    cbar_ligand.ax.tick_params(labelsize=8)
                    
                    # Receptor colorbar
                    cbar_ax_receptor = fig.add_subplot(cbar_gs[1])
                    cbar_receptor = fig.colorbar(mappable_receptor, cax=cbar_ax_receptor, orientation='horizontal')
                    # CHANGE: Put label on top
                    cbar_receptor.set_label('Receptor Specificity', size=10)
                    cbar_receptor.ax.xaxis.set_label_position('top')
                    cbar_receptor.set_ticks([0, global_vmax])
                    cbar_receptor.set_ticklabels([f'0', f'{global_vmax:.1f}'])
                    cbar_receptor.ax.tick_params(labelsize=8)
                else:
                    cbar_ax = fig.add_subplot(right_gs[colorbar_row])
                    # CHANGE: Use inset_axes to reduce colorbar height for better appearance
                    cbar_ax_inner = cbar_ax.inset_axes([0.2, 0.3, 0.6, 0.4])  # Reduced height from 1.0 to 0.4
                    cbar = fig.colorbar(mappable, cax=cbar_ax_inner, orientation='horizontal')
                    # CHANGE: Put label on top
                    cbar.set_label('Specificity Score', size=12, labelpad=5)
                    cbar.ax.xaxis.set_label_position('top')
                    cbar.set_ticks([0, global_vmax])
                    cbar.set_ticklabels([f'0.00', f'{global_vmax:.2f}'])
                    cbar.ax.tick_params(labelsize=10)
                    cbar_ax.axis('off')  # Hide the container frame

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    
    
