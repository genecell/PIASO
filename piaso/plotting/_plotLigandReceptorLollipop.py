import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List


def _format_label(label: str, max_length: int, ligand_receptor_sep: str = '-->') -> str:
    """Format labels for better readability"""
    if len(label) <= max_length:
        return label
    
    # Try to split on common separators and truncate intelligently
    if ligand_receptor_sep in label:
        parts = label.split(ligand_receptor_sep)
        if len(parts) == 2:
            ligand, receptor = parts
            if len(ligand) > max_length // 2:
                ligand = ligand[:max_length//2-1] + "…"
            if len(receptor) > max_length // 2:
                receptor = receptor[:max_length//2-1] + "…"
            return f"{ligand}{ligand_receptor_sep}{receptor}"
    
    # Simple truncation as fallback
    return label[:max_length-1] + "…"


def _calculate_circle_size(logfc_value: float, base_size: float, dramatic_level: float, logfc_range: float) -> float:
    """Calculate circle size using consistent exponential scaling"""
    # Normalize the logFC value
    normalized_val = logfc_value / logfc_range
    
    # Use exponential scaling with dramatic_level parameter
    size_multiplier = np.exp(normalized_val * dramatic_level)
    
    # Calculate min and max multipliers for normalization
    min_multiplier = np.exp(-1.0 * dramatic_level)
    max_multiplier = np.exp(1.0 * dramatic_level)
    
    # Map to target range
    target_min = 0.1
    target_max = 5.0 + (dramatic_level * 3.0)
    
    # Normalize to target range
    size_multiplier = target_min + (size_multiplier - min_multiplier) * (target_max - target_min) / (max_multiplier - min_multiplier)
    size_multiplier = np.clip(size_multiplier, target_min, target_max)
    
    return base_size * size_multiplier


def _validate_inputs(interactions_df: pd.DataFrame, cell_type_pairs: List[str], 
                    col_interaction_score: str, col_ligand_specificity: str, 
                    col_receptor_specificity: str, col_ligand_receptor_pair: str, 
                    col_cell_type_pair: str, col_annotation: str, col_circle_size: str,
                    cell_type_sep: str, top_n: int, fig_width: int, 
                    fig_height_per_pair: int, category_agg_method: str, 
                    specificity_df: pd.DataFrame = None) -> None:
    """Validate all input parameters"""
    if not cell_type_pairs or not isinstance(cell_type_pairs, list):
        raise ValueError("cell_type_pairs must be a non-empty list")
    
    if interactions_df.empty:
        raise ValueError("interactions_df is empty")
    
    # Check required columns - specificity columns are optional if specificity_df is provided
    required_cols = [col_interaction_score, col_ligand_receptor_pair, col_cell_type_pair, col_annotation]
    
    # Only require specificity columns if specificity_df is not provided
    if specificity_df is None:
        required_cols.extend([col_ligand_specificity, col_receptor_specificity])
    
    # Add circle size column to required columns if it exists
    if col_circle_size in interactions_df.columns:
        required_cols.append(col_circle_size)
    else:
        print(f"Warning: Column '{col_circle_size}' not found. Using default circle size.")
    
    missing_cols = [col for col in required_cols if col not in interactions_df.columns]
    if missing_cols:
        # Create detailed error message explaining what each column is used for
        col_explanations = {
            col_interaction_score: "interaction strength/score values",
            col_ligand_specificity: "ligand specificity scores (not required if specificity_df is provided)",
            col_receptor_specificity: "receptor specificity scores (not required if specificity_df is provided)", 
            col_ligand_receptor_pair: "ligand-receptor pair identifiers (e.g., 'GENE1-->GENE2')",
            col_cell_type_pair: "cell type pair identifiers (e.g., 'CellTypeA@CellTypeB')",
            col_annotation: "pathway/category annotations for coloring interactions",
            col_circle_size: "values for controlling circle sizes (optional)"
        }
        
        error_details = []
        for col in missing_cols:
            explanation = col_explanations.get(col, "required data")
            error_details.append(f"'{col}' (used for: {explanation})")
        
        raise ValueError(f"Missing required columns in interactions_df:\n" + "\n".join(error_details))
    
    # Validate specificity_df if provided
    if specificity_df is not None:
        if specificity_df.empty:
            raise ValueError("specificity_df is empty")
        if specificity_df.index.empty:
            raise ValueError("specificity_df must have gene names as index")
        if specificity_df.columns.empty:
            raise ValueError("specificity_df must have cell types as columns")
        print(f"Using external specificity dataframe with {len(specificity_df.index)} genes and {len(specificity_df.columns)} cell types.")
    
    # Check if cell type pairs exist in data
    available_pairs = set(interactions_df[col_cell_type_pair].unique())
    missing_pairs = [pair for pair in cell_type_pairs if pair not in available_pairs]
    if missing_pairs:
        raise KeyError(f"Cell type pairs not found in data: {missing_pairs}")
    
    # Validate separators
    for pair in cell_type_pairs:
        if cell_type_sep not in pair:
            raise ValueError(f"Cell type pair '{pair}' does not contain separator '{cell_type_sep}'")
    
    # Check numeric parameters
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if fig_width <= 0 or fig_height_per_pair <= 0:
        raise ValueError("Figure dimensions must be positive")
    if category_agg_method not in ['sum', 'mean']:
        raise ValueError("category_agg_method must be either 'sum' or 'mean'")
    
    # Validate that cell types in cell_type_pairs exist in specificity_df if provided
    if specificity_df is not None:
        all_cell_types = set()
        for pair in cell_type_pairs:
            sender, receiver = pair.split(cell_type_sep)
            all_cell_types.add(sender)
            all_cell_types.add(receiver)
        
        missing_cell_types = all_cell_types - set(specificity_df.columns)
        if missing_cell_types:
            print(f"Warning: The following cell types from cell_type_pairs are not found in specificity_df columns: {missing_cell_types}")
            print("Specificity scores for these cell types will be set to 0.")


def _setup_figure_and_axes(n_pairs: int, vertical_layout: bool, 
                          fig_width: int, fig_height_per_pair: int):
    """Setup figure and axes based on layout"""
    if vertical_layout:
        # For vertical layout: arrange multiple pairs horizontally (side by side)
        fig, axes = plt.subplots(1, n_pairs, figsize=(fig_width * n_pairs, fig_height_per_pair), squeeze=False)
        axes = axes.flatten()
    else:
        # For horizontal layout: arrange multiple pairs vertically (stacked)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(fig_width, fig_height_per_pair * n_pairs), squeeze=False)
        axes = axes.flatten()
    
    return fig, axes


def _process_dataframe_for_plotting(df_pair: pd.DataFrame, top_n: int, 
                                   preserve_input_order: bool, sort_by_category: bool,
                                   col_annotation: str, col_interaction_score: str,
                                   category_agg_method: str) -> pd.DataFrame:
    """Process and sort dataframe for plotting"""
    if preserve_input_order:
        df_plot = df_pair.head(top_n).reset_index(drop=True)
    elif sort_by_category:
        # First extract top_n interactions by score
        df_top = df_pair.nlargest(top_n, col_interaction_score).reset_index(drop=True)
        
        # Then sort these top interactions by category
        if category_agg_method == 'sum':
            category_scores = df_top.groupby(col_annotation)[col_interaction_score].sum().sort_values(ascending=False)
        else:  # mean
            category_scores = df_top.groupby(col_annotation)[col_interaction_score].mean().sort_values(ascending=False)
        
        category_order = category_scores.index
        df_top[col_annotation] = pd.Categorical(df_top[col_annotation], categories=category_order, ordered=True)
        df_plot = df_top.sort_values(by=[col_annotation, col_interaction_score], ascending=[True, False]).reset_index(drop=True)
    else:
        df_plot = df_pair.nlargest(top_n, col_interaction_score).reset_index(drop=True)
    
    return df_plot


def _extract_specificity_from_external_df(df_plot: pd.DataFrame, specificity_df: pd.DataFrame,
                                         col_ligand_receptor_pair: str, ligand_receptor_sep: str,
                                         sender: str, receiver: str) -> pd.DataFrame:
    """Extract specificity scores from external specificity dataframe"""
    ligand_specificity_scores = []
    receptor_specificity_scores = []
    
    for lr_pair in df_plot[col_ligand_receptor_pair]:
        try:
            ligand, receptor = lr_pair.split(ligand_receptor_sep)
        except ValueError:
            print(f"Warning: Could not split ligand-receptor pair '{lr_pair}' using separator '{ligand_receptor_sep}'. Using 0 specificity.")
            ligand_specificity_scores.append(0.0)
            receptor_specificity_scores.append(0.0)
            continue
        
        # Extract ligand specificity for sender cell type
        if ligand in specificity_df.index and sender in specificity_df.columns:
            ligand_spec = specificity_df.loc[ligand, sender]
        else:
            if ligand not in specificity_df.index:
                print(f"Warning: Ligand '{ligand}' not found in specificity_df index. Using 0 specificity.")
            if sender not in specificity_df.columns:
                print(f"Warning: Sender cell type '{sender}' not found in specificity_df columns. Using 0 specificity.")
            ligand_spec = 0.0
        
        # Extract receptor specificity for receiver cell type
        if receptor in specificity_df.index and receiver in specificity_df.columns:
            receptor_spec = specificity_df.loc[receptor, receiver]
        else:
            if receptor not in specificity_df.index:
                print(f"Warning: Receptor '{receptor}' not found in specificity_df index. Using 0 specificity.")
            if receiver not in specificity_df.columns:
                print(f"Warning: Receiver cell type '{receiver}' not found in specificity_df columns. Using 0 specificity.")
            receptor_spec = 0.0
        
        ligand_specificity_scores.append(ligand_spec)
        receptor_specificity_scores.append(receptor_spec)
    
    # Add the extracted specificity scores to the dataframe
    df_plot = df_plot.copy()
    df_plot['extracted_ligand_specificity'] = ligand_specificity_scores
    df_plot['extracted_receptor_specificity'] = receptor_specificity_scores
    
    return df_plot


def _calculate_y_values(df_plot: pd.DataFrame, col_ligand_specificity: str, 
                       col_receptor_specificity: str, col_interaction_score: str,
                       vertical_layout: bool, use_extracted_specificity: bool = False) -> pd.DataFrame:
    """Calculate y-values for lollipop plot"""
    # Use extracted specificity scores if available, otherwise use original columns
    if use_extracted_specificity:
        ligand_spec_col = 'extracted_ligand_specificity'
        receptor_spec_col = 'extracted_receptor_specificity'
    else:
        ligand_spec_col = col_ligand_specificity
        receptor_spec_col = col_receptor_specificity
    
    total_specificity = df_plot[ligand_spec_col] + df_plot[receptor_spec_col]
    total_specificity[total_specificity == 0] = 1  # Avoid division by zero
    
    # For vertical layout: ligand on LEFT (negative), receptor on RIGHT (positive)
    # For horizontal layout: ligand on TOP (positive), receptor on BOTTOM (negative)
    if vertical_layout:
        df_plot['ligand_y_original'] = -(df_plot[ligand_spec_col] / total_specificity) * df_plot[col_interaction_score]
        df_plot['receptor_y_original'] = (df_plot[receptor_spec_col] / total_specificity) * df_plot[col_interaction_score]
        df_plot['ligand_y'] = df_plot['ligand_y_original'].copy()
        df_plot['receptor_y'] = df_plot['receptor_y_original'].copy()
    else:
        df_plot['ligand_y_original'] = (df_plot[ligand_spec_col] / total_specificity) * df_plot[col_interaction_score]
        df_plot['receptor_y_original'] = -(df_plot[receptor_spec_col] / total_specificity) * df_plot[col_interaction_score]
        df_plot['ligand_y'] = df_plot['ligand_y_original'].copy()
        df_plot['receptor_y'] = df_plot['receptor_y_original'].copy()
    
    # Track which values are clipped for marking
    df_plot['ligand_clipped'] = False
    df_plot['receptor_clipped'] = False
    
    return df_plot


def _apply_score_range_limits(df_plot: pd.DataFrame, score_range_min: float, 
                             score_range_max: float, vertical_layout: bool) -> pd.DataFrame:
    """Apply score range limits and mark clipped values"""
    if score_range_min is not None:
        if vertical_layout:
            # For vertical: ligand values are negative, need to clip at -score_range_max
            ligand_mask = df_plot['ligand_y'] < (-score_range_max if score_range_max else df_plot['ligand_y'].min())
            df_plot.loc[ligand_mask, 'ligand_clipped'] = True
            df_plot['ligand_y'] = df_plot['ligand_y'].clip(lower=-score_range_max if score_range_max else None)
            
            receptor_mask = df_plot['receptor_y'] < score_range_min
            df_plot.loc[receptor_mask, 'receptor_clipped'] = True
            df_plot['receptor_y'] = df_plot['receptor_y'].clip(lower=score_range_min)
        else:
            # For horizontal: receptor values are negative
            ligand_mask = df_plot['ligand_y'] < score_range_min
            df_plot.loc[ligand_mask, 'ligand_clipped'] = True
            df_plot['ligand_y'] = df_plot['ligand_y'].clip(lower=score_range_min)
            
            receptor_mask = df_plot['receptor_y'] < (-score_range_max if score_range_max else df_plot['receptor_y'].min())
            df_plot.loc[receptor_mask, 'receptor_clipped'] = True
            df_plot['receptor_y'] = df_plot['receptor_y'].clip(lower=-score_range_max if score_range_max else None)
    
    if score_range_max is not None:
        if vertical_layout:
            # For vertical: ligand values are negative, receptor values are positive
            ligand_mask = df_plot['ligand_y'] > (-score_range_min if score_range_min else 0)
            df_plot.loc[ligand_mask, 'ligand_clipped'] = True
            df_plot['ligand_y'] = df_plot['ligand_y'].clip(upper=-score_range_min if score_range_min else None)
            
            receptor_mask = df_plot['receptor_y'] > score_range_max
            df_plot.loc[receptor_mask, 'receptor_clipped'] = True
            df_plot['receptor_y'] = df_plot['receptor_y'].clip(upper=score_range_max)
        else:
            # For horizontal: ligand values are positive, receptor values are negative
            ligand_mask = df_plot['ligand_y'] > score_range_max
            df_plot.loc[ligand_mask, 'ligand_clipped'] = True
            df_plot['ligand_y'] = df_plot['ligand_y'].clip(upper=score_range_max)
            
            receptor_mask = df_plot['receptor_y'] > (-score_range_min if score_range_min else 0)
            df_plot.loc[receptor_mask, 'receptor_clipped'] = True
            df_plot['receptor_y'] = df_plot['receptor_y'].clip(upper=-score_range_min if score_range_min else None)
    
    return df_plot


def _prepare_labels(df_plot: pd.DataFrame, col_ligand_receptor_pair: str, 
                   max_label_length: int, ligand_receptor_sep: str,
                   vertical_layout: bool) -> List[str]:
    """Prepare formatted labels for display"""
    formatted_labels = []
    seen_labels = {}
    for label in df_plot[col_ligand_receptor_pair]:
        formatted_label = _format_label(label, max_label_length, ligand_receptor_sep)
        # If we've seen this label before, add a small indicator to make it unique
        if formatted_label in seen_labels:
            seen_labels[formatted_label] += 1
            formatted_label = f"{formatted_label} ({seen_labels[formatted_label]})"
        else:
            seen_labels[formatted_label] = 1
        formatted_labels.append(formatted_label)
    
    # Prepare labels for display based on layout
    if vertical_layout:
        # For vertical layout, we need to reverse the labels since we reversed the dataframe
        formatted_labels_display = formatted_labels[::-1]
    else:
        # For horizontal layout, use labels as is
        formatted_labels_display = formatted_labels
    
    return formatted_labels_display


def _plot_vertical_layout(ax, df_plot: pd.DataFrame, circle_sizes: np.ndarray, 
                         color_map: dict, col_annotation: str, background_colors: bool,
                         show_grid: bool, score_range_min: float, score_range_max: float,
                         range_padding: float, formatted_labels_display: List[str],
                         color_labels_by_annotation: bool) -> None:
    """Handle plotting for vertical layout"""
    
    for j, row in df_plot.iterrows():
        color = color_map.get(row[col_annotation], 'grey')
        size = circle_sizes[j]
        
        ax.hlines(y=j, xmin=row['ligand_y'], xmax=row['receptor_y'], color=color, alpha=0.7, linewidth=2)
        ax.scatter(row['ligand_y'], j, color=color, s=size)
        ax.scatter(row['receptor_y'], j, color=color, s=size)
        
        # Add red dot in center if the value was clipped
        if row['ligand_clipped']:
            ax.scatter(row['ligand_y'], j, color='red', s=size/10, zorder=10)
        if row['receptor_clipped']:
            ax.scatter(row['receptor_y'], j, color='red', s=size/10, zorder=10)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    # Set y-axis labels and ticks only once
    ax.set_yticks(df_plot.index)
    ax.set_yticklabels(formatted_labels_display)
    
    # Set y-axis limits to reduce white space
    ax.set_ylim(-0.5, len(df_plot) - 0.5)
    
    # Set x-axis range and ticks
    if score_range_min is not None and score_range_max is not None:
        # Use user-specified range
        max_abs = score_range_max
        
        # Add padding to accommodate circles at the edges
        # Use the range_padding parameter to control padding amount
        padding = max_abs * range_padding
        
        ax.set_xlim(-max_abs - padding, max_abs + padding)
        
        # But keep tick labels at the specified range
        tick_values = np.linspace(-max_abs, max_abs, 5)
        tick_labels = [f'{abs(val):.1f}' for val in tick_values]
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)
    else:
        # Auto-scale based on data
        xlim = ax.get_xlim()
        max_abs = max(abs(xlim[0]), abs(xlim[1]))
        ax.set_xlim(-max_abs, max_abs)
        
        # Create symmetric ticks with absolute values
        tick_values = np.linspace(-max_abs, max_abs, 5)
        tick_labels = [f'{abs(val):.1f}' for val in tick_values]
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)
    
    # Add background colors after setting xlim - fix overlap issue
    if background_colors:
        actual_xlim = ax.get_xlim()
        y_limits = ax.get_ylim()
        
        # Create non-overlapping background regions
        # Ligand side (left half) - from left edge to center line
        ax.axvspan(actual_xlim[0], 0, ymin=0, ymax=1, alpha=0.1, color='blue', zorder=0)
        # Receptor side (right half) - from center line to right edge
        ax.axvspan(0, actual_xlim[1], ymin=0, ymax=1, alpha=0.1, color='red', zorder=0)
    
    # Color y-axis labels if requested
    if color_labels_by_annotation:
        for tick, annotation in zip(ax.get_yticklabels(), df_plot[col_annotation].iloc[::-1]):
            tick.set_color(color_map.get(annotation, 'black'))
    
    # Add grid for better readability
    if show_grid:
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.grid(axis='y', linestyle=':', alpha=0.2)
    else:
        # Explicitly turn off grid when show_grid is False
        ax.grid(False)


def _plot_horizontal_layout(ax, df_plot: pd.DataFrame, circle_sizes: np.ndarray, 
                           color_map: dict, col_annotation: str, background_colors: bool,
                           show_grid: bool, score_range_min: float, score_range_max: float,
                           range_padding: float, formatted_labels_display: List[str],
                           color_labels_by_annotation: bool) -> None:
    """Handle plotting for horizontal layout"""
    for j, row in df_plot.iterrows():
        color = color_map.get(row[col_annotation], 'grey')
        size = circle_sizes[j]
        
        # For horizontal layout, swap x and y coordinates
        ax.vlines(x=j, ymin=row['receptor_y'], ymax=row['ligand_y'], color=color, alpha=0.7, linewidth=2)
        ax.scatter(j, row['ligand_y'], color=color, s=size)
        ax.scatter(j, row['receptor_y'], color=color, s=size)
        
        # Add red dot in center if the value was clipped
        if row['ligand_clipped']:
            ax.scatter(j, row['ligand_y'], color='red', s=size/10, zorder=10)
        if row['receptor_clipped']:
            ax.scatter(j, row['receptor_y'], color='red', s=size/10, zorder=10)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Set x-axis ticks at the correct positions (where the lollipops are)
    tick_positions = df_plot.index
    ax.set_xticks(tick_positions)
    
    # Set x-axis limits to reduce white space
    ax.set_xlim(-0.5, len(df_plot) - 0.5)
    
    # Set y-axis range and ticks FIRST before adding labels
    if score_range_min is not None and score_range_max is not None:
        # Use user-specified range
        max_abs = score_range_max
        
        # Add padding to accommodate circles at the edges
        padding = max_abs * range_padding
        
        ax.set_ylim(-max_abs - padding, max_abs + padding)
        
        # But keep tick labels at the specified range
        tick_values = np.linspace(-max_abs, max_abs, 5)
        tick_labels = [f'{abs(val):.1f}' for val in tick_values]
        ax.set_yticks(tick_values)
        ax.set_yticklabels(tick_labels)
    else:
        # Auto-scale based on data
        ylim = ax.get_ylim()
        max_abs = max(abs(ylim[0]), abs(ylim[1]))
        ax.set_ylim(-max_abs, max_abs)
        
        # Create symmetric ticks with absolute values
        tick_values = np.linspace(-max_abs, max_abs, 5)
        tick_labels = [f'{abs(val):.1f}' for val in tick_values]
        ax.set_yticks(tick_values)
        ax.set_yticklabels(tick_labels)
    
    # Add background colors after setting ylim
    if background_colors:
        actual_ylim = ax.get_ylim()
        # Ligand side (top half)
        ax.axhspan(0, actual_ylim[1], alpha=0.1, color='blue', zorder=0)
        # Receptor side (bottom half)
        ax.axhspan(actual_ylim[0], 0, alpha=0.1, color='red', zorder=0)
    
    # NOW set the x-axis labels with proper positioning and colors
    # Clear default x-axis labels
    ax.set_xticklabels([])
    
    # Get the transform for data coordinates
    trans = ax.get_xaxis_transform()
    
    # Position labels at the bottom of the plot (y=0 in axes coordinates)
    for i, (pos, label) in enumerate(zip(tick_positions, formatted_labels_display)):
        # Get the color for this label if color_labels_by_annotation is True
        if color_labels_by_annotation:
            annotation = df_plot.iloc[i][col_annotation]
            label_color = color_map.get(annotation, 'black')
        else:
            label_color = 'black'
        
        # Use transform to place text at the correct position
        # y=0 in axes coordinates means bottom of axes
        ax.text(pos, 0, label, 
                transform=trans,
                rotation=45, 
                ha='right', 
                va='top',
                color=label_color)
    
    # Add grid for better readability
    if show_grid:
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', linestyle=':', alpha=0.2)
    else:
        # Explicitly turn off grid when show_grid is False
        ax.grid(False)


def _create_legends(fig, color_map: dict, filtered_df: pd.DataFrame, col_circle_size: str,
                   circle_size_title: str, base_circle_size: float, size_dramatic_level: float,
                   logfc_range: float, vertical_layout: bool) -> None:
    """Create and position legends"""
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    
    # Create circle size legend if the column exists - using SAME calculation function
    size_legend_elements = []
    if col_circle_size in filtered_df.columns:
        # Create more granular size legend with 5 different sizes
        logfc_values = [-logfc_range, -logfc_range/2, 0, logfc_range/2, logfc_range]
        logfc_labels = [f'{val:.1f}' for val in logfc_values]
        
        for val, label in zip(logfc_values, logfc_labels):
            # Use the EXACT SAME function as in the main plot
            actual_plot_size = _calculate_circle_size(val, base_circle_size, size_dramatic_level, logfc_range)
            
            # For scatter plots: 's' parameter is the marker area in points^2
            # For Line2D in legend: 'markersize' is the marker diameter in points
            # Mathematical relationship: area = π * (diameter/2)^2
            # Therefore: diameter = 2 * sqrt(area/π)
            
            # Direct conversion from scatter area to Line2D diameter
            marker_diameter = 2.0 * np.sqrt(actual_plot_size / np.pi)
            
            # NO correction factor - use the exact mathematical conversion
            # If circles appear different, we may need to adjust, but let's start with pure math
            marker_size = marker_diameter
            
            # Ensure reasonable bounds for display
            marker_size = np.clip(marker_size, 1, 50)
            
            size_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                                  markersize=marker_size, label=label, 
                                                  markeredgecolor='gray', markeredgewidth=0.5))
    
    if vertical_layout:
        # Position legends for vertical layout (multiple plots side by side)
        fig.legend(handles=legend_patches, title='Annotation', bbox_to_anchor=(1.01, 0.85), 
                  loc='upper left', frameon=False)
        if size_legend_elements:
            size_legend = fig.legend(handles=size_legend_elements, title=circle_size_title, 
                                   bbox_to_anchor=(1.01, 0.45), loc='upper left', frameon=False)
        plt.tight_layout(rect=[0, 0.03, 0.82, 0.97])
    else:
        # Position legends for horizontal layout (multiple plots stacked vertically)
        if size_legend_elements:
            size_legend = fig.legend(handles=size_legend_elements, title=circle_size_title, 
                                   bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=5, frameon=False)
            # Annotation legend below the circle size legend with more spacing
            fig.legend(handles=legend_patches, title='Annotation', bbox_to_anchor=(0.5, -0.15), 
                      loc='upper center', ncol=min(len(legend_patches), 4), frameon=False)
            plt.tight_layout(rect=[0, 0.20, 1, 0.98])
        else:
            # Only annotation legend if no circle size legend
            fig.legend(handles=legend_patches, title='Annotation', bbox_to_anchor=(0.5, -0.02), 
                      loc='upper center', ncol=min(len(legend_patches), 4), frameon=False)
            plt.tight_layout(rect=[0, 0.08, 1, 0.98])


def plotLigandReceptorLollipop(
    interactions_df: pd.DataFrame,
    cell_type_pairs: List[str],
    top_n: int = 25,
    sort_by_category: bool = False,
    preserve_input_order: bool = False,
    palette: str = 'tab20',
    vertical_layout: bool = True,
    fig_width: int = 10,
    fig_height_per_pair: int = 6,
    save_path: str = None,
    cell_type_sep: str = '@',
    ligand_receptor_sep: str = '-->',
    col_interaction_score: str = 'interaction_score',
    col_ligand_specificity: str = 'ligand_specificity',
    col_receptor_specificity: str = 'receptor_specificity',
    col_ligand_receptor_pair: str = 'ligandXreceptor',
    col_cell_type_pair: str = 'CellTypeXCellType',
    col_annotation: str = 'annotation',
    category_agg_method: str = 'sum',
    color_labels_by_annotation: bool = False,
    max_label_length: int = 25,
    background_colors: bool = False,
    col_circle_size: str = 'avg_log2FC',
    circle_size_title: str = 'avg_log2FC',
    base_circle_size: float = 15,
    circle_size_scale: float = 20,
    logfc_range: float = 1.0,
    size_dramatic_level: float = 1.5,
    show_grid: bool = True,
    score_range_min: float = None,
    score_range_max: float = None,
    range_padding: float = 0.15,
    specificity_df: pd.DataFrame = None
):
    """
    Generates advanced bidirectional lollipop plots for one or more cell-type interactions
    with support for both vertical and horizontal layouts.

    Args:
        interactions_df (pd.DataFrame): DataFrame with interaction and specificity data.
        cell_type_pairs (List[str]): A list of 'CellTypeXCellType' strings to plot.
        top_n (int): The number of top interactions to display per plot.
        sort_by_category (bool): If True, sort interactions by category first, then by score.
        preserve_input_order (bool): If True, use the original DataFrame order.
        palette (str): Seaborn color palette name for coloring categories.
        vertical_layout (bool): If True, subplots are arranged vertically. Otherwise, horizontally.
        fig_width (int): Base width for the figure (for vertical layout) or width per subplot (for horizontal).
        fig_height_per_pair (int): Height per subplot (for vertical layout) or base height (for horizontal).
        save_path (str, optional): Path to save the figure (e.g., 'plot.png').
        cell_type_sep (str): Separator for sender/receiver cell types (e.g., '@').
        ligand_receptor_sep (str): Separator for ligand/receptor pairs (e.g., '-->').
        col_interaction_score (str): Column name for interaction scores.
        col_ligand_specificity (str): Column name for ligand specificity scores.
        col_receptor_specificity (str): Column name for receptor specificity scores.
        col_ligand_receptor_pair (str): Column name for ligand-receptor pair strings.
        col_cell_type_pair (str): Column name for cell type pair strings.
        col_annotation (str): Column name for pathway/annotation data (default: 'annotation').
        category_agg_method (str): Method to aggregate interaction scores by category ('sum' or 'mean').
        color_labels_by_annotation (bool): If True, color ligand-receptor labels by their annotation.
        max_label_length (int): Maximum length for y-axis labels before truncation.
        background_colors (bool): If True, add transparent background colors to distinguish ligand/receptor sides.
        col_circle_size (str): Column name for controlling circle sizes (default: 'avg_log2FC').
        circle_size_title (str): Title for the circle size legend.
        base_circle_size (float): Base size for circles when col_circle_size value is 0.
        circle_size_scale (float): Scaling factor for circle sizes based on col_circle_size values.
        logfc_range (float): Expected range of logFC values for proper scaling (e.g., 1.0 for -1 to +1 range).
        size_dramatic_level (float): Controls how dramatic the size differences are (0.5=subtle, 1.0=moderate, 2.0=very dramatic).
        show_grid (bool): If True, show grid lines on the plot.
        score_range_min (float, optional): Minimum value for interaction score axis. Values below this will be clipped.
        score_range_max (float, optional): Maximum value for interaction score axis. Values above this will be clipped.
        range_padding (float): Padding to add beyond the score range to accommodate circles (default 0.15 = 15%).
        specificity_df (pd.DataFrame, optional): DataFrame with gene names as index and cell types as columns containing specificity scores. 
                                               If provided, will override col_ligand_specificity and col_receptor_specificity columns.
    
    Examples:
        # Vertical layout with default settings
        plotLigandReceptorLollipop(
            interactions_df=specific_interactions,
            cell_type_pairs=['L5 NP@SST-Chrna2'],
            vertical_layout=True
        )
        
        # Using external specificity dataframe
        plotLigandReceptorLollipop(
            interactions_df=specific_interactions,
            cell_type_pairs=['L5 NP@SST-Chrna2', 'L5 PT@SST-Chrna2'],
            specificity_df=gene_specificity_matrix,
            vertical_layout=False,
            save_path='lollipop_plot.png'
        )
    """
    
    # --- 1. Error Handling and Validation ---
    _validate_inputs(interactions_df, cell_type_pairs, col_interaction_score, 
                    col_ligand_specificity, col_receptor_specificity, 
                    col_ligand_receptor_pair, col_cell_type_pair, col_annotation, 
                    col_circle_size, cell_type_sep, top_n, fig_width, 
                    fig_height_per_pair, category_agg_method, specificity_df)
    
    # --- 2. Pre-computation and Setup ---
    n_pairs = len(cell_type_pairs)
    
    # Filter for relevant data and create a global color map for consistency
    filtered_df = interactions_df[interactions_df[col_cell_type_pair].isin(cell_type_pairs)].copy()
    if filtered_df.empty:
        raise ValueError("None of the specified cell_type_pairs were found in the data.")
        
    unique_categories = filtered_df[col_annotation].unique()
    colors = sns.color_palette(palette, len(unique_categories))
    color_map = dict(zip(unique_categories, colors))

    # --- 3. Figure and Axes Setup ---
    fig, axes = _setup_figure_and_axes(n_pairs, vertical_layout, fig_width, fig_height_per_pair)

    # --- 4. Loop and Plot for each Cell Type Pair ---
    for i, pair in enumerate(cell_type_pairs):
        ax = axes[i]
        
        # Turn off any default grid first
        ax.grid(False)
        
        df_pair = filtered_df[filtered_df[col_cell_type_pair] == pair].copy()
        
        # Extract sender and receiver cell types
        try:
            sender, receiver = pair.split(cell_type_sep)
        except ValueError:
            raise ValueError(f"Cannot split cell type pair '{pair}' using separator '{cell_type_sep}'")

        # Process dataframe for plotting
        df_plot = _process_dataframe_for_plotting(df_pair, top_n, preserve_input_order, 
                                                 sort_by_category, col_annotation, 
                                                 col_interaction_score, category_agg_method)

        if df_plot.empty:
            ax.text(0.5, 0.5, f'No data for\n{sender} to {receiver}', ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Extract specificity scores from external dataframe if provided
        use_extracted_specificity = False
        if specificity_df is not None:
            df_plot = _extract_specificity_from_external_df(df_plot, specificity_df, 
                                                           col_ligand_receptor_pair, 
                                                           ligand_receptor_sep, sender, receiver)
            use_extracted_specificity = True

        # Calculate y-values for lollipop plot
        df_plot = _calculate_y_values(df_plot, col_ligand_specificity, col_receptor_specificity, 
                                     col_interaction_score, vertical_layout, use_extracted_specificity)
        
        # Apply score range limits if specified
        df_plot = _apply_score_range_limits(df_plot, score_range_min, score_range_max, vertical_layout)
        
        # For horizontal layout, we want highest scores on the LEFT, so don't reverse
        # For vertical layout, we want highest scores on TOP, so reverse
        if vertical_layout:
            df_plot = df_plot.iloc[::-1].reset_index(drop=True)

        # Calculate circle sizes using the CONSISTENT function
        if col_circle_size in df_plot.columns:
            circle_sizes = [_calculate_circle_size(val, base_circle_size, size_dramatic_level, logfc_range) 
                           for val in df_plot[col_circle_size]]
            circle_sizes = np.array(circle_sizes)
        else:
            circle_sizes = np.array([base_circle_size] * len(df_plot))

        # Prepare labels for display
        formatted_labels_display = _prepare_labels(df_plot, col_ligand_receptor_pair, 
                                                  max_label_length, ligand_receptor_sep, 
                                                  vertical_layout)

        # Plotting logic based on layout
        if vertical_layout:
            _plot_vertical_layout(ax, df_plot, circle_sizes, color_map, col_annotation, 
                                 background_colors, show_grid, score_range_min, score_range_max, 
                                 range_padding, formatted_labels_display, color_labels_by_annotation)
        else:
            _plot_horizontal_layout(ax, df_plot, circle_sizes, color_map, col_annotation, 
                                   background_colors, show_grid, score_range_min, score_range_max, 
                                   range_padding, formatted_labels_display, color_labels_by_annotation)

        # Set title with improved formatting
        title = f'Interactions: {sender} → {receiver}'
        ax.set_title(title, fontsize=14, pad=10)

    # --- 5. Final Touches ---
    # Set appropriate labels based on layout
    if vertical_layout:
        fig.supxlabel('Interaction Score (Ligand ← | → Receptor)', y=0.05, fontsize=14)
    else:
        fig.supylabel('Interaction Score (← Receptor | Ligand →)', x=0.02, fontsize=16)
    
    # Create legends
    _create_legends(fig, color_map, filtered_df, col_circle_size, circle_size_title, 
                   base_circle_size, size_dramatic_level, logfc_range, vertical_layout)

    # --- 6. Save and Display ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Lollipop plot saved to {save_path}")

    plt.show()