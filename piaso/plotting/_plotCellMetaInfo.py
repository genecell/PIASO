import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

def plotConfusionMatrix(
    data,
    groupby_query,
    groupby_reference, 
    normalize='query',
    figsize=(11.5, 10),
    cmap='Purples',
    annot=False,
    fmt='.2f',
    title=None,
    save_path=None,
    dpi=300,
    return_objects=False,
    show_group_color_bars=False,
    **kwargs
):
    """
    Plot a normalized and reordered confusion matrix from clustering results.
    
    This function creates a confusion matrix heatmap with SVD-based reordering for 
    better visualization of cluster relationships. The matrix can be normalized 
    in different ways and customized extensively.
    
    Parameters:
        data (pandas.DataFrame or AnnData): DataFrame or AnnData object containing the data.
                                          If AnnData, will use data.obs for the analysis.
        groupby_query (str): Column name for the query labels (typically predicted clusters).
        groupby_reference (str): Column name for the reference labels (typically true labels).
        normalize (str): How to normalize the confusion matrix. Options:
                        - 'query': normalize by query (row-wise) - default
                        - 'reference': normalize by reference (column-wise)  
                        - 'all': normalize by total count
                        - None: no normalization
        figsize (tuple): Figure size for the plot. Default is (11.5, 10).
        cmap (str): Colormap for the heatmap. Default is 'Purples'.
        annot (bool): Whether to show annotations in cells. Default is False.
        fmt (str): Format for annotations. Default is '.2f'.
        title (str): Custom title for the plot. If None, generates automatic title.
        save_path (str): Path to save the figure. If None, only displays.
        dpi (int): DPI for saved figure. Default is 300.
        return_objects (bool): If True, returns (confusion_matrix, fig, ax). Default is False.
        show_group_color_bars (bool): If True, shows colored bars next to ticks for categories 
                               that have colors defined in adata.uns (e.g., 'CellTypes_colors'). 
                               Default is False.
        **kwargs: Additional arguments passed to sns.heatmap()
    
    Returns:
        None (default) or tuple: If return_objects=True, returns 
        (reordered_confusion_matrix, fig, ax) for further customization
    
    Examples:
        Basic usage with AnnData object:
        >>> import scanpy as sc
        >>> import pandas as pd
        >>> # Load your data
        >>> adata = sc.read_h5ad('your_data.h5ad')
        >>> # Plot confusion matrix between cell types and Leiden clusters
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden')
        
        Using a pandas DataFrame:
        >>> df = pd.DataFrame({
        ...     'CellTypes': ['T_cell', 'B_cell', 'Monocyte', 'T_cell', 'B_cell'],
        ...     'Leiden': ['0', '1', '2', '0', '1']
        ... })
        >>> plotConfusionMatrix(df, groupby_query='CellTypes', groupby_reference='Leiden')
        
        Different normalization methods:
        >>> # Normalize by reference (column-wise)
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden', 
        ...                     normalize='reference')
        >>> 
        >>> # No normalization, show raw counts
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden', 
        ...                     normalize=None)
        >>> 
        >>> # Normalize by total count
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden', 
        ...                     normalize='all')
        
        Customization options:
        >>> # Custom colors and show detailed values
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden',
        ...                     cmap='viridis', annot=True)
        >>> 
        >>> # Custom figure size and save to file
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden',
        ...                     figsize=(15, 12), 
        ...                     save_path='confusion_matrix.png',
        ...                     title='Cell Types vs Leiden Clusters')
        
        Show detailed values in the plot:
        >>> # Display percentage values in each cell
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden',
        ...                     annot=True, fmt='.1%')
        >>> 
        >>> # Display raw counts (with no normalization)
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden',
        ...                     normalize=None, annot=True, fmt='d')
        
        Show color bars for categories:
        >>> # Display colored bars next to ticks (requires colors in adata.uns)
        >>> plotConfusionMatrix(adata, groupby_query='CellTypes', groupby_reference='Leiden',
        ...                     show_group_color_bars=True)
        >>> # This will look for 'CellTypes_colors' and 'Leiden_colors' in adata.uns
        
        Advanced usage - getting results for further analysis:
        >>> conf_matrix, fig, ax = plotConfusionMatrix(adata, 
        ...                                           groupby_query='CellTypes', 
        ...                                           groupby_reference='Leiden',
        ...                                           return_objects=True)
        >>> # Access the reordered confusion matrix
        >>> print(conf_matrix.head())
        >>> # Further customize the plot
        >>> ax.set_title('Custom Title', fontsize=16)
        >>> plt.show()
        
        Using with different data sources:
        >>> # From Seurat object converted to pandas
        >>> seurat_df = pd.read_csv('seurat_metadata.csv')
        >>> plotConfusionMatrix(seurat_df, groupby_query='CellTypes', groupby_reference='Leiden')
        >>> 
        >>> # From flow cytometry data
        >>> flow_df = pd.read_csv('flow_cytometry_results.csv')
        >>> plotConfusionMatrix(flow_df, groupby_query='CellTypes', groupby_reference='Leiden',
        ...                     normalize='reference', cmap='Reds', annot=True)
    """
    
    # Handle different data types
    if hasattr(data, 'obs'):  # AnnData object
        df = data.obs
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("Data must be a pandas DataFrame or AnnData object")
    
    # Validate required columns
    missing_cols = [col for col in [groupby_query, groupby_reference] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Remove rows with NaN values in the specified columns
    df_clean = df[[groupby_query, groupby_reference]].dropna()
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after removing NaN values")
    
    if len(df_clean) < len(df):
        warnings.warn(f"Removed {len(df) - len(df_clean)} rows with NaN values")
    
    # Create the confusion matrix
    conf_matrix = pd.crosstab(df_clean[groupby_query], df_clean[groupby_reference])
    
    # Apply normalization
    if normalize == 'query':
        conf_matrix_norm = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
        vmax = 1.0
        cbar_label = 'Percentage of query prediction'
        cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
        cbar_ticklabels = ['0%', '25%', '50%', '75%', '100%']
    elif normalize == 'reference':
        conf_matrix_norm = conf_matrix.div(conf_matrix.sum(axis=0), axis=1)
        vmax = 1.0
        cbar_label = 'Percentage of reference prediction'
        cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
        cbar_ticklabels = ['0%', '25%', '50%', '75%', '100%']
    elif normalize == 'all':
        conf_matrix_norm = conf_matrix / conf_matrix.sum().sum()
        vmax = conf_matrix_norm.max().max()
        cbar_label = 'Percentage of total'
        cbar_ticks = None
        cbar_ticklabels = None
    elif normalize is None:
        conf_matrix_norm = conf_matrix
        vmax = conf_matrix_norm.max().max()
        cbar_label = 'Count'
        cbar_ticks = None
        cbar_ticklabels = None
        fmt = 'd' if fmt == '.2f' else fmt  # Use integer format for counts
    else:
        raise ValueError("normalize must be 'query', 'reference', 'all', or None")
    
    # Handle case where matrix has only one row or column
    if conf_matrix_norm.shape[0] <= 1 or conf_matrix_norm.shape[1] <= 1:
        warnings.warn("Confusion matrix has only one row or column. Skipping SVD reordering.")
        row_order = np.arange(conf_matrix_norm.shape[0])
        col_order = np.arange(conf_matrix_norm.shape[1])
    else:
        # Singular Value Decomposition (SVD) to find the optimal ordering
        try:
            U, s, Vt = np.linalg.svd(conf_matrix_norm.fillna(0), full_matrices=False)
            row_scores = U[:, 0] * s[0]
            col_scores = Vt.T[:, 0] * s[0]
            # Sorting indices based on scores
            row_order = np.argsort(row_scores)
            col_order = np.argsort(col_scores)
        except np.linalg.LinAlgError:
            warnings.warn("SVD failed. Using original order.")
            row_order = np.arange(conf_matrix_norm.shape[0])
            col_order = np.arange(conf_matrix_norm.shape[1])
    
    # Reorder the confusion matrix
    conf_matrix_reordered = conf_matrix_norm.iloc[row_order, col_order]
    
    # Get color information if show_group_color_bars is True and data is AnnData
    query_colors = None
    reference_colors = None
    if show_group_color_bars and hasattr(data, 'uns'):
        query_color_key = f"{groupby_query}_colors"
        reference_color_key = f"{groupby_reference}_colors"
        
        if query_color_key in data.uns:
            # Get unique categories in original order, then reorder
            query_cats = conf_matrix_norm.index.tolist()
            if len(data.uns[query_color_key]) >= len(query_cats):
                # Create mapping from category to color
                original_query_cats = data.obs[groupby_query].cat.categories.tolist() if hasattr(data.obs[groupby_query], 'cat') else sorted(data.obs[groupby_query].unique())
                query_color_map = {cat: data.uns[query_color_key][i] for i, cat in enumerate(original_query_cats) if i < len(data.uns[query_color_key])}
                # Get colors for reordered categories
                query_colors = [query_color_map.get(cat, '#808080') for cat in query_cats]
                query_colors = [query_colors[i] for i in row_order]
        
        if reference_color_key in data.uns:
            # Get unique categories in original order, then reorder
            reference_cats = conf_matrix_norm.columns.tolist()
            if len(data.uns[reference_color_key]) >= len(reference_cats):
                # Create mapping from category to color
                original_ref_cats = data.obs[groupby_reference].cat.categories.tolist() if hasattr(data.obs[groupby_reference], 'cat') else sorted(data.obs[groupby_reference].unique())
                reference_color_map = {cat: data.uns[reference_color_key][i] for i, cat in enumerate(original_ref_cats) if i < len(data.uns[reference_color_key])}
                # Get colors for reordered categories
                reference_colors = [reference_color_map.get(cat, '#808080') for cat in reference_cats]
                reference_colors = [reference_colors[i] for i in col_order]
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Adjust plot area if color bars are shown to make room for ticks outside color bars
    if show_group_color_bars:
        # Add padding: left for y-axis ticks, top for x-axis ticks
        ax.set_position([0.12, 0.1, 0.75, 0.75])
    
    # Prepare heatmap kwargs
    heatmap_kwargs = {
        'annot': annot,
        'cmap': cmap,
        'fmt': fmt,
        'norm': Normalize(vmin=0, vmax=vmax),
        'cbar_kws': {'ticks': cbar_ticks} if cbar_ticks else {},
        'linewidths': 0.5,  # Add borders between cells
        'linecolor': 'lightgrey',  # Light grey color for borders
        'ax': ax,
        **kwargs
    }
    
    sns.heatmap(conf_matrix_reordered, **heatmap_kwargs)
    
    # Add color bars if requested and colors are available
    if show_group_color_bars:
        # Add query color bar (left side) - closer to heatmap, before ticks
        if query_colors:
            # Create a thicker bar on the left, adjacent to the heatmap
            for i, color in enumerate(query_colors):
                rect = plt.Rectangle((-0.3, i), 0.3, 1, facecolor=color, 
                                   edgecolor='white', linewidth=0.5, clip_on=False)
                ax.add_patch(rect)
        
        # Add reference color bar (top) - positioned at the top, adjacent to heatmap
        if reference_colors:
            # Create a thicker bar on the top, ABOVE the heatmap
            # The top of the heatmap is at y = number of rows
            for i, color in enumerate(reference_colors):
                rect = plt.Rectangle((i, conf_matrix_reordered.shape[0]), 1, 0.3, 
                                   facecolor=color, edgecolor='white', linewidth=0.5, clip_on=False)
                ax.add_patch(rect)
    
    # Customize axes
    ax.xaxis.tick_top()
    ax.set_xlabel(f'Reference: {groupby_reference}')
    ax.set_ylabel(f'Query: {groupby_query}')
    ax.xaxis.set_label_position('top')
    
    # Adjust tick positions when color bars are shown (smaller padding)
    if show_group_color_bars:
        # Move ticks outward just enough to clear the color bars
        ax.tick_params(axis='y', pad=8)  # Small padding for y-axis ticks
        ax.tick_params(axis='x', pad=8)  # Small padding for x-axis ticks
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    if cbar_ticklabels:
        cbar.set_ticklabels(cbar_ticklabels)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    
    # Set title
    if title is None:
        normalize_str = normalize if normalize else 'unnormalized'
        title = f'Confusion Matrix ({normalize_str.title()} Normalization)'
    
    # Adjust title position if color bars are shown
    if show_group_color_bars:
        ax.set_title(title, pad=45)  # More padding when color bars are shown
    else:
        ax.set_title(title, pad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    if return_objects:
        return conf_matrix_reordered, fig, ax