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
    order_query=None,
    order_reference=None,
    normalize='query',
    figsize=None,
    square=True,
    cmap='Purples',
    annot=False,
    fmt='.2f',
    title=None,
    save_path=None,
    dpi=300,
    return_objects=False,
    show_group_color_bars=False,
    show=None,
    **kwargs
):
    """
    Plot a normalized and reordered confusion matrix from clustering results.
    
    This function creates a confusion matrix heatmap with automatic reordering
    for better visualization of cluster relationships: a clustered order shared
    by both axes when they carry the same labels, and an SVD order otherwise. The matrix can be normalized 
    in different ways and customized extensively.
    
    Parameters:
        data (pandas.DataFrame or AnnData): DataFrame or AnnData object containing the data.
                                          If AnnData, will use data.obs for the analysis.
        groupby_query (str): Column name for the query labels (typically predicted clusters).
        groupby_reference (str): Column name for the reference labels (typically true labels).
        order_query, order_reference: how to order each axis. An explicit
            sequence of labels, None (keep the categorical order), or a
            strategy name: 'auto' (the default — one clustered order on both
            axes when the two label sets match, otherwise SVD on both, or
            barycentre for the free axis when the other is pinned), 'svd',
            'hclust' (average linkage on that axis's own profiles), or
            'hclust_symmetric' (one clustered order on both axes; requires
            matching label sets).
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
            Saved with TrueType fonts.
        dpi (int): DPI for saved figure. Default is 300.
        show (bool): Call ``plt.show()`` (default). When not shown the figure
            is returned (without ``return_objects``), and closed if saved.
        return_objects (bool): If True, returns (confusion_matrix, fig, ax). Default is False.
        show_group_color_bars (bool): If True, shows colored bars next to ticks for categories 
                               that have colors defined in adata.uns (e.g., 'CellTypes_colors'). 
                               Default is False.
        figsize (tuple, optional): Figure size. None (default) derives it from
            the matrix shape so the cells come out square, which is what makes
            a confusion matrix readable: an off-diagonal block is judged by
            area, and a fixed 11.5x10 stretched a 30x6 matrix into cells five
            times wider than tall.
        square (bool): Force square cells (default True). Set False to let the
            heatmap fill whatever figsize is given.
        **kwargs: Additional arguments passed to sns.heatmap()
    
    Returns:
        None (default) or tuple: If return_objects=True, returns 
        (reordered_confusion_matrix, fig, ax) for further customization
    
    Examples:
        Basic usage with AnnData object:
        >>> import anndata
        >>> import pandas as pd
        >>> # Load your data
        >>> adata = anndata.read_h5ad('your_data.h5ad')
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
    from ..utils._cytome_compat import is_cytome_input
    _uns = None  # for color bar support
    if is_cytome_input(data):
        # A store, as a path or a Dataset: read through cytome's own API,
        # opened read-only (this was raw SQL on a read-write connection)
        from ._store_io import read_cells_columns
        df = read_cells_columns(data, [groupby_query, groupby_reference])
    elif hasattr(data, 'obs'):  # AnnData object
        df = data.obs
        _uns = getattr(data, 'uns', None)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(
            "Data must be a pandas DataFrame, AnnData object, cytome Dataset, "
            "or path to a .cytome file"
        )

    # Validate required columns
    missing_cols = [col for col in [groupby_query, groupby_reference] if col not in df.columns]
    if missing_cols:
        available = list(df.columns)
        raise ValueError(
            f"Column(s) {missing_cols} not found in data. "
            f"Available columns: {available}"
        )
    
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
    
    # Both axes SVD-ordered was the only behaviour; `order_query` and
    # `order_reference` now also take an explicit sequence, or "auto" to
    # follow whichever axis the other one pins. Pinning one and asking SVD for
    # the other is not a thing SVD can do -- it orders the two together -- so
    # "auto" against a fixed axis means the barycentre, which is the
    # conditional answer to that question.
    from ._seriation import resolve_pair_order

    _q_none = order_query is None
    _r_none = order_reference is None
    rows, cols = resolve_pair_order(
        conf_matrix_norm.fillna(0),
        left_order="auto" if _q_none else order_query,
        right_order="auto" if _r_none else order_reference,
        default_left=list(conf_matrix_norm.index),
        default_right=list(conf_matrix_norm.columns),
    )
    row_order = np.array([conf_matrix_norm.index.get_loc(r) for r in rows])
    col_order = np.array([conf_matrix_norm.columns.get_loc(c) for c in cols])

    # Reorder the confusion matrix
    conf_matrix_reordered = conf_matrix_norm.iloc[row_order, col_order]
    
    # Group colours by name, from the resolver every plot uses: the stored
    # colours of each column (AnnData uns, a store's categories), else the
    # PIASO default. (Only AnnData uns was read, positionally.)
    query_colors = None
    reference_colors = None
    if show_group_color_bars:
        from ._categories import resolve_category_style
        _src = data if (hasattr(data, 'obs') or is_cytome_input(data)) else None
        q_style = resolve_category_style(_src, groupby_query, df[groupby_query],
                                         read_stored=_src is not None)
        r_style = resolve_category_style(_src, groupby_reference, df[groupby_reference],
                                         read_stored=_src is not None)
        query_colors = [q_style.color(c) for c in conf_matrix_reordered.index]
        reference_colors = [r_style.color(c) for c in conf_matrix_reordered.columns]

    # Plotting
    if figsize is None:
        # ~0.34 in per cell, plus room for the labels, clamped so a 3x3 is not
        # a postage stamp and a 100x100 is not unopenable.
        n_r, n_c = conf_matrix_reordered.shape
        cell = 0.34
        figsize = (float(np.clip(n_c * cell + 4.0, 5.0, 22.0)),
                   float(np.clip(n_r * cell + 2.5, 4.0, 22.0)))

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
        # Square cells make a tall matrix genuinely tall, and a colorbar that
        # tracks the full height then dominates the figure. Shrink it toward a
        # readable strip rather than a column.
        'cbar_kws': dict({'ticks': cbar_ticks} if cbar_ticks else {},
                         shrink=float(np.clip(6.0 / max(figsize[1], 1e-6), 0.25, 1.0))),
        'linewidths': 0.5,  # Add borders between cells
        'linecolor': 'lightgrey',  # Light grey color for borders
        'ax': ax,
        **kwargs
    }
    
    if square:
        heatmap_kwargs.setdefault('square', True)
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
        
        # Reference colour bar on top, beside the reference labels (the x
        # ticks are moved to the top below). seaborn's y axis runs downward,
        # so the top edge is y = 0: the bar spans y in [-0.3, 0]. It was at
        # y = n_rows, which is the BOTTOM edge, away from its labels.
        if reference_colors:
            for i, color in enumerate(reference_colors):
                rect = plt.Rectangle((i, -0.3), 1, 0.3,
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
    
    from . import _figure
    _figure.layout(fig, quiet=False)
    out = _figure.finish(fig, created=True, show=_figure.resolve_show(show),
                         save=save_path, writekey="confusion_matrix", dpi=dpi,
                         literal=True)

    if return_objects:
        return conf_matrix_reordered, fig, ax
    return out