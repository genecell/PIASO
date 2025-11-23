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

        
# Adapted from https://github.com/theislab/scanpy/issues/137
def _build_subplots(n,
                    ncol=None,
                    dpi=80,
                    col_size:int=5,
                    row_size:int=5,):
    """
    Build a grid of subplots.

    Parameters
    ----------
    n : int
        The total number of subplots.
    ncol : int or None, optional (default: None)
        If specified, defines the number of columns per row. If None, the number of columns is computed 
        as the ceiling of n divided by the integer square root of n.
    dpi : int, optional (default: 80)
        Dots per inch (DPI) setting for the figure.
    col_size : int, optional (default=5)
        Width (in inches) of each subplot column.
    row_size : int, optional (default=5)
        Height (in inches) of each subplot row.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    axs : ndarray of matplotlib.axes.Axes
        The array of Axes objects for the subplots.
    nrow : int
        The number of rows in the subplot grid.
    ncol : int
        The number of columns in the subplot grid.
    """
    if ncol is None:
        nrow = int(np.sqrt(n))
        ncol = int(np.ceil(n / nrow))
    else:
        nrow = int(np.ceil(n / ncol))
    
    # Assumes col_size and row_size are defined in the outer scope.
    fig, axs = plt.subplots(nrow, ncol, dpi=dpi, figsize=(ncol * col_size, nrow * row_size))
    return fig, axs, nrow, ncol
        
import matplotlib.pyplot as plt
import numpy as np

def _create_global_legend(
    fig,
    axes,
    legend_loc="center left", 
    bbox_to_anchor=(1.02, 0.5), 
    frameon:bool=False,  
    marker_size:float=1.0,  
    max_rows_per_col:int=12, 
    **legend_kwargs):
    """
    Collects unique legend entries from all subplots and creates a single global legend.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
    axes : array-like
        A list or array of Matplotlib Axes objects (e.g., from `plt.subplots`).
    legend_loc : str, optional (default="center left")
        The location of the global legend.
    bbox_to_anchor : tuple, optional (default=(1.02, 0.5))
        The positioning of the legend outside the main figure.
    frameon : bool, optional (default=False)
        Whether to display the legend box.
    marker_size : float, optional (default=1.0)
        Scaling factor for legend markers (dot size).
    max_rows_per_col : int, optional (default=12)
        Maximum number of rows before creating a new column in the legend.
    **legend_kwargs : dict, optional
        Additional keyword arguments passed to `fig.legend()` for styling.

    Returns
    -------
    None
        Displays the figure with a global legend.
    """
    legend_items = {}  # Dictionary to store unique legend items

    # # Collect legend handles and labels from all subplots
    # for ax in axes.flat:
    #     handles, labels = ax.get_legend_handles_labels()
    #     for handle, label in zip(handles, labels):
    #         legend_items[label] = handle  # Ensures unique labels
            
    # Collect legend handles and labels only from visible subplots with artists
    for ax in axes.flat:
        if ax.get_visible():  # Only process visible axes, this is very important, because some subplots will be empty
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Ensure there are artists with labels
                for handle, label in zip(handles, labels):
                    legend_items[label] = handle  # Add unique labels

            
    # Modify marker size in the legend
    legend_handles = list(legend_items.values())
    for handle in legend_handles:
        if hasattr(handle, "set_markersize"):  # Check if it's a Line2D object with markers
            handle.set_markersize(handle.get_markersize() * marker_size)
            
            
    # Determine number of columns dynamically
    num_items = len(legend_items)
    ncol = max(1, -(-num_items // max_rows_per_col))  # Equivalent to math.ceil(num_items / max_rows_per_col)


    # Create the global legend outside the figure if there are labels
    if legend_items:
        fig.legend(
            handles=legend_handles,
            labels=legend_items.keys(),
            loc=legend_loc,
            bbox_to_anchor=bbox_to_anchor,
            frameon=frameon,  # Remove or keep the legend box
            markerscale=marker_size,  # Adjust marker scale
            handlelength=2,  # Adjust spacing between legend symbols and text
            ncol=ncol,  # Automatically set number of columns
            **legend_kwargs  # Pass additional customization options
        )

    plt.tight_layout()  # Adjust layout

    
### Plot embeddings side by side
import matplotlib.pyplot as plt
       
def plot_embeddings_split(adata,
                          color,
                          splitby,
                          ncol:int=None,
                          dpi:int=80,
                          col_size:int=5,
                          row_size:int=5,
                          vmax:float=None,
                          vmin:float=None,
                          show_figure:bool=True,
                          save:bool=None,
                          layer:str=None,
                          basis:str='X_umap',
                          fix_coordinate_ratio:bool=True, 
                          show_axis_ticks:bool=False, 
                          margin_ratio:float=0.05, 
                          legend_fontsize:int=10,
                          legend_fontoutline:int=2,
                          legend_loc:str='right margin',
                          legend_marker_size: float=1.6,
                          x_min=None,
                          x_max=None,
                          y_min=None,
                          y_max=None,
                          **kwargs):
    """
    Plot cell embeddings side by side based on a categorical variable.

    The plots are split by a specified categorical variable, with each unique category producing a separate subplot.
    Data points in each subplot are colored according to the `color` variable.

    Parameters
    ----------
    adata : AnnData
        An AnnData object.
    color : str
        Used to specify a gene name to plot, or a key in `adata.obs` used to assign colors to the cells in the embedding plot.
    splitby : str
        Key in `adata.obs` used to split the dataset into multiple panels. Each unique value under this key
        will result in a separate subplot.
    ncol : int or None, optional (default: None)
        If specified, defines the number of columns per row. If None, the number of columns is computed 
        as the ceiling of n divided by the integer square root of n.
    dpi : int, optional (default: 80)
        Dots per inch (DPI) setting for the figure.
    col_size : int, optional (default=5)
        Width (in inches) of each subplot column.
    row_size : int, optional (default=5)
        Height (in inches) of each subplot row.
    vmax : float or None, optional (default=None)
        Maximum value for the color scale. If not provided, the upper limit is determined automatically.
    vmin : float or None, optional (default=None)
        Minimum value for the color scale. If not provided, the lower limit is determined automatically.
    show_figure : bool, optional (default=True)
        Whether to display the figure after plotting.
    save : str or None, optional (default=None)
        File path to save the resulting figure. If None, the figure will not be saved.
    layer : str or None, optional (default=None)
        If specified, the name of the layer in `adata.layers` from which to obtain the gene expression values.
    basis : str, optional (default='X_umap')
        Key in `adata.obsm` that contains the embedding coordinates (e.g., `X_umap` or `X_pca`).
    fix_coordinate_ratio : bool, optional (default=True)
        If True, the aspect ratio of each subplot is fixed so that the x- and y-axes are scaled equally.
    show_axis_ticks : bool, optional (default=False)
        Whether to display axis ticks and tick labels on the plots.
    margin_ratio : float, optional (default=0.05)
        Margin ratio for both the x-axis and y-axis limits, relative to the range of the data. This provides
        additional spacing around the plotted points.
    legend_fontsize: int, optional (default=9)
        Font size in pt.
    legend_fontoutline: int, optional (default=2)
        Line width of the legend font outline in pt. 
    legend_loc: str, optional (default='right margin')
        Location of legend, defaults to 'right margin'.
    legend_marker_size: float, optional (default=1.5)
        Scaling factor for legend markers (dot size).
    x_min : float or None, optional (default=None)
        Minimum limit for the x-axis. If None, the limit is computed automatically based on the data.
    x_max : float or None, optional (default=None)
        Maximum limit for the x-axis. If None, the limit is computed automatically based on the data.
    y_min : float or None, optional (default=None)
        Minimum limit for the y-axis. If None, the limit is computed automatically based on the data.
    y_max : float or None, optional (default=None)
        Maximum limit for the y-axis. If None, the limit is computed automatically based on the data.
    **kwargs : dict
        Additional keyword arguments passed to the `scanpy.pl.embedding` function.

    Returns
    -------
    None.

    Examples
    --------
    >>> import scanpy as sc
    >>> import piaso
    >>> adata = sc.datasets.pbmc3k()  # Load an example dataset
    >>> # Plot embeddings colored by a gene expression value and split by clusters
    >>> piaso.pl.plot_embeddings_split(adata, color='CDK9', splitby='louvain', col_size=6, row_size=6)
    >>> # Save the figure to a file
    >>> piaso.pl.plot_embeddings_split(adata, color='CDK9', splitby='louvain', save='./CST3_embeddingsSplit.pdf')
    """


    # --- Internal Helper: Adjust Colorbar Aspect Ratio ---
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ### Adapted from https://stackoverflow.com/questions/29516157/set-equal-aspect-in-plot-with-colorbar
    def adjustColorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    # --- 1. Setup Layout ---

    # Robust 'splitby' Variable Extraction & Safety Checks
    if splitby not in adata.obs.columns:
         raise ValueError(f"The splitby key '{splitby}' was not found in adata.obs.")

    # Check if explicitly categorical
    if isinstance(adata.obs[splitby].dtype, pd.CategoricalDtype):
        variables = adata.obs[splitby].cat.categories
    else:
        # --- GUARDRAILS FOR NON-CATEGORICAL COLUMNS ---
        col_data = adata.obs[splitby]
        
        # Reject Float (Continuous)
        if pd.api.types.is_float_dtype(col_data):
            raise ValueError(f"The column '{splitby}' is float (continuous). "
                             "Cannot split plots by a continuous variable.")
            
        # Check Integer Cardinality
        # If it is integer, we must check if it's actually a cluster label (few values) 
        # or a count (many values).
        if pd.api.types.is_integer_dtype(col_data):
            unique_vals = col_data.unique()
            n_unique = len(unique_vals)
            
            # Threshold: If > 50 subplots, assume it's a mistake unless explicitly cast to category
            if n_unique > 50:
                raise ValueError(f"The integer column '{splitby}' has {n_unique} unique values. "
                                 "Splitting by this variable would generate too many subplots (looks like continuous data). "
                                 f"If you really intend to create {n_unique} plots, please convert it explicitly:\n"
                                 f"adata.obs['{splitby}'] = adata.obs['{splitby}'].astype('category')")
            
            variables = np.sort(unique_vals)
            print(f"Note: '{splitby}' is integer type. Treating as {n_unique} discrete categories.")
            
        # Handle Strings/Objects
        else:
            variables = col_data.unique()
            variables = variables[~pd.isnull(variables)] # Remove NaNs
            try:
                variables = np.sort(variables)
            except:
                pass # Sort failed (mixed types), keep order
                
            if len(variables) > 50:
                 print(f"Warning: Variable '{splitby}' will generate {len(variables)} subplots. This may be slow.")
    
    fig, axs, nrow, ncol = _build_subplots(len(variables), ncol=ncol, dpi=dpi, col_size=col_size, row_size=row_size)
    
    if type(axs) != np.ndarray:
        axs = [axs]
    else:
        axs = axs.ravel()
    
    # --- 2. Pre-calculate Limits (Efficiency) ---
    emd_df = pd.DataFrame(adata.obsm[basis]) 
    
    if all(v is not None for v in [x_min, y_min, x_max, y_max]):
        xy_min = np.array([x_min, y_min])
        xy_max = np.array([x_max, y_max])
    else:     
        xy_min = emd_df.min(axis=0).values
        xy_max = emd_df.max(axis=0).values

    xy_margin = (xy_max - xy_min) * margin_ratio

    for ax in axs:
        ax.set_xlim(xy_min[0] - xy_margin[0], xy_max[0] + xy_margin[0])
        ax.set_ylim(xy_min[1] - xy_margin[1], xy_max[1] + xy_margin[1])

    # --- 3. Pre-fetch Data (Efficiency) ---
    is_obs = np.isin(color, adata.obs.columns)
    
    if is_obs:
        full_color_data = adata.obs[color].values
        is_numeric = isinstance(adata.obs[color].iloc[0], (np.floating, float, np.integer, int, np.int_))
    else:
        full_color_data = adata.obs_vector(color, layer=layer)
        is_numeric = True 

    expr_max, expr_min = None, None
    if is_numeric:
        expr_max = vmax if vmax is not None else np.nanmax(full_color_data)
        expr_min = vmin if vmin is not None else np.nanmin(full_color_data)

    # --- 4. Main Plotting Loop ---
    for i in range(len(axs)):
        if i < len(variables):
            category = variables[i]
            
            # Create mask and convert to numpy array immediately
            mask = (adata.obs[splitby] == category)
            mask_values = mask.values 
            
            if np.sum(mask_values) == 0: 
                axs[i].set_visible(False)
                continue

            # Lightweight Proxy AnnData
            adata_sub = sc.AnnData(obs=adata.obs.iloc[mask_values].copy())
            
            # Copy coordinates
            adata_sub.obsm[basis] = adata.obsm[basis][mask_values]
            
            # Inject color data
            adata_sub.obs[color] = full_color_data[mask_values]


            # If the color is categorical and a palette exists in the original object,
            # copy it to the proxy object's .uns.
            if not is_numeric:
                palette_key = f'{color}_colors'
                if palette_key in adata.uns:
                    adata_sub.uns[palette_key] = adata.uns[palette_key]

            plot_kwargs = dict(
                basis=basis,
                color=color, 
                title=f"{color} in\n{category}", 
                legend_fontsize=legend_fontsize,
                legend_fontoutline=legend_fontoutline,
                ncols=4,
                show=False,
                ax=axs[i],
                return_fig=False,
                colorbar_loc=None,
                **kwargs
            )

            if is_numeric:
                plot_kwargs['vmax'] = expr_max
                plot_kwargs['vmin'] = expr_min

            if not is_numeric: 
                if legend_loc == 'on data':
                    plot_kwargs['legend_loc'] = 'on data'
                elif legend_loc != 'right margin':
                     plot_kwargs['legend_loc'] = legend_loc

            # --- Single Plotting Call ---
            sc.pl.embedding(adata_sub, **plot_kwargs)

            # --- Post-Plot Adjustments ---
            if fix_coordinate_ratio:
                axs[i].set_aspect('equal')
            else:
                axs[i].set_aspect('auto')

            if is_numeric:
                if axs[i].collections:
                    mappable = axs[i].collections[0]
                    mappable.set_clim(vmin=expr_min, vmax=expr_max)
                    adjustColorbar(mappable)
        else:
            axs[i].set_visible(False)

    # --- 5. Global Legend ---
    if not is_numeric and legend_loc == 'right margin':
        for ax in axs:
            try:
                if ax.get_visible():
                    ax.legend().set_visible(False)
            except:
                pass
        
        _create_global_legend(fig, axs, legend_loc="center left", bbox_to_anchor=(1.02, 0.5),
                              frameon=False, marker_size=legend_marker_size,
                              fontsize=legend_fontsize, max_rows_per_col=18)

    # --- 6. Final Polish ---
    if show_axis_ticks:
        for ax in axs:
            ax.grid(False)
            ax.set_xticks(np.arange(xy_min[0]-xy_margin[0], xy_max[0]+xy_margin[0], (xy_max[0]-xy_min[0])/4))
            ax.set_yticks(np.arange(xy_min[1]-xy_margin[1], xy_max[1]+xy_margin[1], (xy_max[1]-xy_min[1])/4))

    if show_figure:
        plt.show() 

    if save:
        fig.savefig(save, bbox_inches='tight') 
        print("Figure saved to: ", save)
        plt.close(fig) 
    elif not show_figure:
        plt.close(fig)

        
from functools import wraps
# Create the alias
@wraps(plot_embeddings_split)
def plotEmbeddingsSplit(*args, **kwargs):
    """
    Alias for :func:`plot_embeddings_split`.
    
    Please refer to the main function for full documentation.
    """
    return plot_embeddings_split(*args, **kwargs)