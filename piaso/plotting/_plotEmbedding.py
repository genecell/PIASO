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
        
        
### Plot embeddings side by side
import matplotlib.pyplot as plt
def plot_embeddings_split(adata,
                          color,
                          splitby,
                          ncol:int=None,
                          dpi:int=80,
                          col_size:int=5,
                          row_size:int=5,
                          save=None,
                          vmax=None,
                          vmin=None,
                          show_figure=True,
                          layer:str=None,
                          basis:str='X_umap',
                          fix_coordinate_ratio:bool=True, ### Fix the coordinate ratio
                          show_axis_ticks:bool=False, ### Whether to show the axis ticks and tick labels
                          margin_ratio:float=0.05, ### Set the margin ratio for both x-axis and y-axis, relative to the x-axis intervals and y-axis intervals, respectively
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
    save : str or None, optional (default=None)
        File path to save the resulting figure. If None, the figure will not be saved.
    vmax : float or None, optional (default=None)
        Maximum value for the color scale. If not provided, the upper limit is determined automatically.
    vmin : float or None, optional (default=None)
        Minimum value for the color scale. If not provided, the lower limit is determined automatically.
    show_figure : bool, optional (default=True)
        Whether to display the figure after plotting.
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

    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #Adapted from https://stackoverflow.com/questions/29516157/set-equal-aspect-in-plot-with-colorbar
    def adjustColorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    

    ### Create the unique variables
    variables=adata.obs[splitby].cat.categories
    ### Build the layout
    fig, axs, nrow, ncol = _build_subplots(len(variables), ncol=ncol, dpi=dpi, col_size=col_size, row_size=row_size)
    
    
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace=0.2)

    
    if type(axs) != np.ndarray:
        axs = [axs]
    else:
        axs = axs.ravel()
    
    
    ### in obs
    if np.isin(color, adata.obs.columns):
        
        
        emd_df=pd.DataFrame(adata.obsm[basis].copy())
          
        if all(v is not None for v in [x_min, y_min, x_max, y_max]):
            
            xy_min=np.array([x_min, y_min])
            xy_max=np.array([x_max, y_max])

        else:    
        
            xy_min=emd_df.min(axis=0).values
            xy_max=emd_df.max(axis=0).values

        xy_margin=(xy_max-xy_min)*margin_ratio

        for ax in axs:
            ax.set_xlim(xy_min[0]-xy_margin[0], xy_max[0]+xy_margin[0])
            ax.set_ylim(xy_min[1]-xy_margin[1], xy_max[1]+xy_margin[1])
        
        
        ### 240706, add the np.int_, to check whether it's int or not, because int could also be continous
        if isinstance(adata.obs[color][0], np.floating) or isinstance(adata.obs[color][0], np.int_):
            ## is continous
            if vmax is None:
                expr_max=adata.obs[color].max()
            else:
                expr_max=vmax
                
            if vmin is None:
                expr_min=adata.obs[color].min()
            else:
                expr_min=vmin
            
            
            for i in range(len(axs)):
                if i<len(variables):
                    if basis=='X_umap':
                        fig_tmp=sc.pl.umap(adata[adata.obs[splitby]==variables[i]],
                           color=color,
                           vmax=expr_max,
                           vmin=expr_min,
                           title=color+' in '+variables[i],
                           legend_fontsize=12,
                           legend_fontoutline=2,
                           ncols=4,
                           return_fig=False,
                           colorbar_loc=None,
                           show=False,ax=axs[i],  **kwargs
                           )
                    else:
                        fig_tmp=sc.pl.embedding(
                            adata[adata.obs[splitby]==variables[i]],
                            basis=basis,
                            color=color,
                            vmax=expr_max,
                            vmin=expr_min,
                            title=color+' in '+variables[i],
                            legend_fontsize=12,
                            legend_fontoutline=2,
                            ncols=4,
                            return_fig=False,
                            colorbar_loc=None,
                            show=False,ax=axs[i],  **kwargs
                        )
                        
                        
                    ### Fix the coordinates ratio
                    if fix_coordinate_ratio:
                        axs[i].set_aspect('equal')
                        ### Set the color bar range
                        axs[i].collections[0].set_clim(vmin=expr_min, vmax=expr_max)
                        
                        ### Refer to https://stackoverflow.com/questions/48131232/matplotlib-get-colorbar-mappable-from-an-axis
                        adjustColorbar(axs[i].collections[0])
                    else:
                        axs[i].set_aspect('auto')
                        ### Set the color bar range
                        axs[i].collections[0].set_clim(vmin=expr_min, vmax=expr_max)
                        
                        ### Refer to https://stackoverflow.com/questions/48131232/matplotlib-get-colorbar-mappable-from-an-axis
                        adjustColorbar(axs[i].collections[0])
                    
                    
                    
                else:
                    axs[i].set_visible(False)  
        else:
            ### Categorical variable
            for i in range(len(axs)):
                if i<len(variables):
                    
                    if basis=='X_umap':
                        sc.pl.umap(
                            adata[adata.obs[splitby]==variables[i]],
                            color=color,
                            title=color+' in '+variables[i],
                            legend_fontsize=12,
                            legend_fontoutline=2,
                            ncols=4,
                            show=False, ax=axs[i],
                            **kwargs
                           )
                    else:
                        sc.pl.embedding(
                            adata[adata.obs[splitby]==variables[i]],
                            basis=basis,
                            color=color,
                            title=color+' in '+variables[i],
                            legend_fontsize=12,
                            legend_fontoutline=2,
                            ncols=4,
                            show=False, ax=axs[i],
                            **kwargs
                        )
                        
                    ### Fix the coordinates ratio
                    if fix_coordinate_ratio:
                        axs[i].set_aspect('equal')
                    
                else:
                    axs[i].set_visible(False)  

        

    ### for gene
    else:
    
        ### Calculate the max values for all the plots
        if layer is not None:
            gene_df=pd.DataFrame(np.ravel(adata.layers[layer][:,np.where(adata.var_names==color)[0]].todense()))
        elif sparse.isspmatrix(adata.X):
            gene_df=pd.DataFrame(np.ravel(adata.X[:,np.where(adata.var_names==color)[0]].todense()))
        else:
            ### Check whether adata.X is in sparse format or not
            gene_df=pd.DataFrame(np.ravel(adata.X[:,np.where(adata.var_names==color)[0]]))

        
        if vmax is None:
            expr_max=gene_df.max().values[0]
        else:
            expr_max=vmax
            
        if vmin is None:
            expr_min=gene_df.min().values[0]
        else:
            expr_min=vmin

        emd_df=pd.DataFrame(adata.obsm[basis].copy())
        
        if all(v is not None for v in [x_min, y_min, x_max, y_max]):
            
            xy_min=np.array([x_min, y_min])
            xy_max=np.array([x_max, y_max])

        else:    
        
            xy_min=emd_df.min(axis=0).values
            xy_max=emd_df.max(axis=0).values
        
      

        xy_margin=(xy_max-xy_min)*0.05

        for ax in axs:
            ax.set_xlim(xy_min[0]-xy_margin[0], xy_max[0]+xy_margin[0])
            ax.set_ylim(xy_min[1]-xy_margin[1], xy_max[1]+xy_margin[1])


        for i in range(len(axs)):
            if i<len(variables):
                if basis=='X_umap':
                    fig_tmp=sc.pl.umap(adata[adata.obs[splitby]==variables[i]],
                       color=color,
                       vmax=expr_max,
                        vmin=expr_min,
                       layer=layer,
                       title=color+' in '+variables[i],
                               show=False,ax=axs[i], 
                        
                        return_fig=False,
                        colorbar_loc=None,
                               **kwargs
                       )
                else:
                    fig_tmp=sc.pl.embedding(
                        adata[adata.obs[splitby]==variables[i]],
                        basis=basis,
                        color=color,
                        vmax=expr_max,
                        vmin=expr_min,
                        layer=layer,
                        title=color+' in '+variables[i],
                               show=False,ax=axs[i], 
                        return_fig=False,
                        colorbar_loc=None,
                               **kwargs)
                
                
                ### Fix the coordinates ratio
                if fix_coordinate_ratio:
                    axs[i].set_aspect('equal')
                    ### Set the color bar range
                    axs[i].collections[0].set_clim(vmin=expr_min, vmax=expr_max)
                    ### Refer to https://stackoverflow.com/questions/48131232/matplotlib-get-colorbar-mappable-from-an-axis
                    adjustColorbar(axs[i].collections[0])
                else:
                    axs[i].set_aspect('auto')
                    ### Set the color bar range
                    axs[i].collections[0].set_clim(vmin=expr_min, vmax=expr_max)
                    ### Refer to https://stackoverflow.com/questions/48131232/matplotlib-get-colorbar-mappable-from-an-axis
                    adjustColorbar(axs[i].collections[0])
                
         

                        
            else:
                axs[i].set_visible(False)  
    
    # ### Fix the coordinates ratio
    # if fix_coordinate_ratio:
    #     for ax in axs:
    #         ax.set_aspect('equal')
            
    ### Show the axis ticks
    if show_axis_ticks:
        for ax in axs:
            ax.grid(False)
            ax.set_xticks(np.arange(xy_min[0]-xy_margin[0], xy_max[0]+xy_margin[0], (xy_max[0]-xy_min[0])/4))
            ax.set_yticks(np.arange(xy_min[1]-xy_margin[1], xy_max[1]+xy_margin[1], (xy_max[1]-xy_min[1])/4))


        
    ### Save the figure
    if save is not None:
        if show_figure:
            plt.show()
        fig.savefig(save, bbox_inches='tight')   # save the figure to file
        print("Figure saved to: ", save)
        plt.close(fig)    # close the figure window   
        
