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

### Plot embeddings side by side
import matplotlib.pyplot as plt
def plot_embeddings_split(adata,
                          color,
                          splitby,
                          col_size=5,
                          row_size=5,
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
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #Adapted from https://stackoverflow.com/questions/29516157/set-equal-aspect-in-plot-with-colorbar
    def adjustColorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    
    
    # https://github.com/theislab/scanpy/issues/137
    def _build_subplots(n):
        '''
        Build subplots grid
        n: number of subplots
        '''
        nrow = int(np.sqrt(n))
        ncol = int(np.ceil(n / nrow))
        fig, axs = plt.subplots(nrow, ncol, dpi=80, figsize=(ncol*col_size, nrow*row_size))

        return fig, axs, nrow, ncol

    ### Create the unique variables
    variables=adata.obs[splitby].cat.categories
    fig, axs, nrow, ncol = _build_subplots(len(variables))
    
    
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
        
