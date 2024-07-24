from typing import Iterable, Union, Optional
import matplotlib.pyplot as plt
import scanpy as sc
def plot_features_violin(adata, 
                         feature_list,
                         groupby: Optional[str] = None,
                         use_raw: Optional[bool] = None,
                         layer: Optional[str] = None,
                         width_single: float = 14.0,
                         height_single: float=2.0,
                         size:float=0.1,
                         show_grid: bool = True,
                         show_figure:bool=True,
                         save: Optional[str] = None
                        ):
    """
    Plots a violin plot for each feature specified in `feature_list` using the Scanpy library (sc.pl.violin).
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.
    feature_list : List[str]
        A list of strings denoting the feature names (gene names and cell metrics, e.g., number of genes detected and doublet score) to be visualized.
    groupby : str, optional
        A key in the observation DataFrame (adata.obs) used to group data points in the violin plot. Default is None, which means no grouping is applied.
    use_raw : bool, optional
        A boolean indicating whether to use the raw attribute of adata. If True, uses raw data if available.
    layer : str, optional
        A key from the layers of adata. If provided, the specified layer is used for visualization.
    width_single : float, optional
        The width of each subplot. Default is 14.0.
    height_single : float, optional
        The height of each subplot. Default is 2.0.
    size : float, optional
        The size of the jitter points in the violin plot. Default is 0.1.
    show_grid : bool, optional
        Whether to display grid lines in the plots. Default is True (grid lines shown).
    show_figure : bool, optional
        Whether to show the figure (plt.show()). Default is True.
    save : str, optional
        If provided, the path where the plot should be saved, e.g., `./violin_plot_by_piaso.pdf`. If None, the plot is not saved to a file.
        
    Returns
    -------
    None
        This function does not return any value but visualizes the violin plots and optionally saves the figure.
    """
    n_features=len(feature_list)
    if n_features>1:
        fig, axs = plt.subplots(nrows=n_features, ncols=1, figsize=(width_single, height_single*n_features))

        # Go through the feature list
        for i, feature in enumerate(feature_list):
            if groupby:
                ax = sc.pl.violin(adata,
                    [feature],
                    groupby=groupby,
                    use_raw=use_raw,
                    layer=layer,
                    rotation=90,
                    jitter=0.4,
                    ax=axs[i],
                    size=size,
                    show=False
                    )
                # Only show x-axis labels on the bottom subplot
                if i < len(feature_list)-1:
                    axs[i].set(xticks=[], xlabel='')
            else:
                ax = sc.pl.violin(
                    adata=adata,
                    keys=feature,
                    use_raw=use_raw,
                    layer=layer,
                    rotation=90,
                    jitter=0.4,
                    ax=axs[i],
                    size=size,
                    show=False
                                 )
                
                axs[i].set_ylabel(feature)
                
                # Only show x-axis labels on the bottom subplot
                if i < len(feature_list)-1:
                    axs[i].set(xticks=[], xlabel='')
                else:
                    axs[i].set_xticks([0])  # Set ticks at position 0
                    axs[i].set_xticklabels(['All cells']) 
            axs[i].set_title(feature)
            if not show_grid:
                axs[i].grid(show_grid)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(width_single, height_single*len(feature_list)))
        # Go through the feature list
        for i, feature in enumerate(feature_list):
            if groupby:
                ax = sc.pl.violin(adata, [feature],
                 groupby=groupby,
                                   use_raw=use_raw,
                                  layer=layer,
                   rotation=90,
                 jitter=0.4,
                    ax=axs,
                                  size=size,
                     show=False
                    )
                # Only show x-axis labels on the bottom subplot
                if i < len(feature_list)-1:
                    axs[i].set(xticks=[], xlabel='')
            else:
                ax = sc.pl.violin(
                    
                    adata=adata,
                    keys=feature,
                    use_raw=use_raw,
                    layer=layer,
                    rotation=90,
                    jitter=0.4,
                    ax=axs,
                    size=size,
                    show=False                    
                 )
                
                axs.set_ylabel(feature)
                
                
                # Only show x-axis labels on the bottom subplot
                if i < len(feature_list)-1:
                    axs.set(xticks=[], xlabel='')
                else:
                    axs.set_xticks([0])  # Set ticks at position 0
                    axs.set_xticklabels(['All cells'])
                
            axs.set_title(feature)
            if not show_grid:
                axs.grid(show_grid)
        
    ### Save the figure
    if save is not None:
        if show_figure:
            plt.show()
        fig.savefig(save, bbox_inches='tight')   
        plt.close(fig)   
        print(f'Figure saved to: {save}')