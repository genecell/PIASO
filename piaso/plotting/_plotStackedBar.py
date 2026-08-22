"""Stacked barplot for cell composition across conditions.

Shows the proportion (or count) of each cell group within different conditions/samples.
Supports AnnData and cytome Dataset.
"""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ..utils._cytome_compat import read_cells_columns as _read_cells_columns


def _get_composition_data(data, groupby, splitby):
    """Get a cross-tabulation of groupby x splitby.

    Returns DataFrame: rows = splitby categories, columns = groupby categories, values = counts.
    """
    if _is_cytome_input(data):
        df = _read_cells_columns(data, [groupby, splitby])
    else:
        df = data.obs[[groupby, splitby]].copy()

    ct = pd.crosstab(df[splitby], df[groupby])
    return ct


def stacked_barplot(
    data,
    groupby: str = 'leiden',
    splitby: str = 'batch',
    normalize: bool = True,
    sort_groups: bool = False,
    palette=None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    legend_ncol: Optional[int] = None,
    legend_fontsize: int = 9,
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
):
    """Plot a stacked barplot of cell composition.

    Parameters
    ----------
    data : AnnData or cytome Dataset/path
        Input data.
    groupby : str
        Column for cell groups (bar segments).
    splitby : str
        Column for conditions/samples (bar positions on X axis).
    normalize : bool
        Normalize to fractions per splitby category.
    sort_groups : bool
        Order the stacked segments largest-proportion-first instead of
        following the group's category order. Default ``False``: the segment
        order then matches the legend, and the same colour sits at the same
        height in every bar, which is what makes two bars comparable by eye.
        Set ``True`` when a single bar's composition is the question and the
        ranking is the answer. The ranking uses the mean proportion across
        ``splitby`` categories, so one order applies to the whole plot; a
        per-bar ranking would put a different cell type at the bottom of every
        bar and make the plot unreadable.
    palette : list or dict, optional
        Colors for groups. Falls back to ``adata.uns`` then ``d_color4``.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    legend_ncol : int, optional
        Legend columns. Auto-calculated if None.
    legend_fontsize : int
        Legend font size.
    show : bool
        Call plt.show().
    save : str, optional
        Save path.
    ax : Axes, optional
        Pre-existing axes.
    return_fig : bool
        Return (fig, ax).
    """
    from . import color as _color_mod

    ct = _get_composition_data(data, groupby, splitby)

    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0)

    if sort_groups:
        # Rank on the normalized composition even when the bars are counts,
        # otherwise a large sample decides the order for every other one.
        share = ct.div(ct.sum(axis=1), axis=0) if not normalize else ct
        ct = ct[share.mean(axis=0).sort_values(ascending=False).index]

    groups = list(ct.columns)

    # Resolve palette
    if palette is None:
        if isinstance(palette, dict):
            pass
        elif not _is_cytome_input(data) and hasattr(data, 'uns'):
            key = f'{groupby}_colors'
            if key in data.uns and len(data.uns[key]) >= len(groups):
                palette = list(data.uns[key][:len(groups)])
        if palette is None:
            palette = [_color_mod.d_color4[i % len(_color_mod.d_color4)]
                       for i in range(len(groups))]
    elif isinstance(palette, dict):
        palette = [palette.get(g, _color_mod.d_color4[i % len(_color_mod.d_color4)])
                   for i, g in enumerate(groups)]

    if figsize is None:
        figsize = (max(6, len(ct) * 0.6 + 2), 5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bottom = np.zeros(len(ct))
    x = np.arange(len(ct))

    for i, g in enumerate(groups):
        vals = ct[g].values
        c = palette[i] if isinstance(palette, list) else palette[i % len(palette)]
        ax.bar(x, vals, bottom=bottom, label=str(g), color=c, width=0.8,
               edgecolor='white', linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ct.index, rotation=90, ha='center', fontsize=9)
    ax.set_ylabel('Fraction' if normalize else 'Count')
    if title:
        ax.set_title(title)

    if legend_ncol is None:
        legend_ncol = max(1, -(-len(groups) // 12))

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False,
              fontsize=legend_fontsize, ncol=legend_ncol)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    from ..settings import _savefig
    _savefig(fig, save, writekey='stacked_barplot')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax


@wraps(stacked_barplot)
def stackedBarplot(*args, **kwargs):
    """Alias for :func:`stacked_barplot`."""
    return stacked_barplot(*args, **kwargs)
