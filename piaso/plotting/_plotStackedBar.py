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
from ._store_io import read_cells_columns as _read_cells_columns  # read-only


def _get_composition_data(data, groupby, splitby):
    """Get a cross-tabulation of groupby x splitby.

    Returns DataFrame: rows = splitby categories, columns = groupby categories, values = counts.
    """
    from ._categories import resolve_category_style, label_strings
    if _is_cytome_input(data):
        df = _read_cells_columns(data, [groupby, splitby])
    else:
        df = data.obs[[groupby, splitby]].copy()

    # Rows and columns in the order every plot uses (stored or categorical,
    # else natural): crosstab sorts strings, so clusters ran 0, 1, 10, 2.
    rows = resolve_category_style(data, splitby, df[splitby]).present(df[splitby])
    cols = resolve_category_style(data, groupby, df[groupby]).present(df[groupby])
    ct = pd.crosstab(label_strings(df[splitby]), label_strings(df[groupby]))
    return ct.reindex(index=rows, columns=cols, fill_value=0)


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
    show: Optional[bool] = None,
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
        Colors for groups. Without it, the column's stored colours (by name:
        ``adata.uns``, a cytome's ``set_categories``), then ``d_color4``.
        Colours follow the category, not its position, so ``sort_groups``
        moves a segment without recolouring it.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    legend_ncol : int, optional
        Legend columns. Auto-calculated if None.
    legend_fontsize : int
        Legend font size.
    show : bool, optional
        Call ``plt.show()``; ``None`` (default) shows unless ``ax`` is given.
    save : str, optional
        Save path.
    ax : Axes, optional
        Pre-existing axes.
    return_fig : bool
        Return (fig, ax). Otherwise the figure is returned when it is not shown.
    """
    from . import _figure
    from ._categories import resolve_category_style

    ct = _get_composition_data(data, groupby, splitby)

    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0)

    if sort_groups:
        # Rank on the normalized composition even when the bars are counts,
        # otherwise a large sample decides the order for every other one.
        share = ct.div(ct.sum(axis=1), axis=0) if not normalize else ct
        ct = ct[share.mean(axis=0).sort_values(ascending=False).index]

    groups = list(ct.columns)

    # Colours by name over the whole column: the list used to be matched to
    # the segments in the order drawn, so sort_groups swapped colours.
    style = resolve_category_style(data, groupby, palette=palette)

    if figsize is None:
        figsize = (max(6, len(ct) * 0.6 + 2), 5)

    show = _figure.resolve_show(show, ax)
    fig, ax, _created = _figure.begin(ax, figsize=figsize)

    bottom = np.zeros(len(ct))
    x = np.arange(len(ct))

    for g in groups:
        vals = ct[g].values
        ax.bar(x, vals, bottom=bottom, label=str(g), color=style.color(g), width=0.8,
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

    _figure.layout(fig)
    out = _figure.finish(fig, created=_created, show=show, save=save,
                         writekey='stacked_barplot')
    if return_fig:
        return fig, ax
    return out


@wraps(stacked_barplot)
def stackedBarplot(*args, **kwargs):
    """Alias for :func:`stacked_barplot`."""
    return stacked_barplot(*args, **kwargs)
