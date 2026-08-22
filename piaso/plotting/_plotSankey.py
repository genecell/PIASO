"""Sankey (alluvial) diagram for visualizing cell flow between two categorizations.

Pure matplotlib implementation — no extra dependencies. Works alongside plotConfusionMatrix
for similar inputs but provides a flow-based visualization.

Supports AnnData and cytome Dataset.
"""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from functools import wraps


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ..utils._cytome_compat import read_cells_columns as _read_cells_columns
from ._group_order import resolve_group_order


def _get_category_pairs(data, left_col, right_col):
    """Get paired category values from data.

    Returns DataFrame with two columns.
    """
    if _is_cytome_input(data):
        return _read_cells_columns(data, [left_col, right_col])
    return data.obs[[left_col, right_col]].copy()


def _bezier_path(x0, y0_top, y0_bot, x1, y1_top, y1_bot, n_points=50):
    """Create a Bezier-curve path between two vertical bar segments."""
    t = np.linspace(0, 1, n_points)
    xmid = (x0 + x1) / 2

    # Top curve
    top_x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * xmid + t ** 2 * x1
    top_y = (1 - t) ** 2 * y0_top + 2 * (1 - t) * t * (y0_top + y1_top) / 2 + t ** 2 * y1_top

    # Bottom curve (reversed)
    bot_x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * xmid + t ** 2 * x0
    bot_y = (1 - t) ** 2 * y1_bot + 2 * (1 - t) * t * (y0_bot + y1_bot) / 2 + t ** 2 * y0_bot

    verts = list(zip(top_x, top_y)) + list(zip(bot_x, bot_y)) + [(top_x[0], top_y[0])]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def sankey(
    data,
    left: str,
    right: str,
    palette=None,
    color_by: str = 'left',
    figsize: Optional[tuple] = None,
    alpha: float = 0.4,
    node_width: float = 0.08,
    gap: float = 0.03,
    title: Optional[str] = None,
    fontsize: int = 9,
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
):
    """Plot a Sankey (alluvial) diagram between two categorical columns.

    Parameters
    ----------
    data : AnnData or cytome Dataset/path
        Input data.
    left : str
        Left-side category column.
    right : str
        Right-side category column.
    palette : list or dict, optional
        Colors. If None, uses ``d_color4``.
    color_by : str
        Color flows by ``'left'`` or ``'right'`` categories.
    figsize : tuple, optional
        Figure size.
    alpha : float
        Flow ribbon transparency.
    node_width : float
        Width of category bars (in data coords).
    gap : float
        Vertical gap between bars (fraction of total height).
    title : str, optional
        Title.
    fontsize : int
        Label font size.
    show, save, ax, return_fig
        Output options.
    """
    from . import color as _color_mod

    df = _get_category_pairs(data, left, right)
    df = df.dropna()
    df[left] = df[left].astype(str)
    df[right] = df[right].astype(str)

    left_cats = resolve_group_order(df[left])
    right_cats = resolve_group_order(df[right])

    ct = pd.crosstab(df[left], df[right])
    total = ct.values.sum()

    # Resolve palette
    all_cats = left_cats + right_cats
    if palette is None:
        if not _is_cytome_input(data) and hasattr(data, 'uns'):
            key = f'{left}_colors' if color_by == 'left' else f'{right}_colors'
            if key in data.uns:
                palette = list(data.uns[key])
        if palette is None:
            palette = _color_mod.d_color4
    if isinstance(palette, dict):
        color_list = [palette.get(c, _color_mod.d_color4[i % len(_color_mod.d_color4)])
                      for i, c in enumerate(all_cats)]
    else:
        color_list = [palette[i % len(palette)] for i in range(len(all_cats))]

    left_colors = {c: color_list[i] for i, c in enumerate(left_cats)}
    right_colors = {c: color_list[len(left_cats) + i] for i, c in enumerate(right_cats)}

    if figsize is None:
        figsize = (8, max(4, max(len(left_cats), len(right_cats)) * 0.4 + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute bar positions
    def _compute_positions(cats, counts_per_cat, x_pos):
        total_height = 1.0
        n = len(cats)
        total_gap = gap * (n - 1)
        available = total_height - total_gap
        total_count = sum(counts_per_cat.values())
        positions = {}
        y = 0
        for cat in cats:
            h = (counts_per_cat[cat] / total_count) * available if total_count > 0 else 0
            positions[cat] = (y, y + h)
            y += h + gap
        return positions

    left_counts = {c: ct.loc[c].sum() if c in ct.index else 0 for c in left_cats}
    right_counts = {c: ct[c].sum() if c in ct.columns else 0 for c in right_cats}

    left_pos = _compute_positions(left_cats, left_counts, 0)
    right_pos = _compute_positions(right_cats, right_counts, 1)

    x_left = 0
    x_right = 1

    # Draw bars
    for cat in left_cats:
        y0, y1 = left_pos[cat]
        ax.barh(y=(y0 + y1) / 2, width=node_width, height=y1 - y0,
                left=x_left - node_width / 2, color=left_colors[cat],
                edgecolor='white', linewidth=0.5, zorder=5)
        ax.text(x_left - node_width / 2 - 0.02, (y0 + y1) / 2, cat,
                ha='right', va='center', fontsize=fontsize)

    for cat in right_cats:
        y0, y1 = right_pos[cat]
        ax.barh(y=(y0 + y1) / 2, width=node_width, height=y1 - y0,
                left=x_right - node_width / 2, color=right_colors[cat],
                edgecolor='white', linewidth=0.5, zorder=5)
        ax.text(x_right + node_width / 2 + 0.02, (y0 + y1) / 2, cat,
                ha='left', va='center', fontsize=fontsize)

    # Draw flows
    left_cursor = {c: left_pos[c][0] for c in left_cats}
    right_cursor = {c: right_pos[c][0] for c in right_cats}
    total_count = ct.values.sum()
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())

    total_gap_l = gap * (len(left_cats) - 1)
    total_gap_r = gap * (len(right_cats) - 1)
    avail_l = 1.0 - total_gap_l
    avail_r = 1.0 - total_gap_r

    for lc in left_cats:
        for rc in right_cats:
            if lc not in ct.index or rc not in ct.columns:
                continue
            count = ct.loc[lc, rc]
            if count == 0:
                continue

            h_left = (count / left_total) * avail_l if left_total > 0 else 0
            h_right = (count / right_total) * avail_r if right_total > 0 else 0

            y0_top = left_cursor[lc]
            y0_bot = left_cursor[lc] + h_left
            y1_top = right_cursor[rc]
            y1_bot = right_cursor[rc] + h_right

            path = _bezier_path(
                x_left + node_width / 2, y0_top, y0_bot,
                x_right - node_width / 2, y1_top, y1_bot
            )
            color = left_colors[lc] if color_by == 'left' else right_colors[rc]
            patch = mpatches.PathPatch(path, facecolor=color, alpha=alpha,
                                       edgecolor='none', zorder=2)
            ax.add_patch(patch)

            left_cursor[lc] += h_left
            right_cursor[rc] += h_right

    ax.set_xlim(-0.3, 1.3)
    # `_compute_positions` fits the bars AND their gaps into [0, 1] --
    # `available = 1 - total_gap` -- so the diagram already spans exactly one
    # unit. The old limit added `gap * n_categories` on top of that, padding
    # the axis for space the layout had already reserved; with 27 categories
    # and gap=0.03 the top of the axis reached 1.86 and nearly half the figure
    # came out blank. The padding has to be constant, not per-category.
    ax.set_ylim(-0.05, 1.05)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # Column labels
    ax.text(x_left, -0.03, left, ha='center', va='top', fontsize=fontsize + 1,
            fontweight='bold')
    ax.text(x_right, -0.03, right, ha='center', va='top', fontsize=fontsize + 1,
            fontweight='bold')

    plt.tight_layout()

    from ..settings import _savefig
    _savefig(fig, save, writekey='sankey')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax


@wraps(sankey)
def plotSankey(*args, **kwargs):
    """Alias for :func:`sankey`."""
    return sankey(*args, **kwargs)
