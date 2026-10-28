"""Sankey (alluvial) diagram for visualizing cell flow between two categorizations.

Pure matplotlib implementation — no extra dependencies. Works alongside plotConfusionMatrix
for similar inputs but provides a flow-based visualization.

Supports AnnData and cytome Dataset.
"""

import warnings
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from functools import wraps


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ._store_io import read_cells_columns as _read_cells_columns  # read-only
from ._group_order import resolve_group_order
from ._seriation import resolve_pair_order


def _get_category_pairs(data, left_col, right_col):
    """Get paired category values from data.

    Returns DataFrame with two columns.
    """
    if _is_cytome_input(data):
        return _read_cells_columns(data, [left_col, right_col])
    return data.obs[[left_col, right_col]].copy()


def _effective_gap(n: int, gap: float, max_share: float = 0.5) -> float:
    """Per-gap height, shrunk so the gaps cannot eat the whole axis.

    The gaps between bars total ``gap * (n - 1)`` out of a unit height. At the
    0.03 default that is 87% of the axis for 30 categories -- the bars become
    slivers -- and at 35 categories it exceeds 1.0, so ``available`` goes
    negative and every bar is drawn with negative height. Capping the share
    the gaps may take keeps the layout sane at any category count and changes
    nothing below the cap.
    """
    if n < 2:
        return gap
    return min(gap, max_share / (n - 1))


_FLOW_SCALES = ("linear", "sqrt", "log")

_FLOW_FUNCS = {
    "linear": lambda v: v,
    "sqrt": np.sqrt,
    "log": np.log1p,            # log1p, so a one-cell flow is finite
}

_SIZE_LEGEND_LOCS = ("lower right", "lower left", "upper right", "upper left")

#: Node sizing uses the same three shapes as flow sizing, for the same reason.
_NODE_SCALES = _FLOW_SCALES
_NODE_FUNCS = _FLOW_FUNCS


def _solve_flow_affine(f_sum, f_min, n_flows, avail, min_width):
    """Solve ``width = m * f(count) + b`` for one rail, returning ``(m, b)``.

    Two constraints pin it: the ribbons fill the rail (``sum(width) == avail``)
    and the thinnest one is exactly ``min_width``. Both ``f`` and the offset are
    monotone in the count, so neither can reorder flows -- what they cost is
    proportionality, not rank.

    ``min_width`` is the input rather than ``b`` on purpose. What a diagram can
    afford is ``avail / n_flows``, which depends on the flow *count*, and that
    is not a number anyone knows while typing the call.
    """
    if n_flows == 0 or f_sum <= 0:
        return 0.0, 0.0
    m_prop = avail / f_sum
    if min_width <= 0 or m_prop * f_min >= min_width:
        # Proportional already clears the floor; an offset would take height
        # from flows that never needed it, so don't charge for one.
        return m_prop, 0.0
    if n_flows * min_width >= avail:        # the caller reports this in points
        raise ValueError("min_flow_width leaves no room for the flows")
    denom = f_sum - n_flows * f_min
    if denom <= 0:
        # Every flow is already the same size; no slope can separate them.
        return 0.0, avail / n_flows
    m = (avail - n_flows * min_width) / denom
    return m, min_width - m * f_min


def _solve_node_heights(sizes, avail, min_axis, scale):
    """Node heights on one rail: ``height = M * g(size) + B``, same solver.

    A node has no size of its own by default -- it is exactly as tall as the
    ribbons landing on it, which is what makes them meet. Giving it one means
    solving the rail the same way the ribbons are solved, then splitting each
    node's height across its own ribbons in proportion to their counts. The
    ribbons still meet and still fill the rail; what they lose is a *global*
    count-to-width map, because the slope now depends on the node.

    That loss is the whole reason only one size key is ever drawn.
    """
    sizes = np.asarray(sizes, dtype=float)
    nz = sizes > 0
    n = int(nz.sum())
    g = np.zeros_like(sizes)
    if n:
        g[nz] = _NODE_FUNCS[scale](sizes[nz])
    g_sum = float(g.sum())
    g_min = float(g[nz].min()) if n else 0.0
    m, b = _solve_flow_affine(g_sum, g_min, n, avail, min_axis)
    return np.where(nz, m * g + b, 0.0)


def _split_node_height(w, heights, axis):
    """Rescale each node's ribbons so they sum to that node's height.

    ``axis=1`` walks the left rail (a node is a row), ``axis=0`` the right one
    (a node is a column). A node with no ribbons keeps its zeros -- there is
    nothing to give the height to, and it is not drawn.
    """
    w = np.asarray(w, dtype=float).copy()
    sums = w.sum(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(sums > 0, heights / np.where(sums > 0, sums, 1.0), 0.0)
    return w * (factor[:, None] if axis == 1 else factor[None, :])


def _points_per_unit(ax, layout, r_in=1.0, r_out=None):
    """Rendered points per one unit of rail position ``t``.

    ``min_flow_width`` is in points because that is what "too thin to see"
    means: a fraction of the axis says nothing without a figure size. This is
    the only place the two are related, and it needs a settled axes, which is
    why the limits are fixed before anything is drawn.
    """
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    if layout == "vertical":                # ribbons run across, thickness in x
        span = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
        px = bb.width
    else:
        span = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        px = bb.height
    if span <= 0 or px <= 0:
        return 0.0
    per_data = (px * 72.0 / fig.dpi) / span
    if layout == "circular":
        # t spans a semicircle, so one t-unit is an arc of length pi * r_in.
        return per_data * np.pi * r_in
    return per_data


def _legend_sample_counts(lo, hi, max_entries=4):
    """Round counts spanning the observed range, both ends always included."""
    lo, hi = int(lo), int(hi)
    if lo >= hi:
        return [lo]
    vals = {lo, hi}
    p = 0 if lo < 1 else int(np.floor(np.log10(lo))) + 1
    while 10 ** p < hi:
        if 10 ** p > lo:
            vals.add(10 ** p)
        p += 1
    out = sorted(vals)
    while len(out) > max_entries:
        out.pop(len(out) // 2)              # thin the middle, keep the extremes
    return out


def _rail_span_fraction(ax, layout, node_width):
    """Axes-fraction bounds of the field between the two rails.

    The category labels sit outside the rails, so this is the width the key can
    use without covering them. Circular has no outside to speak of -- the
    labels ring the annulus -- so it gets the whole axes.
    """
    if layout == "circular":
        return 0.0, 1.0
    # `base` is measured across the swatch: axes-fraction y for `vertical`
    # (whose rails run left-to-right), axes-fraction x otherwise.
    lo, hi = ax.get_ylim() if layout == "vertical" else ax.get_xlim()
    a, b = node_width / 2, 1.0 - node_width / 2
    span = hi - lo
    if span == 0:
        return 0.0, 1.0
    f = sorted(((a - lo) / span, (b - lo) / span))
    return f[0], f[1]


def _draw_size_legend(ax, counts, thicknesses, layout, r_in, pt_per_unit,
                      loc, fontsize, color, title, base_lo=0.0, base_hi=1.0):
    """Key mapping a drawn thickness back to a cell count.

    Only meaningful because the map it describes is global -- one slope and one
    offset for every item of that kind in the figure. Exactly one map can be
    global at a time: give the nodes a size of their own and each node's
    ribbons are renormalised to fit it, so the ribbon map turns per-node and a
    ribbon key becomes a lie. The caller picks the ``kind`` accordingly and
    draws one key or none; this routine is the same drawing either way, which
    is the point of there being one of it.

    ``thicknesses`` arrive in rail-space units and the swatches are drawn
    through the same transform the ribbons are, so a swatch is literally the
    size of the ribbon it stands for. That is why a thin swatch is not padded
    up to a legible minimum: the row *spacing* absorbs the labels instead. A
    key drawn at a different scale from the thing it describes is worse than
    no key at all -- it would make the thin end look bigger than it is, which
    is the exact misreading the key exists to prevent.
    """
    from matplotlib.transforms import blended_transform_factory as _blend

    if not counts or pt_per_unit <= 0:
        return
    along_x = layout == "vertical"          # thickness runs across the rails
    # An arc, not a chord: one rail unit spans pi * r_in of a circular diagram.
    scale = np.pi * r_in if layout == "circular" else 1.0
    thick = [t * scale for t in thicknesses]

    # Position across the swatch in axes fraction (robust to any limits),
    # thickness in data units (so it is the ribbon's own size, exactly).
    trans = (_blend(ax.transData, ax.transAxes) if along_x
             else _blend(ax.transAxes, ax.transData))
    lo, hi = ax.get_xlim() if along_x else ax.get_ylim()
    lo, hi = min(lo, hi), max(lo, hi)
    span = hi - lo
    line = fontsize * 1.6 / pt_per_unit * scale      # a text line, data units
    pad = line * 0.35
    length, edge, tick = 0.09, 0.02, 0.012          # axes fraction

    advance = [max(t, line * 0.9) for t in thick]   # spacing, never thickness
    block = sum(advance) + pad * len(thick) + line * 1.2
    far = loc.endswith("right") if along_x else loc.startswith("upper")
    v0 = (hi - span * 0.03 - block) if far else (lo + span * 0.03)
    v = v0
    side = loc.startswith("upper") if along_x else loc.endswith("right")
    # Anchored to the rails, not the axes edge. Outside the rails is where the
    # category labels live, and a key drawn on top of them is not a key.
    base = (base_hi - edge - length) if side else (base_lo + edge)
    tx = (base - tick) if side else (base + length + tick)
    fs = max(fontsize - 1, 1)

    # Backing, so the key survives landing on a ribbon. Inside the rails is the
    # only space a Sankey reliably has, and it is never empty.
    room = 0.075                                    # for the count labels
    bx = (base - room - tick) if side else base
    ax.add_patch(mpatches.Rectangle(
        ((v0 - pad, bx) if along_x else (bx, v0 - pad)),
        *((block + 2 * pad, length + room + tick) if along_x
          else (length + room + tick, block + 2 * pad)),
        transform=trans, clip_on=False, facecolor="white", alpha=0.78,
        edgecolor="none", zorder=5.8))

    for c, t, adv in zip(counts, thick, advance):
        mid = v + adv / 2
        label = f"{int(round(c)):,}"
        if along_x:
            ax.add_patch(mpatches.Rectangle(
                (mid - t / 2, base), t, length, transform=trans, clip_on=False,
                facecolor=color, edgecolor="none", zorder=6))
            ax.text(mid, tx, label, transform=trans, ha="center",
                    va="top" if side else "bottom", rotation=90,
                    fontsize=fs, zorder=6)
        else:
            ax.add_patch(mpatches.Rectangle(
                (base, mid - t / 2), length, t, transform=trans, clip_on=False,
                facecolor=color, edgecolor="none", zorder=6))
            ax.text(tx, mid, label, transform=trans,
                    ha="right" if side else "left", va="center",
                    fontsize=fs, zorder=6)
        v += adv + pad

    if along_x:
        ax.text(v, tx, title, transform=trans, ha="left",
                va="top" if side else "bottom", rotation=90,
                fontsize=fs, fontweight="bold", zorder=6)
    else:
        ax.text(tx if side else base, v, title, transform=trans,
                ha="right" if side else "left", va="bottom",
                fontsize=fs, fontweight="bold", zorder=6)


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



# ---------------------------------------------------------------------------
# Layout
#
# Every layout places the same two things: a rail of category blocks at each
# end, and a ribbon per (left, right) pair. The geometry is computed once in
# "rail space" -- rail index r in {0, 1}, position t along the rail in [0, 1]
# -- and each layout only says where a rail-space point lands on the axes.
# ---------------------------------------------------------------------------

_LAYOUTS = ("horizontal", "vertical", "circular")


def _rail_pt_to_xy(rail_x, t, layout):
    """A point in rail space -> axes coordinates.

    ``rail_x`` runs 0..1 across the gap between the rails, ``t`` runs 0..1
    along a rail. This is the same convention ``_rail_to_xy`` uses for the
    blocks, which is the point: the ribbons have to be flipped exactly as the
    blocks are, or they attach in the opposite vertical order.
    """
    if layout == "vertical":
        return t, 1.0 - rail_x
    return rail_x, 1.0 - t


def _rail_to_xy(r, t, layout, radius=1.0):
    """Rail space -> axes coordinates.

    horizontal: rail 0 on the left, rail 1 on the right, t running
                downward from the top so the first category is first.
    vertical:   rail 0 on top, rail 1 at the bottom, t left to right.
    circular:   rail 0 is the left semicircle, rail 1 the right one, both
                running from the top downward so matching categories face
                each other across the middle.
    """
    if layout == "vertical":
        return t, 1.0 - r
    if layout == "circular":
        ang = np.pi / 2 + (t * np.pi if r == 0 else -t * np.pi)
        return radius * np.cos(ang), radius * np.sin(ang)
    # 1 - t, not t: rails stack from t=0, and matplotlib's y grows upward, so
    # returning t put the FIRST category at the bottom -- a Sankey read
    # 5,4,3,2,1,0 top to bottom while the legend beside it read 0,1,2. The
    # circular layout already ran top-down; this makes horizontal agree.
    return r, 1.0 - t


def _circular_ribbon(t0a, t0b, t1a, t1b, r_in, n=40):
    """A chord from an arc on the left semicircle to one on the right.

    Two arcs joined by two quadratic curves whose control point is the origin,
    which is what gives a chord diagram its waist.
    """
    def arc(rail, ta, tb):
        ts = np.linspace(ta, tb, n)
        pts = [_rail_to_xy(rail, t, "circular", r_in) for t in ts]
        return np.asarray(pts)

    left = arc(0, t0a, t0b)
    right = arc(1, t1b, t1a)

    def bez(p0, p1, steps=n):
        u = np.linspace(0, 1, steps)[:, None]
        ctrl = np.zeros(2)          # through the centre
        return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * ctrl + u ** 2 * p1

    verts = np.vstack([left, bez(left[-1], right[0]), right,
                       bez(right[-1], left[0])])
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2)
             + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


def sankey(
    data,
    left: str,
    right: str,
    palette=None,
    color_by: str = 'left',
    left_order: Optional[Sequence] = None,
    right_order: Optional[Sequence] = None,
    layout: str = 'horizontal',
    flow_scale: str = 'linear',
    min_flow_width: Optional[float] = None,
    node_scale: str = 'linear',
    min_node_size: Optional[float] = None,
    show_size_legend: Optional[bool] = None,
    size_legend_loc: str = 'lower right',
    figsize: Optional[tuple] = None,
    alpha: float = 0.4,
    node_width: float = 0.08,
    gap: float = 0.03,
    title: Optional[str] = None,
    fontsize: int = 9,
    show: Optional[bool] = None,
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
    palette : list, dict or str, optional
        Colors. A dict maps category names (on either rail) to colours; a list
        or a colormap name colours the ``color_by`` column, matched to its
        category order. Without it, each column's stored colours (by name:
        ``adata.uns``, a cytome's ``set_categories``), then ``d_color4``. A
        rail without colours of its own continues the palette after the
        coloured column's categories, so the two rails do not repeat colours.
    color_by : str
        Color flows by ``'left'`` or ``'right'`` categories.

        *Changed*: ``color_by='right'`` coloured the right column's categories
        with the palette offset by the number of left categories, and
        ``left_order='auto'`` recoloured the reordered nodes; colours now
        follow each category by name.
    flow_scale : {'linear', 'sqrt', 'log'}, default ``'linear'``
        Shape of the count-to-width map. ``'linear'`` is proportional and is
        what a Sankey normally promises. ``'sqrt'`` and ``'log'`` (``log1p``)
        compress the range so the thin end stays legible, at the cost of the
        thick end. Reach for one when the flows span orders of magnitude: on a
        2000-fold range half the ribbons come out under a hairline, and no
        figure size fixes that, because it is each flow's *share* of the
        diagram that is too small.
    min_flow_width : float, optional
        Thickness of the thinnest ribbon, in points. This is an affine offset
        applied to every flow -- ``width = m * f(count) + b``, solved so the
        ribbons still fill the rail -- not a clamp, so nothing is tied and no
        flow overtakes another. It costs ``n_flows * min_flow_width`` of the
        diagram, which means what limits it is the flow *count*; a request
        that does not fit raises, having done that arithmetic for you.
        Composes with ``flow_scale``: ``'sqrt'`` shrinks the range first, so it
        needs far less of a baseline than ``'linear'`` to clear the same
        hairline.
    node_scale : {'linear', 'sqrt', 'log'}, default ``'linear'``
        Shape of the size-to-height map for the category blocks, mirroring
        ``flow_scale``. ``'linear'`` is the default because a node has no
        height of its own: it is the sum of its ribbons, which is what makes
        them meet.

        *Added in 1.2.4.*
    min_node_size : float, optional
        Height of the shortest category block, in points -- the node-level
        parallel of ``min_flow_width``, solved the same affine way, so blocks
        stay monotone in group size and nothing is tied. It costs
        ``n_nodes * min_node_size`` of the rail and raises with that
        arithmetic when it does not fit.

        Reach for it when a small category is invisible: without it a node's
        height is ``m * size + b * fan_out``, so how tall a small group draws
        depends on how many ways it splits, which is the wrong variable.

        **What it costs.** Each node's ribbons are renormalised to fill its
        new height, so the count-to-thickness map becomes per-node: the same
        flow can be drawn at two thicknesses at its two ends. Ribbon widths
        therefore stop being comparable across nodes, ``min_flow_width``
        stops being exact, and the size key switches to describing nodes
        (see ``show_size_legend``).

        *Added in 1.2.4.*
    show_size_legend : bool, optional
        Draw a key mapping a drawn thickness back to a cell count. ``None``
        (default) turns it on exactly when a map stops being proportional,
        because a non-proportional size with no key invites the reader to
        divide two of them and get the wrong ratio.

        **One key, never two**, because only one map is global at a time:

        =========================== ==========================================
        state                        key
        =========================== ==========================================
        both maps proportional       none (``True`` forces the flow key)
        flows non-proportional       flows -- "cells per flow"
        nodes non-proportional       nodes -- "cells per node"; the flow key
                                     is suppressed, since per-node
                                     renormalisation has made it untrue
        =========================== ==========================================

        *Renamed in 1.2.4* from ``show_width_legend``.
    size_legend_loc : str, default ``'lower right'``
        Corner for the key: ``'lower right'``, ``'lower left'``,
        ``'upper right'`` or ``'upper left'``. It is drawn inside the axes, so
        move it if it lands on a ribbon.

        *Renamed in 1.2.4* from ``width_legend_loc``.
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
    show : bool, optional
        Call ``plt.show()``; ``None`` (default) shows unless ``ax`` is given.
    save, ax, return_fig
        Output options. Without ``return_fig`` the figure is returned when it
        is not shown. A figure passed in through ``ax`` is never closed.

    Notes
    -----
    Three properties -- node height proportional to size, ribbon width
    proportional to flow, node height equal to the sum of its ribbons -- and
    any mechanism for making something thin visible costs one of them. The
    default keeps all three and pays in visibility instead.

    A node is exactly as tall as the ribbons landing on it, so the moment the
    width map stops being proportional, node height stops tracking group size:
    it becomes ``m * size + b * fan_out``, and a category that splits many ways
    is drawn taller than an equally large one that maps cleanly.
    ``min_node_size`` fixes that by giving the *node* the affine map and
    splitting its height across its ribbons -- which spends the third property
    instead, since a ribbon's thickness then depends on which node it leaves.

    So at most one of the two maps is ever global, and exactly the global one
    gets a key. That is why there is one ``show_size_legend`` rather than a
    node key and a flow key that could be drawn together: two keys in a corner
    would not be clutter, they would be a contradiction.
    """
    from . import color as _color_mod

    # Checked before anything is built: a typo that raises after the figure
    # exists leaves a half-drawn axes behind in a notebook.
    if layout not in _LAYOUTS:
        raise ValueError(f"sankey: layout={layout!r} is not one of {_LAYOUTS}")
    if flow_scale not in _FLOW_SCALES:
        raise ValueError(f"sankey: flow_scale={flow_scale!r} is not one of "
                         f"{_FLOW_SCALES}")
    if node_scale not in _NODE_SCALES:
        raise ValueError(f"sankey: node_scale={node_scale!r} is not one of "
                         f"{_NODE_SCALES}")
    if min_flow_width is not None and min_flow_width < 0:
        raise ValueError(f"sankey: min_flow_width={min_flow_width!r} is a "
                         f"thickness in points and cannot be negative")
    if min_node_size is not None and min_node_size < 0:
        raise ValueError(f"sankey: min_node_size={min_node_size!r} is a "
                         f"height in points and cannot be negative")
    if size_legend_loc not in _SIZE_LEGEND_LOCS:
        raise ValueError(f"sankey: size_legend_loc={size_legend_loc!r} is "
                         f"not one of {_SIZE_LEGEND_LOCS}")

    from ._categories import resolve_category_style, missing_mask, label_strings
    from . import _figure

    df_all = _get_category_pairs(data, left, right)
    # a cell is drawn only when both of its labels are known
    keep = ~(missing_mask(df_all[left]) | missing_mask(df_all[right]))
    df = df_all[keep].copy()

    # Order first, stringify second. `.astype(str)` throws away the categorical
    # dtype, and the categorical order is what plotEmbedding draws the legend
    # from -- resolving afterwards could only ever fall back to a sort, which
    # is why the Sankey disagreed with the UMAP about the same cells.
    # "auto" is resolved from the counts, so the order it wants does not exist
    # until the crosstab does. Everything else resolves from the labels.
    _auto_l = isinstance(left_order, str) and left_order == "auto"
    _auto_r = isinstance(right_order, str) and right_order == "auto"

    # Colours: each rail from its own column, by name, over the whole column
    # (the resolver every plot uses), so neither color_by nor a reordering can
    # move one. The coloured column takes `palette` (a list or colormap name);
    # the other rail's defaults continue that palette after the coloured
    # column's categories, as the two rails always did, so they do not repeat.
    from . import color as _color_mod
    _coloured, _other = (left, right) if color_by == 'left' else (right, left)
    style_c = resolve_category_style(data, _coloured, df_all[_coloured], palette=palette)
    _base = (list(palette) if isinstance(palette, (list, tuple, np.ndarray)) and len(palette)
             else list(_color_mod.d_color4))
    _n_c = len(style_c.canonical)
    _rotated = [_base[(_n_c + i) % len(_base)] for i in range(len(_base))]
    style_o = resolve_category_style(
        data, _other, df_all[_other],
        palette=palette if isinstance(palette, dict) else None,
        default_palette=_rotated)
    style_l, style_r = (style_c, style_o) if color_by == 'left' else (style_o, style_c)

    # display orders: the caller's, else the stored/categorical one, else
    # natural (resolve_group_order's rule, now reading a store's order too)
    left_cats = resolve_category_style(
        data, left, df[left], order=None if _auto_l else left_order).present(df[left])
    right_cats = resolve_category_style(
        data, right, df[right], order=None if _auto_r else right_order).present(df[right])
    df[left] = label_strings(df[left])
    df[right] = label_strings(df[right])

    ct = pd.crosstab(df[left], df[right])
    total = ct.values.sum()

    if _auto_l or _auto_r:
        left_cats, right_cats = resolve_pair_order(
            ct,
            left_order="auto" if _auto_l else left_cats,
            right_order="auto" if _auto_r else right_cats,
            default_left=left_cats, default_right=right_cats,
        )

    left_colors = {c: style_l.color(c) for c in left_cats}
    right_colors = {c: style_r.color(c) for c in right_cats}

    if figsize is None:
        figsize = (8, max(4, max(len(left_cats), len(right_cats)) * 0.4 + 1))

    show = _figure.resolve_show(show, ax)
    fig, ax, _created = _figure.begin(ax, figsize=figsize)

    def _compute_positions(cats, heights):
        """Stack blocks of given height down the rail, gap between each."""
        eff = _effective_gap(len(cats), gap)
        positions = {}
        y = 0.0
        for cat in cats:
            h = float(heights.get(cat, 0.0))
            positions[cat] = (y, y + h)
            y += h + eff
        return positions

    x_left = 0
    x_right = 1

    # Radii for the circular layout: the blocks sit in the annulus between
    # them and the ribbons spring from its inner edge.
    r_in, r_out = 1.0, 1.0 + node_width

    # Limits before anything is drawn. min_flow_width is in points, and points
    # only mean something once the axes knows how much data it spans -- so the
    # measurement has to come after the limits and before the geometry.
    if layout == "circular":
        pad = r_out + 0.45
        ax.set_xlim(-pad, pad)
        ax.set_ylim(-pad, pad)
        ax.set_aspect("equal")
    elif layout == "vertical":
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.3, 1.3)
    else:
        # `_compute_positions` fits the bars AND their gaps into [0, 1], so the
        # diagram already spans exactly one unit; the padding is constant, not
        # per-category. The old per-category padding reached 1.86 at 27
        # categories and left nearly half the figure blank.
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.05, 1.05)
    ax.set_axis_off()

    # ---- counts -> widths -------------------------------------------------
    # Solved once, for the blocks and the ribbons together. A node's height is
    # the sum of its ribbons; transform one and not the other and they stop
    # meeting.
    eff_l = _effective_gap(len(left_cats), gap)
    eff_r = _effective_gap(len(right_cats), gap)
    avail_l = 1.0 - eff_l * (len(left_cats) - 1)
    avail_r = 1.0 - eff_r * (len(right_cats) - 1)

    ctv = (ct.reindex(index=left_cats, columns=right_cats)
             .fillna(0).values.astype(float))
    nz = ctv > 0
    n_flows = int(nz.sum())
    fx = np.zeros_like(ctv)
    if n_flows:
        fx[nz] = _FLOW_FUNCS[flow_scale](ctv[nz])
    fx_sum = float(fx.sum())
    fx_min = float(fx[nz].min()) if n_flows else 0.0

    # Non-proportional as *requested*, not as realised: a baseline that turns
    # out not to bind still leaves the caller expecting a key, and a scale bar
    # on a proportional diagram is honest either way.
    flow_non_proportional = flow_scale != "linear" or bool(min_flow_width)
    node_non_proportional = node_scale != "linear" or bool(min_node_size)
    # Exactly one map can be global, so exactly one kind of key can be true.
    # Nodes win when they are sized, because that is what made the flow map
    # per-node in the first place.
    legend_kind = "node" if node_non_proportional else "flow"
    want_legend = (flow_non_proportional or node_non_proportional
                   if show_size_legend is None else bool(show_size_legend))

    if node_non_proportional and min_flow_width:
        warnings.warn(
            "sankey: min_flow_width is not exact alongside min_node_size / "
            "node_scale -- each node's ribbons are renormalised to fill its "
            "height, so the thinnest ribbon may come out thinner than asked. "
            "The size key describes nodes for the same reason.", stacklevel=2)

    pt_per_unit = (_points_per_unit(ax, layout, r_in, r_out)
                   if (min_flow_width or min_node_size or want_legend) else 0.0)

    min_width_axis = 0.0
    if min_flow_width:
        if pt_per_unit <= 0:
            raise ValueError(
                "sankey: the axes has no drawable extent, so min_flow_width "
                "cannot be converted to a width; pass a figsize.")
        min_width_axis = float(min_flow_width) / pt_per_unit
        avail_min = min(avail_l, avail_r)
        if n_flows and n_flows * min_width_axis >= avail_min:
            afford = avail_min * pt_per_unit / n_flows
            msg = (f"sankey: min_flow_width={min_flow_width:g} pt cannot be "
                   f"met. This diagram has {n_flows} flows and can give every "
                   f"flow at most {afford:.2g} pt. A baseline costs "
                   f"n_flows * min_flow_width, so what limits it is how many "
                   f"flows there are, not how big they are -- enlarge figsize "
                   f"or lower min_flow_width.")
            if flow_scale == "linear":
                msg += (" A non-linear flow_scale ('sqrt', or 'log' for the "
                        "widest ranges) lifts the thin end without spending "
                        "height per flow.")
            raise ValueError(msg)

    m_l, b_l = _solve_flow_affine(fx_sum, fx_min, n_flows, avail_l, min_width_axis)
    m_r, b_r = _solve_flow_affine(fx_sum, fx_min, n_flows, avail_r, min_width_axis)
    w_l = np.where(nz, m_l * fx + b_l, 0.0)
    w_r = np.where(nz, m_r * fx + b_r, 0.0)

    # ---- sizes -> node heights -------------------------------------------
    # Solved on the same rail and with the same affine solver as the ribbons,
    # then each node's ribbons are rescaled to fill it. Done after the ribbons
    # rather than instead of them, because the *split* is by ribbon count and
    # the two rails have to agree flow by flow.
    sizes_l = ctv.sum(axis=1)
    sizes_r = ctv.sum(axis=0)
    h_l = h_r = None
    if node_non_proportional:
        min_node_axis = 0.0
        if min_node_size:
            if pt_per_unit <= 0:
                raise ValueError(
                    "sankey: the axes has no drawable extent, so min_node_size "
                    "cannot be converted to a height; pass a figsize.")
            min_node_axis = float(min_node_size) / pt_per_unit
            n_nodes = max(int((sizes_l > 0).sum()), int((sizes_r > 0).sum()))
            avail_min = min(avail_l, avail_r)
            if n_nodes and n_nodes * min_node_axis >= avail_min:
                afford = avail_min * pt_per_unit / n_nodes
                raise ValueError(
                    f"sankey: min_node_size={min_node_size:g} pt cannot be "
                    f"met. This diagram has {n_nodes} categories on its "
                    f"longest rail and can give every node at most "
                    f"{afford:.2g} pt. A floor costs n_nodes * min_node_size, "
                    f"so what limits it is how many categories there are, not "
                    f"how big they are -- enlarge figsize, lower "
                    f"min_node_size, or use node_scale='sqrt', which lifts "
                    f"the small end without spending height per node.")
        h_l = _solve_node_heights(sizes_l, avail_l, min_node_axis, node_scale)
        h_r = _solve_node_heights(sizes_r, avail_r, min_node_axis, node_scale)
        w_l = _split_node_height(w_l, h_l, axis=1)
        w_r = _split_node_height(w_r, h_r, axis=0)

    left_pos = _compute_positions(left_cats, dict(zip(left_cats, w_l.sum(axis=1))))
    right_pos = _compute_positions(right_cats, dict(zip(right_cats, w_r.sum(axis=0))))

    def _draw_rail(cats, pos, rail, colors):
        for cat in cats:
            t0, t1 = pos[cat]
            if layout == "circular":
                arc_t = np.linspace(t0, t1, 24)
                inner = [_rail_to_xy(rail, t, layout, r_in) for t in arc_t]
                outer = [_rail_to_xy(rail, t, layout, r_out) for t in arc_t[::-1]]
                verts = inner + outer + [inner[0]]
                codes = ([MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2)
                         + [MplPath.CLOSEPOLY])
                ax.add_patch(mpatches.PathPatch(
                    MplPath(verts, codes), facecolor=colors[cat],
                    edgecolor='white', linewidth=0.5, zorder=5))
                lx, ly = _rail_to_xy(rail, (t0 + t1) / 2, layout, r_out + 0.06)
                # On the left half the tangent angle points backwards, so the
                # label reads right-to-left and its anchor sits on the wrong
                # end -- flip it and anchor from the other side.
                deg = np.degrees(np.arctan2(ly, lx))
                flip = deg > 90 or deg < -90
                ax.text(lx, ly, cat,
                        ha='right' if flip else 'left', va='center',
                        fontsize=fontsize,
                        rotation=deg + 180 if flip else deg,
                        rotation_mode='anchor')
                continue

            (xa, ya) = _rail_to_xy(rail, t0, layout)
            (xb, yb) = _rail_to_xy(rail, t1, layout)
            if layout == "vertical":
                ax.add_patch(mpatches.Rectangle(
                    (xa, ya - node_width / 2), xb - xa, node_width,
                    facecolor=colors[cat], edgecolor='white', linewidth=0.5,
                    zorder=5))
                off = -0.02 if rail == 1 else 0.02
                ax.text((xa + xb) / 2, ya + off + (node_width / 2) * (1 if rail == 0 else -1),
                        cat, ha='center',
                        va='bottom' if rail == 0 else 'top',
                        fontsize=fontsize, rotation=90 if len(cats) > 12 else 0)
            else:
                # ya > yb now that the rail runs downward, so anchor at the
                # lower corner and give a positive height: a Rectangle with a
                # negative height still draws, which is how this would have
                # gone unnoticed outside a test that measures it.
                ax.add_patch(mpatches.Rectangle(
                    (xa - node_width / 2, min(ya, yb)), node_width, abs(yb - ya),
                    facecolor=colors[cat], edgecolor='white', linewidth=0.5,
                    zorder=5))
                side = -1 if rail == 0 else 1
                ax.text(xa + side * (node_width / 2 + 0.02), (ya + yb) / 2, cat,
                        ha='right' if rail == 0 else 'left', va='center',
                        fontsize=fontsize)

    _draw_rail(left_cats, left_pos, 0, left_colors)
    _draw_rail(right_cats, right_pos, 1, right_colors)

    # Draw flows
    left_cursor = {c: left_pos[c][0] for c in left_cats}
    right_cursor = {c: right_pos[c][0] for c in right_cats}
    for i, lc in enumerate(left_cats):
        for j, rc in enumerate(right_cats):
            if not nz[i, j]:
                continue
            h_left = w_l[i, j]
            h_right = w_r[i, j]

            # Rail space: t grows downward from the first category. The blocks
            # are drawn through _rail_to_xy, which flips it because matplotlib's
            # y grows upward. The ribbons were built from the raw cursor and so
            # were laid out in the opposite vertical order from the blocks they
            # were meant to join -- the flow with the largest category's height
            # sitting where the smallest category's block was. Flip here too,
            # from the one place that knows the convention.
            t0_top = left_cursor[lc]
            t0_bot = left_cursor[lc] + h_left
            t1_top = right_cursor[rc]
            t1_bot = right_cursor[rc] + h_right

            if layout == "circular":
                path = _circular_ribbon(t0_top, t0_bot, t1_top, t1_bot, r_in)
            else:
                # Built in rail space, then mapped -- the same way the blocks
                # are placed, so the two cannot disagree about which end is up.
                rail = _bezier_path(
                    x_left + node_width / 2, t0_top, t0_bot,
                    x_right - node_width / 2, t1_top, t1_bot
                )
                path = MplPath([_rail_pt_to_xy(x, t, layout)
                                for x, t in rail.vertices], rail.codes)
            color = left_colors[lc] if color_by == 'left' else right_colors[rc]
            patch = mpatches.PathPatch(path, facecolor=color, alpha=alpha,
                                       edgecolor='none', zorder=2)
            ax.add_patch(patch)

            left_cursor[lc] += h_left
            right_cursor[rc] += h_right

    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # Column labels
    if layout == "circular":
        ax.text(-(r_out + 0.30), 0, left, ha='center', va='center',
                fontsize=fontsize + 1, fontweight='bold', rotation=90)
        ax.text(r_out + 0.30, 0, right, ha='center', va='center',
                fontsize=fontsize + 1, fontweight='bold', rotation=-90)
    elif layout == "vertical":
        ax.text(-0.03, 1.0, left, ha='right', va='center',
                fontsize=fontsize + 1, fontweight='bold')
        ax.text(-0.03, 0.0, right, ha='right', va='center',
                fontsize=fontsize + 1, fontweight='bold')
    else:
        ax.text(x_left, -0.03, left, ha='center', va='top',
                fontsize=fontsize + 1, fontweight='bold')
        ax.text(x_right, -0.03, right, ha='center', va='top',
                fontsize=fontsize + 1, fontweight='bold')

    _figure.layout(fig, quiet=False)

    # tight_layout resizes the axes to fit the labels, so the conversion above
    # is an estimate. An axis-off subplot normally expands, making ribbons
    # thicker than asked, but long category labels can push the other way -- and
    # a min_flow_width that quietly went unmet is worse than not having one.
    if min_flow_width and pt_per_unit > 0 and not node_non_proportional:
        realised = min_width_axis * _points_per_unit(ax, layout, r_in, r_out)
        if realised < float(min_flow_width) * 0.95:
            warnings.warn(
                f"sankey: the thinnest ribbon came out {realised:.2g} pt, short "
                f"of the {min_flow_width:g} pt asked for -- the axes shrank to "
                f"fit the category labels. Enlarge figsize or shorten the "
                f"labels.", stacklevel=2)
    # Same check for the node floor, and the same asymmetry: coming out taller
    # than asked is fine (an axis-off subplot usually expands), coming out
    # shorter is the promise going unmet, which is worse than not making it.
    if min_node_size and pt_per_unit > 0 and h_l is not None:
        realised = float(min(h_l[h_l > 0].min(), h_r[h_r > 0].min()))
        realised *= _points_per_unit(ax, layout, r_in, r_out)
        if realised < float(min_node_size) * 0.95:
            warnings.warn(
                f"sankey: the shortest node came out {realised:.2g} pt, short "
                f"of the {min_node_size:g} pt asked for -- the axes shrank to "
                f"fit the category labels. Enlarge figsize or shorten the "
                f"labels.", stacklevel=2)

    # Drawn after tight_layout, not before. The key sits inside the axes, but
    # with clip_on=False tight_layout counts it in the tight bbox and shrinks
    # the axes to make room -- by half, measured -- which silently undoes the
    # min_flow_width it was drawn to explain. Laying it out last also lets it
    # use the settled scale rather than the estimate.
    if want_legend and n_flows:
        pt_per_unit = _points_per_unit(ax, layout, r_in, r_out)
        # One key, for whichever map is still global. Sampled and drawn off the
        # LEFT rail in both cases: the two rails carry different gap totals, so
        # a key measured from the other one would be off by that difference.
        bits = []
        if legend_kind == "node":
            if node_scale != "linear":
                bits.append({"sqrt": "square-root", "log": "log1p"}[node_scale]
                            + " scale")
            if min_node_size:
                bits.append(f"{min_node_size:g} pt floor")
            pos_l = sizes_l[sizes_l > 0]
            samples = _legend_sample_counts(pos_l.min(), pos_l.max())
            # Interpolated through the solved map rather than read off a node,
            # so a round sample count gets its true height even when no
            # category happens to have exactly that many cells.
            g = _NODE_FUNCS[node_scale](np.asarray(samples, dtype=float))
            gp = _NODE_FUNCS[node_scale](pos_l.astype(float))
            slope = ((h_l[sizes_l > 0].max() - h_l[sizes_l > 0].min())
                     / (gp.max() - gp.min())) if gp.max() > gp.min() else 0.0
            base = h_l[sizes_l > 0].min() - slope * gp.min()
            thicknesses = [float(slope * gi + base) for gi in g]
            title = "cells per node"
        else:
            if flow_scale != "linear":
                bits.append({"sqrt": "square-root", "log": "log1p"}[flow_scale]
                            + " scale")
            if min_flow_width:
                bits.append(f"{min_flow_width:g} pt baseline")
            samples = _legend_sample_counts(ctv[nz].min(), ctv[nz].max())
            thicknesses = [m_l * _FLOW_FUNCS[flow_scale](float(c)) + b_l
                           for c in samples]
            title = "cells per flow"
        _draw_size_legend(
            ax, samples, thicknesses,
            layout, r_in, pt_per_unit, size_legend_loc, fontsize, '0.45',
            title + (f" ({', '.join(bits)})" if bits else ""),
            *_rail_span_fraction(ax, layout, node_width))


    out = _figure.finish(fig, created=_created, show=show, save=save, writekey='sankey')
    if return_fig:
        return fig, ax
    return out


@wraps(sankey)
def plotSankey(*args, **kwargs):
    """Alias for :func:`sankey`."""
    return sankey(*args, **kwargs)
