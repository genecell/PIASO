from anndata import AnnData
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



from ._group_order import natural_key
from ..utils._cytome_compat import is_cytome_input as _is_cytome_input  # backward-compat re-export
from ..utils._cytome_compat import open_cytome as _open_cytome


# Canonical legend_loc values shared by plotEmbedding and plot_embeddings_split.
# 'right'   -> legend box / shared legend in the right margin
# 'on_data' -> category labels drawn at cluster centroids (note the underscore)
# 'none'    -> no legend
#: ``'both'`` draws the labels on the clusters *and* the legend beside the
#: panel. While you are still working out what a cluster is, you want the label
#: where the points are; to read off a colour you want the list. Neither alone
#: does both jobs, and drawing them twice was the workaround.
_VALID_LEGEND_LOC = ("right", "on_data", "both", "none")


def _is_numeric_dtype(values) -> bool:
    """Is this column continuous?

    ``np.issubdtype(values.dtype, np.number)`` answered this until pandas 3,
    which stores an object string column as ``StringDtype`` -- an extension
    dtype numpy cannot interpret, so the check *raised* rather than returning
    False and every categorical embedding crashed. pandas' own predicate
    understands both its extension dtypes and plain numpy ones.

    Integer columns stay continuous on purpose: ``n_counts`` is a metric, not a
    category. **Booleans do not**, and that is the one place the two libraries
    disagree: ``np.issubdtype(bool, np.number)`` is False while pandas'
    predicate says True, so switching to it wholesale would have turned every
    True/False mask into a colour bar.
    """
    from pandas.api.types import is_bool_dtype as _pd_is_bool
    from pandas.api.types import is_numeric_dtype as _pd_is_numeric

    dtype = getattr(values, "dtype", None)
    if dtype is None:
        return False
    try:
        if _pd_is_bool(dtype):
            return False
        return bool(_pd_is_numeric(dtype))
    except TypeError:                       # a dtype neither library knows
        return False


def _style_for_values(values, palette=None, category_order=None, na_color="lightgray"):
    """A CategoryStyle for bare values (no stored column), honouring a caller's
    palette and order the way ``_render_embedding`` always has: a list palette
    is positional over ``category_order`` followed by the other values in
    natural order."""
    from ._categories import resolve_category_style, label_strings
    if category_order:
        lab = label_strings(values)
        present = {v for v in lab if v is not None}
        cats, seen = [], set()
        for c in [str(c) for c in category_order]:
            if c not in seen:
                seen.add(c)
                cats.append(c)
        cats += sorted(present - seen, key=natural_key)
        values = pd.Categorical(lab, categories=cats)
    return resolve_category_style(values=values, palette=palette, na_color=na_color,
                                  read_stored=False)


def _render_embedding(ax, coords, values, is_cat, *, color="value", palette=None,
                      point_size=1.0, alpha=1.0, rasterized=True, groups=None,
                      na_color="lightgray", na_label="other",
                      legend_loc="right", legend_ncol=None,
                      legend_marker_size=None, legend_fontsize=10,
                      legend_fontoutline=None, legend_fontbackground="white",
                      legend_fontbackground_alpha=0.7,
                      cmap=None, vmin=None, vmax=None,
                      add_colorbar=True, category_order=None, strict_groups=True,
                      linewidths=0.0, edgecolors="none",
                      style=None, sort_order="category", vcenter=None):
    """Render one embedding scatter (categorical or continuous) into ``ax`` from
    plain arrays — no AnnData / cytome dependency. Shared by ``plotEmbedding`` and
    ``plot_embeddings_split`` so the split needs no proxy-AnnData round-trip.

    **Colours come from** ``style``, a :class:`~piaso.plotting._categories.CategoryStyle`
    the caller resolved over the *whole* column (stored order and colours,
    by name). The values drawn here may be a subset -- a ``cell_mask``, one
    panel of a split -- and that subset must not move a colour: positional
    colouring over the categories present here drew Inh in Exc's colour as
    soon as a mask left Exc out. Without ``style`` (a direct call), one is
    built from these values, ``palette`` (list or dict) and ``category_order``.

    **Missing labels** (None, NaN, the text ``'nan'`` a store holds for a NaN)
    are drawn first, behind everything, in ``na_color``, with a legend entry
    ``"NA"``; they used to become a ``'nan'`` category with a palette colour.
    Continuous NaN values are drawn in ``na_color`` too; matplotlib maps them
    to a transparent colour, which hid them.

    ``sort_order``: ``'category'`` (default) draws the categories in display
    order, each over the ones before; ``'size'`` draws the largest first, so a
    rare type ends on top (100 astrocytes among 20,000 neurons had no visible
    pixel in category order); ``'random'`` draws every cell in one shuffled,
    seeded pass, so no category covers another systematically.

    ``linewidths`` / ``edgecolors`` default to **no stroke**. Matplotlib
    otherwise strokes every marker at ``rcParams['patch.linewidth']`` (1.0 pt)
    in ``edgecolors='face'``, which for cell-sized points is most of the dot:
    ``point_size=0.2`` is a disc 0.45 pt across inside a 1.0 pt stroke, so the
    marker draws **3.2x** the requested size, and with ``alpha < 1`` the stroke
    and the fill composite twice around the rim — the ringed, washed-out dots
    reported at 600 dpi. A stroke is a no-op on a 20 pt marker and a
    misrepresentation on a 0.5 pt one, so it is off unless asked for.

    Returns the continuous mappable (for a shared colorbar) or None.
    """
    from . import color as _color_mod
    from ._categories import label_strings, draw_sequence, NA_LABEL, SORT_ORDERS
    if sort_order is None:
        sort_order = "category"
    if sort_order not in SORT_ORDERS:
        raise ValueError(f"sort_order={sort_order!r}: choose from {SORT_ORDERS}")
    scatter_kw = dict(s=point_size, alpha=alpha, rasterized=rasterized,
                      linewidths=linewidths, edgecolors=edgecolors)
    if is_cat:
        if style is None:
            style = _style_for_values(values, palette, category_order, na_color)
        labels = label_strings(values)
        missing = np.asarray(pd.isna(labels), dtype=bool)
        # natural_key, not plain sorted, decides any category the stored
        # order does not: Leiden labels are strings, so a lexicographic sort
        # gives 0, 1, 10, 11, 2 in the legend while the Sankey beside it --
        # which resolves order through resolve_group_order -- gives 0, 1, 2.
        categories = style.present(labels)
        codes = style.codes(labels)
        index = {c: i for i, c in enumerate(style.order)}

        plot_categories = categories
        # Whether anything was pushed into the background. Only then does the
        # legend need to say what the grey means.
        greyed_out = False
        groups_list = None
        if groups is not None:
            groups_list = [str(g) for g in (
                groups if isinstance(groups, (list, tuple, set)) else [groups])]
            # Validated against the whole column, not the cells drawn here: a
            # group a cell_mask or a split panel leaves out is not a typo,
            # just nothing to highlight in this view.
            unknown = [g for g in groups_list if g not in index]
            if unknown and strict_groups:
                raise ValueError(
                    f"plotEmbedding(): groups={unknown} not found among the "
                    f"categories of color={color!r}. Available: {style.order}."
                )
            plot_categories = [c for c in categories if c in groups_list]

        if missing.any():
            # labelled, so a figure legend collected from the panels
            # (split_embedding) lists it too; the axes legend below is built
            # from its own handles either way
            ax.scatter(coords[missing, 0], coords[missing, 1], c=[style.na_color],
                       label=NA_LABEL, **scatter_kw)
        if groups_list is not None:
            keep = np.isin(codes, [index[c] for c in plot_categories])
            bg = ~missing & ~keep
            greyed_out = bool(bg.any())
            if greyed_out:
                ax.scatter(coords[bg, 0], coords[bg, 1], c=na_color,
                           label='_nolegend_', **scatter_kw)

        if sort_order == "random":
            sel = np.flatnonzero(np.isin(codes, [index[c] for c in plot_categories]))
            np.random.default_rng(0).shuffle(sel)
            ax.scatter(coords[sel, 0], coords[sel, 1], c=style.rgba(labels[sel]),
                       label='_nolegend_', **scatter_kw)
            # one labelled, empty artist per category, so a figure legend that
            # collects labels from the axes (split_embedding) still finds them
            for cat in plot_categories:
                ax.scatter([], [], c=[style.color(cat)], label=cat, **scatter_kw)
        else:
            for cat in draw_sequence(style, labels, sort_order, categories=plot_categories):
                mask = codes == index[cat]
                ax.scatter(coords[mask, 0], coords[mask, 1], c=[style.color(cat)],
                           label=cat, **scatter_kw)

        if legend_loc in ("right", "both"):
            from matplotlib.lines import Line2D
            # Fixed, readable legend-dot size INDEPENDENT of point_size (Round 26):
            # proxy handles drawn at a constant markersize so the legend marker no
            # longer shrinks/grows when the data point_size changes.
            _ms = legend_marker_size if legend_marker_size is not None else 6.0
            # Built from the categories actually drawn, not from every category
            # in the column. With groups= the rest of the data is a single grey
            # cloud, so a legend listing all forty original labels in their full
            # palette claims colours that are nowhere on the plot -- and omits
            # the one colour that is.
            def _handle(colour, label):
                return Line2D([0], [0], marker='o', linestyle='None', markersize=_ms,
                              markerfacecolor=colour, markeredgecolor='none', label=label)
            _handles = [_handle(style.color(cat), cat) for cat in plot_categories]
            if missing.any():
                _handles.append(_handle(style.na_color, NA_LABEL))
            if greyed_out:
                _handles.append(_handle(na_color, na_label))
            _ncol = legend_ncol if legend_ncol is not None else max(1, -(-len(_handles) // 12))
            ax.legend(
                handles=_handles, bbox_to_anchor=(1.05, 1), loc="upper left",
                fontsize=legend_fontsize, frameon=False, ncol=_ncol,
                handletextpad=0.5, columnspacing=1.2, labelspacing=0.4,
            )
        if legend_loc in ("on_data", "both"):
            for cat in plot_categories:
                mask = codes == index[cat]
                cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
                text_kwargs = dict(
                    fontsize=legend_fontsize - 1, ha="center", va="center",
                    fontweight="bold",
                )
                # A label sitting on its own cluster is unreadable against the
                # cluster colour, so it gets a background patch. The colour and
                # opacity are arguments because the useful value depends on the
                # palette: the default white at 0.7 disappears against white
                # figure background, which is exactly where a sparse cluster's
                # label lands.
                if legend_fontbackground is not None:
                    text_kwargs["bbox"] = dict(
                        boxstyle="round,pad=0.2", fc=legend_fontbackground,
                        ec="none", alpha=legend_fontbackground_alpha,
                    )
                if legend_fontoutline is not None and legend_fontoutline > 0:
                    import matplotlib.patheffects as patheffects
                    text_kwargs['path_effects'] = [
                        patheffects.withStroke(linewidth=legend_fontoutline, foreground='white')
                    ]
                ax.text(cx, cy, cat, **text_kwargs)
        return None
    else:
        if cmap is None:
            cmap = _color_mod.c_color1
        vals = pd.to_numeric(pd.Series(np.asarray(values, dtype=object).ravel()),
                             errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(vals)
        if (~finite).any():
            ax.scatter(coords[~finite, 0], coords[~finite, 1], c=[na_color],
                       label='_nolegend_', **scatter_kw)
        idx = np.flatnonzero(finite)
        idx = idx[np.argsort(vals[idx], kind="stable")]
        if idx.size == 0 and vmin is None and vmax is None:
            vmin, vmax = 0.0, 1.0           # nothing finite: an honest empty scale
        norm_kw = dict(vmin=vmin, vmax=vmax)
        if vcenter is not None:
            # a diverging map centred on a value (0 for a signed score): the
            # two sides are stretched separately, so the centre colour sits
            # at vcenter whatever the range on either side
            from matplotlib.colors import TwoSlopeNorm
            lo = float(vmin) if vmin is not None else float(np.nanmin(vals[idx]))
            hi = float(vmax) if vmax is not None else float(np.nanmax(vals[idx]))
            if not lo < float(vcenter) < hi:
                raise ValueError(f"vcenter={vcenter} must lie between the colour "
                                 f"limits ({lo:g}, {hi:g}).")
            norm_kw = dict(norm=TwoSlopeNorm(vcenter=float(vcenter), vmin=lo, vmax=hi))
        sc = ax.scatter(coords[idx, 0], coords[idx, 1], c=vals[idx], cmap=cmap,
                        **norm_kw, **scatter_kw)
        if add_colorbar:
            ax.figure.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        return sc


def _resolve_cytome_basis(ds, basis, modality=None, max_rank=2):
    """Resolve a basis name against a cytome dataset's embeddings.

    Tries exact match first, then a substring lookup so callers can write
    ``basis='X_umap'`` even when the embedding is registered as
    ``'RNA_obsm_X_umap'`` or ``'RNA_umap'``.

    A multiome store holds one embedding per modality (``RNA_umap`` and
    ``ATAC_umap``), and ``'X_umap'`` matches both equally well. ``modality``
    breaks the tie; without it the shorter name wins as before, and a warning
    names the alternatives -- the choice used to be silent, so an ATAC panel
    could be drawn on the RNA layout without a word.

    ``max_rank=1`` accepts only a name that is the key behind a prefix
    (``RNA_pca``, ``RNA_obsm_X_pca``), not one that merely contains it.
    """
    emb_names = ds.list_embeddings()
    if basis in emb_names:
        return basis
    keyword = basis.lower().replace("x_", "")
    # Rank candidates instead of taking the last substring hit. Plain
    # `in` made basis='spatial' resolve to 'spatial_aligned' as soon as an
    # aligned copy existed -- so adding an embedding silently changed what an
    # existing call meant. Prefer the name that IS the key (possibly behind a
    # modality prefix) over one that merely contains it.
    def _rank(name):
        low = name.lower()
        stem = low.split("_", 1)[1] if "_" in low else low
        if low == keyword or stem == keyword:
            return 0                       # 'spatial' / 'RNA_spatial'
        if low.endswith("_" + keyword):
            return 1                       # 'RNA_obsm_spatial'
        return 2                           # 'spatial_aligned', 'spatial_chip'

    def _prefix(name):
        return name.split("_", 1)[0].lower() if "_" in name else ""

    matches = [n for n in emb_names if keyword in n.lower() and _rank(n) <= max_rank]
    if not matches:
        raise KeyError(f"Embedding '{basis}' not found. Available: {emb_names}")
    mod = str(modality).lower() if modality else None
    matches.sort(key=lambda n: (0 if mod and _prefix(n) == mod else 1, _rank(n), len(n)))
    best = matches[0]
    if not (mod and _prefix(best) == mod):
        rivals = [n for n in matches[1:] if _rank(n) == _rank(best)
                  and _prefix(n) != _prefix(best) and _prefix(n)]
        if rivals:
            import warnings
            warnings.warn(
                f"basis={basis!r} matches {[best] + rivals} equally; using "
                f"{best!r}. Pass the full name (basis={rivals[0]!r}) or "
                f"modality= to choose.", UserWarning, stacklevel=3)
    return best


def _cytome_extract(source, basis, obs_columns=None, modality=None):
    """Extract embedding coords + requested obs columns from a cytome source.

    Reads via the public ``ds.list_embeddings`` / ``ds.cells[col]`` API, no
    raw SQL and no proxy AnnData. Accepts either a Dataset or a string path.

    Returns
    -------
    coords : ndarray (n, 2)
    obs_df : pandas.DataFrame
        DataFrame indexed 0..n-1 with one column per name in ``obs_columns``
        that exists in ``ds.cells``. Missing columns are silently skipped
        (so that ``color`` can be a gene name handled by the caller).
    palette_map : dict
        Maps ``f'{col}_colors'`` -> palette list for any obs_columns that
        have a stored palette in ``ds.metadata``. Mirrors AnnData's
        ``adata.uns[f'{col}_colors']`` convention.
    """
    obs_columns = list(obs_columns or [])
    with _open_cytome(source, mode="r") as ds:
        emb_name = _resolve_cytome_basis(ds, basis, modality=modality)
        coords = np.asarray(ds.embeddings[emb_name])
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)

        obs_data = {}
        palette_map = {}
        for col in obs_columns:
            if col is None:
                continue
            if col in ds.cells:
                obs_data[col] = np.asarray(ds.cells[col])
                pkey = f'{col}_colors'
                pval = ds.metadata.get(pkey)
                if pval is not None:
                    palette_map[pkey] = list(pval)

    obs_df = pd.DataFrame(obs_data, index=range(coords.shape[0]))
    return coords, obs_df, palette_map


def _resolve_categorical_style(data, color, user_palette=None):
    """Resolve ``(palette, category_order)`` for a categorical colour.

    The older interface, kept for callers outside the plotters: the plots now
    take order and colours from
    :func:`piaso.plotting._categories.resolve_category_style`, which assigns
    colours by name. This reads the same stored style through the same reader
    (``stored_category_style``: a cytome's ``set_categories``, else the
    categorical order an AnnData carried into the store; AnnData's categorical
    dtype order and ``uns``), so the two cannot drift apart.

    Honors, in priority order:

    1. an explicit ``user_palette`` (list or dict) — returned as-is for colour,
       and, when it is a **dict**, its key order becomes the category order,
    2. the stored order and colours (a dict by name, or a list positional over
       the stored order),

    and returns ``(palette, category_order)`` where ``palette`` is a dict
    ``{cat: hex}``, a list, or None, and ``category_order`` is a list or None.

    ``user_palette`` may also be the NAME of a matplotlib colormap, which is
    sampled evenly across the categories in their resolved order -- useful
    when the categories are ordered (ages, timepoints) and a sequential ramp
    carries that order.
    """
    from ._categories import stored_category_style
    palette = user_palette
    user_dict_order = ([str(k) for k in user_palette]
                       if isinstance(user_palette, dict) else None)
    _cmap_palette = None
    if isinstance(palette, str):
        _cmap_palette = palette
        palette = None
    try:
        category_order, stored = stored_category_style(data, color)
    except Exception:
        category_order, stored = None, None
    if category_order is not None:
        category_order = [str(c) for c in category_order]
    if palette is None and stored is not None:
        palette = dict(stored) if isinstance(stored, dict) else list(stored)

    if _cmap_palette is not None:
        # Sample the colormap once per category, in the resolved order, so the
        # ramp runs along the categories rather than along whatever order the
        # renderer happens to see.
        try:
            import matplotlib as _mpl
            cats = category_order
            if cats is None:
                vals = (data.cells[color] if _is_cytome_input(data)
                        else data.obs[color])
                cats = sorted({str(v) for v in np.asarray(vals)})
            cm = _mpl.colormaps.get_cmap(_cmap_palette)
            n = max(len(cats), 1)
            palette = {str(c): _mpl.colors.to_hex(cm(i / max(n - 1, 1)))
                       for i, c in enumerate(cats)}
        except Exception:
            palette = None          # unknown colormap: fall back to the default
    if user_dict_order:
        category_order = user_dict_order
    return palette, category_order


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


#: Inches of clear space between one panel's legend and the next panel.
_LEGEND_GUTTER_IN = 0.25


def _reserve_legend_space(fig, gutter=_LEGEND_GUTTER_IN):
    """Widen a multi-column figure so each panel's legend fits beside it.

    Per-axes legends are anchored at axes coordinates ``(1.05, 1)`` — just
    outside the axes. Matplotlib's subplot grid does not know they are there,
    so in a multi-column figure panel N's legend is drawn across panel N+1's
    cell, and because N+1 is drawn later its opaque background paints over the
    labels. On a 27-category figure two legends came out interleaved and
    clipped.

    Shrinking the panels cannot fix it: a legend of 27 long labels is wider
    than the panel it belongs to, so there is no axes width that leaves room.
    The space has to be added rather than borrowed. This measures the widest
    legend in each column, grows the figure by exactly that much, and re-lays
    the panels at their original size — the panels do not shrink, the figure
    gets wider.

    Single-column figures are left untouched: their right margin is already
    free, and the legend cannot collide with anything.
    """
    axes_with_legend = [ax for ax in fig.axes if ax.get_legend() is not None]
    if not axes_with_legend:
        return

    def _spec(ax):
        getter = getattr(ax, "get_subplotspec", None)
        return getter() if getter is not None else None

    specs = {id(ax): _spec(ax) for ax in fig.axes}
    grids = {s.get_gridspec() for s in specs.values() if s is not None}
    if len(grids) != 1:
        return                          # hand-placed axes; not ours to move
    gs = grids.pop()
    if gs.ncols < 2:
        return                          # a stack has no right-hand neighbour

    fig.canvas.draw()          # legend extents are only known after a render
    fig_w, fig_h = fig.get_size_inches()
    inv = fig.transFigure.inverted()

    # Positions in inches, captured before the resize: `get_position` returns
    # figure *fractions*, whose meaning changes the moment the width does.
    geom = {}
    for ax in fig.axes:
        p = ax.get_position()
        geom[id(ax)] = (p.x0 * fig_w, p.y0, p.width * fig_w, p.height)

    legend_w = {}
    for ax in axes_with_legend:
        spec = specs[id(ax)]
        if spec is None:
            return
        col = spec.colspan.start
        w = ax.get_legend().get_window_extent().transformed(inv).width * fig_w
        legend_w[col] = max(legend_w.get(col, 0.0), w)
    if not legend_w:
        return

    panel_w = max(g[2] for g in geom.values())
    left = min(g[0] for g in geom.values())
    right = fig_w - max(g[0] + g[2] for g in geom.values())
    inner_gap = max(0.0, (fig_w - left - right - gs.ncols * panel_w)
                    / max(1, gs.ncols - 1))

    # One x offset per column: panel, then its legend, then the gutter.
    col_x, x = {}, left
    for c in range(gs.ncols):
        col_x[c] = x
        x += panel_w + legend_w.get(c, 0.0)
        if c < gs.ncols - 1:
            x += max(inner_gap, gutter)
    new_w = x + right
    if new_w <= fig_w + 1e-6:
        return

    fig.set_size_inches(new_w, fig_h)
    for ax in fig.axes:
        spec = specs[id(ax)]
        if spec is None:
            continue
        _, y0, _, h = geom[id(ax)]
        ax.set_position([col_x[spec.colspan.start] / new_w, y0,
                         panel_w / new_w, h])
        
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

def _create_global_legend(
    fig,
    axes,
    legend_loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon:bool=False,
    marker_size:float=6.0,
    max_rows_per_col:int=12,
    order=None,
    **legend_kwargs):
    """
    Collects unique legend entries from all subplots and creates a single global legend.

    Uses Line2D marker handles for consistent, controllable legend dot sizes
    regardless of scatter point size in the plot.

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
    marker_size : float, optional (default=6.0)
        Size of legend marker dots (diameter in points).
    max_rows_per_col : int, optional (default=12)
        Maximum number of rows before creating a new column in the legend.
    order : sequence of str, optional
        Order of the entries (labels not in it follow, as first seen). The
        panels are drawn one category at a time, so first-seen order is the
        order of the first panel, not of the column.
    **legend_kwargs : dict, optional
        Additional keyword arguments passed to `fig.legend()` for styling.

    Returns
    -------
    None
        Displays the figure with a global legend.
    """
    # Collect unique (label → color) from all visible subplots
    label_color = {}
    axes_iter = axes.flat if hasattr(axes, 'flat') else axes
    for ax in axes_iter:
        if not ax.get_visible():
            continue
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label in label_color:
                continue
            # Extract color from any handle type
            if hasattr(handle, 'get_facecolor'):
                fc = handle.get_facecolor()
                if hasattr(fc, '__len__') and len(fc) > 0:
                    label_color[label] = fc[0] if hasattr(fc[0], '__len__') else fc
                else:
                    label_color[label] = fc
            elif hasattr(handle, 'get_color'):
                label_color[label] = handle.get_color()
            else:
                label_color[label] = 'gray'

    if not label_color:
        from . import _figure
        _figure.layout(fig)
        return

    if order is not None:
        rank = {str(c): i for i, c in enumerate(order)}
        label_color = dict(sorted(label_color.items(),
                                  key=lambda kv: rank.get(kv[0], len(rank))))

    # Build Line2D handles with explicit, fixed marker size
    legend_handles = []
    legend_labels = []
    for label, color in label_color.items():
        h = mlines.Line2D([], [], marker='o', color='none',
                          markerfacecolor=color, markeredgecolor='none',
                          markersize=marker_size, linestyle='None')
        legend_handles.append(h)
        legend_labels.append(label)

    # Determine number of columns dynamically
    num_items = len(legend_labels)
    ncol = max(1, -(-num_items // max_rows_per_col))

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon,
        markerscale=1.0,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.35,
        ncol=ncol,
        **legend_kwargs
    )

    from . import _figure
    _figure.layout(fig)

    
### Plot embeddings side by side
import matplotlib.pyplot as plt
       
def plot_embeddings_split(data,
                          color,
                          splitby,
                          ncol:int=None,
                          dpi:int=80,
                          col_size:int=5,
                          row_size:int=5,
                          alpha:float=1.0,
                          vmax:float=None,
                          vmin:float=None,
                          vmin_pct:float=None,
                          vmax_pct:float=None,
                          show_figure:bool=True,
                          save:bool=None,
                          layer:str=None,
                          basis:str='X_umap',
                          fix_coordinate_ratio:bool=True,
                          show_axis_ticks:bool=False,
                          margin_ratio:float=0.05,
                          legend_fontsize:int=10,
                          legend_fontoutline:int=2,
                          legend_fontbackground="white",
                          legend_fontbackground_alpha:float=0.7,
                          legend_loc:str='right',
                          legend_marker_size: float=6.0,
                          groups=None,
                          point_size: float=None,
                          palette=None,
                          cmap=None,
                          frameon:bool=False,
                          rasterized:bool=True,
                          modality:str=None,
                          cytome_layer:str="counts",
                          compute_on_fly:bool=True,
                          use_cached_stats:bool=True,
                          show_modality_in_title:bool=False,
                          x_min=None,
                          x_max=None,
                          y_min=None,
                          y_max=None,
                          image=False,
                          img_key="hires",
                          image_alpha=1.0,
                          linewidths: float = 0.0,
                          edgecolors="none",
                          sort_order="category",
                          na_color="lightgray",
                          vcenter=None,
                          **kwargs):
    """
    Plot cell embeddings side by side based on a categorical variable.

    The plots are split by a specified categorical variable, with each unique category producing a separate subplot.
    Data points in each subplot are colored according to the `color` variable.

    Supports AnnData, cytome Dataset, or path to .cytome file.

    Parameters
    ----------
    data : AnnData, cytome.Dataset, or str
        An AnnData object, cytome Dataset, or path to .cytome file.
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
        Whether to display the figure after plotting. When it is not shown the
        figure is returned, and closed only if it was saved.
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
    legend_fontsize: int, optional (default=10)
        Font size in pt.
    legend_fontoutline: int, optional (default=2)
        Line width of the legend font outline in pt. 
    legend_fontbackground: str or None, optional (default='white')
        Colour of the patch behind each ``'on_data'`` label; None for no patch.
    legend_fontbackground_alpha: float, optional (default=0.7)
        Opacity of that patch.
    legend_loc: str, optional (default='right')
        Where to put the category labels. One of:

        - ``'right'`` — a single shared legend in the figure's right margin.
        - ``'on_data'`` — labels drawn at each cluster's centroid, in every panel.
        - ``'both'`` — centroid labels *and* the shared right-margin legend.
          Use this while you are still working out what a cluster is: the
          on-data label tells you which cluster you are looking at, the margin
          legend lets you read a colour off the list. Neither alone does both.
        - ``'none'`` — no legend.

        Any other value raises ``ValueError`` rather than silently drawing
        nothing.
    legend_marker_size: float, optional (default=6.0)
        Legend dot size. In the right-margin legend it is the marker scale
        (relative to the data points); in the global/multi-panel legend it is
        the absolute marker size in points. ``None`` auto-sizes from the data
        ``point_size`` (capped to avoid oversized dots on large datasets).
    x_min : float or None, optional (default=None)
        Minimum limit for the x-axis. If None, the limit is computed automatically based on the data.
    x_max : float or None, optional (default=None)
        Maximum limit for the x-axis. If None, the limit is computed automatically based on the data.
    y_min : float or None, optional (default=None)
        Minimum limit for the y-axis. If None, the limit is computed automatically based on the data.
    y_max : float or None, optional (default=None)
        Maximum limit for the y-axis. If None, the limit is computed automatically based on the data.
    point_size : float, optional
        Scatter point size. An explicit value always overrides the auto-size.
        If None, auto-scaled (``max(0.1, min(3, 22000 / n_cells))``,
        clamped to [0.1, 8]). Accepts ``size=`` as an alias for
        scanpy-style call sites.
    palette : list[str] or dict, optional
        Categorical palette. If None, falls back to
        ``adata.uns['{color}_colors']`` (or the cytome metadata
        equivalent), then to the PIASO default ``d_color4``. Mapping is
        held consistent across panels so the same category gets the same
        colour in every subplot.
    cmap : str or Colormap, optional
        Colourmap for numeric ``color`` values. Forwarded to each panel.
    frameon : bool, optional (default False)
        Whether to show axis spines on each panel and on the global
        legend frame. Mirrors ``piaso.pl.plotEmbedding(frameon=...)``.
    rasterized : bool, optional (default True)
        Forward to per-panel scatter for compact vector output.
    linewidths : float, optional (default 0)
        Marker stroke width, in points. **Zero by default.** Matplotlib
        otherwise strokes every point at ``rcParams['patch.linewidth']``
        (1.0 pt) in ``edgecolors='face'``, which for cell-sized markers is
        most of the dot: ``point_size=0.2`` is a disc 0.45 pt across inside a
        1.0 pt stroke, so it draws 3.2x the size asked for, and under
        ``alpha < 1`` the stroke and fill composite twice around the rim —
        ringed dots with a washed-out core, visible once a PDF is enlarged.
        Pass a positive value to outline markers deliberately.

        *Changed in 1.2.4*: was matplotlib's 1.0 pt default.
    edgecolors : color or 'none', optional (default 'none')
        Marker stroke colour. Only meaningful with ``linewidths > 0``.

        *Changed in 1.2.4*: was matplotlib's ``'face'``.
    sort_order : {'category', 'size', 'random'}, default ``'category'``
        Drawing order of a categorical colour within each panel; see
        :func:`plotEmbedding`. ``'size'`` puts rare types on top.
    na_color : str, default ``'lightgray'``
        Colour of cells whose value is missing (and of cells left out by
        ``groups``); a categorical colour lists missing values as ``"NA"``.
    vcenter : float, optional
        Centre of a diverging map for a continuous colour; see
        :func:`plotEmbedding`. Shared by every panel.
    **kwargs : dict
        Accepts the aliases ``ncols`` (→ ``ncol``), ``size``
        (→ ``point_size``) and ``show`` (→ ``show_figure``). Anything else
        raises ``TypeError``.

        *Changed in 1.2.4*: unrecognised keywords used to be accepted and
        silently discarded, so a caller styling the panels got no error and no
        effect — and ``show=False``, which every other ``piaso.pl`` function
        understands, displayed the figure regardless.

    image : bool or str, default False
        Draw the library's tissue image under the spots (``basis='spatial'``
        workflows). ``True`` auto-selects when exactly one library has an
        image (or the plotted cells span one) and raises naming the options
        otherwise; a string selects that library. Sources: AnnData
        ``uns['spatial']`` or a cytome's stored images (cytome ≥ 0.2.6). The
        y-axis follows the image convention (top of the tissue at the top)
        only when an image is drawn.
    img_key : str, default 'hires'
        Which image of the library to draw ('hires', 'lowres', ...).
    image_alpha : float, default 1.0
        Opacity of the tissue image.
    alpha : float, default 1.0
        Point opacity.
    vmin_pct, vmax_pct : float, optional
        Colour-scale limits as percentiles of the values, when ``vmin`` /
        ``vmax`` are not given.
    groups : sequence of str, optional
        For a categorical ``color``: the categories to highlight in every
        panel; the other cells are drawn grey (``na_color``), as in
        :func:`plotEmbedding`. ``None`` highlights all. It does not choose
        ``splitby`` panels (the docstring said it did), and a numeric
        ``color`` has no categories to highlight, so there it warns and is
        ignored.
    show_modality_in_title : bool, default False
        Cytome only: prefix panel titles with the modality.
    modality : str, optional
        Cytome only: the modality whose features to read (``'RNA'``,
        ``'ATAC'``, ``'tiles'``, ``'GA'``). ``None`` finds the modality that
        holds the feature, and raises if several do.
    cytome_layer : str, default 'counts'
        Cytome only: the layer to read the values from. The default is the
        raw counts -- unlike AnnData's ``.X``, which is usually normalised, so
        the same data can read differently; pass ``'log1p'`` or ``'infog'``
        (RNA) or ``'tfidf'`` (ATAC) for normalised values.
    compute_on_fly : bool, default True
        Cytome only: when ``cytome_layer`` names a normalised layer the store
        does not hold (``'log1p'``, ``'infog'``, ``'tfidf'``), compute it per
        feature from the counts. It does not normalise the default
        ``'counts'``.
    use_cached_stats : bool, default True
        Cytome only: reuse normalisation statistics already stored in the
        cytome's metadata rather than recomputing them.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure when it is not shown (``show_figure=False``), else None.

    Examples
    --------
    >>> import anndata
    >>> import piaso
    >>> adata = anndata.read_h5ad('pbmc3k.h5ad')  # Load an example dataset
    >>> # Plot embeddings colored by a gene expression value and split by clusters
    >>> piaso.pl.plot_embeddings_split(adata, color='CDK9', splitby='louvain', col_size=6, row_size=6)
    >>> # Save the figure to a file
    >>> piaso.pl.plot_embeddings_split(adata, color='CDK9', splitby='louvain', save='./CST3_embeddingsSplit.pdf')
    """
    # --- 0. scanpy-style kwarg aliases (be lenient at the call site) ---
    if 'ncols' in kwargs and ncol is None:
        ncol = kwargs.pop('ncols')
    elif 'ncols' in kwargs:
        kwargs.pop('ncols')
    if 'size' in kwargs and point_size is None:
        point_size = kwargs.pop('size')
    elif 'size' in kwargs:
        kwargs.pop('size')
    # `show` is what every other piaso.pl function calls this, and it was being
    # dropped here -- so `show=False` displayed the figure anyway. Accept it.
    if 'show' in kwargs:
        _show = kwargs.pop('show')
        if show_figure is True:            # untouched default, so honour the alias
            show_figure = _show
    # Anything left was accepted and then silently dropped -- the shape of bug
    # that already cost this function its vmin_pct/vmax_pct (see below), and
    # cost `show=False` its meaning. Raise as plotEmbedding does, so a caller
    # learns at the call site instead of from a figure that looks almost right.
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "plot_embeddings_split() got unexpected keyword argument(s): "
            f"{unexpected}"
        )

    from . import color as _color_mod

    # The modality (if any) carrying the user's `color` feature, used to
    # optionally enrich panel titles. Set by the cytome path; stays None
    # for AnnData and obs-column colours.
    resolved_modality_for_title = None

    # --- 1. Unified data extraction (cytome / AnnData) ---
    if _is_cytome_input(data):
        coords, obs_df, palette_map = _cytome_extract(
            data, basis=basis,
            obs_columns=[splitby] + ([] if _is_value_array(color) else [color])
            + ["barcode"], modality=modality,
        )
        if splitby not in obs_df.columns:
            with _open_cytome(data, mode="r") as _ds:
                avail = sorted(_ds.cells.columns)
            raise ValueError(
                f"The splitby key '{splitby}' was not found in cytome cells "
                f"table. Available: {avail}"
            )
        splitby_series = obs_df[splitby]
        if _is_value_array(color):
            full_color_data = _as_value_array(
                color, coords.shape[0],
                names=(obs_df["barcode"].to_numpy() if "barcode" in obs_df.columns
                       else None))
            is_numeric = _is_numeric_dtype(full_color_data)
        elif color in obs_df.columns:
            full_color_data = obs_df[color].values
            is_numeric = _is_numeric_dtype(obs_df[color])
        else:
            with _open_cytome(data, mode="r") as _ds:
                full_color_data, resolved_modality_for_title = (
                    _resolve_cytome_feature_values(
                        _ds, color,
                        modality=modality,
                        cytome_layer=(cytome_layer if cytome_layer != "counts"
                                      else (layer or cytome_layer)),
                        compute_on_fly=compute_on_fly,
                        use_cached_stats=use_cached_stats,
                    )
                )
            is_numeric = True
    else:
        adata_input = data
        if splitby not in adata_input.obs.columns:
            raise ValueError(
                f"The splitby key '{splitby}' was not found in adata.obs."
            )
        coords = np.asarray(adata_input.obsm[basis])
        splitby_series = adata_input.obs[splitby]
        if _is_value_array(color):
            full_color_data = _as_value_array(
                color, coords.shape[0], names=np.asarray(adata_input.obs_names))
            is_numeric = _is_numeric_dtype(full_color_data)
        elif color in adata_input.obs.columns:
            full_color_data = adata_input.obs[color].values
            is_numeric = _is_numeric_dtype(adata_input.obs[color])
        else:
            full_color_data = adata_input.obs_vector(color, layer=layer)
            is_numeric = True
        palette_map = {
            k: list(v) for k, v in (adata_input.uns or {}).items()
            if k.endswith('_colors')
        }

    # --- 2. Resolve splitby categories ---
    # A cytome column arrives as plain objects, so an ordering stored with
    # ``ds.set_categories`` would otherwise be lost and the panels fall back to
    # alphabetical -- which puts E10.5 before E9.5 and reads as a bug.
    _stored_order = None
    if _is_cytome_input(data):
        try:
            with _open_cytome(data, mode="r") as _ds:
                _cats = _ds.get_categories(splitby)
                if isinstance(_cats, dict):        # {'order': [...], ...}
                    _cats = _cats.get("order")
                _stored_order = list(_cats) if _cats else None
        except Exception:
            _stored_order = None
    if _stored_order:
        present = set(pd.unique(splitby_series.values))
        variables = [c for c in _stored_order if c in present]
        variables += [c for c in pd.unique(splitby_series.values)
                      if c not in set(_stored_order) and not pd.isnull(c)]
    elif isinstance(splitby_series.dtype, pd.CategoricalDtype):
        variables = list(splitby_series.cat.categories)
    elif pd.api.types.is_float_dtype(splitby_series):
        raise ValueError(
            f"The column '{splitby}' is float (continuous). "
            "Cannot split plots by a continuous variable."
        )
    elif pd.api.types.is_integer_dtype(splitby_series):
        unique_vals = pd.unique(splitby_series.values)
        n_unique = len(unique_vals)
        if n_unique > 50:
            raise ValueError(
                f"The integer column '{splitby}' has {n_unique} unique values. "
                "Splitting by this variable would generate too many subplots "
                "(looks like continuous data). If you really intend to create "
                f"{n_unique} plots, convert it explicitly to categorical."
            )
        variables = list(np.sort(unique_vals))
        print(f"Note: '{splitby}' is integer type. Treating as {n_unique} discrete categories.")
    else:
        unique_vals = pd.unique(splitby_series.values)
        unique_vals = unique_vals[~pd.isnull(unique_vals)]
        try:
            unique_vals = np.sort(unique_vals)
        except Exception:
            pass
        variables = list(unique_vals)
        if len(variables) > 50:
            print(
                f"Warning: Variable '{splitby}' will generate {len(variables)} "
                "subplots. This may be slow."
            )

    if groups is not None and is_numeric:
        import warnings
        warnings.warn("plot_embeddings_split: groups= highlights categories of a "
                      "categorical color; a numeric color has none, so it is ignored.",
                      UserWarning, stacklevel=2)

    # --- 3. Auto point_size (matches plotEmbedding's formula) ---
    if point_size is None:
        n_cells = coords.shape[0]
        point_size = max(0.1, min(3, 22000 / n_cells))

    # --- 4. Subplot grid ---
    fig, axs, nrow, ncol = _build_subplots(
        len(variables), ncol=ncol, dpi=dpi, col_size=col_size, row_size=row_size
    )
    axs = [axs] if not isinstance(axs, np.ndarray) else axs.ravel()

    # --- 5. Pre-compute uniform axis + colour limits ---
    if all(v is not None for v in [x_min, y_min, x_max, y_max]):
        xy_min = np.array([x_min, y_min])
        xy_max = np.array([x_max, y_max])
    else:
        xy_min = np.nanmin(coords[:, :2], axis=0)
        xy_max = np.nanmax(coords[:, :2], axis=0)
    xy_margin = (xy_max - xy_min) * margin_ratio

    expr_min, expr_max = None, None
    if is_numeric:
        try:
            # Percentile limits, matching plotEmbedding. These were reaching
            # this function through **kwargs and being silently dropped, so a
            # caller asking for 10/90 got the full range and no error --
            # exactly the shape of bug that survives review, because the plot
            # still renders and only the contrast is wrong.
            _fcd = np.asarray(full_color_data, dtype=float)
            _fcd = _fcd[np.isfinite(_fcd)]
            if vmax is None and vmax_pct is not None and _fcd.size:
                vmax = float(np.percentile(_fcd, vmax_pct))
            if vmin is None and vmin_pct is not None and _fcd.size:
                vmin = float(np.percentile(_fcd, vmin_pct))
            expr_max = vmax if vmax is not None else float(np.nanmax(full_color_data))
            expr_min = vmin if vmin is not None else float(np.nanmin(full_color_data))
        except (ValueError, TypeError):
            expr_min, expr_max = vmin, vmax

    # --- 6. Resolve a GLOBAL category→colour mapping for categorical color ---
    # Honors the cytome ``set_categories`` store (order + colors) / AnnData
    # categorical dtype + uns, via the shared resolver.
    # An array passed as `color` is not the name of anything, so titles and
    # stored-palette lookups need a label instead.
    _color_name = _value_array_name(color) if _is_value_array(color) else color

    style = None
    if not is_numeric:
        from ._categories import resolve_category_style
        if _is_value_array(color):
            _order = None
            if isinstance(color, pd.Categorical):
                _order = [str(c) for c in color.categories]
            elif isinstance(color, pd.Series) and isinstance(
                    getattr(color, "dtype", None), pd.CategoricalDtype):
                _order = [str(c) for c in color.cat.categories]
            style = _style_for_values(full_color_data, palette, _order, na_color)
        else:
            # Over the whole column, by name: every panel gets the same
            # category -> colour map whatever it happens to contain.
            style = resolve_category_style(data, color, full_color_data,
                                           palette=palette, na_color=na_color)

    # --- 7. Per-panel: build a tiny proxy AnnData and delegate to plotEmbedding ---
    splitby_arr = np.asarray(splitby_series.values)
    if legend_loc is None:
        legend_loc = "none"
    if legend_loc not in _VALID_LEGEND_LOC:
        raise ValueError(
            f"Invalid legend_loc={legend_loc!r}. Choose from "
            f"{sorted(_VALID_LEGEND_LOC)} (note the underscore in 'on_data')."
        )
    # per-panel labels only for 'on_data'; 'right' -> one shared global legend
    # (drawn below); 'none' -> no legend at all.
    panel_legend = 'on_data' if legend_loc in ('on_data', 'both') else 'none'

    import anndata as _ad
    # Auto point_size once (consistent dot size across panels); an explicit
    # point_size= always overrides.
    _panel_point_size = (point_size if point_size is not None
                         else max(0.1, min(3, 22000 / coords.shape[0])))

    # ---- optional tissue image per panel ----
    # A panel gets an image only when its cells resolve to ONE library:
    # per-panel when splitting by the library column, the single library
    # otherwise. Panels that mix libraries draw no image (overlaying several
    # tissues in one panel is not meaningful) — warned once, not fatal.
    _spatial_store = {}
    _lib_for_all = None
    _warned_mixed = [False]
    if image:
        from ._spatial_image import _spatial_uns_from
        _spatial_store = _spatial_uns_from(data)
        if not _spatial_store:
            import warnings as _w
            _w.warn("image= requested but no spatial images found")
            image = False
        elif isinstance(image, str):
            if image not in _spatial_store:
                raise KeyError(f"library {image!r} has no stored image; "
                               f"available: {sorted(_spatial_store)}")
            _lib_for_all = image
        elif len(_spatial_store) == 1:
            _lib_for_all = next(iter(_spatial_store))

    def _panel_image_ctx(panel_categories_value, mask):
        if not image:
            return None
        from ._spatial_image import _resolve_spatial_image
        lib = _lib_for_all
        if lib is None:
            # splitting by the library column itself?
            if str(panel_categories_value) in _spatial_store:
                lib = str(panel_categories_value)
            else:
                if not _warned_mixed[0]:
                    import warnings as _w
                    _w.warn(
                        f"image=True with {len(_spatial_store)} libraries: "
                        f"panels that do not correspond to a single library "
                        f"are drawn without an image. Split by the library "
                        f"column or pass image='<library_id>'.")
                    _warned_mixed[0] = True
                return None
        return _resolve_spatial_image(data, lib, img_key)

    for i in range(len(axs)):
        if i >= len(variables):
            axs[i].set_visible(False)
            continue
        category = variables[i]
        mask = (splitby_arr == category)
        if int(np.sum(mask)) == 0:
            axs[i].set_visible(False)
            continue

        # Every panel gets the global style (category -> colour by name,
        # resolved once above). A list here was built in sorted order while
        # the renderer indexed it in the column's category order, so wherever
        # the two orders differed the colours landed on the wrong categories.

        # Inject the resolved modality into per-panel titles when requested.
        if show_modality_in_title and resolved_modality_for_title:
            panel_title = f"{_color_name} ({resolved_modality_for_title}) in\n{category}"
        else:
            panel_title = f"{_color_name} in\n{category}"

        _img_ctx = _panel_image_ctx(category, mask)
        if _img_ctx is not None:
            from ._spatial_image import _draw_image_overlay
            _draw_image_overlay(axs[i], _img_ctx, image_alpha)

        # Render natively from arrays — no proxy AnnData round-trip.
        _render_embedding(
            axs[i],
            np.asarray(coords[mask], dtype=np.float64),
            np.asarray(full_color_data)[mask],
            (not is_numeric),
            color=_color_name, style=style, sort_order=sort_order,
            na_color=na_color,
            point_size=_panel_point_size,
            alpha=alpha, rasterized=rasterized, legend_loc=panel_legend,
            # For a continuous colour the colorbar is the legend, so
            # legend_loc='none' has to reach it -- it did not, and asking for
            # no legend still got one per panel.
            add_colorbar=(legend_loc != "none"),
            # The order is resolved ONCE above, across every panel, and was
            # then not handed to the renderer -- which re-sorted each panel
            # alphabetically, so a dict palette's order (and a stored
            # categorical order) survived into the colours but not into the
            # legend. Panels must also agree with each other.
            legend_fontbackground=legend_fontbackground,
            legend_fontbackground_alpha=legend_fontbackground_alpha,
            groups=groups, strict_groups=False,
            cmap=cmap, vmin=expr_min, vmax=expr_max, vcenter=vcenter,
            legend_fontsize=legend_fontsize, legend_fontoutline=legend_fontoutline,
            linewidths=linewidths, edgecolors=edgecolors,
        )
        if not frameon:
            for spine in axs[i].spines.values():
                spine.set_visible(False)
        axs[i].set_title(panel_title)

        # --- per-panel polish that plotEmbedding doesn't expose ---
        axs[i].set_xlim(xy_min[0] - xy_margin[0], xy_max[0] + xy_margin[0])
        if _img_ctx is not None:
            # image convention: y increases downward
            axs[i].set_ylim(xy_max[1] + xy_margin[1], xy_min[1] - xy_margin[1])
        else:
            axs[i].set_ylim(xy_min[1] - xy_margin[1], xy_max[1] + xy_margin[1])
        # Both branches must act. Only the False branch did, so the documented
        # default (True) left matplotlib's 'auto' in place and every panel was
        # stretched to its subplot box -- which on spatial coordinates makes
        # the tissue look squashed.
        axs[i].set_aspect('equal' if fix_coordinate_ratio else 'auto')

    # --- 8. Global legend coalesces labels from all panels ---
    if not is_numeric and legend_loc in ('right', 'both'):
        # Each panel has scatter handles with `label=cat` attached but no
        # per-panel legend rendered. _create_global_legend de-duplicates
        # labels across panels and draws one figure-level legend.
        from ._categories import NA_LABEL
        _create_global_legend(
            fig, axs, legend_loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=frameon, marker_size=legend_marker_size,
            fontsize=legend_fontsize,
            max_rows_per_col=max(8, int(fig.get_figheight() * 1.8)),
            order=(list(style.order) + [NA_LABEL]) if style is not None else None,
        )

    # --- 9. Final polish ---
    # Tick visibility is controlled by show_axis_ticks ONLY (independent of
    # frameon, which controls the spines). Default show_axis_ticks=False must
    # actually hide ticks + labels on every panel — mirror plotEmbedding's
    # else-branch. Pre-fix this had no else, so the default leaked matplotlib's
    # visible ticks through.
    if show_axis_ticks:
        for ax in axs:
            ax.grid(False)
            ax.tick_params(labelbottom=True, labelleft=True, length=4)
            ax.set_xticks(np.arange(
                xy_min[0] - xy_margin[0], xy_max[0] + xy_margin[0],
                (xy_max[0] - xy_min[0]) / 4,
            ))
            ax.set_yticks(np.arange(
                xy_min[1] - xy_margin[1], xy_max[1] + xy_margin[1],
                (xy_max[1] - xy_min[1]) / 4,
            ))
    else:
        for ax in axs:
            ax.grid(False)
            ax.tick_params(labelbottom=False, labelleft=False, length=0)

    from . import _figure
    return _figure.finish(fig, created=True, show=bool(show_figure), save=save,
                          writekey='plotEmbeddingsSplit')

        
from functools import wraps
# Create the alias
@wraps(plot_embeddings_split)
def plotEmbeddingsSplit(*args, **kwargs):
    """
    Alias for :func:`plot_embeddings_split`.
    
    Please refer to the main function for full documentation.
    """
    return plot_embeddings_split(*args, **kwargs)


# ---------------------------------------------------------------------------
# Cytome-native plotEmbedding / plotUMAP
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def _is_value_array(color):
    """True when ``color`` is the values themselves rather than a name.

    A list or tuple is excluded on purpose: those already mean "one panel per
    colour", and that is a far more common intent than a per-cell Python list.
    Pass a numpy array or a pandas Series to colour by values.
    """
    return isinstance(color, (np.ndarray, pd.Series, pd.Categorical, pd.Index))


def _n_cells_of(data):
    """How many cells ``data`` has, or None when that needs opening a file."""
    for attr in ("n_obs", "n_cells"):
        n = getattr(data, attr, None)
        if isinstance(n, (int, np.integer)):
            return int(n)
    return None


def _keys_from_array(color, n_cells):
    """An array of *names* handed to ``color=`` -- ``names['Right'].values[:10]``,
    a ``var_names`` slice, a pandas Index of genes -- means one panel per name,
    exactly as a list does. It is told apart from per-cell values by length:
    a string array with one entry per cell colours by those values; any other
    length is a list of keys. Returns the list, or None when ``color`` is not
    such an array."""
    if not isinstance(color, (np.ndarray, pd.Index, pd.Series)):
        return None
    arr = np.asarray(color)
    if arr.ndim != 1 or arr.dtype.kind not in "OUS" or arr.size == 0:
        return None
    if n_cells is not None and arr.size == n_cells:
        return None
    # a fixed-width string array is all strings by construction; only an
    # object array needs the walk (a Series of mixed values, a None in it)
    if arr.dtype.kind == "O" and not all(isinstance(x, str) for x in arr.tolist()):
        return None
    return arr.tolist()


def _check_keys_resolve(keys, data, n_cells):
    """When an array was read as names, say so in the error if none of them is
    a column or a feature: a per-cell array of the wrong length would otherwise
    surface as "'x' not found", two calls deep, with no mention of the length."""
    obs = getattr(data, "obs", None)
    var_names = getattr(data, "var_names", None)
    if obs is None or var_names is None:
        return
    known = set(map(str, obs.columns)) | set(map(str, var_names))
    if any(k in known for k in keys):
        return
    shown = ", ".join(map(repr, keys[:5])) + (", ..." if len(keys) > 5 else "")
    raise KeyError(
        f"color= was an array of {len(keys)} strings, read as a list of keys "
        f"(one panel per name) because the data has {n_cells} cells; none of them "
        f"is an .obs column or a feature: {shown}. Pass per-cell values as an "
        f"array with one entry per cell, or keys that exist in the data.")


def _value_array_name(color, default="value"):
    """A label for an array passed as ``color`` -- a Series' name, or nothing."""
    name = getattr(color, "name", None)
    return str(name) if name is not None else default


def _align_series(color, names, what="color"):
    """A Series passed as ``color``, in the data's cell order.

    Five cases, decided by the Series' index against the data's cell names:

    * the names are not unique (AnnData allows it; a cytome's ``barcode`` is
      not unique across samples imported without a prefix) -- alignment is
      ambiguous, so use position and say so;
    * a ``RangeIndex`` -- position, which is what a Series built from an
      array means;
    * every label resolves to a cell name -- reindex to the data's order;
      anything that came out of a ``groupby``, a ``sort`` or a ``.loc`` is
      then coloured correctly instead of silently permuted;
    * no label resolves -- position;
    * some resolve and some do not -- raise, naming the count, because
      neither reading is right.
    """
    import warnings
    idx = color.index
    if names is None or isinstance(idx, pd.RangeIndex):
        return color.to_numpy()
    names = pd.Index(names)
    if not names.is_unique:
        warnings.warn(
            f"{what}= is a Series but the cell names are not unique, so it "
            f"cannot be aligned by name; using it in positional order.",
            stacklevel=4)
        return color.to_numpy()
    hit = idx.isin(names)
    n_hit = int(hit.sum())
    if n_hit == 0:
        return color.to_numpy()
    if n_hit < len(idx):
        raise ValueError(
            f"{what}= is a Series whose index matches {n_hit} of its "
            f"{len(idx)} labels to cell names and leaves {len(idx) - n_hit} "
            f"unmatched; neither positional nor name-aligned use is safe. "
            f"Reindex it to the cells first.")
    if not idx.is_unique:
        raise ValueError(f"{what}= is a Series with duplicate index labels.")
    if len(idx) != len(names):
        raise ValueError(
            f"{what}= names {len(idx)} cells but the data has {len(names)}; "
            f"a subset cannot be aligned. Reindex it to the cells first.")
    return color.reindex(names).to_numpy()


def _as_value_array(color, n_cells, what="color", names=None):
    """Validate an array passed as ``color`` and return it as a flat ndarray,
    in the data's cell order (a Series is aligned by name when it can be)."""
    if isinstance(color, pd.Series):
        values = _align_series(color, names, what)
    elif isinstance(color, pd.Categorical):
        values = np.asarray(color)
    else:
        values = np.asarray(color)
    if values.ndim != 1:
        values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(
            f"{what}= was given an array of shape {np.shape(color)}; colouring "
            f"needs one value per cell.")
    if values.shape[0] != n_cells:
        raise ValueError(
            f"{what}= has {values.shape[0]} values but the data has {n_cells} "
            f"cells. Pass one value per cell, in the same order.")
    return values


def _prefetch_cytome_features(data, names, layer=None, modality=None,
                              cytome_layer="counts", compute_on_fly=True,
                              use_cached_stats=True):
    """``{feature: (values, modality)}`` for the features among ``names``, read
    in ONE pass over the store's matrix, for a grid of panels.

    Each panel used to read its own feature, so a grid of N genes streamed the
    matrix N times. Cells columns are left to the panels (they are cheap), and
    when the batch reader cannot serve a feature -- it has no fallback for a
    store without ``{modality}_counts``, which the single-feature reader reads
    as ``{modality}_data`` with a warning -- nothing is prefetched and the
    panels read one by one, as before.
    """
    if layer is not None and cytome_layer == "counts":
        cytome_layer = layer            # the legacy mapping of _get_embedding_and_color
    with _open_cytome(data, mode="r") as ds:
        cells = set(ds.cells.columns)
        feats = [n for n in dict.fromkeys(names)
                 if isinstance(n, str) and n not in cells]
        if len(feats) < 2:
            return {}
        try:
            return _resolve_cytome_feature_values_batch(
                ds, feats, modality=modality, cytome_layer=cytome_layer,
                compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats)
        except Exception:
            return {}


def _get_embedding_and_color(
    data, basis, color, layer=None,
    modality=None, cytome_layer="counts",
    compute_on_fly=True, use_cached_stats=True, prefetched=None,
):
    """Extract embedding coords and color values from cytome Dataset, path,
    or AnnData.

    Parameters
    ----------
    data : AnnData, cytome.Dataset, or str
        Path strings ending in ``.cytome``/``.db`` are opened, read from,
        and closed by this helper.
    basis : str
        Embedding key.
    color : str
        obs column or gene/feature name.
    layer : str, optional
        AnnData layer to read gene expression from. For cytome, prefer
        ``cytome_layer``; if ``layer`` is given (and ``cytome_layer`` is
        the default ``"counts"``), it is honoured as the cytome layer
        for backward compatibility.
    modality : str, optional
        Cytome modality (``"RNA"``/``"GA"``/``"ATAC"``/``"tiles"``); auto-
        detect if None. AnnData inputs are unaffected.
    cytome_layer : str
        Cytome layer to read for feature values (default ``"counts"``).
        Combined with ``modality`` to produce the matrix name
        ``{modality}_{cytome_layer}``.
    compute_on_fly : bool
        If True (default), compute log1p / infog / tfidf per-feature
        when the matrix isn't materialised; uses (and populates)
        per-modality cached stats in ``ds.metadata``.
    use_cached_stats : bool
        If True (default), reuse cached per-modality params from
        ``ds.metadata``; if False, ignore cache and recompute fresh.

    Returns
    -------
    coords : ndarray (n, 2)
    values : ndarray (n,)
    is_categorical : bool
    resolved_modality : str or None
        For cytome feature lookups, the modality that actually carried
        the feature (so callers can render it in titles). ``None`` for
        obs columns and AnnData inputs.
    """
    # Backward-compat: legacy `layer` kwarg maps to cytome_layer for cytome inputs.
    if _is_cytome_input(data) and layer is not None and cytome_layer == "counts":
        cytome_layer = layer
    # `color` may be the values themselves rather than the name of a column.
    # Handled before anything tries to look the name up: a numpy array reached
    # `color in data.obs.columns` and came back as
    # "TypeError: unhashable type: 'numpy.ndarray'", which says nothing about
    # what was wrong or what to do.
    if _is_value_array(color):
        if _is_cytome_input(data):
            coords, _obs_df, _pal = _cytome_extract(data, basis=basis,
                                                    obs_columns=["barcode"],
                                                    modality=modality)
            _names = (_obs_df["barcode"].to_numpy() if "barcode" in _obs_df.columns
                      else None)
        else:
            coords = np.asarray(data.obsm[basis])
            _names = np.asarray(data.obs_names)
        values = _as_value_array(color, coords.shape[0], names=_names)
        return coords, values, not _is_numeric_dtype(values), None, True

    if _is_cytome_input(data) and prefetched is not None:
        # a grid's panel: the feature was read with the others in one pass
        coords, _obs, _pal = _cytome_extract(data, basis=basis, obs_columns=[],
                                             modality=modality)
        values, resolved = prefetched
        return coords, values, False, resolved, False

    if _is_cytome_input(data):
        coords, obs_df, _palette_map = _cytome_extract(
            data, basis=basis, obs_columns=[color], modality=modality,
        )
        if color in obs_df.columns:
            values = obs_df[color].values
            # Continuous = any numeric dtype (int OR float). Only object/string/
            # bool/category columns (e.g. Leiden, CellTypes) are categorical.
            # (Was `np.floating`, which mis-flagged INTEGER metrics like
            # n_fragments / n_counts as categorical.)
            is_cat = not _is_numeric_dtype(values)
            return coords, values, is_cat, None, True
        # color is not an obs column — resolve via the modality registry.
        with _open_cytome(data, mode="r") as ds:
            values, resolved = _resolve_cytome_feature_values(
                ds, color,
                modality=modality, cytome_layer=cytome_layer,
                compute_on_fly=compute_on_fly,
                use_cached_stats=use_cached_stats,
            )
        return coords, values, False, resolved, False

    # AnnData path
    coords = np.asarray(data.obsm[basis])
    if color in data.obs.columns:
        values = np.asarray(data.obs[color])
        # numeric (int/float) → continuous; object/str/bool/category → categorical
        is_cat = not _is_numeric_dtype(values)
    else:
        values = np.asarray(data.obs_vector(color, layer=layer))
        is_cat = False
        return coords, values, is_cat, None, False
    return coords, values, is_cat, None, True


# --------------------------------------------------------------------------
# Modality registry — single source of truth for the (modality → entity, id_cols)
# mapping used by the feature resolver. Order matters for auto-detect:
# RNA-first matches the typical workflow where gene names are the canonical
# colour-by, then GA (gene-named pseudo-counts), then ATAC (peak strings),
# then tiles (binned-coord strings). Open for proteins later.
# --------------------------------------------------------------------------
# Modality registry sourced from cytome's public API, resolved lazily.
#
# Two separate things, deliberately not conflated:
#
# * cytome is a *required* dependency (see pyproject.toml), so this import
#   cannot fail on a correct install.
# * It is still function-local, because `import piaso` should not pay to parse
#   cytome for the AnnData path that never touches it. Every other cytome
#   import in the package is function-local too; 1.2.0 briefly had this one at
#   module scope and that made `import piaso` fail outright when cytome was
#   merely an extra. See piaso/tests/test_declared_dependencies.py, which pins
#   both properties.
#
# The names come from `cytome` directly rather than `cytome.utils.modality`:
# the top level is cytome's documented, conformance-tested surface, so the
# 4-tuple registry shape and these signatures cannot change under us without
# cytome's own CI failing first.
_CYTOME_HINT = (
    "Reading features from a cytome dataset needs the `cytome` package, which "
    "PIASO requires.\n"
    "    pip install -U 'cytome>=0.2.3'\n"
    "If you see this, the install is incomplete -- cytome is a hard dependency."
)


def _cytome_modality():
    """Return the cytome module, or raise with an actionable message."""
    try:
        import cytome
    except ImportError as exc:  # pragma: no cover - only on a broken install
        raise ImportError(_CYTOME_HINT) from exc
    return cytome


def _plot_modality_registry():
    """``(modality, entity, id_cols)`` triples re-derived from cytome's registry.

    The plot side has always unpacked 3-tuples while cytome's canonical
    registry carries 4, so the projection lives here and call sites are
    unchanged. Order matters for auto-detect: RNA first (gene names are the
    typical colour-by), then GA, ATAC, tiles.
    """
    return [
        (mod, entity, id_cols)
        for mod, entity, _idx, id_cols in _cytome_modality().MODALITY_REGISTRY
    ]


def _modality_has_feature(ds, mod_name, entity, id_cols, feature):
    """Wrapper around cytome's helper preserving the plot-side calling
    convention (entity / id_cols already destructured by caller). Always
    delegates to the cytome registry under the hood."""
    return _cytome_modality().modality_has_feature(ds, mod_name, feature)


def _read_feature_column(ds, modality, layer_name, feat_idx, batch_size=2048):
    """Delegates to ``cytome.read_feature_column``."""
    return _cytome_modality().read_feature_column(
        ds, modality, layer_name, feat_idx, batch_size=batch_size,
    )


def _read_feature_columns_cytome(ds, modality, layer_name, feat_idx, batch_size=2048):
    """Delegates to ``cytome.read_feature_columns``."""
    return _cytome_modality().read_feature_columns(
        ds, modality, layer_name, feat_idx, batch_size,
    )


def _modality_cell_depth(ds, modality, use_cached_stats=True, batch_size=2048):
    """Delegates to ``cytome.modality_cell_depth``."""
    return _cytome_modality().modality_cell_depth(
        ds, modality, use_cached_stats=use_cached_stats, batch_size=batch_size,
    )





# Moved to piaso/tools/_normalize_resolve.py. Imported back under the old
# names because COSG imports `_ensure_infog_params` from this module by path,
# and tidying PIASO's namespace is not a reason to break a downstream package.
from ..tools._normalize_resolve import (  # noqa: F401
    _cached_stat_is_fresh, _ensure_infog_params, _ensure_tfidf_params,
    _params_feature_len_matches_modality, _tfidf_idf_matches_modality,
)



def _resolve_cytome_feature_values(
    ds, feature,
    modality=None,
    cytome_layer="counts",
    compute_on_fly=True,
    use_cached_stats=True,
    batch_size=2048,
):
    """Read a single feature (gene / peak / tile) as a per-cell vector
    from a cytome dataset, with explicit modality + layer routing.

    Parameters
    ----------
    ds : cytome.Dataset
    feature : str
        Feature name — gene_id, peak_id, tile_id, etc.
    modality : str, optional
        ``"RNA"`` / ``"GA"`` / ``"ATAC"`` / ``"tiles"``. If None, auto-detect:
        the registry is iterated and the function looks for the feature in
        each modality's entity table. Raises ``ValueError`` if the feature
        is found in MORE THAN ONE modality (caller must disambiguate).
    cytome_layer : str
        Matrix suffix to read. ``"counts"`` (default), ``"log1p"``, ``"infog"``,
        ``"tfidf"``, or any custom layer name. The matrix read is
        ``{modality}_{cytome_layer}``.
    compute_on_fly : bool
        When True (default) AND the requested ``{modality}_{cytome_layer}``
        is NOT materialised in the cytome, compute the per-feature value on
        the fly from ``{modality}_counts`` using the per-modality cached
        stats (or compute + cache them on first call). Supported on-the-fly
        layers: ``log1p``, ``infog``, ``tfidf``. For unknown layers the
        function falls back to the strict behaviour (raise actionable error
        if the matrix is missing).
        When False, the matrix MUST be materialised; missing matrix raises.
    use_cached_stats : bool
        When True (default), per-modality cached stats (``{modality}_cell_depth``,
        ``{modality}_infog_params``, ``{modality}_tfidf_params``) are read
        from ``ds.metadata`` if available. When False, the stats are
        recomputed fresh and the cache is overwritten (useful when the
        matrix has changed structurally e.g. after a subset).
    batch_size : int
        Chunk size for streaming reads/computations.

    Returns
    -------
    values : ndarray (n_cells,)
        Per-cell values for the requested feature × layer.
    resolved_modality : str
        The modality that was actually used (useful for downstream title
        injection / error messages).

    Raises
    ------
    ValueError
        - Feature is ambiguous (present in multiple modalities) AND
          ``modality`` is None.
        - User specified ``modality`` but feature is not in that
          modality's entity table.
        - Required layer is missing AND ``compute_on_fly=False`` (or the
          layer doesn't have an on-the-fly path).
    KeyError
        Feature not found in any modality.
    """
    # ------------------------------------------------------------------
    # 1. Determine candidate modalities.
    # ------------------------------------------------------------------
    registry = _plot_modality_registry()
    if modality is None:
        # Search all registry entries.
        candidates = []
        for mod, entity, id_cols in registry:
            hit = _modality_has_feature(ds, mod, entity, id_cols, feature)
            if hit is not None:
                feat_idx, _ = hit
                candidates.append((mod, entity, feat_idx))
        if not candidates:
            tried = [m for m, _, _ in registry]
            raise KeyError(
                f"Feature '{feature}' not found in any modality. "
                f"Modalities checked: {tried}."
            )
        if len(candidates) > 1:
            mods_with = [m for m, _, _ in candidates]
            raise ValueError(
                f"Feature '{feature}' is ambiguous — present in modalities: "
                f"{mods_with}. Specify modality= explicitly to disambiguate "
                f"(e.g. modality='{mods_with[0]}')."
            )
        resolved_modality, _, feat_idx = candidates[0]
    else:
        # Find the user-specified modality in the registry.
        match = next(
            ((m, e, ic) for m, e, ic in registry if m == modality),
            None,
        )
        if match is None:
            raise ValueError(
                f"Unknown modality '{modality}' for feature resolution. "
                f"Known: {[m for m, _, _ in registry]}."
            )
        _, entity, id_cols = match
        hit = _modality_has_feature(ds, modality, entity, id_cols, feature)
        if hit is None:
            # Look in OTHER modalities so the error message is helpful.
            other_hits = []
            for m, e, ic in registry:
                if m == modality:
                    continue
                if _modality_has_feature(ds, m, e, ic, feature) is not None:
                    other_hits.append(m)
            hint = (
                f" Found in: {other_hits}. Pass modality='{other_hits[0]}'."
                if other_hits else ""
            )
            raise ValueError(
                f"Feature '{feature}' not in modality '{modality}'.{hint}"
            )
        feat_idx, _ = hit
        resolved_modality = modality

    # ------------------------------------------------------------------
    # 2. Try to read from the materialised matrix first.
    # ------------------------------------------------------------------
    matrix_name = f"{resolved_modality}_{cytome_layer}"
    matrix_present = ds.matrix_meta(matrix_name) is not None

    if matrix_present:
        values = _read_feature_column(
            ds, resolved_modality, cytome_layer, feat_idx, batch_size,
        )
        return values, resolved_modality

    # ------------------------------------------------------------------
    # 3. Matrix missing — on-the-fly compute, if eligible.
    # ------------------------------------------------------------------
    if not compute_on_fly:
        raise ValueError(
            f"Matrix '{matrix_name}' not found in cytome and compute_on_fly=False. "
            f"Either materialise it (e.g. piaso.pp.normalize_log1p(ds, modality='{resolved_modality}', save_layer=True)) "
            f"or pass compute_on_fly=True to compute it per-feature."
        )

    counts_name = f"{resolved_modality}_counts"
    if ds.matrix_meta(counts_name) is None:
        # cytome >= 0.3.0 only writes `{modality}_counts` for raw integer
        # counts, so a file converted from normalised data has none. For
        # PLOTTING that is not fatal -- the values the user wants to see are in
        # the matrix that was adata.X. Use it, and say so. (Analysis functions
        # that genuinely need counts, INFOG above all, still refuse.)
        _main = f"{resolved_modality}_data"
        if ds.matrix_meta(_main) is not None:
            import warnings
            warnings.warn(
                f"'{counts_name}' is absent -- cytome >= 0.3.0 reserves that "
                f"name for raw integer counts. Reading '{_main}', the matrix "
                f"that was adata.X, as given. Pass cytome_layer='data' to "
                f"silence this, or re-convert with counts_layer= if you meant "
                f"to normalise from counts.",
                UserWarning, stacklevel=3,
            )
            return (_read_feature_column(
                ds, resolved_modality, "data", feat_idx, batch_size),
                resolved_modality)
        _present = [r[0] for r in ds._conn.execute(
            "SELECT matrix_name FROM matrix_meta").fetchall()]
        raise ValueError(
            f"Cannot compute '{cytome_layer}' on-the-fly: required source "
            f"matrix '{counts_name}' is missing from the cytome. "
            f"Available: {_present}."
        )

    # Read the per-feature counts column once.
    counts_col = _read_feature_column(
        ds, resolved_modality, "counts", feat_idx, batch_size,
    )

    if cytome_layer == "log1p":
        cell_depth = _modality_cell_depth(
            ds, resolved_modality, use_cached_stats=use_cached_stats,
            batch_size=batch_size,
        )
        params = ds.metadata.get(f"{resolved_modality}_log1p_params")
        scale_factor = float(params["scale_factor"]) if params else 1e4
        depth = np.where(cell_depth == 0, 1.0, cell_depth)
        return np.log1p(counts_col / depth * scale_factor).astype(np.float32), resolved_modality

    if cytome_layer == "infog":
        params = _ensure_infog_params(
            ds, resolved_modality, use_cached_stats=use_cached_stats,
            batch_size=batch_size,
        )
        if params is None:
            raise ValueError(
                f"infog params not available for modality '{resolved_modality}'. "
                f"Run piaso.tl.infog(ds, save_layer=False) first."
            )
        cd = np.asarray(params["cell_depth"], dtype=np.float64)
        ig = np.asarray(params["inv_gene_depth"], dtype=np.float64)
        scale = float(params["scale"])
        counts_sum = float(params["counts_sum"])
        thr = params.get("threshold")
        # Per-feature equivalent of _normalize_chunk_infog (one column).
        x = np.asarray(counts_col, dtype=np.float64)
        cd_safe = np.where(cd == 0, 1.0, cd)
        normalized = x * (scale / cd_safe)
        info_factor = x * (counts_sum / cd_safe) * float(ig[feat_idx])
        product = normalized * info_factor
        result = np.sqrt(np.maximum(product, 0.0))
        if thr is not None:
            np.minimum(result, float(thr), out=result)
        return result.astype(np.float32), resolved_modality

    if cytome_layer == "tfidf":
        params = _ensure_tfidf_params(
            ds, resolved_modality, use_cached_stats=use_cached_stats,
            batch_size=batch_size,
        )
        if params is None:
            raise ValueError(
                f"tfidf params not available for modality '{resolved_modality}'. "
                f"Run piaso.tl.compute_tfidf_stats(ds, modality='{resolved_modality}') first."
            )
        cd = np.asarray(params["cell_depth"], dtype=np.float64)
        idf = np.asarray(params["idf"], dtype=np.float64)
        scale_factor = float(params.get("scale_factor", 1e4))
        cd_safe = np.where(cd == 0, 1.0, cd)
        tf = np.asarray(counts_col, dtype=np.float64) / cd_safe * scale_factor
        tf = np.log1p(tf)
        return (tf * float(idf[feat_idx])).astype(np.float32), resolved_modality

    # Unknown layer with no on-the-fly path → actionable error.
    available = [r[0] for r in ds._conn.execute(
        "SELECT matrix_name FROM matrix_meta WHERE matrix_name LIKE ?",
        (f"{resolved_modality}_%",),
    ).fetchall()]
    raise ValueError(
        f"Layer '{cytome_layer}' for modality '{resolved_modality}' has no "
        f"on-the-fly path and matrix '{matrix_name}' is not in the cytome. "
        f"Materialise it first, or pick a known on-the-fly layer "
        f"(log1p, infog, tfidf). Available {resolved_modality} layers: "
        f"{available}."
    )


def _resolve_cytome_feature_values_batch(
    ds, features, modality=None, cytome_layer="counts",
    compute_on_fly=True, use_cached_stats=True, batch_size=2048,
):
    """Batched analogue of :func:`_resolve_cytome_feature_values`: resolve MANY
    features with a **single streaming pass** over each (modality, layer) matrix
    instead of one pass per feature (the dotplot footgun — N genes = N scans).

    Returns ``{feature: (values, resolved_modality)}``. Numerically identical to
    calling the single-feature resolver per feature. Features that resolve to the
    same modality are read together; the materialised-matrix path reads the layer
    directly, the on-the-fly path reads ``counts`` once and vectorises the
    log1p/infog/tfidf normalisation across the requested columns.
    """
    # 1. Resolve each feature → (resolved_modality, feat_idx).
    resolved = {}                                    # feature -> (mod, feat_idx)
    for feat in features:
        if modality is not None:
            hit = _modality_has_feature(ds, modality, None, None, feat)
            if hit is None:
                raise ValueError(
                    f"Feature '{feat}' not found in modality '{modality}'.")
            resolved[feat] = (modality, hit[0])
        else:
            cands = [(m, _modality_has_feature(ds, m, e, ic, feat))
                     for m, e, ic in _plot_modality_registry()]
            cands = [(m, h[0]) for m, h in cands if h is not None]
            if not cands:
                raise KeyError(f"Feature '{feat}' not found in any modality.")
            if len(cands) > 1:
                raise ValueError(
                    f"Feature '{feat}' is ambiguous across modalities "
                    f"{[m for m, _ in cands]}; pass modality= to disambiguate.")
            resolved[feat] = cands[0]

    # 2. Group features by resolved modality.
    by_mod = {}
    for feat, (mod, fidx) in resolved.items():
        by_mod.setdefault(mod, []).append((feat, fidx))

    out = {}
    for mod, items in by_mod.items():
        feats = [f for f, _ in items]
        fidx = np.asarray([i for _, i in items], dtype=np.int64)
        matrix_name = f"{mod}_{cytome_layer}"
        if ds.matrix_meta(matrix_name) is not None:
            vals2d = _read_feature_columns_cytome(ds, mod, cytome_layer, fidx, batch_size)
        else:
            if not compute_on_fly:
                raise ValueError(
                    f"Matrix '{matrix_name}' not found and compute_on_fly=False.")
            counts_name = f"{mod}_counts"
            if ds.matrix_meta(counts_name) is None:
                raise ValueError(
                    f"Cannot compute '{cytome_layer}' on-the-fly: '{counts_name}' missing.")
            x = _read_feature_columns_cytome(ds, mod, "counts", fidx, batch_size).astype(np.float64)
            if cytome_layer == "log1p":
                cd = _modality_cell_depth(ds, mod, use_cached_stats=use_cached_stats,
                                          batch_size=batch_size)
                params = ds.metadata.get(f"{mod}_log1p_params")
                sf = float(params["scale_factor"]) if params else 1e4
                depth = np.where(cd == 0, 1.0, cd)[:, None]
                vals2d = np.log1p(x / depth * sf)
            elif cytome_layer == "infog":
                params = _ensure_infog_params(ds, mod, use_cached_stats=use_cached_stats,
                                              batch_size=batch_size)
                if params is None:
                    raise ValueError(f"infog params not available for '{mod}'.")
                cd = np.asarray(params["cell_depth"], np.float64)
                ig = np.asarray(params["inv_gene_depth"], np.float64)[fidx][None, :]
                scale = float(params["scale"]); csum = float(params["counts_sum"])
                thr = params.get("threshold")
                cd_safe = np.where(cd == 0, 1.0, cd)[:, None]
                normalized = x * (scale / cd_safe)
                info = x * (csum / cd_safe) * ig
                vals2d = np.sqrt(np.maximum(normalized * info, 0.0))
                if thr is not None:
                    np.minimum(vals2d, float(thr), out=vals2d)
            elif cytome_layer == "tfidf":
                params = _ensure_tfidf_params(ds, mod, use_cached_stats=use_cached_stats,
                                              batch_size=batch_size)
                if params is None:
                    raise ValueError(f"tfidf params not available for '{mod}'.")
                cd = np.asarray(params["cell_depth"], np.float64)
                idf = np.asarray(params["idf"], np.float64)[fidx][None, :]
                sf = float(params.get("scale_factor", 1e4))
                cd_safe = np.where(cd == 0, 1.0, cd)[:, None]
                vals2d = np.log1p(x / cd_safe * sf) * idf
            else:
                raise ValueError(
                    f"Layer '{cytome_layer}' for '{mod}' has no on-the-fly path "
                    f"and '{matrix_name}' is not materialised.")
        vals2d = np.asarray(vals2d, dtype=np.float32)
        for col, feat in enumerate(feats):
            out[feat] = (vals2d[:, col], mod)
    return out


def plotEmbedding(
    data,
    color="leiden",
    basis="X_umap",
    layer=None,
    title=None,
    figsize=None,
    point_size=None,
    alpha=1.0,
    frameon=None,
    save=None,
    show=None,
    ax=None,
    palette=None,
    legend_loc="right",
    legend_fontsize=10,
    legend_fontoutline=None,
    legend_fontbackground="white",
    legend_fontbackground_alpha=0.7,
    legend_ncol=None,
    rasterized=True,
    dpi=None,
    vmin=None,
    vmax=None,
    vmin_pct=None,
    vmax_pct=None,
    cmap=None,
    show_axes_arrow=False,
    axes_arrow_loc="bottom_left",
    modality=None,
    cytome_layer="counts",
    compute_on_fly=True,
    use_cached_stats=True,
    show_modality_in_title=False,
    ncol=None,
    col_size=4.0,
    row_size=4.0,
    fix_coordinate_ratio=True,
    show_axis_ticks=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    legend_marker_size=None,
    hspace=None,
    wspace=None,
    groups=None,
    na_color="lightgray",
    na_label="other",
    image=False,
    img_key="hires",
    image_alpha=1.0,
    image_crop=True,
    cell_mask=None,
    return_fig=False,
    linewidths: float = 0.0,
    edgecolors="none",
    sort_order="category",
    vcenter=None,
    **kwargs,
):
    """Plot a 2-D embedding colored by a cell annotation or continuous value.

    cell_mask : array-like, optional
        Plot only these cells: a boolean mask of length ``n_cells`` or
        integer indices. Applied after the coordinates and colour values are
        resolved, so both stay aligned — pairs with a cytome's
        ``cells_in_region(x=, y=)`` for ROI views without writing a subset
        file.
    image : bool or str, default False
        Draw the library's tissue image under the spots (``basis='spatial'``
        workflows). ``True`` auto-selects when exactly one library has an
        image (or the plotted cells span one) and raises naming the options
        otherwise; a string selects that library. Sources: AnnData
        ``uns['spatial']`` or a cytome's stored images (cytome ≥ 0.2.6). The
        y-axis follows the image convention (top of the tissue at the top)
        only when an image is drawn.
    img_key : str, default 'hires'
        Which image of the library to draw ('hires', 'lowres', ...).
    image_alpha : float, default 1.0
        Opacity of the tissue image.
    image_crop : bool, default True
        With an image, crop to the plotted cells' bounding box padded by one
        spot diameter instead of showing the whole slide.
    Returns ``None`` by default — pass ``return_fig=True`` to get
    ``(fig, ax)`` back.

    Supports both ``cytome.Dataset`` and ``AnnData`` objects.

    Parameters
    ----------
    data
        A ``cytome.Dataset`` or ``AnnData`` object.
    color : str, list of str, ndarray, or pandas.Series
        Column in ``cells`` / ``obs`` for colouring, or a **feature name**
        (gene, peak, tile, or GA gene — resolved via the cytome modality
        registry; pass ``modality=`` to disambiguate). A single ``str`` draws one
        panel; a ``list``/``tuple`` of strings draws a multi-panel grid (one
        panel per entry, scanpy ``sc.pl.umap``-style).

        May also be the **values themselves** — one per cell, in order — as a
        numpy array, a pandas Series, or a Categorical, for anything computed on
        the fly and not written back to the object. Numeric values get a
        colorbar, everything else a category legend; a Categorical's own order
        is kept, and a named Series titles the panel. Note that a plain
        ``list``/``tuple`` still means multiple panels, so pass an array.
    basis : str
        Embedding key (e.g. ``'X_umap'``, ``'X_svd'``).
    layer : str, optional
        AnnData layer to read the feature value from when *color* is a feature
        name. If None, reads from ``adata.X``.
    title : str or list of str, optional
        Plot title. Defaults to *color*. For a list ``color``: a single string
        becomes a figure-level suptitle over the grid; a list of titles (length
        must match ``color``) sets one title per panel.
    figsize : tuple, optional
        Figure size in inches.  If None, uses ``rcParams['figure.figsize']``
        (set via ``piaso.settings.set_figure_params(figsize=...)``).
    point_size : float or None
        Scatter point size.  If None (default), automatically calculated
        from the number of cells: ``22000 / n_cells`` clamped to [0.1, 3]
        (explicit value overrides).
    alpha : float
        Point transparency.  Default 1.0 (opaque).
    frameon : bool, optional
        Whether to show axis frame (spines). ``None`` follows
        ``piaso.settings._frameon``, which starts out ``False``.
    save : str, bool, or None
        Figure save behavior:
        - ``None``: don't save.
        - Full path (``'/path/to/fig.png'``): save directly.
        - Suffix (``'_leiden'``): save to ``piaso.settings.figdir``.
        - ``True``: auto-name and save to ``piaso.settings.figdir``.
    show : bool, optional
        Whether to call ``plt.show()``. ``None`` (default) shows unless ``ax``
        is given: drawing into a caller's axes leaves the showing to the caller.
        When not shown, the figure is returned; a figure this function made is
        closed only once it is saved (``save=``).

        *Changed*: the default was True, which called ``plt.show()`` from
        inside a composed figure.
    ax : matplotlib Axes, optional
        Pre-existing axes to draw on. Its figure is never closed here.
    palette : list[str], dict, or str, optional
        Color palette for categorical data. A dict maps category to colour (by
        name) and its key order leads the legend; a list is matched to the
        column's category order; a string names a colormap or palette. If
        None, the stored colours are used -- ``adata.uns['{color}_colors']``,
        a cytome's ``set_categories``, or the colours an AnnData carried into
        the store -- then ``piaso.pl.color.d_color4``. Colours are assigned by
        name over the whole column, so ``cell_mask`` and ``groups`` never move
        one.
    sort_order : {'category', 'size', 'random'}, default ``'category'``
        Drawing order of a categorical colour. ``'category'`` draws the
        categories in legend order, each over the ones before; ``'size'``
        draws the largest first so rare types end on top (a rare type drawn
        first under a large one can have no visible pixel); ``'random'``
        draws all cells in one shuffled, seeded pass.
    groups : str or list of str, optional
        For a **categorical** ``color``: highlight only these category values —
        they keep their palette colour while all other cells are greyed out
        (drawn behind), like ``sc.pl.umap(groups=...)``. Palette indices come
        from the full category list, so colours match an unfiltered plot. Raises
        ``ValueError`` if a name is not among the categories.

        The legend then lists the highlighted groups and one final entry for
        the grey — the colours on the plot, and nothing else.
    na_color : str, default ``'lightgray'``
        Colour for the greyed-out (non-``groups``) cells, and for cells whose
        value is missing (NaN, None), which are drawn first and, for a
        categorical colour, listed as ``"NA"`` in the legend.
    na_label : str, default ``'other'``
        Legend label for that colour. Only appears when ``groups`` actually
        left something out.
    legend_loc : str, default ``'right'``
        Where to put the category labels. One of:

        - ``'right'`` — legend box outside the axes, in the right margin.
        - ``'on_data'`` — labels drawn at each category's centroid.
        - ``'both'`` — centroid labels *and* the right-margin legend, so you can
          name a cluster where the points are and still read its colour off the
          list. Neither alone does both jobs.
        - ``'none'`` — no legend.

        Any other value raises ``ValueError`` rather than silently drawing
        nothing.
    legend_fontsize : int
        Font size for legend labels.
    legend_fontoutline : float or None
        Width of text outline for ``legend_loc='on_data'`` labels.
    legend_fontbackground : str or None, default ``'white'``
        Colour of the patch drawn behind each ``'on_data'`` label. ``None``
        draws the text with no background. The default is readable over a
        cluster and invisible over the figure background, which is where a
        small or sparse cluster's label ends up — pass a tinted colour
        (``'0.9'``, ``'#fffbe6'``) when labels need to read in both places.
    legend_fontbackground_alpha : float, default ``0.7``
        Opacity of that patch. Raise it towards 1.0 to hide the points
        underneath a label; lower it to let them show through.
        Adds a contrasting stroke around text for readability.  If None
        (default), no outline is drawn.
    legend_ncol : int, optional
        Number of legend columns.  If None, auto-calculated (~12 per column).
    rasterized : bool
        Rasterize scatter points for smaller vector files.
    linewidths : float, optional (default 0)
        Marker stroke width, in points. **Zero by default.** Matplotlib
        otherwise strokes every point at ``rcParams['patch.linewidth']``
        (1.0 pt) in ``edgecolors='face'``, which for cell-sized markers is
        most of the dot: ``point_size=0.2`` is a disc 0.45 pt across inside a
        1.0 pt stroke, so it draws 3.2x the size asked for, and under
        ``alpha < 1`` the stroke and fill composite twice around the rim —
        ringed dots with a washed-out core, visible once a PDF is enlarged.
        Pass a positive value to outline markers deliberately.

        *Changed in 1.2.4*: was matplotlib's 1.0 pt default.
    edgecolors : color or 'none', optional (default 'none')
        Marker stroke colour. Only meaningful with ``linewidths > 0``.

        *Changed in 1.2.4*: was matplotlib's ``'face'``.
    dpi : int, optional
        Display DPI. If None, uses ``rcParams['figure.dpi']``.
    vmin, vmax : float
        Limits for continuous color scale.
    vmin_pct, vmax_pct : float, optional
        Percentile (0-100) used to derive ``vmin``/``vmax`` for continuous
        features when the explicit limit is not given — e.g. ``vmax_pct=99``
        clips the top 1% of cells so a few outliers don't wash out the colour
        scale. ``None`` (default) = no percentile clipping. Ignored for
        categorical colours and overridden by an explicit ``vmin``/``vmax``.
    cmap
        Colormap for continuous data.
    vcenter : float, optional
        For a continuous colour on a diverging map (the default for a numeric
        cell column is ``'Spectral_r'``): the value drawn in the map's centre
        colour, e.g. ``0`` for a signed score. Each side is stretched to its
        own half of the map. Must lie between the colour limits.
    show_axes_arrow : bool
        Draw small coordinate arrows at a corner of the plot.
    axes_arrow_loc : str
        Position of axes arrow: ``'bottom_left'`` or ``'bottom_right'``.
    modality : str, optional
        Cytome modality for feature lookup: ``'RNA'``, ``'GA'``,
        ``'ATAC'``, ``'tiles'``, or ``None`` for auto-detect. Auto-detect
        raises ``ValueError`` if the feature is in multiple modalities;
        pass an explicit string to disambiguate. Ignored for AnnData
        inputs.
    cytome_layer : str, default ``'counts'``
        Cytome matrix suffix to read; combined with ``modality`` to form
        ``{modality}_{cytome_layer}``. Common values: ``'counts'``,
        ``'log1p'``, ``'infog'``, ``'tfidf'``.
    compute_on_fly : bool, default ``True``
        If the requested ``{modality}_{cytome_layer}`` matrix isn't
        materialised in the cytome, compute the value per-feature on the
        fly from ``{modality}_counts`` using cached / freshly-computed
        params. Supported on-the-fly layers: ``'log1p'``, ``'infog'``,
        ``'tfidf'``. Set ``False`` for strict mode (raise on missing
        matrix).
    use_cached_stats : bool, default ``True``
        Reuse per-modality cached params from ``ds.metadata`` (e.g.
        ``'{modality}_infog_params'``) when computing on the fly. Set
        ``False`` to ignore the cache and recompute.
    show_modality_in_title : bool, default ``False``
        Append ``(modality)`` to the panel title when the colour was
        resolved through a cytome modality (e.g. ``'Sox2 (RNA)'``).
        Affects feature colours only — obs columns are unchanged.
    ncol : int, optional
        Columns in the multi-panel grid when ``color`` is a list.
        Defaults to ``ceil(sqrt(n_colors))``. Aliased as ``ncols``
        (scanpy convention). Ignored when ``color`` is a single string.
    col_size : float, default ``4.0``
        Per-panel width in inches when ``color`` is a list.
    row_size : float, default ``4.0``
        Per-panel height in inches when ``color`` is a list.
    fix_coordinate_ratio : bool, default ``True``
        If ``True``, sets ``ax.set_aspect('equal')`` so x and y axes are
        scaled equally — appropriate for UMAP / t-SNE / spatial coords
        where distances are meaningful. Set ``False`` to use
        ``'auto'`` (let matplotlib stretch to fit the axes).
    show_axis_ticks : bool, default ``False``
        Whether to display axis tick marks and labels. Off by default
        for embedding-style plots where coordinates are abstract.
    x_min, x_max, y_min, y_max : float, optional
        Custom axis limits. Each is independently optional; pass
        only the ones you want to override (others use the data range).
    legend_marker_size : float, optional
        Marker scale for the legend (categorical colours only).
        If ``None``, auto-computed from ``point_size`` as
        ``max(3, 12 / point_size)``.
    hspace : float, optional
        Vertical spacing between rows of the multi-panel grid (only
        applies when ``color`` is a list). If ``None`` (default), uses
        ``0.1`` when ``show_axis_ticks=False`` and ``0.25`` when
        ``show_axis_ticks=True``. Pass an explicit value to override
        (e.g. ``0.05`` for very tight, ``0.4`` for wide spacing).
    wspace : float, optional
        Horizontal spacing between columns of the multi-panel grid
        (only applies when ``color`` is a list). If ``None`` (default),
        uses ``0.2``. Override to tighten or widen.

    `color` accepts ``str`` (single panel) OR ``list[str]`` / ``tuple[str]``
    (one panel per entry, on a ``ncol``-column grid). When a list is
    given together with ``ax``, raises ``ValueError`` since a single
    axes can't host a multi-panel grid.
    image : bool or str, default False
        Draw the tissue image behind spatial coordinates: ``True`` for the
        first library, a string for that library. Sources: AnnData
        ``uns['spatial']`` or a cytome's stored images.
    img_key : str, default 'hires'
        Which image of the library to draw.
    image_alpha : float, default 1.0
        Opacity of the tissue image.
    image_crop : bool, default True
        Crop the image to the plotted coordinates.
    cell_mask : ndarray, optional
        Plot these cells only (boolean mask of length ``n_cells``, or integer
        indices). Colours and the category order come from the whole column.
    return_fig : bool, default False
        Return ``(fig, ax)`` (``(fig, [axes])`` for a list ``color``).

    Returns
    -------
    fig, ax
        With ``return_fig=True``, for single-color: ``(fig, ax)`` where
        ``ax`` is the matplotlib Axes that was drawn into.
    fig, axs
        With ``return_fig=True``, for list-color: ``(fig, [ax0, ax1, ...])``
        — one Axes per entry in ``color``; trailing empty grid cells are
        hidden via ``set_visible(False)``.
    fig
        Otherwise the figure when it is not shown, and None when it is.
    """
    # Every argument of this call, for the panels of a list `color`: a grid
    # forwarded a hand-written subset, and cell_mask, image*, linewidths,
    # edgecolors and na_label silently fell out of it (a mask of 412 of 600
    # cells drew all 600).
    _call = dict(locals())
    _call.update(_call.pop("kwargs"))
    from . import color as _color_mod
    from .. import settings as _settings
    from . import _figure

    # Resolve defaults from settings
    if frameon is None:
        frameon = _settings._frameon

    # Validate legend_loc strictly — silently rendering no legend on a typo
    # (e.g. 'on data', 'ondata') is confusing. Canonical set is the underscore
    # form shared with plot_embeddings_split.
    if legend_loc is None:
        legend_loc = "none"
    if legend_loc not in _VALID_LEGEND_LOC:
        raise ValueError(
            f"Invalid legend_loc={legend_loc!r}. Choose from "
            f"{sorted(_VALID_LEGEND_LOC)} (note the underscore in 'on_data')."
        )

    # ------------------------------------------------------------------
    # Multi-color dispatch: when `color` is a list (or tuple) of
    # column / feature names, build a subplot grid and recurse once per
    # entry. Mirrors scanpy's sc.pl.umap(adata, color=['g1','g2',...])
    # semantics. `ncol` defaults to ceil(sqrt(n)); `ax=` cannot host a
    # grid so combining the two raises a clean TypeError.
    # ------------------------------------------------------------------
    # A grid's panels receive their feature's values read in one pass with
    # the others (private; see _prefetch_cytome_features).
    _prefetched = kwargs.pop('_prefetched', None)
    _call.pop('_prefetched', None)
    # `ncols` accepted as an alias for `ncol` (scanpy convention).
    if 'ncols' in kwargs:
        if ncol is None:
            ncol = kwargs.pop('ncols')
        else:
            kwargs.pop('ncols')
    if kwargs:
        # Unknown extras — mirror Python's normal "unexpected keyword" UX.
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(
            f"plotEmbedding() got unexpected keyword argument(s): {unexpected}"
        )

    _n_cells = _n_cells_of(data)
    _keys = _keys_from_array(color, _n_cells)
    if _keys is not None:
        _check_keys_resolve(_keys, data, _n_cells)
        color = _keys
    if isinstance(color, (list, tuple)):
        color_list = list(color)
        if ax is not None:
            raise ValueError(
                "Cannot pass `color` as a list together with `ax`: a single "
                "axes can't host a multi-panel grid. Drop `ax=` to let "
                "plotEmbedding build the grid, or call plotEmbedding once "
                "per color with the matching `ax`."
            )
        n = len(color_list)
        if n == 0:
            raise ValueError("`color` was an empty list.")
        if n == 1:
            # Trivial single-element list — fall through to the scalar path.
            color = color_list[0]
        else:
            fig, axs, _nrow, _ncol = _build_subplots(
                n, ncol=ncol, dpi=dpi or 80,
                col_size=col_size, row_size=row_size,
            )
            axs_flat = list(axs.ravel()) if hasattr(axs, "ravel") else [axs]
            # Per-panel titles: a list/tuple of titles maps 1:1 to the colour
            # panels (must match length); a single string becomes a figure-level
            # suptitle over the grid; None keeps each panel's default (its color).
            if isinstance(title, (list, tuple)):
                if len(title) != n:
                    raise ValueError(
                        f"plotEmbedding(): `title` list length ({len(title)}) "
                        f"must match `color` list length ({n})."
                    )
                _panel_titles = list(title)
                _suptitle = None
            else:
                _panel_titles = [None] * n
                _suptitle = title
            # The grid's own arguments stay here; everything else reaches
            # every panel unchanged.
            _grid_only = {"data", "color", "title", "figsize", "save", "show", "ax",
                          "return_fig", "ncol", "ncols", "col_size", "row_size",
                          "hspace", "wspace"}
            _panel_kw = {k: v for k, v in _call.items() if k not in _grid_only}
            # On a store, the grid's features are read in one pass, not one
            # pass per panel (review I2 finding 15).
            _pre = (_prefetch_cytome_features(
                        data, color_list, layer=layer, modality=modality,
                        cytome_layer=cytome_layer, compute_on_fly=compute_on_fly,
                        use_cached_stats=use_cached_stats)
                    if _is_cytome_input(data) else {})
            for i, color_i in enumerate(color_list):
                _extra = ({'_prefetched': _pre[color_i]}
                          if isinstance(color_i, str) and color_i in _pre else {})
                plotEmbedding(
                    data, color=color_i, title=_panel_titles[i],
                    figsize=None, save=None, show=False, ax=axs_flat[i],
                    return_fig=False, **_panel_kw, **_extra,
                )
            # Hide trailing empty axes if n_colors < nrow*ncol
            for j in range(n, len(axs_flat)):
                axs_flat[j].set_visible(False)
            # Tighten the inter-panel spacing. Matplotlib's default
            # hspace=0.2 reserves room for x-tick labels we've hidden;
            # 0.1 is much closer for clean embedding grids. User can
            # override via hspace= / wspace= or pass None to keep
            # matplotlib defaults.
            _h = hspace if hspace is not None else (
                0.25 if show_axis_ticks else 0.1
            )
            _w = wspace if wspace is not None else 0.2
            fig.subplots_adjust(hspace=_h, wspace=_w)
            # Must come after subplots_adjust: it reads the final axes
            # positions and shrinks them in place.
            _reserve_legend_space(fig)
            if _suptitle is not None:
                fig.suptitle(_suptitle)
            out = _figure.finish(fig, created=True, show=_figure.resolve_show(show),
                                 save=save, writekey='plotEmbedding')
            if return_fig:
                return fig, axs_flat
            return out

    coords, values, is_cat, resolved_modality, _from_obs = _get_embedding_and_color(
        data, basis, color, layer=layer,
        modality=modality, cytome_layer=cytome_layer,
        compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
        prefetched=_prefetched,
    )

    # Order and colours of a categorical colour are resolved here, over the
    # WHOLE column, before any cell_mask: a category the mask leaves out must
    # keep its place, or every later category takes the colour of the one
    # before it (Inh was drawn in Exc's colour once a mask dropped Exc).
    style = None
    if is_cat:
        from ._categories import resolve_category_style
        if _is_value_array(color):
            # An array passed as `color` names no stored palette, so only the
            # user's own palette applies. A pandas Categorical does carry an
            # order, though, and honouring it is the point of passing one.
            _order = None
            if isinstance(color, pd.Categorical):
                _order = [str(c) for c in color.categories]
            elif isinstance(color, pd.Series) and isinstance(
                    getattr(color, "dtype", None), pd.CategoricalDtype):
                _order = [str(c) for c in color.cat.categories]
            style = _style_for_values(values, palette, _order, na_color)
        else:
            style = resolve_category_style(data, color, values, palette=palette,
                                           na_color=na_color)

    # Plot a subset without writing a subset file: a boolean mask (length
    # n_cells) or integer indices, applied after the coordinates and colour
    # values are resolved so both stay aligned. Pairs with a cytome's
    # ``cells_in_region`` for ROI views.
    if cell_mask is not None:
        _m = np.asarray(cell_mask)
        if _m.dtype == bool:
            if _m.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"cell_mask has length {_m.shape[0]} but the embedding has "
                    f"{coords.shape[0]} cells")
            _sel = np.where(_m)[0]
        else:
            _sel = _m.astype(np.int64)
        if _sel.size == 0:
            raise ValueError("cell_mask selects no cells")
        coords = coords[_sel]
        if values is not None:
            values = np.asarray(values)[_sel]


    # Robust color limits for continuous features: when vmin/vmax are not set
    # explicitly, derive them from percentiles of the values (0-100 scale, like
    # numpy). vmax_pct=99 clips the top 1% of outliers so the colour scale isn't
    # dominated by a few extreme cells. Ignored for categorical colours.
    if not is_cat and values is not None:
        _v = np.asarray(values, dtype=float)
        if vmax is None and vmax_pct is not None and _v.size:
            vmax = float(np.nanpercentile(_v, vmax_pct))
        if vmin is None and vmin_pct is not None and _v.size:
            vmin = float(np.nanpercentile(_v, vmin_pct))

    # Auto point_size: scale inversely with cell count
    if point_size is None:
        n_cells = coords.shape[0]
        point_size = max(0.1, min(3, 22000 / n_cells))

    show = _figure.resolve_show(show, ax)
    # figsize=None → matplotlib uses rcParams['figure.figsize']
    fig, ax, _created = _figure.begin(ax, nrows=1, ncols=1, figsize=figsize, dpi=dpi)

    # Tissue image under the scatter (basis='spatial' workflows). The image
    # is drawn in full-resolution coordinate units with y increasing downward
    # (see _spatial_image), so the scatter needs no changes; only the axis
    # orientation follows the image.
    _img_ctx = None
    if image:
        from ._spatial_image import (_draw_image_overlay, _image_axis_limits,
                                     _resolve_spatial_image)
        _img_ctx = _resolve_spatial_image(data, image, img_key)
        if _img_ctx is not None:
            _draw_image_overlay(ax, _img_ctx, image_alpha)

    # Continuous colour map depends on WHERE the values came from, because the
    # two cases are read differently. A gene/peak value is an expression level:
    # sequential, zero is meaningful, and ``c_color1`` puts the eye on the high
    # end. A numeric cell-metadata column -- a regulon score, a stage index, a
    # QC metric -- is a quantity spread across cells with no privileged zero,
    # and reads better on a diverging map. Explicit ``cmap=`` always wins.
    if cmap is None and not is_cat and _from_obs:
        cmap = "Spectral_r"

    # Continuous colour maps default by SOURCE, not by dtype: a feature
    # (gene, peak, gene-activity) keeps PIASO's sequential `color_1`, while a
    # numeric CELL column -- regulon activity, a QC metric, pseudotime -- gets
    # `Spectral_r`. Both are continuous, but they answer different questions,
    # and using one ramp for both makes a metadata panel read as expression.
    _cmap = cmap
    if _cmap is None and not is_cat:
        _cmap = "Spectral_r" if _from_obs else _color_mod.c_color1
    _render_embedding(
        ax, coords, values, is_cat,
        color=(_value_array_name(color) if _is_value_array(color) else color),
        style=style, sort_order=sort_order,
        point_size=point_size, alpha=alpha, rasterized=rasterized,
        groups=groups, na_color=na_color, na_label=na_label,
        legend_loc=legend_loc,
        add_colorbar=(legend_loc != "none"),
        legend_ncol=legend_ncol, legend_marker_size=legend_marker_size,
        legend_fontsize=legend_fontsize, legend_fontoutline=legend_fontoutline,
        legend_fontbackground=legend_fontbackground,
        legend_fontbackground_alpha=legend_fontbackground_alpha,
        cmap=_cmap, vmin=vmin, vmax=vmax, vcenter=vcenter,
        linewidths=linewidths, edgecolors=edgecolors,
    )

    # --- frameon ---
    if not frameon:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if title is None:
        # An array passed as `color` has no name to put here unless it is a
        # named Series, so fall back to a neutral one rather than str() of the
        # array.
        _color_name = (_value_array_name(color) if _is_value_array(color)
                       else color)
        title_str = _color_name
        if show_modality_in_title and resolved_modality:
            title_str = f"{_color_name} ({resolved_modality})"
        ax.set_title(title_str)
    else:
        ax.set_title(title)
    # --- Aspect ratio (fix_coordinate_ratio) ---
    ax.set_aspect("equal" if fix_coordinate_ratio else "auto")
    # --- Axis ticks (show_axis_ticks) ---
    if show_axis_ticks:
        ax.tick_params(labelbottom=True, labelleft=True)
    else:
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if _img_ctx is not None and image_crop and x_min is None and x_max is None \
            and y_min is None and y_max is None:
        from ._spatial_image import _image_axis_limits
        _image_axis_limits(ax, coords, _img_ctx)
    elif _img_ctx is not None:
        # explicit limits: still honour the image's top-down orientation
        lo, hi = ax.get_ylim()
        if lo < hi:
            ax.set_ylim(hi, lo)

    # --- Custom axis limits (x_min/x_max/y_min/y_max) ---
    if x_min is not None or x_max is not None:
        cur = ax.get_xlim()
        ax.set_xlim(x_min if x_min is not None else cur[0],
                    x_max if x_max is not None else cur[1])
    if y_min is not None or y_max is not None:
        cur = ax.get_ylim()
        ax.set_ylim(y_min if y_min is not None else cur[0],
                    y_max if y_max is not None else cur[1])

    # --- Axes arrow (coordinate compass) ---
    if show_axes_arrow:
        _draw_axes_arrow(ax, basis, loc=axes_arrow_loc)

    out = _figure.finish(fig, created=_created, show=show, save=save,
                         writekey='plotEmbedding')
    if return_fig:
        return fig, ax
    return out


def _draw_axes_arrow(ax, basis, loc="bottom_left"):
    """Draw small coordinate arrows at a corner of the embedding plot."""
    import matplotlib.patches as mpatches

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    arrow_len = 0.1 * min(x_range, y_range)

    if loc == "bottom_right":
        ox = xlim[1] - 0.15 * x_range
        oy = ylim[0] + 0.05 * y_range
    else:  # bottom_left
        ox = xlim[0] + 0.05 * x_range
        oy = ylim[0] + 0.05 * y_range

    # Derive labels from basis
    keyword = basis.replace('X_', '').upper()
    label_x = f"{keyword}-1"
    label_y = f"{keyword}-2"

    arrow_props = dict(arrowstyle='->', color='black', lw=1.5)
    ax.annotate('', xy=(ox + arrow_len, oy), xytext=(ox, oy),
                arrowprops=arrow_props)
    ax.annotate('', xy=(ox, oy + arrow_len), xytext=(ox, oy),
                arrowprops=arrow_props)
    ax.text(ox + arrow_len * 0.5, oy - 0.03 * y_range, label_x,
            fontsize=7, ha='center', va='top')
    ax.text(ox - 0.03 * x_range, oy + arrow_len * 0.5, label_y,
            fontsize=7, ha='right', va='center', rotation=90)


def plotUMAP(data, color="leiden", **kwargs):
    """Convenience wrapper for :func:`plotEmbedding` with ``basis='X_umap'``."""
    return plotEmbedding(data, color=color, basis="X_umap", **kwargs)