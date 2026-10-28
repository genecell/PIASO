"""Scatter plot for two features (cell × cell or gene × gene).

Supports coloring by a categorical or continuous variable.
Works with AnnData and cytome Dataset.
"""

from typing import Optional
import warnings
import numpy as np
from ._group_order import natural_key
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ._store_io import read_cells_column as _read_cells_column  # read-only


def _get_feature_vector(data, feature, layer=None, use_raw=None,
                        modality=None, cytome_layer="counts",
                        compute_on_fly=True, use_cached_stats=True):
    """Extract a single feature's values from data.

    For cytome inputs, prefers ``ds.cells[feature]`` (e.g. pre-computed
    score columns); falls through to the modality registry resolver if
    the feature is a gene/peak/tile name.
    """
    if _is_cytome_input(data):
        from ..utils._cytome_compat import open_cytome as _open_cytome
        from ._plotEmbedding import _resolve_cytome_feature_values
        with _open_cytome(data, mode="r") as ds:
            cells_cols = set(ds.cells.columns)
        if feature in cells_cols:
            return pd.to_numeric(
                pd.Series(_read_cells_column(data, feature)), errors='coerce'
            ).values
        with _open_cytome(data, mode="r") as ds:
            vals, _ = _resolve_cytome_feature_values(
                ds, feature,
                modality=modality, cytome_layer=cytome_layer,
                compute_on_fly=compute_on_fly,
                use_cached_stats=use_cached_stats,
            )
        return np.asarray(vals, dtype=float)

    import scipy.sparse as sp
    adata = data
    if feature in adata.obs.columns:
        return np.asarray(adata.obs[feature].values, dtype=float)
    elif use_raw and adata.raw is not None:
        v = adata.raw[:, feature].X
        if sp.issparse(v):
            v = v.toarray()
        return np.asarray(v).flatten()
    elif layer is not None:
        return np.asarray(adata.obs_vector(feature, layer=layer), dtype=float)
    else:
        return np.asarray(adata.obs_vector(feature), dtype=float)


def _get_color_values(data, color,
                      modality=None, cytome_layer="counts",
                      compute_on_fly=True, use_cached_stats=True):
    """Get color column values and determine if categorical.

    One rule with the embedding plot (``_is_numeric_dtype``): a column is
    continuous when its dtype is numeric, booleans excepted. A store's
    ``leiden`` column is text, so it is categorical; it used to be parsed as
    numbers ("0", "1", ...) and drawn on a colorbar here while the embedding
    beside it drew categories.
    """
    from ._plotEmbedding import _is_numeric_dtype
    if color is None:
        return None, False

    if _is_cytome_input(data):
        from ..utils._cytome_compat import open_cytome as _open_cytome
        from ._plotEmbedding import _resolve_cytome_feature_values
        with _open_cytome(data, mode="r") as ds:
            cells_cols = set(ds.cells.columns)
            if color in cells_cols:
                vals = np.asarray(ds.cells[color])
                if _is_numeric_dtype(vals):
                    return np.asarray(vals, dtype=float), False
                return vals, True
        # Resolve via modality registry (gene/peak/tile name)
        with _open_cytome(data, mode="r") as ds:
            vals, _ = _resolve_cytome_feature_values(
                ds, color,
                modality=modality, cytome_layer=cytome_layer,
                compute_on_fly=compute_on_fly,
                use_cached_stats=use_cached_stats,
            )
        return np.asarray(vals, dtype=float), False

    adata = data
    if color in adata.obs.columns:
        col = adata.obs[color]
        if _is_numeric_dtype(col):
            return col.to_numpy(dtype=float, na_value=np.nan), False
        return col.values, True
    # Try as gene
    try:
        return np.asarray(adata.obs_vector(color), dtype=float), False
    except KeyError:
        raise ValueError(f"'{color}' not found in obs columns or var_names.")


def _store_entity_table(ds, entity):
    """The store table ``on=entity`` names, through cytome's public accessors.

    ``ds.features(modality)`` for a feature table (``genes``, ``GA_genes``,
    ``peaks``, ``tiles``, from cytome's own modality registry) and
    ``ds.samples`` for the samples. A table the store lacks has no columns.
    """
    import cytome
    getters = {table: (lambda m=modality: ds.features(m))
               for modality, (table, _id) in cytome.MODALITY_VAR_ENTITY.items()}
    getters['samples'] = lambda: ds.samples
    table = getters[entity]() if entity in getters else None
    if table is None or not table.columns:
        have = sorted(name for name, get in getters.items() if get().columns)
        raise ValueError(f"on={entity!r}: no such feature table (have: {have}).")
    return table


def _get_entity_column(data, entity, col):
    """Per-FEATURE values for ``on != 'cells'`` — one value per row of a
    feature entity (``peaks`` / ``genes`` / ``tiles`` / ``GA_genes``, or
    ``samples``, for a cytome; ``adata.var`` for AnnData). ``'width'`` is
    derived from ``end_ - start`` (cytome) or ``end - start`` (AnnData) when
    present.

    Returns ``(values, is_categorical)``; a categorical's missing values stay
    missing (drawn in ``na_color`` as NA, as for ``on='cells'``).
    """
    if _is_cytome_input(data):
        from ..utils._cytome_compat import open_cytome as _open_cytome
        with _open_cytome(data, mode="r") as ds:
            table = _store_entity_table(ds, entity)
            if col == 'width' and 'start' in table and 'end_' in table:
                return (np.asarray(table['end_'], dtype=float)
                        - np.asarray(table['start'], dtype=float)), False
            if col not in table:
                raise ValueError(
                    f"on={entity!r}: column {col!r} not found "
                    f"(have: {sorted(table.columns)}; or 'width' if start/end_ exist).")
            s = pd.Series(table[col], dtype=object)
        numeric = pd.to_numeric(s, errors='coerce')
        if numeric.notna().sum() > 0.5 * len(s):
            return numeric.values, False
        return np.array([None if pd.isna(v) else str(v) for v in s], dtype=object), True

    # AnnData: features live in .var
    var = data.var
    if col == 'width' and {'start', 'end'} <= set(var.columns):
        return (np.asarray(var['end'].values, float)
                - np.asarray(var['start'].values, float)), False
    if col not in var.columns:
        raise ValueError(f"on={entity!r}: {col!r} not in adata.var "
                         f"(have: {list(var.columns)}).")
    vals = var[col].values
    if hasattr(vals, 'categories') or vals.dtype == object:
        return np.asarray(vals).astype(str), True
    return np.asarray(vals, dtype=float), False


def _attach_colorbar(fig, ax, mappable, label=None, side_ax=None):
    """Add a colorbar in its OWN axes so the main plot keeps its size.

    ``fig.colorbar(mappable, ax=ax)`` steals ~20% of the axes width, visibly
    squeezing a non-square scatter. A dedicated divider cax (4% width) sits
    flush against the plot and matches its height without deforming it. In
    marginals mode (``side_ax`` given) the colorbar is placed alongside the
    right marginal instead.
    """
    if side_ax is not None:
        return fig.colorbar(mappable, ax=side_ax, fraction=0.12, pad=0.05,
                            label=label)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    return fig.colorbar(mappable, cax=cax, label=label)


def scatter(
    data,
    x: str,
    y: str,
    color: Optional[str] = None,
    on: str = 'cells',
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    palette=None,
    cmap=None,
    point_size: Optional[float] = None,
    alpha: float = 1.0,
    density: str = 'auto',
    density_threshold: int = 20000,
    gridsize: int = 60,
    logx: bool = False,
    logy: bool = False,
    marginals: bool = False,
    vlines=None,
    hlines=None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_loc: str = 'right',
    legend_fontsize: int = 9,
    legend_marker_size: float = 6.0,
    square: bool = True,
    rasterized: bool = True,
    frameon: Optional[bool] = True,  # Scatter axes ARE meaningful (unlike UMAP); show by default.
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: Optional[bool] = None,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    modality: Optional[str] = None,
    cytome_layer: str = "counts",
    compute_on_fly: bool = True,
    use_cached_stats: bool = True,
    linewidths: float = 0.0,
    edgecolors="none",
    na_color="lightgray",
    sort_order="category",
):
    """Scatter plot of two features colored by a third variable.

    Parameters
    ----------
    data : AnnData or cytome Dataset/path
        Input data.
    x : str
        Feature for X axis (gene name, obs column, or peak).
    y : str
        Feature for Y axis.
    color : str, optional
        Column for coloring points. Can be categorical (e.g. ``'leiden'``)
        or continuous (e.g. a gene name).  If None, all points are grey.
    layer : str, optional
        AnnData layer. Ignored for cytome.
    use_raw : bool, optional
        Use raw attribute. Ignored for cytome.
    palette : list or dict, optional
        Colors for categorical data. Without it, the column's stored colours
        (by name, as the embedding plot uses them), then the PIASO default.
    na_color : str, default ``'lightgray'``
        Colour of points whose colour value is missing; a categorical colour
        lists them as ``"NA"`` (they used to be left out of the plot).
    sort_order : {'category', 'size', 'random'}, default ``'category'``
        Drawing order of a categorical colour; ``'size'`` puts rare
        categories on top. See ``plotEmbedding``.
    cmap : str, optional
        Colormap. Defaults to ``'magma_r'`` when ``color`` is continuous
        (sequential, perceptually uniform, colourblind-safe) and to PIASO's
        density map when ``density=True``.
        Colormap for continuous data.
    point_size : float or None
        Point size.  If None, auto-calculated from cell count.
    alpha : float
        Point transparency.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.  Default to feature names.
    legend_loc : str
        ``'right'``, ``'on_data'``, or ``'none'``.
    legend_fontsize : int
        Legend font size.
    legend_marker_size : float
        Fixed legend dot size (pt), independent of ``point_size``.
    square : bool, default True
        Keep the scatter axes square (``set_box_aspect(1)``) and, for a
        categorical ``color`` with ``legend_loc='right'``, put the legend in its
        own panel so a long category legend no longer squeezes the plot.
    rasterized : bool
        Rasterize points for smaller vector files.
    frameon : bool, optional
        Show axis frame. If None, uses ``piaso.settings._frameon``.
    vmin, vmax : float, optional
        Continuous colorscale limits.
    on : str, default ``'cells'``
        What each point is. ``'cells'`` → one point per cell, ``x``/``y`` are
        per-cell features (gene/peak/tile name or a ``cells`` column). A feature
        entity (``'peaks'`` / ``'genes'`` / ``'tiles'`` / ``'GA_genes'``, or
        ``'samples'``, for a cytome, or ``'var'``/any for AnnData) → one point
        per row of that table, ``x``/``y`` are its columns (e.g.
        ``'neg_log10_pvalue'``, ``'score'``, or the derived ``'width'`` =
        ``end_ - start``).
    density : ``'auto'`` | ``'scatter'`` | ``'hexbin'``
        ``'auto'`` switches to a log-count hexbin above ``density_threshold``
        points (unless colour is categorical). ``'scatter'`` forces points;
        ``'hexbin'`` forces density. With a continuous ``color`` the hexbin shows
        the per-cell mean.
    density_threshold : int
        Point count above which ``density='auto'`` uses hexbin.
    gridsize : int
        Hexbin grid resolution.
    logx, logy : bool
        Log-scale the X / Y axis.
    marginals : bool
        Add marginal histograms of ``x``/``y`` (only when ``ax`` is None).
    vlines, hlines : list of float, optional
        Vertical / horizontal reference lines (e.g. a ``min_length`` or
        ``score_cutoff`` threshold).
    show : bool, optional
        Call ``plt.show()``; ``None`` (default) shows unless ``ax`` is given.
    save, ax, return_fig
        Output options. Without ``return_fig`` the figure is returned when it
        is not shown.
    linewidths : float, default 0.0
        Marker edge width.
    edgecolors : str, default 'none'
        Marker edge colour.
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
    Optionally ``(fig, ax)``.
    """
    from . import color as _color_mod
    from .. import settings as _settings
    from . import _figure

    if frameon is None:
        frameon = _settings._frameon
    _created = ax is None
    show = _figure.resolve_show(show, ax)

    _cytome_kwargs = dict(
        modality=modality, cytome_layer=cytome_layer,
        compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
    )
    if on == 'cells':
        x_vals = _get_feature_vector(data, x, layer, use_raw, **_cytome_kwargs)
        y_vals = _get_feature_vector(data, y, layer, use_raw, **_cytome_kwargs)
    else:
        x_vals, _ = _get_entity_column(data, on, x)
        y_vals, _ = _get_entity_column(data, on, y)
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    n_pts = len(x_vals)
    if point_size is None:
        point_size = max(0.1, min(8, 50000 / max(n_pts, 1)))

    if figsize is None:
        figsize = plt.rcParams.get('figure.figsize', (5, 5))

    # Resolve the colour variable (per-cell or per-feature).
    color_vals, is_cat = (None, False)
    if color is not None:
        if on == 'cells':
            color_vals, is_cat = _get_color_values(data, color, **_cytome_kwargs)
        else:
            color_vals, is_cat = _get_entity_column(data, on, color)

    # Density (hexbin) rendering — for large point counts where a scatter is an
    # unreadable blob. Auto-on above density_threshold; never for categorical
    # colour (needs per-point colours).
    use_density = (density == 'hexbin') or (
        density == 'auto' and n_pts > density_threshold
        and not (color is not None and is_cat))
    if density == 'scatter':
        use_density = False

    # A categorical colour with a right-side legend gets its OWN panel so the
    # scatter stays square instead of being squeezed by a long legend (#12).
    _legend_panel = (
        ax is None and not marginals and color is not None and is_cat
        and not use_density and legend_loc == 'right')

    # Figure / axes, with optional marginal histograms.
    ax_top = ax_right = None
    leg_ax = None
    if ax is None:
        if marginals:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                                  wspace=0.04, hspace=0.04)
            ax = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        elif _legend_panel:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0.02)
            ax = fig.add_subplot(gs[0, 0])
            leg_ax = fig.add_subplot(gs[0, 1])
            leg_ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        if marginals:
            warnings.warn("scatter(marginals=True) is ignored when an existing "
                          "ax is passed.", stacklevel=2)

    if use_density:
        if cmap is None:
            cmap = _color_mod.c_color1
        xscale = 'log' if logx else 'linear'
        yscale = 'log' if logy else 'linear'
        if color is not None and not is_cat:
            hb = ax.hexbin(x_vals, y_vals, C=np.asarray(color_vals, float),
                           reduce_C_function=np.mean, gridsize=gridsize, cmap=cmap,
                           mincnt=1, xscale=xscale, yscale=yscale,
                           vmin=vmin, vmax=vmax, rasterized=rasterized)
            _attach_colorbar(fig, ax, hb, label=color, side_ax=ax_right)
        else:
            hb = ax.hexbin(x_vals, y_vals, gridsize=gridsize, cmap=cmap, bins='log',
                           mincnt=1, xscale=xscale, yscale=yscale, rasterized=rasterized)
            _attach_colorbar(fig, ax, hb, label='log10 count', side_ax=ax_right)
    elif color is not None and is_cat:
        from ._categories import (resolve_category_style, label_strings,
                                  draw_sequence, NA_LABEL)
        # Order and colours by name over the whole column -- the resolver the
        # embedding uses -- but the stored style only for per-cell colours
        # (`on='cells'`); feature-table categoricals have no stored style.
        style = resolve_category_style(
            data if on == 'cells' else None, color if on == 'cells' else None,
            color_vals, palette=palette, na_color=na_color,
            read_stored=(on == 'cells'))
        labels = label_strings(color_vals)
        missing = np.asarray(pd.isna(labels), dtype=bool)
        categories = style.present(labels)
        kw = dict(s=point_size, alpha=alpha, rasterized=rasterized,
                  linewidths=linewidths, edgecolors=edgecolors)
        if missing.any():
            # drawn first, behind the categories; they used to be dropped
            ax.scatter(x_vals[missing], y_vals[missing], c=[na_color],
                       label=NA_LABEL, **kw)
        if sort_order == 'random':
            sel = np.flatnonzero(~missing)
            np.random.default_rng(0).shuffle(sel)
            ax.scatter(x_vals[sel], y_vals[sel], c=style.rgba(labels[sel]),
                       label='_nolegend_', **kw)
        else:
            for cat in draw_sequence(style, labels, sort_order, categories=categories):
                mask = labels == cat
                ax.scatter(x_vals[mask], y_vals[mask], c=[style.color(cat)],
                           label=cat, **kw)

        if legend_loc == 'right':
            from matplotlib.lines import Line2D
            # Fixed legend-dot size, independent of point_size.
            def _h(colour, label):
                return Line2D([0], [0], marker='o', linestyle='None',
                              markersize=legend_marker_size,
                              markerfacecolor=colour, markeredgecolor='none', label=label)
            _handles = [_h(style.color(cat), cat) for cat in categories]
            if missing.any():
                _handles.append(_h(na_color, NA_LABEL))
            ncol = max(1, -(-len(_handles) // 12))
            if leg_ax is not None:
                # Dedicated legend panel keeps the scatter square (#12).
                leg_ax.legend(handles=_handles, loc='center left',
                              fontsize=legend_fontsize, frameon=False, ncol=ncol,
                              handletextpad=0.5, columnspacing=1.2, labelspacing=0.4)
            else:
                ax.legend(handles=_handles, bbox_to_anchor=(1.05, 1),
                          loc='upper left', fontsize=legend_fontsize, frameon=False,
                          ncol=ncol, handletextpad=0.5, columnspacing=1.2,
                          labelspacing=0.4)
        elif legend_loc != 'none':
            ax.legend(fontsize=legend_fontsize, frameon=False)
    elif color is not None:
        if cmap is None:
            # A continuous colour dimension -- QC fractions, scores, expression
            # -- is sequential: low to high, no meaningful midpoint. 'magma_r'
            # is perceptually uniform and colourblind-safe, so equal steps in
            # the value look like equal steps in the colour, which the
            # density-map default (c_color1) is not built for.
            #
            # Pass cmap='Spectral_r' or 'RdBu_r' for the diverging look, or
            # cmap='viridis' for the matplotlib default.
            cmap = 'magma_r'
        cv = np.asarray(color_vals, dtype=float)
        finite = np.isfinite(cv)
        kw = dict(s=point_size, alpha=alpha, rasterized=rasterized,
                  linewidths=linewidths, edgecolors=edgecolors)
        if (~finite).any():
            # NaN maps to a transparent colour; draw those points visibly
            ax.scatter(x_vals[~finite], y_vals[~finite], c=[na_color],
                       label='_nolegend_', **kw)
        order = np.flatnonzero(finite)
        order = order[np.argsort(cv[order], kind="stable")]
        sc = ax.scatter(x_vals[order], y_vals[order], c=cv[order], cmap=cmap,
                        vmin=vmin, vmax=vmax, **kw)
        _attach_colorbar(fig, ax, sc, label=color, side_ax=ax_right)
    else:
        ax.scatter(x_vals, y_vals, c='grey', s=point_size, alpha=alpha,
                   rasterized=rasterized,
                   linewidths=linewidths, edgecolors=edgecolors)

    # Log axes (hexbin already applied its own scale above).
    if logx and not use_density:
        ax.set_xscale('log')
    if logy and not use_density:
        ax.set_yscale('log')

    # Reference / threshold lines.
    for vx in (vlines or []):
        ax.axvline(vx, color='#D55E00', ls='--', lw=1.0)
    for hy in (hlines or []):
        ax.axhline(hy, color='#CC79A7', ls='--', lw=1.0)

    # Marginal histograms.
    if ax_top is not None:
        xf = x_vals[np.isfinite(x_vals)]; yf = y_vals[np.isfinite(y_vals)]
        ax_top.hist(xf, bins=80, color='#999999', linewidth=0.3, edgecolor='white')
        ax_right.hist(yf, bins=80, orientation='horizontal', color='#999999',
                      linewidth=0.3, edgecolor='white')
        ax_top.tick_params(labelbottom=False, labelsize=7)
        ax_right.tick_params(labelleft=False, labelsize=7)
        for _a in (ax_top, ax_right):
            for _s in ('top', 'right'):
                _a.spines[_s].set_visible(False)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        (ax_top if ax_top is not None else ax).set_title(title)

    if not frameon:
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        # Convention for scatter: hide top/right spines but keep bottom/left
        # so the axes read like a publication-style scatter plot.
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # Square plot box so a long legend no longer squeezes the scatter (#12).
    # box_aspect=1 fixes the axes box shape regardless of the data range; skip
    # when marginals share the axis (the gridspec already governs that shape).
    if square and ax_top is None:
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    # tight_layout emits a UserWarning when a right-margin legend +
    # colorbar can't both fit. The figure still renders correctly;
    # silence the warning rather than spamming the notebook.
    _figure.layout(fig)
    out = _figure.finish(fig, created=_created, show=show, save=save, writekey='scatter')
    if return_fig:
        return fig, ax
    return out


@wraps(scatter)
def plotScatter(*args, **kwargs):
    """Alias for :func:`scatter`.
    """
    return scatter(*args, **kwargs)
