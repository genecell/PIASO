from typing import Iterable, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ._store_io import read_cells_columns as _read_cells_columns  # read-only


def _read_cytome_features(source, feature_list, groupby=None):
    """Read feature columns from cytome cells table via the public ds.cells API."""
    needed = list(feature_list) + ([groupby] if groupby else [])
    return _read_cells_columns(source, needed)


def _count_groups(data, feature, groupby, use_raw, layer, **kw) -> int:
    """How many groups the violin will draw. 1 when there is no grouping."""
    if groupby is None:
        return 1
    _, groups = _get_feature_values(data, feature, groupby, use_raw, layer, **kw)
    return max(1, len(groups))


def _get_feature_values(
    data, feature, groupby=None, use_raw=None, layer=None,
    modality=None, cytome_layer="counts",
    compute_on_fly=True, use_cached_stats=True,
):
    """Extract feature values from AnnData or cytome for violin plotting.

    For cytome inputs:
      1. If ``feature`` is a column of ``ds.cells``, read it (e.g.
         pre-computed scores, QC metrics, or numeric annotations).
      2. Otherwise resolve via the modality registry (RNA → genes,
         GA → GA_genes, ATAC → peaks, tiles → tiles) and read the
         per-feature column from ``{modality}_{cytome_layer}``. Supports
         on-the-fly log1p/infog/tfidf via the standard PIASO
         normalization helpers (no full-matrix materialisation
         required).

    Returns ``(values, groups_dict_or_None)``.
    """
    if _is_cytome_input(data):
        from ..utils._cytome_compat import open_cytome as _open_cytome
        from ._plotEmbedding import _resolve_cytome_feature_values

        # Read groupby + feature if it's a cells column; otherwise read
        # only groupby and resolve the feature via the matrix path.
        df_cols = [groupby] if groupby else []
        df = _read_cells_columns(data, df_cols) if df_cols else None

        with _open_cytome(data, mode="r") as ds:
            cells_cols = list(ds.cells.columns)
        if feature in cells_cols:
            extra = _read_cytome_features(data, [feature], groupby)
            values = pd.to_numeric(extra[feature], errors='coerce').values.astype(float)
            if df is None:
                df = extra
        else:
            with _open_cytome(data, mode="r") as ds:
                values, _ = _resolve_cytome_feature_values(
                    ds, feature,
                    modality=modality, cytome_layer=cytome_layer,
                    compute_on_fly=compute_on_fly,
                    use_cached_stats=use_cached_stats,
                )
            values = np.asarray(values, dtype=float)
        if groupby and df is not None and groupby in df.columns:
            return values, _split_by_group(data, groupby, df[groupby], values)
        return values, None

    # AnnData path
    adata = data
    if feature in adata.obs.columns:
        values = adata.obs[feature].values
    elif use_raw and adata.raw is not None:
        values = adata.raw[:, feature].X
        if hasattr(values, 'toarray'):
            values = values.toarray()
        values = np.asarray(values).flatten()
    elif layer is not None:
        values = adata.obs_vector(feature, layer=layer)
    else:
        values = adata.obs_vector(feature)

    values = np.asarray(values, dtype=float)

    if groupby and groupby in adata.obs.columns:
        return values, _split_by_group(adata, groupby, adata.obs[groupby], values)
    return values, None


def _split_by_group(data, groupby, labels, values):
    """``{group: values}`` in the column's order (stored or categorical, else
    natural) -- on a store the stored ``set_categories`` order was ignored."""
    from ._categories import resolve_category_style, label_strings
    lab = label_strings(labels)
    groups = resolve_category_style(data, groupby, labels).present(labels)
    return {g: values[lab == g] for g in groups}


def _resolve_group_palette(data, groupby, groups, palette):
    """``{group: colour}`` for the violins, by name.

    From the resolver every plot uses: a user palette (list or dict), else the
    stored colours (``adata.uns``, a cytome's ``set_categories``, the colours
    an AnnData carried into the store), else the PIASO default, matched to the
    column's own order. The previous list was matched to the groups drawn, so
    a dropped group shifted every colour after it.
    """
    from ._categories import resolve_category_style
    if groupby is None:
        return None
    style = resolve_category_style(data, groupby, palette=palette)
    return {str(g): style.color(g) for g in groups}


def _plot_violin_on_ax(ax, values, groups_dict, feature, size, show_grid,
                       is_last, jitter=False, palette=None,
                       show_median=True, median_color='lightgrey', violin_width=0.7):
    """Draw a single violin plot on the given axes."""
    if groups_dict is not None:
        groups = list(groups_dict.keys())
        # Drop non-finite values per group before anything else. A single NaN
        # makes the gaussian KDE's covariance NaN, and matplotlib then draws an
        # invisible violin without raising -- so one missing measurement in a
        # group silently removes that group from the panel. Seen on a Xenium
        # object where `nucleus_area` is NaN for 1% of cells: every cluster but
        # one vanished, and the one that survived was the only one with no NaN.
        # The ungrouped branch below already filters this way.
        group_data = [np.asarray(groups_dict[g], dtype=float) for g in groups]
        group_data = [d[np.isfinite(d)] for d in group_data]
        # Filter out empty groups
        non_empty = [(g, d) for g, d in zip(groups, group_data) if len(d) > 0]
        if not non_empty:
            ax.set_title(feature)
            return
        groups, group_data = zip(*non_empty)
        groups, group_data = list(groups), list(group_data)

        # Colours by group name: after empty groups are dropped, a list
        # indexed by position put each later group in the colour of the one
        # before it.
        from . import color as _color_mod
        if isinstance(palette, dict):
            default = _color_mod.d_color4
            colors = [palette.get(str(g), default[i % len(default)])
                      for i, g in enumerate(groups)]
        elif palette is not None:
            colors = list(palette)[:len(groups)]
        else:
            colors = [_color_mod.d_color4[i % len(_color_mod.d_color4)]
                      for i in range(len(groups))]

        parts = ax.violinplot(group_data, positions=range(len(groups)),
                              widths=violin_width,
                              showmedians=show_median, showextrema=False)
        # Leave air either side. With one group matplotlib auto-scales x to
        # about [-0.5, 0.5], so the violin fills the panel edge to edge and
        # reads as a block rather than a distribution.
        pad = 0.35 if len(groups) <= 2 else 0.08
        ax.set_xlim(-0.5 - pad, len(groups) - 0.5 + pad)
        # Color each violin body by group
        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c)
            pc.set_alpha(1.0)
        # Style median line
        if show_median and 'cmedians' in parts:
            parts['cmedians'].set_color(median_color)
            parts['cmedians'].set_linewidth(1.5)
        # Jitter strip points (optional, off by default)
        if jitter:
            jitter_width = 0.4
            for pos, (gd, c) in enumerate(zip(group_data, colors)):
                if len(gd) > 0:
                    x_jitter = pos + np.random.default_rng(42).uniform(
                        -jitter_width * 0.3, jitter_width * 0.3, size=len(gd))
                    ax.scatter(x_jitter, gd, s=size, alpha=0.3, color=c,
                               zorder=3, rasterized=True,
                               linewidths=0, edgecolors='none')
        ax.set_xticks(range(len(groups)))
        if is_last:
            ax.set_xticklabels(groups, rotation=90)
        else:
            ax.set_xticklabels([])
    else:
        clean = values[np.isfinite(values)]
        if len(clean) > 0:
            parts = ax.violinplot([clean], positions=[0],
                                  widths=violin_width,
                                  showmedians=show_median, showextrema=False)
            ax.set_xlim(-0.85, 0.85)
            for pc in parts['bodies']:
                pc.set_alpha(1.0)
            if show_median and 'cmedians' in parts:
                parts['cmedians'].set_color(median_color)
                parts['cmedians'].set_linewidth(1.5)
            if jitter:
                x_jitter = np.random.default_rng(42).uniform(
                    -0.12, 0.12, size=len(clean))
                ax.scatter(x_jitter, clean, s=size, alpha=0.3, color='black',
                           zorder=3, rasterized=True,
                           linewidths=0, edgecolors='none')
        if is_last:
            ax.set_xticks([0])
            ax.set_xticklabels(['All cells'])
        else:
            ax.set_xticks([])

    ax.set_ylabel(feature)
    ax.set_title(feature)
    if not show_grid:
        ax.grid(False)


def plot_features_violin(data,
                         feature_list,
                         groupby: Optional[str] = None,
                         use_raw: Optional[bool] = None,
                         layer: Optional[str] = None,
                         palette=None,
                         jitter: bool = False,
                         width_single: float = None,
                         height_single: float = 2.0,
                         ncol: Optional[int] = None,
                         violin_width: float = 0.7,
                         size: float = 0.1,
                         show_grid: bool = True,
                         show_median: bool = True,
                         median_color: str = 'lightgrey',
                         show_figure: bool = True,
                         save: Optional[str] = None,
                         modality: Optional[str] = None,
                         cytome_layer: str = "counts",
                         compute_on_fly: bool = True,
                         use_cached_stats: bool = True,
                         show: Optional[bool] = None,
                         return_fig: bool = False,
                         ):
    """
    Plots a violin plot for each feature specified in `feature_list`.

    Uses matplotlib directly (no scanpy dependency). Supports AnnData and
    cytome Dataset / .cytome file path.

    ``show`` is accepted as an alias for ``show_figure``: every other plotting
    function in ``piaso.pl`` takes ``show``, and a call that works for
    ``pl.embedding`` and ``pl.dotplot`` should not raise here for the sake of
    one function's parameter name. When both are given, ``show`` wins.

    Parameters
    ----------
    data : anndata.AnnData, cytome.Dataset, or str
        The data source. For AnnData, reads from obs, layers, or raw.
        For cytome, reads from the cells SQL table.
    feature_list : List[str]
        Feature names to visualize. For cytome, these must be column names
        in the cells table.
    groupby : str, optional
        Column to group data points by. Default is None.
    use_raw : bool, optional
        Use raw attribute of adata. Ignored for cytome.
    layer : str, optional
        AnnData layer to use. Ignored for cytome.
    palette : list or dict, optional
        Color palette for groups. If None, the column's stored colours by name
        (``adata.uns['{groupby}_colors']``, a cytome's ``set_categories``),
        then the default PIASO palette.
    jitter : bool, optional
        Show jitter scatter points on violins. Default is False.
    width_single : float, optional
        Figure width in inches. ``None`` (default) derives it from the number
        of groups — ``1.6 + 0.42 * n_groups``, clamped to [3, 16] — so a
        two-sample plot is not the same width as a forty-cluster one. Pass a
        number to override.
        Width of each subplot. Default is 14.0.
    violin_width : float, optional
        Width of each violin body in x-axis units, where 1.0 is the spacing
        between groups. Default 0.7.
    ncol : int, optional
        Panels per row. ``None`` (default) puts ungrouped features two to a
        row and grouped ones one to a row, since a grouped panel is already
        as wide as its group count requires.
    height_single : float, optional
        Height of each subplot. Default is 2.0.
    size : float, optional
        Jitter point size (only used when jitter=True). Default is 0.1.
    show_grid : bool, optional
        Show grid lines. Default is True.
    show_median : bool, optional
        Show median line on violins. Default is True.
    median_color : str, optional
        Color of the median line. Default is ``'lightgrey'``.
    show_figure : bool, optional
        Show figure (plt.show()). Default is True. When not shown the figure
        is returned, and closed only if it was saved.
    save : str, optional
        Path to save the figure. Default is None.
    return_fig : bool, optional
        Return ``(fig, axes)`` instead of closing the figure.
    
    show : bool, optional
        Show the figure; ``None`` follows ``show_figure``.
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
    """
    if show is not None:
        show_figure = show

    n_features = len(feature_list)
    _kw = dict(modality=modality, cytome_layer=cytome_layer,
               compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats)
    # Each feature is read once. The first was read three times (to count the
    # groups, to peek at them for the palette, and to draw), each a pass over
    # the store's matrix for a gene. The others are read as they are drawn, so
    # only one feature's values are held at a time.
    first = (_get_feature_values(data, feature_list[0], groupby, use_raw, layer, **_kw)
             if feature_list else (None, None))

    # Width from the number of groups, not a constant. A fixed 14 inches is
    # right for ~30 Leiden clusters and absurd for two samples: the same plot
    # then arrives as a wide strip with two violins lost in it. Scale with the
    # group count, clamped so one group is still readable and eighty do not
    # produce a figure no screen can show.
    _first_groups = first[1]
    n_groups = 1 if groupby is None else max(1, len(_first_groups or {}))
    if width_single is None:
        # The floor used to be 3.0, which is most of a text column for a plot
        # holding one violin. An ungrouped QC panel needs far less than that.
        floor = 2.2 if n_groups <= 2 else 3.0
        width_single = float(np.clip(1.4 + 0.40 * n_groups, floor, 16.0))

    # Several narrow panels side by side read better than a tall single column
    # of them, so ungrouped features default to two per row. A grouped panel is
    # already wide, so it keeps the column.
    if ncol is None:
        ncol = 1 if (groupby is not None and n_groups > 2) else min(n_features, 2)
    ncol = max(1, min(int(ncol), max(n_features, 1)))
    nrow = int(np.ceil(max(n_features, 1) / ncol))

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                             figsize=(width_single * ncol, height_single * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for extra_ax in axes[n_features:]:
        extra_ax.axis("off")

    # Resolve palette once for all features, by group name
    resolved_palette = None
    if groupby is not None and _first_groups is not None:
        resolved_palette = _resolve_group_palette(data, groupby, list(_first_groups),
                                                  palette)

    for i, feature in enumerate(feature_list):
        values, groups_dict = (first if i == 0 else
                               _get_feature_values(data, feature, groupby, use_raw, layer, **_kw))
        # x labels belong on the bottom row of the grid, which is not the
        # last feature once ncol > 1.
        is_last = (i >= n_features - ncol)
        _plot_violin_on_ax(axes[i], values, groups_dict, feature, size,
                           show_grid, is_last, jitter=jitter,
                           palette=resolved_palette,
                           show_median=show_median,
                           median_color=median_color,
                           violin_width=violin_width)

    from . import _figure
    _figure.layout(fig)
    out = _figure.finish(fig, created=True, show=bool(show_figure), save=save,
                         writekey='violin')
    if return_fig:
        return fig, axes[:n_features]
    return out


from functools import wraps
from ._group_order import resolve_group_order
# Create the alias
@wraps(plot_features_violin)
def plotFeaturesViolin(*args, **kwargs):
    """
    Alias for :func:`plot_features_violin`.

    Please refer to the main function for full documentation.
    """
    return plot_features_violin(*args, **kwargs)