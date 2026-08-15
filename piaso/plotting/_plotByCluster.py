from typing import Iterable, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ..utils._cytome_compat import read_cells_columns as _read_cells_columns


def _read_cytome_features(source, feature_list, groupby=None):
    """Read feature columns from cytome cells table via the public ds.cells API."""
    needed = list(feature_list) + ([groupby] if groupby else [])
    return _read_cells_columns(source, needed)


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

        with _open_cytome(data) as ds:
            cells_cols = list(ds.cells.columns)
        if feature in cells_cols:
            extra = _read_cytome_features(data, [feature], groupby)
            values = pd.to_numeric(extra[feature], errors='coerce').values.astype(float)
            if df is None:
                df = extra
        else:
            with _open_cytome(data) as ds:
                values, _ = _resolve_cytome_feature_values(
                    ds, feature,
                    modality=modality, cytome_layer=cytome_layer,
                    compute_on_fly=compute_on_fly,
                    use_cached_stats=use_cached_stats,
                )
            values = np.asarray(values, dtype=float)
        if groupby and df is not None and groupby in df.columns:
            group_labels = df[groupby].values
            groups = sorted(set(str(g) for g in group_labels if pd.notna(g)))
            mask_for = lambda g: np.array(
                [str(x) == g for x in group_labels], dtype=bool,
            )
            groups_dict = {g: values[mask_for(g)] for g in groups}
            return values, groups_dict
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
        group_labels = adata.obs[groupby].values
        groups = sorted(set(str(g) for g in group_labels if pd.notna(g)))
        groups_dict = {g: values[np.array([str(x) == g for x in group_labels])]
                       for g in groups}
        return values, groups_dict
    return values, None


def _resolve_group_palette(data, groupby, groups, palette):
    """Resolve color palette for groups.

    Priority:
    1. User-supplied palette (list or dict)
    2. adata.uns[f'{groupby}_colors'] (AnnData convention)
    3. Default PIASO palette (d_color4)
    """
    from . import color as _color_mod

    if palette is not None:
        if isinstance(palette, dict):
            return [palette.get(g, _color_mod.d_color4[i % len(_color_mod.d_color4)])
                    for i, g in enumerate(groups)]
        return list(palette)

    # Check adata.uns for stored palette
    if not _is_cytome_input(data):
        palette_key = f'{groupby}_colors'
        if hasattr(data, 'uns') and palette_key in data.uns:
            stored = data.uns[palette_key]
            if len(stored) >= len(groups):
                return list(stored[:len(groups)])

    # Check cytome metadata for stored palette via the public API
    if _is_cytome_input(data):
        from ..utils._cytome_compat import get_metadata as _get_metadata
        try:
            stored = _get_metadata(data, f'{groupby}_colors')
            if stored is not None and len(stored) >= len(groups):
                return list(stored[:len(groups)])
        except Exception:
            pass

    return [_color_mod.d_color4[i % len(_color_mod.d_color4)] for i in range(len(groups))]


def _plot_violin_on_ax(ax, values, groups_dict, feature, size, show_grid,
                       is_last, jitter=False, palette=None,
                       show_median=True, median_color='lightgrey'):
    """Draw a single violin plot on the given axes."""
    if groups_dict is not None:
        groups = list(groups_dict.keys())
        group_data = [groups_dict[g] for g in groups]
        # Filter out empty groups
        non_empty = [(g, d) for g, d in zip(groups, group_data) if len(d) > 0]
        if not non_empty:
            ax.set_title(feature)
            return
        groups, group_data = zip(*non_empty)
        groups, group_data = list(groups), list(group_data)

        # Resolve colors for each group
        if palette is None:
            from . import color as _color_mod
            colors = [_color_mod.d_color4[i % len(_color_mod.d_color4)]
                      for i in range(len(groups))]
        else:
            colors = palette[:len(groups)]

        parts = ax.violinplot(group_data, positions=range(len(groups)),
                              showmedians=show_median, showextrema=False)
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
                               zorder=3, rasterized=True)
        ax.set_xticks(range(len(groups)))
        if is_last:
            ax.set_xticklabels(groups, rotation=90)
        else:
            ax.set_xticklabels([])
    else:
        clean = values[np.isfinite(values)]
        if len(clean) > 0:
            parts = ax.violinplot([clean], positions=[0],
                                  showmedians=show_median, showextrema=False)
            for pc in parts['bodies']:
                pc.set_alpha(1.0)
            if show_median and 'cmedians' in parts:
                parts['cmedians'].set_color(median_color)
                parts['cmedians'].set_linewidth(1.5)
            if jitter:
                x_jitter = np.random.default_rng(42).uniform(
                    -0.12, 0.12, size=len(clean))
                ax.scatter(x_jitter, clean, s=size, alpha=0.3, color='black',
                           zorder=3, rasterized=True)
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
                         width_single: float = 14.0,
                         height_single: float = 2.0,
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
                         ):
    """
    Plots a violin plot for each feature specified in `feature_list`.

    Uses matplotlib directly (no scanpy dependency). Supports AnnData and
    cytome Dataset / .cytome file path.

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
        Color palette for groups. If None, checks ``adata.uns['{groupby}_colors']``
        first, then falls back to the default PIASO palette.
    jitter : bool, optional
        Show jitter scatter points on violins. Default is False.
    width_single : float, optional
        Width of each subplot. Default is 14.0.
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
        Show figure (plt.show()). Default is True.
    save : str, optional
        Path to save the figure. Default is None.
    """
    n_features = len(feature_list)
    fig, axes = plt.subplots(nrows=max(n_features, 1), ncols=1,
                             figsize=(width_single, height_single * max(n_features, 1)),
                             squeeze=False)
    axes = axes.ravel()

    # Resolve palette once for all features
    resolved_palette = None
    if groupby is not None:
        # Peek at groups from first feature to build palette
        _, groups_dict_peek = _get_feature_values(
            data, feature_list[0], groupby, use_raw, layer,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
        )
        if groups_dict_peek is not None:
            groups_peek = list(groups_dict_peek.keys())
            resolved_palette = _resolve_group_palette(data, groupby, groups_peek, palette)

    for i, feature in enumerate(feature_list):
        values, groups_dict = _get_feature_values(
            data, feature, groupby, use_raw, layer,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
        )
        is_last = (i == n_features - 1)
        _plot_violin_on_ax(axes[i], values, groups_dict, feature, size,
                           show_grid, is_last, jitter=jitter,
                           palette=resolved_palette,
                           show_median=show_median,
                           median_color=median_color)

    plt.tight_layout()

    from ..settings import _savefig
    _savefig(fig, save, writekey='violin')
    if show_figure:
        plt.show()
    else:
        plt.close(fig)


from functools import wraps
# Create the alias
@wraps(plot_features_violin)
def plotFeaturesViolin(*args, **kwargs):
    """
    Alias for :func:`plot_features_violin`.

    Please refer to the main function for full documentation.
    """
    return plot_features_violin(*args, **kwargs)