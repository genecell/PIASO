"""Plot per-group (cell-type) summary metrics from ``piaso.pp.calculateGroupMetrics``.

Two layouts:
- ``kind='bar'`` (default): a faceted grid, one small barplot per metric (groups
  on x). Metrics have very different scales, so each gets its own panel/axis.
- ``kind='heatmap'``: a groups × metrics matrix, each column normalised
  (min-max or z-score) for comparability, with the raw value annotated per cell.

Bars are coloured by the cell-type colours from the ``set_categories`` store when
available (via ``df.attrs['colors']`` set by ``calculateGroupMetrics``, an
explicit ``palette``, or by re-resolving from ``data`` + ``groupby``).
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def _resolve_group_colors(df, data, groupby, palette):
    """Return ``{group: hex}`` for the df index, best-effort."""
    groups = [str(g) for g in df.index]
    from . import color as _color_mod
    default = _color_mod.d_color4

    # 1. explicit palette (dict or list)
    if isinstance(palette, dict):
        return {g: palette.get(g, default[i % len(default)]) for i, g in enumerate(groups)}
    if isinstance(palette, (list, tuple)) and len(palette):
        return {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    # 2. colours stashed on the df by calculateGroupMetrics
    stashed = df.attrs.get("colors")
    if isinstance(stashed, dict) and stashed:
        return {g: stashed.get(g, default[i % len(default)]) for i, g in enumerate(groups)}

    # 3. re-resolve from the source via the shared category store
    gb = groupby or df.attrs.get("groupby")
    if data is not None and gb is not None:
        try:
            from ._plotEmbedding import _resolve_categorical_style
            pal, _order = _resolve_categorical_style(data, gb)
            if isinstance(pal, dict) and pal:
                return {g: pal.get(g, default[i % len(default)]) for i, g in enumerate(groups)}
        except Exception:
            pass

    # 4. default palette per group
    return {g: default[i % len(default)] for i, g in enumerate(groups)}


def plotGroupMetrics(
    df,
    data=None,
    groupby: Optional[str] = None,
    metrics: Optional[list] = None,
    kind: str = "bar",
    palette=None,
    ncol: Optional[int] = None,
    figsize: Optional[tuple] = None,
    normalize: str = "minmax",
    annotate: bool = True,
    cmap: str = "Blues",
    fontsize: Optional[float] = None,
    rotation: float = 45,
    save: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
):
    """Plot the per-group metrics DataFrame from :func:`piaso.pp.calculateGroupMetrics`.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``calculateGroupMetrics`` (rows = groups, cols = metrics).
    data, groupby : optional
        Source dataset + grouping column, only used to re-resolve cell-type
        colours when the df doesn't already carry them in ``df.attrs['colors']``.
    metrics : list of str, optional
        Subset / order of metric columns to plot. Default: all numeric columns.
    kind : ``'bar'`` | ``'heatmap'``
        ``'bar'`` (default): one panel per metric. ``'heatmap'``: groups × metrics
        with per-column normalisation.
    palette : dict or list, optional
        Override group colours (``{group: hex}`` or an ordered list).
    ncol : int, optional
        Columns in the faceted bar grid (default: ~sqrt(n_metrics)).
    figsize : tuple, optional
        Auto-sized if None.
    normalize : ``'minmax'`` | ``'zscore'`` | ``None``
        Per-column scaling for the heatmap (ignored for bars).
    annotate : bool, default True
        Write the raw value in each heatmap cell.
    cmap : str, default ``'Blues'``
        Heatmap colormap.
    fontsize : float, optional
        Base font size (defaults to ``rcParams['font.size']``).
    rotation : float, default 45
        Group-label rotation (anchored to the ticks so names don't overlap).
    save, show, return_fig
        Output options.
    """
    import pandas as pd

    _fs = float(fontsize) if fontsize is not None else float(
        plt.rcParams.get("font.size", 10) or 10)

    num = df.select_dtypes(include=[np.number])
    cols = [c for c in (metrics or list(num.columns)) if c in num.columns]
    if not cols:
        raise ValueError("No numeric metric columns to plot.")
    groups = [str(g) for g in df.index]
    colors = _resolve_group_colors(df, data, groupby, palette)

    if kind == "bar":
        n = len(cols)
        if ncol is None:
            ncol = max(1, int(np.ceil(np.sqrt(n))))
        nrow = int(np.ceil(n / ncol))
        if figsize is None:
            figsize = (ncol * max(3.0, 0.32 * len(groups) + 1.2), nrow * 2.8)
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        bar_colors = [colors[g] for g in groups]
        x = np.arange(len(groups))
        for k, col in enumerate(cols):
            ax = axs[k // ncol][k % ncol]
            ax.bar(x, df[col].to_numpy(), color=bar_colors, edgecolor="none")
            ax.set_title(col, fontsize=_fs)
            ax.set_xticks(x)
            # Rotate + anchor to the ticks so long cell-type names don't overlap.
            ax.set_xticklabels(groups, rotation=rotation, ha="right",
                               rotation_mode="anchor", fontsize=_fs * 0.8)
            ax.tick_params(axis="y", labelsize=_fs * 0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        # Hide unused panels.
        for k in range(n, nrow * ncol):
            axs[k // ncol][k % ncol].axis("off")
        fig.tight_layout()

    elif kind == "heatmap":
        M = df[cols].to_numpy(dtype=float)
        Mn = M.copy()
        if normalize == "minmax":
            lo = np.nanmin(M, axis=0); hi = np.nanmax(M, axis=0)
            rng = np.where((hi - lo) == 0, 1.0, hi - lo)
            Mn = (M - lo) / rng
        elif normalize == "zscore":
            mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0)
            Mn = (M - mu) / np.where(sd == 0, 1.0, sd)
        if figsize is None:
            figsize = (max(4.0, 0.7 * len(cols) + 2.0), max(3.0, 0.4 * len(groups) + 1.5))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(Mn, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=rotation, ha="right",
                           rotation_mode="anchor", fontsize=_fs * 0.8)
        ax.set_yticks(np.arange(len(groups)))
        ax.set_yticklabels(groups, fontsize=_fs * 0.8)
        if annotate:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    txt = f"{v:.0f}" if abs(v) >= 10 else f"{v:.2g}"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=_fs * 0.6,
                            color="white" if Mn[i, j] > 0.6 else "black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=_fs * 0.6)
        cbar.set_label(f"{normalize}-scaled" if normalize else "value", fontsize=_fs * 0.7)
        fig.tight_layout()
    else:
        raise ValueError(f"kind must be 'bar' or 'heatmap', got {kind!r}.")

    from ..settings import _savefig
    _savefig(fig, save, writekey="group_metrics")
    if show:
        plt.show()
    else:
        plt.close(fig)
    if return_fig:
        return fig


# snake_case alias, consistent with the other piaso.pl entry points
plot_group_metrics = plotGroupMetrics
