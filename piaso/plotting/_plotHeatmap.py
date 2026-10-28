"""Expression heatmap for features across cell groups.

Displays mean expression as a color matrix (groups x features).
Supports AnnData and cytome Dataset.
"""

from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import wraps


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ._store_io import read_cells_columns as _read_cells_columns  # read-only
from ._group_order import resolve_group_order


def _resolve_features(features):
    """Resolve *features* into a flat list plus optional group annotations.

    Accepted inputs: list (unchanged), dict (keys=group labels, values=gene lists),
    or DataFrame with 'group' and 'gene'/'feature' columns.

    Returns (features_flat, var_group_labels_or_None, var_group_positions_or_None).
    """
    if isinstance(features, dict):
        flat, labels, positions = [], [], []
        for label, genes in features.items():
            if isinstance(genes, str):
                genes = [genes]
            start = len(flat)
            flat.extend(genes)
            labels.append(label)
            positions.append((start, len(flat) - 1))
        return flat, labels, positions

    if isinstance(features, pd.DataFrame):
        gene_col = 'gene' if 'gene' in features.columns else 'feature'
        if gene_col not in features.columns or 'group' not in features.columns:
            raise ValueError(
                "DataFrame features must have 'group' and 'gene' (or 'feature') columns."
            )
        flat, labels, positions = [], [], []
        for grp, sub in features.groupby('group', sort=False):
            start = len(flat)
            flat.extend(sub[gene_col].tolist())
            labels.append(str(grp))
            positions.append((start, len(flat) - 1))
        return flat, labels, positions

    return list(features), None, None


def _feature_block(adata, feats, layer=None, use_raw=None):
    """``(n_cells, len(feats))`` values of matrix features, sparse or dense."""
    if use_raw and adata.raw is not None:
        return adata.raw[:, feats].X
    if layer is not None:
        return adata[:, feats].layers[layer]
    return adata[:, feats].X


def _nan_group_means(block, indicator):
    """Group means that skip NaN (as ``np.nanmean``): ``(means, n_finite)``.

    One sparse product over all cells for all features at once; the per-group
    string comparison it replaces was a Python loop over every cell, for every
    group and every feature (5.2 s against 0.08 s at 50,000 cells).
    """
    import scipy.sparse as sp
    sizes = np.asarray(indicator.sum(axis=1), dtype=float).ravel()[:, None]
    if sp.issparse(block):
        block = block.tocsr().astype(np.float64)
        bad = ~np.isfinite(block.data)
        if bad.any():
            nanpart = block.copy()
            nanpart.data = bad.astype(np.float64)
            block = block.copy()
            block.data[bad] = 0.0
            n_bad = np.asarray((indicator @ nanpart).todense(), dtype=float)
        else:
            n_bad = 0.0
        sums = np.asarray((indicator @ block).todense(), dtype=float)
        counts = sizes - n_bad
    else:
        dense = np.asarray(block, dtype=np.float64)
        if dense.ndim == 1:
            dense = dense[:, None]
        finite = np.isfinite(dense)
        sums = np.asarray(indicator @ np.where(finite, dense, 0.0), dtype=float)
        counts = np.asarray(indicator @ finite.astype(np.float64), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    # a group with no cells reads 0, as before; one whose values are all NaN
    # reads NaN, as np.nanmean gives
    means = np.where(np.broadcast_to(sizes, means.shape) > 0, means, 0.0)
    return means, counts


def _anndata_group_means(adata, features, groupby, groups, layer=None, use_raw=None,
                         skip_missing=False):
    """Per-group mean of each feature (``obs`` column or gene) -> groups x features.

    NaN values are left out of a group's mean; a group with no cells reads 0.
    ``skip_missing`` drops features that are neither an ``obs`` column nor a
    gene, instead of raising.
    """
    from ._plotDotplot import _group_indicator
    labels = adata.obs[groupby].values
    indicator = _group_indicator(labels, groups)
    names = (adata.raw.var_names if (use_raw and adata.raw is not None)
             else adata.var_names)
    obs_feats = [f for f in features if f in adata.obs.columns]
    mat_feats = [f for f in features if f not in adata.obs.columns]
    if skip_missing:
        mat_feats = [f for f in mat_feats if f in names]
    blocks = {}
    if mat_feats:
        means, _ = _nan_group_means(_feature_block(adata, mat_feats, layer, use_raw),
                                    indicator)
        for j, f in enumerate(mat_feats):
            blocks[f] = means[:, j]
    if obs_feats:
        vals = np.column_stack([pd.to_numeric(adata.obs[f], errors="coerce")
                                .to_numpy(dtype=float, na_value=np.nan)
                                for f in obs_feats])
        means, _ = _nan_group_means(vals, indicator)
        for j, f in enumerate(obs_feats):
            blocks[f] = means[:, j]
    keep = [f for f in features if f in blocks]
    out = pd.DataFrame({f: blocks[f] for f in keep}, index=[str(g) for g in groups])
    return out


def _group_order(data, groupby, labels, categories_order=None):
    """Groups present, in the caller's order, else the stored/categorical one."""
    from ._categories import resolve_category_style
    style = resolve_category_style(data, groupby, labels, order=categories_order)
    return style.present(labels)


def _get_mean_expression(data, features, groupby, layer=None, use_raw=None,
                         modality=None, cytome_layer="counts",
                         compute_on_fly=True, use_cached_stats=True):
    """Compute mean expression per group.

    Returns DataFrame: groups x features.
    """
    if _is_cytome_input(data):
        # Reuse the dotplot helper which routes obs/cells columns to the
        # cells path and gene/peak features to the matrix path via
        # _resolve_cytome_feature_values.
        from ._plotDotplot import _get_expression_data_cytome
        _, mean_df = _get_expression_data_cytome(
            data, list(features), groupby,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly,
            use_cached_stats=use_cached_stats,
        )
        return mean_df

    # AnnData
    adata = data
    groups = _group_order(adata, groupby, adata.obs[groupby])
    mean_df = _anndata_group_means(adata, list(features), groupby, groups,
                                   layer=layer, use_raw=use_raw)
    mean_df.index.name = groupby
    return mean_df


def _get_cell_expression(data, features, groupby, layer=None, use_raw=None,
                         max_cells_per_group=100, random_state=42,
                         categories_order=None,
                         modality=None, cytome_layer="counts",
                         compute_on_fly=True, use_cached_stats=True):
    """Get per-cell expression matrix with group-stratified sampling.

    Returns (matrix, group_labels, group_sizes) where:
    - matrix: ndarray (n_sampled_cells, n_features)
    - group_labels: list of group names per row
    - group_sizes: dict mapping group → n_cells in matrix, in display order

    The groups follow ``categories_order``, else the column's stored or
    categorical order (natural order otherwise) -- the order every other plot
    uses. The cytome path sorted them as strings (0, 1, 10, 2).
    """
    import scipy.sparse as sp
    from ._categories import label_strings

    if _is_cytome_input(data):
        from ..utils._cytome_compat import open_cytome as _open_cytome
        from ._plotEmbedding import _resolve_cytome_feature_values_batch

        with _open_cytome(data, mode="r") as ds:
            labels_all = np.asarray(ds.cells[groupby])
            cells_cols = set(ds.cells.columns)
            cells_feats = [f for f in features if f in cells_cols]
            matrix_feats = [f for f in features if f not in cells_cols]
            feature_vecs = {}
            for f in cells_feats:
                feature_vecs[f] = pd.to_numeric(pd.Series(np.asarray(ds.cells[f])),
                                                errors='coerce').to_numpy(dtype=float)
            if matrix_feats:
                # one streaming pass for every feature, not one per feature
                batched = _resolve_cytome_feature_values_batch(
                    ds, list(matrix_feats), modality=modality,
                    cytome_layer=cytome_layer, compute_on_fly=compute_on_fly,
                    use_cached_stats=use_cached_stats)
                for f in matrix_feats:
                    feature_vecs[f] = np.asarray(batched[f][0], dtype=float)
        full = (np.column_stack([feature_vecs[f] for f in features])
                if features else np.zeros((len(labels_all), 0)))
    else:
        labels_all = data.obs[groupby]
        full = None

    groups = _group_order(data, groupby, labels_all, categories_order=categories_order)
    lab = label_strings(labels_all)
    rng = np.random.RandomState(random_state)
    picked, labels, sizes = [], [], {}
    for g in groups:
        cell_idx = np.flatnonzero(lab == g)
        n = min(len(cell_idx), max_cells_per_group)
        if n < len(cell_idx):
            cell_idx = rng.choice(cell_idx, n, replace=False)
        picked.append(cell_idx)
        labels.extend([g] * n)
        sizes[g] = n
    if not picked:
        return np.zeros((0, len(features))), [], {}
    rows = np.concatenate(picked)
    if full is not None:
        return full[rows], labels, sizes

    # AnnData: only the sampled cells are read, all features at once
    adata = data
    cols = []
    mat_feats = [f for f in features if f not in adata.obs.columns]
    block = None
    if mat_feats:
        block = _feature_block(adata[rows], mat_feats, layer, use_raw)
        block = np.asarray(block.toarray() if sp.issparse(block) else block, dtype=float)
    j = 0
    for f in features:
        if f in adata.obs.columns:
            cols.append(pd.to_numeric(adata.obs[f].iloc[rows], errors="coerce")
                        .to_numpy(dtype=float, na_value=np.nan))
        else:
            cols.append(block[:, j])
            j += 1
    matrix = np.column_stack(cols) if cols else np.zeros((len(rows), 0))
    return matrix, labels, sizes


def heatmap(
    data,
    features: list,
    groupby: str = 'leiden',
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    standard_scale: Optional[str] = None,
    log: bool = False,
    cmap: str = 'viridis',
    figsize: Optional[tuple] = None,
    categories_order: Optional[list] = None,
    var_names_order: Optional[list] = None,
    dendrogram: bool = False,
    swap_axes: bool = False,
    title: Optional[str] = None,
    show_values: bool = False,
    fmt: str = '.2f',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cell_level: bool = False,
    max_cells_per_group: int = 100,
    show_group_colors: bool = True,
    show: Optional[bool] = None,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    modality: Optional[str] = None,
    cytome_layer: str = "counts",
    compute_on_fly: bool = True,
    use_cached_stats: bool = True,
    palette=None,
):
    """Plot an expression heatmap.

    Two modes:
    - **Group-level** (default): mean expression per group (groups x features).
    - **Cell-level** (``cell_level=True``): per-cell expression with group-stratified
      sampling, up to ``max_cells_per_group`` cells per group.

    Parameters
    ----------
    data : AnnData or cytome Dataset/path
        Input data.
    features : list, dict, or DataFrame
        Feature names.  A plain list of strings, a dict mapping group labels
        to gene lists, or a DataFrame with ``'group'`` and ``'gene'``/``'feature'``
        columns.  When dict/DataFrame, gene groups are shown as colored brackets
        (cell-level) or vertical separators (group-level).
    groupby : str
        Grouping column.
    layer, use_raw
        AnnData layer / raw attribute. Ignored for cytome.
    standard_scale : str, optional
        ``'var'`` to z-score per feature, ``'group'`` per group (group-level);
        the colorbar then reads "z-score". (``dotplot`` min-max scales
        instead.) In the cell-level mode only ``'var'`` applies.
    log : bool
        Log1p transform.
    cmap : str
        Colormap.
    figsize : tuple, optional
        Auto-calculated if None.
    categories_order : list, optional
        Custom group order.
    var_names_order : list, optional
        Custom feature order.
    dendrogram : bool
        Cluster groups by hierarchical clustering (group-level only).
    swap_axes : bool
        Transpose: features on Y, groups on X.
    title : str, optional
        Title.
    show_values : bool
        Annotate cells with values (group-level only).
    fmt : str
        Number format for annotations.
    vmin, vmax : float, optional
        Colorscale limits.
    cell_level : bool
        If True, show per-cell expression instead of group means.
    max_cells_per_group : int
        Maximum cells per group when ``cell_level=True``. Default 100.
    show_group_colors : bool
        Show colored sidebar for groups when ``cell_level=True``. Its colours
        are the column's own (stored, by name) -- the colours of the embedding.
    palette : list or dict, optional
        Colours for that sidebar, overriding the stored ones.
    show : bool, optional
        Call ``plt.show()``; ``None`` (default) shows unless ``ax`` is given.
    save, ax, return_fig
        Output options. Without ``return_fig`` the figure is returned when it
        is not shown.
    
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
    # --- Resolve dict / DataFrame features ---
    features, _auto_labels, _auto_positions = _resolve_features(features)

    from . import _figure
    show = _figure.resolve_show(show, ax)

    if cell_level:
        # The cytome arguments were used inside and never passed in, so every
        # cell-level call raised NameError (audit 2026-10-27, finding 7).
        return _heatmap_cell_level(
            data, features, groupby, layer, use_raw, standard_scale, log,
            cmap, figsize, categories_order, var_names_order, title,
            vmin, vmax, max_cells_per_group, show_group_colors,
            show, save, ax, return_fig,
            var_group_labels=_auto_labels, var_group_positions=_auto_positions,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
            palette=palette)

    mean_df = _get_mean_expression(
        data, features, groupby, layer, use_raw,
        modality=modality, cytome_layer=cytome_layer,
        compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
    )

    # Reorder groups
    if categories_order is not None:
        mean_df = mean_df.loc[[c for c in categories_order if c in mean_df.index]]
    elif dendrogram and mean_df.shape[0] >= 2:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        dist = pdist(mean_df.values, metric='euclidean')
        Z = linkage(dist, method='ward')
        order = leaves_list(Z)
        mean_df = mean_df.iloc[order]

    # Reorder features
    if var_names_order is not None:
        mean_df = mean_df[[f for f in var_names_order if f in mean_df.columns]]

    # Transform
    if log:
        mean_df = np.log1p(mean_df)
    if standard_scale == 'var':
        mean_df = (mean_df - mean_df.mean()) / (mean_df.std() + 1e-12)
    elif standard_scale == 'group':
        mean_df = mean_df.T
        mean_df = (mean_df - mean_df.mean()) / (mean_df.std() + 1e-12)
        mean_df = mean_df.T

    if swap_axes:
        mean_df = mean_df.T

    if figsize is None:
        n_rows, n_cols = mean_df.shape
        figsize = (max(3, n_cols * 0.4 + 1.5), max(2.5, n_rows * 0.3 + 1))

    fig, ax, _created = _figure.begin(ax, figsize=figsize)

    im = ax.imshow(mean_df.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    ax.set_xticks(range(mean_df.shape[1]))
    ax.set_xticklabels(mean_df.columns, rotation=90, ha='center', fontsize=9)
    ax.set_yticks(range(mean_df.shape[0]))
    ax.set_yticklabels(mean_df.index, fontsize=9)

    if show_values:
        for i in range(mean_df.shape[0]):
            for j in range(mean_df.shape[1]):
                val = mean_df.values[i, j]
                color = 'white' if val > (mean_df.values.max() + mean_df.values.min()) / 2 else 'black'
                ax.text(j, i, format(val, fmt), ha='center', va='center',
                        fontsize=7, color=color)

    # Gene group separator lines
    if _auto_positions is not None:
        for i, (start, end) in enumerate(_auto_positions[:-1]):
            next_start = _auto_positions[i + 1][0]
            sep_x = (end + next_start) / 2
            ax.axvline(sep_x, color='white', lw=2, zorder=2)

    # the bar says what the colours are: a z-score once standard_scale is set
    _cbar_label = ('Mean expression' if standard_scale not in ('var', 'group')
                   else f"z-score (per {'feature' if standard_scale == 'var' else 'group'})")
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label=_cbar_label)

    if title:
        ax.set_title(title)

    _figure.layout(fig)
    out = _figure.finish(fig, created=_created, show=show, save=save, writekey='heatmap')
    if return_fig:
        return fig, ax
    return out


def _heatmap_cell_level(data, features, groupby, layer, use_raw,
                        standard_scale, log, cmap, figsize,
                        categories_order, var_names_order,
                        title, vmin, vmax, max_cells_per_group,
                        show_group_colors, show, save, ax, return_fig,
                        var_group_labels=None, var_group_positions=None,
                        modality=None, cytome_layer="counts",
                        compute_on_fly=True, use_cached_stats=True, palette=None):
    """Cell-level heatmap with group-stratified sampling."""
    from . import _figure
    from ._categories import resolve_category_style

    # Reorder features if requested
    if var_names_order is not None:
        features = [f for f in var_names_order if f in features]

    matrix, labels, sizes = _get_cell_expression(
        data, features, groupby, layer, use_raw,
        max_cells_per_group=max_cells_per_group,
        categories_order=categories_order,
        modality=modality, cytome_layer=cytome_layer,
        compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats)

    # matrix: (n_cells, n_features)
    if log:
        matrix = np.log1p(matrix)
    if standard_scale == 'var':
        mu = np.nanmean(matrix, axis=0, keepdims=True)
        sd = np.nanstd(matrix, axis=0, keepdims=True) + 1e-12
        matrix = (matrix - mu) / sd

    groups = list(sizes.keys())
    n_cells = matrix.shape[0]
    n_feats = matrix.shape[1]

    if figsize is None:
        figsize = (max(4, n_feats * 0.4 + 1.5),
                   max(3, n_cells * 0.012 + 1.5))

    _created = ax is None
    if show_group_colors and ax is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.03, 0.97],
                               wspace=0.01, figure=fig)
        ax_colors = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
    elif ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax_colors = None
    else:
        fig = ax.figure
        ax_colors = None

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(features, rotation=90, ha='center', fontsize=9)
    ax.set_yticks([])

    # Gene group separator lines
    if var_group_positions is not None:
        for i, (start, end) in enumerate(var_group_positions[:-1]):
            next_start = var_group_positions[i + 1][0]
            sep_x = (end + next_start) / 2
            ax.axvline(sep_x, color='white', lw=2, zorder=2)

    # Group color sidebar
    if ax_colors is not None:
        # Colour by name, over the whole column, as every other plot: the
        # sidebar indexed the palette by its own group list, so a group the
        # sample or categories_order left out moved every colour after it.
        style = resolve_category_style(data, groupby, palette=palette)
        color_arr = np.zeros((n_cells, 1, 3))
        row = 0
        for g in groups:
            c = plt.matplotlib.colors.to_rgb(style.color(g))
            n = sizes[g]
            color_arr[row:row + n, 0, :] = c
            row += n
        ax_colors.imshow(color_arr, aspect='auto', interpolation='nearest')
        ax_colors.set_xticks([])

        # Group tick labels on color sidebar
        cumulative = 0
        tick_positions = []
        tick_labels = []
        for g in groups:
            n = sizes[g]
            tick_positions.append(cumulative + n / 2)
            tick_labels.append(g)
            cumulative += n
        ax_colors.set_yticks(tick_positions)
        ax_colors.set_yticklabels(tick_labels, fontsize=8)
        for spine in ax_colors.spines.values():
            spine.set_visible(False)
        ax_colors.tick_params(left=False, bottom=False)

    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02,
                 label='z-score (per feature)' if standard_scale == 'var' else 'Expression')

    if title:
        ax.set_title(title)

    _figure.layout(fig)
    out = _figure.finish(fig, created=_created, show=show, save=save,
                         writekey='heatmap_cells')
    if return_fig:
        return fig, ax
    return out


@wraps(heatmap)
def plotHeatmap(*args, **kwargs):
    """Alias for :func:`heatmap`.
    """
    return heatmap(*args, **kwargs)
