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
from ..utils._cytome_compat import read_cells_columns as _read_cells_columns


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
    import scipy.sparse as sp
    adata = data
    groups = sorted(adata.obs[groupby].dropna().unique(), key=str)
    group_labels = adata.obs[groupby].values

    result = {}
    for feat in features:
        if feat in adata.obs.columns:
            vals = np.asarray(adata.obs[feat].values, dtype=float)
        elif use_raw and adata.raw is not None:
            v = adata.raw[:, feat].X
            if sp.issparse(v):
                v = v.toarray()
            vals = np.asarray(v).flatten()
        elif layer is not None:
            vals = np.asarray(adata.obs_vector(feat, layer=layer), dtype=float)
        else:
            vals = np.asarray(adata.obs_vector(feat), dtype=float)

        means = {}
        for g in groups:
            mask = np.array([str(x) == str(g) for x in group_labels])
            gv = vals[mask]
            means[str(g)] = np.nanmean(gv) if len(gv) > 0 else 0.0
        result[feat] = means

    mean_df = pd.DataFrame(result)
    mean_df.index.name = groupby
    return mean_df


def _get_cell_expression(data, features, groupby, layer=None, use_raw=None,
                         max_cells_per_group=100, random_state=42,
                         categories_order=None,
                         modality=None, cytome_layer="counts",
                         compute_on_fly=True, use_cached_stats=True):
    """Get per-cell expression matrix with group-stratified sampling.

    Returns (matrix, group_labels, group_boundaries) where:
    - matrix: ndarray (n_sampled_cells, n_features)
    - group_labels: list of group names per row
    - group_sizes: dict mapping group → n_cells in matrix
    """
    import scipy.sparse as sp

    if _is_cytome_input(data):
        from ..utils._cytome_compat import open_cytome as _open_cytome
        from ._plotEmbedding import _resolve_cytome_feature_values

        groupby_df = _read_cells_columns(data, [groupby])
        groups = categories_order or sorted(
            groupby_df[groupby].dropna().unique(), key=str,
        )
        group_labels_all = groupby_df[groupby].astype(str).values

        # Pre-fetch each feature as an n_cells vector. Cells columns first
        # (cheap), matrix-resolved features second (one streaming column
        # read per feature).
        with _open_cytome(data) as ds:
            cells_cols = set(ds.cells.columns)
        cells_feats = [f for f in features if f in cells_cols]
        matrix_feats = [f for f in features if f not in cells_cols]
        feature_vecs = {}
        if cells_feats:
            extra = _read_cells_columns(data, list(cells_feats))
            for f in cells_feats:
                feature_vecs[f] = pd.to_numeric(extra[f], errors='coerce').values.astype(float)
        if matrix_feats:
            with _open_cytome(data) as ds:
                for f in matrix_feats:
                    vals, _ = _resolve_cytome_feature_values(
                        ds, f,
                        modality=modality, cytome_layer=cytome_layer,
                        compute_on_fly=compute_on_fly,
                        use_cached_stats=use_cached_stats,
                    )
                    feature_vecs[f] = np.asarray(vals, dtype=float)

        rng = np.random.RandomState(random_state)
        sampled = []
        labels = []
        sizes = {}
        for g in groups:
            mask = group_labels_all == str(g)
            cell_idx = np.where(mask)[0]
            n = min(len(cell_idx), max_cells_per_group)
            if n < len(cell_idx):
                cell_idx = rng.choice(cell_idx, n, replace=False)
            row_vals = [feature_vecs[f][cell_idx] for f in features]
            cell_matrix = np.column_stack(row_vals) if row_vals else np.zeros((n, 0))
            sampled.append(cell_matrix)
            labels.extend([str(g)] * n)
            sizes[str(g)] = n
        if not sampled:
            return np.zeros((0, len(features))), [], {}
        return np.vstack(sampled), labels, sizes

    # AnnData path
    adata = data
    group_labels_all = adata.obs[groupby].values
    groups = categories_order or sorted(set(str(g) for g in group_labels_all if pd.notna(g)))
    rng = np.random.RandomState(random_state)

    sampled = []
    labels = []
    sizes = {}
    for g in groups:
        mask = np.array([str(x) == str(g) for x in group_labels_all])
        cell_idx = np.where(mask)[0]
        n = min(len(cell_idx), max_cells_per_group)
        if n < len(cell_idx):
            cell_idx = rng.choice(cell_idx, n, replace=False)

        row_vals = []
        for feat in features:
            if feat in adata.obs.columns:
                v = np.asarray(adata.obs[feat].values[cell_idx], dtype=float)
            elif use_raw and adata.raw is not None:
                v = adata.raw[cell_idx, feat].X
                if sp.issparse(v):
                    v = v.toarray()
                v = np.asarray(v).flatten()
            elif layer is not None:
                v = np.asarray(adata[cell_idx, feat].layers[layer], dtype=float)
                if sp.issparse(v):
                    v = v.toarray()
                v = v.flatten()
            else:
                v = adata[cell_idx, feat].X
                if sp.issparse(v):
                    v = v.toarray()
                v = np.asarray(v, dtype=float).flatten()
            row_vals.append(v)
        # shape: (n_cells, n_features)
        cell_matrix = np.column_stack(row_vals)
        sampled.append(cell_matrix)
        labels.extend([str(g)] * n)
        sizes[str(g)] = n

    return np.vstack(sampled), labels, sizes


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
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    modality: Optional[str] = None,
    cytome_layer: str = "counts",
    compute_on_fly: bool = True,
    use_cached_stats: bool = True,
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
        ``'var'`` to z-score per feature, ``'group'`` per group.
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
        Show colored sidebar for groups when ``cell_level=True``.
    show, save, ax, return_fig
        Output options.
    """
    # --- Resolve dict / DataFrame features ---
    features, _auto_labels, _auto_positions = _resolve_features(features)

    if cell_level:
        return _heatmap_cell_level(
            data, features, groupby, layer, use_raw, standard_scale, log,
            cmap, figsize, categories_order, var_names_order, title,
            vmin, vmax, max_cells_per_group, show_group_colors,
            show, save, ax, return_fig,
            var_group_labels=_auto_labels, var_group_positions=_auto_positions)

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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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

    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label='Mean expression')

    if title:
        ax.set_title(title)

    plt.tight_layout()

    from ..settings import _savefig
    _savefig(fig, save, writekey='heatmap')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax


def _heatmap_cell_level(data, features, groupby, layer, use_raw,
                        standard_scale, log, cmap, figsize,
                        categories_order, var_names_order,
                        title, vmin, vmax, max_cells_per_group,
                        show_group_colors, show, save, ax, return_fig,
                        var_group_labels=None, var_group_positions=None):
    """Cell-level heatmap with group-stratified sampling."""
    from . import color as _color_mod

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
        # Build color column
        palette = _color_mod.d_color4
        if not _is_cytome_input(data) and hasattr(data, 'uns'):
            key = f'{groupby}_colors'
            if key in data.uns and len(data.uns[key]) >= len(groups):
                palette = list(data.uns[key])

        color_arr = np.zeros((n_cells, 1, 3))
        row = 0
        for i, g in enumerate(groups):
            c = plt.matplotlib.colors.to_rgb(palette[i % len(palette)])
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

    plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, label='Expression')

    if title:
        ax.set_title(title)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    from ..settings import _savefig
    _savefig(fig, save, writekey='heatmap_cells')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax


@wraps(heatmap)
def plotHeatmap(*args, **kwargs):
    """Alias for :func:`heatmap`."""
    return heatmap(*args, **kwargs)
