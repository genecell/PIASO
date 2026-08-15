"""Dotplot for marker gene visualization.

Displays fraction of expressing cells (dot size) and mean expression (dot color)
for a set of features across cell groups. Supports AnnData and cytome Dataset.
"""

from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PathCollection
from functools import wraps

# Minimum visible scatter size for a POSITIVE fraction (fraction 0 -> size 0).
_MIN_DOT = 1.5


def _nice_legend_fracs(dot_max, n_target=5):
    """Even "nice-number" fraction series for the dot-size legend.

    Picks a step from ``{1, 2, 2.5, 5}×10ⁿ`` so there are ~``n_target`` ticks
    spanning ``(0, dot_max]`` (e.g. dot_max=0.25 → 5/10/15/20/25%; dot_max=0.6
    → 20/40/60%). Always returns ≥1 positive tick, and includes ``dot_max`` as
    the top tick when the series doesn't already land near it.
    """
    import math
    if not np.isfinite(dot_max) or dot_max <= 0:
        return [dot_max if dot_max and dot_max > 0 else 1.0]
    raw = dot_max / max(1, n_target)
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw - 1e-12),
                10 * mag)
    ticks = []
    v = step
    while v <= dot_max + 1e-9:
        ticks.append(round(v, 4))
        v += step
    if not ticks or abs(ticks[-1] - dot_max) > 0.25 * step:
        ticks.append(round(dot_max, 4))
    return sorted({t for t in ticks if t > 0})


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ..utils._cytome_compat import read_cells_columns as _read_cells_columns


def _resolve_features(features):
    """Resolve *features* into a flat list plus optional group annotations.

    Accepted inputs
    ---------------
    list : returned unchanged, no groups.
    dict : ``{'Group A': ['gene1', 'gene2'], 'Group B': ['gene3']}``
        Keys become ``var_group_labels``, values are concatenated in order.
    DataFrame : must contain a ``'group'`` column and either ``'gene'`` or
        ``'feature'`` column.  Each unique group becomes a label.

    Returns
    -------
    features_flat : list[str]
    var_group_labels : list[str] or None
    var_group_positions : list[tuple[int,int]] or None
    """
    if isinstance(features, str):          # a single feature name → one-element list
        return [features], None, None
    if isinstance(features, dict):
        flat = []
        labels = []
        positions = []
        for label, genes in features.items():
            if isinstance(genes, str):
                genes = [genes]
            start = len(flat)
            flat.extend(genes)
            end = len(flat) - 1
            labels.append(label)
            positions.append((start, end))
        return flat, labels, positions

    if isinstance(features, pd.DataFrame):
        # Expect columns: group + (gene | feature)
        gene_col = 'gene' if 'gene' in features.columns else 'feature'
        if gene_col not in features.columns or 'group' not in features.columns:
            raise ValueError(
                "DataFrame features must have 'group' and 'gene' (or 'feature') columns."
            )
        flat = []
        labels = []
        positions = []
        for grp, sub in features.groupby('group', sort=False):
            start = len(flat)
            flat.extend(sub[gene_col].tolist())
            end = len(flat) - 1
            labels.append(str(grp))
            positions.append((start, end))
        return flat, labels, positions

    # Plain list — no groups
    return list(features), None, None


def _get_expression_data(
    data, features, groupby, layer=None, use_raw=None,
    modality=None, cytome_layer="counts",
    compute_on_fly=True, use_cached_stats=True,
    mean_only_expressed=False,
):
    """Extract expression matrix for features x groups.

    Returns
    -------
    fraction_df : DataFrame (groups x features) — fraction of cells with expression > 0
    mean_df : DataFrame (groups x features) — mean expression. Over ALL cells in
        the group by default, or only the expressing (>0) cells when
        ``mean_only_expressed=True``.
    """
    if _is_cytome_input(data):
        return _get_expression_data_cytome(
            data, features, groupby,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly,
            use_cached_stats=use_cached_stats,
            mean_only_expressed=mean_only_expressed,
        )
    return _get_expression_data_anndata(data, features, groupby, layer, use_raw,
                                        mean_only_expressed=mean_only_expressed)


def _group_mean(gv, mean_only_expressed):
    """Mean over a group's values — all cells, or only expressing (>0) cells."""
    if gv.size == 0:
        return 0.0
    if mean_only_expressed:
        ev = gv[gv > 0]
        return float(ev.mean()) if ev.size else 0.0
    return float(gv.mean())


def _get_expression_data_anndata(adata, features, groupby, layer=None, use_raw=None,
                                 mean_only_expressed=False):
    import scipy.sparse as sp

    groups = sorted(adata.obs[groupby].dropna().unique(), key=str)
    group_labels = adata.obs[groupby].values

    fraction_data = {}
    mean_data = {}

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

        frac_row = {}
        mean_row = {}
        for g in groups:
            mask = np.array([str(x) == str(g) for x in group_labels])
            gv = vals[mask]
            frac_row[str(g)] = np.mean(gv > 0) if len(gv) > 0 else 0.0
            mean_row[str(g)] = _group_mean(np.asarray(gv, dtype=float), mean_only_expressed)
        fraction_data[feat] = frac_row
        mean_data[feat] = mean_row

    groups_str = [str(g) for g in groups]
    fraction_df = pd.DataFrame(fraction_data, index=groups_str)
    mean_df = pd.DataFrame(mean_data, index=groups_str)
    return fraction_df, mean_df


def _get_expression_data_cytome(
    source, features, groupby,
    modality=None, cytome_layer="counts",
    compute_on_fly=True, use_cached_stats=True,
    mean_only_expressed=False,
):
    """Read per-feature expression for a cytome dataset.

    For each feature: if it's a column of ``ds.cells`` (e.g. a
    pre-computed score), use it; otherwise resolve through the modality
    registry to read its column from ``{modality}_{cytome_layer}``.

    Returns ``(fraction_df, mean_df)`` indexed by group label, columns =
    features, mirroring the AnnData path.
    """
    from ..utils._cytome_compat import open_cytome as _open_cytome
    from ._plotEmbedding import _resolve_cytome_feature_values

    # Always read the groupby column from cells; features may or may not
    # be in cells depending on the modality.
    groupby_df = _read_cells_columns(source, [groupby])
    groups = sorted(groupby_df[groupby].dropna().unique(), key=str)
    groups_str = [str(g) for g in groups]
    group_labels = groupby_df[groupby].values

    # Discover which features are obs/cells columns (cheap path) and which
    # require matrix lookup.
    with _open_cytome(source) as ds:
        cells_cols = set(ds.cells.columns)
    cells_feats = [f for f in features if f in cells_cols]
    matrix_feats = [f for f in features if f not in cells_cols]

    fraction_data = {}
    mean_data = {}

    if cells_feats:
        cells_df = _read_cells_columns(source, list(cells_feats))
        for feat in cells_feats:
            vals = pd.to_numeric(cells_df[feat], errors='coerce').values
            fraction_data[feat] = {}
            mean_data[feat] = {}
            for g in groups:
                mask = group_labels == g
                gv = vals[mask]
                gv = gv[~np.isnan(gv)]
                fraction_data[feat][str(g)] = (
                    float((gv > 0).mean()) if gv.size else 0.0
                )
                mean_data[feat][str(g)] = _group_mean(np.asarray(gv, dtype=float), mean_only_expressed)

    if matrix_feats:
        from ._plotEmbedding import _resolve_cytome_feature_values_batch
        with _open_cytome(source) as ds:
            # ONE streaming pass for all genes (was one pass per gene — the slow
            # path for many markers). Numerically identical to per-feature reads.
            batched = _resolve_cytome_feature_values_batch(
                ds, list(matrix_feats),
                modality=modality, cytome_layer=cytome_layer,
                compute_on_fly=compute_on_fly,
                use_cached_stats=use_cached_stats,
            )
            for feat in matrix_feats:
                vals = np.asarray(batched[feat][0], dtype=float)
                fraction_data[feat] = {}
                mean_data[feat] = {}
                for g in groups:
                    mask = group_labels == g
                    gv = vals[mask]
                    fraction_data[feat][str(g)] = (
                        float((gv > 0).mean()) if gv.size else 0.0
                    )
                    mean_data[feat][str(g)] = _group_mean(np.asarray(gv, dtype=float), mean_only_expressed)

    fraction_df = pd.DataFrame(fraction_data, index=groups_str)
    mean_df = pd.DataFrame(mean_data, index=groups_str)
    # Preserve the user's feature order (cells_feats first then matrix_feats
    # in dict order would lose ordering). Reindex columns explicitly.
    fraction_df = fraction_df.reindex(columns=list(features))
    mean_df = mean_df.reindex(columns=list(features))
    return fraction_df, mean_df


def _compute_dendrogram(data, groupby, features=None, layer=None, use_raw=None,
                        use_rep=None, method='ward',
                        modality=None, cytome_layer="counts",
                        compute_on_fly=True, use_cached_stats=True):
    """Compute hierarchical clustering on group centroids.

    Parameters
    ----------
    data : AnnData or cytome
    groupby : str
    features : list, optional
        Gene features to compute mean expression for distance calculation.
        Ignored if use_rep is provided.
    use_rep : str, optional
        Key in adata.obsm to use for computing group centroids (e.g. 'X_gdr', 'X_svd').
        Takes priority over features if both are provided.
    method : str
        Linkage method (default: 'ward').

    Returns linkage matrix suitable for scipy.cluster.hierarchy.dendrogram.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    if use_rep is not None and not _is_cytome_input(data):
        # Use cell embeddings to compute group centroids
        embeddings = data.obsm[use_rep]
        group_labels = data.obs[groupby].values
        groups = sorted(set(str(g) for g in group_labels if pd.notna(g)))
        centroids = []
        for g in groups:
            mask = np.array([str(x) == g for x in group_labels])
            centroids.append(embeddings[mask].mean(axis=0))
        centroid_matrix = np.vstack(centroids)

        if centroid_matrix.shape[0] < 2:
            return None, groups

        dist = pdist(centroid_matrix, metric='euclidean')
        Z = linkage(dist, method=method)
    else:
        if features is None:
            return None, []
        _, mean_df = _get_expression_data(
            data, features, groupby, layer, use_raw,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
        )
        if mean_df.shape[0] < 2:
            return None, list(mean_df.index)
        groups = list(mean_df.index)
        dist = pdist(mean_df.values, metric='euclidean')
        Z = linkage(dist, method=method)

    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)
    ordered_groups = [groups[i] for i in order]
    return Z, ordered_groups


def dotplot(
    data,
    features: list,
    groupby: str = 'leiden',
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Optional[str] = None,
    log: bool = False,
    cmap: str = 'Reds',
    dot_max: Optional[float] = None,
    dot_min: float = 0,
    size_scale: float = 200,
    figsize: Optional[tuple] = None,
    square: bool = True,
    categories_order: Optional[list] = None,
    var_names_order: Optional[list] = None,
    var_group_labels: Optional[list] = None,
    var_group_positions: Optional[list] = None,
    dendrogram: bool = False,
    dendro_method: str = 'ward',
    use_rep: Optional[str] = None,
    swap_axes: bool = False,
    title: Optional[str] = None,
    fontsize: Optional[float] = None,
    grid: bool = False,
    show_border: bool = True,
    edgecolor: str = 'none',
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    palette=None,
    modality: Optional[str] = None,
    cytome_layer: str = "counts",
    compute_on_fly: bool = True,
    use_cached_stats: bool = True,
):
    """Plot a dotplot of feature expression across cell groups.

    Parameters
    ----------
    data : AnnData or cytome Dataset/path
        Input data.
    features : list, dict, or DataFrame
        Feature names.  A plain list of strings, a dict mapping group labels
        to gene lists (``{'Excitatory': ['Slc17a7', 'Satb2'], ...}``), or a
        DataFrame with ``'group'`` and ``'gene'``/``'feature'`` columns.
        When a dict or DataFrame is provided, ``var_group_labels`` and
        ``var_group_positions`` are inferred automatically.
    groupby : str
        Column in obs/cells for grouping.
    layer : str, optional
        AnnData layer. Ignored for cytome.
    use_raw : bool, optional
        Use raw attribute. Ignored for cytome.
    expression_cutoff : float
        Threshold for "expressing" (default 0).
    mean_only_expressed : bool
        Compute mean over expressing cells only.
    standard_scale : str, optional
        Min-max standardise the colour (mean expression) to ``[0, 1]``
        (scanpy semantics): ``'var'`` per gene (column), ``'group'`` per group
        (row), or ``None`` for raw means. (Previously z-scored, which produced
        negative values — now min-max, so the colour bar is always ``[0, 1]``.)
    log : bool
        Log1p transform expression values.
    cmap : str
        Colormap for mean expression.
    dot_max : float, optional
        Max dot size (fraction). Default: data max.
    dot_min : float
        Min dot size (fraction).
    size_scale : float
        Scaling factor for dot size.
    figsize : tuple, optional
        Auto-calculated if None.
    square : bool, default True
        Make every grid block a square (uniform per-cell sizing +
        ``ax.set_aspect('equal')``) so the dots sit centered in square cells.
        ``False`` restores the legacy width∝features / height∝groups sizing.
    categories_order : list, optional
        Custom group ordering on Y axis.
    var_names_order : list, optional
        Custom gene ordering on X axis.
    var_group_labels : list, optional
        Labels for gene groups (displayed as brackets).
    var_group_positions : list of tuple, optional
        Start/end positions for gene group brackets, e.g. ``[(0,3), (4,7)]``.
    dendrogram : bool
        Show dendrogram on group axis.
    dendro_method : str
        Linkage method for dendrogram (default: 'ward').
    use_rep : str, optional
        Key in adata.obsm for computing dendrogram from cell embeddings
        (e.g. 'X_gdr', 'X_svd'). If None, uses mean expression of features.
    swap_axes : bool
        Genes on Y, groups on X.
    title : str, optional
        Plot title.
    grid : bool
        Show light grid lines.
    show_border : bool
        Show outer border around the plot area (default: True).
    edgecolor : str
        Edge color for dots (default: 'none').
    show : bool
        Call plt.show().
    save : str, optional
        Save path.
    ax : Axes, optional
        Pre-existing axes.
    return_fig : bool
        Return (fig, ax) tuple.
    palette
        Not used for dotplot (color is continuous). Reserved for API consistency.
    """
    # --- Resolve dict / DataFrame features ---
    features, auto_labels, auto_positions = _resolve_features(features)
    if auto_labels is not None:
        if var_group_labels is None:
            var_group_labels = auto_labels
        if var_group_positions is None:
            var_group_positions = auto_positions

    # --- Get data ---
    fraction_df, mean_df = _get_expression_data(
        data, features, groupby, layer, use_raw,
        modality=modality, cytome_layer=cytome_layer,
        compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
        mean_only_expressed=mean_only_expressed,
    )

    # --- Reorder groups ---
    dendro_linkage = None
    if categories_order is not None:
        fraction_df = fraction_df.loc[[c for c in categories_order if c in fraction_df.index]]
        mean_df = mean_df.loc[[c for c in categories_order if c in mean_df.index]]
    elif dendrogram and fraction_df.shape[0] >= 2:
        dendro_linkage, ordered_groups = _compute_dendrogram(
            data, groupby, features, layer, use_raw,
            use_rep=use_rep, method=dendro_method,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats)
        fraction_df = fraction_df.loc[ordered_groups]
        mean_df = mean_df.loc[ordered_groups]
    elif auto_labels is not None:
        # Diagonal layout: when features come as a dict / DataFrame keyed by group
        # (e.g. {cell_type: markers} or a COSG result), default the ROW order to
        # the group-key order so each group's marker block lands on the diagonal.
        # Groups present in the data but absent from the keys are appended last.
        key_order = [str(g) for g in auto_labels if str(g) in fraction_df.index]
        rest = [g for g in fraction_df.index if g not in key_order]
        ordered = key_order + rest
        fraction_df = fraction_df.loc[ordered]
        mean_df = mean_df.loc[ordered]

    # --- Reorder features ---
    if var_names_order is not None:
        fraction_df = fraction_df[[f for f in var_names_order if f in fraction_df.columns]]
        mean_df = mean_df[[f for f in var_names_order if f in mean_df.columns]]

    # --- Transform ---
    if log:
        mean_df = np.log1p(mean_df)

    # Min-max standardisation to [0, 1], matching scanpy's dotplot semantics
    # ("subtract the minimum and divide by the maximum"). NOTE: this replaces
    # the previous z-score, which produced confusing negative colour values.
    if standard_scale == 'var':            # per-variable (per-gene = per-column)
        mean_df = mean_df - mean_df.min(axis=0)
        mean_df = (mean_df / mean_df.max(axis=0).replace(0, np.nan)).fillna(0.0)
    elif standard_scale == 'group':        # per-group (= per-row)
        mean_df = mean_df.sub(mean_df.min(axis=1), axis=0)
        mean_df = mean_df.div(mean_df.max(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    elif standard_scale not in (None, 'none'):
        raise ValueError(
            f"Invalid standard_scale={standard_scale!r}. Choose from "
            "{'var', 'group', None}."
        )

    groups = list(fraction_df.index)
    feat_names = list(fraction_df.columns)
    n_groups = len(groups)
    n_feats = len(feat_names)

    if swap_axes:
        fraction_df = fraction_df.T
        mean_df = mean_df.T
        groups, feat_names = feat_names, groups
        n_groups, n_feats = n_feats, n_groups

    # --- Figure layout ---
    has_dendro = (dendrogram and dendro_linkage is not None and not swap_axes)
    has_brackets = bool(var_group_labels and var_group_positions and not swap_axes)

    # Resolve the base font size: explicit fontsize= wins, else the global
    # rcParams['font.size'] (so piaso.settings.set_figure_params(fontsize=14)
    # actually enlarges the dotplot labels). Derived label sizes scale off it.
    _fs = float(fontsize) if fontsize is not None else float(
        plt.rcParams.get('font.size', 10) or 10)
    _fs_tick = _fs
    _fs_bracket = _fs * 0.8
    _fs_legend = _fs * 0.8
    _fs_cbar_title = _fs * 0.75
    _fs_cbar_tick = _fs * 0.65
    _fs_title = _fs * 1.1
    # Font-aware scaling factor (1.0 at the 10pt baseline) so bigger fonts get
    # bigger blocks/margins instead of overflowing fixed-size cells.
    _fscl = _fs / 10.0

    if figsize is None:
        if square:
            # Uniform per-cell size for both axes so every block is a square
            # (paired with ax.set_aspect('equal') below). Block size and margins
            # both scale with the font so large fonts don't overflow the cells.
            _cell = 0.3 * max(1.0, _fscl)
            w = n_feats * _cell + 4.0 * _fscl
            h = n_groups * _cell + 2.0 * _fscl
        else:
            w = max(6, n_feats * 0.55 * _fscl + 3.5 * _fscl)
            h = max(4.0, n_groups * 0.38 * _fscl + 2.0 * _fscl)
        if has_dendro:
            w += 1.5  # extra width for dendrogram panel
        if has_brackets:
            h += 0.6 * _fscl  # extra bottom margin for the bracket labels
        figsize = (w, h)

    # The dendrogram is appended to the LEFT via make_axes_locatable AFTER the
    # main axes geometry (incl. square set_aspect) is finalized, so it tracks the
    # main axes' real drawn box instead of a GridSpec column that ignores the
    # aspect-shrunk height — the old source of the misalignment. The shared
    # divider is created here and reused for the right colorbar/legend panel.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax_dendro = None
    _divider = None

    # --- Dot sizes and colors ---
    frac_vals = fraction_df.values  # (n_groups, n_feats)
    mean_vals = mean_df.values

    if dot_max is None:
        dot_max = frac_vals.max() if frac_vals.max() > 0 else 1.0
    frac_scaled = np.clip((frac_vals - dot_min) / (dot_max - dot_min + 1e-12), 0, 1)

    # --- Plot all dots in a single vectorized scatter call ---
    # Build coordinate arrays
    x_coords = np.tile(np.arange(n_feats), n_groups)
    y_coords = np.repeat(np.arange(n_groups), n_feats)
    # Dot area ∝ fraction. A fraction of 0 (gene undetected in the group) draws
    # NO dot (size 0, scanpy-like); positive fractions are floored to a small
    # visible size so very-lowly-expressing dots don't vanish.
    _raw_sizes = frac_scaled.flatten() * size_scale
    sizes = np.where(_raw_sizes > 0, np.maximum(_raw_sizes, _MIN_DOT), 0.0)
    colors = mean_vals.flatten()

    vmin = np.nanmin(mean_vals)
    vmax = np.nanmax(mean_vals)
    if vmin == vmax:
        vmax = vmin + 1

    sc = ax.scatter(x_coords, y_coords, s=sizes, c=colors,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors=edgecolor, linewidths=0.5, zorder=3)

    # --- Grid ---
    if grid:
        for gi in range(n_groups):
            ax.axhline(gi, color='#e8e8e8', lw=0.4, zorder=1)
        for fi in range(n_feats):
            ax.axvline(fi, color='#e8e8e8', lw=0.4, zorder=1)

    # --- Labels ---
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(feat_names, rotation=90, ha='center', fontsize=_fs_tick)
    # Move gene names to the top of the plot
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(groups, fontsize=_fs_tick)
    ax.set_xlim(-0.5, n_feats - 0.5)
    ax.set_ylim(-0.5, n_groups - 0.5)
    ax.invert_yaxis()
    if square:
        # Equal data aspect → each grid cell is a true square (the dots, already
        # circles, then sit centered in square blocks). The size-legend/colorbar
        # panel is appended afterwards and scales with the (now square) axes.
        ax.set_aspect('equal')

    # --- Draw dendrogram (appended-left, tracks the final/aspect-adjusted box) ---
    if has_dendro and dendro_linkage is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from scipy.cluster.hierarchy import dendrogram as scipy_dendro
        _divider = make_axes_locatable(ax)
        ax_dendro = _divider.append_axes("left", size="15%", pad=0.05)
        dendro_result = scipy_dendro(
            dendro_linkage, orientation='left', ax=ax_dendro,
            no_labels=True, color_threshold=0,
            above_threshold_color='#333333')
        # scipy places leaves at 5, 15, 25, … (step 10); the dot rows sit at
        # 0..n-1 inside ylim(-0.5, n-0.5). Both put their k-th leaf at the same
        # fractional height ((k+0.5)/n), so matching the inverted ylim aligns the
        # branches with the rows — now that the panel shares the main axes' box.
        leaves = dendro_result['leaves']
        dendro_ymin = 5
        dendro_ymax = 5 + 10 * (len(leaves) - 1)
        ax_dendro.set_ylim(dendro_ymax + 5, dendro_ymin - 5)  # inverted
        ax_dendro.set_axis_off()

    # --- Gene group brackets ---
    # Brackets sit just below the bottom frame. x = data coords (gene
    # positions); y is anchored at the axes bottom (axes-fraction 0) and then
    # offset DOWN by a fixed number of inches — so the gap is constant regardless
    # of plot height (axes-fraction offsets ballooned on tall square grids and
    # pushed the brackets far from the frame). Labels hug the bracket line.
    if has_brackets:
        from matplotlib.transforms import (blended_transform_factory,
                                           ScaledTranslation)
        _base = blended_transform_factory(ax.transData, ax.transAxes)
        _in = fig.dpi_scale_trans
        _line_in = 0.08                       # bracket line: 0.08" below frame
        _text_in = _line_in + (_fs_bracket + 3) / 72.0   # label below the line
        line_trans = _base + ScaledTranslation(0, -_line_in, _in)
        tick_trans = _base + ScaledTranslation(0, -_line_in, _in)
        text_trans = _base + ScaledTranslation(0, -_text_in, _in)
        for label, (start, end) in zip(var_group_labels, var_group_positions):
            mid = (start + end) / 2
            # Horizontal line (thin, understated)
            ax.plot([start - 0.3, end + 0.3], [0, 0],
                    color='#444444', lw=0.8, clip_on=False, transform=line_trans)
            # Tick marks at ends (small vertical ticks turning up toward frame)
            for xt in (start - 0.3, end + 0.3):
                ax.plot([xt, xt], [0, 0], marker='|', markersize=3,
                        color='#444444', lw=0.6, clip_on=False,
                        transform=tick_trans)
            # Label (scales with the global font; hugs the bracket line)
            ax.text(mid, 0, label, ha='center', va='top',
                    fontsize=_fs_bracket, fontweight='normal', clip_on=False,
                    transform=text_trans)
        # Separator lines at group boundaries (kept; lightened)
        for i, (start, end) in enumerate(var_group_positions[:-1]):
            next_start = var_group_positions[i + 1][0]
            sep_x = (end + next_start) / 2
            ax.axvline(sep_x, color='#bbbbbb', lw=0.6, ls='--', zorder=2)

    # --- Size legend fractions: an even "nice-number" series up to dot_max ---
    # (The old quartile logic collapsed to a single tick when dot_max was just
    # above 0.25 — e.g. a max of 25% showed only one dot.)
    legend_fracs = _nice_legend_fracs(dot_max, n_target=5)

    # Cap the legend handle dot size so the largest circle doesn't overrun its
    # own label (#9). The dots are still proportional to each other up to the cap.
    _LEG_MAX = 130.0
    legend_elements = []
    for f in legend_fracs:
        s = ((f - dot_min) / (dot_max - dot_min + 1e-12)) * size_scale
        legend_elements.append(
            plt.scatter([], [], s=min(max(s, _MIN_DOT), _LEG_MAX), c='grey',
                        edgecolors='black', linewidths=0.5, label=f'{f:.0%}')
        )

    # --- Colorbar + size legend in DEDICATED axes alongside the grid ---
    # Using fig.colorbar(..., ax=ax) and ax.legend(bbox_to_anchor=...) steals
    # axes width / drifts with figsize (overlapping the grid, esp. swap_axes +
    # small figsize). A divider gives each its own axes that scale WITH the main
    # axes, so they never overlap the dots and don't drift. Reuse the dendrogram
    # divider so left + right panels share one geometry.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = _divider if _divider is not None else make_axes_locatable(ax)
    # ONE dedicated panel holding a COMPACT colorbar (short + thin) on top and the
    # size legend below — both scale with the axes (no figsize drift / overlap).
    lax = divider.append_axes("right", size="24%", pad=0.35)
    lax.axis('off')
    # Colorbar pinned to the TOP of the panel; the size legend grows up from the
    # bottom. A wide vertical gap between them keeps the two titles from colliding.
    cax = lax.inset_axes([0.0, 0.78, 0.12, 0.20])   # x, y, w, h in panel fraction
    cbar = fig.colorbar(sc, cax=cax)
    cax.set_title('Mean\nexpression', fontsize=_fs_cbar_title, pad=3)
    cbar.ax.tick_params(labelsize=_fs_cbar_tick, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    # Bound the size legend to the BOTTOM ~60% of the panel (its own inset) so its
    # "Fraction expressing" title can never ride up into the colorbar above it.
    # labelspacing/handletextpad opened up so the (capped) dots clear their labels.
    leg_inset = lax.inset_axes([0.0, 0.0, 1.0, 0.60])
    leg_inset.axis('off')
    leg_inset.legend(handles=legend_elements, title='Fraction\nexpressing',
                     loc='upper left', bbox_to_anchor=(0.0, 1.0),
                     frameon=False, fontsize=_fs_legend, title_fontsize=_fs_legend,
                     labelspacing=1.4, handletextpad=1.2, borderpad=0.3)

    if title:
        ax.set_title(title, fontsize=_fs_title, pad=10)

    # --- Border / spines ---
    if show_border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor('#333333')
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, top=False)

    # The divider axes carry the colorbar/legend, so we only need to reserve
    # vertical room for the brackets (no right-margin reservation needed).
    if has_dendro:
        fig.subplots_adjust(left=0.01, bottom=0.15 if has_brackets else 0.08, top=0.78)
    else:
        try:
            fig.tight_layout(rect=[0, 0.12 if has_brackets else 0, 1, 1])
        except Exception:
            pass

    from ..settings import _savefig
    _savefig(fig, save, writekey='dotplot')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax


@wraps(dotplot)
def plotDotplot(*args, **kwargs):
    """Alias for :func:`dotplot`."""
    return dotplot(*args, **kwargs)
