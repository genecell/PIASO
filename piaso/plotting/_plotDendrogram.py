"""Dendrogram plot for cell type hierarchy.

Visualizes hierarchical relationships between cell groups based on
mean expression profiles. Supports AnnData and cytome Dataset.
"""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps


from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ..utils._cytome_compat import open_cytome as _open_cytome
from ..utils._cytome_compat import read_cells_columns as _read_cells_columns


def _get_group_means(data, groupby, features=None, layer=None, use_raw=None, n_top_genes=50,
                     modality=None, cytome_layer="counts",
                     compute_on_fly=True, use_cached_stats=True):
    """Compute mean expression per group for clustering.

    If features is None, uses highly variable genes or top-variance genes.
    Returns DataFrame: groups x features.

    For cytome inputs, features that are columns of ``ds.cells`` are
    read from the cells table; features that are gene/peak names are
    resolved through the modality registry (RNA → genes, GA → GA_genes,
    etc.) and read from ``{modality}_{cytome_layer}``.
    """
    if _is_cytome_input(data):
        # If user provided no features, fall back to numeric ds.cells columns
        # (the previous behaviour) — auto-discovering matrix-side HVGs would
        # need a streaming pass we don't want by default.
        if features is None:
            with _open_cytome(data) as ds:
                col_names = [c for c in ds.cells.columns if c != groupby]
                numeric_cols = []
                for c in col_names:
                    arr = np.asarray(ds.cells[c])
                    if np.issubdtype(arr.dtype, np.number):
                        numeric_cols.append(c)
                        if len(numeric_cols) >= n_top_genes:
                            break
            features = numeric_cols

        # Reuse the dotplot helper which now handles BOTH cells-column and
        # matrix-resolved features via _resolve_cytome_feature_values.
        from ._plotDotplot import _get_expression_data_cytome
        _, mean_df = _get_expression_data_cytome(
            data, list(features), groupby,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly,
            use_cached_stats=use_cached_stats,
        )
        return mean_df

    # AnnData path
    import scipy.sparse as sp
    adata = data

    if features is None:
        if 'highly_variable' in adata.var.columns:
            hv_mask = adata.var['highly_variable'].values
            features = list(adata.var_names[hv_mask][:n_top_genes])
        else:
            # Top variance genes
            if sp.issparse(adata.X):
                var = np.asarray(adata.X.power(2).mean(axis=0)).flatten() - \
                      np.asarray(adata.X.mean(axis=0)).flatten() ** 2
            else:
                var = np.var(adata.X, axis=0)
            top_idx = np.argsort(var)[-n_top_genes:]
            features = list(adata.var_names[top_idx])

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
            try:
                vals = np.asarray(adata.obs_vector(feat), dtype=float)
            except KeyError:
                continue

        means = {}
        for g in groups:
            mask = np.array([str(x) == str(g) for x in group_labels])
            gv = vals[mask]
            means[str(g)] = np.nanmean(gv) if len(gv) > 0 else 0.0
        result[feat] = means

    return pd.DataFrame(result)


_EMBEDDING_PRIORITY = ('X_gdr', 'X_pca', 'X_svd', 'X_diffmap', 'X_umap')


def _detect_embedding(data):
    """First available cell embedding from the standard priority list, or None.

    PCA/SVD-like reps come first (better-behaved distances); ``X_umap`` is the
    last resort. Used when ``use_rep='auto'``.
    """
    try:
        if _is_cytome_input(data):
            with _open_cytome(data) as ds:
                present = set(ds.list_embeddings())
        else:
            present = set(getattr(data, 'obsm', {}) or {})
    except Exception:
        return None
    for key in _EMBEDDING_PRIORITY:
        if key in present:
            return key
    return None


def _group_centroids_from_rep(data, groupby, use_rep):
    """Per-group centroids of a cell embedding → DataFrame (groups × dims).

    Returns None if the embedding (or the groupby column) is unavailable, so the
    caller can fall back to the marker-gene mode.
    """
    if _is_cytome_input(data):
        with _open_cytome(data) as ds:
            if use_rep not in ds.list_embeddings() or groupby not in ds.cells.columns:
                return None
            emb = np.asarray(ds.embeddings[use_rep])
            labels = np.asarray(ds.cells[groupby])
    else:
        obsm = getattr(data, 'obsm', None)
        if obsm is None or use_rep not in obsm or groupby not in data.obs:
            return None
        emb = np.asarray(data.obsm[use_rep])
        labels = np.asarray(data.obs[groupby].values)
    if emb.ndim != 2 or emb.shape[0] != labels.shape[0]:
        return None
    groups = sorted(set(str(g) for g in labels if pd.notna(g)), key=str)
    rows = []
    for g in groups:
        mask = np.array([str(x) == g for x in labels])
        rows.append(emb[mask].mean(axis=0))
    if len(rows) < 2:
        return pd.DataFrame(np.vstack(rows) if rows else np.empty((0, emb.shape[1])),
                            index=groups)
    return pd.DataFrame(np.vstack(rows), index=groups)


def plot_dendrogram(
    data,
    groupby: str = 'leiden',
    features: Optional[list] = None,
    use_rep: Optional[str] = 'auto',
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    n_top_genes: int = 50,
    method: str = 'average',
    metric: str = 'euclidean',
    orientation: str = 'top',
    palette=None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    return_linkage: bool = False,
    modality: Optional[str] = None,
    cytome_layer: str = "counts",
    compute_on_fly: bool = True,
    use_cached_stats: bool = True,
):
    """Plot a dendrogram of cell groups based on expression similarity.

    Parameters
    ----------
    data : AnnData or cytome Dataset/path
        Input data.
    groupby : str
        Grouping column.
    features : list, optional
        Features to use for clustering (marker-gene mode). If None, uses highly
        variable genes or top-variance genes. Only consulted when ``use_rep`` is
        ``None`` or resolves to no embedding.
    use_rep : str or None, default ``'auto'``
        Build the group tree from per-group **centroids of a cell embedding**
        (the standard scanpy-style dendrogram), which is usually what you want.

        - ``'auto'`` (default): use the first available embedding from
          ``X_gdr`` → ``X_pca`` → ``X_svd`` → ``X_diffmap`` → ``X_umap``; if none
          exist, fall back to marker-gene expression similarity.
        - an explicit key (e.g. ``'X_gdr'``, ``'X_svd'``): use that embedding.
        - ``None``: **marker-gene mode** — cluster groups by similarity of their
          mean expression over ``features`` (or the top-``n_top_genes`` variable
          genes / numeric ``cells`` columns). Use this when you specifically want
          the tree to reflect marker-gene programs rather than the global
          embedding geometry (e.g. comparing COSG top-``n_top_genes`` markers).
    layer, use_raw
        AnnData layer / raw. Ignored for cytome.
    n_top_genes : int
        Number of top genes to use in marker-gene mode when ``features`` is None.
    method : str
        Linkage method (``'average'``, ``'ward'``, ``'complete'``, ``'single'``).
    metric : str
        Distance metric.
    orientation : str
        Dendrogram orientation: ``'top'``, ``'bottom'``, ``'left'``, ``'right'``.
    palette : list or dict, optional
        Colors for leaf labels.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Title.
    show, save, ax, return_fig
        Output options.
    return_linkage : bool
        Also return the linkage matrix ``Z``.

    Returns
    -------
    Optionally (fig, ax) and/or linkage matrix.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendro
    from scipy.spatial.distance import pdist
    from . import color as _color_mod

    # Resolve use_rep: 'auto' picks a standard embedding if one exists; an
    # explicit key is used directly; None forces marker-gene mode. If the
    # embedding can't be read we fall back to marker-gene expression similarity.
    mean_df = None
    resolved_rep = None
    if use_rep == 'auto':
        resolved_rep = _detect_embedding(data)
    elif use_rep is not None:
        resolved_rep = use_rep
    if resolved_rep is not None:
        mean_df = _group_centroids_from_rep(data, groupby, resolved_rep)
        if mean_df is None and use_rep not in ('auto', None):
            import warnings as _w
            _w.warn(f"plot_dendrogram: use_rep={use_rep!r} not found; "
                    "falling back to marker-gene expression similarity.",
                    stacklevel=2)

    if mean_df is None:
        mean_df = _get_group_means(
            data, groupby, features, layer, use_raw, n_top_genes,
            modality=modality, cytome_layer=cytome_layer,
            compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
        )

    if mean_df.shape[0] < 2:
        print(f"Only {mean_df.shape[0]} group(s) — need ≥2 for dendrogram.")
        return

    dist = pdist(mean_df.values, metric=metric)
    Z = linkage(dist, method=method)

    groups = list(mean_df.index)

    # Resolve leaf colors
    leaf_colors = None
    if palette is not None:
        if isinstance(palette, dict):
            leaf_colors = {g: palette.get(g, 'black') for g in groups}
        else:
            leaf_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    elif not _is_cytome_input(data) and hasattr(data, 'uns'):
        key = f'{groupby}_colors'
        if key in data.uns and len(data.uns[key]) >= len(groups):
            leaf_colors = {g: data.uns[key][i] for i, g in enumerate(groups)}

    if figsize is None:
        if orientation in ('top', 'bottom'):
            figsize = (max(6, len(groups) * 0.6 + 1), 4)
        else:
            figsize = (5, max(4, len(groups) * 0.4 + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    dendro_result = scipy_dendro(
        Z, labels=groups, ax=ax, orientation=orientation,
        color_threshold=0, above_threshold_color='#333333',
        leaf_rotation=90 if orientation in ('top', 'bottom') else 0,
        leaf_font_size=9,
    )

    # Color leaf labels
    if leaf_colors is not None:
        if orientation in ('top', 'bottom'):
            labels = ax.get_xticklabels()
        else:
            labels = ax.get_yticklabels()
        for lbl in labels:
            txt = lbl.get_text()
            if txt in leaf_colors:
                lbl.set_color(leaf_colors[txt])
                lbl.set_fontweight('bold')

    if title:
        ax.set_title(title)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()

    from ..settings import _savefig
    _savefig(fig, save, writekey='plot_dendrogram')
    if show:
        plt.show()
    else:
        plt.close(fig)

    result = []
    if return_fig:
        result.append((fig, ax))
    if return_linkage:
        result.append(Z)

    if len(result) == 1:
        return result[0]
    elif len(result) > 1:
        return tuple(result)


@wraps(plot_dendrogram)
def plotDendrogram(*args, **kwargs):
    """Alias for :func:`plot_dendrogram`."""
    return plot_dendrogram(*args, **kwargs)
