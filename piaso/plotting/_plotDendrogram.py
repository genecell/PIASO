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
from ._store_io import read_cells_columns as _read_cells_columns  # read-only
from ._group_order import resolve_group_order


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
        # No features given: the store's highly variable genes, as on AnnData.
        # The old fallback took the first numeric columns of the cells table --
        # cell_idx, n_fragments, QC metrics -- and clustered the groups on
        # those, silently (audit of 2026-10-27, part 3, critical 3).
        if features is None:
            features = _store_variable_genes(data, modality, n_top_genes)

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

    groups = _group_list(adata, groupby, adata.obs[groupby])
    from ._plotHeatmap import _anndata_group_means
    return _anndata_group_means(adata, list(features), groupby, groups,
                                layer=layer, use_raw=use_raw, skip_missing=True)


def _store_variable_genes(data, modality, n_top_genes):
    """The first ``n_top_genes`` highly variable genes a store records.

    A store converted from AnnData keeps ``var['highly_variable']`` in its
    feature table. Without that column there is nothing principled to cluster
    on without a pass over the matrix, so this raises and says what to pass.
    """
    with _open_cytome(data, mode="r") as ds:
        mods = [m for m in (modality, "RNA", "GA") if m]
        for mod in mods:
            try:
                table = ds.features(mod)
                cols = list(table.columns)
            except Exception:
                continue
            if "highly_variable" not in cols:
                continue
            try:
                import cytome
                _, _, name_col = cytome.modality_feature_table_info(ds, mod)
            except Exception:
                name_col = "gene_id"
            hv = np.asarray(table["highly_variable"]).astype(bool)
            names = np.asarray(table[name_col])[hv]
            if names.size:
                return [str(g) for g in names[:n_top_genes]]
        embeddings = list(ds.list_embeddings())
    raise ValueError(
        "plot_dendrogram: this store has no cell embedding the dendrogram "
        f"recognises and no highly variable genes to fall back on (embeddings: "
        f"{embeddings}). Pass use_rep='<embedding name>' to build the tree "
        "from an embedding, or features=[...] for marker-gene mode.")


def _group_list(data, groupby, labels):
    """The groups present, in the column's stored or categorical order."""
    from ._categories import resolve_category_style
    style = resolve_category_style(data, groupby, labels)
    return style.present(labels)


_EMBEDDING_PRIORITY = ('X_gdr', 'X_pca', 'X_svd', 'X_diffmap', 'X_umap')


def _detect_embedding(data, modality=None):
    """First available cell embedding from the standard priority list, or None,
    as the name it has in ``data``.

    PCA/SVD-like reps come first (better-behaved distances); ``X_umap`` is the
    last resort. Used when ``use_rep='auto'``.

    A store names its embeddings ``<modality>_<key>`` (``RNA_pca``,
    ``ATAC_svd``; older files ``RNA_obsm_X_pca``), so the priority keys are
    resolved through the same lookup the embedding plot uses. Only a name that
    *is* the key behind a prefix counts, not one that merely contains it --
    ``ATAC_svd_TMP_GA``, gene activity's temporary embedding, is not an SVD of
    the cells to cluster on.
    """
    try:
        if _is_cytome_input(data):
            from ._plotEmbedding import _resolve_cytome_basis
            with _open_cytome(data, mode="r") as ds:
                for key in _EMBEDDING_PRIORITY:
                    try:
                        return _resolve_cytome_basis(ds, key, modality=modality,
                                                     max_rank=1)
                    except KeyError:
                        continue
            return None
        present = set(getattr(data, 'obsm', {}) or {})
    except Exception:
        return None
    for key in _EMBEDDING_PRIORITY:
        if key in present:
            return key
    return None


def _group_centroids_from_rep(data, groupby, use_rep, modality=None):
    """Per-group centroids of a cell embedding → DataFrame (groups × dims).

    Returns None if the embedding (or the groupby column) is unavailable, so the
    caller can fall back to the marker-gene mode. On a store ``use_rep`` is
    resolved like ``basis`` (``'X_pca'`` finds ``RNA_pca``).
    """
    from ._categories import label_strings
    if _is_cytome_input(data):
        from ._plotEmbedding import _resolve_cytome_basis
        with _open_cytome(data, mode="r") as ds:
            if groupby not in ds.cells.columns:
                return None
            try:
                name = _resolve_cytome_basis(ds, use_rep, modality=modality)
            except KeyError:
                return None
            emb = np.asarray(ds.embeddings[name])
            labels = np.asarray(ds.cells[groupby])
    else:
        obsm = getattr(data, 'obsm', None)
        if obsm is None or use_rep not in obsm or groupby not in data.obs:
            return None
        emb = np.asarray(data.obsm[use_rep])
        labels = data.obs[groupby]
    if emb.ndim != 2 or emb.shape[0] != len(labels):
        return None
    groups = _group_list(data, groupby, labels)
    lab = label_strings(labels)
    # one pass over the cells (the per-group string comparison was a Python
    # loop over every cell for every group)
    frame = pd.DataFrame(emb[pd.notna(lab)])
    frame["_g"] = lab[pd.notna(lab)]
    means = frame.groupby("_g", sort=False).mean()
    return means.reindex(groups)


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
    show: Optional[bool] = None,
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
          exist, fall back to marker-gene expression similarity. On a cytome
          the keys resolve to the store's names (``RNA_pca``, ``ATAC_svd``);
          ``modality`` chooses between modalities.
        - an explicit key (e.g. ``'X_gdr'``, ``'X_svd'``, ``'RNA_pca'``): use
          that embedding.
        - ``None``: **marker-gene mode** — cluster groups by similarity of their
          mean expression over ``features`` (or the top-``n_top_genes`` highly
          variable genes; on AnnData without them, the top-variance genes).
          Use this when you specifically want the tree to reflect marker-gene
          programs rather than the global embedding geometry (e.g. comparing
          COSG top-``n_top_genes`` markers).

        *Changed*: on a cytome, ``'auto'`` used to find no embedding (it looked
        for AnnData's ``X_`` names only) and clustered the groups on the cells
        table's numeric columns -- cell index, fragment counts -- without a
        warning. A store without an embedding or highly variable genes now
        raises, naming what to pass.
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
        Colors for leaf labels. Without it, the column's stored colours (the
        same the embedding uses, by name); leaves are then coloured only when
        such colours exist.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Title.
    show : bool, optional
        Call ``plt.show()``; ``None`` (default) shows unless ``ax`` is given.
    save, ax, return_fig
        Output options. Without ``return_fig`` or ``return_linkage`` the figure
        is returned when it is not shown.
    return_linkage : bool
        Also return the linkage matrix ``Z``.
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
    Optionally (fig, ax) and/or linkage matrix.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendro
    from scipy.spatial.distance import pdist
    from . import _figure

    # Resolve use_rep: 'auto' picks a standard embedding if one exists; an
    # explicit key is used directly; None forces marker-gene mode. If the
    # embedding can't be read we fall back to marker-gene expression similarity.
    mean_df = None
    resolved_rep = None
    if use_rep == 'auto':
        resolved_rep = _detect_embedding(data, modality=modality)
    elif use_rep is not None:
        resolved_rep = use_rep
    if resolved_rep is not None:
        mean_df = _group_centroids_from_rep(data, groupby, resolved_rep,
                                            modality=modality)
        if mean_df is None and use_rep not in ('auto', None):
            import warnings as _w
            _w.warn(f"plot_dendrogram: use_rep={use_rep!r} not found; "
                    "falling back to marker-gene expression similarity.",
                    stacklevel=2)
    elif use_rep == 'auto':
        import warnings as _w
        _w.warn("plot_dendrogram: no embedding found (looked for "
                f"{list(_EMBEDDING_PRIORITY)}); building the tree from marker-gene "
                "expression similarity instead.", stacklevel=2)

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

    groups = [str(g) for g in mean_df.index]

    # Leaf colours by name, from the resolver every plot uses. They were
    # positional over the tree's group list -- built from a plain array, so a
    # categorical's order was gone and uns colours landed on other groups.
    leaf_colors = None
    from ._categories import resolve_category_style, stored_category_style
    if palette is not None or stored_category_style(data, groupby)[1] is not None:
        style = resolve_category_style(data, groupby, palette=palette)
        leaf_colors = {g: style.color(g) for g in groups}

    if figsize is None:
        if orientation in ('top', 'bottom'):
            figsize = (max(6, len(groups) * 0.6 + 1), 4)
        else:
            figsize = (5, max(4, len(groups) * 0.4 + 1))

    show = _figure.resolve_show(show, ax)
    fig, ax, _created = _figure.begin(ax, figsize=figsize)

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

    _figure.layout(fig)
    out = _figure.finish(fig, created=_created, show=show, save=save,
                         writekey='plot_dendrogram')

    result = []
    if return_fig:
        result.append((fig, ax))
    if return_linkage:
        result.append(Z)

    if len(result) == 1:
        return result[0]
    elif len(result) > 1:
        return tuple(result)
    return out


@wraps(plot_dendrogram)
def plotDendrogram(*args, **kwargs):
    """Alias for :func:`plot_dendrogram`.
    """
    return plot_dendrogram(*args, **kwargs)
