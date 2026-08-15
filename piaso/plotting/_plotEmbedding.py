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



from ..utils._cytome_compat import is_cytome_input as _is_cytome_input  # backward-compat re-export
from ..utils._cytome_compat import open_cytome as _open_cytome


# Canonical legend_loc values shared by plotEmbedding and plot_embeddings_split.
# 'right'   -> legend box / shared legend in the right margin
# 'on_data' -> category labels drawn at cluster centroids (note the underscore)
# 'none'    -> no legend
_VALID_LEGEND_LOC = ("right", "on_data", "none")


def _render_embedding(ax, coords, values, is_cat, *, color="value", palette=None,
                      point_size=1.0, alpha=1.0, rasterized=True, groups=None,
                      na_color="lightgray", legend_loc="right", legend_ncol=None,
                      legend_marker_size=None, legend_fontsize=10,
                      legend_fontoutline=None, cmap=None, vmin=None, vmax=None,
                      add_colorbar=True, category_order=None, strict_groups=True):
    """Render one embedding scatter (categorical or continuous) into ``ax`` from
    plain arrays — no AnnData / cytome dependency. Shared by ``plotEmbedding`` and
    ``plot_embeddings_split`` so the split needs no proxy-AnnData round-trip.

    ``palette`` may be a **list** (indexed by the resolved category order) or a
    **dict** mapping ``category -> hex`` (categories absent from the dict fall back
    to the default palette). ``category_order`` overrides the default alphabetical
    ordering — categories listed there come first (in that order), any remaining
    present categories follow alphabetically. Both are populated from a cytome's
    ``ds.set_categories`` store / AnnData categorical dtype + ``uns`` by the callers.

    Returns the continuous mappable (for a shared colorbar) or None.
    """
    from . import color as _color_mod
    if is_cat:
        present = set(str(v) for v in values)
        if category_order:
            ordered = [str(c) for c in category_order if str(c) in present]
            categories = ordered + sorted(present - set(ordered))
        else:
            categories = sorted(present)
        str_values = np.array([str(v) for v in values])
        if palette is None:
            palette = _color_mod.d_color4
        _pal_is_dict = isinstance(palette, dict)

        def _color_for(cat, i):
            if _pal_is_dict:
                return palette.get(
                    str(cat), _color_mod.d_color4[i % len(_color_mod.d_color4)])
            return palette[i % len(palette)]

        plot_categories = categories
        if groups is not None:
            groups_list = [str(g) for g in (
                groups if isinstance(groups, (list, tuple, set)) else [groups])]
            unknown = [g for g in groups_list if g not in categories]
            # In a split panel a requested group may be absent from THIS panel
            # (present in another) — that's not an error, just nothing to
            # highlight here. Only the single-plot path validates strictly.
            if unknown and strict_groups:
                raise ValueError(
                    f"plotEmbedding(): groups={unknown} not found among the "
                    f"categories of color={color!r}. Available: {categories}."
                )
            bg = ~np.isin(str_values, groups_list)
            if bg.any():
                ax.scatter(coords[bg, 0], coords[bg, 1], c=na_color,
                           s=point_size, alpha=alpha, rasterized=rasterized,
                           label='_nolegend_')
            plot_categories = [c for c in categories if c in groups_list]

        for cat in plot_categories:
            i = categories.index(cat)               # stable palette index
            mask = str_values == cat
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[_color_for(cat, i)],
                       s=point_size, alpha=alpha, label=cat, rasterized=rasterized)

        if legend_loc == "right":
            from matplotlib.lines import Line2D
            # Fixed, readable legend-dot size INDEPENDENT of point_size (Round 26):
            # proxy handles drawn at a constant markersize so the legend marker no
            # longer shrinks/grows when the data point_size changes.
            _ms = legend_marker_size if legend_marker_size is not None else 6.0
            _handles = [
                Line2D([0], [0], marker='o', linestyle='None', markersize=_ms,
                       markerfacecolor=_color_for(cat, categories.index(cat)),
                       markeredgecolor='none', label=cat)
                for cat in categories
            ]
            _ncol = legend_ncol if legend_ncol is not None else max(1, -(-len(categories) // 12))
            ax.legend(
                handles=_handles, bbox_to_anchor=(1.05, 1), loc="upper left",
                fontsize=legend_fontsize, frameon=False, ncol=_ncol,
                handletextpad=0.5, columnspacing=1.2, labelspacing=0.4,
            )
        elif legend_loc == "on_data":
            for cat in plot_categories:
                mask = str_values == cat
                cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
                text_kwargs = dict(
                    fontsize=legend_fontsize - 1, ha="center", va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
                )
                if legend_fontoutline is not None and legend_fontoutline > 0:
                    import matplotlib.patheffects as patheffects
                    text_kwargs['path_effects'] = [
                        patheffects.withStroke(linewidth=legend_fontoutline, foreground='white')
                    ]
                ax.text(cx, cy, cat, **text_kwargs)
        return None
    else:
        import matplotlib.pyplot as plt
        if cmap is None:
            cmap = _color_mod.c_color1
        order = np.argsort(values)
        sc = ax.scatter(coords[order, 0], coords[order, 1],
                        c=values[order], cmap=cmap, s=point_size, alpha=alpha,
                        rasterized=rasterized, vmin=vmin, vmax=vmax)
        if add_colorbar:
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        return sc


def _resolve_cytome_basis(ds, basis):
    """Resolve a basis name against a cytome dataset's embeddings.

    Tries exact match first, then a substring lookup so callers can write
    ``basis='X_umap'`` even when the embedding is registered as
    ``'RNA_obsm_X_umap'``.
    """
    emb_names = ds.list_embeddings()
    if basis in emb_names:
        return basis
    keyword = basis.lower().replace("x_", "")
    matches = [n for n in emb_names if keyword in n.lower()]
    if matches:
        return matches[-1]
    raise KeyError(f"Embedding '{basis}' not found. Available: {emb_names}")


def _cytome_extract(source, basis, obs_columns=None):
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
    with _open_cytome(source) as ds:
        emb_name = _resolve_cytome_basis(ds, basis)
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

    Honors, in priority order:

    1. an explicit ``user_palette`` (list or dict) — returned as-is for colour,
    2. a cytome ``ds.set_categories(color, order=, colors=)`` store
       (``ds.metadata['categories'][color]`` → ``{order, colors}``),
       or AnnData's categorical dtype order + ``uns[f'{color}_colors']``,
    3. the legacy ``ds.metadata[f'{color}_colors']`` / ``uns[f'{color}_colors']``
       palette list (no explicit order).

    Returns ``(palette, category_order)`` where ``palette`` is a dict
    ``{cat: hex}``, a list, or None, and ``category_order`` is a list or None.
    The shared renderer applies both.
    """
    palette = user_palette
    category_order = None
    try:
        if _is_cytome_input(data):
            with _open_cytome(data) as ds:
                entry = None
                getter = getattr(ds, "get_categories", None)
                if getter is not None:
                    entry = getter(color)
                if entry:
                    category_order = entry.get("order")
                    if palette is None and entry.get("colors"):
                        palette = dict(entry["colors"])
                if palette is None:
                    stored = ds.metadata.get(f"{color}_colors")
                    if stored is not None:
                        palette = list(stored)
        elif hasattr(data, "obs") and color in getattr(data, "obs", {}):
            col = data.obs[color]
            if isinstance(col.dtype, pd.CategoricalDtype):
                category_order = list(col.cat.categories)
            if palette is None:
                key = f"{color}_colors"
                if getattr(data, "uns", None) is not None and key in data.uns:
                    palette = list(data.uns[key])
    except Exception:
        # Styling is best-effort: never let a metadata read break plotting.
        return user_palette, None
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
        plt.tight_layout()
        return

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

    plt.tight_layout()

    
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
                          show_figure:bool=True,
                          save:bool=None,
                          layer:str=None,
                          basis:str='X_umap',
                          fix_coordinate_ratio:bool=True,
                          show_axis_ticks:bool=False,
                          margin_ratio:float=0.05,
                          legend_fontsize:int=10,
                          legend_fontoutline:int=2,
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
        Whether to display the figure after plotting.
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
    legend_fontsize: int, optional (default=9)
        Font size in pt.
    legend_fontoutline: int, optional (default=2)
        Line width of the legend font outline in pt. 
    legend_loc: str, optional (default='right margin')
        Location of legend, defaults to 'right margin'.
    legend_marker_size: float, optional (default=4.0)
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
        If None, auto-scaled (``max(0.1, min(4, 30000 / n_cells))``,
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
    **kwargs : dict
        Forwarded verbatim to :func:`piaso.pl.plotEmbedding` for each
        panel. Accepts the scanpy-style aliases ``ncols`` (→ ``ncol``)
        and ``size`` (→ ``point_size``) for compatibility with existing
        call sites.

    Returns
    -------
    None.

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

    from . import color as _color_mod

    # The modality (if any) carrying the user's `color` feature, used to
    # optionally enrich panel titles. Set by the cytome path; stays None
    # for AnnData and obs-column colours.
    resolved_modality_for_title = None

    # --- 1. Unified data extraction (cytome / AnnData) ---
    if _is_cytome_input(data):
        coords, obs_df, palette_map = _cytome_extract(
            data, basis=basis, obs_columns=[splitby, color]
        )
        if splitby not in obs_df.columns:
            with _open_cytome(data) as _ds:
                avail = sorted(_ds.cells.columns)
            raise ValueError(
                f"The splitby key '{splitby}' was not found in cytome cells "
                f"table. Available: {avail}"
            )
        splitby_series = obs_df[splitby]
        if color in obs_df.columns:
            full_color_data = obs_df[color].values
            is_numeric = pd.api.types.is_numeric_dtype(obs_df[color])
        else:
            with _open_cytome(data) as _ds:
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
        if color in adata_input.obs.columns:
            full_color_data = adata_input.obs[color].values
            is_numeric = pd.api.types.is_numeric_dtype(adata_input.obs[color])
        else:
            full_color_data = adata_input.obs_vector(color, layer=layer)
            is_numeric = True
        palette_map = {
            k: list(v) for k, v in (adata_input.uns or {}).items()
            if k.endswith('_colors')
        }

    # --- 2. Resolve splitby categories ---
    if isinstance(splitby_series.dtype, pd.CategoricalDtype):
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

    # --- 3. Auto point_size (matches plotEmbedding's formula) ---
    if point_size is None:
        n_cells = coords.shape[0]
        point_size = max(0.1, min(4, 30000 / n_cells))

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
            expr_max = vmax if vmax is not None else float(np.nanmax(full_color_data))
            expr_min = vmin if vmin is not None else float(np.nanmin(full_color_data))
        except (ValueError, TypeError):
            expr_min, expr_max = vmin, vmax

    # --- 6. Resolve a GLOBAL category→colour mapping for categorical color ---
    # Honors the cytome ``set_categories`` store (order + colors) / AnnData
    # categorical dtype + uns, via the shared resolver.
    cat_to_color = {}
    if not is_numeric:
        present = set(str(v) for v in full_color_data)
        resolved_palette, category_order = _resolve_categorical_style(data, color, palette)
        if category_order:
            ordered = [str(c) for c in category_order if str(c) in present]
            all_color_cats = ordered + sorted(present - set(ordered))
        else:
            all_color_cats = sorted(present)
        if isinstance(resolved_palette, dict):
            cat_to_color = {
                cat: resolved_palette.get(
                    cat, _color_mod.d_color4[i % len(_color_mod.d_color4)])
                for i, cat in enumerate(all_color_cats)
            }
        else:
            if resolved_palette is not None:
                global_palette = list(resolved_palette)
            else:
                stored = palette_map.get(f'{color}_colors')
                global_palette = list(stored) if stored else list(_color_mod.d_color4)
            cat_to_color = {
                cat: global_palette[i % len(global_palette)]
                for i, cat in enumerate(all_color_cats)
            }

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
    panel_legend = 'on_data' if legend_loc == 'on_data' else 'none'

    import anndata as _ad
    # Auto point_size once (consistent dot size across panels); an explicit
    # point_size= always overrides.
    _panel_point_size = (point_size if point_size is not None
                         else max(0.1, min(4, 30000 / coords.shape[0])))

    for i in range(len(axs)):
        if i >= len(variables):
            axs[i].set_visible(False)
            continue
        category = variables[i]
        mask = (splitby_arr == category)
        if int(np.sum(mask)) == 0:
            axs[i].set_visible(False)
            continue

        # Per-panel palette: subset of the global mapping in alphabetical
        # order so the renderer's `sorted(set(...))` gives consistent
        # colours across panels.
        panel_palette = None
        if not is_numeric:
            if groups is not None:
                # Greying via `groups` is delegated to _render_embedding; pass the
                # GLOBAL category→colour dict so colours stay consistent while
                # non-group cells are greyed out.
                panel_palette = dict(cat_to_color)
            else:
                panel_cats = sorted(set(str(v) for v in full_color_data[mask]))
                panel_palette = [cat_to_color[c] for c in panel_cats]

        # Inject the resolved modality into per-panel titles when requested.
        if show_modality_in_title and resolved_modality_for_title:
            panel_title = f"{color} ({resolved_modality_for_title}) in\n{category}"
        else:
            panel_title = f"{color} in\n{category}"

        # Render natively from arrays — no proxy AnnData round-trip.
        _render_embedding(
            axs[i],
            np.asarray(coords[mask], dtype=np.float64),
            np.asarray(full_color_data)[mask],
            (not is_numeric),
            color=color, palette=panel_palette, point_size=_panel_point_size,
            alpha=alpha, rasterized=rasterized, legend_loc=panel_legend,
            groups=groups, strict_groups=False,
            cmap=cmap, vmin=expr_min, vmax=expr_max,
            legend_fontsize=legend_fontsize, legend_fontoutline=legend_fontoutline,
        )
        if not frameon:
            for spine in axs[i].spines.values():
                spine.set_visible(False)
        axs[i].set_title(panel_title)

        # --- per-panel polish that plotEmbedding doesn't expose ---
        axs[i].set_xlim(xy_min[0] - xy_margin[0], xy_max[0] + xy_margin[0])
        axs[i].set_ylim(xy_min[1] - xy_margin[1], xy_max[1] + xy_margin[1])
        if not fix_coordinate_ratio:
            axs[i].set_aspect('auto')

    # --- 8. Global legend coalesces labels from all panels ---
    if not is_numeric and legend_loc == 'right':
        # Each panel has scatter handles with `label=cat` attached but no
        # per-panel legend rendered. _create_global_legend de-duplicates
        # labels across panels and draws one figure-level legend.
        _create_global_legend(
            fig, axs, legend_loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=frameon, marker_size=legend_marker_size,
            fontsize=legend_fontsize,
            max_rows_per_col=max(8, int(fig.get_figheight() * 1.8)),
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

    from ..settings import _savefig
    _savefig(fig, save, writekey='plotEmbeddingsSplit')

    if show_figure:
        plt.show()
    if save and not show_figure:
        plt.close(fig)

        
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


def _get_embedding_and_color(
    data, basis, color, layer=None,
    modality=None, cytome_layer="counts",
    compute_on_fly=True, use_cached_stats=True,
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
    if _is_cytome_input(data):
        coords, obs_df, _palette_map = _cytome_extract(
            data, basis=basis, obs_columns=[color]
        )
        if color in obs_df.columns:
            values = obs_df[color].values
            # Continuous = any numeric dtype (int OR float). Only object/string/
            # bool/category columns (e.g. Leiden, CellTypes) are categorical.
            # (Was `np.floating`, which mis-flagged INTEGER metrics like
            # n_fragments / n_counts as categorical.)
            is_cat = not np.issubdtype(values.dtype, np.number)
            return coords, values, is_cat, None
        # color is not an obs column — resolve via the modality registry.
        with _open_cytome(data) as ds:
            values, resolved = _resolve_cytome_feature_values(
                ds, color,
                modality=modality, cytome_layer=cytome_layer,
                compute_on_fly=compute_on_fly,
                use_cached_stats=use_cached_stats,
            )
        return coords, values, False, resolved

    # AnnData path
    coords = np.asarray(data.obsm[basis])
    if color in data.obs.columns:
        values = np.asarray(data.obs[color])
        # numeric (int/float) → continuous; object/str/bool/category → categorical
        is_cat = not np.issubdtype(values.dtype, np.number)
    else:
        values = np.asarray(data.obs_vector(color, layer=layer))
        is_cat = False
    return coords, values, is_cat, None


# --------------------------------------------------------------------------
# Modality registry — single source of truth for the (modality → entity, id_cols)
# mapping used by the feature resolver. Order matters for auto-detect:
# RNA-first matches the typical workflow where gene names are the canonical
# colour-by, then GA (gene-named pseudo-counts), then ATAC (peak strings),
# then tiles (binned-coord strings). Open for proteins later.
# --------------------------------------------------------------------------
# Modality registry sourced from cytome.utils.modality (cytome 0.1.1+).
# We re-derive the legacy plot-side tuple shape (mod, entity, id_cols)
# from the canonical cytome MODALITY_REGISTRY so existing call sites that
# unpacked 3-tuples keep working without churn.
from cytome.utils.modality import MODALITY_REGISTRY as _CYTOME_MODALITY_REGISTRY
from cytome.utils.modality import (
    modality_has_feature as _modality_has_feature_cytome,
    read_feature_column as _read_feature_column_cytome,
    read_feature_columns as _read_feature_columns_cytome,
    modality_cell_depth as _modality_cell_depth_cytome,
)
_PLOT_MODALITY_REGISTRY = [
    (mod, entity, id_cols) for mod, entity, _idx, id_cols in _CYTOME_MODALITY_REGISTRY
]


def _modality_has_feature(ds, mod_name, entity, id_cols, feature):
    """Wrapper around cytome's helper preserving the plot-side calling
    convention (entity / id_cols already destructured by caller). Always
    delegates to the cytome registry under the hood."""
    return _modality_has_feature_cytome(ds, mod_name, feature)


def _read_feature_column(ds, modality, layer_name, feat_idx, batch_size=2048):
    """Delegates to ``cytome.utils.modality.read_feature_column``."""
    return _read_feature_column_cytome(
        ds, modality, layer_name, feat_idx, batch_size=batch_size,
    )


def _modality_cell_depth(ds, modality, use_cached_stats=True, batch_size=2048):
    """Delegates to ``cytome.utils.modality.modality_cell_depth``."""
    return _modality_cell_depth_cytome(
        ds, modality, use_cached_stats=use_cached_stats, batch_size=batch_size,
    )


def _cached_stat_is_fresh(params, ds):
    """A cached normalization-params dict is stale once its cell-indexed
    ``cell_depth`` no longer matches ``ds.n_cells`` (e.g. after a filter_cells
    on a cytome predating the subset-invalidation fix). Returns False so the
    caller recomputes instead of broadcasting a wrong-length vector."""
    if not isinstance(params, dict) or "cell_depth" not in params:
        return True  # nothing cell-indexed to check
    try:
        return int(np.asarray(params["cell_depth"]).shape[0]) == int(ds.n_cells)
    except Exception:
        return True


def _params_feature_len_matches_modality(params, ds, modality, feature_key):
    """True if a cached normalization payload's per-feature vector
    (``params[feature_key]``) length matches ``modality``'s feature count.

    Guards against returning a payload computed for a DIFFERENT modality — e.g.
    the modality-blind legacy 'tfidf_params' (ATAC idf) for a 'tiles' request, or
    a legacy 'infog_params' (RNA inv_gene_depth) for an ATAC request. Permissive
    (returns True) when the feature count can't be determined, so behaviour is
    unchanged where the meta is unavailable."""
    if not isinstance(params, dict) or feature_key not in params:
        return True
    try:
        mm = ds.matrix_meta(f"{modality}_counts")
        n_feat = mm.get("n_cols") if isinstance(mm, dict) else None
        if n_feat is None:
            return True
        return int(np.asarray(params[feature_key]).shape[0]) == int(n_feat)
    except Exception:
        return True


def _tfidf_idf_matches_modality(params, ds, modality):
    """TF-IDF specialization of :func:`_params_feature_len_matches_modality`
    (per-feature vector = ``idf``)."""
    return _params_feature_len_matches_modality(params, ds, modality, "idf")


def _ensure_infog_params(ds, modality, use_cached_stats=True, batch_size=2048):
    """Return cached ``{modality}_infog_params`` (or compute now and cache).
    Falls through to legacy ``infog_params`` once with a DeprecationWarning
    when only the unprefixed key is present. A cached payload whose
    ``cell_depth`` length no longer matches ``ds.n_cells`` is treated as a miss
    (self-heals a cytome filtered before the cached-stats invalidation fix)."""
    if use_cached_stats:
        new_key = f"{modality}_infog_params"
        v = ds.metadata.get(new_key)
        if (v is not None and _cached_stat_is_fresh(v, ds)
                and _params_feature_len_matches_modality(v, ds, modality, "inv_gene_depth")):
            return v
        legacy = ds.metadata.get("infog_params")
        # Feature-count guard: the un-prefixed legacy 'infog_params' is modality-
        # blind (historically RNA); do NOT return it for a request on a modality
        # with a different feature count (e.g. ATAC peaks) — route to recompute.
        if (legacy is not None and _cached_stat_is_fresh(legacy, ds)
                and _params_feature_len_matches_modality(legacy, ds, modality, "inv_gene_depth")):
            import warnings as _warnings
            _warnings.warn(
                f"Using legacy 'infog_params' as '{new_key}'. Recompute "
                f"with piaso.tl.infog(ds, modality='{modality}') to refresh.",
                DeprecationWarning, stacklevel=3,
            )
            return legacy
    # Cache miss / forced refresh — compute via piaso.tl.infog (streaming, lazy)
    from ..tools._normalization import infog as _infog
    _infog(ds, save_layer=False, streaming=True, batch_size=batch_size, verbosity=0)
    return ds.metadata.get(f"{modality}_infog_params") or ds.metadata.get("infog_params")


def _ensure_tfidf_params(ds, modality, use_cached_stats=True, batch_size=2048):
    """Return cached ``{modality}_tfidf_params`` (or compute now and cache).

    Delegates to ``_runTFIDF._load_or_compute_tfidf_stats`` so the runSVD
    ``auto_tfidf=True`` path, COSG ``layer='tfidf'``, and plotting
    ``cytome_layer='tfidf'`` all share the same cache-or-compute helper.
    """
    # Legacy un-prefixed key compatibility (preserved for older cytomes). A
    # cached payload is stale/wrong when its cell_depth length no longer matches
    # n_cells (filtered cytome) OR — critically for the modality-blind legacy
    # 'tfidf_params' — its idf length does not match THIS modality's feature
    # count. The un-prefixed key does not record which modality it was computed
    # for, so a payload built for ATAC peaks (idf ~n_peaks) must NOT be returned
    # for a 'tiles' request (idf ~n_tiles); the feature-count guard routes it to
    # recompute instead of broadcasting a wrong-length idf.
    if use_cached_stats:
        new_key = f"{modality}_tfidf_params"
        if (new_key in ds.metadata and _cached_stat_is_fresh(ds.metadata[new_key], ds)
                and _tfidf_idf_matches_modality(ds.metadata[new_key], ds, modality)):
            return ds.metadata[new_key]
        legacy = ds.metadata.get("tfidf_params")
        if (legacy is not None and _cached_stat_is_fresh(legacy, ds)
                and _tfidf_idf_matches_modality(legacy, ds, modality)):
            import warnings as _warnings
            _warnings.warn(
                f"Using legacy 'tfidf_params' as '{new_key}'. "
                f"Recompute with piaso.tl.compute_tfidf_stats(ds, modality='{modality}') "
                f"to refresh.", DeprecationWarning, stacklevel=3,
            )
            return legacy
    from ..tools._runTFIDF import _load_or_compute_tfidf_stats
    return _load_or_compute_tfidf_stats(
        ds, modality=modality, batch_size=batch_size,
        force_recompute=not use_cached_stats,
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
    if modality is None:
        # Search all registry entries.
        candidates = []
        for mod, entity, id_cols in _PLOT_MODALITY_REGISTRY:
            hit = _modality_has_feature(ds, mod, entity, id_cols, feature)
            if hit is not None:
                feat_idx, _ = hit
                candidates.append((mod, entity, feat_idx))
        if not candidates:
            tried = [m for m, _, _ in _PLOT_MODALITY_REGISTRY]
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
            ((m, e, ic) for m, e, ic in _PLOT_MODALITY_REGISTRY if m == modality),
            None,
        )
        if match is None:
            raise ValueError(
                f"Unknown modality '{modality}' for feature resolution. "
                f"Known: {[m for m, _, _ in _PLOT_MODALITY_REGISTRY]}."
            )
        _, entity, id_cols = match
        hit = _modality_has_feature(ds, modality, entity, id_cols, feature)
        if hit is None:
            # Look in OTHER modalities so the error message is helpful.
            other_hits = []
            for m, e, ic in _PLOT_MODALITY_REGISTRY:
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
        raise ValueError(
            f"Cannot compute '{cytome_layer}' on-the-fly: required source "
            f"matrix '{counts_name}' is missing from the cytome."
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
                     for m, e, ic in _PLOT_MODALITY_REGISTRY]
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
    show=True,
    ax=None,
    palette=None,
    legend_loc="right",
    legend_fontsize=10,
    legend_fontoutline=None,
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
    return_fig=False,
    **kwargs,
):
    """Plot a 2-D embedding colored by a cell annotation or continuous value.

    Round 7: now returns ``None`` by default — pass ``return_fig=True``
    to get ``(fig, ax)`` back (the legacy behaviour).

    Supports both ``cytome.Dataset`` and ``AnnData`` objects.

    Parameters
    ----------
    data
        A ``cytome.Dataset`` or ``AnnData`` object.
    color : str or list of str
        Column in ``cells`` / ``obs`` for colouring, or a **feature name**
        (gene, peak, tile, or GA gene — resolved via the cytome modality
        registry; pass ``modality=`` to disambiguate). A single ``str`` draws one
        panel; a ``list``/``tuple`` of strings draws a multi-panel grid (one
        panel per entry, scanpy ``sc.pl.umap``-style).
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
        from the number of cells: ``30000 / n_cells`` clamped to [0.1, 4]
        (explicit value overrides).
    alpha : float
        Point transparency.  Default 1.0 (opaque).
    frameon : bool, optional
        Whether to show axis frame (spines).  If None, uses
        ``piaso.settings._frameon`` (default False).
    save : str, bool, or None
        Figure save behavior:
        - ``None``: don't save.
        - Full path (``'/path/to/fig.png'``): save directly.
        - Suffix (``'_leiden'``): save to ``piaso.settings.figdir``.
        - ``True``: auto-name and save to ``piaso.settings.figdir``.
    show : bool
        Whether to call ``plt.show()``.
    ax : matplotlib Axes, optional
        Pre-existing axes to draw on.
    palette : list[str], optional
        Color palette for categorical data.  If None, checks
        ``adata.uns['{color}_colors']`` first, then falls back to
        ``piaso.pl.color.d_color4``.
    groups : str or list of str, optional
        For a **categorical** ``color``: highlight only these category values —
        they keep their palette colour while all other cells are greyed out
        (drawn behind), like ``sc.pl.umap(groups=...)``. Palette indices come
        from the full category list, so colours match an unfiltered plot. Raises
        ``ValueError`` if a name is not among the categories.
    na_color : str, default ``'lightgray'``
        Colour for the greyed-out (non-``groups``) cells.
    legend_loc : str
        ``'right'`` (outside), ``'on_data'`` (centroid labels), or ``'none'``.
    legend_fontsize : int
        Font size for legend labels.
    legend_fontoutline : float or None
        Width of text outline for ``legend_loc='on_data'`` labels.
        Adds a contrasting stroke around text for readability.  If None
        (default), no outline is drawn.
    legend_ncol : int, optional
        Number of legend columns.  If None, auto-calculated (~12 per column).
    rasterized : bool
        Rasterize scatter points for smaller vector files.
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

    Returns
    -------
    fig, ax
        For single-color: ``(fig, ax)`` where ``ax`` is the matplotlib
        Axes that was drawn into.
    fig, axs
        For list-color: ``(fig, [ax0, ax1, ...])`` — one Axes per entry
        in ``color``; trailing empty grid cells are hidden via
        ``set_visible(False)``.
    """
    from . import color as _color_mod
    from .. import settings as _settings

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
            for i, color_i in enumerate(color_list):
                plotEmbedding(
                    data, color=color_i, basis=basis, layer=layer,
                    title=_panel_titles[i],             # per-panel title (or None)
                    figsize=None,
                    point_size=point_size, alpha=alpha,
                    frameon=frameon,
                    save=None,                          # save the parent fig
                    show=False, ax=axs_flat[i],
                    palette=palette,
                    legend_loc=legend_loc, legend_fontsize=legend_fontsize,
                    legend_fontoutline=legend_fontoutline, legend_ncol=legend_ncol,
                    rasterized=rasterized, dpi=dpi,
                    vmin=vmin, vmax=vmax, vmin_pct=vmin_pct, vmax_pct=vmax_pct,
                    cmap=cmap,
                    show_axes_arrow=show_axes_arrow,
                    axes_arrow_loc=axes_arrow_loc,
                    modality=modality, cytome_layer=cytome_layer,
                    compute_on_fly=compute_on_fly,
                    use_cached_stats=use_cached_stats,
                    show_modality_in_title=show_modality_in_title,
                    fix_coordinate_ratio=fix_coordinate_ratio,
                    show_axis_ticks=show_axis_ticks,
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    legend_marker_size=legend_marker_size,
                    groups=groups, na_color=na_color,
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
            if _suptitle is not None:
                fig.suptitle(_suptitle)
            from ..settings import _savefig
            _savefig(fig, save, writekey='plotEmbedding')
            if show:
                plt.show()
            if return_fig:
                return fig, axs_flat
            return None

    coords, values, is_cat, resolved_modality = _get_embedding_and_color(
        data, basis, color, layer=layer,
        modality=modality, cytome_layer=cytome_layer,
        compute_on_fly=compute_on_fly, use_cached_stats=use_cached_stats,
    )

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
        point_size = max(0.1, min(4, 30000 / n_cells))

    if ax is None:
        # figsize=None → matplotlib uses rcParams['figure.figsize']
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # Resolve the categorical palette + display order from the cytome
    # ``set_categories`` store (or AnnData categorical dtype + uns) before
    # delegating to the shared array-based renderer (no proxy AnnData needed).
    category_order = None
    if is_cat:
        palette, category_order = _resolve_categorical_style(data, color, palette)
    _render_embedding(
        ax, coords, values, is_cat, color=color, palette=palette,
        category_order=category_order,
        point_size=point_size, alpha=alpha, rasterized=rasterized,
        groups=groups, na_color=na_color, legend_loc=legend_loc,
        legend_ncol=legend_ncol, legend_marker_size=legend_marker_size,
        legend_fontsize=legend_fontsize, legend_fontoutline=legend_fontoutline,
        cmap=cmap, vmin=vmin, vmax=vmax,
    )

    # --- frameon ---
    if not frameon:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if title is None:
        title_str = color
        if show_modality_in_title and resolved_modality:
            title_str = f"{color} ({resolved_modality})"
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

    from ..settings import _savefig
    _savefig(fig, save, writekey='plotEmbedding')
    if show:
        plt.show()
    if return_fig:
        return fig, ax
    return None


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