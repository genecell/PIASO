"""One answer to "which colour is this category?", for every PIASO plot.

Every plot that colours cells or groups by a label used to decide the colours
itself, and most decided by *position*: the i-th category drawn got the i-th
palette entry. Position is not identity. The i-th category drawn depends on
what this particular plot draws -- a ``cell_mask`` that leaves out a type, a
group dropped for having no data, a Sankey's two rails concatenated, a stacked
bar sorted by size -- so the same type changed colour from one panel of a
figure to the next. Seven such sites were found in the audit of 2026-10-27.

The rule here:

* **Colours are assigned once per column, by name, over the column's
  canonical order** -- the stored order (a cytome's ``set_categories``, or the
  categorical order an AnnData carried into the store), else the pandas
  categorical order, followed by any other value in natural order. A palette
  given as a list, and the default palette, are matched to that order; a dict
  palette, and a store's colour map, are matched by name. Nothing a plot
  leaves out shifts anything.
* **The display order is separate** (``order=``, a dict palette's key order,
  or the canonical order). A plot may draw a subset, or sort, without
  touching a colour.
* **Missing labels are not a category.** None, NaN, ``pd.NA`` -- and the
  string ``'nan'``, which is what a cytome text column holds for a NaN
  written into it -- get ``na_color`` and the legend label ``"NA"``.

A store converted from AnnData keeps the categorical order only in its
``_column_meta`` table, and the ``uns`` colour list (positional, scanpy's
convention) in ``metadata['<col>_colors']``. Plotting never read the order, so
the colour list was matched to a natural sort instead and every type changed
colour. It is read here through cytome's own reader
(``cytome.io.convert_anndata._restore_column_dtypes``, the function
``to_anndata`` uses), not SQL, and only when it still covers the column's
values; an order the column has outgrown is stale and ignored, as
``ds.get_categories`` does for its own entries.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._group_order import natural_key

__all__ = [
    "CategoryStyle",
    "DEFAULT_NA_COLOR",
    "NA_LABEL",
    "missing_mask",
    "label_strings",
    "resolve_category_style",
    "stored_category_style",
    "SORT_ORDERS",
    "draw_sequence",
]

#: Legend label for cells whose label is missing.
NA_LABEL = "NA"
#: Colour for cells whose label is missing (and, in the embedding, for cells
#: pushed into the background by ``groups=``).
DEFAULT_NA_COLOR = "lightgray"

#: Draw orders for categorical scatters (``sort_order=``).
SORT_ORDERS = ("category", "size", "random")


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def _as_object(values) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=object)
    if isinstance(values, pd.Categorical):
        return np.asarray(values, dtype=object)
    return np.asarray(values, dtype=object).ravel()


def missing_mask(values) -> np.ndarray:
    """True where a label is missing: None, NaN, ``pd.NA`` or the text ``'nan'``.

    ``'nan'`` counts because that is what a cytome stores when a NaN is written
    into a text column (``str(np.nan)``); read back, it is a string, and
    treating it as a category drew a "nan" swatch in the palette's colour.
    """
    arr = _as_object(values)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    miss = np.asarray(pd.isna(arr), dtype=bool)
    # elementwise == on an object array runs in C and is False for non-strings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        miss |= np.asarray(arr == "nan", dtype=bool)
    return miss


def label_strings(values) -> np.ndarray:
    """The labels as strings (object array); missing labels become None."""
    arr = _as_object(values)
    out = np.empty(arr.shape, dtype=object)
    miss = missing_mask(arr)
    if (~miss).any():
        # pandas' str conversion runs in C; a Python loop costs ~0.3 s per
        # million cells
        out[~miss] = pd.Series(arr[~miss], dtype=object).astype(str).to_numpy(dtype=object)
    out[miss] = None
    return out


def _present(values) -> set:
    labels = label_strings(values)
    out = set(pd.unique(labels[labels != None]))  # noqa: E711 (elementwise)
    return out


def _categorical_order(values) -> Optional[List[str]]:
    if isinstance(values, pd.Categorical):
        return [str(c) for c in values.categories]
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        return [str(c) for c in values.cat.categories]
    return None


# ---------------------------------------------------------------------------
# what the object stores
# ---------------------------------------------------------------------------

def _column_meta_categories(ds, column) -> Optional[List[str]]:
    """The categorical order ``from_anndata`` recorded for a cells column.

    Through cytome's own reader, so the schema stays cytome's business. The
    reader is private to cytome; if a future cytome renames it, this returns
    None and the plot falls back to the order it can see.
    """
    try:
        from cytome.io.convert_anndata import _restore_column_dtypes
    except Exception:
        return None
    try:
        frame = pd.DataFrame({column: pd.Series([], dtype=object)})
        frame = _restore_column_dtypes(ds._conn, "cells", frame)
    except Exception:
        return None
    col = frame[column]
    if isinstance(col.dtype, pd.CategoricalDtype):
        return [str(c) for c in col.cat.categories]
    return None


def stored_category_style(data, column):
    """``(order, colors)`` as the object stores them, either may be None.

    ``colors`` is a dict (by name) or a list (positional over ``order``, or
    over the natural order of the values when there is no stored order).

    AnnData: the categorical order and ``uns['<column>_colors']``.
    cytome: ``ds.get_categories(column)`` ({order, colors}); without it, the
    categorical order kept from the AnnData it was converted from, and
    ``metadata['<column>_colors']``. ``get_categories`` drops a stale entry by
    writing, which a store opened read-only refuses; that error means the
    entry is stale, so it is treated as absent rather than ending the lookup.
    """
    from ..utils._cytome_compat import is_cytome_input, open_cytome

    if data is None or column is None:
        return None, None
    try:
        if is_cytome_input(data):
            with open_cytome(data, mode="r") as ds:
                try:
                    actual = ds.cells.resolve_column(column)
                except Exception:
                    actual = column if column in ds.cells.columns else None
                if actual is None:
                    return None, None
                entry = None
                getter = getattr(ds, "get_categories", None)
                if getter is not None:
                    try:
                        entry = getter(actual)
                    except Exception:
                        entry = None
                order = list(entry["order"]) if entry and entry.get("order") else None
                colors = dict(entry["colors"]) if entry and entry.get("colors") else None
                if order is None:
                    order = _column_meta_categories(ds, actual)
                if colors is None:
                    stored = ds.metadata.get(f"{actual}_colors")
                    if stored is None and actual != column:
                        stored = ds.metadata.get(f"{column}_colors")
                    if stored is not None:
                        colors = (dict(stored) if isinstance(stored, dict)
                                  else [str(c) for c in list(stored)])
                return order, colors
        obs = getattr(data, "obs", None)
        if obs is None or column not in obs.columns:
            return None, None
        order = _categorical_order(obs[column])
        colors = None
        uns = getattr(data, "uns", None)
        key = f"{column}_colors"
        if uns is not None and key in uns:
            stored = uns[key]
            colors = (dict(stored) if isinstance(stored, dict)
                      else [str(c) for c in list(stored)])
        return order, colors
    except Exception:
        # Styling is best-effort: a metadata read must never break a plot.
        return None, None


def _column_values(data, column):
    """The full column, keeping a categorical dtype where one exists."""
    from ..utils._cytome_compat import is_cytome_input, open_cytome

    if is_cytome_input(data):
        with open_cytome(data, mode="r") as ds:
            return np.asarray(ds.cells[column])
    return data.obs[column]


# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------

def _palette_from_name(name: str, n: int) -> Optional[List[str]]:
    """``n`` colours from a named colormap or palette, or None if unknown.

    A qualitative colormap (``tab10``, ``Set2``, ``Paired``: a ListedColormap
    of fewer than 256 colours) gives its colours in order, as seaborn does;
    sampling one evenly would skip most of them. A continuous map is sampled
    evenly from end to end -- the ramp for ordered categories (ages, stages).
    """
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, to_hex

    try:
        cm = mpl.colormaps.get_cmap(name)
    except (ValueError, KeyError, TypeError):
        cm = None
    if cm is not None:
        if isinstance(cm, ListedColormap) and cm.N < 256:
            return [to_hex(cm.colors[i % cm.N]) for i in range(n)]
        return [to_hex(cm(i / max(n - 1, 1))) for i in range(n)]
    try:
        import seaborn as sns
        return [to_hex(c) for c in sns.color_palette(name, n)]
    except Exception:
        return None


def _default_palette():
    from . import color as _color_mod
    return list(_color_mod.d_color4)


# ---------------------------------------------------------------------------
# the style
# ---------------------------------------------------------------------------

class CategoryStyle:
    """Order and colours for one categorical column.

    ``order`` is the display order and lists every category of the column
    (unused levels of a categorical included); ``colors`` maps each of them to
    a colour. ``present(values)`` is the part of ``order`` a plot draws.
    """

    def __init__(self, column, order, colors, na_color=DEFAULT_NA_COLOR,
                 canonical=None):
        self.column = column
        self.order: List[str] = list(order)
        self.colors: Dict[str, object] = dict(colors)
        self.na_color = na_color
        self.canonical: List[str] = list(canonical if canonical is not None else order)

    def __repr__(self):
        return (f"CategoryStyle(column={self.column!r}, order={self.order!r}, "
                f"na_color={self.na_color!r})")

    def color(self, category):
        """The colour of ``category``; ``na_color`` for a missing label."""
        if category is None:
            return self.na_color
        return self.colors.get(str(category), self.na_color)

    def present(self, values) -> List[str]:
        """The categories that occur in ``values``, in display order."""
        seen = _present(values)
        return [c for c in self.order if c in seen]

    def codes(self, values) -> np.ndarray:
        """Index of each value's category in ``order``; -1 for a missing one."""
        labels = label_strings(values)
        return np.asarray(pd.Categorical(labels, categories=self.order).codes,
                          dtype=np.int64)

    def rgba(self, values) -> np.ndarray:
        """One RGBA row per value; missing labels in ``na_color``."""
        from matplotlib.colors import to_rgba
        table = np.array([to_rgba(self.colors[c]) for c in self.order] +
                         [to_rgba(self.na_color)], dtype=float).reshape(-1, 4)
        codes = self.codes(values)
        codes = np.where(codes < 0, len(self.order), codes)
        return table[codes]

    def palette_for(self, categories: Iterable) -> List[object]:
        """Colours for ``categories``, in that order."""
        return [self.color(c) for c in categories]


def resolve_category_style(data=None, column=None, values=None, *, palette=None,
                           order: Optional[Sequence] = None,
                           na_color=DEFAULT_NA_COLOR,
                           default_palette: Optional[Sequence] = None,
                           read_stored: bool = True) -> CategoryStyle:
    """Resolve the order and colours of a categorical column.

    Parameters
    ----------
    data
        AnnData or a cytome Dataset/path, for the stored style; may be None.
    column
        The column's name, for the stored style; may be None.
    values
        The column's values, ALL of them (a plot of a subset still passes the
        whole column, so a category the subset leaves out keeps its place). A
        pandas categorical keeps its order. Read from ``data[column]`` when
        None.
    palette
        A dict (by name; its key order also leads the display order), a list
        (positional over the canonical order), or the name of a colormap or
        seaborn palette. Entries it does not cover fall back to the stored
        colours, then to the default palette.
    order
        A display order to lead with (e.g. ``categories_order``). Categories
        it omits follow in the canonical order. It does not move colours.
    default_palette
        The fallback palette (PIASO's ``d_color4`` when None), positional over
        the canonical order.
    read_stored
        Read the stored order and colours (False for values that name no
        stored column).
    """
    if values is None:
        if data is None or column is None:
            raise ValueError("resolve_category_style needs values or (data, column)")
        values = _column_values(data, column)

    present = _present(values)
    stored_order, stored_colors = (stored_category_style(data, column)
                                   if read_stored else (None, None))
    cat_order = _categorical_order(values)

    # --- canonical order: what colours are positioned over ------------------
    base = None
    for candidate in (stored_order, cat_order):
        if candidate:
            cand = [str(c) for c in candidate]
            # an order that does not cover the column's values is stale
            if present <= set(cand):
                base = cand
                break
    if base is None and stored_order and cat_order is None:
        # The order recorded for this column (set_categories, or the AnnData
        # it was converted from) predates its values: the column was rewritten
        # since. Positions in a colour list stored with it no longer say which
        # colour is whose, so both are dropped, and the plot says so.
        warnings.warn(
            f"the category order recorded for {column!r} does not cover its "
            f"values (e.g. {sorted(present - set(map(str, stored_order)))[:3]}), "
            f"so the column was rewritten since; that order and the colours "
            f"stored with it are ignored.", UserWarning, stacklevel=3)
        if not isinstance(stored_colors, dict):
            stored_colors = None
    seen = set()
    canonical: List[str] = []
    for c in (base or []):
        if c not in seen:
            seen.add(c)
            canonical.append(c)
    canonical += sorted(present - seen, key=natural_key)

    # --- display order --------------------------------------------------------
    lead = []
    if order is not None:
        lead = [str(c) for c in order]
    elif isinstance(palette, dict):
        lead = [str(c) for c in palette]
    canon_set = set(canonical)
    display: List[str] = []
    seen = set()
    for c in lead:
        if c in canon_set and c not in seen:
            seen.add(c)
            display.append(c)
    display += [c for c in canonical if c not in seen]

    # --- colours ----------------------------------------------------------------
    default = list(default_palette) if default_palette is not None else _default_palette()
    colors: Dict[str, object] = {c: default[i % len(default)]
                                 for i, c in enumerate(canonical)}

    if isinstance(stored_colors, dict):
        colors.update({str(k): v for k, v in stored_colors.items() if str(k) in canon_set})
    elif stored_colors is not None:
        positional = base if base is not None else canonical
        if len(stored_colors) >= len(positional):
            colors.update({c: stored_colors[i] for i, c in enumerate(positional)})
        else:
            warnings.warn(
                f"the stored colours for {column!r} ({len(stored_colors)}) are fewer "
                f"than its categories ({len(positional)}), so they no longer say "
                f"which colour is whose; the default palette is used.",
                UserWarning, stacklevel=3)

    if isinstance(palette, dict):
        colors.update({str(k): v for k, v in palette.items() if str(k) in canon_set})
    elif isinstance(palette, str):
        named = _palette_from_name(palette, len(display))
        if named is None:
            warnings.warn(f"palette={palette!r} is not a colormap or palette name; "
                          f"using the default colours.", UserWarning, stacklevel=3)
        else:
            # a ramp runs along the order the categories are shown in
            colors.update(dict(zip(display, named)))
    elif palette is not None:
        pal = list(palette)
        if pal:
            colors.update({c: pal[i % len(pal)] for i, c in enumerate(canonical)})

    return CategoryStyle(column, display, colors, na_color=na_color, canonical=canonical)


# ---------------------------------------------------------------------------
# draw order
# ---------------------------------------------------------------------------

def draw_sequence(style: CategoryStyle, values, sort_order: Optional[str] = "category",
                  categories: Optional[Sequence[str]] = None):
    """Categories in the order to draw them, for ``sort_order``.

    ``'category'`` (default): the display order, each category drawn over the
    ones before it. ``'size'``: the largest first, so a rare type is drawn
    last, on top -- 100 astrocytes among 20,000 excitatory neurons had no
    visible pixel in category order. ``'random'`` is handled by the caller
    (one scatter, cells shuffled); here it returns the display order.
    """
    cats = list(categories) if categories is not None else style.present(values)
    if sort_order in (None, "category", "random"):
        return cats
    if sort_order == "size":
        labels = label_strings(values)
        counts = pd.Series([v for v in labels if v is not None]).value_counts()
        # stable: ties keep the display order
        return sorted(cats, key=lambda c: -int(counts.get(c, 0)))
    raise ValueError(f"sort_order={sort_order!r}: choose from {SORT_ORDERS}")
