"""One definition of "what order do the groups go in".

Every plot that groups cells by a label has to decide the order, and until
this existed each one decided separately with ``sorted(set(str(g)))``. That is
a lexicographic sort, so Leiden clusters came out 0, 1, 10, 11, 2 — while
``plotEmbedding`` used the pandas categorical order and produced 0, 1, 2, 3.
The UMAP legend and the violin below it then disagreed about which cluster was
which colour, on the same figure.

The order here is the embedding's order:

1. an explicit ``categories_order`` if the caller passed one;
2. the pandas categorical order, which is what AnnData carries and what
   ``plotEmbedding`` reads;
3. otherwise a natural sort, so "2" comes before "10" rather than after it.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["natural_key", "resolve_group_order"]


def natural_key(value) -> tuple:
    """Sort key that reads digit runs as numbers: c2 < c10, and "10" > "2"."""
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def resolve_group_order(
    labels: Iterable,
    categories_order: Optional[Sequence] = None,
    present_only: bool = True,
) -> list:
    """Return the group labels of ``labels``, as strings, in display order.

    Parameters
    ----------
    labels
        The per-cell group labels: a Series, a categorical, or any array.
    categories_order
        An explicit order. Entries not present in ``labels`` are dropped when
        ``present_only``.
    present_only
        Drop categories with no cells. Categorical columns routinely carry
        levels left behind by a subset, and an empty violin is not informative.
    """
    ser = labels if isinstance(labels, pd.Series) else pd.Series(np.asarray(labels))
    present = {str(g) for g in ser.dropna().unique()}

    if categories_order is not None:
        order = [str(g) for g in categories_order]
    elif isinstance(ser.dtype, pd.CategoricalDtype):
        order = [str(g) for g in ser.cat.categories]
    else:
        return sorted(present, key=natural_key)

    seen, out = set(), []
    for g in order:
        if g in seen:
            continue
        seen.add(g)
        if not present_only or g in present:
            out.append(g)
    # anything present but unlisted still has to be drawn somewhere
    out.extend(sorted(present - seen, key=natural_key))
    return out
