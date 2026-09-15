"""Ordering two label sets so the figure between them reads.

A Sankey or a confusion matrix has two axes of categories and a table of
counts joining them. Which order to draw them in is a choice, and the wrong
choice makes agreement look like disagreement: ribbons cross that did not have
to, and a diagonal that is really there does not appear.

Two questions, two answers:

``svd_order``
    Nothing is fixed; order both axes together. The first singular vector of
    the normalised table is a one-dimensional summary of "which end of the
    other axis does this category sit at", and sorting both by it brings the
    mass onto the diagonal. This is what ``plotConfusionMatrix`` has always
    done.

``barycentre_order``
    One axis is already decided -- by the reference's own categorical order,
    say -- and the other should follow it. Each free category is placed at the
    mean position of its mass along the fixed axis, which is the standard
    crossing-minimisation heuristic.

``hclust_order``
    Average-linkage on each axis's own profile, the clustermap convention.
    Unlike SVD it groups categories that behave alike even when the table has
    no single dominant direction, which is where the first singular vector
    stops being a useful summary.

``hclust_symmetric_order``
    For a **square table over one label set** -- a confusion matrix of two
    annotations of the same categories, a leave-one-type-out similarity --
    where the same order belongs on both axes. Symmetrises, clusters once,
    and returns that order twice. Applying two independent orders to such a
    table moves the diagonal off the diagonal, which is the one thing the
    picture is for.

The distinction matters and is easy to miss: **SVD orders the two axes
jointly**, so it cannot honour an axis you have fixed. Asking it to is asking
for a different answer than the one it computes.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["svd_order", "barycentre_order", "hclust_order",
           "hclust_symmetric_order", "resolve_pair_order"]


def _normalised(ct: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise, so a large category does not dominate the direction."""
    totals = ct.sum(axis=1).replace(0, np.nan)
    return ct.div(totals, axis=0).fillna(0.0)


def svd_order(ct: pd.DataFrame) -> tuple[list, list]:
    """Order both axes by the first singular vector. Returns (rows, cols)."""
    if ct.shape[0] <= 1 or ct.shape[1] <= 1:
        return list(ct.index), list(ct.columns)
    norm = _normalised(ct)
    try:
        u, s, vt = np.linalg.svd(norm.values, full_matrices=False)
    except np.linalg.LinAlgError:
        warnings.warn("svd_order: SVD did not converge; keeping the input order.",
                      RuntimeWarning, stacklevel=2)
        return list(ct.index), list(ct.columns)
    rows = [ct.index[i] for i in np.argsort(u[:, 0] * s[0])]
    cols = [ct.columns[i] for i in np.argsort(vt.T[:, 0] * s[0])]
    return rows, cols


def barycentre_order(ct: pd.DataFrame, fixed: Sequence, axis: int = 0) -> list:
    """Order the free axis to line up with an axis that is already fixed.

    Parameters
    ----------
    ct
        Counts, rows against columns.
    fixed
        The order of the fixed axis.
    axis
        0 if the rows are fixed and the columns are being ordered, 1 for the
        other way round.
    """
    if axis == 0:
        table, free = ct.reindex(index=list(fixed)).fillna(0.0), list(ct.columns)
        mass = table.values                      # fixed x free
    else:
        table, free = ct.reindex(columns=list(fixed)).fillna(0.0), list(ct.index)
        mass = table.values.T                    # fixed x free

    pos = np.arange(len(fixed), dtype=float)[:, None]
    weight = mass.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = (mass * pos).sum(axis=0) / weight
    # A category with no cells on the fixed axis has no barycentre; keep it at
    # the end rather than letting NaN decide where it lands.
    centre = np.where(weight > 0, centre, np.inf)
    return [free[i] for i in np.argsort(centre, kind="stable")]


def _leaf_order(dist_square: np.ndarray, method: str = "average") -> np.ndarray:
    """Leaf order of an average-linkage tree over a square distance matrix."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = dist_square.shape[0]
    if n <= 2:
        return np.arange(n)
    d = np.asarray(dist_square, dtype=np.float64).copy()
    d = 0.5 * (d + d.T)          # squareform(checks=False) would silently use
    np.fill_diagonal(d, 0.0)     # the upper triangle of an asymmetric input
    d[~np.isfinite(d)] = 0.0
    return np.asarray(leaves_list(linkage(squareform(d, checks=False),
                                          method=method)))


def _profile_distance(values: np.ndarray) -> np.ndarray:
    """Correlation distance between rows, falling back to Euclidean.

    A row with no variance has no correlation with anything; rather than
    return NaN for it, those pairs fall back to a normalised Euclidean
    distance so the row still lands somewhere sensible.
    """
    v = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(v)
    d = 1.0 - c
    bad = ~np.isfinite(d)
    if bad.any():
        diff = v[:, None, :] - v[None, :, :]
        e = np.sqrt((diff ** 2).sum(axis=-1))
        scale = e.max() or 1.0
        d[bad] = (e / scale)[bad]
    np.fill_diagonal(d, 0.0)
    return d


def hclust_order(ct: pd.DataFrame, method: str = "average") -> tuple[list, list]:
    """Order each axis by clustering its own profiles. Returns (rows, cols)."""
    rows, cols = list(ct.index), list(ct.columns)
    if ct.shape[0] > 2:
        norm = _normalised(ct).values
        rows = [ct.index[i] for i in _leaf_order(_profile_distance(norm), method)]
    if ct.shape[1] > 2:
        normT = _normalised(ct.T).values
        cols = [ct.columns[i] for i in _leaf_order(_profile_distance(normT), method)]
    return rows, cols


def hclust_symmetric_order(ct: pd.DataFrame, method: str = "average") -> list:
    """One order for both axes of a square, same-label-set table.

    The table is read as a similarity: it is symmetrised, turned into a
    distance by ``1 - s`` after scaling to [0, 1], and clustered once.

    Raises
    ------
    ValueError
        If the two axes do not carry the same label set — in which case one
        order cannot apply to both, and there is nothing sensible to return.
    """
    if list(ct.index) != list(ct.columns):
        if set(ct.index) != set(ct.columns):
            raise ValueError(
                "hclust_symmetric_order needs the same labels on both axes; "
                f"got {len(set(ct.index) - set(ct.columns))} row label(s) "
                f"absent from the columns and "
                f"{len(set(ct.columns) - set(ct.index))} the other way. Use "
                "'hclust' to order each axis independently.")
        ct = ct.loc[:, list(ct.index)]      # same set, different order
    if ct.shape[0] <= 2:
        return list(ct.index)
    s = np.asarray(ct.values, dtype=np.float64)
    s = np.nan_to_num(s)
    hi = np.abs(s).max() or 1.0
    d = 1.0 - 0.5 * (s + s.T) / hi
    return [ct.index[i] for i in _leaf_order(d, method)]


def resolve_pair_order(
    ct: pd.DataFrame,
    left_order: Optional[Sequence] = None,
    right_order: Optional[Sequence] = None,
    default_left: Optional[Sequence] = None,
    default_right: Optional[Sequence] = None,
) -> tuple[list, list]:
    """Work out both axis orders from what the caller pinned.

    ``left_order``/``right_order`` may each be an explicit sequence, None
    (keep the default order passed in), or one of the strings:

    ``"auto"``
        Both sides ``"auto"`` orders them jointly — by ``hclust_symmetric``
        when the two axes carry the **same label set**, since one order then
        belongs on both, and by SVD otherwise. One side ``"auto"`` makes that
        side follow the other by barycentre.
    ``"svd"``
        Force the singular-vector order even on a square table.
    ``"hclust"``
        Average-linkage on that axis's own profiles.
    ``"hclust_symmetric"``
        One clustered order on both axes; requires matching label sets.
    """
    _STRATEGIES = {"auto", "svd", "hclust", "hclust_symmetric"}
    left_str = left_order if isinstance(left_order, str) else None
    right_str = right_order if isinstance(right_order, str) else None
    for name, val in (("left_order", left_str), ("right_order", right_str)):
        if val is not None and val not in _STRATEGIES:
            raise ValueError(
                f"{name}={val!r} is not a known strategy; use one of "
                f"{sorted(_STRATEGIES)}, an explicit sequence, or None.")

    left_auto = left_str == "auto"
    right_auto = right_str == "auto"

    left = list(default_left) if left_order is None or left_str else list(left_order)
    right = list(default_right) if right_order is None or right_str else list(right_order)

    # Explicit strategies first — they say what to do without inference.
    if left_str == "hclust_symmetric" or right_str == "hclust_symmetric":
        order = hclust_symmetric_order(ct)
        return ([r for r in order if r in set(left)],
                [c for c in order if c in set(right)])
    if left_str == "hclust" and right_str == "hclust":
        rows, cols = hclust_order(ct)
        return ([r for r in rows if r in set(left)],
                [c for c in cols if c in set(right)])
    if left_str == "svd" and right_str == "svd":
        rows, cols = svd_order(ct)
        return ([r for r in rows if r in set(left)],
                [c for c in cols if c in set(right)])
    if left_str in ("hclust", "svd") and right_str is None:
        rows = (hclust_order(ct)[0] if left_str == "hclust" else svd_order(ct)[0])
        return [r for r in rows if r in set(left)], right
    if right_str in ("hclust", "svd") and left_str is None:
        cols = (hclust_order(ct)[1] if right_str == "hclust" else svd_order(ct)[1])
        return left, [c for c in cols if c in set(right)]

    if left_auto and right_auto:
        # A square table over one label set gets one order on both axes:
        # two independent orders would move a real diagonal off the diagonal.
        if set(ct.index) == set(ct.columns) and ct.shape[0] == ct.shape[1]:
            order = hclust_symmetric_order(ct)
            return ([r for r in order if r in set(left)],
                    [c for c in order if c in set(right)])
        rows, cols = svd_order(ct)
        return ([r for r in rows if r in set(left)],
                [c for c in cols if c in set(right)])
    if right_auto:
        ordered = barycentre_order(ct, [l for l in left if l in ct.index], axis=0)
        return left, [c for c in ordered if c in set(right)]
    if left_auto:
        ordered = barycentre_order(ct, [r for r in right if r in ct.columns], axis=1)
        return [r for r in ordered if r in set(left)], right
    return left, right
