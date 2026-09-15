"""Hierarchical orders for a confusion matrix, and one order for a square one.

`svd_order` summarises the table by its first singular vector, which stops
being a useful summary when there is no single dominant direction. `hclust`
groups categories that behave alike; `hclust_symmetric` is for a table over
ONE label set, where two independent orders would move a real diagonal off
the diagonal.
"""

import numpy as np
import pandas as pd
import pytest

from piaso.plotting._seriation import (
    hclust_order,
    hclust_symmetric_order,
    resolve_pair_order,
    svd_order,
)


def _blocks(labels, noise=0.0, seed=0):
    """A square matrix with a strong diagonal, rows/cols shuffled."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    m = np.eye(n) * 10.0 + rng.random((n, n)) * noise
    return pd.DataFrame(m, index=labels, columns=labels)


L = ["a", "b", "c", "d", "e", "f"]


# ------------------------------------------------------------- symmetric --


def test_symmetric_order_keeps_the_diagonal_on_the_diagonal():
    ct = _blocks(L, noise=0.5)
    shuffled = ct.iloc[[3, 0, 5, 1, 4, 2], [3, 0, 5, 1, 4, 2]]
    order = hclust_symmetric_order(shuffled)
    got = shuffled.loc[order, order].values
    assert (np.diag(got) == got.max(axis=1)).all()


def test_symmetric_order_returns_one_order_used_on_both_axes():
    ct = _blocks(L, noise=0.3)
    rows, cols = resolve_pair_order(ct, "hclust_symmetric", "hclust_symmetric",
                                    list(ct.index), list(ct.columns))
    assert rows == cols


def test_symmetric_order_reorders_columns_to_match_when_the_sets_agree():
    ct = _blocks(L, noise=0.2)
    scrambled = ct[list(reversed(L))]          # same set, different column order
    assert len(hclust_symmetric_order(scrambled)) == len(L)


def test_symmetric_order_refuses_mismatched_label_sets():
    ct = pd.DataFrame(np.ones((3, 2)), index=["a", "b", "c"], columns=["x", "y"])
    with pytest.raises(ValueError, match="same labels on both axes"):
        hclust_symmetric_order(ct)


def test_tiny_tables_are_returned_untouched():
    ct = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    assert hclust_symmetric_order(ct) == ["a", "b"]


# ----------------------------------------------------------- independent --


def test_hclust_groups_columns_that_behave_alike():
    # x1/x2 are duplicates, and so are y1/y2: each pair must end up adjacent
    rows = [f"r{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    a, b = rng.random(8), rng.random(8)
    ct = pd.DataFrame({"x1": a, "y1": b, "x2": a, "y2": b}, index=rows)
    _, cols = hclust_order(ct)
    assert abs(cols.index("x1") - cols.index("x2")) == 1
    assert abs(cols.index("y1") - cols.index("y2")) == 1


def test_hclust_handles_a_constant_row_without_nan():
    ct = pd.DataFrame(np.vstack([np.eye(4) * 5, np.ones((1, 4))]),
                      index=list("abcde"), columns=list("wxyz"))
    rows, cols = hclust_order(ct)
    assert sorted(rows) == list("abcde") and sorted(cols) == list("wxyz")


def test_rectangular_tables_are_ordered_per_axis():
    rng = np.random.default_rng(1)
    ct = pd.DataFrame(rng.random((9, 4)),
                      index=[f"c{i}" for i in range(9)], columns=list("wxyz"))
    rows, cols = hclust_order(ct)
    assert len(rows) == 9 and len(cols) == 4


# ------------------------------------------------------------- dispatch --


def test_auto_prefers_the_symmetric_order_when_the_labels_match():
    ct = _blocks(L, noise=0.4)
    rows, cols = resolve_pair_order(ct, "auto", "auto", list(ct.index), list(ct.columns))
    assert rows == cols == hclust_symmetric_order(ct)


def test_auto_still_uses_svd_when_the_labels_differ():
    rng = np.random.default_rng(2)
    ct = pd.DataFrame(rng.random((7, 4)),
                      index=[f"c{i}" for i in range(7)], columns=list("wxyz"))
    rows, cols = resolve_pair_order(ct, "auto", "auto", list(ct.index), list(ct.columns))
    want_r, want_c = svd_order(ct)
    assert rows == want_r and cols == want_c


def test_svd_can_be_forced_on_a_square_table():
    ct = _blocks(L, noise=0.4)
    rows, cols = resolve_pair_order(ct, "svd", "svd", list(ct.index), list(ct.columns))
    want_r, want_c = svd_order(ct)
    assert rows == want_r and cols == want_c


def test_one_side_hclust_leaves_the_other_alone():
    rng = np.random.default_rng(3)
    ct = pd.DataFrame(rng.random((6, 4)),
                      index=[f"c{i}" for i in range(6)], columns=list("wxyz"))
    rows, cols = resolve_pair_order(ct, "hclust", None, list(ct.index), list(ct.columns))
    assert cols == list(ct.columns) and sorted(rows) == sorted(ct.index)


def test_an_unknown_strategy_name_raises_rather_than_being_ignored():
    ct = _blocks(L)
    with pytest.raises(ValueError, match="not a known strategy"):
        resolve_pair_order(ct, "dendrogram", None, list(ct.index), list(ct.columns))


def test_explicit_sequences_are_still_honoured():
    ct = _blocks(L)
    want = list(reversed(L))
    rows, cols = resolve_pair_order(ct, want, want, list(ct.index), list(ct.columns))
    assert rows == want and cols == want
