"""The Sankey layout has to survive a lot of categories.

Two defects lived in the same arithmetic. The axis limit added
``gap * n_categories`` on top of a layout that already fitted into [0, 1], so
with 30 categories nearly half the canvas came out blank -- that one is fixed
and covered here. The second is that the gaps themselves are unbounded:
``gap * (n - 1)`` at the 0.03 default reaches 1.0 at 35 categories, leaving
negative height for every bar.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import piaso
from piaso.plotting._plotSankey import _effective_gap


def _frame(n_left: int, n_right: int, n: int = 2000):
    obs = pd.DataFrame(
        {"l": [f"L{i % n_left}" for i in range(n)],
         "r": [f"R{i % n_right}" for i in range(n)]},
        index=[str(i) for i in range(n)])
    return ad.AnnData(np.zeros((n, 2), dtype="float32"), obs=obs)


@pytest.mark.parametrize("n", [2, 5, 10])
def test_small_category_counts_are_untouched(n):
    """The cap must not change layouts that were already fine."""
    assert _effective_gap(n, 0.03) == 0.03


@pytest.mark.parametrize("n", [35, 60, 200])
def test_gaps_never_consume_the_whole_axis(n):
    """Without the cap, `available` is negative here and bars invert."""
    eff = _effective_gap(n, 0.03)
    assert 1.0 - eff * (n - 1) >= 0.5 - 1e-9


def test_forty_categories_render_with_positive_bars():
    fig, ax = piaso.pl.sankey(_frame(40, 38), left="l", right="r",
                              show=False, return_fig=True)
    bars = [p for p in ax.patches if hasattr(p, "get_height")]
    heights = [p.get_height() for p in bars]
    assert bars, "no bars drawn"
    assert min(heights) > 0, f"non-positive bar height: {min(heights)}"
    tops = [p.get_y() + p.get_height() for p in bars]
    assert max(tops) <= 1.0 + 1e-6
    assert min(p.get_y() for p in bars) >= -1e-6


def test_the_axis_is_not_padded_per_category():
    """The regression that left half the figure blank.

    y-limits must be the constant padding, not something that grows with the
    number of categories.
    """
    _f1, ax_small = piaso.pl.sankey(_frame(3, 3), left="l", right="r",
                                    show=False, return_fig=True)
    _f2, ax_big = piaso.pl.sankey(_frame(30, 28), left="l", right="r",
                                  show=False, return_fig=True)
    assert ax_small.get_ylim() == pytest.approx(ax_big.get_ylim())
    assert ax_big.get_ylim()[1] < 1.2, ax_big.get_ylim()
