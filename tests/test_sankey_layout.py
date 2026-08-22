"""The Sankey diagram must fill its axes, not sit in the bottom third.

`_compute_positions` fits the bars *and* the gaps between them into [0, 1]
(`available = 1 - total_gap`). The y-limit used to add `gap * n_categories` on
top of that, padding for space the layout had already reserved. With 27
categories and the default gap that put the top of the axis at 1.86, so nearly
half the published figure was blank.

The padding has to be constant. These tests pin that: the drawn content spans
the axis regardless of how many categories there are.
"""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
ad = pytest.importorskip("anndata")
piaso = pytest.importorskip("piaso")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _toy(n_left=5, n_right=27, n=2000):
    rng = np.random.default_rng(0)
    a = ad.AnnData(np.zeros((n, 2), dtype="float32"))
    a.obs["left"] = pd.Categorical([str(i % n_left) for i in range(n)])
    a.obs["right"] = pd.Categorical(
        [f"R{v}" for v in rng.integers(0, n_right, n)])
    return a


def _fig(adata):
    fig, ax = piaso.pl.sankey(adata, left="left", right="right",
                              show=False, return_fig=True)
    return fig, ax


@pytest.mark.parametrize("n_right", [3, 15, 27, 40])
def test_the_content_fills_the_axis_at_any_category_count(n_right):
    fig, ax = _fig(_toy(n_right=n_right))
    lo, hi = ax.get_ylim()
    # The bars span [0, 1] by construction; the axis should hug that with a
    # small constant margin rather than growing with the category count.
    assert hi - lo < 1.3, (
        f"ylim spans {hi - lo:.2f} for {n_right} categories — the axis is "
        "padded for space the layout already reserved")
    assert lo < 0 <= 1 < hi, "the [0, 1] band the bars occupy must be inside"
    plt.close(fig)


def test_the_bars_actually_reach_the_top_of_their_band():
    """Guards the other direction: content must not stop short of 1.0."""
    a = _toy(n_right=27)
    fig, ax = _fig(a)
    # ax.patches holds the node bars (Rectangle) and the flow ribbons
    # (PathPatch); only the bars carry the band the layout reserved.
    from matplotlib.patches import Rectangle
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert bars, "no node bars were drawn"
    assert max(b.get_y() + b.get_height() for b in bars) == pytest.approx(
        1.0, abs=0.02)
    assert min(b.get_y() for b in bars) == pytest.approx(0.0, abs=0.02)
    plt.close(fig)


def test_a_lopsided_diagram_still_fills_the_axis():
    """Left and right have different counts, so different per-side gap totals."""
    fig, ax = _fig(_toy(n_left=2, n_right=30))
    lo, hi = ax.get_ylim()
    assert hi - lo < 1.3
    plt.close(fig)
