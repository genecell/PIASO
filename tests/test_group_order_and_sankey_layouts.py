"""Every figure of the same cells must agree on the order of their groups.

Leiden labels are strings, so a lexicographic sort puts cluster 10 between 1
and 2. The embedding legend did that while the Sankey beside it resolved order
through `resolve_group_order` and got 0, 1, 2 -- so two figures of the same
cells disagreed about which cluster was which colour. A cytome round-trip
stores the column as plain strings, so there is no categorical order left to
fall back on and the sort is what decides.

The Sankey had a second, quieter version of the same bug: it called
`.astype(str)` *before* resolving the order, which threw away the categorical
dtype, so the caller's own ordering could never be honoured.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clustered():
    anndata = pytest.importorskip("anndata")
    rs = np.random.RandomState(0)
    n = 400
    lab = [str(i) for i in rs.randint(0, 12, n)]
    other = [l if rs.rand() < 0.8 else str(rs.randint(0, 12)) for l in lab]
    return anndata.AnnData(
        X=np.zeros((n, 2), dtype=np.float32),
        obs=pd.DataFrame({"leiden": lab, "pred": other},
                         index=[f"c{i}" for i in range(n)]),
    )


def test_natural_order_puts_10_after_9():
    from piaso.plotting._group_order import resolve_group_order

    labels = pd.Series([str(i) for i in range(13)])
    order = resolve_group_order(labels)
    assert order[:4] == ["0", "1", "2", "3"], order
    assert order.index("10") > order.index("9"), (
        f"lexicographic sort leaked through: {order}")


def test_sankey_honours_an_explicit_order(clustered):
    """The caller's order is the one drawn, on each side independently."""
    import piaso

    want = [str(i) for i in range(11, -1, -1)]
    out = piaso.pl.sankey(clustered, left="leiden", right="pred",
                          left_order=want, show=False, return_fig=True)
    fig = out[0] if isinstance(out, tuple) else out
    drawn = [t.get_text() for t in fig.axes[0].texts
             if t.get_text().isdigit()]
    assert drawn[:3] == want[:3], (
        f"left_order was not honoured; drawn starts {drawn[:5]}")


def test_sankey_default_order_is_natural(clustered):
    """Without an explicit order, 10 still comes after 9."""
    import piaso

    out = piaso.pl.sankey(clustered, left="leiden", right="pred",
                          show=False, return_fig=True)
    fig = out[0] if isinstance(out, tuple) else out
    drawn = [t.get_text() for t in fig.axes[0].texts if t.get_text().isdigit()]
    left = drawn[:12]
    assert left.index("10") > left.index("9"), left


@pytest.mark.parametrize("layout", ["horizontal", "vertical", "circular"])
def test_every_layout_draws_the_same_flows(clustered, layout):
    import piaso

    out = piaso.pl.sankey(clustered, left="leiden", right="pred",
                          layout=layout, show=False, return_fig=True)
    fig = out[0] if isinstance(out, tuple) else out
    ax = fig.axes[0]
    assert len(ax.patches) > 12, f"{layout}: too few artists"
    labels = {t.get_text() for t in ax.texts if t.get_text().isdigit()}
    assert len(labels) == 12, f"{layout}: drew {len(labels)} category labels"


def test_unknown_layout_is_rejected(clustered):
    import piaso

    with pytest.raises(ValueError, match="layout="):
        piaso.pl.sankey(clustered, left="leiden", right="pred",
                        layout="spiral", show=False)
