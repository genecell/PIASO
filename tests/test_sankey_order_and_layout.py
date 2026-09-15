"""A Sankey must agree with the legend beside it, and take an order.

Three things went wrong together and only show up in combination:

* ``astype(str)`` ran before the order was resolved, so the column's own
  categorical order was thrown away and every Sankey fell through to a natural
  sort -- which is not what the UMAP beside it shows;
* the caller had no way to say what order they wanted;
* the rail stacked from ``t=0`` while matplotlib's y grows upward, so the first
  category was drawn at the *bottom*. A Sankey read 5,4,3,2,1,0 downward while
  the legend read 0,1,2 -- the same labels, opposite ends.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def labelled():
    import anndata as ad

    rs = np.random.RandomState(0)
    n = 400
    leiden = pd.Categorical(
        [str(i) for i in rs.randint(0, 12, n)],
        categories=[str(i) for i in range(12)], ordered=True)
    return ad.AnnData(
        X=np.zeros((n, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"leiden": leiden,
             "CellTypes": rs.choice(["Astro", "ExN", "InN", "Oligo"], n)},
            index=[f"c{i}" for i in range(n)]))


def _fig(out):
    return out[0] if isinstance(out, tuple) else out


def _rail_top_down(fig, wanted):
    items = [(t.get_position()[1], t.get_text()) for t in fig.axes[0].texts
             if t.get_text() in wanted]
    return [lab for _, lab in sorted(items, reverse=True)]


def test_first_category_is_at_the_top(labelled):
    """Top to bottom must follow the resolved order, not reverse it."""
    import piaso

    fig = _fig(piaso.pl.sankey(labelled, left="leiden", right="CellTypes",
                               show=False, return_fig=True))
    got = _rail_top_down(fig, {str(i) for i in range(12)})
    assert got == [str(i) for i in range(12)], (
        f"rail reads {got[:4]}... downward; the legend reads 0, 1, 2, 3")


def test_categorical_order_survives_the_string_cast(labelled):
    """The column's own order is the default, not a natural sort of strings."""
    import piaso

    # reverse the categorical order; the Sankey must follow it
    labelled.obs["leiden"] = labelled.obs["leiden"].cat.reorder_categories(
        [str(i) for i in reversed(range(12))])
    fig = _fig(piaso.pl.sankey(labelled, left="leiden", right="CellTypes",
                               show=False, return_fig=True))
    got = _rail_top_down(fig, {str(i) for i in range(12)})
    assert got == [str(i) for i in reversed(range(12))], (
        f"reordered categories were ignored; got {got[:4]}...")


def test_explicit_order_wins(labelled):
    import piaso

    order = ["Oligo", "InN", "ExN", "Astro"]
    fig = _fig(piaso.pl.sankey(labelled, left="leiden", right="CellTypes",
                               right_order=order, show=False, return_fig=True))
    assert _rail_top_down(fig, set(order)) == order


def test_sankey_and_embedding_agree(labelled):
    """The two must resolve the same labels the same way."""
    import piaso
    from piaso.plotting._group_order import resolve_group_order

    fig = _fig(piaso.pl.sankey(labelled, left="leiden", right="CellTypes",
                               show=False, return_fig=True))
    got = _rail_top_down(fig, {str(i) for i in range(12)})
    assert got == resolve_group_order(labelled.obs["leiden"])


@pytest.mark.parametrize("layout", ["horizontal", "vertical", "circular"])
def test_every_layout_draws_something(labelled, layout):
    import piaso

    fig = _fig(piaso.pl.sankey(labelled, left="leiden", right="CellTypes",
                               layout=layout, show=False, return_fig=True))
    ax = fig.axes[0]
    assert ax.patches, f"{layout}: nothing drawn"
    labels = {t.get_text() for t in ax.texts}
    assert "Oligo" in labels, f"{layout}: category labels missing"


def test_unknown_layout_is_rejected(labelled):
    import piaso

    with pytest.raises(ValueError, match="layout"):
        piaso.pl.sankey(labelled, left="leiden", right="CellTypes",
                        layout="spiral", show=False)
