"""Two things a caller says that were being heard as something else.

Both reported from use, and both are cases where an argument was accepted,
partially honoured, and the rest of it dropped without a word:

* ``palette={"Neuron": ..., "Glia": ...}`` set the colours and lost the order,
  because the categories were re-sorted alphabetically afterwards. A dict
  written out by hand states a sequence; Python has preserved it since 3.7.
* ``legend_loc="none"`` suppressed a categorical legend but not a continuous
  colour bar — and for a continuous colour the colour bar *is* the legend.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def adata():
    import anndata as ad

    rng = np.random.default_rng(0)
    n = 180
    obs = pd.DataFrame(
        {"ct": pd.Categorical(rng.choice(["Glia", "Neuron", "VLMC"], n),
                              # deliberately NOT alphabetical, and not the
                              # palette order either: three orders in play.
                              categories=["VLMC", "Glia", "Neuron"]),
         "s": rng.choice(["x", "y"], n)},
        index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(np.abs(rng.normal(size=(n, 3))).astype(np.float32), obs=obs,
                   var=pd.DataFrame(index=["g1", "g2", "g3"]))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    return a


PALETTE = {"Neuron": "#ff0000", "VLMC": "#00ff00", "Glia": "#0000ff"}


def _drawn_order(ax):
    return [c.get_label() for c in ax.collections]


def _face(c):
    return tuple(np.round(c.get_facecolor()[0][:3], 3))


# ------------------------------------------------ a dict states an order ---


def test_a_dict_palette_sets_the_category_order(adata):
    import piaso

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, color="ct", basis="X_umap", palette=PALETTE,
                           ax=ax, show=False)
    assert _drawn_order(ax) == list(PALETTE), (
        f"the dict said {list(PALETTE)}, the plot drew {_drawn_order(ax)}")
    plt.close("all")


def test_the_colours_still_follow_the_dict(adata):
    """Order is new; colour was always right and must stay right."""
    import piaso

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, color="ct", basis="X_umap", palette=PALETTE,
                           ax=ax, show=False)
    got = {c.get_label(): _face(c) for c in ax.collections}
    assert got == {"Neuron": (1.0, 0.0, 0.0), "VLMC": (0.0, 1.0, 0.0),
                   "Glia": (0.0, 0.0, 1.0)}, got
    plt.close("all")


def test_the_dict_outranks_the_stored_categorical_order(adata):
    """Both are deliberate; the one typed at the call site is more specific."""
    import piaso

    assert list(adata.obs["ct"].cat.categories) == ["VLMC", "Glia", "Neuron"]
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, color="ct", basis="X_umap", palette=PALETTE,
                           ax=ax, show=False)
    assert _drawn_order(ax) == ["Neuron", "VLMC", "Glia"]
    plt.close("all")


def test_without_a_dict_the_stored_order_still_wins(adata):
    """The new rule must not cost the old one."""
    import piaso

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, color="ct", basis="X_umap", ax=ax, show=False)
    assert _drawn_order(ax) == ["VLMC", "Glia", "Neuron"]
    plt.close("all")


def test_a_partial_dict_leads_and_the_rest_follow(adata):
    """Naming two of three categories is a legitimate thing to do."""
    import piaso

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, color="ct", basis="X_umap",
                           palette={"Neuron": "#ff0000"}, ax=ax, show=False)
    order = _drawn_order(ax)
    assert order[0] == "Neuron", order
    assert sorted(order[1:]) == ["Glia", "VLMC"], order
    plt.close("all")


def test_a_list_palette_is_unaffected(adata):
    """A list carries colours only -- it cannot state an order."""
    import piaso

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, color="ct", basis="X_umap",
                           palette=["#111111", "#222222", "#333333"],
                           ax=ax, show=False)
    assert _drawn_order(ax) == ["VLMC", "Glia", "Neuron"]
    plt.close("all")


def test_the_split_honours_a_dict_order_too(adata):
    import piaso

    piaso.pl.plot_embeddings_split(adata, color="ct", splitby="s",
                                   basis="X_umap", palette=PALETTE,
                                   show_figure=False)
    panels = [a for a in plt.gcf().axes if a.collections]
    assert panels
    for ax in panels:
        drawn = [c.get_label() for c in ax.collections if c.get_label() in PALETTE]
        assert drawn == [c for c in PALETTE if c in drawn], (ax.get_title(), drawn)
    plt.close("all")


# --------------------------------------- 'none' means none, colour bar too --


def _colorbars(fig, main_axes):
    return [a for a in fig.axes if a not in main_axes]


@pytest.mark.parametrize("loc,want", [("right", 1), ("none", 0)])
def test_plot_embedding_colorbar_follows_legend_loc(adata, loc, want):
    import piaso

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, color="g1", basis="X_umap", legend_loc=loc,
                           ax=ax, show=False)
    assert len(_colorbars(fig, {ax})) == want
    plt.close("all")


@pytest.mark.parametrize("loc,want_any", [("right", True), ("none", False)])
def test_split_colorbar_follows_legend_loc(adata, loc, want_any):
    """The reported case: a continuous colour split into panels."""
    import piaso

    piaso.pl.plot_embeddings_split(adata, color="g1", splitby="s",
                                   basis="X_umap", legend_loc=loc,
                                   show_figure=False)
    bars = [a for a in plt.gcf().axes if a.get_label() == "<colorbar>"]
    assert bool(bars) is want_any, f"legend_loc={loc!r} drew {len(bars)} colorbars"
    plt.close("all")


def test_none_still_suppresses_the_categorical_legend(adata):
    import piaso

    piaso.pl.plot_embeddings_split(adata, color="ct", splitby="s",
                                   basis="X_umap", legend_loc="none",
                                   show_figure=False)
    fig = plt.gcf()
    assert not fig.legends
    assert not any(a.get_legend() for a in fig.axes)
    plt.close("all")
