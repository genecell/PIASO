"""With ``groups=``, the legend should describe the plot in front of you.

``plotEmbedding(groups=['A','B'])`` draws A and B in their palette colours and
everything else as one grey cloud. The legend was built from the full category
list, so on a forty-cluster column it listed forty labels in forty colours,
thirty-eight of which appeared nowhere on the figure -- while the one colour
that covered most of the points, the grey, had no entry at all.

It now lists the highlighted groups, then one entry for the grey, and only when
something was actually greyed out.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def adata():
    import anndata as ad
    rng = np.random.default_rng(0)
    n = 400
    cats = np.array(["A", "B", "C", "D", "E"])[rng.integers(0, 5, n)]
    a = ad.AnnData(
        X=rng.random((n, 3)).astype("float32"),
        obs=pd.DataFrame({"g": pd.Categorical(cats)},
                         index=[f"c{i}" for i in range(n)]),
    )
    a.obsm["X_umap"] = rng.random((n, 2))
    return a


def _legend(adata, **kw):
    import piaso
    fig, ax = plt.subplots()
    try:
        piaso.pl.plotEmbedding(adata, basis="X_umap", color="g", ax=ax,
                               show=False, **kw)
        leg = ax.get_legend()
        labels = [t.get_text() for t in leg.get_texts()]
        colors = [mcolors.to_hex(h.get_markerfacecolor()) for h in leg.legend_handles]
        return labels, colors
    finally:
        plt.close(fig)


def test_legend_lists_the_groups_then_the_grey(adata):
    labels, colors = _legend(adata, groups=["A", "B"])
    assert labels == ["A", "B", "other"]
    assert colors[-1] == mcolors.to_hex("lightgray")


def test_no_other_entry_when_nothing_is_greyed_out(adata):
    """Asking for every category greys out nothing, so there is no 'other'."""
    labels, _ = _legend(adata, groups=["A", "B", "C", "D", "E"])
    assert labels == ["A", "B", "C", "D", "E"]


def test_legend_unchanged_without_groups(adata):
    labels, _ = _legend(adata)
    assert labels == ["A", "B", "C", "D", "E"]


def test_highlighted_colours_still_come_from_the_full_category_list(adata):
    """A group must keep the colour it has on an unfiltered plot.

    This is the property that makes a highlighted panel comparable with the
    full one; rebuilding the legend from `plot_categories` must not quietly
    re-index the palette.
    """
    full_labels, full_colors = _legend(adata)
    full = dict(zip(full_labels, full_colors))
    for selection in (["A"], ["C"], ["B", "E"]):
        labels, colors = _legend(adata, groups=selection)
        assert labels[:-1] == selection
        for lab, col in zip(labels[:-1], colors[:-1]):
            assert col == full[lab], f"{lab} changed colour under groups={selection}"


def test_na_label_is_configurable(adata):
    labels, colors = _legend(adata, groups=["A"], na_label="rest",
                             na_color="#cccccc")
    assert labels == ["A", "rest"]
    assert colors[-1] == "#cccccc"


def test_ncol_follows_the_entries_actually_shown(adata):
    """The column count was computed from the full category list too.

    With many categories and a two-group highlight that produced a legend laid
    out for forty entries and holding three.
    """
    import anndata as ad
    import piaso
    rng = np.random.default_rng(1)
    n = 600
    cats = np.array([f"c{i}" for i in range(40)])[rng.integers(0, 40, n)]
    big = ad.AnnData(
        X=rng.random((n, 3)).astype("float32"),
        obs=pd.DataFrame({"g": pd.Categorical(cats)},
                         index=[f"x{i}" for i in range(n)]),
    )
    big.obsm["X_umap"] = rng.random((n, 2))

    fig, ax = plt.subplots()
    try:
        piaso.pl.plotEmbedding(big, basis="X_umap", color="g",
                               groups=["c0", "c1"], ax=ax, show=False)
        leg = ax.get_legend()
        assert len(leg.get_texts()) == 3
        assert leg._ncols == 1, f"3 entries laid out in {leg._ncols} columns"
    finally:
        plt.close(fig)
