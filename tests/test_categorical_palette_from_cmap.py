"""A colormap NAME should work as `palette=` for an ordered categorical.

Ages, timepoints and stages have a direction, and a sequential ramp says so.
`palette="Spectral_r"` used to be accepted and silently ignored, which is
worse than refusing it: the figure came back in the default categorical
colours and nothing said why.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso


def _adata(cats=("E15.5", "E18", "P0", "P14", "P56"), n=200):
    import anndata as ad
    rs = np.random.RandomState(0)
    a = ad.AnnData(X=sp.csr_matrix(rs.poisson(1.0, (n, 5)).astype(np.float32)))
    a.obs["age"] = pd.Categorical(rs.choice(cats, n), categories=list(cats))
    a.obsm["X_umap"] = rs.rand(n, 2)
    return a


def _legend_colors(fig):
    """Handles may be Line2D or a collection depending on how the legend was
    built, so read whichever colour accessor exists."""
    leg = fig.legends[0] if fig.legends else fig.axes[0].get_legend()
    out = []
    for h in leg.legend_handles:
        for attr in ("get_facecolor", "get_markerfacecolor", "get_color"):
            fn = getattr(h, attr, None)
            if fn is None:
                continue
            c = fn()
            c = np.ravel(np.asarray(matplotlib.colors.to_rgba(c)
                                    if isinstance(c, str) else c))
            if c.size >= 3:
                out.append(c[:3])
                break
    return out


def test_cmap_name_samples_that_colormap_across_the_categories():
    """The colours must BE the colormap, sampled in category order.

    An earlier version of this test asserted the two ends were the most
    different pair, which is false for a diverging map like Spectral: both
    ends are dark and saturated, so a mid-range step can exceed them. The
    real contract is exactness against the colormap.
    """
    from piaso.plotting._plotEmbedding import _resolve_categorical_style

    cats = ["E15.5", "E18", "P0", "P14", "P56"]
    a = _adata(cats=cats)
    palette, order = _resolve_categorical_style(a, "age", "Spectral_r")

    assert order == cats
    cm = matplotlib.colormaps.get_cmap("Spectral_r")
    expected = {c: matplotlib.colors.to_hex(cm(i / (len(cats) - 1)))
                for i, c in enumerate(cats)}
    assert palette == expected
    assert len(set(palette.values())) == len(cats)   # no category shares a colour


def test_default_is_unchanged_without_a_palette():
    a = _adata()
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="age", basis="X_umap", ax=ax, show=False)
    fig.canvas.draw()
    assert len(_legend_colors(fig)) == 5
    plt.close(fig)


def test_an_unknown_colormap_name_does_not_break_the_plot():
    a = _adata()
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="age", basis="X_umap",
                           palette="not_a_colormap", ax=ax, show=False)
    fig.canvas.draw()
    assert len(_legend_colors(fig)) == 5      # fell back, still drew
    plt.close(fig)


def test_an_explicit_list_still_wins():
    a = _adata()
    mine = ["#000000", "#111111", "#222222", "#333333", "#444444"]
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="age", basis="X_umap", palette=mine,
                           ax=ax, show=False)
    fig.canvas.draw()
    cols = [matplotlib.colors.to_hex(c) for c in _legend_colors(fig)]
    assert cols == mine
    plt.close(fig)
