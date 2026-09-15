"""A NaN in one group must not delete that group's violin.

matplotlib's violinplot does not raise on non-finite input: the gaussian KDE's
covariance comes out NaN and the body is drawn with no visible extent. So a
single missing measurement silently removed a whole group from the panel.

Found on a Xenium object where `nucleus_area` is NaN for ~1% of cells: of 27
clusters, the only one that rendered was the only one with no NaN.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import piaso


@pytest.fixture
def adata_with_gaps():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    n = 600
    obs = pd.DataFrame({
        "group": pd.Categorical(np.repeat(list("abcd"), n // 4)),
        "clean": rng.normal(10, 2, n),
        "gappy": rng.normal(10, 2, n),
    })
    # one NaN in every group but 'd'
    for g, idx in ((("a"), 5), (("b"), n // 4 + 5), (("c"), n // 2 + 5)):
        obs.loc[obs.index[idx], "gappy"] = np.nan
    X = rng.normal(size=(n, 3)).astype(np.float32)
    return ad.AnnData(X=X, obs=obs,
                      var=pd.DataFrame(index=["g0", "g1", "g2"]))


def _drawn_groups(ax):
    """Groups whose violin has a non-degenerate vertical extent."""
    out = 0
    for coll in ax.collections:
        paths = coll.get_paths()
        for pth in paths:
            v = pth.vertices
            if len(v) and np.ptp(v[:, 1]) > 1e-9:
                out += 1
    return out


def test_a_nan_does_not_remove_the_group(adata_with_gaps):
    fig, axes = piaso.pl.plotFeaturesViolin(
        adata_with_gaps, ["clean", "gappy"], groupby="group",
        show_figure=False, return_fig=True)
    clean_n = _drawn_groups(axes[0])
    gappy_n = _drawn_groups(axes[1])
    assert clean_n >= 4, f"the clean feature should draw 4 violins, drew {clean_n}"
    assert gappy_n >= 4, (
        f"a single NaN per group removed violins: {gappy_n} drawn where the "
        f"same data without NaN draws {clean_n}"
    )
    plt.close("all")


def test_an_all_nan_group_is_dropped_not_crashed(adata_with_gaps):
    """A group with nothing finite left has no violin to draw, and that is fine
    as long as the others still appear."""
    a = adata_with_gaps
    a.obs.loc[a.obs["group"] == "a", "gappy"] = np.nan
    fig, axes = piaso.pl.plotFeaturesViolin(
        a, ["gappy"], groupby="group", show_figure=False, return_fig=True)
    assert _drawn_groups(axes[0]) >= 3
    plt.close("all")
