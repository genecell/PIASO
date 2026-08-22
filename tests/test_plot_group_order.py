"""Every grouped plot must order groups the way plotEmbedding does.

Each of these used ``sorted(set(str(g)))``, a lexicographic sort, so Leiden
clusters came out 0, 1, 10, 11, 2 while the UMAP legend used the categorical
order and said 0, 1, 2, 3. The same figure then disagreed with itself about
which cluster was which colour.
"""
import numpy as np
import pandas as pd
import pytest

from piaso.plotting._group_order import natural_key, resolve_group_order


def test_natural_sort_puts_2_before_10():
    got = sorted(["10", "2", "1", "20", "3"], key=natural_key)
    assert got == ["1", "2", "3", "10", "20"]
    assert sorted(["c10", "c2"], key=natural_key) == ["c2", "c10"]


def test_categorical_order_is_respected():
    cats = [str(i) for i in range(12)]
    s = pd.Series(pd.Categorical(["11", "0", "5"], categories=cats))
    # present-only, but in the categorical order, not lexicographic
    assert resolve_group_order(s) == ["0", "5", "11"]


def test_plain_strings_fall_back_to_natural_sort():
    s = pd.Series(["10", "2", "0"])
    assert resolve_group_order(s) == ["0", "2", "10"]


def test_explicit_order_wins_and_extras_are_still_drawn():
    s = pd.Series(["a", "b", "c"])
    assert resolve_group_order(s, categories_order=["c", "a"]) == ["c", "a", "b"]


def test_unused_categories_are_dropped():
    s = pd.Series(pd.Categorical(["a"], categories=["a", "b", "c"]))
    assert resolve_group_order(s) == ["a"]
    assert resolve_group_order(s, present_only=False) == ["a", "b", "c"]


@pytest.mark.parametrize("module,attr", [
    ("piaso.plotting._plotByCluster", "resolve_group_order"),
    ("piaso.plotting._plotDotplot", "resolve_group_order"),
    ("piaso.plotting._plotHeatmap", "resolve_group_order"),
    ("piaso.plotting._plotDendrogram", "resolve_group_order"),
    ("piaso.plotting._plotSankey", "resolve_group_order"),
])
def test_every_grouped_plot_uses_the_shared_helper(module, attr):
    """A new lexicographic sort in one of these would silently re-diverge."""
    import importlib
    assert hasattr(importlib.import_module(module), attr)


def test_violin_group_order_matches_the_categorical(tmp_path):
    """End to end: the violin's x tick labels, in order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    anndata = pytest.importorskip("anndata")
    import piaso

    rng = np.random.default_rng(0)
    n = 400
    lab = [str(i) for i in rng.integers(0, 12, n)]
    a = anndata.AnnData(X=rng.random((n, 3)).astype(np.float32))
    a.var_names = ["g1", "g2", "g3"]
    a.obs["leiden"] = pd.Categorical(lab, categories=[str(i) for i in range(12)])

    _, axes = piaso.pl.violin(a, ["g1"], groupby="leiden", show=False,
                              return_fig=True)
    ax = axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [str(i) for i in range(12)], labels
    plt.close("all")
