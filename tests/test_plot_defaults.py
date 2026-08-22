"""Defaults that were changed deliberately, pinned so they are not drifted back."""
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
anndata = pytest.importorskip("anndata")
import piaso  # noqa: E402


@pytest.fixture
def adata():
    rng = np.random.default_rng(1)
    n = 600
    a = anndata.AnnData(X=np.abs(rng.random((n, 4))).astype(np.float32))
    a.var_names = ["g1", "g2", "g3", "g4"]
    a.obsm["X_umap"] = rng.random((n, 2))
    a.obs["leiden"] = pd.Categorical([str(i) for i in rng.integers(0, 5, n)],
                                     categories=[str(i) for i in range(5)])
    a.obs["sample"] = pd.Categorical(rng.choice(["s1", "s2"], n))
    a.obs["qc"] = rng.random(n)
    return a


def test_violin_puts_ungrouped_features_two_per_row(adata):
    fig, _ = piaso.pl.violin(adata, ["g1", "g2", "g3", "g4"], show=False,
                             return_fig=True)
    # 4 features, 2 per row -> 2 rows of 2, so the leftmost axes share an x
    xs = sorted({round(ax.get_position().x0, 3) for ax in fig.axes})
    assert len(xs) == 2, f"expected a 2-column grid, got columns at {xs}"
    plt.close("all")


def test_violin_grouped_stays_one_per_row(adata):
    fig, _ = piaso.pl.violin(adata, ["g1", "g2"], groupby="leiden", show=False,
                             return_fig=True)
    xs = {round(ax.get_position().x0, 3) for ax in fig.axes}
    assert len(xs) == 1
    plt.close("all")


def test_violin_ncol_is_overridable(adata):
    fig, _ = piaso.pl.violin(adata, ["g1", "g2", "g3"], ncol=3, show=False,
                             return_fig=True)
    xs = {round(ax.get_position().x0, 3) for ax in fig.axes}
    assert len(xs) == 3
    plt.close("all")


def test_embedding_point_size_default_is_the_smaller_one(adata):
    """22000/n capped at 3, not the old 30000/n capped at 4."""
    n = adata.n_obs
    assert max(0.1, min(3, 22000 / n)) == 3          # 600 cells -> capped
    piaso.pl.embedding(adata, basis="X_umap", color="leiden", show=False)
    ax = [a for a in plt.gcf().axes if a.collections][0]
    sizes = np.concatenate([c.get_sizes() for c in ax.collections if len(c.get_sizes())])
    assert sizes.max() <= 3.0 + 1e-9, sizes.max()
    plt.close("all")


def test_stacked_barplot_default_keeps_category_order(adata):
    fig, ax = piaso.pl.stackedBarplot(adata, groupby="leiden", splitby="sample",
                                      show=False, return_fig=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [str(i) for i in range(5)], labels
    plt.close("all")


def test_stacked_barplot_sort_groups_ranks_by_proportion(adata):
    # make one group dominant so the ranking is unambiguous
    lab = np.array(["0"] * 500 + ["1"] * 60 + ["2"] * 40)
    adata.obs["leiden"] = pd.Categorical(lab, categories=["0", "1", "2"])
    fig, ax = piaso.pl.stackedBarplot(adata, groupby="leiden", splitby="sample",
                                      sort_groups=True, show=False, return_fig=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[0] == "0", labels
    plt.close("all")


def test_confusion_matrix_cells_are_square_by_default(adata):
    piaso.pl.plotConfusionMatrix(adata, "leiden", "sample")
    ax = plt.gcf().axes[0]
    assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal", ax.get_aspect()
    plt.close("all")
