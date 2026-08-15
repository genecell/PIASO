"""2026-06-04: plotEmbedding title (list -> per-panel) + scanpy-style groups=."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def adata():
    import anndata
    rng = np.random.RandomState(0)
    a = anndata.AnnData(rng.rand(60, 4).astype("float32"))
    a.var_names = ["GeneA", "GeneB", "GeneC", "GeneD"]
    a.obsm["X_umap"] = rng.rand(60, 2)
    a.obs["leiden"] = np.array(["0", "1", "2"] * 20)
    return a


def test_title_list_sets_per_panel_titles(adata):
    """The exact user call: list title matching list color -> one title/panel."""
    import piaso
    r = piaso.pl.umap(adata, color=["leiden", "leiden"], title=["a", "b"],
                      ncol=1, show=False, return_fig=True)
    fig = r[0] if isinstance(r, tuple) else r
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["a", "b"]
    plt.close("all")


def test_title_list_length_mismatch_raises(adata):
    import piaso
    with pytest.raises(ValueError, match="must match"):
        piaso.pl.umap(adata, color=["leiden", "leiden"], title=["only_one"],
                      show=False, return_fig=True)
    plt.close("all")


def test_title_scalar_is_suptitle_for_grid(adata):
    import piaso
    r = piaso.pl.umap(adata, color=["leiden", "GeneA"], title="GRID",
                      show=False, return_fig=True)
    fig = r[0] if isinstance(r, tuple) else r
    assert fig._suptitle is not None and fig._suptitle.get_text() == "GRID"
    plt.close("all")


def test_groups_greys_out_non_selected(adata):
    import piaso
    r = piaso.pl.plotEmbedding(adata, color="leiden", groups=["0"],
                               show=False, return_fig=True)
    ax = r[1] if isinstance(r, tuple) else plt.gca()
    labels = [c.get_label() for c in ax.collections]
    assert "_nolegend_" in labels          # grey background layer
    assert "0" in labels                    # selected group coloured
    assert "1" not in labels and "2" not in labels  # others greyed
    plt.close("all")


def test_groups_unknown_category_raises(adata):
    import piaso
    with pytest.raises(ValueError, match="not found among the categories"):
        piaso.pl.plotEmbedding(adata, color="leiden", groups=["nope"],
                               show=False, return_fig=True)
    plt.close("all")


def test_groups_single_str_and_na_color(adata):
    import piaso
    r = piaso.pl.plotEmbedding(adata, color="leiden", groups="0",
                               na_color="black", show=False, return_fig=True)
    ax = r[1] if isinstance(r, tuple) else plt.gca()
    assert "0" in [c.get_label() for c in ax.collections]
    plt.close("all")
