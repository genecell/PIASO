"""plotEmbedding's grid forwards every argument; plots keep the caller's figure.

Regressions for the audit of 2026-10-27 (review I2, findings 13, 14 and 20):

* a list ``color`` drew each panel from a hand-written subset of the
  arguments, so ``cell_mask`` (and image*, linewidths, edgecolors, na_label)
  fell out of it: a mask of 412 of 600 cells still drew all 600;
* ``show`` defaulted to True inside a caller's axes, functions given ``ax=``
  closed the caller's figure, and ``plt.tight_layout()`` acted on ``gcf()``;
* ``basis='X_umap'`` on a store holding RNA_umap and ATAC_umap took RNA
  without a word.
"""
import warnings

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso


def _adata(n=600, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        "ct": pd.Categorical(rng.choice(["A", "B", "C"], n), categories=["A", "B", "C"]),
        "score": rng.normal(size=n),
    }, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1.0, (n, 4)).astype(np.float32)), obs=obs,
                   var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    return a


def _n_points(ax):
    return sum(len(c.get_offsets()) for c in ax.collections)


def test_grid_panels_receive_cell_mask_and_styling():
    a = _adata()
    mask = np.zeros(a.n_obs, dtype=bool)
    mask[:412] = True
    plt.close("all")
    fig, axs = piaso.pl.embedding(a, color=["ct", "score"], cell_mask=mask,
                                  linewidths=0.7, edgecolors="black",
                                  show=False, return_fig=True)
    for ax in axs[:2]:
        assert _n_points(ax) == 412
        coll = max(ax.collections, key=lambda c: len(c.get_offsets()))
        assert np.allclose(coll.get_linewidths(), 0.7)
    plt.close(fig)


def test_grid_panels_receive_na_label_and_sort_order():
    a = _adata()
    plt.close("all")
    fig, axs = piaso.pl.embedding(a, color=["ct", "ct"], groups=["A"], na_label="rest",
                                  sort_order="size", show=False, return_fig=True)
    for ax in axs[:2]:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels[-1] == "rest"
    plt.close(fig)


# --- figure life cycle ---------------------------------------------------------

def _callers():
    """Each public plot, called on toy data into a given axes."""
    a = _adata()
    a.obs["batch"] = np.where(np.arange(a.n_obs) % 2, "s1", "s2")
    return a, [
        ("embedding", lambda ax: piaso.pl.embedding(a, color="ct", ax=ax)),
        ("dotplot", lambda ax: piaso.pl.dotplot(a, ["g0", "g1"], groupby="ct", ax=ax)),
        ("heatmap", lambda ax: piaso.pl.heatmap(a, ["g0", "g1"], groupby="ct", ax=ax)),
        ("scatter", lambda ax: piaso.pl.scatter(a, "g0", "g1", color="ct", ax=ax)),
        ("sankey", lambda ax: piaso.pl.sankey(a, "ct", "batch", ax=ax)),
        ("stacked_barplot", lambda ax: piaso.pl.stacked_barplot(a, groupby="ct",
                                                                splitby="batch", ax=ax)),
        ("plot_dendrogram", lambda ax: piaso.pl.plot_dendrogram(a, groupby="ct",
                                                                use_rep=None, ax=ax)),
    ]


@pytest.mark.parametrize("name", [n for n, _ in _callers()[1]])
def test_a_callers_figure_is_neither_shown_nor_closed(name, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
    _, callers = _callers()
    call = dict(callers)[name]
    plt.close("all")
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call(ax)
    assert shown == [], f"{name} called plt.show() inside the caller's axes"
    assert plt.fignum_exists(fig.number), f"{name} closed the caller's figure"
    plt.close("all")


def test_tight_layout_acts_on_the_figure_drawn_not_the_current_one():
    a = _adata()
    plt.close("all")
    a.obs["batch"] = np.where(np.arange(a.n_obs) % 2, "s1", "s2")
    target, ax = plt.subplots()
    # the current figure, not the target, with labels tight_layout would move
    other, oax = plt.subplots()
    oax.set_ylabel("a long label\nover two lines", fontsize=20)
    before = (other.subplotpars.left, other.subplotpars.bottom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piaso.pl.stacked_barplot(a, groupby="ct", splitby="batch", ax=ax, show=False)
    assert (other.subplotpars.left, other.subplotpars.bottom) == before
    plt.close("all")


def test_not_shown_returns_the_figure_and_keeps_it_for_plt_savefig(tmp_path):
    a = _adata()
    plt.close("all")
    out = piaso.pl.embedding(a, color="ct", show=False)
    assert isinstance(out, plt.Figure)
    assert plt.gcf() is out                  # plt.savefig() after show=False still works
    plt.close("all")


def test_saved_and_not_shown_figures_do_not_accumulate(tmp_path):
    a = _adata()
    plt.close("all")
    for i in range(3):
        piaso.pl.dotplot(a, ["g0"], groupby="ct", show=False,
                         save=str(tmp_path / f"d{i}.png"))
    assert plt.get_fignums() == []
    plt.close("all")


def test_saved_pdf_embeds_truetype_fonts(tmp_path):
    a = _adata()
    path = tmp_path / "e.pdf"
    plt.close("all")
    with matplotlib.rc_context({"pdf.fonttype": 3}):      # matplotlib's default
        piaso.pl.dotplot(a, ["g0"], groupby="ct", show=False, save=str(path))
    data = path.read_bytes()
    assert b"/FontType 3" not in data and b"/Subtype /Type3" not in data
    assert b"TrueType" in data or b"FontFile2" in data


# --- basis on a multiome store -------------------------------------------------

def test_ambiguous_basis_warns_and_modality_chooses(tmp_path):
    import cytome
    from piaso.plotting._plotEmbedding import _resolve_cytome_basis
    p = str(tmp_path / "m.cytome")
    ds = cytome.create(p)
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"b{i}" for i in range(20)]}))
    rng = np.random.default_rng(0)
    ds.add_embedding("RNA_umap", rng.normal(size=(20, 2)))
    ds.add_embedding("ATAC_umap", rng.normal(size=(20, 2)))
    ds.flush()
    with pytest.warns(UserWarning, match="ATAC_umap"):
        assert _resolve_cytome_basis(ds, "X_umap") == "RNA_umap"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_cytome_basis(ds, "X_umap", modality="ATAC") == "ATAC_umap"
        assert _resolve_cytome_basis(ds, "ATAC_umap") == "ATAC_umap"
    ds.close()


def test_vcenter_puts_the_map_centre_at_the_value():
    """Improvement 6 of review I2: a diverging map centred on a value."""
    a = _adata()
    a.obs["signed"] = np.linspace(-1.0, 3.0, a.n_obs)
    plt.close("all")
    fig, ax = piaso.pl.embedding(a, color="signed", vcenter=0.0, show=False,
                                 return_fig=True)
    sc = max(ax.collections, key=lambda c: len(c.get_offsets()))
    assert sc.norm(0.0) == pytest.approx(0.5)
    assert sc.norm(-1.0) == pytest.approx(0.0) and sc.norm(3.0) == pytest.approx(1.0)
    plt.close(fig)
    with pytest.raises(ValueError, match="vcenter"):
        piaso.pl.embedding(a, color="signed", vcenter=5.0, show=False)
    plt.close("all")


def test_grid_on_a_store_reads_its_genes_in_one_pass(tmp_path, monkeypatch):
    """Review I2 finding 15: each panel of a grid streamed the store's matrix
    for its own gene. The grid now reads them in one pass; the colours are
    the ones each panel draws alone."""
    import cytome
    import piaso.plotting._plotEmbedding as pe
    a = _adata()
    a.layers["counts"] = a.X.copy()
    path = str(tmp_path / "grid.cytome")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = cytome.from_anndata(a, output=path, counts_layer="counts")
        ds.close()
    genes = ["g0", "g1", "g2"]
    single = {}
    for g in genes:
        plt.close("all")
        fig, ax = piaso.pl.embedding(path, color=g, show=False, return_fig=True)
        sc = max(ax.collections, key=lambda c: len(c.get_offsets()))
        single[g] = np.sort(np.asarray(sc.get_array(), dtype=float))
        plt.close(fig)
    calls = []
    real = pe._resolve_cytome_feature_values
    monkeypatch.setattr(pe, "_resolve_cytome_feature_values",
                        lambda *x, **k: calls.append(1) or real(*x, **k))
    plt.close("all")
    fig, axs = piaso.pl.embedding(path, color=genes, show=False, return_fig=True)
    assert calls == []                              # no per-panel matrix read
    for g, ax in zip(genes, axs):
        sc = max(ax.collections, key=lambda c: len(c.get_offsets()))
        np.testing.assert_allclose(np.sort(np.asarray(sc.get_array(), dtype=float)),
                                   single[g], rtol=1e-6)
    plt.close("all")
