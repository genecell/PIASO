"""Each category keeps its colour in every panel of plot_embeddings_split.

Without ``groups=`` each panel was handed its palette as a list in sorted
order, while the renderer indexes it in the column's category order. When
the two orders differed -- a categorical or stored order such as MGE-like
before CGE-like -- the colours landed on the wrong categories, silently.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import anndata as ad
import piaso

CATS = ["MGE-like", "CGE-like", "Astro"]        # deliberately not sorted


def _adata(n=600, drop_from_p7=None):
    rng = np.random.default_rng(0)
    cls = rng.choice(CATS, n)
    age = rng.choice(["P1", "P7"], n)
    if drop_from_p7 is not None:                 # a panel that lacks a category
        cls[(age == "P7") & (cls == drop_from_p7)] = CATS[0]
    a = ad.AnnData(np.zeros((n, 2), dtype=np.float32),
                   obs=pd.DataFrame({"cls": pd.Categorical(cls, categories=CATS), "age": age},
                                    index=[f"c{i}" for i in range(n)]))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    return a


def _colours():
    """{category: colour} of every axis with categorical points."""
    out = []
    for ax in plt.gcf().axes:
        d = {c.get_label(): mc.to_hex(c.get_facecolor()[0]) for c in ax.collections
             if c.get_label() and not c.get_label().startswith("_")}
        if d:
            out.append(d)
    plt.close("all")
    return out


def _single(data, **kw):
    piaso.pl.plotEmbedding(data, color="cls", basis="X_umap", show=False, **kw)
    return _colours()[0]


def _split(data, **kw):
    piaso.pl.plot_embeddings_split(data, color="cls", splitby="age", basis="X_umap",
                                   show=False, **kw)
    return _colours()


@pytest.mark.parametrize("groups", [None, CATS])
def test_categorical_order(groups):
    a = _adata()
    ref = _single(a)
    for panel in _split(a, groups=groups):
        for cat, col in panel.items():
            assert col == ref[cat], (cat, panel, ref)


def test_a_panel_without_one_category():
    a = _adata(drop_from_p7="CGE-like")
    ref = _single(a)
    panels = _split(a)
    assert "CGE-like" not in panels[1]
    for panel in panels:
        for cat, col in panel.items():
            assert col == ref[cat], (cat, panel, ref)


def test_user_palette_list():
    a = _adata()
    pal = ["#111111", "#222222", "#333333"]
    ref = _single(a, palette=pal)
    assert ref == dict(zip(CATS, pal))
    for panel in _split(a, palette=pal):
        for cat, col in panel.items():
            assert col == ref[cat]


def test_stored_order_on_a_cytome(tmp_path):
    cytome = pytest.importorskip("cytome")
    a = _adata()
    a.obs["cls"] = a.obs["cls"].astype(str)      # a store keeps plain strings
    path = str(tmp_path / "s.cytome")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = cytome.from_anndata(a, output=path)
        ds.add_embedding("X_umap", np.asarray(a.obsm["X_umap"]))
        ds.flush()
        ds.set_categories("cls", order=CATS, colors=["#aa0000", "#00aa00", "#0000aa"])
        ds.close()
    want = dict(zip(CATS, ["#aa0000", "#00aa00", "#0000aa"]))
    assert _single(path) == want
    for panel in _split(path):
        for cat, col in panel.items():
            assert col == want[cat], (cat, panel)
