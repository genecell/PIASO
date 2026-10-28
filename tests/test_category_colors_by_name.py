"""A category keeps its colour whatever a plot leaves out, sorts or stores.

Regressions for the audit of 2026-10-27 (review I2, findings 4, 5, 6 and 12):
colours were assigned by position among the categories a plot happened to draw,
so a ``cell_mask``, a store converted from AnnData, a Sankey's ``color_by``, a
sorted stacked bar, a dropped empty group or a dendrogram's leaves each
recoloured some types. One resolver now assigns colours by name over the
column's full order (``piaso.plotting._categories``).
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso

CATS = ["Exc", "Inh", "Astro"]                  # not alphabetical on purpose
COLS = {"Exc": "#ff0000", "Inh": "#00ff00", "Astro": "#0000ff"}



def _is_violin_body(c):
    """A violin's body: a PolyCollection, or matplotlib 3.10's FillBetweenPolyCollection (its
    subclass, which seaborn's violins are drawn with there), so an exact class name finds none."""
    from matplotlib.collections import PolyCollection
    return isinstance(c, PolyCollection)

def _adata(n=600, seed=0):
    rng = np.random.default_rng(seed)
    lab = pd.Categorical(rng.choice(CATS, n), categories=CATS)
    X = rng.poisson(1.0, size=(n, 6)).astype(np.float32)
    obs = pd.DataFrame({"celltype": lab,
                        "batch": rng.choice(["s1", "s2"], n),
                        "n_counts": rng.poisson(100, n).astype(float)},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(sp.csr_matrix(X), obs=obs,
                   var=pd.DataFrame(index=[f"g{i}" for i in range(6)]))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obsm["X_pca"] = rng.normal(size=(n, 5))
    a.uns["celltype_colors"] = [COLS[c] for c in CATS]
    return a


def _store(tmp_path, adata, name="emb.cytome"):
    import cytome
    path = str(tmp_path / name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = cytome.from_anndata(adata, output=path)
        ds.close()
    return path


def _scatter_colours(ax):
    out = {}
    for coll in ax.collections:
        lab = coll.get_label()
        if lab and not lab.startswith("_") and len(coll.get_facecolor()):
            out[lab] = mcolors.to_hex(coll.get_facecolor()[0])
    return out


def _legend_colours(legend):
    out = {}
    for h, t in zip(legend.legend_handles, legend.get_texts()):
        for attr in ("get_markerfacecolor", "get_facecolor", "get_color"):
            fn = getattr(h, attr, None)
            if fn is None:
                continue
            c = fn()
            c = c if isinstance(c, str) else np.ravel(np.asarray(c))[:4]
            out[t.get_text()] = mcolors.to_hex(c)
            break
    return out


# --- the resolver ------------------------------------------------------------

def test_resolver_assigns_by_name_over_the_full_order():
    from piaso.plotting._categories import resolve_category_style
    a = _adata()
    style = resolve_category_style(a, "celltype")
    assert style.order == CATS
    assert style.colors == COLS
    # a subset of the values changes what is present, never a colour
    sub = a.obs["celltype"][a.obs["celltype"] != "Exc"]
    assert style.present(sub) == ["Inh", "Astro"]
    assert style.color("Inh") == COLS["Inh"]


def test_resolver_missing_labels_are_not_a_category():
    from piaso.plotting._categories import resolve_category_style, missing_mask
    vals = np.array(["a", None, np.nan, "nan", "b"], dtype=object)
    assert missing_mask(vals).tolist() == [False, True, True, True, False]
    style = resolve_category_style(values=vals, read_stored=False, na_color="#cccccc")
    assert style.order == ["a", "b"]
    assert style.color(None) == "#cccccc"


def test_resolver_qualitative_name_gives_its_colours_in_order():
    from piaso.plotting._categories import resolve_category_style
    style = resolve_category_style(values=np.array(["x", "y", "z"]), palette="tab10",
                                   read_stored=False)
    cm = matplotlib.colormaps["tab10"]
    assert [style.color(c) for c in "xyz"] == [mcolors.to_hex(cm.colors[i]) for i in range(3)]


# --- plotEmbedding ---------------------------------------------------------------

def test_cell_mask_keeps_each_category_colour():
    a = _adata()
    mask = np.asarray(a.obs["celltype"].isin(["Inh", "Astro"]))
    plt.close("all")
    fig, ax = piaso.pl.embedding(a, color="celltype", cell_mask=mask,
                                 show=False, return_fig=True)
    assert _scatter_colours(ax) == {"Inh": COLS["Inh"], "Astro": COLS["Astro"]}
    assert _legend_colours(ax.get_legend()) == {"Inh": COLS["Inh"], "Astro": COLS["Astro"]}
    plt.close(fig)


def test_store_converted_from_anndata_keeps_the_colours(tmp_path):
    a = _adata()
    path = _store(tmp_path, a)
    plt.close("all")
    fig, ax = piaso.pl.embedding(path, color="celltype", show=False, return_fig=True)
    assert _scatter_colours(ax) == COLS
    # and the legend follows the categorical order, not a natural sort
    assert [t.get_text() for t in ax.get_legend().get_texts()] == CATS
    plt.close(fig)


def test_split_embedding_on_a_converted_store_keeps_the_colours(tmp_path):
    a = _adata()
    path = _store(tmp_path, a)
    plt.close("all")
    piaso.pl.split_embedding(path, color="celltype", splitby="batch", show_figure=False)
    fig = plt.gcf()
    got = {}
    for ax in fig.axes:
        got.update(_scatter_colours(ax))
    assert got == COLS
    plt.close("all")


def test_categorical_nan_is_drawn_in_na_color_with_an_NA_entry():
    a = _adata()
    lab = a.obs["celltype"].astype(object)
    lab.iloc[:50] = np.nan
    a.obs["ct_nan"] = pd.Categorical(lab, categories=CATS)
    plt.close("all")
    fig, ax = piaso.pl.embedding(a, color="ct_nan", show=False, return_fig=True,
                                 na_color="#aaaaaa")
    labels = [c.get_label() for c in ax.collections]
    assert "nan" not in labels
    leg = _legend_colours(ax.get_legend())
    assert leg.get("NA") == "#aaaaaa"
    # every NaN cell is drawn, in na_color
    grey = [c for c in ax.collections
            if len(c.get_facecolor()) and mcolors.to_hex(c.get_facecolor()[0]) == "#aaaaaa"]
    assert sum(len(c.get_offsets()) for c in grey) == 50
    plt.close(fig)


def test_continuous_nan_cells_are_visible():
    a = _adata()
    a.obs["score"] = np.random.default_rng(1).normal(size=a.n_obs)
    a.obs.loc[a.obs.index[:50], "score"] = np.nan
    plt.close("all")
    fig, ax = piaso.pl.embedding(a, color="score", show=False, return_fig=True)
    fig.canvas.draw()          # face colours are mapped through the cmap at draw
    n_drawn = 0
    for coll in ax.collections:
        fc = coll.get_facecolors()
        if len(fc) == 0:
            continue
        n_drawn += int((np.asarray(fc)[:, 3] > 0).sum()) if len(fc) > 1 else (
            len(coll.get_offsets()) if fc[0][3] > 0 else 0)
    assert n_drawn == a.n_obs
    plt.close(fig)


def test_sort_order_size_draws_a_rare_type_on_top():
    rng = np.random.default_rng(2)
    n_big, n_rare = 4000, 40
    xy = rng.normal(scale=0.3, size=(n_big + n_rare, 2))
    xy[:n_rare] = rng.normal(scale=0.03, size=(n_rare, 2))   # inside the dense core
    lab = pd.Categorical(["Astro"] * n_rare + ["Exc"] * n_big,
                         categories=["Astro", "Exc"])      # rare one first
    a = ad.AnnData(np.zeros((n_big + n_rare, 1), dtype=np.float32),
                   obs=pd.DataFrame({"ct": lab}, index=[f"c{i}" for i in range(n_big + n_rare)]))
    a.obsm["X_umap"] = xy
    a.uns["ct_colors"] = ["#ff0000", "#0000ff"]

    def red_pixels(**kw):
        plt.close("all")
        fig, ax = piaso.pl.embedding(a, color="ct", show=False, return_fig=True,
                                     point_size=6, legend_loc="none", **kw)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
        plt.close(fig)
        return int(((rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)).sum())

    hidden = red_pixels()                       # category order: Astro drawn first
    shown = red_pixels(sort_order="size")
    assert shown > 5 * max(hidden, 1)


def test_sort_order_random_is_deterministic_and_keeps_colours():
    a = _adata()
    plt.close("all")
    fig, ax = piaso.pl.embedding(a, color="celltype", show=False, return_fig=True,
                                 sort_order="random")
    assert _legend_colours(ax.get_legend()) == COLS
    pts = [c for c in ax.collections if len(c.get_offsets()) == a.n_obs]
    assert len(pts) == 1
    fc1 = np.asarray(pts[0].get_facecolors()).copy()
    plt.close(fig)
    fig, ax = piaso.pl.embedding(a, color="celltype", show=False, return_fig=True,
                                 sort_order="random")
    pts = [c for c in ax.collections if len(c.get_offsets()) == a.n_obs]
    np.testing.assert_array_equal(fc1, np.asarray(pts[0].get_facecolors()))
    plt.close(fig)


# --- stacked bar, violin, Sankey --------------------------------------------------

def _bar_colours(ax):
    return {lab: mcolors.to_hex(h.patches[0].get_facecolor())
            for h, lab in zip(*ax.get_legend_handles_labels())}


@pytest.mark.parametrize("sort_groups", [False, True])
def test_stacked_bar_colours_follow_names(sort_groups):
    a = _adata()
    plt.close("all")
    fig, ax = piaso.pl.stacked_barplot(a, groupby="celltype", splitby="batch",
                                       sort_groups=sort_groups, show=False, return_fig=True)
    assert _bar_colours(ax) == COLS
    plt.close(fig)


def test_stacked_bar_orders_clusters_naturally():
    rng = np.random.default_rng(4)
    n = 400
    a = ad.AnnData(np.zeros((n, 1), dtype=np.float32),
                   obs=pd.DataFrame({"leiden": rng.choice([str(i) for i in range(12)], n),
                                     "batch": rng.choice(["s1", "s2"], n)},
                                    index=[f"c{i}" for i in range(n)]))
    plt.close("all")
    fig, ax = piaso.pl.stacked_barplot(a, groupby="leiden", splitby="batch",
                                       show=False, return_fig=True)
    labels = ax.get_legend_handles_labels()[1]
    assert labels == [str(i) for i in range(12)]          # 0, 1, 2, ... 10, 11
    plt.close(fig)


def test_violin_dropped_group_keeps_later_colours():
    a = _adata()
    a.obs["n_counts"] = a.obs["n_counts"].astype(float)
    a.obs.loc[a.obs["celltype"] == "Exc", "n_counts"] = np.nan      # Exc: all NaN
    plt.close("all")
    fig, axes = piaso.pl.violin(a, ["n_counts"], groupby="celltype", show=False,
                                return_fig=True)
    ax = axes[0]
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    bodies = [c for c in ax.collections if _is_violin_body(c)]
    got = {t: mcolors.to_hex(b.get_facecolor()[0]) for t, b in zip(ticks, bodies)}
    assert got == {"Inh": COLS["Inh"], "Astro": COLS["Astro"]}
    plt.close(fig)


def _sankey_node_colours(ax, names):
    """{name: colour} for the node blocks, matched to their labels by height."""
    texts = [(t.get_position(), t.get_text()) for t in ax.texts if t.get_text() in names]
    nodes = [p for p in ax.patches if p.get_zorder() == 5]
    out = {}
    for (x, y), name in texts:
        best = min(nodes, key=lambda p: abs(p.get_x() + p.get_width() / 2 - x)
                   + abs(p.get_y() + p.get_height() / 2 - y))
        out[(name, x < 0.5)] = mcolors.to_hex(best.get_facecolor())
    return out


def test_sankey_color_by_right_uses_the_right_columns_colours():
    a = _adata()
    a.obs["region"] = pd.Categorical(np.where(np.arange(a.n_obs) % 3 == 0, "CTX", "HPF"),
                                     categories=["HPF", "CTX"])
    a.uns["region_colors"] = ["#111111", "#eeeeee"]           # HPF, CTX
    plt.close("all")
    fig, ax = piaso.pl.sankey(a, "celltype", "region", color_by="right", show=False,
                              return_fig=True)
    got = _sankey_node_colours(ax, {"HPF", "CTX", *CATS})
    assert got[("HPF", False)] == "#111111" and got[("CTX", False)] == "#eeeeee"
    # flows take the right node's colour
    flows = {mcolors.to_hex(p.get_facecolor()) for p in ax.patches if p.get_zorder() == 2}
    assert flows == {"#111111", "#eeeeee"}
    plt.close(fig)


def test_sankey_auto_order_does_not_recolour():
    a = _adata()
    plt.close("all")
    fig, ax = piaso.pl.sankey(a, "celltype", "batch", show=False, return_fig=True)
    plain = {k: v for k, v in _sankey_node_colours(ax, set(CATS)).items()}
    plt.close(fig)
    fig, ax = piaso.pl.sankey(a, "celltype", "batch", left_order="auto", show=False,
                              return_fig=True)
    auto = {k: v for k, v in _sankey_node_colours(ax, set(CATS)).items()}
    assert plain == auto and {k[0]: v for k, v in plain.items()} == COLS
    plt.close(fig)


def test_store_order_and_colours_reach_violin_bar_sankey(tmp_path):
    import cytome
    a = _adata()
    path = _store(tmp_path, a)
    ds = cytome.open(path)
    ds.set_categories("celltype", order=["Astro", "Inh", "Exc"],
                      colors=["#123456", "#654321", "#abcdef"])
    ds.close()
    want = {"Astro": "#123456", "Inh": "#654321", "Exc": "#abcdef"}
    plt.close("all")
    fig, axes = piaso.pl.violin(path, ["n_counts"], groupby="celltype", show=False,
                                return_fig=True)
    ticks = [t.get_text() for t in axes[0].get_xticklabels()]
    assert ticks == ["Astro", "Inh", "Exc"]
    bodies = [c for c in axes[0].collections if _is_violin_body(c)]
    assert {t: mcolors.to_hex(b.get_facecolor()[0]) for t, b in zip(ticks, bodies)} == want
    plt.close("all")
    fig, ax = piaso.pl.stacked_barplot(path, groupby="celltype", splitby="batch",
                                       show=False, return_fig=True)
    assert _bar_colours(ax) == want
    assert ax.get_legend_handles_labels()[1] == ["Astro", "Inh", "Exc"]
    plt.close("all")
    fig, ax = piaso.pl.sankey(path, "celltype", "batch", show=False, return_fig=True)
    got = {k[0]: v for k, v in _sankey_node_colours(ax, set(CATS)).items()}
    assert got == want
    plt.close("all")
