"""A cell-sized dot was mostly stroke, and the stroke is not ours to draw.

Matplotlib strokes every scatter marker at ``rcParams['patch.linewidth']``
(1.0 pt) in ``edgecolors='face'`` unless told otherwise. That is invisible on a
20 pt dot and ruinous on a cell-sized one:

* ``point_size=0.2`` is a disc **0.45 pt** across. Add a 1.0 pt stroke and it
  draws **1.45 pt** — 3.2x the size asked for.
* With ``alpha < 1`` the stroke and the fill composite *twice* around the rim,
  so the dot renders as an annulus with a washed-out core whose colour is
  neither the palette colour nor the background.

Reported from a 600 dpi render of a published panel, where the workaround was
``plt.rc_context({"patch.linewidth": 0})`` — a global rcParam, because nothing
in the signature reached the scatter call.

These tests pin both halves: the property on every scatter PIASO draws, and the
rendered pixels, since the property is only a proxy for what the reader sees.
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
    n = 300
    a = ad.AnnData(
        np.abs(rng.normal(size=(n, 3))).astype(np.float32),
        obs=pd.DataFrame(
            {"ct": rng.choice(list("abc"), n), "split": rng.choice(["x", "y"], n)},
            index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    return a


def _linewidths(ax):
    return {float(lw) for c in ax.collections for lw in np.atleast_1d(c.get_linewidths())}


def _radial_ink(lw, ec, s=4.0, alpha=0.6):
    """Mean darkness in 1 px rings out from the centre of one isolated marker.

    Rendered at 600 dpi, which is where the artefact was found. Background is
    white, so "ink" is 1 - luminance and a solid disc must peak at the centre.
    """
    fig, ax = plt.subplots(figsize=(1, 1), dpi=600)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()
    ax.scatter([0], [0], s=s, c=["#1f77b4"], alpha=alpha,
               linewidths=lw, edgecolors=ec, rasterized=False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255
    plt.close(fig)
    h, w, _ = img.shape
    ink = 1 - img[..., :3].mean(2)
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2)
    return np.array([ink[(r >= k) & (r < k + 1)].mean() for k in range(14)])


# --------------------------------------------------- the rendered artefact --


def test_the_default_marker_renders_as_a_disc_not_a_ring():
    """The reported symptom, measured in pixels rather than argued from code."""
    after = _radial_ink(0.0, "none")
    assert after.argmax() == 0, (
        "ink should be greatest at the centre of a filled marker; peaked at "
        f"r={after.argmax()} px instead, which is a ring: {after[:6].round(3)}")
    assert np.all(np.diff(after[:6]) <= 0), "a disc fades outward, monotonically"


def test_the_old_default_really_did_draw_a_ring():
    """Guards the premise. If matplotlib ever stops stroking, this fails and
    the fix can be revisited rather than cargo-culted."""
    before = _radial_ink(1.0, "face")
    assert before.argmax() > 0, (
        "expected the 1.0 pt stroke to put peak ink off-centre; if this fails, "
        "matplotlib's patch.linewidth default may have changed")
    assert before[0] < before.max(), "the washed-out core is the whole complaint"


def test_the_stroke_was_making_the_marker_bigger_than_asked():
    before, after = _radial_ink(1.0, "face"), _radial_ink(0.0, "none")

    def half_ink_radius(p):
        return int(np.argmax(p < p.max() * 0.5))

    assert half_ink_radius(before) >= 2 * half_ink_radius(after), (
        "the stroke should roughly double this marker's drawn radius; got "
        f"{half_ink_radius(before)} px vs {half_ink_radius(after)} px")


# ------------------------------------------------- every scatter we render --


@pytest.mark.parametrize("kw", [
    {"color": "ct"},                              # categorical
    {"color": "g1"},                              # continuous
    {"color": "ct", "groups": ["a"]},             # highlight + na_color background
])
def test_plot_embedding_strokes_nothing_by_default(adata, kw):
    fig, ax = plt.subplots()
    piaso_plot(adata, ax=ax, **kw)
    assert _linewidths(ax) == {0.0}, _linewidths(ax)
    for c in ax.collections:
        assert c.get_edgecolors().size == 0 or np.allclose(
            c.get_edgecolors()[:, 3], 0.0), "edges should be invisible"
    plt.close(fig)


def piaso_plot(adata, **kw):
    import piaso

    kw.setdefault("basis", "X_umap")
    kw.setdefault("point_size", 0.2)
    return piaso.pl.plotEmbedding(adata, show=False, **kw)


def test_the_marker_is_exactly_the_size_requested(adata):
    """``point_size`` is an area in pt^2; with no stroke the drawn diameter is
    its square root and nothing more."""
    fig, ax = plt.subplots()
    piaso_plot(adata, color="ct", point_size=0.2)
    drawn = np.sqrt(0.2) + max(_linewidths(ax) | {0.0})
    assert drawn == pytest.approx(np.sqrt(0.2)), (
        f"a 0.2 pt^2 marker should draw {np.sqrt(0.2):.3f} pt across, not {drawn:.3f}")
    plt.close(fig)


def test_split_panels_get_the_same_treatment(adata):
    import piaso

    fig = piaso.pl.plot_embeddings_split(
        adata, color="ct", splitby="split", basis="X_umap",
        point_size=0.2, show_figure=False)
    axes = [a for a in plt.gcf().axes if a.collections]
    assert axes, "no panels drawn"
    for ax in axes:
        assert _linewidths(ax) == {0.0}, (ax.get_title(), _linewidths(ax))
    plt.close("all")


def test_the_feature_scatter_too(adata):
    import piaso

    fig, ax = plt.subplots()
    piaso.pl.scatter(adata, x="g1", y="g2", color="ct", point_size=0.2,
                     ax=ax, show=False)
    assert _linewidths(ax) == {0.0}, _linewidths(ax)
    plt.close("all")


# ------------------------------------------------------------- opting in ---


@pytest.mark.parametrize("fn", ["plotEmbedding", "plot_embeddings_split"])
def test_an_outline_is_still_available_when_asked_for(adata, fn):
    """Removing a default is not removing the option: a deliberate outline on
    a large marker is a legitimate style."""
    import piaso

    if fn == "plotEmbedding":
        fig, ax = plt.subplots()
        piaso.pl.plotEmbedding(adata, color="ct", basis="X_umap", point_size=40,
                               linewidths=1.5, edgecolors="black", ax=ax, show=False)
        axes = [ax]
    else:
        piaso.pl.plot_embeddings_split(
            adata, color="ct", splitby="split", basis="X_umap", point_size=40,
            linewidths=1.5, edgecolors="black", show_figure=False)
        axes = [a for a in plt.gcf().axes if a.collections]
    for ax in axes:
        assert 1.5 in _linewidths(ax), _linewidths(ax)
    plt.close("all")


# ------------------------------------------- kwargs that went nowhere -------


def test_split_refuses_a_keyword_it_cannot_honour(adata):
    """It accepted ``**kwargs`` and dropped them, so styling a split panel
    failed silently. Raise where plotEmbedding raises."""
    import piaso

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        piaso.pl.plot_embeddings_split(
            adata, color="ct", splitby="split", basis="X_umap",
            marker="^", show_figure=False)
    plt.close("all")


@pytest.mark.parametrize("alias,value,attr", [
    ("ncols", 2, None),
    ("size", 3.0, None),
])
def test_the_scanpy_aliases_still_pass(adata, alias, value, attr):
    import piaso

    piaso.pl.plot_embeddings_split(
        adata, color="ct", splitby="split", basis="X_umap",
        show_figure=False, **{alias: value})
    plt.close("all")


def test_show_false_is_honoured_rather_than_dropped(adata, monkeypatch):
    """``show`` is what every other piaso.pl function calls this. It reached
    ``**kwargs`` here and was discarded, so ``show=False`` displayed anyway."""
    import piaso

    shown = {"n": 0}
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.__setitem__("n", shown["n"] + 1))
    piaso.pl.plot_embeddings_split(adata, color="ct", splitby="split",
                                   basis="X_umap", show=False)
    assert shown["n"] == 0, "show=False still displayed the figure"
    plt.close("all")
