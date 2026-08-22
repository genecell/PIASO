"""Multi-panel categorical legends must not be drawn on top of each other.

`plotEmbedding` anchors each panel's categorical legend at axes coordinates
``(1.05, 1)``. In a one-panel figure that is the space to the right of the plot
and is fine. In a multi-panel grid nothing reserves that space, so panel N's
legend is drawn inside panel N+1's cell, where N+1's opaque axes patch then
paints over it -- the labels come out interleaved and clipped
("LongCellTypeName17" and "LongCellTypeName10" overprinting as "peName17 0").

Found on a 27-cell-type PBMC figure. This test pins the geometry so a fix can
be verified and a regression caught: every categorical legend must sit inside
its own panel's horizontal span.
"""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
ad = pytest.importorskip("anndata")
piaso = pytest.importorskip("piaso")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _toy(n_cats=27, n=600):
    rng = np.random.default_rng(0)
    a = ad.AnnData(np.zeros((n, 2), dtype="float32"))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obs["A"] = pd.Categorical([f"LongCellTypeName{i % n_cats}" for i in range(n)])
    a.obs["B"] = pd.Categorical([str(i % 15) for i in range(n)])
    return a


def _as_figure(ret):
    """`return_fig=True` gives (fig, axs) for a grid and a Figure otherwise."""
    return ret[0] if isinstance(ret, tuple) else ret


def _boxes(fig):
    """Axes and legend extents in figure coordinates, per panel."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent().transformed(inv)
        ab = ax.get_position()
        out.append({"ax": (ab.x0, ab.x1), "leg": (lb.x0, lb.x1)})
    return out


def _overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])


def test_no_legend_overlaps_another_panel_or_its_legend():
    """The invariant is non-overlap, not narrowness.

    A legend of 27 long labels is legitimately wider than the panel it belongs
    to; what must not happen is that it lands on a *neighbouring* panel, where
    that panel's opaque background then paints over the labels.
    """
    fig = _as_figure(piaso.pl.embedding(_toy(), basis="X_umap",
                                        color=["A", "B"], show=False,
                                        return_fig=True))
    boxes = _boxes(fig)
    assert len(boxes) == 2, "expected one categorical legend per panel"

    for i, bi in enumerate(boxes):
        for j, bj in enumerate(boxes):
            if i == j:
                continue
            assert _overlap(bi["leg"], bj["ax"]) <= 0, (
                f"legend {i} {bi['leg']} covers panel {j} {bj['ax']}")
            if i < j:
                assert _overlap(bi["leg"], bj["leg"]) <= 0, (
                    f"legends {i} and {j} overlap: {bi['leg']} vs {bj['leg']}")
        assert _overlap(bi["leg"], bi["ax"]) <= 0, (
            f"legend {i} overlaps its own plotting area")


def test_the_figure_is_widened_rather_than_the_panels_squeezed():
    """Room is added, not borrowed — the panels keep their size."""
    narrow = _as_figure(piaso.pl.embedding(_toy(n_cats=3), basis="X_umap",
                                           color=["A", "B"], show=False,
                                           return_fig=True))
    wide = _as_figure(piaso.pl.embedding(_toy(n_cats=27), basis="X_umap",
                                         color=["A", "B"], show=False,
                                         return_fig=True))
    nw, _ = narrow.get_size_inches()
    ww, _ = wide.get_size_inches()
    assert ww > nw, "27 categories should widen the figure beyond 3"

    def panel_inches(f):
        w, _ = f.get_size_inches()
        return max(ax.get_position().width * w for ax in f.axes)

    assert panel_inches(wide) == pytest.approx(panel_inches(narrow), rel=0.02)
    plt.close(narrow)
    plt.close(wide)


def test_a_single_panel_legend_is_unaffected():
    """The single-panel case is correct and must stay that way."""
    fig = _as_figure(piaso.pl.embedding(_toy(), basis="X_umap", color="A",
                                        show=False, return_fig=True))
    boxes = _boxes(fig)
    assert len(boxes) == 1
    assert boxes[0]["leg"][0] >= boxes[0]["ax"][0], (
        "legend should not overlap its own plotting area")
    plt.close(fig)
