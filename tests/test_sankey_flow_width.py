"""A Sankey that cannot draw half its flows is not showing them.

On a real crosstab -- 37k cells, Leiden x Subclass, 107 flows -- the median
flow is 0.09 pt tall at 3 inches and 58 of the 107 fall under a 0.25 pt
hairline. Enlarging the figure does not help: each flow's *share* of the
diagram is fixed, so only the count -> width map can change it.

Two maps do, and they fail in opposite places, which is why both exist:

* a baseline (``min_flow_width``) is an affine offset applied to every flow,
  not a clamp -- nothing is tied, every rank survives -- but it barely
  separates the small flows from each other;
* ``flow_scale`` compresses the range, keeping the small end readable at the
  cost of the large end.

What both give up is proportionality, so the size key defaults on. And
because a node is exactly as tall as the ribbons landing on it, node height
under either map depends on how many ways the node splits -- a documented
consequence, pinned here so it cannot drift into a surprise.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from piaso.plotting._plotSankey import (
    _FLOW_FUNCS, _FLOW_SCALES, _legend_sample_counts, _points_per_unit,
)

HAIRLINE_PT = 0.25


def _adata(counts):
    import anndata as ad

    rows = [(l, r) for (l, r), c in counts.items() for _ in range(int(c))]
    obs = pd.DataFrame(rows, columns=["left", "right"])
    obs.index = pd.Index([f"c{i}" for i in range(len(obs))])
    return ad.AnnData(X=np.zeros((len(obs), 2), dtype=np.float32), obs=obs)


@pytest.fixture
def wide():
    """30 flows spanning 1 to 2000 cells -- a 2000x dynamic range."""
    vals = np.unique(np.geomspace(1, 2000, 30).round().astype(int))
    pairs = [(f"L{i}", f"R{j}") for i in range(6) for j in range(5)]
    return _adata(dict(zip(pairs, vals)))


def _draw(adata, **kw):
    import piaso

    kw.setdefault("figsize", (6, 3))
    out = piaso.pl.sankey(adata, left="left", right="right", show=False,
                          return_fig=True, **kw)
    return out if isinstance(out, tuple) else (out, out.axes[0])


def _ribbons(ax):
    """Left-end thickness of every ribbon, in rail-space units, in draw order.

    ``_bezier_path`` starts the top curve at the left rail and ends the bottom
    curve there, so the first and second-to-last vertices are the two edges of
    the same end.
    """
    out = []
    for p in ax.patches:
        if p.get_zorder() != 2:
            continue
        v = p.get_path().vertices
        out.append(abs(float(v[0][1]) - float(v[-2][1])))
    return np.array(out)


def _nodes(ax):
    """Height of every category block (horizontal layout)."""
    return np.array([p.get_height() for p in ax.patches
                     if p.get_zorder() == 5])


def _legend_bars(ax):
    return [p for p in ax.patches if p.get_zorder() == 6]


def _pt_per_unit(fig, ax):
    """Rendered points per unit of rail-space width, after tight_layout."""
    return _points_per_unit(ax, "horizontal", 1.0, 1.08)


def _expected_counts(adata):
    """Flow counts in the order the ribbons are drawn."""
    ct = pd.crosstab(adata.obs["left"], adata.obs["right"])
    v = ct.values
    return v[v > 0] if v.size else v


# --------------------------------------------------------------- unchanged


def test_the_default_is_the_proportional_layout_it_always_was(wide):
    """linear, no baseline: widths are still count / total * available."""
    fig, ax = _draw(wide)
    got = _ribbons(ax)
    counts = _expected_counts(wide)

    n_left = wide.obs["left"].nunique()
    from piaso.plotting._plotSankey import _effective_gap
    avail = 1.0 - _effective_gap(n_left, 0.03) * (n_left - 1)
    want = counts / counts.sum() * avail

    assert np.allclose(got, want, rtol=1e-9), (
        "linear + no baseline must reproduce the proportional layout exactly; "
        "every existing figure depends on it")


def test_the_default_draws_no_size_legend(wide):
    """Proportional widths need no key -- and no caller asked for one."""
    _, ax = _draw(wide)
    assert _legend_bars(ax) == []


# ------------------------------------------------------------- visibility


def test_linear_leaves_most_of_this_diagram_undrawn(wide):
    """The premise. If this ever stops failing, the rest is unnecessary."""
    fig, ax = _draw(wide)
    pt = _ribbons(ax) * _pt_per_unit(fig, ax)
    assert (pt < HAIRLINE_PT).sum() >= len(pt) // 3, (
        f"expected a wide range to hide many flows, got min {pt.min():.3f} pt")


@pytest.mark.parametrize("kw", [
    {"flow_scale": "log"},
    {"flow_scale": "sqrt", "min_flow_width": 0.25},
    {"min_flow_width": 0.5},
])
def test_every_flow_clears_a_hairline(wide, kw):
    fig, ax = _draw(wide, **kw)
    pt = _ribbons(ax) * _pt_per_unit(fig, ax)
    assert pt.min() >= HAIRLINE_PT, (
        f"{kw} left the thinnest flow at {pt.min():.3f} pt")


def test_a_baseline_delivers_the_width_it_promises(wide):
    """The conversion is taken before tight_layout, which only enlarges the
    axes here -- so the rendered width is at or above the request."""
    fig, ax = _draw(wide, min_flow_width=0.5)
    pt = _ribbons(ax) * _pt_per_unit(fig, ax)
    assert pt.min() >= 0.5, f"asked for 0.5 pt, thinnest rendered {pt.min():.3f}"


# ------------------------------------------------------------------ rank


@pytest.mark.parametrize("kw", [
    {"flow_scale": "sqrt"},
    {"flow_scale": "log"},
    {"min_flow_width": 0.5},
    {"flow_scale": "sqrt", "min_flow_width": 0.25},
])
def test_no_map_reorders_a_single_flow(wide, kw):
    """An offset is not a clamp: it shifts every flow, so nothing ties and
    nothing swaps. A clamp would tie the whole small end at one width."""
    from scipy.stats import spearmanr

    _, ax = _draw(wide, **kw)
    got, counts = _ribbons(ax), _expected_counts(wide)
    assert spearmanr(counts, got).statistic == pytest.approx(1.0), (
        f"{kw} changed the order of the flows")


@pytest.mark.parametrize("kw", [{"min_flow_width": 0.5}, {"flow_scale": "log"}])
def test_distinct_counts_keep_distinct_widths(wide, kw):
    """The failure mode of a clamp, stated as a test."""
    _, ax = _draw(wide, **kw)
    got, counts = _ribbons(ax), _expected_counts(wide)
    assert len(np.unique(np.round(got, 12))) == len(np.unique(counts))


# ------------------------------------------------------------- invariant


@pytest.mark.parametrize("scale", _FLOW_SCALES)
@pytest.mark.parametrize("width", [None, 0.4])
def test_ribbons_still_land_exactly_on_their_blocks(wide, scale, width):
    """The one guarantee the layout makes, under every map."""
    _, ax = _draw(wide, flow_scale=scale, min_flow_width=width)
    ribbons, nodes = _ribbons(ax), _nodes(ax)
    n_left = wide.obs["left"].nunique()
    assert nodes.size == n_left + wide.obs["right"].nunique()
    assert ribbons.sum() == pytest.approx(nodes[:n_left].sum(), rel=1e-9), (
        "the left blocks no longer add up to the ribbons leaving them")


def test_a_fragmented_node_is_drawn_taller_than_a_clean_one():
    """Documented, not accidental: node height is the sum of its ribbons, so
    any non-proportional map pays the baseline once per flow. A reader must
    not take node height for group size under a scale.

    C is here to make the baseline bind. ``min_flow_width`` guarantees a
    minimum rather than adding an offset unconditionally, so on a diagram
    whose flows are all comfortably above the request no offset is applied and
    nothing is distorted -- A and B would come out identical, correctly. It
    takes a wide range for the offset to switch on, which is also the only
    situation anyone would ask for one.
    """
    adata = _adata({("A", "R0"): 300,
                    **{("B", f"R{i}"): 30 for i in range(10)},
                    ("C", "R0"): 20000})
    _, ax = _draw(adata, min_flow_width=0.5)
    a, b = _nodes(ax)[:2]
    assert b > a * 1.2, (
        f"equal-sized A and B drawn {a:.4f} vs {b:.4f}; the fan-out effect the "
        "docstring warns about is missing, so the warning is now wrong")


# -------------------------------------------------------------- the legend


@pytest.mark.parametrize("kw,drawn", [
    ({}, False),
    ({"flow_scale": "sqrt"}, True),
    ({"flow_scale": "log"}, True),
    ({"min_flow_width": 0.4}, True),
    ({"show_size_legend": True}, True),
    ({"flow_scale": "log", "show_size_legend": False}, False),
])
def test_the_legend_appears_exactly_when_width_stops_meaning_count(wide, kw, drawn):
    _, ax = _draw(wide, **kw)
    assert bool(_legend_bars(ax)) is drawn


def test_the_legend_names_the_transform(wide):
    _, ax = _draw(wide, flow_scale="sqrt", min_flow_width=0.3)
    blurb = " ".join(t.get_text() for t in ax.texts)
    assert "square-root" in blurb and "0.3 pt baseline" in blurb


def test_the_legend_bars_are_the_widths_they_claim(wide):
    """A key drawn at a different scale from the ribbons is worse than none."""
    _, ax = _draw(wide, flow_scale="log")
    counts = _expected_counts(wide)
    samples = _legend_sample_counts(counts.min(), counts.max())

    bars = sorted(_legend_bars(ax), key=lambda p: p.get_x())
    assert len(bars) == len(samples)

    ribbons = _ribbons(ax)
    for c, bar in zip(samples, bars):
        if c not in counts:
            continue
        want = ribbons[np.flatnonzero(counts == c)[0]]
        assert bar.get_height() == pytest.approx(want, rel=1e-6), (
            f"legend shows {bar.get_height():.5f} for {c} cells, the ribbon "
            f"is {want:.5f}")


def test_sample_counts_span_the_flows_present():
    got = _legend_sample_counts(1, 2000)
    assert got[0] == 1 and got[-1] == 2000
    assert got == sorted(got) and len(got) <= 4


# -------------------------------------------------------------- rejection


def test_an_unaffordable_baseline_says_what_the_diagram_can_afford(wide):
    """The affordable width depends on the flow count, which the caller has no
    way to know -- so the error has to do the arithmetic."""
    with pytest.raises(ValueError) as e:
        _draw(wide, min_flow_width=40.0)
    msg = str(e.value)
    n = len(_expected_counts(wide))
    assert f"{n} flows" in msg
    assert "can give every flow" in msg
    assert "sqrt" in msg, "linear was the cause; the cheaper map should be named"


def test_sqrt_is_not_told_to_use_sqrt(wide):
    with pytest.raises(ValueError, match="cannot be met") as e:
        _draw(wide, flow_scale="sqrt", min_flow_width=40.0)
    assert "flow_scale='sqrt'" not in str(e.value)


@pytest.mark.parametrize("bad", ["Log", "log1p", "linear ", "none"])
def test_an_unknown_scale_is_rejected(wide, bad):
    with pytest.raises(ValueError, match="flow_scale"):
        _draw(wide, flow_scale=bad)


def test_a_negative_baseline_is_rejected(wide):
    with pytest.raises(ValueError, match="min_flow_width"):
        _draw(wide, min_flow_width=-1.0)


# ---------------------------------------------------------------- layouts


@pytest.mark.parametrize("layout", ["horizontal", "vertical", "circular"])
def test_every_layout_takes_a_scale_and_a_baseline(wide, layout):
    _, ax = _draw(wide, layout=layout, flow_scale="sqrt", min_flow_width=0.3)
    assert (len([p for p in ax.patches if p.get_zorder() == 2])
            == len(_expected_counts(wide)))
    assert _legend_bars(ax), f"{layout} drew no width legend"


def test_log1p_keeps_a_one_cell_flow_positive():
    """log, not log1p, would map a one-cell flow to zero width."""
    assert float(_FLOW_FUNCS["log"](np.array([1.0]))[0]) > 0
