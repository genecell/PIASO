"""A node too small to see is a category the reader will not know about.

By default a node has no size of its own: it is exactly as tall as the ribbons
landing on it, which is what makes them meet. That has a consequence nobody
asks for -- under ``min_flow_width`` a node's height is ``m * size +
b * fan_out``, so how tall a small group draws depends on how many ways it
splits, which is the wrong variable.

``min_node_size`` and ``node_scale`` give the *node* the affine map instead and
split its height across its own ribbons. The ribbons still meet and still fill
the rail; what they give up is a global count-to-thickness map, because the
slope now depends on which node a ribbon leaves.

That is why there is one ``show_size_legend`` and not two keys: at most one map
is global at a time, and only the global one can carry an honest key. These
tests pin that trade, in both directions.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from piaso.plotting._plotSankey import (
    _effective_gap, _points_per_unit, _solve_node_heights, _split_node_height,
)


def _adata(counts):
    import anndata as ad

    rows = [(l, r) for (l, r), c in counts.items() for _ in range(int(c))]
    obs = pd.DataFrame(rows, columns=["left", "right"])
    obs.index = pd.Index([f"c{i}" for i in range(len(obs))])
    return ad.AnnData(X=np.zeros((len(obs), 2), dtype=np.float32), obs=obs)


@pytest.fixture
def lopsided():
    """Five left categories spanning 4000 to 12 cells, each fanning out 3 ways.

    The fan-out is equal on purpose: it isolates *size* as the only thing that
    can make one node shorter than another.
    """
    sizes = {"A": 4000, "B": 800, "C": 200, "D": 40, "E": 12}
    counts = {}
    for cat, n in sizes.items():
        counts[(cat, "x")] = int(n * 0.6)
        counts[(cat, "y")] = int(n * 0.3)
        counts[(cat, "z")] = n - int(n * 0.6) - int(n * 0.3)
    return _adata(counts)


@pytest.fixture
def uneven_fanout():
    """Two left categories of the SAME size, splitting 1 way and 4 ways.

    ``tiny`` is here only to make a flow baseline bind: with every flow already
    an eighth of the rail, ``min_flow_width`` would be slack and the fan-out
    effect it causes would not appear at all.
    """
    return _adata({("wide", "p"): 100, ("wide", "q"): 100,
                   ("wide", "r"): 100, ("wide", "s"): 100,
                   ("narrow", "p"): 400, ("tiny", "q"): 2})


def _draw(adata, **kw):
    import piaso

    kw.setdefault("figsize", (6, 4))
    out = piaso.pl.sankey(adata, left="left", right="right", show=False,
                          return_fig=True, **kw)
    return out if isinstance(out, tuple) else (out, out.axes[0])


def _left_nodes(ax, cats):
    """Height of each left-rail block, in the order ``cats`` gives.

    Paired with the labels rather than assumed from draw order: the rail is in
    the *resolved category* order, which is alphabetical for plain strings and
    need not match the order a test happens to list.
    """
    # The left rail is centred on x=0, so its blocks start at -node_width/2;
    # the right rail's start near 1. zorder 5 is what _draw_rail uses.
    blocks = [p for p in ax.patches if p.get_zorder() == 5 and p.get_x() < 0.5]
    labels = [t for t in ax.texts
              if t.get_position()[0] < 0 and t.get_text() in set(cats)]
    assert len(blocks) == len(cats), (len(blocks), len(cats))
    assert len(labels) == len(cats), [t.get_text() for t in labels]
    by_cat = {}
    for b in blocks:
        centre = b.get_y() + b.get_height() / 2
        near = min(labels, key=lambda t: abs(t.get_position()[1] - centre))
        by_cat[near.get_text()] = b.get_height()
    return np.array([by_cat[c] for c in cats])


def _pt_per_unit(ax):
    return _points_per_unit(ax, "horizontal", 1.0, 1.08)


def _key_title(ax):
    for t in ax.texts:
        txt = t.get_text()
        if txt.startswith("cells per "):
            return txt
    return None


# ------------------------------------------------------------ the premise --


def test_without_a_floor_the_smallest_node_is_invisible(lopsided):
    """If this stops failing, nothing below is needed."""
    _, ax = _draw(lopsided)
    h = _left_nodes(ax, list("ABCDE")) * _pt_per_unit(ax)
    assert h.min() < 1.0, f"expected a hairline node, got {h.min():.2f} pt"


def test_fan_out_not_size_decides_height_when_only_flows_are_floored(uneven_fanout):
    """The wrong variable, measured: two equal groups drawn at different heights.

    This is what ``min_node_size`` exists to fix, and it is a property of the
    default layout rather than a bug -- a node IS its ribbons.
    """
    # 20 pt, because the distortion is worth n_flows * baseline: at 3 pt these
    # two are 9% apart, at 20 pt they are 2x. The mechanism is the same either
    # way -- the size of it is what a reader would or would not notice.
    _, ax = _draw(uneven_fanout, min_flow_width=20.0)
    wide, narrow, _tiny = _left_nodes(ax, ["wide", "narrow", "tiny"])
    assert wide > narrow * 1.5, (
        "a 4-way split should out-draw an equal 1-way split under a flow "
        f"baseline; got {wide:.4f} vs {narrow:.4f}")


# --------------------------------------------------------- the node floor --


def test_the_floor_is_at_least_the_height_it_says(lopsided):
    """A floor is a minimum, and the asymmetry is deliberate.

    The points-per-unit conversion happens before ``tight_layout``, which then
    resizes an axis-off subplot -- usually outward. Coming out taller than
    asked is therefore normal and harmless; coming out *shorter* is the
    promise going unmet, and that warns (tested below).
    """
    _, ax = _draw(lopsided, min_node_size=10.0)
    h = _left_nodes(ax, list("ABCDE")) * _pt_per_unit(ax)
    assert h.min() >= 10.0 * 0.95, (
        f"the shortest node came out {h.min():.2f} pt, under the 10 pt asked for")
    assert h.min() < 10.0 * 3, (
        f"{h.min():.2f} pt for a 10 pt floor means the conversion is wrong, "
        "not that the axes grew a little")


def test_a_floor_the_axes_cannot_honour_warns_rather_than_lying(lopsided, monkeypatch):
    """The conversion runs before ``tight_layout``, which can shrink the axes.

    Forced rather than coaxed with long labels: what matters is that a floor
    which silently went unmet is reported, not which label length triggers it.
    The shrink is applied to every call after the first, which is exactly when
    the real one happens.
    """
    import piaso.plotting._plotSankey as S

    real = S._points_per_unit
    calls = {"n": 0}

    def _shrinking(*a, **k):
        calls["n"] += 1
        return real(*a, **k) * (1.0 if calls["n"] == 1 else 0.5)

    monkeypatch.setattr(S, "_points_per_unit", _shrinking)
    with pytest.warns(UserWarning, match="shortest node came out"):
        _draw(lopsided, min_node_size=10.0)


def test_the_floor_keeps_nodes_monotone_in_size(lopsided):
    """An offset, not a clamp: no two categories are tied and none overtakes."""
    _, ax = _draw(lopsided, min_node_size=10.0)
    h = _left_nodes(ax, list("ABCDE"))          # A..E is descending in size
    assert np.all(np.diff(h) < 0), f"heights not strictly descending: {h}"


def test_the_floor_lifts_the_small_end_without_reordering(lopsided):
    plain = _left_nodes(_draw(lopsided)[1], list("ABCDE"))
    lifted = _left_nodes(_draw(lopsided, min_node_size=10.0)[1], list("ABCDE"))
    assert lifted[-1] / lifted[0] > plain[-1] / plain[0] * 1.5
    assert list(np.argsort(-plain)) == list(np.argsort(-lifted))


def test_nodes_still_fill_the_rail(lopsided):
    """Heights plus gaps span exactly the rail, floored or not."""
    for kw in ({}, {"min_node_size": 10.0}, {"node_scale": "sqrt"}):
        _, ax = _draw(lopsided, **kw)
        h = _left_nodes(ax, list("ABCDE"))
        gaps = _effective_gap(5, 0.03) * 4
        assert h.sum() + gaps == pytest.approx(1.0, abs=1e-6), kw


def test_ribbons_still_add_up_to_their_node(lopsided):
    """The property that makes a Sankey a Sankey: they meet."""
    _, ax = _draw(lopsided, min_node_size=10.0)
    ribbons = []
    for p in ax.patches:
        if p.get_zorder() != 2:
            continue
        v = p.get_path().vertices
        ribbons.append((float(v[0][1]), abs(float(v[0][1]) - float(v[-2][1]))))
    h = _left_nodes(ax, list("ABCDE"))
    # three ribbons leave each node, in draw order
    per_node = [sum(w for _, w in ribbons[i * 3:(i + 1) * 3]) for i in range(5)]
    assert np.allclose(per_node, h, atol=1e-6), (per_node, h)


def test_node_scale_lifts_the_small_end_too(lopsided):
    plain = _left_nodes(_draw(lopsided)[1], list("ABCDE"))
    sqrt = _left_nodes(_draw(lopsided, node_scale="sqrt")[1], list("ABCDE"))
    assert sqrt[-1] > plain[-1]
    assert np.all(np.diff(sqrt) < 0)


def test_node_sizing_makes_equal_groups_equal_again(uneven_fanout):
    """The fix for the premise above: size, not fan-out, sets the height."""
    _, ax = _draw(uneven_fanout, min_node_size=6.0)
    wide, narrow, _tiny = _left_nodes(ax, ["wide", "narrow", "tiny"])
    assert wide == pytest.approx(narrow, rel=1e-6)


# ---------------------------------------------------------------- the key --


@pytest.mark.parametrize("kw,title", [
    ({}, None),
    ({"flow_scale": "sqrt"}, "flow"),
    ({"min_flow_width": 0.4}, "flow"),
    ({"min_node_size": 8.0}, "node"),
    ({"node_scale": "sqrt"}, "node"),
    ({"show_size_legend": True}, "flow"),
    ({"min_node_size": 8.0, "show_size_legend": False}, None),
])
def test_one_key_is_drawn_and_it_describes_the_global_map(lopsided, kw, title):
    _, ax = _draw(lopsided, **kw)
    got = _key_title(ax)
    if title is None:
        assert got is None, f"expected no key, got {got!r}"
    else:
        assert got is not None and got.startswith(f"cells per {title}"), got


def test_the_node_key_suppresses_the_flow_key_rather_than_joining_it(lopsided):
    """Two keys would not be clutter, they would be a contradiction."""
    with pytest.warns(UserWarning, match="min_flow_width is not exact"):
        _, ax = _draw(lopsided, min_node_size=8.0, min_flow_width=0.5,
                      flow_scale="sqrt")
    titles = [t.get_text() for t in ax.texts if t.get_text().startswith("cells per ")]
    assert titles == [t for t in titles if t.startswith("cells per node")]
    assert len(titles) == 1, titles


def test_the_node_key_names_the_transform(lopsided):
    _, ax = _draw(lopsided, node_scale="sqrt", min_node_size=6.0)
    title = _key_title(ax)
    assert "square-root" in title and "6 pt floor" in title


def test_the_node_key_bars_are_the_heights_they_claim(lopsided):
    """A key drawn at a different scale from the thing it describes is worse
    than no key: it would make the small end look bigger than it is."""
    _, ax = _draw(lopsided, min_node_size=10.0)
    bars = [p for p in ax.patches if p.get_zorder() == 6]
    assert bars, "no key bars drawn"
    heights = sorted(p.get_height() for p in bars)
    nodes = sorted(_left_nodes(ax, list("ABCDE")))
    # the key spans the observed range, so its extremes sit inside the nodes'
    assert heights[0] >= nodes[0] * 0.9
    assert heights[-1] <= nodes[-1] * 1.1


# --------------------------------------------------------------- refusals --


def test_a_floor_that_cannot_fit_raises_with_the_node_arithmetic(lopsided):
    with pytest.raises(ValueError) as e:
        _draw(lopsided, min_node_size=200.0)
    msg = str(e.value)
    assert "min_node_size" in msg and "5 categories" in msg
    assert "n_nodes * min_node_size" in msg
    assert "node_scale='sqrt'" in msg          # the way out that costs nothing


@pytest.mark.parametrize("kw,match", [
    ({"min_node_size": -1.0}, "min_node_size"),
    ({"node_scale": "cube"}, "node_scale"),
    ({"size_legend_loc": "middle"}, "size_legend_loc"),
])
def test_bad_arguments_raise_before_anything_is_drawn(lopsided, kw, match):
    import matplotlib.pyplot as plt
    before = len(plt.get_fignums())
    with pytest.raises(ValueError, match=match):
        _draw(lopsided, **kw)
    assert len(plt.get_fignums()) == before, "a half-drawn axes was left behind"


# ------------------------------------------------------------- the solver --


def test_solver_puts_the_smallest_node_exactly_on_the_floor():
    sizes = np.array([100.0, 10.0, 1.0])
    h = _solve_node_heights(sizes, avail=1.0, min_axis=0.2, scale="linear")
    assert h.min() == pytest.approx(0.2)
    assert h.sum() == pytest.approx(1.0)
    assert np.all(np.diff(h) < 0)


def test_solver_leaves_a_proportional_rail_alone_when_the_floor_is_slack():
    sizes = np.array([3.0, 2.0, 1.0])
    h = _solve_node_heights(sizes, avail=1.0, min_axis=0.01, scale="linear")
    assert np.allclose(h, sizes / sizes.sum())


def test_an_empty_node_gets_no_height_and_keeps_its_ribbons_at_zero():
    sizes = np.array([10.0, 0.0, 5.0])
    h = _solve_node_heights(sizes, avail=1.0, min_axis=0.0, scale="linear")
    assert h[1] == 0.0
    w = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 0.0]])
    out = _split_node_height(w, h, axis=1)
    assert np.all(out[1] == 0.0)
    assert np.allclose(out.sum(axis=1), h)
