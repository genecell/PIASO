"""Two axes, one table, and the geometry that joins them.

Ribbons and blocks were placed in opposite vertical conventions: the blocks
went through `_rail_to_xy`, which flips because matplotlib's y grows upward,
while the ribbons were built from the raw cursor. The flow carrying the
largest category's height was drawn where the smallest category's block was.
Every Sankey on the site was wrong in the same way, and it is invisible unless
you check a case whose answer you already know.

Ordering: SVD orders the two axes *together*, so it cannot honour an axis you
have fixed. With one side pinned the conditional answer is the barycentre --
each free category at the mean position of its mass along the fixed axis.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.patches as mpatches


SIZES = {"A": 400, "B": 200, "C": 100, "D": 50}


@pytest.fixture
def one_to_one():
    """A perfect diagonal with very unequal class sizes."""
    anndata = pytest.importorskip("anndata")
    lab = sum([[k] * v for k, v in SIZES.items()], [])
    return anndata.AnnData(
        X=np.zeros((len(lab), 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"L": pd.Categorical(lab, categories=list(SIZES)),
             "R": pd.Categorical(lab, categories=list(SIZES))},
            index=[f"c{i}" for i in range(len(lab))]),
    )


def _blocks_and_ribbons(ax, layout):
    rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
    if layout == "vertical":
        blocks = {(round(p.get_x(), 3), round(p.get_x() + p.get_width(), 3))
                  for p in rects if p.get_y() > 0.5}
        along, across, pick = 0, 1, max
    else:
        blocks = {(round(p.get_y(), 3), round(p.get_y() + p.get_height(), 3))
                  for p in rects if p.get_x() < 0.5}
        along, across, pick = 1, 0, min
    spans = set()
    for r in ax.patches:
        if not isinstance(r, mpatches.PathPatch):
            continue
        v = r.get_path().vertices
        edge = v[:, along][np.isclose(v[:, across], pick(v[:, across]), atol=1e-9)]
        spans.add((round(edge.min(), 3), round(edge.max(), 3)))
    return blocks, spans


@pytest.mark.parametrize("layout", ["horizontal", "vertical"])
def test_every_ribbon_lands_on_its_block(one_to_one, layout):
    """A 1:1 mapping must join each block to itself, at its own height."""
    import piaso

    out = piaso.pl.sankey(one_to_one, left="L", right="R", layout=layout,
                          show=False, return_fig=True)
    ax = (out[1] if isinstance(out, tuple) else out.axes[0])
    blocks, spans = _blocks_and_ribbons(ax, layout)

    assert len(spans) == len(SIZES), f"{layout}: {len(spans)} ribbons"
    for s in spans:
        assert any(abs(b[0] - s[0]) < 2e-3 and abs(b[1] - s[1]) < 2e-3
                   for b in blocks), (
            f"{layout}: ribbon {s} matches no block; blocks {sorted(blocks)}")


def test_barycentre_follows_a_fixed_axis():
    """Right side ordered to align with a left side that is already decided."""
    from piaso.plotting._seriation import barycentre_order

    # L0->rE, L1->rC, L2->rA: the aligning order is E, C, A
    ct = pd.DataFrame([[0, 0, 9], [0, 9, 0], [9, 0, 0]],
                      index=["L0", "L1", "L2"], columns=["rA", "rC", "rE"])
    assert barycentre_order(ct, ["L0", "L1", "L2"], axis=0) == ["rE", "rC", "rA"]


def test_svd_orders_both_axes_together():
    from piaso.plotting._seriation import svd_order

    ct = pd.DataFrame([[9, 0, 0], [0, 9, 0], [0, 0, 9]],
                      index=["a", "b", "c"], columns=["x", "y", "z"])
    rows, cols = svd_order(ct)
    assert sorted(rows) == ["a", "b", "c"] and sorted(cols) == ["x", "y", "z"]
    # the diagonal must survive: row i pairs with col i in both orders
    assert [rows.index(r) for r in ("a", "b", "c")] == \
           [cols.index(c) for c in ("x", "y", "z")]


def test_auto_right_aligns_with_an_explicit_left():
    """The whole point: pin the left, and the right follows it."""
    import piaso
    anndata = pytest.importorskip("anndata")

    rs = np.random.RandomState(0)
    n = 600
    left = [f"L{i}" for i in rs.randint(0, 4, n)]
    m = {"L0": "rD", "L1": "rB", "L2": "rA", "L3": "rC"}
    right = [m[l] for l in left]
    a = anndata.AnnData(
        X=np.zeros((n, 2), dtype=np.float32),
        obs=pd.DataFrame({"L": pd.Categorical(left), "R": pd.Categorical(right)},
                         index=[f"c{i}" for i in range(n)]))

    out = piaso.pl.sankey(a, left="L", right="R", right_order="auto",
                          show=False, return_fig=True)
    ax = (out[1] if isinstance(out, tuple) else out.axes[0])
    drawn = [t.get_text() for t in ax.texts if t.get_text().startswith("r")]
    assert drawn == ["rD", "rB", "rA", "rC"], drawn


def test_confusion_matrix_takes_an_explicit_order():
    import inspect
    import piaso

    sig = inspect.signature(piaso.pl.plotConfusionMatrix)
    assert "order_query" in sig.parameters
    assert "order_reference" in sig.parameters


# ---------------------------------------------------------------------------
# The 1:1 test above joins each block to itself, which catches a block drawn at
# the wrong height but not a cursor that drifts: with many-to-many flows each
# block is filled by several ribbons stacked in turn, and the stack has to fill
# it exactly. If the ribbons are scaled against a different denominator than
# the blocks, they overflow into the gaps -- which is what "the edges are in
# the wrong place" looks like on a real figure.
# ---------------------------------------------------------------------------

_NODE_WIDTH = 0.08


def _tiling_error(ax, side):
    """max |sum of ribbon heights - block height| over the blocks on `side`."""
    import matplotlib.patches as mp

    block_x = -_NODE_WIDTH / 2 if side == "left" else 1 - _NODE_WIDTH / 2
    edge_x = _NODE_WIDTH / 2 if side == "left" else 1 - _NODE_WIDTH / 2

    blocks = [(p.get_y(), p.get_y() + p.get_height()) for p in ax.patches
              if isinstance(p, mp.Rectangle) and abs(p.get_x() - block_x) < 1e-9]
    edges = []
    for p in ax.patches:
        if not isinstance(p, mp.PathPatch):
            continue
        v = p.get_path().vertices
        m = np.isclose(v[:, 0], edge_x, atol=1e-9)
        if m.sum() >= 2:
            edges.append((v[m, 1].min(), v[m, 1].max()))

    worst = 0.0
    for b0, b1 in blocks:
        covered = sum(e[1] - e[0] for e in edges
                      if e[0] >= b0 - 1e-9 and e[1] <= b1 + 1e-9)
        worst = max(worst, abs(covered - (b1 - b0)))
    return worst, len(blocks), len(edges)


def _many_to_many(seed=0, n=1500, n_left=15, n_right=10, unused_levels=0,
                  nan_every=0):
    anndata = pytest.importorskip("anndata")
    rs = np.random.RandomState(seed)
    left = np.array([str(i) for i in rs.randint(0, n_left, n)], dtype=object)
    right = [f"T{i}" for i in rs.randint(0, n_right, n)]
    if nan_every:
        left[::nan_every] = None
    levels = [f"T{i}" for i in range(n_right + unused_levels)]
    obs = pd.DataFrame({"L": list(left),
                        "R": pd.Categorical(right, categories=levels)},
                       index=[f"c{i}" for i in range(n)])
    return anndata.AnnData(X=np.zeros((n, 2), dtype=np.float32), obs=obs)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "kind,kwargs",
    [("plain", {}),
     ("unused categorical levels", {"unused_levels": 8}),
     ("missing pairs", {"n_left": 20, "n_right": 25, "n": 400}),
     ("NaN labels", {"nan_every": 37})],
)
def test_ribbons_tile_their_block_exactly(side, kind, kwargs):
    import piaso

    adata = _many_to_many(**kwargs)
    out = piaso.pl.sankey(adata, left="L", right="R", show=False, return_fig=True)
    ax = (out[1] if isinstance(out, tuple) else out.axes[0])
    err, n_blocks, n_edges = _tiling_error(ax, side)
    assert n_blocks and n_edges, f"{kind}/{side}: nothing drawn"
    assert err < 1e-6, (
        f"{kind}/{side}: a block is over/under-filled by {err:.6f} -- the "
        f"ribbons are scaled against a different total than the blocks")
