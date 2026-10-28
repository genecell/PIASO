"""Each lollipop row is labelled with the interaction it draws, in both layouts.

Regression for the audit of 2026-10-27 (part 3, critical 1). With the default
``vertical_layout=True`` the rows were reversed and their labels reversed a
second time, so every row except the middle one named another pair: scores
5/4/3/2 for A/B/C/D drew 2 in the row labelled A. Present since SCALAR's first
commit (71fd16a, 2025-08-27).

The test reads back what a reader sees: each tick label and the length of the
stem drawn at that tick (receptor end minus ligand end equals the score).
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from piaso.plotting import plotLigandReceptorLollipop

SCORES = {"A-->a": 5.0, "B-->b": 4.0, "C-->c": 3.0, "D-->d": 2.0}
ANNOT = {"A-->a": "P1", "B-->b": "P2", "C-->c": "P1", "D-->d": "P2"}


def _frame():
    return pd.DataFrame({
        "ligandXreceptor": list(SCORES),
        "CellTypeXCellType": ["S@R"] * len(SCORES),
        "interaction_score": list(SCORES.values()),
        # unequal specificities, so the two ends of a stem differ
        "ligand_specificity": [0.25, 0.5, 0.75, 0.5],
        "receptor_specificity": [0.75, 0.5, 0.25, 0.5],
        "annotation": [ANNOT[k] for k in SCORES],
    })


def _draw(**kw):
    plt.close("all")
    out = plotLigandReceptorLollipop(_frame(), ["S@R"], **kw)
    fig = out if isinstance(out, plt.Figure) else plt.gcf()
    ax = fig.axes[0]
    return fig, ax


def _stems(ax, vertical):
    """{row position: (low end, high end)} from the stem collections."""
    out = {}
    for coll in ax.collections:
        if type(coll).__name__ != "LineCollection":
            continue
        for seg in coll.get_segments():
            (x0, y0), (x1, y1) = seg
            if vertical and np.isclose(y0, y1):
                out[float(y0)] = (min(x0, x1), max(x0, x1))
            elif not vertical and np.isclose(x0, x1):
                out[float(x0)] = (min(y0, y1), max(y0, y1))
    return out


def test_vertical_layout_labels_name_the_drawn_rows():
    fig, ax = _draw(vertical_layout=True)
    stems = _stems(ax, vertical=True)
    labels = {float(t.get_position()[1]): t.get_text()
              for t in ax.get_yticklabels()}
    assert len(stems) == len(SCORES) and set(labels) == set(stems)
    for pos, label in labels.items():
        lo, hi = stems[pos]
        assert hi - lo == pytest.approx(SCORES[label]), (
            f"row at y={pos} is labelled {label} but draws {hi - lo}")
    # highest score on top
    top = max(labels)
    assert labels[top] == "A-->a"
    plt.close(fig)


def test_horizontal_layout_labels_name_the_drawn_rows():
    fig, ax = _draw(vertical_layout=False)
    stems = _stems(ax, vertical=False)
    texts = {round(float(t.get_position()[0]), 6): t.get_text()
             for t in ax.texts if t.get_text() in SCORES}
    assert len(stems) == len(SCORES) and set(texts) == {round(p, 6) for p in stems}
    for pos, (lo, hi) in stems.items():
        label = texts[round(pos, 6)]
        assert hi - lo == pytest.approx(SCORES[label])
    # highest score on the left
    assert texts[min(texts)] == "A-->a"
    plt.close(fig)


@pytest.mark.parametrize("vertical", [True, False])
def test_label_colours_follow_their_own_row(vertical):
    fig, ax = _draw(vertical_layout=vertical, color_labels_by_annotation=True)
    if vertical:
        ticks = [t for t in ax.get_yticklabels() if t.get_text()]
    else:
        ticks = [t for t in ax.texts if t.get_text() in SCORES]
    colours = {}
    for t in ticks:
        colours.setdefault(ANNOT[t.get_text()], set()).add(
            matplotlib.colors.to_hex(t.get_color()))
    # one colour per annotation, two annotations, two colours
    assert all(len(v) == 1 for v in colours.values())
    assert len({next(iter(v)) for v in colours.values()}) == 2
    plt.close(fig)


def test_duplicate_labels_suffix_the_lower_ranked_row_in_both_layouts():
    df = _frame()
    df.loc[3, "ligandXreceptor"] = "A-->a"          # a repeated pair, score 2
    for vertical in (True, False):
        plt.close("all")
        out = plotLigandReceptorLollipop(df, ["S@R"], vertical_layout=vertical)
        fig = out if isinstance(out, plt.Figure) else plt.gcf()
        ax = fig.axes[0]
        stems = _stems(ax, vertical=vertical)
        if vertical:
            labels = {float(t.get_position()[1]): t.get_text()
                      for t in ax.get_yticklabels()}
        else:
            labels = {float(t.get_position()[0]): t.get_text()
                      for t in ax.texts if t.get_text().startswith(("A", "B", "C"))}
        by_label = {lab: stems[pos][1] - stems[pos][0] for pos, lab in labels.items()}
        assert by_label["A-->a"] == pytest.approx(5.0)
        assert by_label["A-->a (2)"] == pytest.approx(2.0)
        plt.close(fig)


# --- finding 17 of review I2: the size key and the score-range limits -------

def _frame_with_fc():
    df = _frame()
    df["avg_log2FC"] = [-1.0, -0.5, 0.5, 1.0]
    return df


def test_size_key_dots_are_the_plotted_size():
    """The key drew markersize 2*sqrt(s/pi) for a scatter of area s, so its
    dots were 1.13x wider than the plotted ones; a marker of area s has
    markersize sqrt(s)."""
    from piaso.plotting._plotLigandReceptorLollipop import _calculate_circle_size
    plt.close("all")
    out = plotLigandReceptorLollipop(_frame_with_fc(), ["S@R"])
    fig = out if isinstance(out, plt.Figure) else plt.gcf()
    key = [lg for lg in fig.legends if lg.get_title().get_text() == "avg_log2FC"][0]
    for handle, text in zip(key.legend_handles, key.get_texts()):
        s = _calculate_circle_size(float(text.get_text()), 15, 1.5, 1.0)
        assert handle.get_markersize() == pytest.approx(np.sqrt(s), rel=1e-6)
    # and the same stroke as the plotted dots, so the drawn sizes agree
    dots = [c for c in fig.axes[0].collections if type(c).__name__ == "PathCollection"]
    stroke = {float(lw) for c in dots for lw in c.get_linewidths()}
    assert {h.get_markeredgewidth() for h in key.legend_handles} == stroke
    plt.close(fig)


def test_circle_size_scale_is_reported_as_ineffective():
    with pytest.warns(UserWarning, match="circle_size_scale"):
        _draw(circle_size_scale=40)


@pytest.mark.parametrize("vertical", [True, False])
def test_one_sided_upper_limit_clips_both_ends(vertical):
    """score_range_max alone clipped the receptor end only (and the ligand
    end only in the horizontal layout)."""
    fig, ax = _draw(vertical_layout=vertical, score_range_max=2.0)
    stems = _stems(ax, vertical=vertical)
    for lo, hi in stems.values():
        assert abs(lo) <= 2.0 + 1e-9 and abs(hi) <= 2.0 + 1e-9
    # the 5-score row is 1.25 / 3.75 before clipping: its long end is cut
    assert any(np.isclose(max(abs(lo), abs(hi)), 2.0) for lo, hi in stems.values())
    plt.close(fig)


@pytest.mark.parametrize("vertical", [True, False])
def test_one_sided_lower_limit_lifts_both_ends(vertical):
    # ligand ends 1.25, 2, 2.25, 1 and receptor ends 3.75, 2, 0.75, 1: a
    # floor of 1.5 has to lift ends on both sides
    fig, ax = _draw(vertical_layout=vertical, score_range_min=1.5)
    for lo, hi in _stems(ax, vertical=vertical).values():
        assert abs(lo) >= 1.5 - 1e-9 and abs(hi) >= 1.5 - 1e-9
    plt.close(fig)


@pytest.mark.parametrize("vertical", [True, False])
def test_two_sided_limits_unchanged(vertical):
    fig, ax = _draw(vertical_layout=vertical, score_range_min=1.0, score_range_max=2.0)
    for lo, hi in _stems(ax, vertical=vertical).values():
        for v in (lo, hi):
            assert 1.0 - 1e-9 <= abs(v) <= 2.0 + 1e-9
    plt.close(fig)
