"""The ligand-receptor heatmap labels the row it colours, in every mode.

Regression for the audit of 2026-10-27 (part 3, critical 2). With one colormap
(the default) the horizontal layout's image extent put the ligand values in
the bottom row, while the two-colormap branch put them on top; the labels,
the same in both, read "Ligand" above "Receptor". The labels also sat below
the heatmap rather than beside their rows.

Each test reads what a viewer sees: the pixel colour at the centre of the row
(or column) a label sits on, against the colour of the value it names.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from piaso.plotting import plotLigandReceptorInteraction

# distinct ligand and receptor specificities per row, so a swap shows
LIG = {"L1": 1.0, "L2": 0.6, "L3": 0.2}
REC = {"R1": 0.0, "R2": 0.8, "R3": 0.4}


def _inputs():
    inter = pd.DataFrame({
        "ligandXreceptor": ["L1-->R1", "L2-->R2", "L3-->R3"],
        "CellTypeXCellType": ["S@R"] * 3,
        "interaction_score": [3.0, 2.0, 1.0],
        "annotation": ["P1", "P2", "P1"],
    })
    spec = pd.DataFrame(0.0, index=list(LIG) + list(REC), columns=["S", "R"])
    for g, v in LIG.items():
        spec.loc[g, "S"] = v
    for g, v in REC.items():
        spec.loc[g, "R"] = v
    return inter, spec


def _draw(**kw):
    plt.close("all")
    inter, spec = _inputs()
    out = plotLigandReceptorInteraction(
        inter, spec, ["S@R"], y_max=4, fig_width=10, fig_height_per_pair=6, **kw)
    fig = out if isinstance(out, plt.Figure) else plt.gcf()
    fig.canvas.draw()
    hm = [a for a in fig.axes if a.images][0]
    return fig, hm


def _pixel(fig, ax, x, y):
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h = rgba.shape[0]
    px, py = ax.transData.transform((x, y))
    return rgba[int(round(h - py)), int(round(px)), :3] / 255.0


def _expected(cmap, value, vmax):
    return np.asarray(matplotlib.colormaps[cmap](value / vmax)[:3])


def _label_pos(ax, text, axis):
    """Data coordinate (along ``axis``) of the label reading ``text``."""
    if axis == "y":
        for t in ax.get_yticklabels():
            if t.get_text() == text:
                return float(t.get_position()[1])
    else:
        for t in ax.get_xticklabels():
            if t.get_text() == text:
                return float(t.get_position()[0])
    for t in ax.texts:
        if t.get_text() == text:
            return float(t.get_position()[1 if axis == "y" else 0])
    raise AssertionError(f"no label {text!r}")


MODES = {
    "one colormap": dict(kw={}, lig="Purples", rec="Purples"),
    "two colormaps": dict(kw=dict(heatmap_cmap_ligand="Blues",
                                  heatmap_cmap_receptor="Reds"),
                          lig="Blues", rec="Reds"),
}


@pytest.mark.parametrize("mode", list(MODES))
def test_horizontal_layout_rows_match_their_labels(mode):
    m = MODES[mode]
    fig, hm = _draw(vertical_layout=False, **m["kw"])
    vmax = max(max(LIG.values()), max(REC.values()))
    y_lig = _label_pos(hm, "Ligand", "y")
    y_rec = _label_pos(hm, "Receptor", "y")
    # a label sits at the centre of a row, ligand above receptor
    assert y_lig == pytest.approx(1.5) and y_rec == pytest.approx(0.5)
    # interaction i is drawn in column i (ranked by score: L1, L2, L3)
    for i, (lig, rec) in enumerate(zip(LIG, REC)):
        got_l = _pixel(fig, hm, i, y_lig)
        got_r = _pixel(fig, hm, i, y_rec)
        np.testing.assert_allclose(got_l, _expected(m["lig"], LIG[lig], vmax), atol=0.02)
        np.testing.assert_allclose(got_r, _expected(m["rec"], REC[rec], vmax), atol=0.02)
    plt.close(fig)


@pytest.mark.parametrize("mode", list(MODES))
def test_vertical_layout_columns_match_their_labels(mode):
    m = MODES[mode]
    fig, hm = _draw(vertical_layout=True, **m["kw"])
    vmax = max(max(LIG.values()), max(REC.values()))
    x_lig = _label_pos(hm, "Ligand", "x")
    x_rec = _label_pos(hm, "Receptor", "x")
    col_lig = int(np.floor(x_lig + 0.5))          # column [-0.5, 0.5) is 0
    col_rec = int(np.floor(x_rec + 0.5))
    assert (col_lig, col_rec) == (0, 1)
    # vertical layout: rows bottom-up, the top row is the highest score
    labels_ax = [a for a in fig.axes if a is not hm and any(
        t.get_text() in ("L1-->R1", "L2-->R2", "L3-->R3") for t in a.texts)][0]
    rows = {t.get_text(): float(t.get_position()[1]) for t in labels_ax.texts
            if t.get_text() in ("L1-->R1", "L2-->R2", "L3-->R3")}
    for pair, y in rows.items():
        lig, rec = pair.split("-->")
        np.testing.assert_allclose(_pixel(fig, hm, col_lig, y),
                                   _expected(m["lig"], LIG[lig], vmax), atol=0.02)
        np.testing.assert_allclose(_pixel(fig, hm, col_rec, y),
                                   _expected(m["rec"], REC[rec], vmax), atol=0.02)
    assert rows["L1-->R1"] > rows["L3-->R3"]
    plt.close(fig)


def test_shared_legend_with_fixed_vmax_has_a_finite_colorbar():
    """shared_legend + heatmap_vmax left the colorbar's ticks at -inf."""
    for vertical in (False, True):
        fig, hm = _draw(vertical_layout=vertical, shared_legend=True, heatmap_vmax=2.0)
        def _all_axes(axes):
            for a in axes:
                yield a
                yield from _all_axes(a.child_axes)

        cbars = [a._colorbar for a in _all_axes(fig.axes) if hasattr(a, "_colorbar")]
        assert cbars, "no colorbar drawn"
        for cb in cbars:
            ticks = np.asarray(cb.get_ticks(), dtype=float)
            assert ticks.size and np.all(np.isfinite(ticks)), ticks
            assert cb.norm.vmax == pytest.approx(2.0)
            assert ticks.max() == pytest.approx(2.0)
        plt.close(fig)


def test_preserve_input_order_shared_scale_covers_the_plotted_rows():
    """With preserve_input_order the shared vmax came from the nlargest rows,
    not the rows drawn, so a drawn value above it was clipped silently."""
    inter, spec = _inputs()
    # the first row in input order has the LOWEST score but the highest
    # specificity; top_n=1 with preserve_input_order draws only it
    inter = inter.iloc[::-1].reset_index(drop=True)
    spec.loc["L3", "S"] = 5.0
    plt.close("all")
    out = plotLigandReceptorInteraction(
        inter, spec, ["S@R"], top_n=1, preserve_input_order=True,
        shared_legend=True, y_max=4, fig_width=10, fig_height_per_pair=6)
    fig = out if isinstance(out, plt.Figure) else plt.gcf()
    hm = [a for a in fig.axes if a.images][0]
    assert hm.images[0].norm.vmax == pytest.approx(5.0)
    plt.close(fig)
