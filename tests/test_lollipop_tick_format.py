"""Lollipop tick labels must carry enough decimals for the axis range."""
from piaso.plotting._plotLigandReceptorLollipop import _lollipop_tick_fmt


def test_small_axes_do_not_collapse_to_zero():
    labels = _lollipop_tick_fmt([-0.0004, -0.0002, 0.0, 0.0002, 0.0004])
    assert labels[0] != "0.0" and labels[-1] == "0.00040"
    assert len(set(labels)) == 3          # symmetric axis: 3 distinct magnitudes


def test_ordinary_axes_keep_one_decimal():
    assert _lollipop_tick_fmt([-0.6, -0.3, 0.0, 0.3, 0.6]) == ["0.6", "0.3", "0.0", "0.3", "0.6"]


def test_all_zero_axis_is_harmless():
    assert _lollipop_tick_fmt([0.0, 0.0]) == ["0.0", "0.0"]
