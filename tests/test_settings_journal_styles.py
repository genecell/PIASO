"""piaso.settings.set_figure_params journal style presets + always-on fonttype 42."""

import matplotlib as mpl
import pytest

import piaso


def test_default_no_style():
    piaso.settings.set_figure_params()
    assert tuple(mpl.rcParams["figure.figsize"]) == (4.0, 4.0)
    assert mpl.rcParams["font.size"] == 12.0
    assert piaso.settings.style is None
    # fonttype 42 always on
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42


def test_nature_methods_preset():
    piaso.settings.set_figure_params(style="nature_methods")
    w, h = mpl.rcParams["figure.figsize"]
    assert abs(w - 89.0 / 25.4) < 1e-6 and abs(h - 89.0 / 25.4) < 1e-6
    assert mpl.rcParams["font.size"] == 7.0
    assert piaso.settings.style == "nature_methods"
    assert mpl.rcParams["pdf.fonttype"] == 42


def test_explicit_args_override_preset():
    piaso.settings.set_figure_params(style="cell", fontsize=9, figsize=(3, 2))
    assert tuple(mpl.rcParams["figure.figsize"]) == (3.0, 2.0)
    assert mpl.rcParams["font.size"] == 9.0


@pytest.mark.parametrize("style", [
    "nature", "cell", "science", "pnas", "elife",
    "bioinformatics", "genome_biology", "nucleic_acids_research",
    "Genome Biology", "Nucleic-Acids-Research",  # normalization
])
def test_all_presets_apply(style):
    piaso.settings.set_figure_params(style=style)
    assert mpl.rcParams["pdf.fonttype"] == 42


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        piaso.settings.set_figure_params(style="vibe")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
