"""KAMON is the marker-based naming method's name (2026-10-28): ``piaso.tl.kamon`` is the function, and
``piaso.tl.predictCellTypeByMarker``, its earlier descriptive name, is the same object, kept for every existing
script. The API reference documents it once, under kamon, with the other name listed."""
import importlib.util
from pathlib import Path

import pytest

import piaso

# The API reference and its generator are the site's sources (web/), which the public tree does not ship.
_WEB = Path(__file__).resolve().parents[1] / "web"
needs_site = pytest.mark.skipif(not (_WEB / "scripts" / "build_api_pages.py").exists(),
                                reason="the site's sources (web/) are not in this tree")


def test_kamon_and_predictCellTypeByMarker_are_one_function():
    assert piaso.tl.kamon is piaso.tl.predictCellTypeByMarker
    assert piaso.tl.kamon.__name__ == "kamon"
    assert piaso.tools.kamon is piaso.tl.kamon
    assert "predictCellTypeByMarker" in piaso.tl.kamon.__doc__


@needs_site
def test_the_api_page_documents_kamon_once_and_lists_the_other_name():
    page = (_WEB / "src/content/docs/api/tools.md").read_text()
    assert "\n## kamon\n" in page
    assert "\n## predictCellTypeByMarker\n" not in page
    assert "Also available as `predictCellTypeByMarker`: the same function." in page
    assert "| [`predictCellTypeByMarker`](#kamon) | The same function as `kamon`. |" in page


@needs_site
def test_the_generator_lists_an_alias_under_its_function():
    gen = _WEB / "scripts" / "build_api_pages.py"
    spec = importlib.util.spec_from_file_location("build_api_pages", gen)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    documented, _, aliases = mod.collect("tl", mod.held_modules())
    names = [n for n, _ in documented]
    assert "kamon" in names and "predictCellTypeByMarker" not in names
    assert aliases.get("kamon") == ["predictCellTypeByMarker"]
