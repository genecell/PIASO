"""matplotlib compatibility regression tests.

piaso/plotting/color.py builds colormaps at import time; an unguarded
matplotlib.cm.get_cmap call there (removed upstream in newer matplotlib)
makes `import piaso` fail outright, not just a plot call.
"""

import importlib
import inspect
import sys

import matplotlib
import matplotlib.cm
import pytest


def test_color_module_imports_without_cm_get_cmap(monkeypatch):
    """import must survive a matplotlib without cm.get_cmap."""
    monkeypatch.delattr(matplotlib.cm, "get_cmap", raising=False)
    import piaso.plotting.color as color_mod
    importlib.reload(color_mod)
    assert color_mod.c_color1.N == 256
    assert color_mod.c_color2.N == 256
    assert color_mod.c_color3.N == 256


def test_no_module_level_cm_get_cmap_calls():
    """Guard against reintroducing module-level cm.get_cmap calls."""
    import piaso.plotting.color as color_mod
    src = inspect.getsource(color_mod)
    for line in src.splitlines():
        code = line.split("#")[0]
        if "cm.get_cmap(" in code:
            # only the guarded fallback inside _get_cmap may call it
            assert "return cm.get_cmap" in code, (
                f"unguarded cm.get_cmap call: {line!r}"
            )
