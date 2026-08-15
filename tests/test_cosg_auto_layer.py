"""Unit tests for the modality-resolving `layer='auto'` default in run_cosg_cytome.

RNA/GA → log1p, ATAC/tiles → tfidf; explicit layers pass through unchanged;
unknown modality under 'auto' raises. `import cosg` must stay piaso-free.
"""
import subprocess
import sys

import pytest


def test_auto_resolves_by_modality():
    from cosg._cytome_streaming import _resolve_auto_layer
    assert _resolve_auto_layer("auto", "RNA") == "log1p"
    assert _resolve_auto_layer("auto", "GA") == "log1p"
    assert _resolve_auto_layer("auto", "ATAC") == "tfidf"
    assert _resolve_auto_layer("auto", "tiles") == "tfidf"


def test_explicit_layer_passthrough():
    from cosg._cytome_streaming import _resolve_auto_layer
    for lyr in ("counts", "log1p", "infog", "tfidf", "RNA_infog"):
        assert _resolve_auto_layer(lyr, "RNA") == lyr
        assert _resolve_auto_layer(lyr, "ATAC") == lyr


def test_auto_unknown_modality_raises():
    from cosg._cytome_streaming import _resolve_auto_layer
    with pytest.raises(ValueError, match="no default normalization"):
        _resolve_auto_layer("auto", "not_a_modality")


def test_default_layer_is_auto():
    import inspect
    from cosg._cytome_streaming import run_cosg_cytome
    assert inspect.signature(run_cosg_cytome).parameters["layer"].default == "auto"


def test_import_cosg_is_piaso_free():
    """`import cosg` must not import piaso (piaso is imported lazily, per-call)."""
    code = "import sys, cosg; assert 'piaso' not in sys.modules, sorted(m for m in sys.modules if 'piaso' in m)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"`import cosg` pulled in piaso:\n{r.stderr}"
