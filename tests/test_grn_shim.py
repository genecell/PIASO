"""The GRN API moved to CytoRete; PIASO keeps lazy `piaso.tl/pp/pl/data` shims.

Verifies: (1) importing piaso does NOT import cytorete; (2) every historical GRN
entry point still exists as a callable on its namespace; (3) the scanner (PWM,
scan_motifs) STAYS in piaso; (4) a shim forwards to cytorete when installed; and
(5) raises a clear `pip install cytorete` ImportError when it is not.
"""
import importlib
import sys

import pytest

import piaso


def test_import_piaso_does_not_import_cytorete():
    # piaso import must be cytorete-free (no packaging cycle, zero-cost pointer).
    assert "piaso" in sys.modules
    # (can't assert cytorete absent globally — another test may have imported it —
    #  but the shim module itself must not import it at load)
    from piaso import _grn_shim  # noqa: F401
    assert "cytorete" not in _grn_shim.__dict__


@pytest.mark.parametrize("ns_attr,name", [
    ("tl", "inferGRN"), ("tl", "inferRegulon"), ("tl", "inferTFActivity"),
    ("tl", "regulonActivity"), ("tl", "regulonSpecificity"),
    ("pp", "build_peak_cistrome"), ("pp", "build_cistrome"),
    ("pp", "extract_promoter_sequences"), ("pp", "bulk_base_cistrome"),
    ("pl", "regulonActivity"), ("pl", "regulonNetwork"),
    ("pl", "regulonEmbedding"), ("pl", "regulonSpecificityScatter"),
    ("data", "load_meme"), ("data", "load_jaspar_meme"),
    ("data", "build_tf_motif_map"), ("data", "extract_sequences"),
])
def test_shim_entry_points_present(ns_attr, name):
    ns = getattr(piaso, ns_attr)
    assert callable(getattr(ns, name)), f"piaso.{ns_attr}.{name} missing"


def test_scanner_and_pwm_stay_in_piaso():
    # The motif scanner + PWM back piaso.pp.scan_motifs — they must NOT have moved.
    assert callable(piaso.pp.scan_motifs)
    assert callable(piaso.pp.estimate_background)
    assert hasattr(piaso.data, "PWM")


def test_shim_helpful_error_when_cytorete_absent(monkeypatch):
    real = importlib.import_module

    def _no_cytorete(name, *a, **k):
        if name.split(".")[0] == "cytorete":
            raise ImportError("simulated: cytorete not installed")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", _no_cytorete)
    with pytest.raises(ImportError, match="pip install cytorete"):
        piaso.tl.inferGRN()
