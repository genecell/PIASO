"""Regression test for the TSS-QC barcode/delimiter bug (2026-06-03).

`step_tss_enrichment.py` recomputes TSS from raw fragment files when the
cytome has no inline TSS scores. For multi-library imports it must map each
file's RAW barcodes to the cytome's prefixed/suffixed cell barcodes using the
*importer's* delimiter. The old code hardcoded ``"_"`` (and never declared
``--prefix-delimiter``), so a dataset imported with the default ``:`` delimiter
either crashed argparse or silently matched ZERO cells.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "workflow", "scripts", "step_tss_enrichment.py"
)
_spec = importlib.util.spec_from_file_location("step_tss_enrichment", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_map = _mod._file_barcode_mapping


def test_prefix_mode_colon_delimiter_selects_and_strips():
    bl = ["lib1:AAAA", "lib1:CCCC", "lib2:GGGG"]
    raws, retag = _map(bl, 0, prefixes=["lib1", "lib2"], prefix_delimiter=":")
    assert raws == ["AAAA", "CCCC"]            # stripped to raw for the file
    assert retag("AAAA") == "lib1:AAAA"        # re-tagged back to cytome cell
    raws2, retag2 = _map(bl, 1, prefixes=["lib1", "lib2"], prefix_delimiter=":")
    assert raws2 == ["GGGG"]
    assert retag2("GGGG") == "lib2:GGGG"


def test_prefix_mode_dash_delimiter():
    bl = ["lib1-AAAA", "lib2-GGGG"]
    raws, retag = _map(bl, 0, prefixes=["lib1", "lib2"], prefix_delimiter="-")
    assert raws == ["AAAA"]
    assert retag("AAAA") == "lib1-AAAA"


def test_hardcoded_underscore_bug_is_fixed():
    """With the importer's default ':' delimiter, the old hardcoded '_' match
    would have returned an empty list (the silent-wrong-TSS bug)."""
    bl = ["lib1:AAAA", "lib1:CCCC"]
    raws, _ = _map(bl, 0, prefixes=["lib1"], prefix_delimiter=":")
    assert raws == ["AAAA", "CCCC"]            # not empty


def test_suffix_mode():
    bl = ["AAAA-0", "CCCC-0", "GGGG-1"]
    raws, retag = _map(bl, 0, suffixes=["0", "1"], suffix_delimiter="-")
    assert raws == ["AAAA", "CCCC"]
    assert retag("AAAA") == "AAAA-0"


def test_single_file_passthrough():
    bl = ["AAAA", "CCCC"]
    raws, retag = _map(bl, 0)
    assert raws == ["AAAA", "CCCC"]
    assert retag("AAAA") == "AAAA"


def test_argparse_accepts_delimiter_flags():
    """The flags the Snakefile forwards (via fragment_args) must be declared."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, _PATH, "--help"], capture_output=True, text=True
    )
    assert "--prefix-delimiter" in out.stdout
    assert "--suffix-delimiter" in out.stdout
    assert "--barcode-suffixes" in out.stdout
