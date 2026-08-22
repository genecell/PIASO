"""The motif and sequence loaders must ship, and must actually run.

``piaso.pp.scan_motifs`` is public. Until 1.2.0 its *inputs* were not: the
motif-DB loaders and the .2bit sequence access were forwarded to Cytorete, which
is not published, so every one of them raised

    ImportError: piaso.data.load_meme has moved to the Cytorete package --
    `pip install cytorete`

on a command that cannot succeed. A public user could scan, but had no supported
way to obtain a PWM or a sequence to scan with -- the feature was advertised and
unreachable.

These tests pin the two halves of that fix: the names resolve to real modules
rather than a shim, and a PWM survives a write/read round trip into the scanner.
"""
from __future__ import annotations

import numpy as np
import pytest

import piaso


# The loaders that were shim-forwarded before 1.2.0.
MOTIF_API = [
    "load_meme", "load_jaspar_meme", "load_cisbp_meme", "load_cisbp",
    "load_tf_list", "fetch_jaspar", "resolve_jaspar_path", "fetch_cisbp",
    "resolve_cisbp_meme_path", "fetch_cistarget_motifs", "load_cistarget_motifs",
    "resolve_cistarget_paths", "write_meme", "fetch_animaltfdb_tf_list",
    "build_tf_motif_map",
]
SEQUENCE_API = ["fetch_2bit", "resolve_2bit_path", "extract_sequences", "revcomp"]


@pytest.mark.parametrize("name", MOTIF_API + SEQUENCE_API)
def test_loader_is_not_a_shim(name):
    """Resolving to ``piaso._grn_shim`` means it raises for every public user."""
    fn = getattr(piaso.data, name, None)
    assert fn is not None, f"piaso.data.{name} is missing"
    module = getattr(fn, "__module__", "") or ""
    assert "shim" not in module, (
        f"piaso.data.{name} resolves to {module}: it is still forwarded to an "
        "unpublished package, so calling it raises ImportError."
    )
    assert module.startswith("piaso.data"), (
        f"piaso.data.{name} comes from {module}, expected piaso.data.*"
    )


@pytest.mark.parametrize("name", ["loadMotifs", "loadTFList", "buildTFMotifMap",
                                  "fetchJASPAR", "fetchCISBP", "fetchGenomeFasta"])
def test_camelcase_aliases_still_resolve(name):
    """The camelCase aliases are the documented spelling in older notebooks."""
    assert callable(getattr(piaso.data, name, None)), f"piaso.data.{name} is not callable"


def test_write_meme_load_meme_round_trip(tmp_path):
    """A PWM survives the round trip, so the loaders agree on the format."""
    from piaso.data import PWM, write_meme, load_meme

    # probs is (4, width) -- rows are A/C/G/T. Deliberately non-uniform, so a
    # transposed or reordered matrix cannot pass by symmetry.
    probs = np.array([
        [0.70, 0.10, 0.05],   # A
        [0.10, 0.60, 0.05],   # C
        [0.10, 0.20, 0.80],   # G
        [0.10, 0.10, 0.10],   # T
    ], dtype=float)
    pwm = PWM(motif_id="TEST1", tf_name="TESTTF", probs=probs, source="test")

    path = tmp_path / "motifs.meme"
    write_meme([pwm], str(path))
    assert path.exists() and path.stat().st_size > 0

    back = load_meme(str(path))
    assert len(back) == 1
    got = back[0]
    assert got.tf_name == "TESTTF"
    assert got.probs.shape == probs.shape, "width/alphabet axes swapped"
    np.testing.assert_allclose(got.probs, probs, atol=1e-3)


def test_scan_motifs_accepts_a_loaded_pwm(tmp_path):
    """The point of the move: loader output feeds the public scanner.

    This is the workflow that was broken -- not because scan_motifs was absent,
    but because nothing public could produce its `pwms` argument.
    """
    from piaso.data import PWM, write_meme, load_meme

    probs = np.array([
        [0.97, 0.01, 0.01],   # A
        [0.01, 0.97, 0.01],   # C
        [0.01, 0.01, 0.97],   # G
        [0.01, 0.01, 0.01],   # T
    ], dtype=float)
    path = tmp_path / "m.meme"
    write_meme([PWM(motif_id="ACG", tf_name="T", probs=probs, source="test")], str(path))
    pwms = load_meme(str(path))
    assert pwms, "load_meme returned nothing to scan with"

    hits = piaso.pp.scan_motifs(pwms, ["TTTTACGTTTT", "TTTTTTTTTTT"], pvalue=0.01)
    assert isinstance(hits, dict)


def test_revcomp_is_a_real_implementation():
    from piaso.data import revcomp

    assert revcomp("ACGT") == "ACGT"
    assert revcomp("AAAC") == "GTTT"
