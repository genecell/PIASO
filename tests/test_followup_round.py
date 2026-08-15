"""Follow-up round tests:
- #1 encoder consolidation (importer reuses the lib's encoder).
- #2 fragment_format_version assert (importer + reader panic on incompatible).
- #3 publish.yml builds the Rust extension via maturin (not hatch), version matrix.
"""
import os
import subprocess

import pandas as pd
import pytest

import cytome

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------
# #1 — encoder consolidation
# ---------------------------------------------------------------------

def test_importer_encoder_consolidated():
    """The importer's local encoder.rs is gone; encode functions live in the
    lib's encoder.rs (single owner of the encoding=1 format)."""
    assert not os.path.exists(
        os.path.join(REPO, "src/bin/cytome_import_fragments/encoder.rs")
    ), "importer should reuse the lib encoder, not keep a local copy"
    lib_enc = open(os.path.join(REPO, "src/encoder.rs")).read()
    for fn in ("encode_starts_delta", "encode_ends_length", "encode_cell_indices"):
        assert f"pub fn {fn}" in lib_enc, f"{fn} should be in the lib encoder"
    main_rs = open(os.path.join(REPO, "src/bin/cytome_import_fragments/main.rs")).read()
    assert "use _piaso::encoder" in main_rs


# ---------------------------------------------------------------------
# #2 — fragment_format_version assert
# ---------------------------------------------------------------------

def _binary(name):
    for p in (os.path.join(REPO, "bin", name),
              os.path.join(REPO, "target", "release", name)):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


@pytest.fixture
def tiny_cytome_bad_format(tmp_path):
    """A minimal cytome whose _manifest.format_version is an incompatible major."""
    path = str(tmp_path / "bad.cytome")
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": [0, 1], "barcode": ["AAA", "BBB"],
    }))
    ds.flush()
    ds._conn.execute(
        "INSERT OR REPLACE INTO _manifest(key, value) VALUES('format_version', '\"9.0.0\"')"
    )
    ds._conn.commit()
    ds.close()
    return path


def test_format_assert_importer_append(tiny_cytome_bad_format):
    binary = _binary("cytome-import-fragments")
    if binary is None:
        pytest.skip("cytome-import-fragments binary not built")
    frag = os.path.join(
        REPO, "..", "..", "data", "MouseE18Brain",
        "e18_mouse_brain_fresh_5k_atac_fragments_onlychr.tsv.gz",
    )
    if not os.path.exists(frag):
        frag = "/nonexistent.tsv.gz"  # assert fires before fragments are read
    r = subprocess.run(
        [binary, "--cytome", tiny_cytome_bad_format, "--fragments", frag,
         "--genome", "mm10"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "format_version" in r.stderr and "Incompatible" in r.stderr


def test_format_assert_reader_callpeaks(tiny_cytome_bad_format, tmp_path):
    binary = _binary("piaso-atac")
    if binary is None:
        pytest.skip("piaso-atac binary not built")
    clusters = tmp_path / "cl.tsv"
    clusters.write_text("barcode\tcluster\nAAA\t0\nBBB\t1\n")
    r = subprocess.run(
        [binary, "call-peaks", "--cytome", tiny_cytome_bad_format,
         "--clusters", str(clusters), "--genome", "mm10",
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "format_version" in r.stderr and "Incompatible" in r.stderr


def test_supported_format_major_constant():
    src = open(os.path.join(REPO, "src/cytome_reader.rs")).read()
    assert "pub const SUPPORTED_FORMAT_MAJOR: u32 = 1;" in src
    assert "pub fn assert_compatible_format" in src


# ---------------------------------------------------------------------
# #3 — publish.yml builds the Rust extension via maturin
# ---------------------------------------------------------------------

def test_publish_yml_uses_maturin_matrix():
    import yaml
    path = os.path.join(REPO, ".github/workflows/publish.yml")
    with open(path) as f:
        d = yaml.safe_load(f)
    text = open(path).read()
    assert "hatch build" not in text, "must not use hatch (won't compile Rust)"
    assert "maturin-action" in text, "must build the Rust extension with maturin"
    mat = d["jobs"]["build-wheels"]["strategy"]["matrix"]["python-version"]
    assert set(mat) >= {"3.10", "3.11", "3.12"}
