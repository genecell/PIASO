"""The Codebook representative PWMs, and the three shapes they arrive in.

Supplementary Data 1 of Jolma, Laverty, Fathi et al., Nature 657:275-283
(2026): 1,685 motif files for 1,421 human TFs, CC-BY-4.0, served by the
publisher. It roughly doubles the motif universe over JASPAR CORE vertebrates
on human data, which is the reason to have it at all.

Three things in the archive are not what a scanner expects, each found by
reading every file rather than the paper, and each pinned here:

* 34 files name the *construct* assayed (``CASZ1.FL``, ``ZNF729.DBD3``) rather
  than the gene, so they would never match ``var_names``;
* three carry clone identifiers (``AC023509.3``) that are not genes at all;
* five are CIS-BP 1.94d conversions with all-zero positions, which have no
  information and make a log-odds undefined.

The network tests are skipped unless the archive is already cached -- a test
suite that downloads is a test suite that fails on a train.
"""
from __future__ import annotations

import os
import zipfile

import numpy as np
import pytest

from piaso.data import _motifs as M
from piaso.data import build_tf_motif_map, load_codebook


def _pwm_text(tf, motif_id, rows):
    head = f"TF\t{tf}\nMotif\t{motif_id}\nPos\tA\tC\tG\tT\n"
    body = "".join(f"{i + 1}\t" + "\t".join(f"{v}" for v in row) + "\n"
                   for i, row in enumerate(rows))
    return head + body


@pytest.fixture
def archive(tmp_path):
    """A miniature of the real archive, carrying each of its three quirks."""
    files = {
        "SupplementaryData1/SOX9.txt": _pwm_text(
            "SOX9", "M01234_1.94d",
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.25] * 4]),
        # two motifs for one TF, as 264 TFs have
        "SupplementaryData1/PAX6_1.txt": _pwm_text(
            "PAX6", "M02000_1.94d", [[0.7, 0.1, 0.1, 0.1]] * 4),
        "SupplementaryData1/PAX6_2.txt": _pwm_text(
            "PAX6", "M02001_1.94d", [[0.1, 0.7, 0.1, 0.1]] * 5),
        # a construct name, not a gene
        "SupplementaryData1/CASZ1.txt": _pwm_text(
            "CASZ1.FL", "PWM00551", [[0.4, 0.2, 0.2, 0.2]] * 6),
        "SupplementaryData1/ZNF729_3.txt": _pwm_text(
            "ZNF729.DBD3", "PWM00552", [[0.2, 0.4, 0.2, 0.2]] * 6),
        # a clone id, not a gene
        "SupplementaryData1/AC023509.3.txt": _pwm_text(
            "AC023509.3", "PWM00553", [[0.25] * 4] * 5),
        # the CIS-BP conversion defect: an all-zero position, and a row that
        # sums to 0.8 rather than 1
        "SupplementaryData1/PPARG.txt": _pwm_text(
            "PPARG", "M10282_1.94d",
            [[0.9, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0], [0.4, 0.2, 0.1, 0.1]]),
    }
    p = tmp_path / "codebook.zip"
    with zipfile.ZipFile(p, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return str(p)


# ------------------------------------------------------------ the parsing --


def test_it_reads_the_archive_without_unpacking_it(archive):
    pwms = load_codebook(archive)
    assert [p.source for p in pwms] == ["codebook"] * len(pwms)
    assert {p.tf_name for p in pwms} == {"SOX9", "PAX6", "CASZ1", "ZNF729", "PPARG"}


def test_a_tf_with_two_motifs_contributes_both(archive):
    pwms = load_codebook(archive)
    pax6 = [p for p in pwms if p.tf_name == "PAX6"]
    assert len(pax6) == 2
    assert {p.motif_id for p in pax6} == {"M02000_1.94d", "M02001_1.94d"}
    assert sorted(p.width for p in pax6) == [4, 5]


def test_construct_suffixes_are_stripped_so_the_symbol_matches(archive):
    """``CASZ1.FL`` cannot be found in ``var_names``; ``CASZ1`` can."""
    names = {p.tf_name for p in load_codebook(archive)}
    assert "CASZ1" in names and "CASZ1.FL" not in names
    assert "ZNF729" in names and "ZNF729.DBD3" not in names

    verbatim = {p.tf_name for p in load_codebook(archive,
                                                 keep_construct_variants=True)}
    assert "CASZ1.FL" in verbatim and "ZNF729.DBD3" in verbatim


def test_clone_identifiers_are_dropped(archive):
    assert not [p for p in load_codebook(archive)
                if p.tf_name.startswith("AC0")]


def test_zero_positions_are_dropped_and_the_rest_renormalised(archive):
    """An all-zero column carries no information and breaks the log-odds."""
    pparg = next(p for p in load_codebook(archive) if p.tf_name == "PPARG")
    assert pparg.width == 2, "the all-zero position should be gone"
    assert np.allclose(pparg.probs.sum(axis=0), 1.0, atol=1e-5)


def test_every_matrix_is_a_usable_probability_matrix(archive):
    for p in load_codebook(archive):
        assert p.probs.shape[0] == 4 and p.width > 0
        assert np.allclose(p.probs.sum(axis=0), 1.0, atol=1e-5)
        assert np.isfinite(p.pssm(np.array([0.29, 0.21, 0.21, 0.29]))).all()


def test_a_directory_of_files_works_as_well_as_the_zip(archive, tmp_path):
    out = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)
    assert ({p.tf_name for p in load_codebook(str(out))}
            == {p.tf_name for p in load_codebook(archive)})


def test_the_motifs_feed_the_tf_map_like_any_other_source(archive):
    """Case-insensitive remapping is what makes a human DB work on mouse."""
    pwms = load_codebook(archive)
    human = build_tf_motif_map(pwms, tf_list=None,
                               gene_universe=["SOX9", "PAX6", "GAPDH"])
    assert set(human) == {"SOX9", "PAX6"}
    mouse = build_tf_motif_map(pwms, tf_list=None,
                               gene_universe=["Sox9", "Pax6", "Actb"])
    assert set(mouse) == {"Sox9", "Pax6"}, "ALL-CAPS motifs must find Title-case genes"


# ------------------------------------------------------------- the refusals -


def test_a_missing_path_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="Codebook"):
        load_codebook(str(tmp_path / "nope.zip"))


def test_an_archive_with_no_pwms_names_the_fetcher(tmp_path):
    p = tmp_path / "empty.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("readme.md", "nothing here")
    with pytest.raises(ValueError, match="fetch_codebook"):
        load_codebook(str(p))


def test_a_malformed_file_is_skipped_not_fatal(tmp_path):
    """One bad file in 1,685 should not cost the other 1,684."""
    p = tmp_path / "mixed.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("d/SOX9.txt", _pwm_text("SOX9", "M1", [[1.0, 0, 0, 0]]))
        zf.writestr("d/BROKEN.txt", "TF\tBROKEN\nMotif\tM2\nPos\tA\tC\tG\tT\n"
                                    "1\tnot\ta\tnumber\there\n")
    assert {x.tf_name for x in load_codebook(str(p))} == {"SOX9"}


# ------------------------------------------------- the real archive, if cached


CACHED = M._codebook_archive_path()
needs_archive = pytest.mark.skipif(
    not os.path.exists(CACHED),
    reason="run piaso.data.fetch_codebook() once to enable (1 MB)")


@needs_archive
def test_the_real_archive_is_the_size_the_paper_says():
    pwms = load_codebook(CACHED)
    tfs = {p.tf_name for p in pwms}
    assert len(pwms) > 1600, len(pwms)
    assert 1400 <= len(tfs) <= 1425, len(tfs)
    assert not [t for t in tfs if "." in t], "construct suffixes survived"
    widths = [p.width for p in pwms]
    assert min(widths) >= 4 and max(widths) <= 45


@needs_archive
def test_the_real_archive_needs_no_repair_after_loading():
    for p in load_codebook(CACHED):
        assert np.allclose(p.probs.sum(axis=0), 1.0, atol=1e-4), p.motif_id


@needs_archive
def test_it_covers_jaspar_and_then_some():
    """The reason to offer it: about twice the TFs, nearly a superset."""
    from piaso.data import load_jaspar_meme, resolve_jaspar_path

    jaspar_file = resolve_jaspar_path()
    if not os.path.exists(jaspar_file):
        pytest.skip("JASPAR not cached")
    jaspar = set()
    for p in load_jaspar_meme(jaspar_file):
        jaspar.update(t.upper() for t in p.tf_name.split("::"))
    codebook = {p.tf_name.upper() for p in load_codebook(CACHED)}
    assert len(codebook) > len(jaspar) * 1.7
    assert len(jaspar - codebook) < 20, sorted(jaspar - codebook)
