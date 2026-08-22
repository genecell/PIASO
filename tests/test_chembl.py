"""ChEMBL drug-target gene sets in ``piaso.data``.

The filter reproduces drug2cell's `filter_activities` so the gene sets can be
built with pandas alone. Two of its steps are easy to get subtly wrong and are
what these tests pin:

* `add_drug_mechanism` and `include_active` do NOT remove rows — they mark rows
  as protected, so a later assay-type or potency cut cannot drop them. Getting
  that backwards silently shrinks the dictionary.
* the pChEMBL cut-off is per target class, because 30 nM is unremarkable for a
  kinase and exceptional for an ion channel. A single global threshold would
  keep the wrong compounds for half the classes.

The real table is 2.7 GB, so these run on a fixture carrying one row per branch.
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
piaso = pytest.importorskip("piaso")

from piaso.data import _chembl as C  # noqa: E402


def _row(**kw):
    base = {
        C.COL_MAX_PHASE: 4,
        C.COL_CHEMBL_ID: "CHEMBL1",
        C.COL_PREF_NAME: "DRUG A",
        C.COL_MECHANISM: None,
        C.COL_ASSAY_TYPE: "F",
        C.COL_ACTIVITY_COMMENT: None,
        C.COL_PCHEMBL: 9.0,
        C.COL_SYNONYMS: "GENE1",
        "target_class": "none",
    }
    base.update(kw)
    return base


@pytest.fixture
def table():
    return pd.DataFrame([
        _row(),                                                  # 0 plain pass
        _row(**{C.COL_MAX_PHASE: 2}),                            # 1 preclinical
        _row(**{C.COL_ASSAY_TYPE: "B"}),                         # 2 wrong assay
        _row(**{C.COL_ASSAY_TYPE: "B", C.COL_MECHANISM: 42}),    # 3 protected
        _row(**{C.COL_ACTIVITY_COMMENT: "Not Active"}),          # 4 inactive
        _row(**{C.COL_PCHEMBL: 3.0}),                            # 5 too weak
        _row(**{C.COL_PCHEMBL: 3.0,
                C.COL_ACTIVITY_COMMENT: "Active"}),              # 6 protected
        _row(**{C.COL_PCHEMBL: 6.5, "target_class": "Kinase"}),  # 7 weak kinase
        _row(**{C.COL_PCHEMBL: 6.5, "target_class": "Ion Channel"}),  # 8 ok
    ])


def _kept(df):
    return set(df.index)


def test_the_default_filters_keep_only_the_confident_rows(table):
    out = C.filter_chembl_activities(table, verbose=False)
    # 0 passes everything; 3 and 6 are protected; 8 clears the ion-channel bar.
    assert _kept(out) == {0, 3, 6, 8}


def test_a_curated_mechanism_survives_the_wrong_assay_type(table):
    """Row 3 is a binding assay, which the assay filter drops — unless it
    carries a mechanism of action, which is better evidence than the assay."""
    with_protection = C.filter_chembl_activities(table, verbose=False)
    without = C.filter_chembl_activities(table, add_drug_mechanism=False,
                                         verbose=False)
    assert 3 in _kept(with_protection)
    assert 3 not in _kept(without)


def test_an_active_comment_survives_a_failed_potency_cut(table):
    """Row 6 has pChEMBL 3.0, far under the 6.0 bar, but is called Active."""
    with_protection = C.filter_chembl_activities(table, verbose=False)
    without = C.filter_chembl_activities(table, include_active=False,
                                         verbose=False)
    assert 6 in _kept(with_protection)
    assert 6 not in _kept(without)


def test_the_potency_cut_is_per_target_class(table):
    """Rows 7 and 8 have the *same* pChEMBL, 6.5, and different fates.

    A kinase must reach 7.53, an ion channel only 5.0. A single global cut-off
    would keep both or drop both, and either way be wrong for one of them.
    """
    out = C.filter_chembl_activities(table, verbose=False)
    assert 8 in _kept(out), "6.5 clears the 5.0 ion-channel bar"
    assert 7 not in _kept(out), "6.5 misses the 7.53 kinase bar"

    flat = C.filter_chembl_activities(table, pchembl_threshold=6.0,
                                      verbose=False)
    assert {7, 8} <= _kept(flat), "a flat 6.0 cut-off keeps both"


def test_an_unknown_target_class_falls_back_rather_than_raising(table):
    """drug2cell raises when the column has a class the dict lacks. A table
    whose vocabulary has drifted is not a reason to fail the whole build."""
    t = table.copy()
    t.loc[0, "target_class"] = "Transporter"       # not in PCHEMBL_THRESHOLDS
    out = C.filter_chembl_activities(t, default_pchembl=6.0, verbose=False)
    assert 0 in _kept(out), "pChEMBL 9.0 clears the 6.0 fallback"

    strict = C.filter_chembl_activities(t, default_pchembl=9.5, verbose=False)
    assert 0 not in _kept(strict), "the fallback is actually applied"


def test_max_phase_selects_clinical_stage(table):
    out = C.filter_chembl_activities(table, drug_max_phase=2, verbose=False)
    assert _kept(out) == {1}
    both = C.filter_chembl_activities(table, drug_max_phase=[2, 4], verbose=False)
    assert 1 in _kept(both) and 0 in _kept(both)


def test_the_dictionary_splits_and_dedupes_synonyms():
    df = pd.DataFrame([
        _row(**{C.COL_SYNONYMS: "ADRB1|ADRB2"}),
        _row(**{C.COL_SYNONYMS: "ADRB2"}),          # duplicate across rows
        _row(**{C.COL_CHEMBL_ID: "CHEMBL2", C.COL_PREF_NAME: "DRUG B",
                C.COL_SYNONYMS: "HTR2A"}),
    ])
    d = C.chembl_targets_to_dict(df)
    assert d["CHEMBL1|DRUG A"] == ["ADRB1", "ADRB2"]
    assert d["CHEMBL2|DRUG B"] == ["HTR2A"]


def test_a_missing_target_class_column_says_what_to_do(table):
    with pytest.raises(KeyError, match="single float threshold"):
        C.filter_chembl_activities(table, pchembl_target_column="nope",
                                   verbose=False)


def test_resolve_returns_none_when_nothing_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path))
    assert C.resolve_chembl_path(data_dir=str(tmp_path)) is None


def test_the_public_names_are_exported():
    assert piaso.data.load_chembl_targets is C.load_chembl_targets
    assert piaso.data.fetch_chembl is C.fetch_chembl
    assert piaso.data.loadChEMBLTargets is C.load_chembl_targets
    assert piaso.data.PCHEMBL_THRESHOLDS["Kinase"] == 7.53
