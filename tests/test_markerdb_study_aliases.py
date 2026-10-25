"""A renamed study's former name keeps working in analyzeGenes and analyzeMarkers.

PIASOmarkerDB keeps every former study name as an alias, and its API answers them, but the client checked
``studies=`` against the list of current names before asking, so a former name stopped there. The client now
resolves names through the server's ``/api/v1/study-aliases`` first (any case), with one warning per renamed name,
and resolves ``exclude_studies`` the same way. A server without the endpoint leaves the names as given.
"""
from __future__ import annotations

import warnings

import pytest

from piaso.tools import _markerdb

STUDIES = [{"study_publication": "SEAAD2024_MTG_subclass"}, {"study_publication": "DiBellaArlotta2021_Cortex"}]
ALIASES = [{"old_name": "SEAAD2024_MTG_Subclass", "new_name": "SEAAD2024_MTG_subclass"},
           {"old_name": "DiBellaArlotta2021", "new_name": "DiBellaArlotta2021_Cortex"}]


@pytest.fixture
def make(monkeypatch, tmp_path):
    def build(alias_status=None):
        c = _markerdb.PIASOmarkerDB(cache_dir=tmp_path / "cache")
        calls, seen = [], {}

        def request(method, endpoint, params=None, **kw):
            calls.append(endpoint)
            if endpoint == "studies":
                return {"data": STUDIES}
            if endpoint == "study-aliases":
                if alias_status:
                    raise _markerdb.APIError("not found", status_code=alias_status)
                return ALIASES
            raise AssertionError(f"unexpected endpoint {endpoint}")

        def single(**kw):
            seen.update(kw)
            return []                       # no cell type matched; the caller turns it into an empty table

        monkeypatch.setattr(c, "_request", request)
        monkeypatch.setattr(c, "_analyzeSingleGeneList", single)
        return c, seen, calls
    return build


def test_a_former_name_is_resolved_with_a_warning(make):
    c, seen, _ = make()
    with pytest.warns(UserWarning, match="renamed to 'SEAAD2024_MTG_subclass'"):
        c.analyzeGenes(["SST", "PVALB"], studies="SEAAD2024_MTG_Subclass")
    assert seen["studies"] == ["SEAAD2024_MTG_subclass"]


def test_a_former_name_in_another_case_is_resolved(make):
    c, seen, _ = make()
    with pytest.warns(UserWarning, match="DiBellaArlotta2021_Cortex"):
        c.analyzeGenes(["SST"], studies=["dibellaarlotta2021"])
    assert seen["studies"] == ["DiBellaArlotta2021_Cortex"]


def test_a_current_name_passes_without_a_warning(make):
    c, seen, _ = make()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c.analyzeGenes(["SST"], studies=["SEAAD2024_MTG_subclass"])
    assert seen["studies"] == ["SEAAD2024_MTG_subclass"]


def test_an_unknown_name_is_still_refused(make):
    c, _, _ = make()
    with pytest.raises(_markerdb.ValidationError, match="not found"):
        c.analyzeGenes(["SST"], studies=["NoSuchStudy2030"])


def test_exclude_studies_are_resolved_and_conflicts_found_after_resolving(make):
    c, seen, _ = make()
    with pytest.warns(UserWarning):
        c.analyzeGenes(["SST"], exclude_studies=["DiBellaArlotta2021"])
    assert seen["exclude_studies"] == ["DiBellaArlotta2021_Cortex"]
    with pytest.warns(UserWarning), pytest.raises(_markerdb.ValidationError, match="Conflicting"):
        c.analyzeGenes(["SST"], studies=["DiBellaArlotta2021"], exclude_studies=["DiBellaArlotta2021_Cortex"])


def test_the_aliases_are_read_once_per_client(make):
    c, _, calls = make()
    with pytest.warns(UserWarning):
        c.analyzeGenes(["SST"], studies=["SEAAD2024_MTG_Subclass"])
        c.analyzeGenes(["SST"], studies=["DiBellaArlotta2021"])
    assert calls.count("study-aliases") == 1
    assert c.studyAliases() == {d["old_name"]: d["new_name"] for d in ALIASES}


def test_a_server_without_aliases_leaves_the_names_as_given(make):
    c, seen, _ = make(alias_status=404)
    assert c.studyAliases() == {}
    with pytest.raises(_markerdb.ValidationError):                    # as before the change
        c.analyzeGenes(["SST"], studies=["SEAAD2024_MTG_Subclass"])
    c.analyzeGenes(["SST"], studies=["SEAAD2024_MTG_subclass"])
    assert seen["studies"] == ["SEAAD2024_MTG_subclass"]


def test_another_server_error_is_raised(make):
    c, _, _ = make(alias_status=500)
    with pytest.raises(_markerdb.APIError):
        c.analyzeGenes(["SST"], studies=["SEAAD2024_MTG_Subclass"])
