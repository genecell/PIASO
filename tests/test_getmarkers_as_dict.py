"""`as_dict=True` returns the dictionary, not a pair containing it.

`piaso.tl.getMarkers(..., as_dict=True)` used to return `(DataFrame, dict)`.
Almost every use of it wanted the dict -- it is the form
`predictCellTypeByMarker` takes -- and paid for it by unpacking a two-element
tuple and discarding the first element. Asking for a dict now gets a dict.

The old pair survives one release as `as_dict='both'`. The change is announced
once per session on the first `as_dict=True`, because the failure mode when a
caller keeps unpacking is not always loud: with exactly two cell types,
`df, d = getMarkers(as_dict=True)` succeeds and binds two cell-type *names*.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from piaso.tools import _markerdb


@pytest.fixture(autouse=True)
def reset_notice():
    _markerdb._as_dict_change_announced = False
    yield
    _markerdb._as_dict_change_announced = False


@pytest.fixture
def client(monkeypatch):
    """A client whose HTTP layer returns a fixed two-cell-type answer."""
    rows = [
        {"gene": "Foxp2", "cell_type": "L6 CT", "score": 0.9},
        {"gene": "Syt6", "cell_type": "L6 CT", "score": 0.8},
        {"gene": "Cux2", "cell_type": "L2/3 IT", "score": 0.7},
    ]
    c = _markerdb.PIASOmarkerDB()
    monkeypatch.setattr(c, "_request", lambda *a, **k: {"data": rows})
    return c


def _ignore_change_notice():
    warnings.simplefilter("ignore", FutureWarning)


def test_as_dict_true_returns_the_dict(client):
    with warnings.catch_warnings():
        _ignore_change_notice()
        out = client.getMarkers(as_dict=True)
    assert isinstance(out, dict)
    assert set(out) == {"L6 CT", "L2/3 IT"}
    assert out["L6 CT"] == ["Foxp2", "Syt6"]


def test_as_dict_false_is_unchanged(client):
    out = client.getMarkers(as_dict=False)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3


def test_the_default_is_still_the_dataframe(client):
    assert isinstance(client.getMarkers(), pd.DataFrame)


def test_both_returns_the_old_pair_with_a_deprecation(client):
    with pytest.warns(DeprecationWarning, match="as_dict='both' is deprecated"):
        out = client.getMarkers(as_dict="both")
    df, d = out
    assert isinstance(df, pd.DataFrame) and isinstance(d, dict)
    assert len(df) == 3 and set(d) == {"L6 CT", "L2/3 IT"}


def test_the_change_is_announced_once(client):
    with pytest.warns(FutureWarning, match="returns the marker dictionary alone"):
        client.getMarkers(as_dict=True)
    # and not again, so a loop does not fill the terminal
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        client.getMarkers(as_dict=True)
    assert [w for w in rec if issubclass(w.category, FutureWarning)] == []


def test_the_notice_exists_because_two_cell_types_unpack_silently(client):
    """Why the warning is worth its noise: this does not raise."""
    with warnings.catch_warnings():
        _ignore_change_notice()
        a, b = client.getMarkers(as_dict=True)
    assert (a, b) == ("L6 CT", "L2/3 IT"), \
        "unpacking a 2-key dict binds its keys -- silently, hence the notice"


def test_an_unknown_spelling_is_refused(client):
    with pytest.raises(ValueError, match="is not valid"):
        client.getMarkers(as_dict="dict")


def test_get_all_markers_follows_the_same_rule(client, monkeypatch):
    calls = {"n": 0}

    def fake_request(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            return {"data": []}
        return {"data": [{"gene": "Foxp2", "cell_type": "L6 CT", "score": 0.9}]}

    monkeypatch.setattr(client, "_request", fake_request)
    with warnings.catch_warnings():
        _ignore_change_notice()
        out = client.getAllMarkers(as_dict=True)
    assert isinstance(out, dict) and out == {"L6 CT": ["Foxp2"]}
