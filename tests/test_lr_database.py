"""The CellChatDB ligand-receptor loaders in ``piaso.data``.

SCALAR needs an interaction table and does not ship one. These loaders fetch
CellChatDB on demand and cache it, the same shape as the genome and motif
fetchers — so the tests that matter are: the network is not touched when a
cached copy exists, the ``annotation`` column survives (the LR plotting
functions group by it), and a bad species says what the valid ones are.
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
piaso = pytest.importorskip("piaso")

from piaso.data import _lr  # noqa: E402


@pytest.fixture
def fake_db(tmp_path):
    """A minimal table with the columns the plotting functions rely on."""
    df = pd.DataFrame({
        "interaction_name": ["A_B", "C_D", "E_F"],
        "pathway_name": ["P1", "P2", "P3"],
        "ligand": ["TGFB1", "COL1A2", "VIP"],
        "receptor": ["TGFBR2", "SDC4", "SCTR"],
        "annotation": ["Secreted Signaling", "ECM-Receptor",
                       "Secreted Signaling"],
    })
    path = tmp_path / "human_lr_database_CellChatDB.csv"
    df.to_csv(path, index=False)
    return tmp_path, path


def test_a_cached_copy_is_not_redownloaded(fake_db, monkeypatch):
    tmp_path, _ = fake_db

    def explode(*a, **k):
        raise AssertionError("urlretrieve called despite a cached copy")

    monkeypatch.setattr("urllib.request.urlretrieve", explode)
    df = _lr.load_lr_database("human", data_dir=str(tmp_path))
    assert len(df) == 3
    assert {"ligand", "receptor", "annotation"} <= set(df.columns)


def test_annotation_filter_keeps_one_class(fake_db):
    tmp_path, _ = fake_db
    ecm = _lr.load_lr_database("human", annotation="ECM-Receptor",
                               data_dir=str(tmp_path))
    assert len(ecm) == 1
    assert ecm.iloc[0]["ligand"] == "COL1A2"


def test_a_wrong_annotation_lists_the_real_ones(fake_db):
    tmp_path, _ = fake_db
    with pytest.raises(ValueError, match="Secreted Signaling"):
        _lr.load_lr_database("human", annotation="Nonsense",
                             data_dir=str(tmp_path))


def test_an_unsupported_species_names_the_supported_ones():
    with pytest.raises(ValueError, match="human"):
        _lr.fetch_lr_database("zebrafish")


def test_resolve_returns_none_when_nothing_is_cached(tmp_path, monkeypatch):
    """`data_dir` is checked *in addition to* the home cache, not instead.

    So a genuinely-empty state needs the home cache pointed somewhere empty
    too — matching resolve_screen_path, which behaves the same way.
    """
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path))
    assert _lr.resolve_lr_path("human", data_dir=str(tmp_path)) is None


def test_data_dir_is_searched_alongside_the_home_cache(fake_db):
    tmp_path, path = fake_db
    assert _lr.resolve_lr_path("human", data_dir=str(tmp_path)) == str(path)


def test_the_public_names_are_exported():
    assert piaso.data.fetch_lr_database is _lr.fetch_lr_database
    assert piaso.data.load_lr_database is _lr.load_lr_database
    # camelCase aliases match the house style of fetchSCREEN / fetchJASPAR.
    assert piaso.data.loadLRDatabase is _lr.load_lr_database
