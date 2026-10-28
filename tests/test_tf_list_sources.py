"""TF-list sources: cisTarget fetches, everything else comes from a file.

AnimalTFDB was a fetchable source until 1.2.4. Its only host answers every
non-browser client with a WAF captcha page -- served as HTTP 200, ~16 kB of
HTML, which the old parser turned into a dozen junk "symbols" that silently
shrank the TF universe. The host has no mirror (the HUST domain no longer
resolves), so the source was removed rather than retried, and the parser now
refuses that shape of file whatever the host.

For regulon inference neither catalogue is needed: `source="motifdb"` (no
restriction beyond having a motif) is the right answer.
"""

import pathlib

import pytest

from piaso.data import _motifs as M
from piaso.data import fetch_tf_list, load_tf_list


def _write(tmp_path, source, species, body):
    p = tmp_path / M._TF_LIST_STEMS[source][species]
    p.write_text(body)
    return p


def _catalogue(n=60, prefix="TF"):
    return "".join(f"{prefix}{i}\n" for i in range(n))


def _serve(monkeypatch, write):
    """Stand in for the network: ``write(url, path)`` fills the file piaso.data's
    one download helper asks for (a temporary file, renamed once it passes)."""
    from piaso.data import _download
    monkeypatch.setattr(_download, "_fetch_once",
                        lambda url, path, timeout, progress, expected_size: write(url, str(path)))
    monkeypatch.setattr(_download.time, "sleep", lambda s: None)


# --------------------------------------------------------------- sources --


def test_motifdb_is_a_sentinel_meaning_no_restriction():
    assert load_tf_list(source="motifdb") is None


def test_default_source_is_cistarget():
    import inspect
    assert inspect.signature(load_tf_list).parameters["source"].default == "cistarget"
    assert inspect.signature(fetch_tf_list).parameters["source"].default == "cistarget"


def test_cistarget_is_the_only_fetchable_source():
    assert list(M._TF_LIST_SOURCES) == ["cistarget"]


def test_each_source_caches_under_its_own_name(tmp_path):
    assert (M._tf_list_cache_path("human", "cistarget", tmp_path).name
            != M._tf_list_cache_path("human", "animaltfdb", tmp_path).name)


def test_a_cached_list_is_read_without_network(tmp_path, monkeypatch):
    _write(tmp_path, "cistarget", "human",
           "SOX9\nOLIG2\n# a comment\n\nPAX6\n" + _catalogue())
    monkeypatch.setattr(M, "_DEFAULT_CACHE_DIR", pathlib.Path(tmp_path))
    got = load_tf_list("human")
    assert {"SOX9", "OLIG2", "PAX6"} <= got and "# a comment" not in got


def test_a_tsv_with_a_symbol_column_is_understood(tmp_path):
    p = tmp_path / "animaltfdb_saved_from_a_browser.tsv"
    p.write_text("Ensembl\tSymbol\tFamily\nENSMUSG1\tSox9\tSOX\nENSMUSG2\tPax6\tPAX\n")
    assert load_tf_list("mouse", path=str(p)) == {"Sox9", "Pax6"}


def test_an_explicit_path_wins_over_the_cache(tmp_path):
    p = tmp_path / "mine.txt"
    p.write_text("MYTF\n")
    assert load_tf_list("human", path=str(p)) == {"MYTF"}


# ------------------------------------------------------- the file guard ---


def test_an_html_page_is_refused_rather_than_parsed(tmp_path):
    """The exact 1.2.3 failure: a WAF page parsed into a dozen 'symbols'."""
    p = tmp_path / "waf.html"
    p.write_text('<!doctype html>\n<meta charset="UTF-8">\n'
                 '<meta name="aliyun_waf_aa" content="ff926c7f">\n')
    with pytest.raises(ValueError, match="looks like an HTML page"):
        load_tf_list("human", path=str(p))


def test_markup_fragments_anywhere_in_the_file_are_refused(tmp_path):
    p = tmp_path / "half.txt"
    p.write_text("SOX9\n<div class=x>\nPAX6\n")
    with pytest.raises(ValueError, match="cannot be gene symbols"):
        load_tf_list("human", path=str(p))


def test_a_short_users_file_is_fine_but_a_short_catalogue_is_not(tmp_path, monkeypatch):
    """The count check applies to downloads, never to a file the user passed."""
    small = tmp_path / "ten.txt"
    small.write_text("SOX9\nPAX6\nOLIG2\n")
    assert load_tf_list("human", path=str(small)) == {"SOX9", "PAX6", "OLIG2"}

    _write(tmp_path, "cistarget", "mouse", "Sox9\nPax6\n")
    monkeypatch.setattr(M, "_DEFAULT_CACHE_DIR", pathlib.Path(tmp_path))
    with pytest.raises(ValueError, match="fewer than the"):
        load_tf_list("mouse")


def test_a_download_that_returns_a_page_is_not_left_in_the_cache(tmp_path, monkeypatch):
    def _serve_a_captcha(url, filename):
        pathlib.Path(filename).write_text("<!doctype html><title>captcha</title>")

    _serve(monkeypatch, _serve_a_captcha)
    with pytest.raises(RuntimeError, match="Failed to download"):
        fetch_tf_list("human", dest_dir=str(tmp_path))
    assert not M._tf_list_cache_path("human", "cistarget", tmp_path).exists()


# ---------------------------------------------------------------- errors --


def test_missing_cache_names_the_fetcher_for_that_source(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_DEFAULT_CACHE_DIR", pathlib.Path(tmp_path))
    with pytest.raises(RuntimeError, match=r"fetch_tf_list\('human', source='cistarget'\)"):
        load_tf_list("human")


def test_unknown_source_and_species_raise():
    with pytest.raises(ValueError, match="Unknown source"):
        load_tf_list("human", source="nope")
    with pytest.raises(ValueError, match="Unsupported species"):
        load_tf_list("zebrafish", source="cistarget")
    with pytest.raises(ValueError, match="Unknown source"):
        fetch_tf_list("human", source="nope")


@pytest.mark.parametrize("call", ["load", "fetch"])
def test_animaltfdb_explains_the_removal_and_points_at_path(call):
    """Not 'unknown source' -- it worked in 1.2.3 and users have that code."""
    fn = load_tf_list if call == "load" else fetch_tf_list
    with pytest.raises(ValueError) as e:
        fn("human", source="animaltfdb")
    msg = str(e.value)
    assert "WAF" in msg and "guolab.wchscu.cn" in msg
    assert "path=" in msg and "browser" in msg
    assert "fetch_tf_list('human')" in msg           # the working default


def test_the_1_2_3_fetcher_raises_without_touching_the_network(monkeypatch):
    from piaso.data import fetch_animaltfdb_tf_list

    def _never(*a, **k):                       # noqa: ARG001
        raise AssertionError("the shim must not make a request")

    _serve(monkeypatch, _never)
    with pytest.raises(RuntimeError) as e:
        fetch_animaltfdb_tf_list("mouse")
    msg = str(e.value)
    assert "removed in PIASO 1.2.4" in msg
    assert "load_tf_list" in msg and "fetch_tf_list('mouse')" in msg


def test_other_sources_fail_with_the_plain_message(tmp_path, monkeypatch):
    import urllib.error

    def _boom(url, filename):
        raise urllib.error.URLError("no route")

    _serve(monkeypatch, _boom)
    with pytest.raises(RuntimeError, match="Failed to download the cistarget"):
        fetch_tf_list("mouse", dest_dir=str(tmp_path))


def test_fetch_writes_where_load_will_look(tmp_path, monkeypatch):
    def _fake(url, filename):
        pathlib.Path(filename).write_text("SOX9\nPAX6\n" + _catalogue())

    _serve(monkeypatch, _fake)
    monkeypatch.setattr(M, "_DEFAULT_CACHE_DIR", pathlib.Path(tmp_path))
    fetch_tf_list("human")
    assert {"SOX9", "PAX6"} <= load_tf_list("human")
