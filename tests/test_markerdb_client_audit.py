"""The PIASOmarkerDB client after the code audit of 2026-10-27 (part 3 minor; review I1's marker DB minors and
improvement 6/7). No request leaves the machine: a localhost server or a fake ``_request`` answers.

- transient failures (503, dropped connections) are retried with backoff;
- a non-JSON answer raises APIError, not a raw JSONDecodeError;
- no directory is made in $HOME; cache_dir is an opt-in response cache;
- as_dict lists a gene once per cell type and warns when contexts are merged;
- a page larger than the server's 100,000-row cap is asked for in pages (it came back short, silently);
- a cell type with a comma in its name can be asked for;
- getRecommendedStudy returns current study names;
- analyzeGenes asks for the marker rows of all its gene sets together, with the per-set results unchanged.
"""
from __future__ import annotations

import json
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from piaso.tools import _markerdb


@pytest.fixture
def server():
    """A localhost server answering from a script: a list of (status, content type, body)."""
    state = {"script": [], "n": 0}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            i = min(state["n"], len(state["script"]) - 1)
            state["n"] += 1
            status, ctype, body = state["script"][i]
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()


def _json(obj):
    return (200, "application/json", json.dumps(obj).encode())


def test_a_transient_503_is_retried(server):
    url, state = server
    state["script"] = [(503, "text/plain", b"restarting"), _json({"data": ["S1", "S2"]})]
    c = _markerdb.PIASOmarkerDB(base_url=url, timeout=5)
    assert c.listStudies() == ["S1", "S2"]
    assert state["n"] == 2


def test_retries_give_up_with_the_status(server):
    url, state = server
    state["script"] = [(503, "text/plain", b"down")]
    c = _markerdb.PIASOmarkerDB(base_url=url, timeout=5, retries=1)
    with pytest.raises(_markerdb.APIError) as e:
        c.listStudies()
    assert e.value.status_code == 503 and state["n"] == 2


def test_a_non_json_answer_raises_api_error(server):
    url, state = server
    state["script"] = [(200, "text/html", b"<html>maintenance</html>")]
    c = _markerdb.PIASOmarkerDB(base_url=url, timeout=5)
    with pytest.raises(_markerdb.APIError, match="did not answer with JSON.*text/html.*maintenance"):
        c.listStudies()


def test_no_directory_in_home_and_an_opt_in_cache(server, tmp_path, monkeypatch):
    url, state = server
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    c = _markerdb.PIASOmarkerDB(base_url=url, timeout=5)
    assert c.cache_dir is None and not (tmp_path / "home").exists()
    state["script"] = [_json({"data": ["S1"]}), _json({"data": ["S2"]})]
    cached = _markerdb.PIASOmarkerDB(base_url=url, timeout=5, cache_dir=tmp_path / "cache")
    assert cached.listStudies() == ["S1"] and cached.listStudies() == ["S1"]
    assert state["n"] == 1, "the second call was answered from the cache"
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    expired = _markerdb.PIASOmarkerDB(base_url=url, timeout=5, cache_dir=tmp_path / "cache", cache_ttl=0)
    assert expired.listStudies() == ["S2"] and state["n"] == 2


ROWS = [
    {"gene": "Gad2", "cell_type": "Inh", "specificity_score": 0.9, "study_publication": "S1", "species": "Mouse",
     "tissue": "Brain", "condition": "Control"},
    {"gene": "Gad1", "cell_type": "Inh", "specificity_score": 0.8, "study_publication": "S2", "species": "Mouse",
     "tissue": "Brain", "condition": "Control"},
    {"gene": "Gad2", "cell_type": "Inh", "specificity_score": 0.7, "study_publication": "S2", "species": "Mouse",
     "tissue": "Brain", "condition": "Control"},
    {"gene": "Slc17a7", "cell_type": "Exc", "specificity_score": 0.6, "study_publication": "S1",
     "species": "Mouse", "tissue": "Brain", "condition": "Control"},
]


def _fake(monkeypatch, c, rows=ROWS, page_cap=None):
    calls = []

    def request(method, endpoint, params=None, **kw):
        calls.append(dict(params or {}))
        out = list(rows)
        if "cell_type" in (params or {}):
            want = set(params["cell_type"].split(","))      # the server splits on commas
            out = [r for r in out if r["cell_type"] in want]
        if "gene" in (params or {}):
            want = {g.upper() for g in params["gene"].split(",")}
            out = [r for r in out if r["gene"].upper() in want]
        total = len(out)
        off = int(params.get("offset", 0)) if params else 0
        lim = params.get("limit") if params else None
        if lim is not None:
            lim = min(int(lim), page_cap) if page_cap else int(lim)
            out = out[off:off + lim]
        else:
            out = out[off:]
        return {"data": out, "total": total, "has_more": off + len(out) < total}

    monkeypatch.setattr(c, "_request", request)
    return calls


def test_as_dict_lists_a_gene_once_and_warns_on_merged_contexts(monkeypatch):
    c = _markerdb.PIASOmarkerDB()
    _fake(monkeypatch, c)
    with pytest.warns(UserWarning, match="1 cell type name.*several contexts.*Inh"):
        d = c.getMarkers(species="Mouse", as_dict=True)
    assert d == {"Inh": ["Gad2", "Gad1"], "Exc": ["Slc17a7"]}


def test_pages_stay_within_the_server_cap(monkeypatch):
    rows = [dict(ROWS[0], gene=f"G{i}") for i in range(23)]
    c = _markerdb.PIASOmarkerDB()
    monkeypatch.setattr(_markerdb, "_MARKER_PAGE_MAX", 10)
    calls = _fake(monkeypatch, c, rows=rows, page_cap=10)
    assert len(c.getAllMarkers(batch_size=1000)) == 23
    assert all(int(q["limit"]) <= 10 for q in calls) and len(calls) == 3
    calls.clear()
    assert len(c.getMarkers(limit=15)) == 15 and [int(q["limit"]) for q in calls] == [10, 5]


def test_a_cell_type_with_a_comma_can_be_asked_for(monkeypatch):
    rows = ROWS + [dict(ROWS[0], cell_type="CD4+, naive"), dict(ROWS[0], cell_type="CD4+")]
    c = _markerdb.PIASOmarkerDB()
    calls = _fake(monkeypatch, c, rows=rows)
    df = c.getMarkers(cell_type="CD4+, naive")
    assert df.cell_type.tolist() == ["CD4+, naive"] and "cell_type" not in calls[-1]
    df = c.getMarkers(cell_type=["CD4+, naive", "Exc"])
    assert sorted(df.cell_type) == ["CD4+, naive", "Exc"]


def test_recommended_studies_carry_current_names():
    c = _markerdb.PIASOmarkerDB()
    assert c.getRecommendedStudy("human", "cortex") == "SEAAD2024_MTG_subclass"
    assert c.getRecommendedStudy("human", "blood") == "AllenHumanImmuneHealthAtlas2025_L2"
    assert c.getRecommendedStudy("human", "brain") == "SilettiLinnarsson2023_WholeBrain_class"
    assert c.getRecommendedStudy("mouse", "cortex") == "AllenWholeMouseBrain2023_isocortex"
    assert c.getRecommendedStudy("mouse", "brain") == "AllenWholeMouseBrain2023_Neuron"


def test_analyze_genes_batches_its_sets(monkeypatch):
    c = _markerdb.PIASOmarkerDB()
    calls = _fake(monkeypatch, c)
    sets = {"c1": ["Gad2", "Gad1"], "c2": ["slc17a7"], "c3": ["Gad2", "Nope"]}
    res, top = c.analyzeGenes(sets)
    assert len(calls) == 1, "one request for the three sets"
    assert top == {"c1": "Inh", "c2": "Exc", "c3": "Inh"}
    # the per-set answer, one query per set as before
    for name, genes in sets.items():
        one = c._resultsToDataFrame(c._analyzeSingleGeneList(genes=genes))
        pd.testing.assert_frame_equal(res[name].reset_index(drop=True), one.reset_index(drop=True))


def test_an_offset_without_a_limit_is_sliced_here(monkeypatch):
    """The server appends OFFSET without LIMIT when no limit is given, which SQLite refuses."""
    c = _markerdb.PIASOmarkerDB()
    calls = _fake(monkeypatch, c)
    df = c.getMarkers(offset=2)
    assert df.gene.tolist() == [r["gene"] for r in ROWS[2:]] and "offset" not in calls[-1]
