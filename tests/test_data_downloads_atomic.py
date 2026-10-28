"""piaso.data downloads: complete or absent, never half; the newest registry wins.

Regression tests for the code audit of 2026-10-27 (review I1, findings 13 and 14, and the [A] minors):

- a registry cached by an older PIASO overrode the newer one the package ships, so new datasets were
  "Unknown"; the highest ``version`` now wins;
- the genome files and the TF list were downloaded straight to their final names, so an interrupted
  download counted as complete ("already available") on the next call; every download now goes through
  one helper (a temporary file beside the destination, checked, renamed; timeouts and retries);
- ``fetch_dataset(force=True)`` left the old store's ``-wal`` for SQLite to replay onto the new file;
- downloads were mode 0600 (``mkstemp``) whatever the umask.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import stat
import threading

import pytest

import piaso.data._datasets as D
from piaso.data import _download, _genome


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PIASO_DATA_DIR", str(tmp_path))
    import piaso
    monkeypatch.setattr(piaso.settings, "data_dir", None, raising=False)
    monkeypatch.setattr(D, "_registry_cache", None)
    monkeypatch.setattr(_download.time, "sleep", lambda s: None)
    return tmp_path


def _builtin():
    with open(os.path.join(os.path.dirname(D.__file__), "_registry_builtin.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ registry

def test_a_newer_packaged_registry_wins_over_an_older_cache(data_root, monkeypatch):
    (data_root / "datasets.json").write_text(json.dumps({"version": 1, "datasets": {"old_only": {}}}))
    monkeypatch.setattr(D, "_fetch_remote_registry", lambda p: pytest.fail("no remote with a cache"))
    reg = D._load_registry()
    assert reg["version"] == _builtin()["version"] and "old_only" not in reg["datasets"]


def test_a_newer_or_equal_cache_is_kept(data_root, monkeypatch):
    v = _builtin()["version"]
    monkeypatch.setattr(D, "_fetch_remote_registry", lambda p: None)
    for version in (v + 1, v):
        monkeypatch.setattr(D, "_registry_cache", None)
        (data_root / "datasets.json").write_text(json.dumps({"version": version, "datasets": {"mine": {}}}))
        assert "mine" in D._load_registry()["datasets"]


def test_without_a_cache_an_older_remote_loses_to_the_package(data_root, monkeypatch):
    monkeypatch.setattr(D, "_fetch_remote_registry", lambda p: {"version": 1, "datasets": {"stale": {}}})
    assert "stale" not in D._load_registry()["datasets"]


# ------------------------------------------------------------------ the helper

class _Truncating(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):                       # noqa: N802
        type(self).hits += 1
        self.send_response(200)
        self.send_header("Content-Length", "1000")
        self.end_headers()
        self.wfile.write(b"x" * 400)        # then the connection closes
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *a):
        pass


def test_a_truncated_transfer_is_retried_and_never_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(_download.time, "sleep", lambda s: None)
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Truncating)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        dest = tmp_path / "f.bed"
        with pytest.raises(Exception):
            _download.download(f"http://127.0.0.1:{srv.server_port}/f.bed", dest, timeout=5, retries=3)
        assert not dest.exists()
        assert not list(tmp_path.glob("*.part"))
        assert _Truncating.hits == 3
    finally:
        srv.shutdown()


def test_a_failed_check_leaves_the_old_file(tmp_path, monkeypatch):
    dest = tmp_path / "x.txt"
    dest.write_text("old")
    monkeypatch.setattr(_download, "_fetch_once", lambda url, p, *a: open(p, "w").write("new"))

    def refuse(path):
        raise ValueError("not what was asked for")
    with pytest.raises(ValueError):
        _download.download("http://x.invalid/x", dest, check=refuse)
    assert dest.read_text() == "old" and not list(tmp_path.glob("*.part"))


def test_downloads_take_the_umask(tmp_path, monkeypatch):
    monkeypatch.setattr(_download, "_fetch_once", lambda url, p, *a: open(p, "w").write("data"))
    old = os.umask(0o022)
    try:
        dest = _download.download("http://x.invalid/x", tmp_path / "x.txt")
    finally:
        os.umask(old)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o644


# ------------------------------------------------------------------ genome files

def test_an_interrupted_genome_download_is_not_taken_for_a_complete_one(tmp_path, monkeypatch):
    monkeypatch.setattr(_download.time, "sleep", lambda s: None)
    files = _genome.GENOME_FILES["mm10"]
    victim = files["promoter"]

    def flaky(url, path, *a):
        with open(path, "w") as fh:
            fh.write("chr1\t1\t2\n" * 10)
            if url.endswith(victim):
                fh.flush()
                raise ConnectionResetError("reset mid-file")
    monkeypatch.setattr(_download, "_fetch_once", flaky)
    with pytest.raises(RuntimeError, match="required file"):
        _genome.fetch_genome("mm10", dest_dir=str(tmp_path), download_gtf=False)
    gdir = tmp_path / "mm10"
    assert not (gdir / victim).exists()               # never half a file under its own name
    assert not list(gdir.glob("*.part"))

    monkeypatch.setattr(_download, "_fetch_once", lambda url, p, *a: open(p, "w").write("chr1\t1\t2\n"))
    out = _genome.fetch_genome("mm10", dest_dir=str(tmp_path), download_gtf=False)
    assert (gdir / victim).exists() and out == str(gdir)


# ------------------------------------------------------------------ TF list

def test_an_interrupted_tf_list_is_not_cached(tmp_path, monkeypatch):
    from piaso.data import _motifs as M
    from piaso.data import fetch_tf_list
    monkeypatch.setattr(_download.time, "sleep", lambda s: None)

    def cut(url, path, *a):
        with open(path, "w") as fh:
            fh.write("".join(f"TF{i}\n" for i in range(5000)))   # long enough to pass the check...
        raise TimeoutError("read timed out")                     # ...but the transfer did not finish
    monkeypatch.setattr(_download, "_fetch_once", cut)
    with pytest.raises(RuntimeError, match="Failed to download"):
        fetch_tf_list("human", dest_dir=str(tmp_path))
    assert not M._tf_list_cache_path("human", "cistarget", tmp_path).exists()


# ------------------------------------------------------------------ fetch_dataset

def test_force_drops_the_old_stores_sidecars(data_root, monkeypatch):
    payload = b"store bytes" * 100
    info = {"filename": "toy.cytome", "title": "toy", "url": "http://x.invalid/toy",
            "md5": hashlib.md5(payload).hexdigest(), "size_bytes": len(payload)}
    monkeypatch.setattr(D, "dataset_info", lambda name: info)
    monkeypatch.setattr(_download, "_fetch_once", lambda url, p, *a: open(p, "wb").write(payload))
    p = D.fetch_dataset("toy")
    for suffix in ("-wal", "-shm"):
        open(f"{p}{suffix}", "wb").write(b"old sidecar")
    old = os.umask(0o022)
    try:
        q = D.fetch_dataset("toy", force=True)
    finally:
        os.umask(old)
    assert q.read_bytes() == payload
    assert not os.path.exists(f"{q}-wal") and not os.path.exists(f"{q}-shm")
    assert stat.S_IMODE(q.stat().st_mode) == 0o644
