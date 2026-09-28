"""fetch_dataset: verify before the final name, never overwrite a working file."""
import hashlib
import os
import time

import pytest

import piaso.data._datasets as D

PAYLOAD = b"cytome bytes" * 1000


def _setup(monkeypatch, tmp_path, payload=PAYLOAD, md5=None, filename="toy.cytome"):
    info = {"filename": filename, "title": "toy", "url": "http://example.invalid/toy",
            "md5": md5 if md5 is not None else hashlib.md5(PAYLOAD).hexdigest(),
            "size_bytes": len(PAYLOAD)}
    monkeypatch.setattr(D, "dataset_info", lambda name: info)
    calls = []

    def fake_download(url, dest, expected_size=0):
        calls.append(str(dest))
        # the destination handed to the downloader is never the final name
        assert not str(dest).endswith(filename)
        with open(dest, "wb") as fh:
            fh.write(payload)
    monkeypatch.setattr(D, "_download_with_progress", fake_download)
    monkeypatch.setattr(D, "_md5_file", _counting_md5(calls))
    return calls


def _counting_md5(calls):
    real = D._md5_file

    def md5(path):
        calls.append(("md5", str(path)))
        return real(path)
    return md5


def test_downloads_verifies_and_renames(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path)
    p = D.fetch_dataset("toy", data_dir=tmp_path)
    assert p.read_bytes() == PAYLOAD
    assert not list(p.parent.glob("*.part"))
    assert len([c for c in calls if isinstance(c, str)]) == 1


def test_existing_file_is_not_read_or_replaced(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path)
    p = D.fetch_dataset("toy", data_dir=tmp_path)
    # the user writes into the downloaded store
    with open(p, "ab") as fh:
        fh.write(b"an embedding")
    n = len(calls)
    q = D.fetch_dataset("toy", data_dir=tmp_path)
    assert q == p
    assert p.read_bytes().endswith(b"an embedding")
    assert len(calls) == n, "a cached file must be neither checksummed nor downloaded"


def test_bad_download_leaves_existing_file_and_no_partial(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    p = D.fetch_dataset("toy", data_dir=tmp_path)
    with open(p, "ab") as fh:
        fh.write(b"work")
    _setup(monkeypatch, tmp_path, payload=b"corrupted")
    with pytest.raises(RuntimeError, match="existing file is unchanged"):
        D.fetch_dataset("toy", data_dir=tmp_path, force=True)
    assert p.read_bytes().endswith(b"work")
    assert not list(p.parent.glob("*.part"))


def test_force_replaces_only_after_verification(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    p = D.fetch_dataset("toy", data_dir=tmp_path)
    with open(p, "ab") as fh:
        fh.write(b"work")
    D.fetch_dataset("toy", data_dir=tmp_path, force=True)
    assert p.read_bytes() == PAYLOAD


def test_stale_partials_are_removed_fresh_ones_kept(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    d = D._datasets_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    old, new = d / ".toy.cytome.old.part", d / ".toy.cytome.new.part"
    old.write_bytes(b"x"); new.write_bytes(b"x")
    past = time.time() - 3 * 86400
    os.utime(old, (past, past))
    D.fetch_dataset("toy", data_dir=tmp_path)
    assert not old.exists() and new.exists()


def test_size_mismatch_on_non_store_warns_without_downloading(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, filename="toy.h5ad")
    p = D.fetch_dataset("toy", data_dir=tmp_path)
    p.write_bytes(b"truncated")
    n = len(calls)
    with pytest.warns(UserWarning, match="force=True"):
        D.fetch_dataset("toy", data_dir=tmp_path)
    assert len(calls) == n and p.read_bytes() == b"truncated"
