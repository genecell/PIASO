"""Tests for cytome's behaviour when ``PRAGMA journal_mode=WAL`` fails.

Background (see docs/discussion/2026-05-03_cytome_nfs_wal_fallback.md):
The user originally reported ``sqlite3.OperationalError: disk I/O
error`` on a Harvard NFS mount during ``cytome.open()``. The error
self-healed after waiting 5–10 minutes — meaning the cause was a
**transient** writer-contention issue (a Snakemake pipeline still
writing), not a fundamental WAL/NFS incompatibility.

A first attempt (Cytome 730d7a7) auto-fell-back to ``journal_mode=DELETE``
with a UserWarning. That fix had two flaws:
  1. ``PRAGMA journal_mode=DELETE`` ALSO requires an exclusive lock,
     so it would fail in the same scenario the WAL pragma failed in,
     just with a more confusing traceback.
  2. It silently downgraded a perfectly good WAL session for the rare
     case where the filesystem truly couldn't support WAL.

Current behaviour (Cytome <next>): re-raise the original
OperationalError with an actionable diagnostic message listing the
three real causes (active writer, stale sidecars, truly-incompatible
filesystem) and what the user can do for each.

Tests below verify:
- Local-disk opens still use WAL with no diagnostic noise.
- mmap_size pragma failure is silently swallowed (SQLite handles it).
- WAL-pragma failure raises with the actionable diagnostic message.
"""
import sqlite3
import warnings

import pytest


# ---------------------------------------------------------------------
# Helper: wrap sqlite3.connect so a chosen PRAGMA raises OperationalError
# ---------------------------------------------------------------------

class _PragmaInterceptingConnection:
    """Thin proxy around a real sqlite3.Connection that raises
    OperationalError on a configurable PRAGMA prefix.

    Used to simulate hostile NFS clients without an actual NFS mount.
    """

    def __init__(self, real_conn, fail_prefix: str):
        self._conn = real_conn
        self._fail_prefix = fail_prefix

    def execute(self, sql, *args, **kw):
        if isinstance(sql, str) and sql.startswith(self._fail_prefix):
            raise sqlite3.OperationalError(
                f"disk I/O error (simulated for {self._fail_prefix!r})"
            )
        return self._conn.execute(sql, *args, **kw)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *args):
        return self._conn.__exit__(*args)


def _make_connect(fail_prefix: str):
    real_connect = sqlite3.connect

    def fake_connect(*args, **kw):
        return _PragmaInterceptingConnection(real_connect(*args, **kw), fail_prefix)

    return fake_connect


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_local_disk_still_uses_wal(tmp_path):
    """On a real local filesystem that supports WAL, no warning is
    emitted and journal_mode actually ends up as WAL."""
    import cytome

    work = tmp_path / "fresh.cytome"
    ds = cytome.create(str(work))
    ds.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds2 = cytome.open(str(work))
        mode = ds2._conn.execute("PRAGMA journal_mode").fetchone()[0]
        ds2.close()

    assert mode.lower() == "wal", (
        f"Expected WAL journal mode on local disk, got {mode!r}"
    )
    assert not caught, (
        f"Did not expect any warning on local-disk open. "
        f"Got: {[str(w.message) for w in caught]}"
    )


def test_mmap_size_failure_is_swallowed(tmp_path, monkeypatch):
    """If PRAGMA mmap_size=... raises (e.g. on extreme NFS), it should
    be silently swallowed — SQLite degrades to read/write() automatically."""
    import cytome
    from cytome.io import sqlite_engine

    work = tmp_path / "fresh.cytome"
    ds = cytome.create(str(work))
    ds.close()

    monkeypatch.setattr(sqlite_engine.sqlite3, "connect",
                        _make_connect("PRAGMA mmap_size"))

    # Should NOT raise even though mmap_size fails
    ds2 = cytome.open(str(work))
    ds2.close()


def test_wal_failure_raises_with_actionable_diagnostic(tmp_path, monkeypatch):
    """When PRAGMA journal_mode=WAL fails, the original OperationalError
    is re-raised with a diagnostic message listing the three real causes
    (active writer, stale sidecars, truly-incompatible filesystem) and
    what the user can do for each."""
    import cytome
    from cytome.io import sqlite_engine

    work = tmp_path / "fresh.cytome"
    ds = cytome.create(str(work))
    ds.close()

    monkeypatch.setattr(sqlite_engine.sqlite3, "connect",
                        _make_connect("PRAGMA journal_mode=WAL"))

    with pytest.raises(sqlite3.OperationalError) as excinfo:
        cytome.open(str(work))

    msg = str(excinfo.value)
    # The message must mention all three causes so the user can pick
    # the right remediation.
    assert "Another process is writing" in msg, (
        f"Diagnostic should mention concurrent writer. Got:\n{msg}"
    )
    assert "Stale -wal/-shm" in msg, (
        f"Diagnostic should mention stale sidecars. Got:\n{msg}"
    )
    assert "Copy the cytome to local disk" in msg, (
        f"Diagnostic should mention local-disk fallback. Got:\n{msg}"
    )
    # And the original error should be chained
    assert excinfo.value.__cause__ is not None
    assert "disk I/O error" in str(excinfo.value.__cause__)


def test_wal_failure_diagnostic_includes_concrete_paths(tmp_path, monkeypatch):
    """The diagnostic message should include the actual file path so
    the user can copy/paste the suggested 'cp ... /tmp/...' command."""
    import cytome
    from cytome.io import sqlite_engine

    work = tmp_path / "named_for_test.cytome"
    ds = cytome.create(str(work))
    ds.close()

    monkeypatch.setattr(sqlite_engine.sqlite3, "connect",
                        _make_connect("PRAGMA journal_mode=WAL"))

    with pytest.raises(sqlite3.OperationalError) as excinfo:
        cytome.open(str(work))

    msg = str(excinfo.value)
    # The 'cp <db> /tmp/local.cytome' suggestion should reference the
    # actual file path, so users can copy/paste.
    assert str(work) in msg or "named_for_test.cytome" in msg, (
        f"Diagnostic should mention the actual cytome path. Got:\n{msg}"
    )
