"""Two SQLite copies in one process: the mechanism, pinned.

``_piaso`` bundles its own SQLite. Two facts decide how Rust may touch a store
that this process's Python connection holds:

1. A *locking* open by the bundled copy cannot see the Python connection (the
   kernel never reports a process's own locks to it), so it takes the
   dead-man-switch lock on the ``-shm`` index, truncates the index and rebuilds
   it. A Python connection that had mapped the part cut off dies with SIGBUS on
   its next large commit. We assert the truncation, not the crash: the crash is
   undefined behaviour, the truncation is its cause.
2. Closing *any* descriptor on the store drops every POSIX lock the process
   holds on it. Python's own SQLite defers such closes; another SQLite copy,
   or plain ``open().close()``, cannot. The Python connection is then
   unlocked, and another process may delete its write-ahead log.

So inside a Python process the store has one owner, the Python connection,
and Rust receives bytes (``svd_pass`` with a chunk source).
"""
import os
import sqlite3
import subprocess
import sys
import textwrap

import pytest

try:
    from piaso import _piaso
    HAVE = hasattr(_piaso, "svd_layout")
except Exception:  # pragma: no cover
    HAVE = False

pytestmark = [pytest.mark.skipif(not HAVE, reason="Rust extension not built"),
              pytest.mark.skipif(sys.platform != "linux", reason="POSIX lock semantics as on Linux")]

SCHEMA = [
    "CREATE TABLE matrix_meta(matrix_name TEXT, n_rows INT, n_cols INT, n_nonzero INT, dtype TEXT, "
    "row_entity TEXT, col_entity TEXT, chunk_size INT, n_chunks INT, has_csc INT, csc_chunk_size INT, "
    "csc_n_chunks INT, layout TEXT)",
    "INSERT INTO matrix_meta VALUES('ATAC_counts',10,10,0,'float32','cells','peaks',128,1,0,NULL,NULL,'csr')",
    "CREATE TABLE matrix_chunks(matrix_name TEXT, chunk_idx INT, row_start INT, row_end INT, n_nonzero INT)",
    "CREATE TABLE matrix_csc_chunks(matrix_name TEXT, chunk_idx INT, col_start INT, col_end INT, n_nonzero INT)",
    "INSERT INTO matrix_chunks VALUES('ATAC_counts',0,0,10,0)",
    "CREATE TABLE pad(b BLOB)",
]


def _store(path, big=False):
    c = sqlite3.connect(str(path))
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA wal_autocheckpoint=0")
    for s in SCHEMA:
        c.execute(s)
    c.commit()
    if big:
        # > 4,062 frames: the index grows a second 32 KB region.
        c.execute("INSERT INTO pad VALUES(?)", (os.urandom(24 << 20),))
        c.commit()
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    return c


@pytest.mark.parametrize("snapshot, truncated", [(False, True), (True, False)])
def test_a_locking_open_by_the_bundled_sqlite_truncates_the_index(tmp_path, snapshot, truncated):
    p = tmp_path / "s.db"
    c = _store(p, big=True)
    shm = str(p) + "-shm"
    before = os.path.getsize(shm)
    assert before >= 65536
    _piaso.svd_layout(str(p), "ATAC_counts", snapshot=snapshot)
    after = os.path.getsize(shm)
    c.close()
    assert (after < before) is truncated, (before, after)


PROBE = textwrap.dedent(r'''
    import fcntl, os, struct, sys
    fd = os.open(sys.argv[1], os.O_RDWR)
    r = fcntl.fcntl(fd, fcntl.F_GETLK, struct.pack("hhqqi", fcntl.F_WRLCK, 0, 0x40000002, 510, 0))
    os.close(fd)
    print("free" if struct.unpack("hhqqi", r)[0] == fcntl.F_UNLCK else "held")
''')


def _lock_on_store(path):
    return subprocess.run([sys.executable, "-c", PROBE, str(path)], capture_output=True,
                          text=True, check=True).stdout.strip()


def _op(name, path):
    if name == "rust_immutable":
        _piaso.svd_layout(str(path), "ATAC_counts", snapshot=True)
    elif name == "plain_open_close":
        open(path, "rb").close()
    elif name == "python_immutable":
        k = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
        k.execute("SELECT count(*) FROM matrix_meta").fetchone()
        k.close()
    elif name == "python_second_connection":
        k = sqlite3.connect(str(path))
        k.execute("SELECT count(*) FROM matrix_meta").fetchone()
        k.close()


@pytest.mark.parametrize("op, lock", [
    ("nothing", "held"),
    ("python_second_connection", "held"),
    ("python_immutable", "held"),
    ("plain_open_close", "free"),
    ("rust_immutable", "free"),
])
def test_which_operations_keep_the_python_connections_lock(tmp_path, op, lock):
    p = tmp_path / "l.db"
    c = _store(p)
    c.execute("SELECT count(*) FROM matrix_meta").fetchone()
    assert _lock_on_store(p) == "held"
    _op(op, p)
    got = _lock_on_store(p)
    c.close()
    assert got == lock
