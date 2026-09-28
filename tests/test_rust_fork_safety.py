"""Every function of the Rust extension works in a process forked after it ran.

rayon's global pool is inherited by a forked child without its threads, and a
child that uses it waits for them for ever. That is Python's
``multiprocessing`` on Linux: the motif scanner hung this way, as the Leiden
did before it. Every parallel function now builds a pool of its own
(``src/threads.rs``).

The list of functions is read from the extension itself. A function added
later without an entry here fails ``test_every_function_has_an_entry``, the
first time the suite runs, instead of hanging a user's pipeline.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import threading
import time

import numpy as np
import pytest
import scipy.sparse as sp

_piaso = pytest.importorskip("piaso._piaso")

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="fork is the Linux default")


def _graph(n=40_000, k=8, seed=0):
    """Symmetric kNN-like graph, above the size the Leiden runs on one thread."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = (rows + rng.integers(1, 50, rows.size)) % n
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    return A


def _leiden():
    A = _graph()
    labels, _ = _piaso.leiden_csr(A.indptr.astype(np.int64), A.indices.astype(np.int32), A.data,
                                  A.shape[0], n_threads=4)
    return int(np.asarray(labels).max())


def _scan():
    rng = np.random.default_rng(1)
    widths = [6, 8, 10]
    pssms = rng.normal(size=5 * sum(widths))
    seqs = rng.integers(0, 4, size=2000 * 200).astype(np.uint8)
    offsets = list(range(0, seqs.size + 1, 200))
    best, count = _piaso.scan_motifs_fwd(pssms.tolist(), widths, [3.0, 4.0, 5.0],
                                         seqs.tobytes(), offsets, n_threads=4)
    return float(np.nansum(best)) + int(count.sum())


def _fused(f32=False):
    rng = np.random.default_rng(2)
    n_rows, n_cols, n_sets, n_ctrl = 300, 50, 3, 4
    A = sp.random(n_rows, n_cols, 0.2, format="csr", random_state=3)
    B = sp.random(n_cols, n_sets * n_ctrl, 0.3, format="csr", random_state=4)
    t = np.float32 if f32 else np.float64
    fn = _piaso.fused_matmul_reduce_f32 if f32 else _piaso.fused_matmul_reduce
    means, _ = fn(A.data.astype(t), A.indices.astype(np.int32), A.indptr.astype(np.int32),
                  n_rows, n_cols, B.data.astype(t), B.indices.astype(np.int32),
                  B.indptr.astype(np.int32), n_sets * n_ctrl, rng.normal(size=n_rows * n_sets),
                  n_sets, n_ctrl, 64, 4, True)
    return float(np.asarray(means).sum())


def _score_complete():
    n_rows, n_cols, k = 300, 50, 5
    A = sp.random(n_rows, n_cols, 0.2, format="csr", random_state=5)
    knn = np.array([(j + d) % n_cols for j in range(n_cols) for d in range(1, k + 1)], dtype=np.int64)
    sets = np.array([0, 1, 2, 10, 11, 12, 13], dtype=np.int32)
    offs = np.array([0, 3, 7], dtype=np.int32)
    out = _piaso.score_complete(A.data, A.indices.astype(np.int32), A.indptr.astype(np.int32),
                                n_rows, n_cols, knn, k, sets, offs,
                                np.ones(7), offs, 4, 0, 64, 4, True)
    return float(np.asarray(out[0]).sum())


def _svd_store(tmp):
    import anndata as ad
    import cytome
    rng = np.random.default_rng(6)
    X = sp.random(400, 300, 0.05, format="csr", random_state=7, dtype=np.float32)
    X.data = rng.poisson(2, X.nnz).astype(np.float32) + 1
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(400)]
    A.var_names = [f"chr1:{1000 * j}-{1000 * j + 500}" for j in range(300)]
    path = str(tmp / "svd.cytome")
    cytome.from_anndata(A, modality="ATAC", output=path).close()
    return path, X


def _svd_args(X):
    n, g = X.shape
    remap = np.arange(g, dtype=np.int32)
    depth = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    idf = np.log1p(n / np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)).astype(np.float32)
    dense = np.random.default_rng(8).standard_normal((g, 5))
    return dense, remap, g, None, n, depth, idf, 1e4


def _svd_pass(path, X):
    got = _piaso.svd_pass(path, "ATAC_counts", "xb", *_svd_args(X), n_threads=4)
    return float(np.asarray(got).sum())


def _svd_pass_source(path, X):
    import cytome
    ds = cytome.open(path)
    try:
        with ds.chunk_source("ATAC_counts", n_readers=2) as src:
            got = _piaso.svd_pass_source(src, "xb", *_svd_args(X), n_threads=4)
    finally:
        ds.close()
    return float(np.asarray(got).sum())


def _svd_layout(path):
    return str(_piaso.svd_layout(path, "ATAC_counts"))


# name -> how to exercise it: a callable taking (path, X), or the reason it
# is not run here. Every function of the extension must be listed.
CALLS = {
    "leiden_csr": lambda p, X: _leiden(),
    "scan_motifs_fwd": lambda p, X: _scan(),
    "fused_matmul_reduce": lambda p, X: _fused(False),
    "fused_matmul_reduce_f32": lambda p, X: _fused(True),
    "score_complete": lambda p, X: _score_complete(),
    "svd_pass": _svd_pass,
    "svd_pass_source": _svd_pass_source,
    "svd_layout": lambda p, X: _svd_layout(p),
    "svd_layout_rule": lambda p, X: str(_piaso.svd_layout_rule(1000, 500, True, True)),
    "chrom_sizes": lambda p, X: len(_piaso.chrom_sizes("hg38")),
    "known_genomes": lambda p, X: len(_piaso.known_genomes()),
    # Built only with --features internal. Each builds its pool per call
    # through threads::count and needs a fragment store.
    "call_peaks_pyo3": "needs a fragment store; pool per call (threads::count)",
    "quantify_peaks_pyo3": "needs a fragment store; pool per call (threads::count)",
    "index_fragments_pyo3": "single-threaded; needs a fragment store",
}


def _functions():
    return sorted(n for n in dir(_piaso)
                  if not n.startswith("_") and callable(getattr(_piaso, n)))


def test_every_function_has_an_entry():
    missing = [n for n in _functions() if n not in CALLS]
    assert not missing, (
        f"{missing}: add an entry to CALLS. A parallel function must build its "
        f"pool with src/threads.rs, and be shown to work after a fork here.")


def _in_child(fn, path, X, q):
    try:
        q.put(("ok", fn(path, X)))
    except BaseException as e:  # noqa: BLE001
        q.put(("error", repr(e)))


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    pytest.importorskip("cytome")
    return _svd_store(tmp_path_factory.mktemp("fork"))


@pytest.mark.parametrize("name", [n for n, c in CALLS.items() if callable(c)])
def test_works_in_a_forked_child(name, store):
    if not hasattr(_piaso, name):
        pytest.skip(f"{name} is not in this build")
    path, X = store
    fn = CALLS[name]
    parent = fn(path, X)                      # the parent's pools, if any, now exist
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    child = ctx.Process(target=_in_child, args=(fn, path, X, q))
    child.start()
    child.join(60)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail(f"{name} hung in a forked child")
    status, value = q.get(timeout=5)
    assert status == "ok", value
    assert value == parent


def test_the_scan_releases_python_and_stops_on_request():
    """Other Python threads run while a scan runs; before, it held the lock."""
    rng = np.random.default_rng(9)
    widths = [12] * 40
    pssms = rng.normal(size=5 * sum(widths)).tolist()
    seqs = rng.integers(0, 4, size=20_000 * 500).astype(np.uint8)
    offsets = list(range(0, seqs.size + 1, 500))
    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0.001)

    t = threading.Thread(target=tick)
    t.start()
    t0 = time.perf_counter()
    _piaso.scan_motifs_fwd(pssms, widths, [6.0] * 40, seqs.tobytes(), offsets, n_threads=2)
    took = time.perf_counter() - t0
    stop.set()
    t.join()
    # about one tick per ms if the lock is free; the old scanner allowed ~3
    assert len(ticks) > 0.2 * took * 1000, (len(ticks), took)


def test_scan_n_threads_does_not_enter_the_result():
    rng = np.random.default_rng(10)
    widths = [6, 9, 15]
    pssms = rng.normal(size=5 * sum(widths)).tolist()
    seqs = rng.integers(0, 5, size=3000 * 120).astype(np.uint8)
    offsets = list(range(0, seqs.size + 1, 120))
    ref = _piaso.scan_motifs_fwd(pssms, widths, [2.0, 3.0, 4.0], seqs.tobytes(), offsets, n_threads=1)
    for t in (2, 5, 0):
        got = _piaso.scan_motifs_fwd(pssms, widths, [2.0, 3.0, 4.0], seqs.tobytes(), offsets, n_threads=t)
        assert np.array_equal(np.asarray(got[0]), np.asarray(ref[0]), equal_nan=True)
        assert np.array_equal(np.asarray(got[1]), np.asarray(ref[1]))


def test_scan_refuses_inconsistent_input():
    with pytest.raises(ValueError, match="do not match"):
        _piaso.scan_motifs_fwd([0.0] * 10, [6], [1.0], b"\x00" * 10, [0, 10])


def test_scan_refuses_codes_above_n():
    with pytest.raises(ValueError, match="0..=4"):
        _piaso.scan_motifs_fwd([0.0] * 30, [6], [1.0], bytes([0, 1, 2, 3, 5, 0]), [0, 6])
