"""PIASO's own Leiden (backend='rust') against igraph's, and its contract."""
from __future__ import annotations

import threading

import numpy as np
import pytest
import scipy.sparse as sp

try:
    from piaso import _piaso
    HAVE = hasattr(_piaso, "leiden_csr")
except Exception:  # pragma: no cover
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="Rust extension without leiden_csr")


def _blobs(n_per=150, k=6, dim=8, noise=1.0, seed=0):
    import anndata as ad
    import piaso
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dim)) * 4
    truth = np.repeat(np.arange(k), n_per)
    X = (centres[truth] + noise * rng.standard_normal((len(truth), dim))).astype(np.float32)
    a = ad.AnnData(X=np.zeros((len(truth), 1), dtype=np.float32))
    a.obsm["X_e"] = X
    piaso.tl.neighbors(a, use_rep="X_e", n_neighbors=15, random_state=0)
    return a, truth


def _modularity(adj, labels, res=1.0):
    import igraph as ig
    c = sp.triu(sp.csr_matrix(adj), k=1).tocoo()
    g = ig.Graph(n=adj.shape[0], edges=np.column_stack([c.row, c.col]).tolist(), edge_attrs={"weight": c.data.tolist()})
    return g.modularity(list(np.asarray(labels).astype(int)), weights="weight", resolution=res)


def test_finds_clear_clusters_as_igraph_does():
    import piaso
    from sklearn.metrics import adjusted_rand_score as ari
    a, truth = _blobs(noise=0.5)
    piaso.tl.leiden(a, key_added="ig", backend="igraph")
    piaso.tl.leiden(a, key_added="rs", backend="rust")
    assert ari(truth, a.obs["rs"]) > 0.99 and ari(a.obs["ig"], a.obs["rs"]) > 0.99


def test_modularity_is_igraphs_on_a_noisy_graph():
    import piaso
    a, _ = _blobs(n_per=300, k=10, noise=2.5, seed=3)
    adj = a.obsp["connectivities"]
    qi, qr = [], []
    for s in range(4):
        piaso.tl.leiden(a, key_added="ig", backend="igraph", random_state=s)
        piaso.tl.leiden(a, key_added="rs", backend="rust", random_state=s)
        qi.append(_modularity(adj, a.obs["ig"])); qr.append(_modularity(adj, a.obs["rs"]))
    assert np.mean(qr) >= np.mean(qi) - 0.003, (qi, qr)


def test_one_seed_one_partition_and_no_seed_needs_none():
    import piaso
    a, _ = _blobs(noise=2.0, seed=1)
    piaso.tl.leiden(a, key_added="a", backend="rust", random_state=7)
    piaso.tl.leiden(a, key_added="b", backend="rust", random_state=7)
    piaso.tl.leiden(a, key_added="c", backend="rust", random_state=None)
    piaso.tl.leiden(a, key_added="d", backend="rust", random_state=None)
    assert (a.obs["a"] == a.obs["b"]).all() and (a.obs["c"] == a.obs["d"]).all()
    # the largest community is "0"
    counts = a.obs["a"].value_counts()
    assert counts.index[0] == "0"


def test_threads_run_side_by_side_with_the_same_labels():
    import piaso
    data = [_blobs(noise=2.0, seed=s)[0] for s in range(4)]
    for a in data:
        piaso.tl.leiden(a, key_added="alone", backend="rust", random_state=3)
    th = [threading.Thread(target=piaso.tl.leiden, args=(a,), kwargs=dict(key_added="together", backend="rust", random_state=3))
          for a in data]
    [t.start() for t in th]; [t.join() for t in th]
    assert all((a.obs["alone"] == a.obs["together"]).all() for a in data)


def test_until_stable_and_resolution():
    import piaso
    a, _ = _blobs(noise=1.5)
    piaso.tl.leiden(a, key_added="r1", backend="rust", n_iterations=-1)
    piaso.tl.leiden(a, key_added="r3", backend="rust", n_iterations=-1, resolution=3.0)
    assert a.obs["r3"].nunique() > a.obs["r1"].nunique()


def test_a_cytome_gets_its_labels(tmp_path):
    import anndata as ad
    import cytome
    import piaso
    a, truth = _blobs(noise=0.5)
    X = sp.random(a.n_obs, 20, density=0.2, format="csr", dtype=np.float32, random_state=0)
    b = ad.AnnData(X=X)
    b.obs_names = [f"c{i}" for i in range(a.n_obs)]
    b.var_names = [f"g{j}" for j in range(20)]
    ds = cytome.from_anndata(b, output=str(tmp_path / "l.cytome"))
    ds.add_graph("connectivities", a.obsp["connectivities"]); ds.flush()
    piaso.tl.leiden(ds, key_added="lr", backend="rust", adjacency_key="connectivities")
    from sklearn.metrics import adjusted_rand_score as ari
    assert ari(truth, np.asarray(ds.cells["lr"]).astype(str)) > 0.99
    ds.close()


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="negative"):
        _piaso.leiden_csr(np.array([0, 1, 2], dtype=np.int64), np.array([1, 0], dtype=np.int32),
                          np.array([-1.0, -1.0]), 2)
    with pytest.raises(ValueError, match="indptr"):
        _piaso.leiden_csr(np.array([0, 1], dtype=np.int64), np.array([1], dtype=np.int32), np.array([1.0]), 2)
    import piaso
    a, _ = _blobs()
    with pytest.raises(ValueError, match="backend"):
        piaso.tl.leiden(a, backend="leidenalg")


# --- parallel and deterministic: what must not enter the result ---------------------------------

def _csr(adj):
    A = sp.csr_matrix(adj).astype(np.float32)
    A.sort_indices()
    return A, np.ascontiguousarray(A.indptr, dtype=np.int64), np.ascontiguousarray(A.indices, dtype=np.int32)


@pytest.fixture(scope="module")
def noisy():
    """A graph above the size where the work is handed to the pool."""
    a, truth = _blobs(n_per=900, k=8, noise=2.2, seed=5)
    return a, truth


@pytest.mark.parametrize("seed", [0, 11, None])
def test_the_number_of_threads_does_not_enter_the_labels(noisy, seed):
    A, ip, ix = _csr(noisy[0].obsp["connectivities"])
    base, q = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 4, seed=seed, n_threads=1)
    for t in (2, 4, 20, 0):
        lab, q_t = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 4, seed=seed, n_threads=t)
        assert np.array_equal(base, lab), f"{t} threads"
        assert q_t == q


def test_the_layout_in_memory_does_not_enter_the_labels(noisy):
    A, ip, ix = _csr(noisy[0].obsp["connectivities"])
    a, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 4, seed=3, reorder=True, n_threads=4)
    b, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 4, seed=3, reorder=False, n_threads=4)
    assert np.array_equal(a, b)


def test_float64_weights_give_the_labels_of_float32(noisy):
    A, ip, ix = _csr(noisy[0].obsp["connectivities"])
    a, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 2, seed=1)
    b, _ = _piaso.leiden_csr(ip, ix, A.data.astype(np.float64), A.shape[0], 1.0, 2, seed=1)
    assert np.array_equal(a, b)


def test_another_seed_is_another_draw(noisy):
    A, ip, ix = _csr(noisy[0].obsp["connectivities"])
    labs = [_piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 2, seed=s)[0] for s in range(4)]
    assert any(not np.array_equal(labs[0], l) for l in labs[1:])


@pytest.mark.parametrize("resolution", [0.4, 1.0, 2.0, 3.0])
def test_modularity_is_igraphs_at_every_resolution(noisy, resolution):
    import piaso
    a = noisy[0]
    adj = a.obsp["connectivities"]
    qi, qr = [], []
    for s in range(3):
        piaso.tl.leiden(a, key_added="ig", backend="igraph", random_state=s, resolution=resolution)
        piaso.tl.leiden(a, key_added="rs", backend="rust", random_state=s, resolution=resolution)
        qi.append(_modularity(adj, a.obs["ig"], resolution)); qr.append(_modularity(adj, a.obs["rs"], resolution))
    assert np.mean(qr) >= np.mean(qi) - 0.003, (resolution, qi, qr)


def test_every_community_is_connected(noisy):
    from scipy.sparse.csgraph import connected_components
    A, ip, ix = _csr(noisy[0].obsp["connectivities"])
    for res in (1.0, 3.0):
        lab, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], res, 3, seed=0)
        C = A.tocoo()
        keep = lab[C.row] == lab[C.col]
        inside = sp.csr_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=A.shape)
        assert connected_components(inside, directed=False)[0] == len(np.unique(lab))


def test_the_reported_modularity_is_the_partitions(noisy):
    A, ip, ix = _csr(noisy[0].obsp["connectivities"])
    lab, q = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 2, seed=0)
    assert abs(q - _modularity(A, lab)) < 1e-6


def test_a_directed_graph_is_refused_and_says_why(noisy):
    A, ip, ix = _csr(sp.triu(noisy[0].obsp["connectivities"], k=1))
    with pytest.raises(ValueError, match="not symmetric"):
        _piaso.leiden_csr(ip, ix, A.data, A.shape[0])


def test_small_and_empty_graphs():
    e = (np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))
    lab, q = _piaso.leiden_csr(*e, 0)
    assert len(lab) == 0 and q == 0.0
    # isolated nodes keep to themselves; a self-loop is no edge
    A = sp.csr_matrix(np.array([[2, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
    lab, _ = _piaso.leiden_csr(np.ascontiguousarray(A.indptr, dtype=np.int64), np.ascontiguousarray(A.indices, dtype=np.int32),
                               A.data, 4, seed=0)
    assert list(lab) == [0, 0, 1, 2]


def test_tl_leiden_takes_n_threads_and_gives_the_same_labels(noisy):
    import piaso
    a = noisy[0]
    piaso.tl.leiden(a, key_added="t1", backend="rust", n_threads=1, n_iterations=3)
    piaso.tl.leiden(a, key_added="t8", backend="rust", n_threads=8, n_iterations=3)
    piaso.tl.leiden(a, key_added="auto", n_iterations=3)
    assert (a.obs["t1"] == a.obs["t8"]).all() and (a.obs["t1"] == a.obs["auto"]).all()


@pytest.mark.parametrize("resolution", [0.3, 1.0, 3.0, 10.0])
def test_the_rounds_end_on_their_own(noisy, resolution):
    """Nodes that join one community in the same round reckon without each other; untested,
    two of them joined, found the community too heavy, left, and joined again, to the guard."""
    import warnings
    A, ip, ix = _csr(noisy[0].obsp["connectivities"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for seed in range(6):
            _piaso.leiden_csr(ip, ix, A.data, A.shape[0], resolution, 5, seed=seed)


# --- what a caller may do around a call ---------------------------------------------------------

def _big(n=300_000, k=8, seed=0):
    """A symmetric graph large enough to be handed to several threads and to run for seconds."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = (rows + rng.integers(1, 200, size=n * k)) % n
    w = rng.random(n * k).astype(np.float32) * 0.9 + 0.05
    A = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.sum_duplicates()
    return A


def test_a_forked_child_clusters_too():
    """One pool shared by the process outlived a fork without its threads, and the child waited
    for them for ever."""
    import os, signal, time
    A = _big(60_000)
    ip, ix = np.ascontiguousarray(A.indptr, dtype=np.int64), np.ascontiguousarray(A.indices, dtype=np.int32)
    a, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 2, seed=0)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            b, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 2, seed=0)
            os.write(w, b"same" if np.array_equal(a, b) else b"differ")
        finally:
            os._exit(0)
    os.close(w)
    deadline = time.time() + 120
    while time.time() < deadline:
        done, _status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.2)
    else:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail("the forked child did not finish")
    assert os.read(r, 16) == b"same"


def test_a_matrix_stored_in_pieces_is_summed_not_refused():
    import piaso
    from piaso.tools._leiden import _leiden_rust
    # 0-1 and 1-2, every entry given twice as two halves, the second row out of order
    indptr = np.array([0, 2, 6, 8], dtype=np.int32)
    indices = np.array([1, 1, 2, 0, 0, 2, 1, 1], dtype=np.int32)
    data = np.array([.5, .5, .25, .4, .6, .25, .25, .25], dtype=np.float32)
    M = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
    assert not M.has_canonical_format
    labels = _leiden_rust(M, 1.0, 2, 0)
    whole = M.copy(); whole.sum_duplicates()
    assert list(labels) == list(_leiden_rust(whole, 1.0, 2, 0))


def test_the_callers_matrix_is_left_as_it_was(noisy):
    from piaso.tools._leiden import _leiden_rust
    A = sp.csr_matrix(noisy[0].obsp["connectivities"]).astype(np.float32).copy()
    lo, hi = A.indptr[5], A.indptr[6]
    A.indices[lo:hi] = A.indices[lo:hi][::-1].copy()
    A.data[lo:hi] = A.data[lo:hi][::-1].copy()
    A.has_sorted_indices = False
    before = (A.indptr.copy(), A.indices.copy(), A.data.copy())
    ordered = A.copy(); ordered.sort_indices()
    assert list(_leiden_rust(A, 1.0, 2, 0)) == list(_leiden_rust(ordered, 1.0, 2, 0))
    for x, y in zip(before, (A.indptr, A.indices, A.data)):
        assert np.array_equal(x, y)


def test_ctrl_c_stops_a_run():
    import os, signal, threading, time
    A = _big()
    ip, ix = np.ascontiguousarray(A.indptr, dtype=np.int64), np.ascontiguousarray(A.indices, dtype=np.int32)
    t0 = time.time()
    _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 1, seed=0)
    one = time.time() - t0
    threading.Timer(0.3, os.kill, args=(os.getpid(), signal.SIGINT)).start()
    t0 = time.time()
    with pytest.raises(KeyboardInterrupt):
        _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 200, seed=0)
    assert time.time() - t0 < 0.3 + 3 * one + 2.0, "the run went on long after the interrupt"
    # and the next call is a normal one
    lab, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 1, seed=0)
    assert len(lab) == A.shape[0]


def test_rayon_num_threads_is_honoured_and_changes_no_label(noisy, monkeypatch):
    A, ip, ix = _csr(_big(40_000))
    base, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 2, seed=0, n_threads=3)
    monkeypatch.setenv("RAYON_NUM_THREADS", "2")
    lab, _ = _piaso.leiden_csr(ip, ix, A.data, A.shape[0], 1.0, 2, seed=0)
    assert np.array_equal(base, lab)
