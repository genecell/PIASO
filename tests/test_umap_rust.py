"""PIASO's own UMAP (backend='rust', which backend='auto' takes): umap-learn's
graph and start, and a layout that is the same at any number of threads and in
any node order."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

try:
    from piaso import _piaso
    HAVE = hasattr(_piaso, "umap_layout_csr")
except Exception:  # pragma: no cover
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="Rust extension without umap_layout_csr")


def _blobs(n_per=300, k=5, dim=8, spread=4.0, noise=1.0, seed=0):
    import anndata as ad
    import pandas as pd
    import piaso
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dim)) * spread
    truth = np.repeat(np.arange(k), n_per)
    X = (centres[truth] + noise * rng.standard_normal((len(truth), dim))).astype(np.float32)
    a = ad.AnnData(obs=pd.DataFrame(index=np.arange(len(X)).astype(str)), obsm={"X_e": X})
    piaso.tl.neighbors(a, use_rep="X_e", n_neighbors=15, random_state=0)
    return a, X, truth


def _ring(n):
    """A symmetric graph with chords, fast to build at any size: node i is
    joined to i +- 1, 2, 5 and 11, each edge's weight a function of its two
    ends (so both directions carry the same value)."""
    i = np.arange(n, dtype=np.int64)
    r = np.concatenate([np.concatenate([i, (i + d) % n]) for d in (1, 2, 5, 11)])
    c = np.concatenate([np.concatenate([(i + d) % n, i]) for d in (1, 2, 5, 11)])
    lo, hi = np.minimum(r, c), np.maximum(r, c)
    w = ((lo * 2654435761 + hi * 40503) % 1000).astype(np.float32) / np.float32(1000) * np.float32(0.8) + np.float32(0.2)
    G = sp.csr_matrix((w.astype(np.float32), (r, c)), shape=(n, n))
    G.sort_indices()
    return G


def _start(n, dim=2):
    t = np.arange(n, dtype=np.float64)
    return np.ascontiguousarray(np.stack([np.sin(t * 0.37), np.cos(t * 0.11)][:dim], 1) * 5 + 5, dtype=np.float32)


def _layout(G, init, threads, epochs=12, blocks=128, seed=3, identity=None):
    return _piaso.umap_layout_csr(np.ascontiguousarray(G.indptr, dtype=np.int64), np.ascontiguousarray(G.indices, dtype=np.int32),
                                  np.ascontiguousarray(G.data, dtype=np.float32), init, epochs, 1.577, 0.895, gamma=1.0,
                                  initial_alpha=1.0, negative_sample_rate=5.0, seed=seed, identity=identity,
                                  n_threads=threads, blocks=blocks)


def test_the_graph_equals_umap_learns():
    umap_ = pytest.importorskip("umap.umap_")
    from piaso.tools import _umap_rust as ur
    a, X, _ = _blobs(n_per=200, spread=1.5)
    idx, dst = a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"]
    ref, s_ref, r_ref = umap_.fuzzy_simplicial_set(X, 15, np.random.RandomState(0), "euclidean", knn_indices=idx, knn_dists=dst)
    ref = ref.tocsr(); ref.sort_indices()
    G, s, r = ur.fuzzy_graph(idx, dst)
    assert np.array_equal(G.indptr, ref.indptr) and np.array_equal(G.indices, ref.indices)
    assert np.abs(G.data - ref.data).max() <= 1e-6
    assert np.abs(s - s_ref).max() <= 1e-6 and np.abs(r - r_ref).max() <= 1e-6
    assert abs(G - G.T).max() == 0


def test_the_layout_is_the_same_at_every_thread_count():
    n = 40000                      # above the one-thread thresholds
    G = _ring(n)
    init = _start(n)
    one = _layout(G, init, 1)
    assert np.isfinite(one).all() and not np.array_equal(one, init)
    for t in (4, 20):
        assert np.array_equal(one, _layout(G, init, t))


def test_the_layout_does_not_depend_on_the_order_of_the_nodes():
    n = 40000
    G = _ring(n)
    init = _start(n)
    base = _layout(G, init, 4)
    perm = np.random.default_rng(1).permutation(n)        # position p holds node perm[p]
    P = G[perm][:, perm].tocsr(); P.sort_indices()
    out = _layout(P, np.ascontiguousarray(init[perm]), 4, identity=perm.astype(np.int64))
    back = np.empty_like(out); back[perm] = out
    assert np.array_equal(back, base)


_M64 = (1 << 64) - 1
_GOLD = 0x9E3779B97F4A7C15


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _hash6(seed, kind, epoch, a, b, c):
    h = _mix(seed ^ _GOLD)
    h = _mix(h ^ kind)
    h = _mix((h + epoch * 0xD1B54A32D192ED03) & _M64)
    h = _mix(h ^ ((a * _GOLD) & _M64))
    h = _mix((h + b * 0xC2B2AE3D27D4EB4F) & _M64)
    return _mix(h ^ ((c * 0x165667B19E3779F9) & _M64))


_KIND_BLOCK, _KIND_NEGATIVE = 0x626c_6f63_6b31, 0x6e65_6761_7469


def _stages(p):
    """The diagonal pairs of blocks, then the circle method's rounds."""
    out = [[(x, x) for x in range(p)]]
    others = list(range(1, p))
    for _ in range(p - 1):
        ring = [0] + others
        out.append([(min(ring[i], ring[p - 1 - i]), max(ring[i], ring[p - 1 - i])) for i in range(p // 2)])
        others = [others[-1]] + others[:-1]
    return out


def _negatives(k, nf, f, r):
    """umap-learn's count of negative samples at an edge's k-th firing, in epoch nf."""
    import math
    fr = f * r
    here = math.floor(nf * fr)
    if k <= 1:
        before = 1
    else:
        target = k - 1.0
        x = float(math.ceil(target / f))
        while x > 1.0 and math.floor((x - 1.0) * f) >= target:
            x -= 1.0
        while math.floor(x * f) < target:
            x += 1.0
        before = math.floor(x * fr)
    return max(here - before, 0)


def _reference_layout(G, init, n_epochs, a, b, gamma, alpha0, nsr, seed, blocks):
    """The layout as its module documents it, one edge at a time in Python:
    the nodes in blocks by a hash of their names (here their positions); every
    pair of blocks once an epoch, stage by stage, the edges of a pair in the
    order of their ends; an edge of rate w / w_max fires in epoch n when
    floor(n w / w_max) grows, each of its two directed copies moving its own
    node at once and adding the pull on the other end to that node's sum, then
    pushing its node from negative samples drawn by hash and read from the
    epoch's start; the sums added when the epoch ends; the step falling
    linearly."""
    import math
    f32 = np.float32
    n, dim = init.shape
    pos = init.astype(f32).copy()
    w = G.data.astype(f32)
    wmax = w.max()
    block = [_mix(v ^ _KIND_BLOCK) % blocks for v in range(n)]
    pairs = {}
    for i in range(n):
        for e in range(G.indptr[i], G.indptr[i + 1]):
            j = int(G.indices[e])
            rate = float(w[e] / wmax)
            if i < j and rate > 0:
                pairs.setdefault((min(block[i], block[j]), max(block[i], block[j])), []).append((i, j, rate))
    for v in pairs.values():
        v.sort()
    clip = lambda x: min(4.0, max(-4.0, x))
    acc = np.zeros((n, dim))
    snap = pos.copy()

    def fire(i, j, h0, m, alpha):
        d2 = 0.0
        for t in range(dim):
            d = pos[i, t] - pos[j, t]
            d2 += float(d * d)
        coeff = -2.0 * a * b * (d2 ** b / d2) / (a * d2 ** b + 1.0) if d2 > 0 else 0.0
        for t in range(dim):
            g = clip(coeff * float(pos[i, t] - pos[j, t]))
            pos[i, t] = f32(float(pos[i, t]) + g * alpha)
            acc[j, t] -= g * alpha
        for s in range(m):
            k = _mix((h0 + (s + 1) * _GOLD) & _M64) % n
            if k == i:
                continue
            d2 = 0.0
            for t in range(dim):
                d = pos[i, t] - snap[k, t]
                d2 += float(d * d)
            if d2 > 0:
                coeff = 2.0 * gamma * b / ((0.001 + d2) * (a * d2 ** b + 1.0))
                for t in range(dim):
                    g = clip(coeff * float(pos[i, t] - snap[k, t]))
                    pos[i, t] = f32(float(pos[i, t]) + g * alpha)

    alpha = alpha0
    for ep in range(n_epochs):
        if ep > 0:
            snap[:] = pos
            for stage in _stages(blocks):
                for key in stage:
                    for u, v, f in pairs.get(key, []):
                        now = math.floor(ep * f)
                        if now <= math.floor((ep - 1) * f):
                            continue
                        m = _negatives(float(now), float(ep), f, nsr)
                        fire(u, v, _hash6(seed, _KIND_NEGATIVE, ep, u, v, 0), m, alpha)
                        fire(v, u, _hash6(seed, _KIND_NEGATIVE, ep, u, v, 1), m, alpha)
            pos[:] = (pos.astype(np.float64) + acc).astype(f32)
            acc[:] = 0.0
        alpha = alpha0 * (1.0 - ep / n_epochs)
    return pos


@pytest.mark.parametrize("blocks", [4, 64, 128])
def test_the_layout_is_its_specification(blocks):
    from piaso.tools import _umap_rust as ur
    a, _, _ = _blobs(n_per=12, k=4, spread=3.0)
    G, _, _ = ur.fuzzy_graph(a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"])
    G = ur.prune(G, 30)
    G.sort_indices()
    init = _start(G.shape[0])
    ref = _reference_layout(G, init, 30, 1.577, 0.895, 1.0, 1.0, 5.0, 11, blocks)
    ours = _layout(G, init, 1, epochs=30, blocks=blocks, seed=11)
    assert not np.array_equal(ours, init)
    assert np.array_equal(ours, ref)



def test_the_default_number_of_blocks_is_128():
    """The layout cuts the nodes into 128 blocks unless told otherwise; 64, the
    count of the first release, gives another layout, as another seed does."""
    from piaso.tools import _umap_rust as ur
    a, _, _ = _blobs(n_per=12, k=4, spread=3.0)
    G, _, _ = ur.fuzzy_graph(a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"])
    G = ur.prune(G, 30)
    G.sort_indices()
    init = _start(G.shape[0])
    default = _piaso.umap_layout_csr(np.ascontiguousarray(G.indptr, dtype=np.int64),
                                     np.ascontiguousarray(G.indices, dtype=np.int32),
                                     np.ascontiguousarray(G.data, dtype=np.float32), init, 30, 1.577, 0.895, seed=11,
                                     n_threads=1)
    assert np.array_equal(default, _layout(G, init, 1, epochs=30, blocks=128, seed=11))
    assert not np.array_equal(default, _layout(G, init, 1, epochs=30, blocks=64, seed=11))

def test_the_spectral_start_is_umap_learns_subspace():
    spectral = pytest.importorskip("umap.spectral")
    from scipy.sparse.csgraph import connected_components
    from piaso.tools import _umap_rust as ur
    a, X, _ = _blobs(n_per=300, spread=1.2)
    G, _, _ = ur.fuzzy_graph(a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"])
    g = ur.prune(G, ur.default_epochs(G.shape[0]))
    assert connected_components(g, directed=False)[0] == 1
    ref = spectral.spectral_layout(X, g.tocoo(), 2, np.random.RandomState(0))
    ours, info = ur.spectral_start(g, 2, data=X)
    q1, _ = np.linalg.qr(ref); q2, _ = np.linalg.qr(ours)
    assert np.linalg.svd(q1.T @ q2, compute_uv=False).min() > 0.99
    assert info["method"] == "chebyshev" and info["converged"]
    # deterministic, whatever the threads
    again, _ = ur.spectral_start(g, 2, data=X, n_threads=1)
    assert np.array_equal(ours, again)
    # each axis's entry of largest magnitude is positive
    for c in range(2):
        assert ours[np.argmax(np.abs(ours[:, c])), c] > 0


def test_the_start_depends_on_neither_the_seed_nor_the_order_of_the_nodes():
    from piaso.tools import _umap_rust as ur
    a, X, _ = _blobs(n_per=300, spread=1.2)
    idx, dst = a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"]
    _, i0 = ur.embed(X, idx, dst, random_state=0, return_info=True)
    _, i7 = ur.embed(X, idx, dst, random_state=7, return_info=True)
    # one start; the seeds differ by umap-learn's noise of 1e-4 alone (a coordinate of 10 at most)
    assert np.abs(i0["init"] - i7["init"]).max() < 5e-3
    g = i0["graph"]
    base, info = ur.spectral_start(g, 2, data=X)
    assert info["method"] == "chebyshev"
    perm = np.random.default_rng(2).permutation(g.shape[0])          # position p holds node perm[p]
    gp = g[perm][:, perm].tocsr(); gp.sort_indices()
    moved, _ = ur.spectral_start(gp, 2, data=X[perm], identity=perm)
    back = np.empty_like(moved); back[perm] = moved
    # the same vectors and signs, up to the solver's tolerance (another order rounds the sums otherwise)
    assert np.abs(back - base).max() < 1e-3


def test_the_filter_in_one_call_equals_its_steps():
    from piaso.tools import _umap_rust as ur
    G = _ring(40000)
    A = ur._NormalizedGraphOps(G, 4)
    rng = np.random.default_rng(5)
    X = np.linalg.qr(rng.standard_normal((G.shape[0], 7)))[0]
    AX = A(X)
    c = e = 0.21
    sigma1 = e / (0.93 - c)
    sigma = sigma1
    Y0, Y1 = X, (AX - c * X) * (sigma1 / e)
    for _ in range(2, 17):
        s_next = 1.0 / (2.0 / sigma1 - sigma)
        Y0, Y1 = Y1, A.filter_step(Y1, Y0, c, 2.0 * s_next / e, sigma * s_next)
        sigma = s_next
    assert np.array_equal(A.chebyshev(X, AX, c, e, sigma1, 16), Y1)
    A1 = ur._NormalizedGraphOps(G, 1)
    assert np.array_equal(A1.chebyshev(X, AX, c, e, sigma1, 16), Y1)


def test_the_fused_filter_step_rounds_as_numpy_does():
    G = _ring(40000)
    rng = np.random.default_rng(4)
    X, Z = rng.standard_normal((G.shape[0], 7)), rng.standard_normal((G.shape[0], 7))
    s = 1.0 / np.sqrt(np.asarray(G.sum(axis=0), dtype=np.float64).ravel())
    args = (G.indptr.astype(np.int64), G.indices.astype(np.int32), G.data.astype(np.float64), s)
    c, k1, k2 = 0.4987, 2.0 * 0.731 / 0.4987, 0.9 * 0.731
    plain = 0.5 * _piaso.umap_graph_matmat(*args, X, 1.0, 4)
    fused = _piaso.umap_graph_matmat(*args, X, 1.0, 4, (Z, 0.5, c, k1, k2))
    assert np.array_equal(fused, (plain - c * X) * k1 - k2 * Z)
    assert np.array_equal(fused, _piaso.umap_graph_matmat(*args, X, 1.0, 1, (Z, 0.5, c, k1, k2)))


def test_the_filtered_start_agrees_with_a_dense_eigendecomposition(monkeypatch):
    from piaso.tools import _umap_rust as ur
    a, X, _ = _blobs(n_per=300, spread=1.2)
    G, _, _ = ur.fuzzy_graph(a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"])
    g = ur.prune(G, ur.default_epochs(G.shape[0]))
    filtered, _ = ur.spectral_start(g, 2, data=X)
    monkeypatch.setattr(ur, "START_DENSE", 10 ** 6)
    dense, info = ur.spectral_start(g, 2, data=X)
    assert info["method"] == "dense"
    q1, _ = np.linalg.qr(filtered); q2, _ = np.linalg.qr(dense)
    assert np.linalg.svd(q1.T @ q2, compute_uv=False).min() > 0.999


def test_quality_is_close_to_umap_learns():
    umap = pytest.importorskip("umap")
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from piaso.tools import _umap_rust as ur
    a, X, truth = _blobs(n_per=300, spread=2.0)
    idx, dst = a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"]

    def keep(Y):
        nb = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        return np.mean([np.isin(nb[i], idx[i, 1:]).mean() for i in range(len(Y))])

    ours = ur.embed(X, idx, dst, random_state=0)
    ref = umap.UMAP(min_dist=0.5, spread=1.0, random_state=0, precomputed_knn=(idx, dst, None)).fit_transform(X)
    assert keep(ours) >= 0.9 * keep(ref)
    assert silhouette_score(ours, truth) >= silhouette_score(ref, truth) - 0.1


def test_a_graph_in_several_pieces():
    from scipy.sparse.csgraph import connected_components
    from piaso.tools import _umap_rust as ur
    a, X, truth = _blobs(n_per=200, k=3, spread=30.0, noise=0.5)
    idx, dst = a.uns["_neighbors_knn_indices"], a.uns["_neighbors_knn_dists"]
    G, _, _ = ur.fuzzy_graph(idx, dst)
    assert connected_components(G, directed=False)[0] >= 2
    emb = ur.embed(X, idx, dst, random_state=0)
    assert np.isfinite(emb).all()
    cent = np.array([emb[truth == c].mean(0) for c in range(3)])
    within = max(np.linalg.norm(emb[truth == c] - cent[c], axis=1).mean() for c in range(3))
    between = min(np.linalg.norm(cent[i] - cent[j]) for i in range(3) for j in range(i + 1, 3))
    assert between > within


def test_tl_umap_rust_on_anndata():
    import piaso
    a, _, _ = _blobs(n_per=100)
    piaso.tl.umap(a, use_rep="X_e", backend="rust", n_threads=2)
    first = a.obsm["X_umap"].copy()
    assert first.shape == (500, 2) and first.dtype == np.float32
    piaso.tl.umap(a, use_rep="X_e", backend="rust", n_threads=1)
    assert np.array_equal(first, a.obsm["X_umap"])
    piaso.tl.umap(a, use_rep="X_e", backend="rust", random_state=7)
    assert not np.array_equal(first, a.obsm["X_umap"])


def test_the_default_is_the_rust_layout_when_a_knn_is_stored():
    import inspect
    import piaso
    assert inspect.signature(piaso.tl.umap).parameters["backend"].default == "auto"
    a, _, _ = _blobs(n_per=100)
    piaso.tl.umap(a, use_rep="X_e", n_threads=2)
    default = a.obsm["X_umap"].copy()
    piaso.tl.umap(a, use_rep="X_e", backend="rust", n_threads=3)
    assert np.array_equal(default, a.obsm["X_umap"])


def test_tl_umap_rust_on_a_cytome(tmp_path):
    import pandas as pd
    import cytome
    import piaso
    _, X, _ = _blobs(n_per=60)
    ds = cytome.create(str(tmp_path / "u.cytome"))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(len(X)), "barcode": [f"c{i}" for i in range(len(X))]}))
    ds.add_embedding("X_e", X)
    ds.flush()
    piaso.tl.neighbors(ds, use_rep="X_e", n_neighbors=15, random_state=0)
    piaso.tl.umap(ds, use_rep="X_e", n_threads=2)
    emb = np.asarray(ds.embeddings["X_umap"])
    assert emb.shape == (len(X), 2) and np.isfinite(emb).all()
    # the default took the Rust layout, and the store says so
    import json
    row = ds._conn.execute("SELECT p.parameters FROM embedding_meta m JOIN _provenance p ON p.id = m.provenance_id "
                           "WHERE m.array_name = 'X_umap'").fetchone()
    assert json.loads(row[0])["backend"] == "rust"


def test_neighbors_can_build_its_graph_in_rust():
    import piaso
    a, _, _ = _blobs(n_per=80)
    ref = a.obsp["connectivities"].tocsr().copy()
    piaso.tl.neighbors(a, use_rep="X_e", n_neighbors=15, random_state=0, graph_backend="rust")
    ours = a.obsp["connectivities"].tocsr()
    assert abs(ours - ref).max() <= 1e-6
    with pytest.raises(ValueError, match="graph_backend"):
        piaso.tl.neighbors(a, use_rep="X_e", graph_backend="bogus")


def test_ctrl_c_stops_a_layout():
    import os, signal, threading, time
    n = 40000
    G = _ring(n)
    init = _start(n)
    t0 = time.time()
    _layout(G, init, 0, epochs=5)
    one = (time.time() - t0) / 5
    threading.Timer(0.3, os.kill, args=(os.getpid(), signal.SIGINT)).start()
    t0 = time.time()
    with pytest.raises(KeyboardInterrupt):
        _layout(G, init, 0, epochs=100000)
    assert time.time() - t0 < 0.3 + 5 * one + 2.0
    assert np.isfinite(_layout(G, init, 0, epochs=2)).all()


def test_arguments_are_checked():
    import piaso
    n = 200
    G = _ring(n)
    init = _start(n)
    asym = G.tolil(); asym[0, 1] = 0.5; asym[1, 0] = 0.25; asym = asym.tocsr(); asym.sort_indices()
    with pytest.raises(ValueError, match="not symmetric"):
        _layout(asym, init, 1)
    with pytest.raises(ValueError, match="permutation"):
        _layout(G, init, 1, identity=np.zeros(n, dtype=np.int64))
    for bad_blocks in (0, 3, 2048):
        with pytest.raises(ValueError, match="blocks"):
            _layout(G, init, 1, blocks=bad_blocks)
    loop = G.tolil(); loop[5, 5] = 0.5; loop = loop.tocsr(); loop.sort_indices()
    with pytest.raises(ValueError, match="itself"):
        _layout(loop, init, 1)
    with pytest.raises(ValueError):
        _layout(G, init[:10], 1)
    bad = init.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _layout(G, bad, 1)
    # an edge listed twice (both directions, same weight, so the graph is symmetric)
    ip, ix, d = G.indptr.astype(np.int64), G.indices.astype(np.int32), G.data.astype(np.float32)
    for r, c in ((0, 1), (1, 0)):
        at = ip[r + 1]
        ix, d = np.insert(ix, at, c), np.insert(d, at, G[r, c])
        ip = ip.copy(); ip[r + 1:] += 1
    with pytest.raises(ValueError, match="twice"):
        _layout(sp.csr_matrix((d, ix, ip), shape=G.shape), init, 1)
    # zero epochs return the start
    assert np.array_equal(_layout(G, init, 1, epochs=0), init)
    a, _, _ = _blobs(n_per=40, k=2)
    with pytest.raises(ValueError, match="backend"):
        piaso.tl.umap(a, use_rep="X_e", backend="bogus")
    with pytest.raises(ValueError, match="random_state"):
        piaso.tl.umap(a, use_rep="X_e", backend="rust", random_state=-1)
    with pytest.raises(ValueError, match="n_threads"):
        piaso.tl.umap(a, use_rep="X_e", backend="rust", n_threads=-2)
    del a.uns["_neighbors_knn_indices"]
    with pytest.raises(ValueError, match="kNN"):
        piaso.tl.umap(a, use_rep="X_e", backend="rust")
