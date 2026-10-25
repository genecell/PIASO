"""UMAP without umap-learn: the fuzzy graph, the spectral start and the layout,
parallel and deterministic (``piaso.tl.umap(backend='rust')``, which the
default, ``backend='auto'``, takes whenever a kNN is stored).

What it computes is umap-learn's (0.5): the fuzzy simplicial set of the kNN
graph, the curve's ``a`` and ``b`` from ``min_dist`` and ``spread``, the edges
below ``max / n_epochs`` dropped, a spectral start scaled as umap-learn scales
it, and the epochs of its schedule. How it computes it differs in three ways,
all to make one graph and one seed give one embedding whatever the number of
threads:

- **The layout** runs each epoch as a schedule of node blocks: the nodes fall
  into 128 blocks by a hash of their names, every pair of blocks meets once an
  epoch, and the pairs that run at the same time share no node, so the
  threads together compute one sequential pass over the edges in a fixed
  order (``piaso._piaso.umap_layout_csr``; the rules are written out in the
  Rust module). A firing moves the edge's own node at once, and the pull it
  gives the other end is added up and applied when the epoch ends
  (umap-learn moves the other end at once). umap-learn applies every move at
  once in one stream of random numbers, which is why a seed makes it
  single-threaded.
- **The spectral start** is the leading eigenvectors of the normalized graph
  ``D^-1/2 G D^-1/2``, the vectors umap-learn asks ARPACK for (the
  Laplacian's smallest), to ARPACK's tolerance on the same residual. They come
  from a Chebyshev-filtered subspace iteration: a small block, a polynomial of
  the graph's product in Rust (each row summed in order) that damps the rest of
  the spectrum (one Rust call per filter step), and the dense steps on one
  BLAS thread. A kNN graph's smallest Laplacian eigenvalues crowd together (on
  199,357 nuclei of the whole brain the first six lie within 2e-4 of each
  other), which is where a Krylov basis of bounded size stops short; the
  filter needs only the block. The start does not depend on the seed, as
  umap-learn's does not: the iteration begins from a block drawn per node
  from a hash of its name, and each vector's sign is fixed (its entry of
  largest magnitude positive), so two seeds start from one picture.
- **Random draws** come from hashes and seeded generators, never from timing:
  the layout's negative samples and the start's noise of 1e-4 (umap-learn's)
  from the seed; the start's block and the placement of tiny components from
  the nodes' names.

A graph in several connected components is laid out as umap-learn lays it out:
each component's own spectral start, placed at its component's position in a
small spectral layout of the components' centroids (or on the axes when there
are few components), scaled to half the distance to the nearest component.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

#: umap-learn's defaults that ``piaso.tl.umap`` does not expose.
LEARNING_RATE = 1.0
REPULSION_STRENGTH = 1.0
NEGATIVE_SAMPLE_RATE = 5.0
LOCAL_CONNECTIVITY = 1.0

#: The spectral start: umap-learn's tolerance, on ARPACK's residual of the
#: Laplacian (``||L x - lambda x|| <= tol * lambda``); the block's columns
#: beyond those wanted; the filter's degree; the most filter steps before it
#: gives up with a warning; graphs this small take a dense eigendecomposition.
START_TOL = 1e-4
START_EXTRA = 5
START_DEGREE = 64
START_MAX_STEPS = 100
START_DENSE = 1000

#: Contexts of the hashes that draw from the nodes' names.
_KIND_START = 0x7370656374
_KIND_META = 0x6d657461
_KIND_COMPONENT = 0x636f6d70
_GOLD = 0x9E3779B97F4A7C15


def _ext():
    from piaso import _piaso
    if not hasattr(_piaso, "umap_layout_csr"):
        raise ImportError("piaso's Rust extension has no umap_layout_csr; rebuild it, or use backend='umap-learn'")
    return _piaso


def available() -> bool:
    try:
        _ext()
        return True
    except Exception:
        return False


def _splitmix(z):
    """splitmix64's finalizer on an array of uint64, wrapping as Rust's does."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _hash_uniform(names, k: int, kind: int):
    """A number in [0, 1) for each node and each of ``k`` columns, from a hash
    of the node's name, the column and ``kind``: the same whatever the seed
    and wherever the node sits in memory."""
    ids = np.asarray(names, dtype=np.uint64).reshape(-1, 1)
    cols = np.arange(k, dtype=np.uint64).reshape(1, -1)
    with np.errstate(over="ignore"):
        z = _splitmix((ids * np.uint64(_GOLD)) ^ np.uint64(kind))
        z = _splitmix(z + (cols + np.uint64(1)) * np.uint64(0xD1B54A32D192ED03))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _fix_signs(V, names):
    """Each column's sign fixed: its entry of largest magnitude positive, and
    among equal magnitudes that of the node with the smallest name."""
    V = np.array(V, dtype=np.float64, copy=True)
    names = np.asarray(names)
    for c in range(V.shape[1]):
        mag = np.abs(V[:, c])
        top = np.flatnonzero(mag == mag.max())
        if V[top[np.argmin(names[top])], c] < 0:
            V[:, c] = -V[:, c]
    return V


def fuzzy_graph(knn_indices, knn_dists, local_connectivity=LOCAL_CONNECTIVITY, n_threads=0):
    """UMAP's fuzzy simplicial set (CSR, float32): umap-learn's
    ``fuzzy_simplicial_set`` with ``set_op_mix_ratio=1``, from a kNN whose first
    column is each node itself."""
    ext = _ext()
    idx = np.ascontiguousarray(knn_indices)
    if idx.dtype not in (np.int32, np.int64):
        idx = idx.astype(np.int64)
    dist = np.ascontiguousarray(knn_dists, dtype=np.float32)
    ip, ix, data, sigmas, rhos = ext.umap_fuzzy_graph(idx, dist, local_connectivity=float(local_connectivity),
                                                       n_threads=int(n_threads))
    n = dist.shape[0]
    return sp.csr_matrix((data, ix, ip), shape=(n, n)), sigmas, rhos


def find_ab_params(spread: float, min_dist: float):
    """umap-learn's fit of ``1 / (1 + a x^(2b))`` to the offset exponential."""
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def default_epochs(n: int) -> int:
    return 500 if n <= 10000 else 200


def prune(graph, n_epochs: int):
    """umap-learn drops the edges whose weight is under ``max / n_epochs``
    (they would never fire)."""
    g = graph.tocsr(copy=True)
    if g.nnz:
        g.data[g.data < (g.data.max() / float(n_epochs))] = 0.0
        g.eliminate_zeros()
    return g


class _NormalizedGraphOps:
    """``A = (I + D^-1/2 G D^-1/2) / 2``: symmetric, eigenvalues in [0, 1],
    the top one 1 with the vector ``sqrt(degree)``; ``A``'s eigenvalue ``t``
    is the Laplacian's ``2 (1 - t)``."""

    def __init__(self, graph, n_threads):
        g = graph.tocsr()
        self.indptr = np.ascontiguousarray(g.indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(g.indices, dtype=np.int32)
        self.data = np.ascontiguousarray(g.data, dtype=np.float64)
        deg = np.asarray(g.sum(axis=0), dtype=np.float64).ravel()
        self.scale = 1.0 / np.sqrt(np.maximum(deg, np.finfo(np.float64).tiny))
        self.trivial = np.sqrt(deg) / np.linalg.norm(np.sqrt(deg))
        self.n = g.shape[0]
        self.n_threads = int(n_threads)
        self._ext = _ext()

    def __call__(self, X):
        return 0.5 * self._ext.umap_graph_matmat(self.indptr, self.indices, self.data, self.scale,
                                                 np.ascontiguousarray(X, dtype=np.float64), 1.0, self.n_threads)

    def filter_step(self, X, Z, c, k1, k2):
        """``(A(X) - c X) k1 - k2 Z`` in one pass over the graph, rounded as
        the numpy expression would be."""
        return self._ext.umap_graph_matmat(self.indptr, self.indices, self.data, self.scale,
                                           np.ascontiguousarray(X, dtype=np.float64), 1.0, self.n_threads,
                                           (np.ascontiguousarray(Z, dtype=np.float64), 0.5, c, k1, k2))

    def chebyshev(self, X, AX, c, e, sigma1, degree):
        """The filter of ``degree`` on the block ``X`` (``AX = A(X)``): the
        steps of ``filter_step`` one after another, in one call."""
        return self._ext.umap_chebyshev_filter(self.indptr, self.indices, self.data, self.scale,
                                               np.ascontiguousarray(X, dtype=np.float64),
                                               np.ascontiguousarray(AX, dtype=np.float64), float(c), float(e),
                                               float(sigma1), int(degree), self.n_threads)

    def deflate(self, X):
        return X - np.outer(self.trivial, self.trivial @ X)


def _rayleigh_ritz(A, Y):
    """The block's Ritz pairs, largest first: ``(theta, X, AX)``."""
    Y, _ = np.linalg.qr(A.deflate(Y))
    AY = A(Y)
    H = Y.T @ AY
    theta, V = np.linalg.eigh(0.5 * (H + H.T))
    return theta[::-1], Y @ V[:, ::-1], AY @ V[:, ::-1]


def _connected_start(graph, dim: int, n_threads: int, identity=None):
    """The leading non-trivial eigenvectors of one connected graph's ``A``
    (``_NormalizedGraphOps``), each vector's sign fixed: ``(coords, info)``.

    Chebyshev-filtered subspace iteration (Zhou and Saad 2007): each step
    applies to the block the Chebyshev polynomial of degree ``START_DEGREE``
    that stays within [-1, 1] on [0, the block's smallest Ritz value] and grows
    fast above it, then takes the Ritz pairs of the filtered block. It stops
    when every wanted pair passes ARPACK's test on the Laplacian, as umap-learn
    asks of ARPACK. The block starts from a hash of the nodes' names
    (``identity``, default positions), not from the seed. The dense steps run
    on one BLAS thread, so with the graph's products in Rust the vectors do not
    depend on the number of threads.
    """
    import warnings
    from threadpoolctl import threadpool_limits
    A = _NormalizedGraphOps(graph, n_threads)
    n = A.n
    names = np.arange(n) if identity is None else np.asarray(identity)
    tiny = np.finfo(np.float64).tiny
    with threadpool_limits(limits=1):
        if n <= START_DENSE:
            # small: every eigenvector at once
            dense = A(np.eye(n))
            theta, V = np.linalg.eigh(0.5 * (dense + dense.T))
            return (_fix_signs(V[:, ::-1][:, 1:dim + 1], names),
                    {"method": "dense", "iterations": 0, "converged": True})
        k = min(dim + START_EXTRA, n - 1)
        theta, X, AX = _rayleigh_ritz(A, 2.0 * _hash_uniform(names, k, _KIND_START) - 1.0)
        residual = None
        for step in range(1, START_MAX_STEPS + 1):
            # the scaled three-term recurrence keeps the block's size bounded
            hi = max(float(theta[-1]), 1e-12)
            e, c = hi / 2.0, hi / 2.0
            sigma1 = e / (theta[0] - c)
            theta, X, AX = _rayleigh_ritz(A, A.chebyshev(X, AX, c, e, sigma1, START_DEGREE))
            R = AX[:, :dim] - X[:, :dim] * theta[:dim]
            residual = np.linalg.norm(R, axis=0) / np.maximum(1.0 - theta[:dim], tiny)
            if np.all(residual <= START_TOL):
                break
    converged = bool(np.all(residual <= START_TOL))
    if not converged:
        warnings.warn(f"umap: the spectral start stopped after {START_MAX_STEPS} filter steps with relative "
                      f"residuals {np.array2string(residual, precision=2)} (asked: {START_TOL}); the layout "
                      f"starts from these vectors", stacklevel=3)
    return _fix_signs(X[:, :dim], names), {"method": "chebyshev", "iterations": step, "converged": converged,
                                           "residual": residual.tolist(),
                                           "eigenvalues": (2.0 * (1.0 - theta[:dim])).tolist()}


def _component_meta(data, labels, n_comp: int, dim: int):
    """umap-learn's component placement: on the axes when there are few
    components, else a spectral layout of the centroids' affinities
    (exp(-d^2)), here by a dense eigendecomposition."""
    if n_comp <= 2 * dim:
        k = int(np.ceil(n_comp / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        return np.vstack([base, -base])[:n_comp]
    X = np.asarray(data, dtype=np.float64)
    cent = np.vstack([X[labels == c].mean(axis=0) for c in range(n_comp)])
    d2 = ((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    A = np.exp(-d2)
    deg = A.sum(1)
    Ls = np.eye(n_comp) - A / np.sqrt(np.outer(deg, deg))
    w, V = np.linalg.eigh(Ls)
    emb = _fix_signs(V[:, 1:dim + 1], np.arange(n_comp)) / np.sqrt(deg)[:, None]
    return emb / np.abs(emb).max()


def spectral_start(graph, dim: int, *, data=None, n_threads: int = 0, identity=None):
    """umap-learn's spectral start of the (pruned) graph: ``(coords, info)``.
    It does not depend on the seed; ``identity`` names the nodes (default:
    their positions)."""
    from scipy.sparse.csgraph import connected_components
    n = graph.shape[0]
    names = np.arange(n) if identity is None else np.asarray(identity)
    n_comp, labels = connected_components(graph, directed=False)
    info = {"n_components": int(n_comp)}
    if n_comp == 1:
        coords, sinfo = _connected_start(graph, dim, n_threads, names)
        info.update(sinfo)
        return coords, info
    meta = _component_meta(data if data is not None else _hash_uniform(names, dim, _KIND_META), labels, n_comp, dim)
    out = np.empty((n, dim), dtype=np.float64)
    g = graph.tocsr()
    for c in range(n_comp):
        members = np.flatnonzero(labels == c)
        dists = np.sqrt(((meta - meta[c]) ** 2).sum(1))
        data_range = dists[dists > 0].min() / 2.0 if np.any(dists > 0) else 1.0
        if len(members) < 2 * dim or len(members) <= dim + 1:
            u = _hash_uniform(names[members], dim, _KIND_COMPONENT)
            out[members] = (2.0 * u - 1.0) * data_range + meta[c]
        else:
            sub = g[members][:, members]
            emb, _ = _connected_start(sub, dim, n_threads, names[members])
            out[members] = emb * (data_range / np.abs(emb).max()) + meta[c]
    return out, info


def scale_start(coords, seed: int):
    """umap-learn's ``noisy_scale_coords`` (largest coordinate 10, then normal
    noise of 0.0001) and its rescale of every dimension to [0, 10]."""
    c = np.asarray(coords, dtype=np.float64)
    c = (c * (10.0 / np.abs(c).max())).astype(np.float32)
    rng = np.random.default_rng([int(seed), 0x6e6f6973])
    c = c + rng.normal(scale=0.0001, size=c.shape).astype(np.float32)
    lo, hi = c.min(0), c.max(0)
    return np.ascontiguousarray(10.0 * (c - lo) / (hi - lo), dtype=np.float32)


def embed(data, knn_indices, knn_dists, n_components=2, min_dist=0.5, spread=1.0, random_state=42, n_threads=0,
          n_epochs=None, return_info=False):
    """The whole of ``piaso.tl.umap(backend='rust')``: graph, start, layout."""
    import time
    seed = 0 if random_state is None else int(random_state)
    if seed < 0:
        raise ValueError("random_state must be a non-negative integer or None")
    t0 = time.time()
    graph, _, _ = fuzzy_graph(knn_indices, knn_dists, n_threads=n_threads)
    t_graph = time.time() - t0
    n = graph.shape[0]
    ne = default_epochs(n) if n_epochs is None else int(n_epochs)
    if ne < 1:
        raise ValueError("n_epochs must be at least 1")
    g = prune(graph, ne)
    a, b = find_ab_params(spread, min_dist)
    t1 = time.time()
    coords, sinfo = spectral_start(g, n_components, data=data, n_threads=n_threads)
    init = scale_start(coords, seed)
    t_start = time.time() - t1
    t2 = time.time()
    ext = _ext()
    emb = ext.umap_layout_csr(np.ascontiguousarray(g.indptr, dtype=np.int64), np.ascontiguousarray(g.indices, dtype=np.int32),
                              np.ascontiguousarray(g.data, dtype=np.float32), init, ne, a, b, gamma=REPULSION_STRENGTH,
                              initial_alpha=LEARNING_RATE, negative_sample_rate=NEGATIVE_SAMPLE_RATE, seed=seed,
                              n_threads=int(n_threads))
    t_layout = time.time() - t2
    if return_info:
        return emb, dict(graph=g, init=init, a=a, b=b, n_epochs=ne, t_graph=t_graph, t_start=t_start,
                         t_layout=t_layout, start=sinfo)
    return emb
