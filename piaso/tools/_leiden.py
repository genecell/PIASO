"""Leiden clustering: PIASO's own implementation, in its Rust extension, or igraph's.

Supports both AnnData and cytome.Dataset as input.
"""

import random as _random
import threading
import numpy as np
import pandas as pd

from ._neighbors import _is_cytome_dataset


# Guards igraph's process-global RNG; see the comment at its use site.
_IGRAPH_RNG_LOCK = threading.Lock()


def _cell_mask_bool(cell_mask, n, what="cell_mask"):
    """A boolean mask of length ``n`` from a boolean mask or from cell indices.

    ``cell_mask`` is documented as a boolean mask or sorted indices, but was
    cast with ``astype(bool)``: indices became a boolean array of their own
    length (an error, or, when as many indices as cells were given, a wrong
    mask).
    """
    arr = np.asarray(cell_mask)
    if arr.dtype == bool:
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"{what} length ({arr.shape[0] if arr.ndim else 0}) must match "
                             f"the graph ({n} rows).")
        return arr
    if arr.ndim != 1 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(f"{what} must be a boolean mask or an array of integer cell indices")
    idx = arr.astype(np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise IndexError(f"{what} holds indices outside 0..{n - 1}")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{what} holds a cell index more than once")
    out = np.zeros(n, dtype=bool)
    out[idx] = True
    return out


def _igraph_from_adjacency(adj):
    """An undirected igraph Graph and its edge weights from a symmetric sparse matrix.

    The upper triangle's stored non-zeros, in row-major order: the edges and
    weights the per-edge lookup ``adj[sources, targets]`` gave before, read
    straight from the matrix (that lookup and the Python list of edge tuples
    were the slow part on large graphs). A matrix with unsummed duplicates is
    summed on a copy first, so each pair is one edge.
    """
    import igraph as ig
    import scipy.sparse as _sp
    A = _sp.csr_matrix(adj)
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()
    coo = A.tocoo()
    keep = (coo.data != 0) & (coo.row <= coo.col)
    edges = np.column_stack([coo.row[keep], coo.col[keep]]).astype(np.int64, copy=False)
    g = ig.Graph(n=A.shape[0], edges=edges, directed=False)
    g.es['weight'] = coo.data[keep].tolist()
    return g


class _SeededRNG:
    """A seeded random number generator compatible with igraph's RNG interface."""

    def __init__(self, seed):
        self._rng = _random.Random(seed)

    def random(self):
        return self._rng.random()

    def randint(self, a, b):
        return self._rng.randint(a, b)

    def gauss(self, mu, sigma):
        return self._rng.gauss(mu, sigma)


def leiden(
    data,
    resolution=1.0,
    modality='RNA',
    n_iterations=10,
    random_state=42,
    key_added='leiden',
    adjacency_key=None,
    neighbors_key=None,
    knn_result=None,
    cell_mask=None,
    backend='auto',
    n_threads=0,
):
    """Leiden clustering of the neighbour graph.

    Reads the connectivities matrix and runs the Leiden algorithm: PIASO's
    own implementation, in its Rust extension, or igraph's
    ``community_leiden()``.

    Parameters
    ----------
    data : AnnData, cytome.Dataset, or the path of a cytome
        If AnnData: reads from obsp, stores in obs.
        If cytome.Dataset or a path: reads connectivities from cytome graphs,
        stores in cells.
    resolution : float
        Resolution parameter controlling cluster granularity: higher gives
        more and smaller clusters.
    n_iterations : int, default 10
        How many times the whole algorithm runs, each run starting from the
        partition the one before it ended with. A negative number repeats
        until a run changes nothing, which with random refinement can take
        long. More runs raise the modularity a little and make two seeds
        agree a little more; on 2.3 million cells 2 runs give 96 to 99 % of
        the partition of 10 in under half the time.
    random_state : int or None, default 42
        The seed. With ``backend='rust'`` one graph and one seed give one
        partition, on any machine and any number of threads; ``None``
        refines without a random choice. With ``backend='igraph'`` it seeds
        igraph's generator, and repeated calls on one machine agree.
    key_added : str
        Column name to store cluster labels.
    neighbors_key : str, optional
        Prefix of the neighbors graph to use (matches the ``key_added`` passed
        to ``piaso.tl.neighbors``). ``neighbors_key='SVD'`` reads
        ``'SVD_connectivities'``; ``None`` / ``'neighbors'`` reads the
        un-prefixed ``'connectivities'``. Mirrors ``umap``'s ``neighbors_key``.
    adjacency_key : str, optional
        Full graph name to read (escape hatch / overrides ``neighbors_key``).
        Defaults to the connectivities graph resolved from ``neighbors_key``.
    knn_result : dict, optional
        Result dict from neighbors() with 'connectivities'. Used for the
        in-memory ndarray / cell_mask path to avoid re-reading from disk.
    modality : str, default 'RNA'
        Cytome only: the modality the graph and the labels belong to.
    cell_mask : ndarray, optional
        Cytome only: boolean mask or sorted indices of the cells the graph was
        built on with the same mask in ``neighbors``.
    backend : {'auto', 'rust', 'igraph'}, default 'auto'
        ``'auto'`` is ``'rust'`` when the extension provides it, else
        ``'igraph'``. ``'rust'``: PIASO's Leiden, parallel and
        deterministic. One graph and one ``random_state`` give one
        partition: the number of threads, the cores of the machine and the
        layout of the graph in memory do not enter the result. Same
        objective and the same randomised refinement as igraph, and the
        same modularity; ``random_state=None`` refines greedily, with no
        random choice. It releases Python's lock while it runs. The graph
        must be symmetric, and a directed one is refused. The graph passed
        in is not modified. Ctrl-C stops a run.

        Everything in PIASO that clusters calls this function, so
        ``leiden_local`` and ``runGDR`` follow the same backend.

        *Changed in 1.2.6*: ``'rust'`` is the default, and the partitions
        differ from those of 1.2.5 by about as much as another seed would
        change them. Pass ``backend='igraph'`` for the partitions of earlier
        versions.
    n_threads : int, default 0
        ``'rust'`` only: threads to use. 0 uses all the cores this process
        may use (on a cluster: the job's allocation), or the number in the
        environment variable ``RAYON_NUM_THREADS`` if it is set. The labels
        are the same for every value. A graph of fewer than 32,768 nodes is
        clustered on one thread whatever is asked, which is faster there.

    Returns
    -------
    None, or np.ndarray when the labels cannot be written
        Nothing, on both backends: the labels are written to
        ``adata.obs[key_added]`` or to ``ds.cells[key_added]`` and read back
        from there. The two paths that **cannot** write still return the
        label array — ``cell_mask`` (the labels cover the masked cells only,
        so writing them full-length would mis-align) and ``data=None`` (no
        object to write to).

        *Changed in 1.2.4*: the AnnData path used to return the labels as
        well as writing them, so the same call had two different contracts
        depending on the backend and echoed an array of strings in a
        notebook. Read them from ``adata.obs[key_added]``.
    """
    # A store that cannot be written fails here, not after the work.
    from ..utils._cytome_compat import require_writable as _require_writable
    _require_writable(data, "piaso.tl.leiden")
    # A cytome path is as valid an input as it is for infog / runSVD / runGDR.
    # Without this a pipeline written entirely with a path worked for three
    # calls and then raised "'str' object has no attribute 'obsm'" on the
    # fourth, which reads like the caller passed the wrong type.
    if isinstance(data, str):
        from ._normalization import _open_cytome
        data = _open_cytome(data)

    from ._neighbors import _neighbors_graph_keys

    # Resolve the adjacency/connectivities graph name: explicit adjacency_key
    # wins; otherwise derive it from the neighbors_key prefix.
    if adjacency_key is None:
        if _is_cytome_dataset(data):
            # Accept both the modality-prefixed name new runs write and the
            # un-prefixed one older files have. Only a cytome has `.metadata`;
            # an AnnData keeps its graph in obsp under the plain key.
            from ._neighbors import _neighbors_graph_keys_for_read
            for _c, _d, _n in _neighbors_graph_keys_for_read(neighbors_key, modality):
                if data.metadata.get(_n) is not None:
                    adjacency_key = _c
                    break
            if adjacency_key is None:
                # Fall back to the UN-prefixed name, because that is what
                # neighbors() writes. Falling back to the prefixed one asked
                # for RNA_connectivities in a file that has 'connectivities',
                # and Leiden then clustered an empty graph into one group.
                adjacency_key = _neighbors_graph_keys(neighbors_key)[0]
        else:
            adjacency_key = _neighbors_graph_keys(neighbors_key)[0]

    # Load adjacency matrix
    if knn_result is not None and 'connectivities' in knn_result:
        adj = knn_result['connectivities']
    elif _is_cytome_dataset(data):
        adj = data.graphs[adjacency_key].to_sparse()
    else:
        adj = data.obsp[adjacency_key]

    # cell_mask: run Leiden on the masked subgraph only. Returns labels
    # of length n_masked. Caller is responsible for placing labels back
    # into the full-cell ordering (e.g. multi-batch GDR writes them
    # into a temp cells column at masked positions).
    if cell_mask is not None:
        cell_mask_arr = _cell_mask_bool(cell_mask, adj.shape[0])
        adj = adj[cell_mask_arr][:, cell_mask_arr]

    if backend == 'auto':
        try:
            from .. import _piaso as _ext
            backend = 'rust' if hasattr(_ext, 'leiden_csr') else 'igraph'
        except Exception:
            backend = 'igraph'
        if backend == 'igraph':
            import warnings
            warnings.warn(
                "leiden: the Rust extension has no leiden_csr, so igraph's Leiden runs instead "
                "(another partition, and not the same at every thread count). Rebuild or reinstall "
                "PIASO's extension, or pass backend='igraph' to silence this.",
                RuntimeWarning, stacklevel=2)
    if backend == 'rust':
        labels = _leiden_rust(adj, resolution, n_iterations, random_state, n_threads)
        return _store_labels(data, labels, key_added, cell_mask)
    if backend != 'igraph':
        raise ValueError(f"leiden: backend must be 'auto', 'rust' or 'igraph', got {backend!r}")
    import igraph as ig

    # Build undirected igraph graph from sparse adjacency.
    # Our connectivities (from fuzzy_simplicial_set) are symmetric,
    # so we take the upper triangle to build an undirected graph
    # without duplicate edges.
    g = _igraph_from_adjacency(adj)

    # igraph's RNG is PROCESS-GLOBAL: set_random_number_generator installs it
    # for the whole interpreter, not for this call. Two threads clustering
    # concurrently therefore reset each other's generator mid-run, and the
    # partitions stop being reproducible — measured on a 38-batch cytome as
    # ARI 0.976 against the serial result, while a 5-batch one happened to come
    # out identical and hid it.
    #
    # The lock makes the set-then-use pair atomic. Leiden is ~6% of a GDR
    # stage-1 batch, so serialising it costs little and keeps the SVD and KNN
    # around it parallel.
    with _IGRAPH_RNG_LOCK:
        ig.set_random_number_generator(_SeededRNG(random_state))

        # Run Leiden (igraph backend, aligned with scanpy's future default)
        partition = g.community_leiden(
            objective_function='modularity',
            weights='weight',
            resolution=resolution,
            n_iterations=n_iterations,
        )

    labels = np.array(partition.membership, dtype=str)
    return _store_labels(data, labels, key_added, cell_mask)


def _leiden_rust(adj, resolution, n_iterations, random_state, n_threads=0):
    """Labels from PIASO's Leiden on a symmetric sparse matrix, as strings."""
    import scipy.sparse as _sp
    from .. import _piaso
    A = _sp.csr_matrix(adj)
    if A.dtype not in (np.float32, np.float64):
        A = A.astype(np.float32)
    if not A.has_canonical_format:
        # Entries out of order, or one (row, column) stored in pieces whose
        # sum is what counts: put in order on a copy, never in the caller's
        # matrix. A matrix already in order is read in place.
        A = A.copy()
        A.sum_duplicates()
    seed = None if random_state is None else int(random_state) & 0xFFFFFFFFFFFFFFFF
    membership, _q = _piaso.leiden_csr(
        np.ascontiguousarray(A.indptr, dtype=np.int64),
        np.ascontiguousarray(A.indices, dtype=np.int32),
        np.ascontiguousarray(A.data), A.shape[0],
        float(resolution), int(n_iterations) if int(n_iterations) != 0 else 1, seed=seed,
        n_threads=int(n_threads))
    return np.asarray(membership).astype(str)


def _label_order(labels) -> list:
    """Numeric-aware category order, so the result is "0","1","2",...,"10"
    rather than the lexical "0","1","10","11",... that plotters fall back to."""
    _levels = list(dict.fromkeys(np.asarray(labels).tolist()))
    try:
        return [str(i) for i in sorted(int(x) for x in _levels)]
    except (ValueError, TypeError):
        return sorted(_levels)


def _store_labels(data, labels, key_added, cell_mask):
    """Write the labels where the caller expects them, or return them."""
    _order = _label_order(labels)

    # Store results.
    # When cell_mask is set, labels apply to masked cells only —
    # writing into ``data.cells[key_added]`` (which expects full-cell
    # length) would mis-align. Skip the write and return labels for
    # the caller to handle. Same when data is None (in-memory only:
    # caller passed knn_result and just wants the labels back).
    if cell_mask is None and data is not None:
        if _is_cytome_dataset(data):
            data.cells[key_added] = labels
            data.flush()
            # Persist the numeric category order so plots show 0,1,2,…,10.
            # Not wrapped in a silent except: the old `except Exception: pass`
            # here swallowed the KeyError from a case-mismatched column name
            # ("leiden" vs an existing "Leiden"), which left the STALE order
            # attached to the NEW labels until a plot noticed. Cytome resolves
            # the casing now; if set_categories still fails, say so — labels
            # without an order beat labels with a wrong one.
            try:
                data.set_categories(key_added, order=_order)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"leiden: labels written to cells[{key_added!r}] but the "
                    f"category order could not be stored ({e}). Plots will "
                    f"fall back to a default order; run "
                    f"ds.set_categories({key_added!r}, order=...) to set one.",
                    stacklevel=2)
            # Strict / self-contained: labels live on the cytome.
            return None
        else:
            data.obs[key_added] = pd.Categorical(labels, categories=_order)
        return None

    # cell_mask, or data=None: nowhere to write them, so hand them back.
    return labels
