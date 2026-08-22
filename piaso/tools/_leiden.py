"""Leiden clustering using igraph directly.

Supports both AnnData and cytome.Dataset as input.
"""

import random as _random
import threading
import numpy as np
import pandas as pd

from ._neighbors import _is_cytome_dataset


# Guards igraph's process-global RNG; see the comment at its use site.
_IGRAPH_RNG_LOCK = threading.Lock()


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
):
    """Leiden clustering using igraph (no scanpy/leidenalg dependency).

    Reads the connectivities matrix and runs the Leiden algorithm
    via igraph's ``community_leiden()``.

    Parameters
    ----------
    data : AnnData or cytome.Dataset
        If AnnData: reads from obsp, stores in obs.
        If cytome.Dataset: reads connectivities from cytome graphs, stores in cells.
    resolution : float
        Resolution parameter controlling cluster granularity.
    n_iterations : int
        Number of Leiden iterations.
    random_state : int
        Random seed for reproducibility. Sets igraph's internal RNG to ensure
        deterministic results across repeated calls.
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

    Returns
    -------
    np.ndarray or None
        For AnnData and the in-memory ``cell_mask`` / ``data=None`` paths,
        returns the string array of cluster labels. For a **cytome.Dataset**
        (no ``cell_mask``) returns ``None`` — labels are written to
        ``ds.cells[key_added]`` and read back from there. Self-contained: pass
        the Dataset and access ``ds.cells[key_added]`` afterwards.
    """
    # A cytome path is as valid an input as it is for infog / runSVD / runGDR.
    # Without this a pipeline written entirely with a path worked for three
    # calls and then raised "'str' object has no attribute 'obsm'" on the
    # fourth, which reads like the caller passed the wrong type.
    if isinstance(data, str):
        from ._normalization import _open_cytome
        data = _open_cytome(data)

    import igraph as ig
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
        cell_mask_arr = np.asarray(cell_mask).astype(bool)
        if cell_mask_arr.shape[0] != adj.shape[0]:
            raise ValueError(
                f"cell_mask length ({cell_mask_arr.shape[0]}) must match "
                f"adjacency matrix ({adj.shape[0]} rows)."
            )
        adj = adj[cell_mask_arr][:, cell_mask_arr]

    # Build undirected igraph graph from sparse adjacency.
    # Our connectivities (from fuzzy_simplicial_set) are symmetric,
    # so we take the upper triangle to build an undirected graph
    # without duplicate edges.
    sources, targets = adj.nonzero()
    mask = sources <= targets
    sources, targets = sources[mask], targets[mask]
    weights = np.array(adj[sources, targets]).flatten()
    g = ig.Graph(n=adj.shape[0],
                 edges=list(zip(sources.tolist(), targets.tolist())),
                 directed=False)
    g.es['weight'] = weights.tolist()

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

    # Numeric-aware category order so the result is "0","1","2",...,"10" rather
    # than the lexical "0","1","10","11",... that plotters fall back to.
    _levels = list(dict.fromkeys(labels.tolist()))
    try:
        _order = [str(i) for i in sorted(int(x) for x in _levels)]
    except (ValueError, TypeError):
        _order = sorted(_levels)

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
            try:
                data.set_categories(key_added, order=_order)
            except Exception:
                pass
            # Strict / self-contained: labels live on the cytome.
            return None
        else:
            data.obs[key_added] = pd.Categorical(labels, categories=_order)

    return labels
