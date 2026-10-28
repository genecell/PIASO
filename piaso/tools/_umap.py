"""UMAP embedding computation from precomputed kNN graph.

Supports both AnnData and cytome.Dataset as input.
"""

import warnings

import numpy as np

from ._neighbors import (
    _is_cytome_dataset,
    _load_embedding_from_cytome,
    reconstruct_knn_from_cytome,
)


def _umap_version() -> str:
    try:
        from importlib.metadata import version
        return version("umap-learn")
    except Exception:                                   # pragma: no cover
        return "unknown"


def _sklearn_version() -> str:
    try:
        from importlib.metadata import version
        return version("scikit-learn")
    except Exception:                                   # pragma: no cover
        return "unknown"


def umap(
    data,
    use_rep=None,
    min_dist=0.5,
    spread=1.0,
    n_components=2,
    random_state=42,
    key_added='X_umap',
    knn_result=None,
    neighbors_key='neighbors',
    modality='RNA',
    backend='auto',
    n_threads=0,
):
    """Compute UMAP embedding from precomputed kNN graph.

    Requires ``piaso.tl.neighbors()`` to have been run first. Pass the
    dict returned by neighbors() as ``knn_result`` for cytome mode.

    Parameters
    ----------
    data : AnnData or cytome.Dataset
        If AnnData: reads kNN from uns, stores UMAP in obsm.
        If cytome.Dataset: reads embedding, stores UMAP embedding in cytome.
    use_rep : str
        Name of the representation whose ARRAY goes with the graph. What it
        controls depends on whether a stored kNN graph is being reused:

        - cytome, or AnnData with a stored piaso kNN: neighbors are **not**
          re-run — the embedding's topology comes wholly from the stored
          graph, and ``use_rep`` only selects the data array (umap-learn's
          API requires one even with ``precomputed_knn``; both backends use
          it only to place the pieces of a graph in several connected
          components). Naming a different representation than the graph was
          built on raises, because it would change nothing while implying it
          had.
        - AnnData with **no** stored kNN: umap-learn computes its own
          neighbors from this array (its defaults, not ours) — announced
          with a warning. Run :func:`piaso.tl.neighbors` first to control
          and reuse the graph.
    min_dist : float
        Minimum distance parameter for UMAP.
    spread : float
        Spread parameter for UMAP.
    n_components : int
        Number of UMAP dimensions.
    random_state : int
        Random seed for reproducibility. With the Rust backend it sets the
        layout's negative samples and the start's noise of 1e-4; the spectral
        start itself does not depend on it, as umap-learn's does not. The
        Rust layout is deterministic, so ``None`` there means seed 0 (with
        umap-learn ``None`` means a fresh random state each call).
    key_added : str
        Key/name for the UMAP coordinates.
    knn_result : dict, optional
        Result dict from neighbors() with 'knn_indices' and 'knn_dists'. Only
        needed for the in-memory ndarray / cell_mask path. For a cytome.Dataset
        it is **not** required — the kNN is reconstructed from the persisted
        ``distances`` graph (self-contained). For AnnData it falls back to uns.
    neighbors_key : str
        Which stored neighbors graph to reuse — matches the ``key_added``
        passed to :func:`piaso.tl.neighbors`. Default ``'neighbors'`` (the
        un-prefixed graph). Honoured on **both** the AnnData and cytome paths;
        naming a graph that was never computed raises rather than quietly
        falling back to the default one.
    modality : str, default 'RNA'
        Cytome only: the modality the graph was stored under, and the one the
        embedding is written under.
    backend : {'auto', 'rust', 'umap-learn'}, default 'auto'
        Who computes the embedding from the stored kNN.

        - ``'rust'``: umap-learn's graph, curve, spectral start and schedule,
          computed by PIASO's extension, in parallel and deterministically:
          one graph and one ``random_state`` give one embedding at any
          ``n_threads``. Each epoch runs as a schedule of node blocks in
          which no two threads touch one node at a time (see
          ``piaso.tools._umap_rust``). Its coordinates differ from
          umap-learn's as two umap-learn seeds do. It needs a stored kNN
          (run :func:`piaso.tl.neighbors` first) and does not import
          umap-learn.
        - ``'umap-learn'``: umap-learn's ``UMAP``, single-threaded whenever
          ``random_state`` is set (its documentation: a seed turns off its
          multi-threading). The embeddings of PIASO 1.2.6 and earlier. It
          takes the stored kNN as umap-learn takes a precomputed one, at its
          own ``n_neighbors`` (15): a kNN of more columns is cut to its first
          15, and one of fewer is set aside for neighbors umap-learn computes
          from the array, with umap-learn's warning. The Rust backend uses
          every column stored.
        - ``'auto'``: ``'rust'`` when a kNN is stored or passed and the
          extension has it, else ``'umap-learn'`` (which then computes its
          own neighbors, with a warning).

        *Changed in 1.2.7*: the default was ``'umap-learn'``. Pass
        ``backend='umap-learn'`` to reproduce an embedding of an earlier
        version.
    n_threads : int, default 0
        Threads for the Rust backend (0: RAYON_NUM_THREADS, else the
        process's cores; one thread below 4,096 cells). The embedding does
        not depend on it. Ignored by umap-learn.

    Returns
    -------
    None, or np.ndarray for an in-memory call
        Nothing, on both backends: the embedding is written to
        ``adata.obsm[key_added]`` or to the cytome under ``key_added``, and
        read back from there. Only the in-memory form — ``data=None`` with a
        ``knn_result`` — returns the coordinates, because there is nowhere to
        write them.

        *Changed in 1.2.4*: the AnnData path used to return the coordinates
        as well as writing them, so the same call had two different contracts
        depending on the backend and echoed a 2-column array in a notebook.
        Read them from ``adata.obsm[key_added]``.
    """
    # A store that cannot be written fails here, not after the work.
    from ..utils._cytome_compat import require_writable as _require_writable
    _require_writable(data, "piaso.tl.umap")
    # A cytome path is as valid an input as it is for infog / runSVD / runGDR.
    # Without this a pipeline written entirely with a path worked for three
    # calls and then raised "'str' object has no attribute 'obsm'" on the
    # fourth, which reads like the caller passed the wrong type.
    if isinstance(data, str):
        from ._normalization import _open_cytome
        data = _open_cytome(data)

    if backend not in ('auto', 'rust', 'umap-learn'):
        raise ValueError(f"umap(backend={backend!r}): use 'auto', 'rust' or 'umap-learn'")
    if not isinstance(n_threads, (int, np.integer)) or n_threads < 0:
        raise ValueError("umap(n_threads=...): a non-negative integer (0: all the process may use)")

    # Get kNN data
    precomputed_knn = None
    if knn_result is not None:
        knn_indices = knn_result['knn_indices']
        knn_dists = knn_result['knn_dists']
        precomputed_knn = (knn_indices, knn_dists, None)
    elif _is_cytome_dataset(data):
        # Self-contained cytome path: reconstruct kNN from the distances graph
        # that piaso.tl.neighbors() persisted — no value passing needed. The
        # embedding is driven entirely by this reused graph; `use_rep` only
        # selects the array fed to UMAP (n_samples + stored raw data), so default
        # it to the rep the graph was built on, and warn on an explicit mismatch.
        from ._neighbors import _neighbors_graph_keys
        _nn_key = _neighbors_graph_keys(neighbors_key)[2]
        stored_use_rep = None
        try:
            stored_use_rep = data.metadata.get(_nn_key.replace('n_neighbors', 'use_rep'))
        except Exception:
            stored_use_rep = None
        if use_rep is None:
            use_rep = stored_use_rep or 'X_svd'
        elif stored_use_rep and use_rep != stored_use_rep:
            # An error, not a warning. With a precomputed graph, use_rep
            # selects DATA, not topology — the embedding's structure comes
            # wholly from the stored graph. A use_rep that names a different
            # representation therefore does nothing topological, and a
            # parameter that silently does nothing is worse than one that
            # stops you: the figure would *imply* it was built on use_rep
            # when it was built on whatever neighbors() used.
            raise ValueError(
                f"umap(use_rep='{use_rep}'): the '{neighbors_key}' neighbors "
                f"graph was built on '{stored_use_rep}', and the UMAP is "
                f"driven by that graph — use_rep only selects the array fed "
                f"to umap-learn, not the topology. Run "
                f"piaso.tl.neighbors(ds, use_rep='{use_rep}') first if you "
                f"want the '{use_rep}' topology, or drop use_rep to use the "
                f"graph as stored.")
        knn_indices, knn_dists = reconstruct_knn_from_cytome(data, neighbors_key, modality)
        precomputed_knn = (knn_indices, knn_dists, None)
    elif data is None:
        # No object and no kNN: there is nothing to embed. Checked here rather
        # than below because the AnnData branch reads `data.uns` immediately.
        raise ValueError(
            "umap(data=None) needs knn_result= from piaso.tl.neighbors: with "
            "no object there is no representation and no stored graph.")
    elif not _is_cytome_dataset(data):
        # `neighbors_key` used to be ignored here: the keys were hard-coded to
        # '_neighbors_*', so neighbors(key_added='x') followed by
        # umap(neighbors_key='x') silently reused whichever graph happened to
        # be un-prefixed. With two graphs on one object -- an SVD one and a GDR
        # one -- that produced a second UMAP bit-identical to the first, which
        # looks like "the method did nothing" rather than like a wiring bug.
        uns_key = neighbors_key if neighbors_key else 'neighbors'
        knn_idx_key = f'_{uns_key}_knn_indices'
        knn_dist_key = f'_{uns_key}_knn_dists'
        if knn_idx_key not in data.uns and uns_key != 'neighbors':
            raise KeyError(
                f"umap: no kNN stored for neighbors_key='{neighbors_key}' "
                f"(looked for adata.uns['{knn_idx_key}']). Available: "
                f"{[k for k in data.uns if k.endswith('_knn_indices')]}. "
                f"Run piaso.tl.neighbors(..., key_added='{neighbors_key}') first.")
        if knn_idx_key in data.uns and knn_dist_key in data.uns:
            knn_indices = data.uns[knn_idx_key]
            knn_dists = data.uns[knn_dist_key]
            precomputed_knn = (knn_indices, knn_dists, None)
        else:
            # The one path where use_rep genuinely decides the topology: with
            # no stored kNN, umap-learn computes its own neighbors from the
            # use_rep array, with its defaults rather than ours. Say so —
            # silently running a second, differently-parameterised kNN is how
            # "why does tl.umap have use_rep, is it re-running neighbors?"
            # becomes a reasonable question with no discoverable answer.
            # (backend='rust' refuses below instead: it computes no kNN.)
            if backend != 'rust':
                warnings.warn(
                    f"umap: no stored kNN found for "
                    f"neighbors_key='{uns_key}' — umap-learn will compute its own "
                    f"neighbors from use_rep='{use_rep or 'X_svd'}' with its "
                    f"internal defaults. Run piaso.tl.neighbors first to control "
                    f"and reuse the graph.", stacklevel=2)
        # Fall back to the representation the graph was built on, but never
        # override one the caller passed explicitly -- doing so silently
        # embedded X_svd when the call said use_rep='X_gdr'.
        if use_rep is None and uns_key in data.uns and 'params' in data.uns[uns_key]:
            use_rep = data.uns[uns_key]['params'].get('use_rep', use_rep)

    # Final fallback so a never-resolved use_rep doesn't index obsm[None].
    if use_rep is None:
        use_rep = 'X_svd'

    # Load embedding
    if data is None:
        # The in-memory path: a knn_result and nothing to read a representation
        # from. umap-learn uses X only for its row count when the graph is
        # precomputed, so the kNN's own shape is the whole requirement -- and
        # asking `None` for `.obsm` is how this path used to end. It was
        # invisible here because the test covering it skips wherever
        # umap-learn and scikit-learn disagree, which is most of our machines.
        X = np.zeros((len(precomputed_knn[0]), 1), dtype=np.float32)
    elif _is_cytome_dataset(data):
        X = _load_embedding_from_cytome(data, use_rep, modality)
    else:
        X = data.obsm[use_rep]

    from . import _umap_rust
    if backend == 'auto':
        backend = 'rust' if (precomputed_knn is not None and _umap_rust.available()) else 'umap-learn'
        if precomputed_knn is not None and backend == 'umap-learn':
            warnings.warn(
                "umap: the Rust extension has no umap_layout_csr, so umap-learn computes the "
                "embedding instead (single-threaded with a seed, and another picture). Rebuild or "
                "reinstall PIASO's extension, or pass backend='umap-learn' to silence this.",
                RuntimeWarning, stacklevel=2)
    if backend == 'rust':
        if precomputed_knn is None:
            raise ValueError("umap(backend='rust') needs a stored kNN graph: run piaso.tl.neighbors first, "
                             "or pass knn_result")
        if random_state is not None and (not isinstance(random_state, (int, np.integer)) or random_state < 0):
            raise ValueError("umap(backend='rust'): random_state must be a non-negative integer or None")
        embedding = _umap_rust.embed(
            X, np.asarray(precomputed_knn[0]), np.asarray(precomputed_knn[1]), n_components=n_components,
            min_dist=min_dist, spread=spread, random_state=random_state, n_threads=int(n_threads))
    else:
        embedding = _umap_learn_embedding(X, precomputed_knn, n_components, min_dist, spread, random_state)
    return _store_umap(data, embedding, key_added, use_rep, modality, backend)


def _umap_learn_embedding(X, precomputed_knn, n_components, min_dist, spread, random_state):
    import umap as umap_module
    # umap-learn's default for precomputed_knn is the sentinel (None, None,
    # None), not None — passing None crashes inside fit at
    # `self.precomputed_knn[0]`. The no-stored-kNN fallback always hit this,
    # which means that path never actually worked; it now runs (with the
    # warning above announcing whose neighbors are being used).
    _umap_kwargs = dict(
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
    )
    if precomputed_knn is not None:
        _umap_kwargs["precomputed_knn"] = precomputed_knn
    reducer = umap_module.UMAP(**_umap_kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*random_state")
        warnings.filterwarnings("ignore", message="precomputed_knn.*transform")
        try:
            embedding = reducer.fit_transform(X)
        except TypeError as exc:
            # umap-learn < 0.5.8 calls sklearn's check_array with
            # `force_all_finite`, renamed in scikit-learn 1.6 and REMOVED in
            # 1.8. The combination raises from three frames inside sklearn
            # with no mention of umap, which reads like a PIASO bug. It is an
            # upstream one, fixed upstream, so say which package to move.
            if "force_all_finite" not in str(exc):
                raise
            raise ImportError(
                "piaso.tl.umap needs umap-learn >= 0.5.8. The installed "
                f"umap-learn ({_umap_version()}) calls scikit-learn's "
                "check_array(force_all_finite=...), which scikit-learn "
                f"{_sklearn_version()} removed.\n"
                "    pip install -U 'umap-learn>=0.5.8'\n"
                "Nothing else needs to change; this is an upstream "
                "incompatibility between those two packages."
            ) from exc
    return embedding


def _store_umap(data, embedding, key_added, use_rep, modality, backend):
    # Store results
    if _is_cytome_dataset(data):
        # Writes the caller's key verbatim, so one cytome can hold RNA_svd
        # (runSVD, via _embedding_names.storage_name) next to X_umap. That
        # inconsistency is real: `ds.embeddings["X_svd"]` raises after
        # runSVD(key_added="X_svd") because the stored name is RNA_svd.
        #
        # Standardising on {modality}_{name} is the consistent choice, but the
        # current names are a tested contract (test_round24, projectGDR's
        # reference lookup), so it is a migration rather than a patch, and it
        # is not attempted here.
        #
        # Reads are unaffected either way: _embedding_names.candidate_names
        # accepts every spelling, which is why this has stayed invisible to
        # anyone using use_rep= rather than indexing ds.embeddings directly.
        from ..settings import _resolve_layer_dtype
        data.add_embedding(
            key_added, embedding,
            dtype=_resolve_layer_dtype(None),
            provenance={"modality": modality, "function": "piaso.tl.umap",
                        "use_rep": use_rep, "key_added": key_added, "backend": backend},
        )
        data.flush()
        # Strict / self-contained: embedding lives on the cytome.
        return None
    if data is None:
        # In-memory: a knn_result and no object to write to, so the array is
        # the only possible result.
        return embedding

    data.obsm[key_added] = embedding
    return None
