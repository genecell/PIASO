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
        Embedding name for the representation to use.
    min_dist : float
        Minimum distance parameter for UMAP.
    spread : float
        Spread parameter for UMAP.
    n_components : int
        Number of UMAP dimensions.
    random_state : int
        Random seed for reproducibility.
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

    Returns
    -------
    np.ndarray or None
        For AnnData and in-memory inputs, returns the UMAP coordinates. For a
        **cytome.Dataset**, returns ``None`` — the embedding is written to the
        cytome under ``key_added`` and read back from there.
    """
    # A cytome path is as valid an input as it is for infog / runSVD / runGDR.
    # Without this a pipeline written entirely with a path worked for three
    # calls and then raised "'str' object has no attribute 'obsm'" on the
    # fourth, which reads like the caller passed the wrong type.
    if isinstance(data, str):
        from ._normalization import _open_cytome
        data = _open_cytome(data)

    import umap as umap_module

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
            warnings.warn(
                f"umap(use_rep='{use_rep}') differs from the representation the "
                f"'{neighbors_key}' neighbors graph was built on ('{stored_use_rep}'). "
                f"The cytome UMAP is driven by the reused graph; use_rep only sets the "
                f"array fed to UMAP. Pass the matching neighbors_key, or rerun "
                f"piaso.tl.neighbors on '{use_rep}'.", stacklevel=2)
        knn_indices, knn_dists = reconstruct_knn_from_cytome(data, neighbors_key, modality)
        precomputed_knn = (knn_indices, knn_dists, None)
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
        # Fall back to the representation the graph was built on, but never
        # override one the caller passed explicitly -- doing so silently
        # embedded X_svd when the call said use_rep='X_gdr'.
        if use_rep is None and uns_key in data.uns and 'params' in data.uns[uns_key]:
            use_rep = data.uns[uns_key]['params'].get('use_rep', use_rep)

    # Final fallback so a never-resolved use_rep doesn't index obsm[None].
    if use_rep is None:
        use_rep = 'X_svd'

    # Load embedding
    if _is_cytome_dataset(data):
        X = _load_embedding_from_cytome(data, use_rep, modality)
    else:
        X = data.obsm[use_rep]

    reducer = umap_module.UMAP(
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        precomputed_knn=precomputed_knn,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*random_state")
        warnings.filterwarnings("ignore", message="precomputed_knn.*transform")
        embedding = reducer.fit_transform(X)

    # Store results
    if _is_cytome_dataset(data):
        # Store as {modality}_{name}, like runSVD. Previously this wrote the
        # caller's key verbatim, so one cytome could hold RNA_svd next to
        # X_umap and a second modality's UMAP had nowhere to go.
        # NOTE: writes the caller's key verbatim. Standardising on
        # {modality}_{name} like runSVD is the consistent choice, but the
        # current names are a tested contract (test_round24, projectGDR's
        # reference lookup), so it is a migration rather than a patch.
        # Reads already accept every spelling via _embedding_names.
        from ..settings import _resolve_layer_dtype
        data.add_embedding(
            key_added, embedding,
            dtype=_resolve_layer_dtype(None),
            provenance={"modality": modality, "function": "piaso.tl.umap",
                        "use_rep": use_rep, "key_added": key_added},
        )
        data.flush()
        # Strict / self-contained: embedding lives on the cytome.
        return None
    else:
        data.obsm[key_added] = embedding

    return embedding
