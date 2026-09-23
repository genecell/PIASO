"""Resolving a normalization from stored parameters.

INFOG and TF-IDF on a cytome record their parameters in ``ds.metadata`` and,
by default, write no matrix: the normalization is a pure per-chunk function of
the raw counts and those parameters, so the layer never needs to exist on disk.

These helpers return the parameters, computing and caching them if they are
absent. They lived in ``piaso/plotting/_plotEmbedding.py``, which meant that a
plotting module owned normalization resolution and that COSG, a separate
package, imported a private symbol from it. Anything importing them from the
old location still works.

Note that the transforms themselves stay where the methods are
(``_normalize_chunk_infog`` in ``_normalization``, ``_normalize_chunk_tfidf``
in ``_runTFIDF``); only the parameter lookup moved.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["ensure_infog_params", "ensure_tfidf_params"]

# These predicates guard a cached payload against being reused for the wrong
# shape or the wrong modality. They came across with the functions that use
# them: leaving them in the plotting module would have made the two modules
# import each other.
def _cached_stat_is_fresh(params, ds):
    """A cached normalization-params dict is stale once its cell-indexed
    ``cell_depth`` no longer matches ``ds.n_cells`` (e.g. after a filter_cells
    on a cytome predating the subset-invalidation fix). Returns False so the
    caller recomputes instead of broadcasting a wrong-length vector."""
    if not isinstance(params, dict) or "cell_depth" not in params:
        return True  # nothing cell-indexed to check
    try:
        return int(np.asarray(params["cell_depth"]).shape[0]) == int(ds.n_cells)
    except Exception:
        return True

def _params_feature_len_matches_modality(params, ds, modality, feature_key):
    """True if a cached normalization payload's per-feature vector
    (``params[feature_key]``) length matches ``modality``'s feature count.

    Guards against returning a payload computed for a DIFFERENT modality — e.g.
    the modality-blind legacy 'tfidf_params' (ATAC idf) for a 'tiles' request, or
    a legacy 'infog_params' (RNA inv_gene_depth) for an ATAC request. Permissive
    (returns True) when the feature count can't be determined, so behaviour is
    unchanged where the meta is unavailable."""
    if not isinstance(params, dict) or feature_key not in params:
        return True
    try:
        mm = ds.matrix_meta(f"{modality}_counts")
        n_feat = mm.get("n_cols") if isinstance(mm, dict) else None
        if n_feat is None:
            return True
        return int(np.asarray(params[feature_key]).shape[0]) == int(n_feat)
    except Exception:
        return True

def _tfidf_idf_matches_modality(params, ds, modality):
    """TF-IDF specialization of :func:`_params_feature_len_matches_modality`
    (per-feature vector = ``idf``)."""
    return _params_feature_len_matches_modality(params, ds, modality, "idf")


def _ensure_infog_params(ds, modality, use_cached_stats=True, batch_size=2048):
    """Return cached ``{modality}_infog_params`` (or compute now and cache).
    Falls through to legacy ``infog_params`` once with a DeprecationWarning
    when only the unprefixed key is present. A cached payload whose
    ``cell_depth`` length no longer matches ``ds.n_cells`` is treated as a miss
    (self-heals a cytome filtered before the cached-stats invalidation fix)."""
    if use_cached_stats:
        new_key = f"{modality}_infog_params"
        v = ds.metadata.get(new_key)
        if (v is not None and _cached_stat_is_fresh(v, ds)
                and _params_feature_len_matches_modality(v, ds, modality, "inv_gene_depth")):
            return v
        legacy = ds.metadata.get("infog_params")
        # Feature-count guard: the un-prefixed legacy 'infog_params' is modality-
        # blind (historically RNA); do NOT return it for a request on a modality
        # with a different feature count (e.g. ATAC peaks) — route to recompute.
        if (legacy is not None and _cached_stat_is_fresh(legacy, ds)
                and _params_feature_len_matches_modality(legacy, ds, modality, "inv_gene_depth")):
            import warnings as _warnings
            _warnings.warn(
                f"Using legacy 'infog_params' as '{new_key}'. Recompute "
                f"with piaso.tl.infog(ds, modality='{modality}') to refresh.",
                DeprecationWarning, stacklevel=3,
            )
            return legacy
    # Cache miss / forced refresh — compute via piaso.tl.infog (streaming, lazy)
    from ..tools._normalization import infog as _infog
    _infog(ds, save_layer=False, streaming=True, batch_size=batch_size, verbosity=0)
    return ds.metadata.get(f"{modality}_infog_params") or ds.metadata.get("infog_params")


def _ensure_tfidf_params(ds, modality, use_cached_stats=True, batch_size=2048):
    """Return cached ``{modality}_tfidf_params`` (or compute now and cache).

    Delegates to ``_runTFIDF._load_or_compute_tfidf_stats`` so the runSVD
    ``auto_tfidf=True`` path, COSG ``layer='tfidf'``, and plotting
    ``cytome_layer='tfidf'`` all share the same cache-or-compute helper.
    """
    # Legacy un-prefixed key compatibility (preserved for older cytomes). A
    # cached payload is stale/wrong when its cell_depth length no longer matches
    # n_cells (filtered cytome) OR — critically for the modality-blind legacy
    # 'tfidf_params' — its idf length does not match THIS modality's feature
    # count. The un-prefixed key does not record which modality it was computed
    # for, so a payload built for ATAC peaks (idf ~n_peaks) must NOT be returned
    # for a 'tiles' request (idf ~n_tiles); the feature-count guard routes it to
    # recompute instead of broadcasting a wrong-length idf.
    from ..tools._runTFIDF import _hydrate_tfidf_params

    def _hydrated(stored):
        """The cached payload with its arrays in place, or None if it cannot be
        read back. Newer cytomes store `cell_depth` and `idf` as table columns
        and keep only their names in metadata; the guards below are written
        against the arrays, so hydrate before checking rather than let a
        column-name payload pass every check vacuously."""
        try:
            out = _hydrate_tfidf_params(ds, stored, modality)
        except Exception:
            return None
        return out if ("cell_depth" in out and "idf" in out) else None

    if use_cached_stats:
        new_key = f"{modality}_tfidf_params"
        cached = _hydrated(ds.metadata[new_key]) if new_key in ds.metadata else None
        if (cached is not None and _cached_stat_is_fresh(cached, ds)
                and _tfidf_idf_matches_modality(cached, ds, modality)):
            return cached
        legacy = ds.metadata.get("tfidf_params")
        legacy = _hydrated(legacy) if legacy is not None else None
        if (legacy is not None and _cached_stat_is_fresh(legacy, ds)
                and _tfidf_idf_matches_modality(legacy, ds, modality)):
            import warnings as _warnings
            _warnings.warn(
                f"Using legacy 'tfidf_params' as '{new_key}'. "
                f"Recompute with piaso.tl.compute_tfidf_stats(ds, modality='{modality}') "
                f"to refresh.", DeprecationWarning, stacklevel=3,
            )
            return legacy
    from ..tools._runTFIDF import _load_or_compute_tfidf_stats
    return _load_or_compute_tfidf_stats(
        ds, modality=modality, batch_size=batch_size,
        force_recompute=not use_cached_stats,
    )



#: Public names. The underscore-prefixed originals stay as aliases because
#: COSG imports them and PIASO should not break a downstream package to tidy
#: its own namespace.
ensure_infog_params = _ensure_infog_params
ensure_tfidf_params = _ensure_tfidf_params
