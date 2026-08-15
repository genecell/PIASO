"""Project unseen cells into an existing (frozen) GDR space.

Public surface is a single function, :func:`projectGDR`. The reference state lives **inside the
reference object** — ``adata.uns['gdr_reference']`` or ``ds.metadata['gdr_reference']`` — written by
``runGDR(..., save_reference=True)``. There is no separate fit call and no external file format: if
you want to share a reference, share the object.

Design decisions (see ``docs/discussion/2026-07-28_gdr_projection_of_unseen_cells.md``):

* **Reference-frozen only.** Reference coordinates never move. Re-fitting is ``concat`` + ``runGDR``.
* ``runGDR`` stores the cheap part of the state (marker sets, block indices, layers, seed). The
  expensive part — the raw score-matrix column norms and the frozen control-gene neighbourhoods — is
  completed on first use by :func:`projectGDR` and **cached back into the reference object**, so the
  cost is paid once and never inside ``runGDR``.
* Two column-scaling modes, both frozen: ``mode='reference'`` uses the stored reference norms,
  ``mode='self'`` uses the query's own (i.e. treats the query as a new batch, which is what ``runGDR``
  does for each reference batch when ``batch_key`` is set).

Why the reference cohort *size* does not matter: the row normalisation that follows divides out any
scaling constant common to all columns, and ``||s_j||_2 = sqrt(N) * RMS_j`` with the same ``sqrt(N)``
for every column, so reference L2 norms and per-column RMS give identical embeddings.

INFOG constants use the same schema as the cytome's existing ``{modality}_infog_params``
(``inv_gene_depth``, ``scale``, ``counts_sum``, ``threshold``), so a cytome that already carries them
needs nothing new.
"""
from typing import Optional, Literal
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

__all__ = ["projectGDR"]

GDR_REFERENCE_KEY = "gdr_reference"
GDR_REFERENCE_SCHEMA_VERSION = 3


# --------------------------------------------------------------------------------------------
# INFOG with frozen reference statistics — same schema as cytome `{modality}_infog_params`
# --------------------------------------------------------------------------------------------
def _infog_reference_stats(counts, var_names=None) -> dict:
    """Cohort-level INFOG constants, in the cytome ``*_infog_params`` schema.

    ``infog`` computes ``out[i,j] = counts[i,j] * sqrt(scale*counts_sum*inv_gene_depth[j]) / cell_depth[i]``
    and trims at ``threshold``. Only ``cell_depth`` is per-cell, so everything else is reusable.
    Deliberately does **not** store ``cell_depth`` — that is per-cohort and must never be reused.
    """
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    cell_depth = np.asarray(counts.sum(axis=1)).ravel()
    gene_depth = np.asarray(counts.sum(axis=0)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_gene_depth = 1.0 / gene_depth
    inv_gene_depth[~np.isfinite(inv_gene_depth)] = 0.0
    return {
        "inv_gene_depth": inv_gene_depth,
        "scale": float(np.median(cell_depth)),
        "counts_sum": float(counts.sum()),
        "threshold": float(np.sqrt(counts.shape[0])),
        "n_ref_cells": int(counts.shape[0]),
        "var_names": None if var_names is None else np.asarray(var_names).astype(str),
    }


def _infog_per_gene(stats: dict) -> np.ndarray:
    """The single per-gene factor the three cohort terms collapse into.

    Computed in explicit float64. ``inv_gene_depth`` is stored float32, and under NEP 50 the
    accumulation dtype otherwise depends on how the two scalars are *typed*: Python floats
    are weak (result stays float32) but ``np.float64`` scalars are strong (result promotes to
    float64). Writing the state to h5ad converts Python floats to ``np.float64``, so without
    this cast an in-memory and a round-tripped reference computed the factor at different
    precision -- the source of the ~1e-05 INFOG / ~1e-08 embedding discrepancy.
    """
    return np.sqrt(np.float64(stats["scale"]) * np.float64(stats["counts_sum"])
                   * np.asarray(stats["inv_gene_depth"], dtype=np.float64))


def _infog_apply(counts, stats: dict, var_names=None, trim: bool = True):
    """Apply frozen reference INFOG constants to new counts, on the reference gene space."""
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    counts = counts.tocsr()

    ref_names = stats.get("var_names")
    gene_recovery = 1.0
    if ref_names is not None and var_names is not None:
        ref_names = np.asarray(ref_names).astype(str)
        var_names = np.asarray(var_names).astype(str)
        pos = pd.Series(np.arange(len(var_names)), index=var_names)
        pos = pos[~pos.index.duplicated()]
        take = pos.reindex(ref_names).values
        found = ~pd.isna(take)
        gene_recovery = float(found.mean())
        # build directly in CSC to avoid an expensive sparsity-structure change
        cols = []
        idx_map = {int(j): int(take[j]) for j in np.where(found)[0]}
        src = counts.tocsc()
        n = counts.shape[0]
        empty = sparse.csc_matrix((n, 1), dtype=np.float64)
        for j in range(len(ref_names)):
            cols.append(src[:, idx_map[j]] if j in idx_map else empty)
        counts = sparse.hstack(cols, format="csr")

    per_gene = _infog_per_gene(stats)
    if counts.shape[1] != len(per_gene):
        raise ValueError(
            f"gene-space mismatch: query has {counts.shape[1]} genes, reference constants have "
            f"{len(per_gene)}. Pass matching `var_names` so genes can be aligned by name.")
    cell_depth = np.asarray(counts.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cell = 1.0 / cell_depth
    per_cell[~np.isfinite(per_cell)] = 0.0

    out = counts.copy().tocsr()
    out.data = out.data.astype(np.float64)
    out.data *= np.repeat(per_cell, np.diff(out.indptr)) * per_gene[out.indices]
    if trim:
        out.data[out.data > stats["threshold"]] = stats["threshold"]
    return out, gene_recovery


# --------------------------------------------------------------------------------------------
# shared normalisation / scoring, so reference and query go through identical code
# --------------------------------------------------------------------------------------------
def _gdr_normalize(score_list, col_norm=None, block_indices=None, eps: float = 1e-12):
    """Column L2 then row L2 — whole-row when ``block_indices`` is None, per-block otherwise."""
    S = np.asarray(score_list, dtype=np.float64).copy()
    if col_norm is None:
        col_norm = np.linalg.norm(S, axis=0)
    col_norm = np.asarray(col_norm, dtype=np.float64).copy()
    col_norm[col_norm < eps] = 1.0
    S /= col_norm
    if block_indices is None:
        S = normalize(S, norm="l2", axis=1)
    else:
        for start, end in zip(block_indices[:-1], block_indices[1:]):
            S[:, start:end] = normalize(S[:, start:end], norm="l2", axis=1)
    return S, col_norm


def _raw_scores(data, marker_gene, layer, max_workers, random_seed, verbosity=0,
                precomputed_knn=None, modality="RNA", batch_size=1024):
    """Raw marker-set scores, via the same helper runGDR uses. Works for AnnData and cytome.

    ``precomputed_knn`` freezes the expression-matched control-gene neighbourhoods; without it
    ``score`` rebuilds them from whatever cells are passed in and the projection drifts with query
    size (the same 200 cells scored in a 2000- vs 200-cell cohort differ by max 0.288, r=0.979).
    """
    from ._runGDR import calculateScoreParallel
    kw = dict(gene_set=marker_gene, score_method="piaso", max_workers=max_workers,
              random_seed=random_seed, verbosity=verbosity, precomputed_knn=precomputed_knn)
    if _is_cytome(data):
        kw.update(modality=modality, cytome_layer=layer, batch_size=batch_size)
    else:
        kw.update(score_layer=layer)
    score_list, names = calculateScoreParallel(data, **kw)
    return np.asarray(score_list, dtype=np.float64), names


def _infog_apply_cytome(ds, stats, *, modality="RNA", counts_layer="counts",
                        key_added="infog_frozen", batch_size=2048, verbosity=1):
    """Write a frozen-INFOG layer into a query cytome using the REFERENCE's constants.

    Streams ``counts`` chunk by chunk, applies the reference per-gene factor with each chunk's own
    cell depths, and stores the result as a new layer. This is what closes the last asymmetry with
    the AnnData path: without it the query is scored on whatever ``infog`` layer it happens to
    carry, which was computed on the query's *own* cohort.

    Genes are aligned to the reference **by name**, like the AnnData path, so a query cytome with
    its own feature table projects fine; ``gene_recovery`` reports the fraction of reference genes
    found.

    Alignment maps the reference *constants* onto the query's gene space rather than reindexing the
    matrix into the reference's. That is deliberate: the written layer must stay the same width as
    the cytome's feature table, because the scoring pass resolves marker-gene names to column
    indices through that table — a reference-width layer on a query-width feature table would index
    the wrong columns and fail silently. Query genes absent from the reference get factor 0, so they
    contribute nothing (the marker sets only name reference genes anyway), and cell depth is summed
    over the matched genes only, which is what ``_infog_apply`` does on the AnnData side (it
    reindexes first, then sums) — so the two paths agree numerically.

    Returns ``(key_added, gene_recovery)``.
    """
    ref_names = stats.get("var_names")
    q_names = np.asarray(_var_names(ds, modality)).astype(str)
    per_gene_ref = _infog_per_gene(stats)
    gene_recovery = 1.0
    if ref_names is None or (len(ref_names) == len(q_names)
                             and np.array_equal(np.asarray(ref_names).astype(str), q_names)):
        if len(per_gene_ref) != len(q_names):
            raise ValueError(f"gene-space mismatch: query cytome has {len(q_names)} features, "
                             f"reference constants have {len(per_gene_ref)}")
        per_gene_q = per_gene_ref
        in_ref = np.ones(len(q_names), dtype=np.float64)
    else:
        ref_names = np.asarray(ref_names).astype(str)
        if pd.Index(ref_names).has_duplicates:
            raise ValueError("reference var_names contain duplicates, so query genes cannot be "
                             "aligned unambiguously; de-duplicate the reference names first.")
        pos = pd.Series(np.arange(len(q_names)), index=q_names)
        pos = pos[~pos.index.duplicated()]
        take = pos.reindex(ref_names).to_numpy()               # reference j -> query column
        found = ~pd.isna(take)
        gene_recovery = float(found.mean())
        qcol = take[found].astype(np.int64)
        per_gene_q = np.zeros(len(q_names), dtype=np.float64)
        per_gene_q[qcol] = per_gene_ref[np.where(found)[0]]
        in_ref = np.zeros(len(q_names), dtype=np.float64)
        in_ref[qcol] = 1.0
        if verbosity:
            print(f"projectGDR: aligning the query cytome to the reference gene space by name — "
                  f"{int(found.sum())}/{len(ref_names)} reference genes found", flush=True)
    thr = float(stats["threshold"])
    chunks, rows = [], 0
    for chunk, _idx in ds.iter_chunks(modality=modality, layer=counts_layer,
                                      batch_size=batch_size):
        c = chunk.tocsr() if sparse.issparse(chunk) else sparse.csr_matrix(chunk)
        # depth over the REFERENCE-matched genes only (c @ mask == masked row sums), matching
        # _infog_apply, which reindexes to the reference space before summing.
        cd = np.asarray(c @ in_ref).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            pc = 1.0 / cd
        pc[~np.isfinite(pc)] = 0.0
        out = c.copy().astype(np.float64)
        out.data *= np.repeat(pc, np.diff(out.indptr)) * per_gene_q[out.indices]
        out.data[out.data > thr] = thr
        out.eliminate_zeros()      # unmatched genes got factor 0; drop them from the structure
        chunks.append(out); rows += out.shape[0]
    M = sparse.vstack(chunks, format="csr")
    name = f"{modality}_{key_added}"
    # add_matrix only BUFFERS the write (ds._pending_writes); without the flush the layer is
    # invisible to the very next iter_chunks that the scoring pass makes.
    ds.add_matrix(name, M)
    ds.flush()
    if verbosity:
        print(f"projectGDR: wrote frozen-INFOG layer '{name}' ({M.shape}) using the reference's "
              f"constants", flush=True)
    return key_added, gene_recovery


def _reference_knn(matrix, n_nearest_neighbors=30, leaf_size=40):
    from ._normalization import _precompute_stats
    return np.asarray(_precompute_stats(matrix, n_nearest_neighbors=n_nearest_neighbors,
                                        leaf_size=leaf_size))


# --------------------------------------------------------------------------------------------
# container helpers — one code path for AnnData and cytome
# --------------------------------------------------------------------------------------------
def _is_cytome(obj) -> bool:
    return type(obj).__name__ == "CytomeDataset" or hasattr(obj, "iter_chunks")


def _open(obj):
    """Accept a cytome path, a CytomeDataset, or an AnnData."""
    if isinstance(obj, str):
        if obj.endswith(".h5ad"):
            import anndata as ad
            return ad.read_h5ad(obj)
        import cytome
        return cytome.open(obj)
    return obj


# --------------------------------------------------------------------------------------------
# cytome metadata is JSON-backed, so the reference state has to be encoded on the way in.
# `ds.metadata[key] = value` serialises a *top-level* DataFrame, but the state is a dict whose
# values include a DataFrame (marker_gene) and several numpy arrays (col_norm, knn_idx,
# novelty_reference, and the infog constants). json.dumps chokes on all of those, and the
# runGDR call site wrapped the write in try/except -- so the write degraded to a warning and
# the cytome silently had NO gdr_reference. Encode/decode explicitly instead.
# --------------------------------------------------------------------------------------------
# Measured on a real reference cytome (ADVIS >=P6 interneurons, 7,784 cells x 32,285 genes), the
# schema-v2 encoding put 1,388,173 bytes of JSON into ONE metadata row -- of which 99.6% was
# either a duplicate of something the cytome already stored or an array with a proper home:
#
#   infog              881,479 B (63%)  <- duplicate of ds.metadata['{modality}_infog_params']
#   var_names          342,380 B (25%)  <- duplicate of the modality's feature table
#   novelty_reference  158,001 B (11%)  <- a per-CELL vector; belongs in ds.cells
#   marker_gene          5,090 B (0.4%) <- near-duplicate of ds.metadata['runGDR_marker_genes']
#   everything else      < 1 KB         <- genuinely config-like; metadata is the right home
#
# So schema v3 keeps only the small config inline and POINTS at the canonical copies. The
# per-cell novelty vector moves to a cells column, where it is also queryable and plottable
# like any other cell annotation. Result: the row drops to ~1 KB with no information loss and
# no second copy that can silently drift from the first.
#
# marker_gene stays inline deliberately (it is 0.4% of the row, it IS the recipe, and keeping it
# makes the reference self-contained) -- but as a plain dict-of-lists, the same shape
# `runGDR_marker_genes` uses, so there is no bespoke encoding tag anywhere in the schema.
_NOVELTY_COL = "gdr_novelty_reference"
_CTRL_KNN_EMB = "gdr_control_knn"


def _fingerprint(names, extra=None):
    """Cheap check that a pointed-at record still describes the same thing.

    ``extra`` is canonically formatted rather than ``str()``-ed: a float that is a Python float
    on one side and a np.float32/np.float64 on the other reprs differently ("8.1" vs
    "8.100000381469727") and would spuriously fail the check. Same class of trap as the
    dtype-promotion bug in ``_infog_per_gene``.
    """
    import hashlib
    tag = "" if extra is None else f"{float(extra):.12g}"
    h = hashlib.sha1(("|".join(map(str, names)) + f"#{tag}").encode()).hexdigest()[:12]
    return f"{len(names)}:{h}"


def _infog_fingerprint(params: dict) -> str:
    """Hash the POINTED-AT record's own content, so replacing it is detected.

    Fingerprinting the feature table instead would miss the case that matters: the gene names are
    unchanged but the INFOG constants themselves were recomputed (a second `infog()` call with
    different parameters), which silently invalidates a frozen reference.
    """
    import hashlib
    igd = np.asarray(params.get("inv_gene_depth", []), dtype=np.float64)
    parts = [str(igd.size)] + [f"{float(params.get(k, 0.0)):.12g}"
                               for k in ("scale", "counts_sum", "threshold")]
    if igd.size:
        parts.append(f"{float(igd.sum()):.12g}")
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    return f"{igd.size}:{h}"


def _encode_state_cytome(ds, state, modality="RNA"):
    """Schema-v3 encode: small config inline, big arrays to their proper cytome homes."""
    out, meta = {}, ds.metadata
    for k, v in state.items():
        if k in ("marker_gene",):
            df = v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)
            out[k] = {str(c): [None if pd.isna(x) else str(x) for x in df[c]] for c in df.columns}
        elif k == "var_names":
            continue                                   # derivable from the feature table
        elif k == "novelty_reference" and v is not None:
            ds.add_cells_column(_NOVELTY_COL, np.asarray(v, dtype=np.float64))
            out["novelty_reference_ref"] = {"store": "cells", "column": _NOVELTY_COL}
        elif k == "knn_idx" and v is not None:
            ds.add_var_embedding(_CTRL_KNN_EMB, np.asarray(v))
            out["knn_idx_ref"] = {"store": "var_embedding", "name": _CTRL_KNN_EMB}
        elif k == "infog" and v is not None:
            key = f"{modality}_infog_params"
            # fingerprint BOTH sides from the same source (the feature table), never from the
            # state's own var_names copy -- otherwise a benign difference in how the two were
            # built fails the check for no reason.
            if key in list(meta.keys()):               # point at the canonical copy
                out["infog_ref"] = {
                    "store": "metadata", "key": key,
                    "fingerprint": _infog_fingerprint(meta[key]),
                }
            else:                                      # no canonical copy: keep it inline
                out["infog"] = {kk: (np.asarray(x).tolist() if isinstance(x, np.ndarray)
                                     else (x.item() if isinstance(x, np.generic) else x))
                                for kk, x in v.items()}
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()                        # small vectors (col_norm, gene_set_names)
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    out["schema_version"] = 3
    return out


def _decode_state_cytome(ds, enc, modality=None):
    """Resolve schema-v3 pointers; schema v2 (inline, tagged) still reads."""
    st = dict(enc)
    modality = modality or st.get("modality", "RNA")
    if isinstance(st.get("marker_gene"), dict):
        mg = st["marker_gene"]
        if "__piaso_df__" in mg:                       # legacy v2
            mg = mg["__piaso_df__"]
        st["marker_gene"] = pd.DataFrame({k: pd.Series(v) for k, v in mg.items()})
    for k in ("col_norm", "gene_set_names", "novelty_reference"):
        if isinstance(st.get(k), dict) and "__piaso_nd__" in st[k]:      # legacy v2
            st[k] = np.asarray(st[k]["__piaso_nd__"])
        elif isinstance(st.get(k), list):
            st[k] = np.asarray(st[k])
    if isinstance(st.get("infog"), dict) and any(
            isinstance(x, dict) and "__piaso_nd__" in x for x in st["infog"].values()):
        st["infog"] = {kk: (np.asarray(x["__piaso_nd__"])
                            if isinstance(x, dict) and "__piaso_nd__" in x else x)
                       for kk, x in st["infog"].items()}
    elif isinstance(st.get("infog"), dict):
        st["infog"] = {kk: (np.asarray(x) if isinstance(x, list) else x)
                       for kk, x in st["infog"].items()}
    # ---- v3 pointers ----
    ref = st.pop("infog_ref", None)
    if ref is not None:
        p = ds.metadata[ref["key"]]
        names = _var_names(ds, modality)
        got = _infog_fingerprint(p)
        if ref.get("fingerprint") and got != ref["fingerprint"]:
            raise ValueError(
                f"gdr_reference points at ds.metadata['{ref['key']}'] but that record no longer "
                f"matches the reference it was built from (fingerprint {ref['fingerprint']} vs "
                f"{got}). Re-run runGDR on this cytome, or restore the original INFOG params.")
        st["infog"] = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in p.items()}
        st["infog"]["var_names"] = names
    ref = st.pop("novelty_reference_ref", None)
    if ref is not None:
        st["novelty_reference"] = np.asarray(
            ds.cells.to_pandas()[ref["column"]].to_numpy(), dtype=np.float64)
    ref = st.pop("knn_idx_ref", None)
    if ref is not None:
        st["knn_idx"] = np.asarray(ds.var_embeddings[ref["name"]])
    if "var_names" not in st.get("infog", {}) and "infog" in st:
        st["infog"]["var_names"] = _var_names(ds, modality)
    return st


def _get_state(obj):
    try:
        if _is_cytome(obj):
            return _decode_state_cytome(obj, obj.metadata[GDR_REFERENCE_KEY])
        return obj.uns[GDR_REFERENCE_KEY]
    except (KeyError, TypeError):
        return None


def _set_state(obj, state):
    if _is_cytome(obj):
        obj.metadata[GDR_REFERENCE_KEY] = _encode_state_cytome(
            obj, state, modality=state.get("modality", "RNA"))
        obj.flush()          # metadata writes are buffered like matrices
    else:
        obj.uns[GDR_REFERENCE_KEY] = state


def _counts_matrix(obj, layer):
    if _is_cytome(obj):
        raise NotImplementedError("frozen-INFOG from raw counts is AnnData-only; for a cytome pass "
                                  "a layer already on the reference scale, or reuse its "
                                  "`{modality}_infog_params`.")
    return obj.layers[layer] if layer is not None else obj.X


def _n_obs(obj):
    return obj.n_obs if hasattr(obj, "n_obs") else len(obj.cells.to_pandas())


def _var_names(obj, modality="RNA"):
    """Feature names in MATRIX COLUMN ORDER.

    The ``ORDER BY {idx_col}`` is essential, not cosmetic. Without it SQLite is free to satisfy
    ``select {name_col} from {tbl}`` from a covering index on the name column, which returns rows
    in *lexicographic* order -- measured on a 4,000-gene cytome the names came back
    ``g0, g1, g10, g100, g1000, ...`` instead of ``g0, g1, g2, ...``. Every consumer here treats
    position i as matrix column i (frozen-INFOG gene alignment, marker-gene recovery), so an
    unordered read silently mis-maps every gene.
    """
    if _is_cytome(obj):
        from cytome import modality_feature_table_info as _mfi
        _tbl, _idx_col, name_col = _mfi(obj, modality)
        # Go through the public accessor rather than hand-rolling SQL: EntityTable.__getitem__
        # already does `ORDER BY ROWID`, so the ordering guarantee lives in one place instead of
        # being re-derived (and forgotten) at each call site. This is exactly what the previous
        # hand-rolled query got wrong.
        return np.asarray(obj.features(modality)[name_col]).astype(str)
    return np.asarray(obj.var_names).astype(str)


# --------------------------------------------------------------------------------------------
# reference state: the recipe is written by runGDR, the heavy parts completed lazily here
# --------------------------------------------------------------------------------------------
def _make_reference_recipe(marker_gene, *, block_indices=None, layer="infog", groupby=None,
                           batch_key=None, random_seed=1927, denovo_labels=None,
                           n_nearest_neighbors=30, leaf_size=40, modality="RNA") -> dict:
    """The cheap state runGDR can record with no extra computation."""
    return {
        "schema_version": GDR_REFERENCE_SCHEMA_VERSION,
        "marker_gene": marker_gene,
        "block_indices": None if block_indices is None else list(map(int, block_indices)),
        "axis1_mode": "whole_row" if block_indices is None else "per_block",
        "layer": layer,
        "groupby": groupby,
        "batch_key": batch_key,
        "random_seed": int(random_seed),
        "n_nearest_neighbors": int(n_nearest_neighbors),
        "leaf_size": int(leaf_size),
        "modality": modality,
        "denovo_labels": denovo_labels,
        "fitted": False,
    }


def _complete_reference(reference_data, state, *, max_workers=8, counts_layer=None,
                        novelty_k=15, batch_size=1024, verbosity=1) -> dict:
    """Fill in the parts that need a pass over the reference, then cache back into the object.

    Computed here rather than in ``runGDR`` so that running GDR costs nothing extra; the one-time
    cost lands on the first ``projectGDR`` call, where a scoring pass is happening anyway.
    """
    if state.get("fitted"):
        return state
    layer = state.get("layer", "infog")
    modality = state.get("modality", "RNA")
    if verbosity:
        print("projectGDR: completing reference state (first use; cached afterwards)", flush=True)

    if _is_cytome(reference_data):
        matrix = None      # knn from the streaming stats path inside score()
    else:
        matrix = reference_data.layers[layer] if layer is not None else reference_data.X
    if matrix is not None:
        state["knn_idx"] = _reference_knn(matrix, state["n_nearest_neighbors"],
                                          state["leaf_size"]).astype(np.int32)

    S, names = _raw_scores(reference_data, state["marker_gene"], layer, max_workers,
                           state["random_seed"], 0, state.get("knn_idx"), modality, batch_size)
    X, col_norm = _gdr_normalize(S, None, state.get("block_indices"))
    state["col_norm"] = col_norm
    state["gene_set_names"] = np.asarray(list(map(str, names)))
    state["n_ref_cells"] = int(S.shape[0])
    state["var_names"] = _var_names(reference_data, modality)

    if counts_layer is not None and not _is_cytome(reference_data):
        state["infog"] = _infog_reference_stats(
            _counts_matrix(reference_data, counts_layer), _var_names(reference_data))
    elif _is_cytome(reference_data):
        try:                                    # reuse the cytome's own stored INFOG params
            p = reference_data.metadata[f"{modality}_infog_params"]
            state["infog"] = {k: p[k] for k in
                              ("inv_gene_depth", "scale", "counts_sum", "threshold") if k in p}
            state["infog"]["var_names"] = _var_names(reference_data, modality)
        except (KeyError, TypeError):
            pass

    # novelty calibration: leave-one-out mean kNN distance over the reference, ALL cells kept
    from sklearn.neighbors import NearestNeighbors
    k = int(min(novelty_k + 1, X.shape[0]))
    d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    self_nov = d[:, 1:].mean(axis=1)
    state["novelty_reference"] = self_nov
    # string keys: float-keyed dicts cannot be written to h5ad
    state["novelty_thresholds"] = {f"{q:g}": float(np.quantile(self_nov, q))
                                   for q in (0.95, 0.99, 0.995, 0.999)}
    state["novelty_k"] = int(novelty_k)
    # X_gdr_reference is NOT stored: it is already obsm['X_gdr'] / ds.embeddings on the object
    state["fitted"] = True
    return state


def _reference_embedding(reference_data, state, key="X_gdr"):
    """The reference coordinates — read from the object rather than duplicated in the state."""
    if _is_cytome(reference_data):
        for k in (key, "X_gdr"):
            try:
                return np.asarray(reference_data.embeddings[k])
            except (KeyError, TypeError):
                continue
        raise KeyError(f"reference cytome has no '{key}' embedding")
    for k in (key, "X_gdr"):
        if k in reference_data.obsm:
            return np.asarray(reference_data.obsm[k])
    raise KeyError(f"reference AnnData has no obsm['{key}']")


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------
def projectGDR(
    data,
    reference,
    mode: Literal["reference", "self"] = "reference",
    layer: Optional[str] = None,
    counts_layer: Optional[str] = None,
    key_added: str = "X_gdr",
    modality: str = "RNA",
    reference_modality: Optional[str] = None,
    max_workers: int = 8,
    batch_size: int = 1024,
    min_cells_self_mode: int = 500,
    novelty_k: int = 15,
    novelty_quantile: float = 0.99,
    write_to_cytome: bool = True,
    copy: bool = False,
    verbosity: int = 1,
):
    """Project ``data`` into the frozen GDR space of ``reference``.

    Parameters
    ----------
    data
        Query cells: an ``AnnData``, a ``CytomeDataset``, or a path to either. Cytome queries are
        scored by streaming; the expression matrix is never materialised.
    reference
        The reference the GDR space was built on — an ``AnnData``/``CytomeDataset``/path carrying
        ``uns['gdr_reference']`` / ``metadata['gdr_reference']`` from ``runGDR(save_reference=True)``.
        Parts of that state are completed on first use and cached back, so the second call is free.
    mode
        ``'reference'`` scales the query's score columns by the **reference** column norms;
        ``'self'`` uses the query's own, i.e. treats it as a new batch (what ``runGDR`` does for each
        reference batch when ``batch_key`` is set). Falls back to ``'reference'`` below
        ``min_cells_self_mode`` cells, where self-scaling is measurably worse.
    modality
        Modality of the **query**. Cytome only — ignored for AnnData inputs, which have a single
        feature space (``var_names``).
    reference_modality
        Modality of the **reference**. Defaults to whatever the saved state recorded, so existing
        calls are unaffected. Supply it only to be explicit, or when the reference has no recorded
        modality.

        These are deliberately two parameters because they are two objects. Projecting a
        gene-activity query onto an RNA reference is a legitimate thing to want — but doing it *by
        accident* is not, and it cannot be caught downstream: RNA and GA are both keyed by gene
        symbol, so the marker-recovery guard below sees matching names and passes while the values
        come from the wrong assay. A mismatch is therefore reported here, loudly.
    layer
        Layer to score. Defaults to whatever the reference used. Applies to **both** AnnData and
        cytome — there is no separate ``cytome_layer``.
    counts_layer
        AnnData only. If given, the query is INFOG-normalised with the **reference's** frozen
        constants before scoring, which is the correct choice: it keeps the query on the reference's
        normalisation scale.

    Notes
    -----
    GDR axes are the reference's marker-set scores, so a cell type absent from the reference still
    receives confident-looking coordinates — measured, a held-out subclass lands on its nearest
    relative (SST-Chodl→SST, PV→PV-Chandelier, VIP→SNCG 95 %) with no intrinsic warning. **Read
    ``obs['<key>_novelty']`` and ``uns['<key>_projection']['novelty_test']` before interpreting the
    coordinates.** The per-cell flag is well calibrated on false positives but low-powered (TPR 0.12
    at q99); the population-level shift is the reliable readout.
    """
    data = _open(data)
    reference = _open(reference)
    if copy:
        data = data.copy()

    state = _get_state(reference)
    if state is None:
        raise ValueError(
            "reference carries no 'gdr_reference' state. Run "
            "`piaso.tl.runGDR(reference, ..., save_reference=True)` first.")

    # ── reconcile query vs reference modality ───────────────────────────────────────────────
    # `modality` describes the QUERY; `state['modality']` describes the REFERENCE. They used to be
    # the same parameter, so a reference built on one modality could be scored against a query
    # read from another with nothing to notice. That cannot be caught later: RNA and GA are both
    # keyed by gene symbol, so the marker-recovery guard below sees matching names and passes.
    ref_mod_recorded = state.get("modality")
    ref_mod = reference_modality if reference_modality is not None else ref_mod_recorded
    if ref_mod is None:
        ref_mod = modality
    if (reference_modality is not None and ref_mod_recorded is not None
            and reference_modality != ref_mod_recorded):
        raise ValueError(
            f"projectGDR: reference_modality={reference_modality!r} contradicts the modality "
            f"recorded in the saved reference state ({ref_mod_recorded!r}). The state is written "
            f"by runGDR(save_reference=True) and describes how the reference was actually built; "
            f"either drop reference_modality or rebuild the reference.")
    if _is_cytome(reference) and ref_mod != modality:
        warnings.warn(
            f"projectGDR: the query modality ({modality!r}) differs from the reference modality "
            f"({ref_mod!r}). This is legitimate — projecting gene activity onto an RNA reference "
            f"is a normal thing to do — but it is not detectable downstream, because both are "
            f"keyed by gene symbol. Pass reference_modality={ref_mod!r} explicitly to silence "
            f"this if it is intended.", UserWarning)
    if verbosity:
        print(f"projectGDR: query modality {modality!r}, reference modality {ref_mod!r}",
              flush=True)

    state = _complete_reference(reference, state, max_workers=max_workers,
                                counts_layer=counts_layer, novelty_k=novelty_k,
                                batch_size=batch_size, verbosity=verbosity)
    _set_state(reference, state)                      # cache the completed state back

    n_query = _n_obs(data)
    if mode == "self" and n_query < min_cells_self_mode:
        warnings.warn(f"projectGDR: mode='self' needs a reasonably sized, mixed query; got "
                      f"{n_query} cells (< {min_cells_self_mode}). Using mode='reference'.",
                      UserWarning)
        mode = "reference"

    use_layer = layer if layer is not None else state.get("layer", "infog")
    gene_recovery = 1.0
    query_use = data

    # 1a. frozen INFOG for a cytome query: materialise a layer with the reference's constants
    if counts_layer is not None and "infog" in state and _is_cytome(data):
        use_layer, gene_recovery = _infog_apply_cytome(
            data, state["infog"], modality=modality, counts_layer=counts_layer,
            batch_size=batch_size, verbosity=verbosity)
        if verbosity:
            print(f"projectGDR: INFOG applied with frozen reference constants (streaming); "
                  f"gene recovery {gene_recovery:.1%}", flush=True)
    # 1b. frozen INFOG (AnnData + counts_layer)
    elif counts_layer is not None and "infog" in state and not _is_cytome(data):
        import anndata as ad
        norm, gene_recovery = _infog_apply(data.layers[counts_layer], state["infog"],
                                           var_names=data.var_names)
        query_use = ad.AnnData(X=norm, obs=data.obs.copy(),
                               var=pd.DataFrame(index=pd.Index(state["infog"]["var_names"])))
        query_use.layers[use_layer] = norm
        if verbosity:
            print(f"projectGDR: INFOG applied with frozen reference constants; "
                  f"gene recovery {gene_recovery:.1%}", flush=True)

    # 2. marker-gene recovery
    mg = state["marker_gene"]
    if not isinstance(mg, pd.DataFrame):
        mg = pd.DataFrame(mg)
    qgenes = set(map(str, _var_names(query_use, modality)))
    rec = pd.Series({c: float(np.mean([str(g) in qgenes for g in mg[c].dropna()]))
                     for c in mg.columns})
    if verbosity:
        print(f"projectGDR: marker-gene recovery — median {rec.median():.1%}, min {rec.min():.1%}, "
              f"{int((rec < 0.6).sum())}/{len(rec)} sets below 60%", flush=True)
    if (rec < 0.2).mean() > 0.5:
        raise ValueError("projectGDR: over half the marker sets recover <20% of their genes in the "
                         "query; the gene spaces are not compatible.")

    # 3. score + frozen normalisation
    knn_idx = state.get("knn_idx")
    if knn_idx is not None:
        knn_idx = np.asarray(knn_idx)
        n_q_genes = len(_var_names(query_use, modality))
        if knn_idx.shape[0] != n_q_genes:
            if verbosity:
                print(f"projectGDR: frozen KNN covers {knn_idx.shape[0]} genes but the query has "
                      f"{n_q_genes}; scoring without it (pass `counts_layer` to realign).",
                      flush=True)
            knn_idx = None
    S, _ = _raw_scores(query_use, mg, use_layer, max_workers, state["random_seed"], 0,
                       knn_idx, modality, batch_size)
    col_norm = None if mode == "self" else np.asarray(state["col_norm"])
    X, _ = _gdr_normalize(S, col_norm, state.get("block_indices"))

    # 4. novelty — per cell, plus the population test that actually works
    Xref = _reference_embedding(reference, state, key_added)
    from sklearn.neighbors import NearestNeighbors
    k = int(min(state.get("novelty_k", novelty_k), len(Xref)))
    dist, _ = NearestNeighbors(n_neighbors=k).fit(Xref).kneighbors(X)
    nov = dist.mean(axis=1)
    thr = state.get("novelty_thresholds", {})
    qkey = f"{novelty_quantile:g}"
    cut = float(thr[qkey]) if qkey in thr else None
    ntest = {}
    ref_nov = state.get("novelty_reference")
    if ref_nov is not None and len(ref_nov):
        from scipy.stats import mannwhitneyu, ks_2samp
        u, pm = mannwhitneyu(nov, np.asarray(ref_nov), alternative="greater")
        ks, pk = ks_2samp(nov, np.asarray(ref_nov))
        ntest = {"median_query": float(np.median(nov)),
                 "median_reference": float(np.median(ref_nov)),
                 "mannwhitney_p": float(pm), "ks_stat": float(ks), "ks_p": float(pk),
                 "fraction_above_threshold": (float(np.mean(nov > cut)) if cut else np.nan),
                 "threshold_quantile": qkey}
        if verbosity:
            print(f"projectGDR: novelty — query median {ntest['median_query']:.4f} vs reference "
                  f"{ntest['median_reference']:.4f}, MWU p={pm:.3g}, KS={ks:.3f}"
                  + (f", {ntest['fraction_above_threshold']:.1%} above q{qkey}" if cut else ""),
                  flush=True)

    # 5. write results
    info = {"mode": mode, "gene_recovery": float(gene_recovery),
            "marker_set_recovery_median": float(rec.median()),
            "n_sets_below_60pct": int((rec < 0.6).sum()),
            "axis1_mode": state.get("axis1_mode", "whole_row"),
            "control_genes_frozen": bool(knn_idx is not None),
            "novelty_test": ntest}
    if _is_cytome(data):
        if write_to_cytome:
            data.add_embedding(key_added, X)
            data.metadata[f"{key_added}_projection"] = info
            if verbosity:
                print(f"projectGDR: wrote embeddings['{key_added}'] ({X.shape})", flush=True)
            return None
        return X
    data.obsm[key_added] = X
    data.obs[f"{key_added}_novelty"] = nov
    if cut is not None:
        data.obs[f"{key_added}_is_novel"] = nov > cut
    data.obs[f"{key_added}_mode"] = mode
    data.uns[f"{key_added}_projection"] = info
    return data if copy else None
