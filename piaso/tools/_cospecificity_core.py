"""RNA trans co-specificity: the released core of the co-specificity family.

`cospecificity_trans` scores candidate feature pairs (e.g. a cistrome's
motif-supported TF -> target map) by how similarly the two features' COSG
cell-type specificity profiles behave — the trans leg regulon assembly
consumes downstream (cytorete). This module also holds the shared plumbing
(the specificity matrix from COSG, the pairwise metrics, the COSG dense
cache) that the rest of the family builds on.

The genomic map/genome-wide/anchored machinery lives in `_cospecificity`,
which is not part of the public distribution; it imports these helpers, so
there is exactly one implementation of each.
"""
from __future__ import annotations

import functools

import warnings
import zlib
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# One predicate for the whole package. Four modules each had their own: three
# identical isinstance checks and one duck-type on `iter_chunks`, none of which
# noticed a *closed* dataset -- which then failed as a sqlite "Cannot operate on
# a closed database" several frames away. The shared one raises where the
# mistake is.
from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome


# ---------------------------------------------------------------------------
# COSG dense matrix caching (in ds.metadata, checked against a digest of its
# inputs, stored as compressed raw arrays: no pickle)
# ---------------------------------------------------------------------------

_COSG_CACHE_KEY_DEFAULT = None  # auto-derive from (mu, expressed_pct, layer)

#: The cache entry's format. An entry without it (the base64 pickles written
#: before 2026-10-27) is a miss and is replaced, never unpickled: loading a
#: pickle from a shared store runs whatever code it holds.
_COSG_CACHE_FORMAT = "piaso-cosg-dense/2"
_COSG_CACHE_DTYPES = ("float32", "float64")


def _resolve_modality_n_features(ds, modality: str) -> int:
    """Return the number of features for the given modality."""
    if modality in ("ATAC", "peaks"):
        return int(ds.n_peaks)
    if modality == "tiles":
        return int(
            ds._conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        )
    if modality in ("RNA", "genes"):
        return int(ds.n_genes)
    raise ValueError(
        f"_resolve_modality_n_features: modality {modality!r} not "
        f"supported. Use 'ATAC', 'tiles', or 'RNA'."
    )


def _filter_top_n_per_group(
    scores_df: pd.DataFrame, n_top: int,
) -> pd.DataFrame:
    """Keep top-``n_top`` features per group (column); zero the rest.

    Used at usage time to reduce a comprehensive cached COSG dense
    matrix to a user-requested top-N view. The underlying cache is
    untouched (read-only); this returns a fresh DataFrame.
    """
    values = scores_df.values
    n_features, n_groups = values.shape
    if n_top is None or n_top >= n_features:
        return scores_df
    out = np.zeros_like(values)
    for g in range(n_groups):
        col = values[:, g]
        if (col > 0).sum() <= n_top:
            # Already at or below cap; keep all positive
            out[:, g] = col
        else:
            idx = np.argpartition(col, -n_top)[-n_top:]
            out[idx, g] = col[idx]
    return pd.DataFrame(out, index=scores_df.index, columns=scores_df.columns)


def _derive_cosg_cache_key(
    *, cosg_mu: float, cosg_expressed_pct: float, cosg_layer: str,
    groupby: str, modality: str, cosg_remove_lowly_expressed: bool = True,
) -> str:
    """Auto-derive the cache key from parameters that change the COSG
    output. ``n_genes_user`` is NOT part of the key because we always
    compute at ``ds.n_peaks`` internally and filter at usage time;
    different ``n_genes_user`` values share the same cache.

    Format: ``cospec_cosg_dense__<groupby>_<modality>_mu=<>_pct=<>_layer=<>``,
    with ``_filter=off`` appended when COSG's detection filter is off (the
    default keeps the key it always had, so existing caches still match).
    """
    return (
        f"cospec_cosg_dense__"
        f"{groupby}_{modality}_"
        f"mu={cosg_mu}_pct={cosg_expressed_pct}_layer={cosg_layer}"
        + ("" if cosg_remove_lowly_expressed else "_filter=off")
    )


def _cosg_cache_inputs(ds, *, groupby: str, modality: str, cosg_layer: str,
                       cosg_params: dict) -> dict:
    """What a cached COSG dense matrix was computed from, as a JSON-able dict
    whose digest keys the entry: the labels in cell order (a digest), the number
    of cells, the store's category order for ``groupby``, the identity of every
    matrix COSG may read for ``(modality, layer)`` (``{modality}_{layer}``, and
    ``{modality}_counts``, which an unmaterialised layer is computed from),
    the normalisation parameters cached for that layer, COSG's version and the
    run's parameters.

    So a relabel, a subset (cytome's ``subset`` copies the metadata, so the
    copied entry meets other cells), a re-quantification or a COSG upgrade makes
    the entry stale, and it is recomputed.
    """
    from ..utils._cytome_compat import require_cells_column
    from ..utils import _store_digest as D
    import cosg

    require_cells_column(ds, groupby, argument="cospecificity: groupby")
    layer = cosg_layer or "counts"
    matrices = [D.matrix_identity(ds, m) for m in dict.fromkeys((f"{modality}_{layer}", f"{modality}_counts"))]
    try:
        cats = ds.metadata.get("categories") or {}
        order = (cats.get(groupby) or {}).get("order") if isinstance(cats, dict) else None
        layer_params = ds.metadata.get(f"{modality}_{layer}_params")
    except Exception:
        order, layer_params = None, None
    return {
        "groupby": groupby,
        "labels": D.labels_digest(D.cell_column(ds, groupby)),
        "n_cells": int(ds.n_cells),
        "category_order": None if order is None else [str(x) for x in order],
        "matrices": matrices,
        "layer_params": None if layer_params is None else D.digest(layer_params),
        "cosg_version": str(getattr(cosg, "__version__", "unknown")),
        "params": cosg_params,
    }


def _save_cosg_cache(ds, key: str, scores_df: pd.DataFrame,
                     cosg_params: dict, inputs: dict, *, verbose: int = 1) -> None:
    """Persist the COSG dense output in ``ds.metadata[key]``.

    Stored as raw arrays, zlib-compressed (COSG's dense output is mostly
    zeros): the values in C order with their dtype and shape, the feature names
    and the group names, plus the digest of the inputs it was computed from
    (:func:`_cosg_cache_inputs`). Nothing in it executes on load. A store
    opened read-only keeps no cache: the result is returned all the same.
    """
    from ..utils import _store_digest as D

    if getattr(ds, "read_only", False):
        if verbose:
            print(f"cospecificity: the store is read-only; the COSG dense matrix is "
                  f"not cached (ds.metadata[{key!r}])")
        return
    values = np.ascontiguousarray(scores_df.to_numpy())
    if values.dtype.name not in _COSG_CACHE_DTYPES:
        values = values.astype(np.float32)
    names = [str(x) for x in scores_df.index]
    if any("\n" in n for n in names):
        raise ValueError("cospecificity: a feature name holds a newline; it cannot be cached")
    payload = {
        "format": _COSG_CACHE_FORMAT,
        "digest": D.digest(inputs),
        "inputs": inputs,
        "cosg_params": cosg_params,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "dtype": values.dtype.name,
        "columns": pd.Index(scores_df.columns).tolist(),
        "index_zlib": zlib.compress("\n".join(names).encode("utf-8"), 1),
        "values_zlib": zlib.compress(values.tobytes(), 1),
    }
    ds.metadata[key] = payload
    ds.flush()


def _load_cosg_cache(ds, key: str, inputs: dict) -> tuple[pd.DataFrame | None, str]:
    """``(frame, why)``: the cached COSG dense matrix when the entry was computed
    from ``inputs`` (its digest matches), else ``(None, reason)``. An entry in an
    older format (the base64 pickles) is a miss: it is never unpickled."""
    from ..utils import _store_digest as D

    payload = ds.metadata.get(key)
    if payload is None:
        return None, "absent"
    if not isinstance(payload, dict) or payload.get("format") != _COSG_CACHE_FORMAT:
        return None, "an older format (replaced)"
    if payload.get("digest") != D.digest(inputs):
        return None, "stale: the labels, the cells, the matrix or COSG changed since it was cached"
    try:
        n, g = (int(x) for x in payload["shape"])
        dtype = str(payload["dtype"])
        if dtype not in _COSG_CACHE_DTYPES:
            return None, f"unexpected dtype {dtype!r}"
        values = np.frombuffer(zlib.decompress(payload["values_zlib"]), dtype=dtype).reshape(n, g)
        index = zlib.decompress(payload["index_zlib"]).decode("utf-8").split("\n") if n else []
        columns = list(payload["columns"])
        if len(index) != n or len(columns) != g:
            return None, "corrupt (shape)"
    except (KeyError, TypeError, ValueError, zlib.error) as exc:
        return None, f"unreadable ({exc})"
    return pd.DataFrame(values.copy(), index=pd.Index(index), columns=pd.Index(columns)), "hit"


def _ensure_cosg_dense(
    ds, *, groupby: str, modality: str,
    cosg_mu: float, cosg_expressed_pct: float,
    cosg_layer: str,
    cosg_batch_size: int | None,
    use_cached: bool, cache_key: str | None,
    verbose: int,
    cosg_remove_lowly_expressed: bool = True,
) -> pd.DataFrame:
    """Load cached COSG dense scores or compute + cache them.

    Always computes COSG at the maximum ``n_genes_user`` (i.e.
    ``ds.n_peaks`` / ``ds.n_genes`` etc.) regardless of caller's
    intent. The user-facing ``cosg_n_genes_user`` is applied as
    a **post-process filter** at the call site (see
    :func:`_filter_top_n_per_group`), so a single cached matrix
    serves both ``cospecificity_map(cosg_n_genes_user=None)`` and
    ``specificity_hotspot(top_n=2000)``.

    The metadata key is auto-derived from
    ``(groupby, modality, mu, expressed_pct, layer)`` when
    ``cache_key=None`` — the knobs that actually change the COSG
    λ values. Typical session: 1-2 cache entries per cytome.

    An entry is used only when it was computed from the same inputs: the
    labels, the number of cells, the matrix COSG reads, COSG's version and the
    parameters (a digest, :func:`_cosg_cache_inputs`); otherwise it is
    recomputed and replaced. So an entry never outlives a relabel, a subset or
    a re-quantification, under the derived key or an explicit ``cache_key``.
    """
    if cache_key is None:
        cache_key = _derive_cosg_cache_key(
            cosg_mu=cosg_mu, cosg_expressed_pct=cosg_expressed_pct,
            cosg_layer=cosg_layer,
            groupby=groupby, modality=modality,
            cosg_remove_lowly_expressed=cosg_remove_lowly_expressed,
        )

    import cosg

    # Always compute at maximum n_genes_user. Resolve from modality.
    feature_count = _resolve_modality_n_features(ds, modality)
    params = {
        "groupby": groupby, "modality": modality,
        "cosg_mu": cosg_mu, "cosg_expressed_pct": cosg_expressed_pct,
        "cosg_n_genes_user": feature_count,
        "layer": cosg_layer,
        "cosg_remove_lowly_expressed": bool(cosg_remove_lowly_expressed),
    }
    inputs = _cosg_cache_inputs(ds, groupby=groupby, modality=modality,
                                cosg_layer=cosg_layer, cosg_params=params)

    if use_cached:
        cached, why = _load_cosg_cache(ds, cache_key, inputs)
        if cached is not None:
            if verbose:
                print(f"cospecificity: reusing cached COSG dense at "
                      f"ds.metadata[{cache_key!r}] "
                      f"(shape {cached.shape})")
            return cached
        if verbose and why != "absent":
            print(f"cospecificity: the cached COSG dense at ds.metadata[{cache_key!r}] "
                  f"is not used: {why}")

    if verbose:
        print(f"cospecificity: running COSG (groupby={groupby!r}, "
              f"modality={modality!r}, layer={cosg_layer!r}, "
              f"n_genes_user={feature_count} = ds.n_{modality.lower()}; "
              f"COMPREHENSIVE — filtered to top-N per group at usage "
              f"time) — this may take a few seconds to minutes "
              f"depending on cytome size...")

    cosg_kwargs = dict(
        groupby=groupby, modality=modality,
        output_format="dense",
        n_genes_user=feature_count,
        mu=cosg_mu, expressed_pct=cosg_expressed_pct,
        remove_lowly_expressed=cosg_remove_lowly_expressed,
        layer=cosg_layer, verbose=bool(verbose),
    )
    if cosg_batch_size is not None:
        cosg_kwargs["batch_size"] = cosg_batch_size
    result = cosg.run_cosg_cytome(ds, **cosg_kwargs)
    scores_df = result["scores_df"]

    # COSG may cache its normalisation on first use (tfidf): describe the inputs as
    # the next reader will see them
    inputs = _cosg_cache_inputs(ds, groupby=groupby, modality=modality,
                                cosg_layer=cosg_layer, cosg_params=params)
    _save_cosg_cache(ds, cache_key, scores_df, params, inputs, verbose=verbose)
    if verbose and not getattr(ds, "read_only", False):
        print(f"cospecificity: cached COSG dense to "
              f"ds.metadata[{cache_key!r}] (shape {scores_df.shape})")
    return scores_df


# ---------------------------------------------------------------------------
# peak coordinate resolution
# ---------------------------------------------------------------------------


_VALID_METRICS = ("geomean", "outer", "cosine", "weighted_cosine")


def _pairwise_metric(
    scores: np.ndarray, metric: str, target_col: int | None,
) -> np.ndarray:
    """Compute the ``(n, n)`` pairwise score matrix.

    Parameters
    ----------
    scores
        ``(n_features, n_celltypes)`` specificity matrix.
    metric
        - ``"geomean"`` (default): ``sqrt(s_t(i) * s_t(j))`` —
          symmetric, weakest-link, clips negatives to 0.
        - ``"outer"``: ``s_t(i) * s_t(j)`` — raw product.
        - ``"cosine"``: cell-type-agnostic profile cosine
          similarity (``target_col`` ignored).
        - ``"weighted_cosine"``: target outer × profile cosine.
    target_col
        Index of the cell-type column used by per-target metrics.
        Required for all metrics except ``"cosine"``.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"metric {metric!r} not in {_VALID_METRICS}"
        )
    if metric == "cosine":
        norms = np.linalg.norm(scores, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        unit = scores / norms
        return unit @ unit.T
    if target_col is None:
        raise ValueError(
            f"metric {metric!r} requires a target cell-type column"
        )
    s = scores[:, target_col].astype(np.float64)
    if metric == "outer":
        return np.outer(s, s).astype(np.float32)
    if metric == "geomean":
        prod = np.outer(s, s)
        return np.sqrt(np.maximum(prod, 0.0)).astype(np.float32)
    # weighted_cosine
    norms = np.linalg.norm(scores, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = scores / norms
    cos = unit @ unit.T
    return (cos * np.outer(s, s)).astype(np.float32)


# ---------------------------------------------------------------------------
# genomic-grid binning
# ---------------------------------------------------------------------------


def _specificity_matrix(
    data, *, groupby: str, modality: str = "RNA",
    cosg_mu: float = 1.0, cosg_expressed_pct: float = 0.1,
    cosg_layer: str = "counts", cosg_batch_size=None,
    use_cached: bool = True, cosg_cache_key: str | None = _COSG_CACHE_KEY_DEFAULT,
    verbose: int = 1, cosg_remove_lowly_expressed: bool = True,
) -> pd.DataFrame:
    """COSG λ specificity matrix ``S`` [features × cell_types], for AnnData or
    cytome. Cytome reuses the cached :func:`_ensure_cosg_dense`; AnnData runs
    ``cosg.cosg`` at full ``n_genes_user`` and pivots ``uns`` into a dense frame.

    ``cosg_remove_lowly_expressed``: COSG's detection filter. With the default
    ``True``, a feature expressed in under ``cosg_expressed_pct`` of a group's
    cells is scored −1 in that group, as cytorete's regulon assembly expects.
    For :func:`piaso.tl.runSCALAR` pass ``False``: SCALAR refuses negative
    entries (two −1 multiply to +1) and applies its own detection floor.

    ``cosg_layer`` is the matrix COSG scores, on both backends: a cytome's
    ``{modality}_{cosg_layer}`` (an unmaterialised ``log1p``/``infog``/``tfidf``
    is computed from the counts), an AnnData's ``layers[cosg_layer]``. On an
    AnnData without a ``'counts'`` layer, ``'counts'`` (the default) is ``.X``,
    as a store's counts matrix is what ``from_anndata`` wrote from ``.X``;
    ``None`` is ``.X``; any other name must be a layer. The caller's AnnData is
    not modified (COSG runs on a lightweight copy that shares its matrix).
    """
    if _is_cytome(data):
        return _ensure_cosg_dense(
            data, groupby=groupby, modality=modality,
            cosg_mu=cosg_mu, cosg_expressed_pct=cosg_expressed_pct,
            cosg_layer=cosg_layer, cosg_batch_size=cosg_batch_size,
            use_cached=use_cached, cache_key=cosg_cache_key, verbose=verbose,
            cosg_remove_lowly_expressed=cosg_remove_lowly_expressed,
        )
    # ---- AnnData path ----
    import anndata as _ad
    import cosg as _cosg
    adata = data
    if groupby not in adata.obs.columns:
        raise KeyError(f"specificity_matrix: groupby={groupby!r} is not a column of adata.obs")
    if cosg_layer is None or (cosg_layer == "counts" and "counts" not in adata.layers):
        X, where = adata.X, ".X"
    elif cosg_layer in adata.layers:
        X, where = adata.layers[cosg_layer], f"layers[{cosg_layer!r}]"
    else:
        raise KeyError(
            f"specificity_matrix: cosg_layer={cosg_layer!r} is not a layer of the AnnData "
            f"(layers: {list(adata.layers)}); pass None for .X")
    # COSG writes its result into .uns: run it on a copy that shares the matrix, so
    # the caller's uns['cosg'] (or anything else) is never overwritten
    work = _ad.AnnData(X=X, obs=adata.obs[[groupby]].copy(), var=pd.DataFrame(index=adata.var_names))
    if verbose:
        print(f"cospecificity_trans: running COSG on AnnData {where} "
              f"(groupby={groupby!r}, n_genes_user={adata.n_vars}, mu={cosg_mu})")
    _cosg.cosg(
        work, groupby=groupby, mu=cosg_mu,
        expressed_pct=cosg_expressed_pct,
        remove_lowly_expressed=cosg_remove_lowly_expressed,
        n_genes_user=int(adata.n_vars), key_added="cosg",
    )
    res = work.uns["cosg"]
    groups = list(res["names"].dtype.names)
    var_names = np.asarray(adata.var_names)
    pos = {g: i for i, g in enumerate(var_names)}
    dense = np.zeros((len(var_names), len(groups)), dtype=np.float32)
    for gi, grp in enumerate(groups):
        names = np.asarray(res["names"][grp])
        scores = np.asarray(res["scores"][grp], dtype=np.float32)
        idx = np.array([pos.get(n, -1) for n in names])
        ok = idx >= 0
        dense[idx[ok], gi] = scores[ok]
    return pd.DataFrame(dense, index=var_names, columns=groups)


def _cospecificity_trans_impl(
    data,
    *,
    groupby: str = "leiden",
    pairs: dict | None = None,
    metric: str = "weighted_cosine",
    modality: str = "RNA",
    cosg_mu: float = 1.0,
    cosg_expressed_pct: float = 0.1,
    cosg_layer: str = "counts",
    cosg_batch_size=None,
    use_cached: bool = True,
    cosg_cache_key: str | None = _COSG_CACHE_KEY_DEFAULT,
    specificity: pd.DataFrame | None = None,
    min_sign: float = 0.0,
    verbose: int = 1,
) -> dict:
    """Trans co-specificity between candidate feature pairs (e.g. TF→target).

    Parameters
    ----------
    data
        AnnData or cytome.Dataset (RNA).
    pairs
        ``{source: [target, ...]}`` candidate edges (e.g. the cistrome's
        motif-supported TF→gene map). Only these edges are scored (keeps it
        tractable). If ``None``, all source×target pairs over the features are
        used (expensive — intended for small inputs/tests).
    metric
        Per-cell-type pairwise metric (``weighted_cosine`` default; also
        ``geomean`` / ``cosine`` / ``outer``) — reuses :func:`_pairwise_metric`.
    specificity
        Optional precomputed ``S`` [features × cell_types]; skips COSG.
    min_sign
        Keep only edges whose specificity-profile Pearson correlation
        (activation sign) ``> min_sign``. Default 0 → positive (activating)
        edges only (design Q6/Q7: no repression inference).

    Returns
    -------
    dict
        ``{"edges": DataFrame[source,target,cosine,sign,cospec_max,best_celltype],
        "celltypes": [...], "per_celltype": ndarray[n_edges, n_celltypes],
        "S": DataFrame}``. ``per_celltype[e, t]`` is the metric for edge ``e`` in
        cell type ``t`` — the basis for cell-type-specific regulons.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric {metric!r} not in {_VALID_METRICS}")
    S = specificity if specificity is not None else _specificity_matrix(
        data, groupby=groupby, modality=modality,
        cosg_mu=cosg_mu, cosg_expressed_pct=cosg_expressed_pct,
        cosg_layer=cosg_layer, cosg_batch_size=cosg_batch_size,
        use_cached=use_cached, cosg_cache_key=cosg_cache_key, verbose=verbose,
    )
    feat_pos = {f: i for i, f in enumerate(S.index)}
    celltypes = list(S.columns)
    Sv = S.to_numpy(dtype=np.float64)
    norms = np.maximum(np.linalg.norm(Sv, axis=1), 1e-12)

    # candidate edges
    if pairs is None:
        srcs = list(S.index)
        pairs = {s: list(S.index) for s in srcs}

    rows_src, rows_tgt = [], []
    e_cos, e_sign, e_pcmax, e_pcarg = [], [], [], []
    per_ct_rows = []
    nct = len(celltypes)
    for src, tgts in pairs.items():
        si = feat_pos.get(src)
        if si is None:
            continue
        ti = np.array([feat_pos[t] for t in tgts if t in feat_pos and feat_pos[t] != si])
        if ti.size == 0:
            continue
        sv = Sv[si]                                   # (nct,)
        tv = Sv[ti]                                   # (k, nct)
        # global cosine (celltype-agnostic profile similarity)
        cos = (tv @ sv) / (norms[ti] * norms[si])     # (k,)
        # activation sign = Pearson corr of specificity profiles across celltypes
        sc = sv - sv.mean()
        tc = tv - tv.mean(axis=1, keepdims=True)
        denom = np.maximum(np.linalg.norm(sc) * np.linalg.norm(tc, axis=1), 1e-12)
        sign = (tc @ sc) / denom                      # (k,)
        # per-cell-type metric
        if metric == "cosine":
            pc = np.repeat(cos[:, None], nct, axis=1)
        else:
            outer = tv * sv[None, :]                   # (k, nct) = S[t,c]*S[s,c]
            if metric == "outer":
                pc = outer
            elif metric == "geomean":
                pc = np.sqrt(np.maximum(outer, 0.0))
            else:  # weighted_cosine
                pc = cos[:, None] * outer
        pcmax = pc.max(axis=1)
        pcarg = pc.argmax(axis=1)
        for j, tgt_idx in enumerate(ti):
            if sign[j] <= min_sign:
                continue
            rows_src.append(src)
            rows_tgt.append(S.index[tgt_idx])
            e_cos.append(float(cos[j]))
            e_sign.append(float(sign[j]))
            e_pcmax.append(float(pcmax[j]))
            e_pcarg.append(celltypes[int(pcarg[j])])
            per_ct_rows.append(pc[j])

    edges = pd.DataFrame({
        "source": rows_src, "target": rows_tgt,
        "cosine": e_cos, "sign": e_sign,
        "cospec_max": e_pcmax, "best_celltype": e_pcarg,
    })
    per_ct = (np.vstack(per_ct_rows).astype(np.float32)
              if per_ct_rows else np.zeros((0, nct), np.float32))
    if verbose:
        print(f"cospecificity_trans: {len(edges)} positive-sign edges "
              f"over {len(celltypes)} cell types (metric={metric})")
    return {"edges": edges, "celltypes": celltypes,
            "per_celltype": per_ct, "S": S}



@functools.wraps(_cospecificity_trans_impl)
def cospecificity_trans(data, **kwargs) -> dict:
    """Trans co-specificity, accepting a cytome path as well as an open one."""
    from ..utils._cytome_compat import _is_path_like, open_cytome

    if _is_path_like(data):
        # A path opened here is closed here; a Dataset handed in belongs to
        # the caller and is left open.
        with open_cytome(data) as ds:
            return _cospecificity_trans_impl(ds, **kwargs)
    return _cospecificity_trans_impl(data, **kwargs)
