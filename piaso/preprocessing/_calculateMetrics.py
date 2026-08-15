### Calculate basic metrics

from typing import Optional
import scipy.sparse
import anndata
import numpy as np
import pandas as pd


from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome
from ..utils._cytome_compat import open_cytome_sync as _open_cytome


# Per-modality cell-metric column names. The sum-per-cell and nnz-per-cell
# metrics mean different things per modality, so they get modality-appropriate
# names (PIASO-internal convention, not scanpy's total_counts/n_genes_by_counts).
# ATAC names are unchanged for back-compat (QC config + existing cytomes read
# n_fragments_in_peak / n_peaks).
_CELL_METRIC_NAMES = {
    "ATAC":  ("n_fragments_in_peak", "n_peaks"),
    "RNA":   ("n_counts", "n_genes"),
    "GA":    ("GA_n_counts", "n_GA_genes"),
    "tiles": ("n_fragments_in_tile", "n_tiles"),
}

# Modalities backed by the fragment_chunks table (so a per-cell ``n_fragments``
# total is meaningful). RNA / GA have no fragments.
_FRAGMENT_BACKED = {"ATAC", "tiles"}


def _canon_modality(modality: str) -> str:
    """Normalise a modality name to its registry key (RNA/GA/ATAC upper, tiles lower)."""
    up = modality.upper()
    return up if up in {"RNA", "GA", "ATAC"} else modality


def _resolve_feature_masks(names, prefix_vars, feature_set_vars):
    """Build per-key boolean feature masks over an ordered array of feature names.

    ``prefix_vars`` maps a key to a prefix string or list of prefixes
    (``startswith`` match, e.g. ``{'mt': 'MT-', 'ribo': ['RPS', 'RPL']}``).
    ``feature_set_vars`` maps a key to an explicit list of exact feature names.
    Returns ``{key: bool mask}`` aligned to ``names`` order.
    """
    names = np.asarray([("" if n is None else str(n)) for n in names], dtype=object)
    masks = {}
    for key, prefixes in (prefix_vars or {}).items():
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        m = np.zeros(len(names), dtype=bool)
        for p in prefixes:
            m |= np.fromiter((n.startswith(p) for n in names), dtype=bool, count=len(names))
        masks[str(key)] = m
    for key, feats in (feature_set_vars or {}).items():
        fset = set(str(f) for f in feats)
        masks[str(key)] = np.fromiter((n in fset for n in names), dtype=bool, count=len(names))
    return masks


def calculateCellMetrics(source, layer: Optional[str] = None,
                         modality: str = "RNA", measurement: str = "counts",
                         batch_size: int = 1024, verbose: bool = True,
                         prefix_vars: Optional[dict] = None,
                         feature_set_vars: Optional[dict] = None,
                         feature_name_column: Optional[str] = None):
    """
    Calculate per-cell QC metrics and store them in adata.obs or cytome cells table.

    Computes two values per cell from the chosen modality's count matrix, plus
    (for fragment-backed modalities) the total fragment count:

    - sum-per-cell  — total counts/fragments (modality-specific column name)
    - nnz-per-cell  — number of active features (modality-specific column name)
    - ``n_fragments`` — total fragments from the fragment_chunks table
      (ATAC / tiles only; cytome mode).
    - ``frip`` — fraction of fragments in peaks =
      ``n_fragments_in_peak / n_fragments`` (ATAC only, cytome mode, written
      automatically when ``n_fragments`` is available). Filter on it directly,
      e.g. ``piaso.pp.filter_cells(ds, modality='ATAC', mask={'frip': (0.2, None)})``.

    The column names depend on ``modality`` (cytome mode):

    ======== ====================== ============ =====================
    modality sum-per-cell           nnz-per-cell fragment total
    ======== ====================== ============ =====================
    ATAC     ``n_fragments_in_peak`` ``n_peaks``  ``n_fragments``
    RNA      ``n_counts``            ``n_genes``  — (no fragments)
    GA       ``GA_n_counts``         ``n_GA_genes`` — (no fragments)
    tiles    ``n_fragments_in_tile`` ``n_tiles``  ``n_fragments``
    ======== ====================== ============ =====================

    Parameters
    ----------
    source : AnnData, cytome.Dataset, or str
        Input data. For AnnData, uses the feature-cell matrix. For cytome
        Dataset or path to .cytome file, streams from the ``{modality}_{measurement}``
        matrix on disk and (ATAC/tiles) reads total fragment counts.
    layer : str, optional
        AnnData layer to use. If None, uses .X. Ignored for cytome.
    modality : str
        Modality: ``'RNA'`` (default), ``'ATAC'``, ``'GA'``, ``'tiles'``. Selects
        the cytome modality to stream AND the per-cell metric column names (see
        table above) — the AnnData path is now modality-aware too (e.g. an RNA
        AnnData gets ``n_counts``/``n_genes``; pass ``modality='ATAC'`` for
        ``n_fragments_in_peak``/``n_peaks``).
    measurement : str
        Cytome measurement/layer (default 'counts'). Ignored for AnnData.
    batch_size : int
        Chunk size for streaming (cytome mode). Default 1024.
    verbose : bool
        Print progress messages.
    prefix_vars : dict, optional
        Map ``{key: prefix}`` or ``{key: [prefixes]}`` for per-cell percentage of
        counts in features whose name starts with any of the prefixes. Writes
        ``n_counts_{key}`` and ``pct_counts_{key}`` (= 100 * masked / total).
        Example: ``{'mt': 'MT-', 'ribo': ['RPS', 'RPL']}`` → ``pct_counts_mt``,
        ``pct_counts_ribo``.
    feature_set_vars : dict, optional
        Map ``{key: [exact feature names]}`` — same outputs as ``prefix_vars`` but
        matched by exact membership rather than prefix.
    feature_name_column : str, optional
        Which feature-name column to match against. AnnData: a ``var`` column
        (default ``var_names``). Cytome: a column of the modality's var table
        (default: the modality's canonical name column, e.g. ``gene_id``).
    """
    if isinstance(source, str) or _is_cytome(source):
        _calculateCellMetrics_cytome(source, modality, measurement, batch_size, verbose,
                                     prefix_vars=prefix_vars,
                                     feature_set_vars=feature_set_vars,
                                     feature_name_column=feature_name_column)
    else:
        _calculateCellMetrics_anndata(source, layer, modality=modality,
                                      prefix_vars=prefix_vars,
                                      feature_set_vars=feature_set_vars,
                                      feature_name_column=feature_name_column)


def _calculateCellMetrics_anndata(adata, layer=None, modality="RNA",
                                  prefix_vars=None, feature_set_vars=None,
                                  feature_name_column=None):
    """AnnData path. Modality-aware column names (sum-per-cell / nnz-per-cell)."""
    matrix = adata.layers[layer] if layer is not None else adata.X

    if not isinstance(matrix, scipy.sparse.spmatrix):
        raise ValueError("The specified layer should contain a sparse matrix.")

    sum_col, nnz_col = _CELL_METRIC_NAMES.get(
        _canon_modality(modality), _CELL_METRIC_NAMES["RNA"])

    # sum-per-cell (total counts/fragments) and nnz-per-cell (active features).
    adata.obs[sum_col] = np.ravel(matrix.sum(axis=1))
    adata.obs[nnz_col] = matrix.getnnz(axis=1)

    # --- Optional per-prefix / per-feature-set percentages ---
    if prefix_vars or feature_set_vars:
        names = (adata.var[feature_name_column].values
                 if feature_name_column is not None else adata.var_names)
        masks = _resolve_feature_masks(names, prefix_vars, feature_set_vars)
        total = np.ravel(matrix.sum(axis=1)).astype(float)
        denom = np.where(total > 0, total, 1.0)
        for key, mask in masks.items():
            sub = np.ravel(matrix[:, mask].sum(axis=1)).astype(float) if mask.any() \
                else np.zeros(matrix.shape[0])
            adata.obs[f'n_counts_{key}'] = sub
            adata.obs[f'pct_counts_{key}'] = 100.0 * sub / denom


def _cytome_feature_names(ds, modality, feature_name_column):
    """Ordered feature-name array (by matrix column index) for a cytome modality."""
    from cytome import modality_feature_table_info
    feature_table, idx_col, name_col = modality_feature_table_info(ds, modality)
    col = feature_name_column or name_col
    rows = ds._conn.execute(
        f"SELECT {col} FROM {feature_table} ORDER BY {idx_col}"
    ).fetchall()
    return np.asarray([r[0] for r in rows], dtype=object)


def _calculateCellMetrics_cytome(source, modality, measurement, batch_size, verbose,
                                 prefix_vars=None, feature_set_vars=None,
                                 feature_name_column=None):
    """Cytome streaming path — modality-aware per-cell metrics."""
    ds = _open_cytome(source) if isinstance(source, str) else source
    n_cells = ds.n_cells

    canon = _canon_modality(modality)
    if canon not in _CELL_METRIC_NAMES:
        raise ValueError(
            f"Unknown modality '{modality}'. Known: {list(_CELL_METRIC_NAMES)}."
        )
    sum_col, nnz_col = _CELL_METRIC_NAMES[canon]
    has_fragments = canon in _FRAGMENT_BACKED

    # Optional per-prefix / per-feature-set masks over the feature axis
    masks = {}
    if prefix_vars or feature_set_vars:
        names = _cytome_feature_names(ds, modality, feature_name_column)
        masks = _resolve_feature_masks(names, prefix_vars, feature_set_vars)
    key_sum = {k: np.zeros(n_cells, dtype=np.float64) for k in masks}

    # Accumulate per-cell sum and nnz from the count-matrix chunks
    cell_sum = np.zeros(n_cells, dtype=np.float64)
    cell_nnz = np.zeros(n_cells, dtype=np.int64)

    matrix_name = f"{modality}_{measurement}"
    n_chunks = 0
    for chunk, indices in ds.iter_chunks(
        modality=modality, layer=measurement, batch_size=batch_size
    ):
        sparse = scipy.sparse.issparse(chunk)
        if sparse:
            cell_sum[indices] = np.ravel(chunk.sum(axis=1))
            cell_nnz[indices] = chunk.getnnz(axis=1)
        else:
            cell_sum[indices] = chunk.sum(axis=1)
            cell_nnz[indices] = (chunk != 0).sum(axis=1)
        for k, m in masks.items():
            if m.any():
                key_sum[k][indices] = np.ravel(chunk[:, m].sum(axis=1))
        n_chunks += 1

    if verbose:
        print(f"[calculateCellMetrics] Streamed {n_chunks} chunks from {matrix_name}")
        print(f"  {sum_col}: median={np.median(cell_sum):.0f}, "
              f"mean={np.mean(cell_sum):.0f}")
        print(f"  {nnz_col}: median={np.median(cell_nnz):.0f}, "
              f"mean={np.mean(cell_nnz):.0f}")

    # Write to cytome cells table
    conn = ds._conn
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()}
    new_cols = [sum_col, nnz_col]
    if has_fragments:
        new_cols.append('n_fragments')
    for col in new_cols:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE cells ADD COLUMN {col} INTEGER")

    # Batch write the sum / nnz metrics under the modality-specific names
    conn.executemany(
        f"UPDATE cells SET {sum_col} = ?, {nnz_col} = ? WHERE cell_idx = ?",
        [(int(cell_sum[i]), int(cell_nnz[i]), i) for i in range(n_cells)]
    )

    # Total fragments per cell from fragment_chunks table (ATAC / tiles only)
    if has_fragments:
        try:
            n_with_frags = conn.execute(
                "SELECT COUNT(*) FROM cells WHERE n_fragments IS NOT NULL AND n_fragments > 0"
            ).fetchone()[0]
            if verbose and n_with_frags > 0:
                frags = [r[0] for r in conn.execute(
                    "SELECT n_fragments FROM cells WHERE n_fragments IS NOT NULL"
                ).fetchall()]
                print(f"  n_fragments (from importer): median={np.median(frags):.0f}, "
                      f"mean={np.mean(frags):.0f}")
            elif verbose:
                print("  n_fragments: not available (run importer with fragment counting)")
        except Exception:
            pass

    # FRiP (ATAC peaks only): fraction of a cell's fragments that fall in peaks.
    #   frip = n_fragments_in_peak / n_fragments
    # Numerator is the per-cell peak-matrix row sum (cell_sum, already computed);
    # denominator is the importer-provided total n_fragments. Auto-written for
    # ATAC when n_fragments is populated. Tiles are excluded on purpose — the
    # "fraction in tiles" is ~1 by construction and isn't FRiP. Matches the
    # pipeline's Stage-2 QC definition (qc_filter_stage2.py).
    if canon == 'ATAC' and has_fragments:
        try:
            nfrag_rows = conn.execute(
                "SELECT n_fragments FROM cells ORDER BY cell_idx"
            ).fetchall()
            n_frags = np.array(
                [(r[0] if r[0] is not None else 0) for r in nfrag_rows],
                dtype=np.float64,
            )
            if np.any(n_frags > 0):
                safe = np.where(n_frags > 0, n_frags, 1.0)
                frip = np.where(n_frags > 0, cell_sum / safe, 0.0).astype(np.float64)
                existing_cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()
                }
                if 'frip' not in existing_cols:
                    conn.execute("ALTER TABLE cells ADD COLUMN frip REAL")
                conn.executemany(
                    "UPDATE cells SET frip = ? WHERE cell_idx = ?",
                    [(float(frip[i]), i) for i in range(n_cells)],
                )
                if verbose:
                    print(f"  frip (fragments-in-peaks / n_fragments): "
                          f"median={np.median(frip):.3f}, mean={np.mean(frip):.3f}")
            elif verbose:
                print("  frip: skipped (n_fragments not populated by importer)")
        except Exception:
            pass

    # Per-prefix / per-feature-set counts + percentages
    if masks:
        denom = np.where(cell_sum > 0, cell_sum, 1.0)
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()}
        for key, ksum in key_sum.items():
            ncol, pcol = f"n_counts_{key}", f"pct_counts_{key}"
            if ncol not in existing_cols:
                conn.execute(f"ALTER TABLE cells ADD COLUMN {ncol} INTEGER")
            if pcol not in existing_cols:
                conn.execute(f"ALTER TABLE cells ADD COLUMN {pcol} REAL")
            pct = 100.0 * ksum / denom
            conn.executemany(
                f"UPDATE cells SET {ncol} = ?, {pcol} = ? WHERE cell_idx = ?",
                [(int(ksum[i]), float(pct[i]), i) for i in range(n_cells)]
            )
            if verbose:
                print(f"  pct_counts_{key}: median={np.median(pct):.2f}%, "
                      f"matched {int(masks[key].sum())} features")

    conn.commit()
    if verbose:
        cols = f"{sum_col}, {nnz_col}"
        if masks:
            cols += ", " + ", ".join(f"pct_counts_{k}" for k in key_sum)
        print(f"  Written to cytome cells table: {cols}")


### Per-group (cell-type) summary metrics

# Default "detected" thresholds per modality: a feature counts as detected in a
# group when it is expressed in > this fraction of that group's cells. RNA genes
# are denser than ATAC peaks, so the peak threshold is lower.
_GROUP_DETECTION_DEFAULTS = {"RNA": 0.10, "GA": 0.10, "ATAC": 0.05, "tiles": 0.05}


def _present_modalities(ds):
    """Auto-detected modalities with a ``{mod}_counts`` matrix, for the default
    metric set. ``tiles`` is **excluded by default** — it is a genome-binned
    re-encoding of the same fragments as ``ATAC`` and adds little beyond the peak
    metrics. Pass ``modalities=['tiles', …]`` explicitly to include it."""
    out = []
    for mod in ("RNA", "GA", "ATAC"):
        try:
            if ds.matrix_meta(f"{mod}_counts") is not None:
                out.append(mod)
        except Exception:
            pass
    return out


def _resolve_group_order_colors(ds, groupby, labels):
    """(ordered group names, {group: hex} or None) — honors the set_categories
    store so the summary rows / bars follow the same order + colors as the plots."""
    present = [g for g in pd.unique(pd.Series(labels).dropna().astype(str))]
    order, colors = None, None
    try:
        getter = getattr(ds, "get_categories", None)
        entry = getter(groupby) if getter is not None else None
        if entry:
            order = entry.get("order")
            colors = entry.get("colors")
    except Exception:
        pass
    if order:
        ordered = [str(g) for g in order if str(g) in set(present)]
        ordered += sorted(set(present) - set(ordered), key=_natural_key)
    else:
        ordered = sorted(present, key=_natural_key)
    if isinstance(colors, dict):
        colors = {str(k): str(v) for k, v in colors.items()}
    else:
        colors = None
    return ordered, colors


def _natural_key(s):
    """Numeric-aware sort key so '2' precedes '10' for integer-like labels."""
    try:
        return (0, int(s))
    except (ValueError, TypeError):
        return (1, str(s))


def calculateGroupMetrics(
    data,
    groupby: str,
    modalities=None,
    detection_pct=None,
    expression_cutoff: float = 0.0,
    measurement: str = "counts",
    batch_size: int = 1024,
    key_added: Optional[str] = None,
    verbose: bool = True,
):
    """Per-group (e.g. per cell type) summary metrics, streamed from a cytome.

    For every group in ``groupby`` and every modality present, computes how many
    features are **detected** — expressed in more than ``detection_pct`` of that
    group's cells — plus per-cell count/feature summaries. Modalities are
    auto-detected (only those with a ``{mod}_counts`` matrix are reported).

    Columns (per group; ``{m}`` = modality):

    - ``n_cells``
    - ``{m}_n_features_detected`` — # features expressed in > ``detection_pct[m]`` of cells
    - ``{m}_counts_mean`` / ``{m}_counts_median`` — per-cell total counts/fragments
    - ``{m}_features_per_cell_median`` — median active features per cell
    - ATAC/tiles: ``{m}_n_fragments_mean`` / ``{m}_n_fragments_median`` (from
      ``cells.n_fragments``), ``{m}_frip_mean`` (if a ``frip`` column exists)

    Parameters
    ----------
    data : cytome.Dataset or str, or AnnData
        Cytome (streamed) or AnnData (single modality, in-memory).
    groupby : str
        Cell-grouping column (cells table / ``obs``).
    modalities : list of str, optional
        Restrict to these modalities. Default: all present.
    detection_pct : float or dict, optional
        Detection fraction, global float or per-modality dict. Default
        ``{'RNA':0.10,'GA':0.10,'ATAC':0.05,'tiles':0.05}``.
    expression_cutoff : float, default 0.0
        A cell "expresses" a feature when its count is ``> expression_cutoff``.
    measurement : str, default ``'counts'``
        Source matrix layer.
    batch_size : int, default 1024
        Streaming chunk size (cytome).
    key_added : str, optional
        If given (cytome), also store the result under ``ds.metadata[key_added]``.
        Default: return-only.
    verbose : bool, default True

    Returns
    -------
    pandas.DataFrame
        Rows = groups (ordered by the ``set_categories`` store when present),
        columns = metrics. ``df.attrs`` carries ``groupby``, ``colors``
        (``{group: hex}`` if the store has them), and ``detection_pct`` so
        :func:`piaso.pl.plotGroupMetrics` can re-use the cell-type colors.
    """
    if _is_cytome(data) or isinstance(data, str):
        return _calculateGroupMetrics_cytome(
            data, groupby, modalities=modalities, detection_pct=detection_pct,
            expression_cutoff=expression_cutoff, measurement=measurement,
            batch_size=batch_size, key_added=key_added, verbose=verbose,
        )
    return _calculateGroupMetrics_anndata(
        data, groupby, modality=(modalities[0] if modalities else "RNA"),
        detection_pct=detection_pct, expression_cutoff=expression_cutoff,
        verbose=verbose,
    )


def _detection_pct_for(modality, detection_pct):
    if detection_pct is None:
        return _GROUP_DETECTION_DEFAULTS.get(modality, 0.10)
    if isinstance(detection_pct, dict):
        return float(detection_pct.get(modality,
                     _GROUP_DETECTION_DEFAULTS.get(modality, 0.10)))
    return float(detection_pct)


def _calculateGroupMetrics_cytome(source, groupby, modalities, detection_pct,
                                  expression_cutoff, measurement, batch_size,
                                  key_added, verbose):
    ds = _open_cytome(source) if isinstance(source, str) else source
    n_cells = ds.n_cells

    if groupby not in ds.cells.columns:
        raise ValueError(f"groupby='{groupby}' not found in cells table.")
    labels = np.asarray([("NA" if v is None else str(v))
                         for v in ds.cells[groupby]], dtype=object)

    ordered_groups, colors = _resolve_group_order_colors(ds, groupby, labels)
    gindex = {g: i for i, g in enumerate(ordered_groups)}
    group_ids = np.array([gindex.get(l, -1) for l in labels], dtype=np.int64)
    n_groups = len(ordered_groups)
    n_per_group = np.bincount(group_ids[group_ids >= 0], minlength=n_groups)

    mods = modalities if modalities is not None else _present_modalities(ds)
    if not mods:
        raise ValueError("No counts matrices found for any modality.")

    result = pd.DataFrame(index=ordered_groups)
    result.index.name = groupby
    result["n_cells"] = n_per_group.astype(int)

    for mod in mods:
        if ds.matrix_meta(f"{mod}_{measurement}") is None:
            if verbose:
                print(f"[calculateGroupMetrics] skip {mod}: no {mod}_{measurement} matrix")
            continue
        n_feat = int(ds.matrix_meta(f"{mod}_{measurement}")["shape"][1]) \
            if isinstance(ds.matrix_meta(f"{mod}_{measurement}"), dict) \
            and "shape" in ds.matrix_meta(f"{mod}_{measurement}") else None

        # Per-(group, feature) expressing-cell counts; per-cell sum + nnz.
        expressing = None
        cell_sum = np.zeros(n_cells, dtype=np.float64)
        cell_nnz = np.zeros(n_cells, dtype=np.int64)
        for chunk, idxs in ds.iter_chunks(
            modality=mod, layer=measurement, batch_size=batch_size
        ):
            if scipy.sparse.issparse(chunk):
                mask = chunk > expression_cutoff      # sparse bool
                cell_sum[idxs] = np.ravel(chunk.sum(axis=1))
                cell_nnz[idxs] = np.ravel(mask.sum(axis=1))
                mask = mask.tocsr()
            else:
                mask = chunk > expression_cutoff
                cell_sum[idxs] = chunk.sum(axis=1)
                cell_nnz[idxs] = mask.sum(axis=1)
            if expressing is None:
                expressing = np.zeros((n_groups, mask.shape[1]), dtype=np.int64)
            gids = group_ids[idxs]
            for g in np.unique(gids):
                if g < 0:
                    continue
                rows = np.where(gids == g)[0]
                expressing[g] += np.asarray(mask[rows].sum(axis=0)).ravel()

        sum_col, nnz_col = _CELL_METRIC_NAMES.get(_canon_modality(mod),
                                                  _CELL_METRIC_NAMES["RNA"])
        pct = _detection_pct_for(mod, detection_pct)
        safe_n = np.where(n_per_group == 0, 1, n_per_group)[:, None]
        frac = expressing / safe_n
        result[f"{mod}_n_features_detected"] = (frac > pct).sum(axis=1).astype(int)

        # Per-cell summaries by group (exact mean+median from the per-cell arrays).
        df_cell = pd.DataFrame({"g": group_ids, "sum": cell_sum, "nnz": cell_nnz})
        df_cell = df_cell[df_cell["g"] >= 0]
        gb = df_cell.groupby("g")
        s_mean = gb["sum"].mean()
        s_med = gb["sum"].median()
        n_med = gb["nnz"].median()
        result[f"{mod}_counts_mean"] = [float(s_mean.get(i, 0.0)) for i in range(n_groups)]
        result[f"{mod}_counts_median"] = [float(s_med.get(i, 0.0)) for i in range(n_groups)]
        result[f"{mod}_features_per_cell_median"] = [float(n_med.get(i, 0.0)) for i in range(n_groups)]

        # Fragment / FRiP summaries for fragment-backed modalities.
        if _canon_modality(mod) in _FRAGMENT_BACKED and "n_fragments" in ds.cells.columns:
            nf = np.asarray([(0 if v is None else float(v)) for v in ds.cells["n_fragments"]])
            dff = pd.DataFrame({"g": group_ids, "nf": nf})
            dff = dff[dff["g"] >= 0]
            fm, fmed = dff.groupby("g")["nf"].mean(), dff.groupby("g")["nf"].median()
            result[f"{mod}_n_fragments_mean"] = [float(fm.get(i, 0.0)) for i in range(n_groups)]
            result[f"{mod}_n_fragments_median"] = [float(fmed.get(i, 0.0)) for i in range(n_groups)]
            if "frip" in ds.cells.columns:
                fr = np.asarray([(0.0 if v is None else float(v)) for v in ds.cells["frip"]])
                dfr = pd.DataFrame({"g": group_ids, "fr": fr})
                dfr = dfr[dfr["g"] >= 0]
                frm = dfr.groupby("g")["fr"].mean()
                result[f"{mod}_frip_mean"] = [float(frm.get(i, 0.0)) for i in range(n_groups)]

        if verbose:
            print(f"[calculateGroupMetrics] {mod}: {result[f'{mod}_n_features_detected'].sum()} "
                  f"group-detections (>{pct:.0%}), {n_groups} groups")

    result.attrs["groupby"] = groupby
    result.attrs["colors"] = colors
    result.attrs["detection_pct"] = {m: _detection_pct_for(m, detection_pct) for m in mods}

    if key_added is not None:
        try:
            ds.metadata[key_added] = {
                "columns": list(result.columns),
                "index": list(result.index),
                "data": result.to_numpy().tolist(),
                "groupby": groupby,
            }
            ds.flush()
        except Exception as e:
            if verbose:
                print(f"[calculateGroupMetrics] could not store under '{key_added}': {e}")

    return result


def _calculateGroupMetrics_anndata(adata, groupby, modality, detection_pct,
                                   expression_cutoff, verbose):
    if groupby not in adata.obs:
        raise ValueError(f"groupby='{groupby}' not found in adata.obs.")
    X = adata.X
    if not scipy.sparse.issparse(X):
        X = scipy.sparse.csr_matrix(X)
    labels = np.asarray([str(v) for v in adata.obs[groupby].values], dtype=object)
    col = adata.obs[groupby]
    if hasattr(col.dtype, "categories"):
        ordered_groups = [str(c) for c in col.cat.categories if str(c) in set(labels)]
    else:
        ordered_groups = sorted(set(labels), key=_natural_key)
    gindex = {g: i for i, g in enumerate(ordered_groups)}
    group_ids = np.array([gindex[l] for l in labels], dtype=np.int64)
    n_groups = len(ordered_groups)
    n_per_group = np.bincount(group_ids, minlength=n_groups)

    sum_col, nnz_col = _CELL_METRIC_NAMES.get(_canon_modality(modality),
                                              _CELL_METRIC_NAMES["RNA"])
    mask = X > expression_cutoff
    cell_sum = np.ravel(X.sum(axis=1))
    cell_nnz = np.ravel(mask.sum(axis=1))
    pct = _detection_pct_for(modality, detection_pct)

    expressing = np.zeros((n_groups, X.shape[1]), dtype=np.int64)
    mask = mask.tocsr()
    for g in range(n_groups):
        rows = np.where(group_ids == g)[0]
        if rows.size:
            expressing[g] = np.asarray(mask[rows].sum(axis=0)).ravel()

    safe_n = np.where(n_per_group == 0, 1, n_per_group)[:, None]
    frac = expressing / safe_n
    result = pd.DataFrame(index=ordered_groups)
    result.index.name = groupby
    result["n_cells"] = n_per_group.astype(int)
    result[f"{modality}_n_features_detected"] = (frac > pct).sum(axis=1).astype(int)
    dfc = pd.DataFrame({"g": group_ids, "sum": cell_sum, "nnz": cell_nnz})
    gb = dfc.groupby("g")
    result[f"{modality}_counts_mean"] = [float(gb["sum"].mean().get(i, 0.0)) for i in range(n_groups)]
    result[f"{modality}_counts_median"] = [float(gb["sum"].median().get(i, 0.0)) for i in range(n_groups)]
    result[f"{modality}_features_per_cell_median"] = [float(gb["nnz"].median().get(i, 0.0)) for i in range(n_groups)]
    result.attrs["groupby"] = groupby
    result.attrs["colors"] = None
    result.attrs["detection_pct"] = {modality: pct}
    if verbose:
        print(f"[calculateGroupMetrics] {modality}: {n_groups} groups (AnnData)")
    return result


### Calculate feature metrics


def calculatePeakMetrics(*args, **kwargs):
    """REMOVED — renamed to :func:`calculateFeatureMetrics`.

    The function now supports all modalities (RNA / GA / ATAC / tiles), so the
    ATAC-specific "peak" name no longer fits. Update your call:

        piaso.pp.calculatePeakMetrics(ds, ...)  →  piaso.pp.calculateFeatureMetrics(ds, ...)
    """
    raise TypeError(
        "calculatePeakMetrics has been renamed to calculateFeatureMetrics "
        "(it now writes per-feature `n_cells` to the modality's var table for "
        "RNA / GA / ATAC / tiles, not just peaks). Replace "
        "`piaso.pp.calculatePeakMetrics(...)` with "
        "`piaso.pp.calculateFeatureMetrics(...)`."
    )


def calculateFeatureMetrics(source, layer: str = None,
                            modality: str = "ATAC", measurement: str = "counts",
                            batch_size: int = 1024, verbose: bool = True):
    """
    Calculate the number of cells in which each feature is active (non-zero counts).

    For AnnData, stores result in ``adata.var['n_cells']``.
    For cytome, stores ``n_cells`` in the modality's var entity table
    (RNA → ``genes``, GA → ``GA_genes``, ATAC → ``peaks``, tiles → ``tiles``),
    resolved via the cytome modality registry.

    Parameters
    ----------
    source : AnnData, cytome.Dataset, or str
        Input data. For cytome, streams the ``{modality}_{measurement}`` matrix
        in chunks.
    layer : str, optional
        AnnData layer to use. Ignored for cytome.
    modality : str
        Cytome modality: ``'ATAC'`` (default), ``'RNA'``, ``'GA'``, ``'tiles'``.
        Ignored for AnnData.
    measurement : str
        Cytome measurement (default 'counts'). Ignored for AnnData.
    batch_size : int
        Chunk size for streaming. Default 1024.
    verbose : bool
        Print progress messages.
    """
    if isinstance(source, str) or _is_cytome(source):
        _calculateFeatureMetrics_cytome(source, modality, measurement, batch_size, verbose)
    else:
        _calculateFeatureMetrics_anndata(source, layer)


def _calculateFeatureMetrics_anndata(adata, layer=None):
    """AnnData path (original implementation)."""
    if layer:
        if layer not in adata.layers.keys():
            raise ValueError(f"Specified layer '{layer}' not found in the provided AnnData object.")
        non_zero_counts = np.sum(adata.layers[layer] > 0, axis=0)
    else:
        non_zero_counts = np.sum(adata.X > 0, axis=0)

    adata.var['n_cells'] = np.squeeze(np.asarray(non_zero_counts))


def _calculateFeatureMetrics_cytome(source, modality, measurement, batch_size, verbose):
    """Cytome streaming path — one pass accumulating col-wise nnz, written to
    the modality's var entity table."""
    from cytome import modality_feature_table_info

    ds = _open_cytome(source) if isinstance(source, str) else source

    # Resolve the var entity table + its index column for this modality
    feature_table, idx_col, _name_col = modality_feature_table_info(ds, modality)

    # Get n_features from matrix metadata
    matrix_name = f"{modality}_{measurement}"
    meta = ds._conn.execute(
        "SELECT n_cols FROM matrix_meta WHERE matrix_name = ?", (matrix_name,)
    ).fetchone()
    if meta is None:
        raise ValueError(f"Matrix '{matrix_name}' not found in cytome dataset.")
    n_features = int(meta[0])
    col_nnz = np.zeros(n_features, dtype=np.int64)

    n_chunks = 0
    for chunk, indices in ds.iter_chunks(
        modality=modality, layer=measurement, batch_size=batch_size
    ):
        if scipy.sparse.issparse(chunk):
            col_nnz += np.ravel((chunk != 0).sum(axis=0))
        else:
            col_nnz += (chunk != 0).sum(axis=0)
        n_chunks += 1

    if verbose:
        print(f"[calculateFeatureMetrics] Streamed {n_chunks} chunks, "
              f"{n_features} features in {feature_table}")
        print(f"  n_cells per feature: median={np.median(col_nnz):.0f}, "
              f"mean={np.mean(col_nnz):.1f}, max={np.max(col_nnz)}")

    # Write to the modality's var entity table
    conn = ds._conn
    existing_cols = {r[1] for r in conn.execute(
        f"PRAGMA table_info({feature_table})"
    ).fetchall()}
    if 'n_cells' not in existing_cols:
        conn.execute(f"ALTER TABLE {feature_table} ADD COLUMN n_cells INTEGER")

    # Batch write keyed on the modality's index column
    conn.executemany(
        f"UPDATE {feature_table} SET n_cells = ? WHERE {idx_col} = ?",
        [(int(col_nnz[i]), i) for i in range(n_features)]
    )
    conn.commit()
    if verbose:
        print(f"  Written to cytome {feature_table} table: n_cells")

### Calculate TSS enrichment score
import os
import warnings
import pandas as pd
def calculateTSSEnrichmentScore(
            fragment_file: str=None,
            adata=None,
            output_dir: str=None,
            barcodes: list=None,
            barcode_file: str=None,
            bedtools_path: str=None,
            tss_bed_file: str=None,
            genome_size_file: str=None,
            slop_l:int=50,
            slop_r:int=50,
            shift:int=1550,
            prefix:str='tss',
            chromosome_selected: list=[],
            method: str='python'):

    """
    Compute the TSS enrichment score for cells based on given parameters and files.

    Parameters:
    :param fragment_file (str): Path to the fragment file.
    :param adata: AnnData object. If provided, TSS_score is written to adata.obs['TSS_score'].
    :param output_dir (str): Directory to save intermediate and output files.
    :param barcodes: list, optional list of barcodes to include. If None and adata is provided, extracted from adata.obs_names.
    :param barcode_file: str, optional path to a file with one barcode per line.
    :param bedtools_path: str, the path to the bedtools binaries. Required if method='bedtools'.
    :param tss_bed_file: str, the path to the input TSS bed file.
    :param genome_size_file: str, the path to the genome size file.
    :param slop_l: int, the amount to slop (extend) the features to the left.
    :param slop_r: int, the amount to slop (extend) the features to the right.
    :param shift: int, the amount to shift the features to the left and right.
    :param prefix: str, a prefix for naming output files. Default is "tss".
    :param chromosome_selected: list, list of chromosomes to keep. Default is empty, which keeps all.
    :param method: str, 'python' (default, no bedtools) or 'bedtools'.

    Returns:
    - pandas.DataFrame: A dataframe with the 'CellBarcode' and 'TSS_score' columns.
    """

    warnings.warn(
        "calculateTSSEnrichmentScore is deprecated. TSS enrichment is now "
        "computed during fragment import by the Rust importer "
        "(`cytome-import-fragments --tss-bed ...`), which writes tss_score / "
        "tss_core_count / tss_flank_count straight into the cytome `cells` "
        "table -- identical results (r=0.9999) at ~2700x the speed. This "
        "function will be removed in a future version.",
        DeprecationWarning, stacklevel=2,
    )
    # Imported lazily: the peak-file and quantification helpers are not part of
    # the public package, so importing them at module level would make the
    # whole preprocessing package unimportable in a public install.
    from ._processPeakFile import processTSSbed, processTSSbed_python
    from ._quantifyPeakActivity import quantifyPeakActivity

    # Process TSS bed
    tss_output_dir = os.path.join(output_dir, 'tss')

    if method == 'python':
        processTSSbed_python(
            tss_bed_file=tss_bed_file,
            genome_size_file=genome_size_file,
            slop_l=slop_l,
            slop_r=slop_r,
            shift=shift,
            prefix=prefix,
            output_dir=tss_output_dir,
            chromosome_selected=chromosome_selected,
        )
    else:
        processTSSbed(
            bedtools_path=bedtools_path,
            tss_bed_file=tss_bed_file,
            genome_size_file=genome_size_file,
            slop_l=slop_l,
            slop_r=slop_r,
            shift=shift,
            prefix=prefix,
            output_dir=tss_output_dir,
            chromosome_selected=chromosome_selected,
        )

    # Auto-extract barcodes from adata if not provided
    if barcodes is None and barcode_file is None and adata is not None:
        barcodes = list(adata.obs_names)

    # Quantify peak activity
    tss_prefix = prefix

    peak_file = os.path.join(output_dir, 'tss', f'{tss_prefix}_slop_l{slop_l}_r{slop_r}.bed')
    adata_tss = quantifyPeakActivity(fragment_file, peak_file, barcodes=barcodes, barcode_file=barcode_file, bedtools_path=bedtools_path)

    peak_file_ls = os.path.join(output_dir, 'tss', f'{tss_prefix}_slop_l{slop_l}_r{slop_r}_leftshift{shift}.bed')
    adata_tss_ls = quantifyPeakActivity(fragment_file, peak_file_ls, barcodes=barcodes, barcode_file=barcode_file, bedtools_path=bedtools_path)

    peak_file_rs = os.path.join(output_dir, 'tss', f'{tss_prefix}_slop_l{slop_l}_r{slop_r}_rightshift{shift}.bed')
    adata_tss_rs = quantifyPeakActivity(fragment_file, peak_file_rs, barcodes=barcodes, barcode_file=barcode_file, bedtools_path=bedtools_path)

    # Get common cells
    common_cells = set(adata_tss.obs_names)
    common_cells.intersection_update(adata_tss_ls.obs_names, adata_tss_rs.obs_names)
    common_cells = np.array(list(common_cells))

    adata_tss = adata_tss[common_cells]
    adata_tss_ls = adata_tss_ls[common_cells]
    adata_tss_rs = adata_tss_rs[common_cells]

    # Compute TSS score (guard against zero flanking counts)
    tss_center = np.ravel(adata_tss.X.sum(axis=1))
    tss_flank = np.ravel((adata_tss_ls.X.sum(axis=1) + adata_tss_rs.X.sum(axis=1)) / 2)
    tss_score = np.divide(tss_center, tss_flank, out=np.zeros_like(tss_center, dtype=float), where=tss_flank > 0)

    tss_score_df = pd.DataFrame({
        'CellBarcode': common_cells,
        'TSS_score': tss_score
    })
    tss_score_output=os.path.join(output_dir, 'tss', f'{tss_prefix}_score.csv')
    tss_score_df.to_csv(tss_score_output, index=None)
    print("Finished. TSS enrichment scores saved in: ", tss_score_output)

    # Write to adata.obs if provided
    if adata is not None:
        barcode_to_score = dict(zip(tss_score_df['CellBarcode'], tss_score_df['TSS_score']))
        adata.obs['TSS_score'] = adata.obs_names.map(lambda x: barcode_to_score.get(x, np.nan))
        print("TSS_score written to adata.obs['TSS_score']")

    return tss_score_df
