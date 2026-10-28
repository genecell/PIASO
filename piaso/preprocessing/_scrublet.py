"""Doublet detection matching the Scrublet algorithm (Wolock et al., 2019).

Reproduces the exact scrublet pipeline:
  1. Total-count normalize (to mean of total counts)
  2. Gene filtering via v-score (Poisson-noise model + percentile)
  3. Simulate doublets (random pairs, sum raw counts)
  4. Re-normalize (to 1e6 = TPM)
  5. Z-score (mean center + variance normalize, computed on observed only)
  6. PCA via sklearn.decomposition.PCA (or IncrementalPCA for streaming)
  7. KNN with adjusted k: k_adj = k * (1 + n_sim/n_obs), each point's own
     entry left out, as scrublet's annoy and scikit-learn searches do
  8. Bayesian doublet scoring
  9. Auto-threshold via skimage or histogram bimodality

A cell that is not scored (its library has fewer than 50 cells, or fewer
genes pass the variability filter than ``n_components``) gets NaN and a
missing doublet call, never 0 / False, and one warning names the libraries.

Supports both AnnData (in-memory) and cytome Dataset (streaming) inputs.
Cytome path uses 4 streaming passes with O(batch_size × n_genes) peak RAM,
enabling parallel per-library execution with bounded memory.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
import scipy.optimize
from typing import Optional


# ============================================================================
# Gene selection — exact scrublet v-score algorithm
# ============================================================================

def _running_quantile(x, y, p, n_bins):
    """Running quantile (percentile) over sorted x bins."""
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]
    dx = (x[-1] - x[0]) / n_bins
    x_out = np.linspace(x[0] + dx / 2, x[-1] - dx / 2, n_bins)
    y_out = np.zeros(x_out.shape)
    for i in range(len(x_out)):
        mask = (x >= x_out[i] - dx / 2) & (x < x_out[i] + dx / 2)
        if mask.any():
            y_out[i] = np.percentile(y[mask], p)
        else:
            y_out[i] = y_out[i - 1] if i > 0 else np.nan
    return x_out, y_out


def _fit_vscores(mu_gene, FF_gene, n_bins=50, fit_percentile=0.1, error_wt=1):
    """Fit Poisson-noise model and compute v-scores from gene-level stats.

    Shared by both in-memory and streaming paths.

    Parameters
    ----------
    mu_gene : array (n_genes,)
        Gene means (on normalized matrix, already filtered to mean > 0).
    FF_gene : array (n_genes,)
        Fano factors (var / mean).

    Returns
    -------
    v_scores : array
    a, b : float
        Fitted Poisson model parameters.
    """
    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = _running_quantile(data_x, data_y, fit_percentile, n_bins)
    valid = ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    g_log = lambda params: np.log(params[1] * np.exp(-params[0]) + params[2])
    h, b_edges = np.histogram(np.log(FF_gene[mu_gene > 0]), bins=200)
    b_centers = b_edges[:-1] + np.diff(b_edges) / 2
    max_ix = np.argmax(h)
    c = max(np.exp(b_centers[max_ix]), 1)
    err_fun = lambda b2: np.sum(abs(g_log([x, c, b2]) - y) ** error_wt)
    b_opt = scipy.optimize.fmin(func=err_fun, x0=[0.1], disp=False)
    a = c / (1 + b_opt[0]) - 1

    v_scores = FF_gene / ((1 + a) * (1 + b_opt[0]) + b_opt[0] * mu_gene)
    return v_scores, a, b_opt[0]


def _get_vscores(E, min_mean=0, n_bins=50, fit_percentile=0.1, error_wt=1):
    """Calculate v-scores from a full normalized matrix (in-memory path)."""
    mu_gene = np.array(E.mean(axis=0)).ravel()
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:, gene_ix].copy()
    tmp.data **= 2
    var_gene = np.array(tmp.mean(axis=0)).ravel() - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    v_scores, a, b = _fit_vscores(mu_gene, FF_gene, n_bins, fit_percentile, error_wt)
    return v_scores, gene_ix, mu_gene, FF_gene, a, b


def _filter_genes(E, min_counts=3, min_cells=3, min_vscore_pctl=85):
    """Filter genes by expression level and v-score percentile (in-memory)."""
    v_scores, gene_ix, mu_gene, FF_gene, a, b = _get_vscores(E)
    ix2 = v_scores > 0
    v_scores = v_scores[ix2]
    gene_ix = gene_ix[ix2]

    min_vscore = np.percentile(v_scores, min_vscore_pctl)

    counts_check = np.array((E[:, gene_ix] >= min_counts).sum(axis=0)).ravel()
    ix = (counts_check >= min_cells) & (v_scores >= min_vscore)

    return gene_ix[ix]


def _filter_genes_streaming(mu_gene, var_gene, gene_cell_counts,
                             min_counts_cells_threshold, min_vscore_pctl):
    """Filter genes from streaming-accumulated stats.

    Parameters
    ----------
    mu_gene : array (n_genes,)
        Gene means on mean-normalized matrix.
    var_gene : array (n_genes,)
        Gene variances on mean-normalized matrix.
    gene_cell_counts : array (n_genes,)
        Number of cells with raw count >= min_counts per gene.
    min_counts_cells_threshold : int
        Minimum cells with >= min_counts expression.
    min_vscore_pctl : float
        V-score percentile threshold.
    """
    valid = mu_gene > 0
    gene_ix = np.nonzero(valid)[0]
    mu = mu_gene[gene_ix]
    var = np.maximum(var_gene[gene_ix], 0)
    FF = var / mu

    v_scores, a, b = _fit_vscores(mu, FF)

    ix2 = v_scores > 0
    v_scores = v_scores[ix2]
    gene_ix = gene_ix[ix2]

    min_vscore = np.percentile(v_scores, min_vscore_pctl)

    counts_check = gene_cell_counts[gene_ix]
    ix = (counts_check >= min_counts_cells_threshold) & (v_scores >= min_vscore)

    return gene_ix[ix]


# ============================================================================
# Total counts normalization — exact scrublet formula
# ============================================================================

def _tot_counts_norm(E, total_counts=None, target_total=None):
    """Total counts normalize sparse matrix (in-memory path)."""
    E = E.tocsc()
    n_cells = E.shape[0]
    if total_counts is None:
        total_counts = np.array(E.sum(axis=1)).ravel()
    if target_total is None:
        target_total = np.mean(total_counts)

    w = sparse.lil_matrix((n_cells, n_cells))
    safe_tc = np.where(total_counts > 0, total_counts, 1e-10)
    w.setdiag(float(target_total) / safe_tc)
    return (w * E).tocsc()


def _normalize_zscore_chunk(chunk, total_counts_chunk, target_total,
                             gene_mean, gene_std):
    """Normalize chunk to target_total, then z-score. Returns dense float64."""
    scale = target_total / np.maximum(total_counts_chunk, 1e-10)
    if sparse.issparse(chunk):
        chunk_dense = chunk.multiply(scale[:, np.newaxis]).toarray()
    else:
        chunk_dense = np.asarray(chunk) * scale[:, np.newaxis]
    chunk_dense = (chunk_dense.astype(np.float64) - gene_mean) / gene_std
    return chunk_dense


# ============================================================================
# Main entry point
# ============================================================================

def scrublet(
    data,
    library_key: Optional[str] = None,
    n_components: int = 30,
    sim_doublet_ratio: float = 2.0,
    expected_doublet_rate: float = 0.06,
    n_neighbors: int = None,
    min_counts: int = 3,
    min_cells: int = 3,
    min_gene_variability_pctl: float = 85,
    random_state: int = 0,
    threshold: Optional[float] = None,
    batch_size: int = 1024,
    verbose: bool = True,
    layer: Optional[str] = None,
    modality: str = "RNA",
):
    """Detect doublets using the Scrublet algorithm (Wolock et al., 2019).

    Processes each library independently. Supports both AnnData and
    cytome Dataset inputs. The cytome path is fully streaming with
    O(batch_size * n_genes) peak RAM per library.

    Parameters
    ----------
    data : AnnData or cytome.Dataset or str
        Input data. For AnnData, uses raw counts from ``X`` or ``raw``
        (or ``layers[layer]``). For cytome, streams from the RNA counts.
    library_key : str, optional
        Column in ``obs``/``cells`` identifying libraries; each library is
        scored on its own. ``None`` scores all cells as one library. A key
        that is not a column, or a column that is missing (NULL / NaN / empty)
        for any cell, raises: those cells would otherwise go unscored, or be
        pooled, without a word -- which is what happened on the pipeline's
        stores, whose ``sample_id`` is empty.
    n_components : int
        Number of PCA components for the manifold.
    sim_doublet_ratio : float
        Ratio of simulated doublets to observed cells.
    expected_doublet_rate : float
        Prior expected doublet rate (for Bayesian scoring).
    n_neighbors : int, optional
        Number of neighbors for KNN. Defaults to
        ``round(0.5 * sqrt(n_cells))``.
    min_counts : int
        Min counts per gene for gene filtering.
    min_cells : int
        Min cells expressing gene for gene filtering.
    min_gene_variability_pctl : float
        V-score percentile threshold for gene filtering (default 85).
    random_state : int
        Random seed for reproducibility.
    threshold : float, optional
        Manual doublet score threshold. If None, auto-detected.
    batch_size : int
        Streaming batch size for cytome path (default 1024).
    verbose : bool
        Print progress messages.
    layer : str, optional
        The raw counts: ``adata.layers[layer]`` on AnnData,
        ``{modality}_{layer}`` on a cytome. ``None`` takes ``.X`` (or
        ``.raw``) on AnnData and ``{modality}_raw_counts`` or
        ``{modality}_counts`` on a cytome, and refuses values that are not
        integers: doublets are simulated by adding two cells' raw counts.
    modality : str, default 'RNA'
        Cytome only: the modality to read.

    Returns
    -------
    None
        Adds ``scrublet_score`` and ``is_doublet`` to obs/cells. Cells that
        are not scored get NaN and a missing call (a nullable boolean on
        AnnData, NULL on a cytome).
    """
    is_cytome = _is_cytome_input(data)

    if is_cytome:
        _scrublet_cytome(
            data, library_key=library_key, n_components=n_components,
            sim_doublet_ratio=sim_doublet_ratio,
            expected_doublet_rate=expected_doublet_rate,
            n_neighbors=n_neighbors, min_counts=min_counts,
            min_cells=min_cells,
            min_gene_variability_pctl=min_gene_variability_pctl,
            random_state=random_state, threshold=threshold,
            batch_size=batch_size, verbose=verbose, layer=layer,
            modality=modality,
        )
    else:
        _scrublet_anndata(
            data, library_key=library_key, n_components=n_components,
            sim_doublet_ratio=sim_doublet_ratio,
            expected_doublet_rate=expected_doublet_rate,
            n_neighbors=n_neighbors, min_counts=min_counts,
            min_cells=min_cells,
            min_gene_variability_pctl=min_gene_variability_pctl,
            random_state=random_state, threshold=threshold,
            verbose=verbose, layer=layer,
        )


# ============================================================================
# Libraries and unscored cells -- one rule for both paths
# ============================================================================

#: A library smaller than this is not scored (its cells get NaN).
_MIN_CELLS_PER_LIBRARY = 50


def _library_groups(labels, library_key, where):
    """``[(name, cell_indices), ...]`` for each library, in sorted order.

    ``labels`` is the library column (None when ``library_key`` is None).
    A missing value (None / NaN / an empty string) is refused: SQL's
    ``WHERE col = NULL`` matched no cell, so every cell of a store with an
    empty sample column was left at score 0 and "not a doublet", and on
    AnnData a NaN label pooled or dropped its cells.
    """
    import pandas as pd
    if library_key is None:
        return None
    lab = pd.Series(np.asarray(labels, dtype=object))
    missing = lab.isna().to_numpy() | (lab.astype(str).str.strip().to_numpy() == "")
    if missing.any():
        raise ValueError(
            f"scrublet: library_key={library_key!r} is missing (NULL, NaN or empty) for "
            f"{int(missing.sum())} of {len(lab)} cells of {where}. Those cells would not be scored. "
            f"Fill the column (one library per value), or pass library_key=None to score all the "
            f"cells as one library.")
    codes, names = pd.factorize(lab.astype(str), sort=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(names) + 1))
    return [(str(names[i]), np.sort(order[bounds[i]:bounds[i + 1]]).astype(np.int64))
            for i in range(len(names))]


def _warn_unscored(skipped, n_total):
    """One warning naming the libraries left unscored, and why."""
    if not skipped:
        return
    import warnings
    n = sum(c for _, c, _ in skipped)
    head = "; ".join(f"{name} ({c} cells: {why})" for name, c, why in skipped[:5])
    warnings.warn(
        f"scrublet: {n} of {n_total} cells were not scored and have NaN scores and a missing "
        f"doublet call: {head}{' ...' if len(skipped) > 5 else ''}.", UserWarning, stacklevel=3)


def _too_few_genes(n_genes_filtered, n_components):
    """The PCA needs more genes than components (scikit-learn's arpack solver
    refuses ``n_components >= n_features``), so a small panel cannot be scored."""
    return n_genes_filtered <= n_components


# ============================================================================
# AnnData path (in-memory, exact scrublet match)
# ============================================================================

def _scrublet_anndata(
    data, library_key, n_components, sim_doublet_ratio,
    expected_doublet_rate, n_neighbors, min_counts, min_cells,
    min_gene_variability_pctl, random_state, threshold, verbose,
    layer=None,
):
    """In-memory scrublet for AnnData objects."""
    import pandas as pd
    counts_full, obs_df = _load_anndata_counts(data, layer=layer)

    n_total = counts_full.shape[0]
    scores = np.full(n_total, np.nan, dtype=np.float64)
    predicted = np.zeros(n_total, dtype=bool)
    scored = np.zeros(n_total, dtype=bool)

    if library_key is not None and library_key not in obs_df.columns:
        raise KeyError(
            f"scrublet: library_key={library_key!r} is not a column of adata.obs "
            f"(columns: {list(obs_df.columns)}).")
    groups = _library_groups(obs_df[library_key].to_numpy() if library_key is not None else None,
                             library_key, "adata.obs")
    if groups is None:
        groups = [("all", np.arange(n_total, dtype=np.int64))]

    skipped = []
    for lib_i, (lib_name, cell_indices) in enumerate(groups):
        n_cells = len(cell_indices)

        if verbose:
            print(f"  Library {lib_i+1}/{len(groups)}: "
                  f"{lib_name} ({n_cells} cells)")

        if n_cells < _MIN_CELLS_PER_LIBRARY:
            if verbose:
                print(f"    Skipping — too few cells ({n_cells} < {_MIN_CELLS_PER_LIBRARY})")
            skipped.append((lib_name, n_cells, f"fewer than {_MIN_CELLS_PER_LIBRARY} cells"))
            continue

        out = _scrublet_library_inmemory(
            counts_full[cell_indices], n_components, sim_doublet_ratio,
            expected_doublet_rate, n_neighbors, min_counts, min_cells,
            min_gene_variability_pctl, random_state, threshold, verbose,
        )
        if out is None:
            skipped.append((lib_name, n_cells, f"no more genes pass the filter than n_components={n_components}"))
            continue
        lib_scores, lib_predicted = out

        scores[cell_indices] = lib_scores
        predicted[cell_indices] = lib_predicted
        scored[cell_indices] = True

    data.obs['scrublet_score'] = scores
    if scored.all():
        data.obs['is_doublet'] = predicted
    else:
        # a missing call, not "not a doublet", for a cell nobody scored
        calls = pd.array(predicted, dtype="boolean")
        calls[~scored] = pd.NA
        data.obs['is_doublet'] = calls
    _warn_unscored(skipped, n_total)

    if verbose:
        total_doublets = int(predicted[scored].sum())
        print(f"  Total doublets: {total_doublets}/{int(scored.sum())} scored cells "
              f"({100*total_doublets/max(int(scored.sum()), 1):.1f}%)")


def _scrublet_library_inmemory(
    X_obs_raw, n_components, sim_doublet_ratio, expected_doublet_rate,
    n_neighbors, min_counts, min_cells, min_gene_variability_pctl,
    random_state, threshold, verbose,
):
    """Run scrublet on one library, fully in memory."""
    n_cells = X_obs_raw.shape[0]
    total_counts_obs = np.array(X_obs_raw.sum(axis=1)).ravel()

    # 1. Normalize observed (to mean of total counts)
    X_obs_norm = _tot_counts_norm(X_obs_raw, total_counts=total_counts_obs)

    # 2. Gene filtering via v-score
    gene_filter = _filter_genes(
        X_obs_norm, min_counts=min_counts, min_cells=min_cells,
        min_vscore_pctl=min_gene_variability_pctl,
    )
    if _too_few_genes(len(gene_filter), n_components):
        if verbose:
            print(f"    Warning: only {len(gene_filter)} genes selected "
                  f"(<= n_components={n_components}), skipping library")
        return None
    if verbose:
        print(f"    Selected {len(gene_filter)} genes "
              f"(v-score pctl={min_gene_variability_pctl})")

    X_obs = X_obs_raw[:, gene_filter]

    # 3. Simulate doublets
    rng = np.random.RandomState(random_state)
    n_sim = int(n_cells * sim_doublet_ratio)
    pair_ix = rng.randint(0, n_cells, size=(n_sim, 2))
    X_sim = X_obs[pair_ix[:, 0]] + X_obs[pair_ix[:, 1]]
    total_counts_sim = total_counts_obs[pair_ix[:, 0]] + total_counts_obs[pair_ix[:, 1]]

    if verbose:
        print(f"    Simulated {n_sim} doublets")

    # 4. Re-normalize to TPM
    X_obs_norm = _tot_counts_norm(X_obs, total_counts=total_counts_obs, target_total=1e6)
    X_sim_norm = _tot_counts_norm(X_sim, total_counts=total_counts_sim, target_total=1e6)

    # 5. Z-score (observed stats, apply to both)
    gene_mean = np.array(X_obs_norm.mean(axis=0)).ravel()
    tmp = X_obs_norm.copy()
    tmp.data **= 2
    gene_var = np.array(tmp.mean(axis=0)).ravel() - gene_mean ** 2
    gene_std = np.sqrt(np.maximum(gene_var, 0))
    gene_std[gene_std == 0] = 1.0
    del tmp

    X_obs_z = _sparse_zscore(X_obs_norm, gene_mean, gene_std)
    X_sim_z = _sparse_zscore(X_sim_norm, gene_mean, gene_std)
    del X_obs_norm, X_sim_norm

    # 6. PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=random_state,
              svd_solver='arpack')
    manifold_obs = pca.fit_transform(X_obs_z)
    manifold_sim = pca.transform(X_sim_z)
    del X_obs_z, X_sim_z

    if verbose:
        mem_mb = (manifold_obs.nbytes + manifold_sim.nbytes) / 1e6
        print(f"    PCA done: obs {manifold_obs.shape}, "
              f"sim {manifold_sim.shape}, {mem_mb:.1f} MB")

    # 7-9. KNN + scoring + threshold
    return _knn_score_threshold(
        manifold_obs, manifold_sim, n_cells, n_sim,
        n_neighbors, expected_doublet_rate, random_state,
        threshold, verbose,
    )


# ============================================================================
# Cytome path (streaming, bounded RAM)
# ============================================================================

def _scrublet_cytome(
    data, library_key, n_components, sim_doublet_ratio,
    expected_doublet_rate, n_neighbors, min_counts, min_cells,
    min_gene_variability_pctl, random_state, threshold, batch_size, verbose,
    layer=None, modality="RNA",
):
    """Streaming scrublet for cytome Dataset objects.

    Uses 4 streaming passes per library with O(batch_size * n_genes)
    peak RAM, independent of total cell count.
    """
    ds, ml = _resolve_cytome_layer(data, layer=layer, modality=modality)
    n_total = ml.shape[0]

    scores = np.full(n_total, np.nan, dtype=np.float64)
    predicted = np.zeros(n_total, dtype=bool)
    scored = np.zeros(n_total, dtype=bool)

    # Determine libraries from cells table: one read of the column, the same
    # rule as AnnData (a missing value raises; SQL's `= NULL` matched nothing)
    conn = ds._conn
    cells_cols = [r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()]
    groups = None
    if library_key is not None:
        if library_key not in cells_cols:
            raise KeyError(
                f"scrublet: library_key={library_key!r} is not a column of the cytome's cells "
                f"table (columns: {cells_cols}).")
        rows = conn.execute(f"SELECT cell_idx, [{library_key}] FROM cells ORDER BY cell_idx").fetchall()
        idx = np.array([r[0] for r in rows], dtype=np.int64)
        groups = _library_groups([r[1] for r in rows], library_key, "the cytome's cells table")
        groups = [(name, idx[pos]) for name, pos in groups]
    if groups is None:
        groups = [("all", np.arange(n_total, dtype=np.int64))]

    skipped = []
    for lib_i, (lib_name, lib_cell_indices) in enumerate(groups):
        n_cells = len(lib_cell_indices)

        if verbose:
            print(f"  Library {lib_i+1}/{len(groups)}: "
                  f"{lib_name} ({n_cells} cells) [streaming]")

        if n_cells < _MIN_CELLS_PER_LIBRARY:
            if verbose:
                print(f"    Skipping — too few cells ({n_cells} < {_MIN_CELLS_PER_LIBRARY})")
            skipped.append((lib_name, n_cells, f"fewer than {_MIN_CELLS_PER_LIBRARY} cells"))
            continue

        out = _scrublet_library_streaming(
            ml, lib_cell_indices, n_components, sim_doublet_ratio,
            expected_doublet_rate, n_neighbors, min_counts, min_cells,
            min_gene_variability_pctl, random_state, threshold,
            batch_size, verbose,
        )
        if out is None:
            skipped.append((lib_name, n_cells, f"no more genes pass the filter than n_components={n_components}"))
            continue
        lib_scores, lib_predicted = out

        scores[lib_cell_indices] = lib_scores
        predicted[lib_cell_indices] = lib_predicted
        scored[lib_cell_indices] = True

    _store_cytome_results(ds, scores, predicted, scored)
    _warn_unscored(skipped, n_total)

    if verbose:
        total_doublets = int(predicted[scored].sum())
        print(f"  Total doublets: {total_doublets}/{int(scored.sum())} scored cells "
              f"({100*total_doublets/max(int(scored.sum()), 1):.1f}%)")


def _scrublet_library_streaming(
    ml, lib_cell_indices, n_components, sim_doublet_ratio,
    expected_doublet_rate, n_neighbors, min_counts, min_cells,
    min_gene_variability_pctl, random_state, threshold,
    batch_size, verbose,
):
    """Streaming scrublet for one library from cytome.

    5 passes through the data:
      Pass 1a: total_counts + gene stats (normalize-to-1 trick)
      Pass 1b: normalized gene cell counts (for exact min_counts filter)
      Pass 2:  IncrementalPCA fit
      Pass 3:  PCA transform observed → manifold_obs
      Pass 4:  Doublet simulation (random access) → manifold_sim

    Peak RAM: O(batch_size * n_genes_filtered) + O(n_cells * n_components)
    """
    n_cells = len(lib_cell_indices)
    n_genes = ml.shape[1]

    # Sort library indices for sequential chunk access
    sort_order = np.argsort(lib_cell_indices)
    sorted_globals = lib_cell_indices[sort_order]

    # === PASS 1a: Total counts + gene stats (single pass) ===
    # Accumulate normalize-to-1 gene stats:
    #   gene_norm1_sum[g] = Σ_i (x_ig / tc_i)
    #   gene_norm1_sq_sum[g] = Σ_i (x_ig / tc_i)²
    # Then: mu_g(T) = T * norm1_sum / n, var_g(T) = T² * (sq/n - (s/n)²)
    gene_norm1_sum = np.zeros(n_genes, dtype=np.float64)
    gene_norm1_sq_sum = np.zeros(n_genes, dtype=np.float64)
    total_counts_sorted = np.zeros(n_cells, dtype=np.float64)

    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        batch_globals = sorted_globals[i:end]
        chunk = ml.rows(batch_globals)

        tc = np.array(chunk.sum(axis=1)).ravel().astype(np.float64)
        total_counts_sorted[i:end] = tc

        # Normalize each cell to 1/tc (sparse row scaling)
        inv_tc = 1.0 / np.maximum(tc, 1e-10)
        chunk_scaled = chunk.multiply(inv_tc[:, np.newaxis]).tocsr()
        gene_norm1_sum += np.array(chunk_scaled.sum(axis=0)).ravel()

        tmp = chunk_scaled.copy()
        tmp.data **= 2
        gene_norm1_sq_sum += np.array(tmp.sum(axis=0)).ravel()
        del tmp, chunk_scaled

    # Unsort total_counts to original library order
    total_counts = np.empty(n_cells, dtype=np.float64)
    total_counts[sort_order] = total_counts_sorted

    # Derive normalized gene stats
    target = np.mean(total_counts)
    mu_gene = target * gene_norm1_sum / n_cells
    var_gene = target ** 2 * (
        gene_norm1_sq_sum / n_cells - (gene_norm1_sum / n_cells) ** 2
    )

    # === PASS 1b: Normalized gene cell counts ===
    # Must use normalized counts (raw * target / tc_i) for the min_counts
    # filter to match scrublet exactly. Requires target from pass 1a.
    gene_cell_counts = np.zeros(n_genes, dtype=np.int64)

    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        batch_globals = sorted_globals[i:end]
        chunk = ml.rows(batch_globals)

        tc = total_counts_sorted[i:end]
        # Normalized count = raw * target / tc_i
        scale = target / np.maximum(tc, 1e-10)
        chunk_norm = chunk.multiply(scale[:, np.newaxis]).tocsr()
        gene_cell_counts += np.array(
            (chunk_norm >= min_counts).sum(axis=0)
        ).ravel().astype(np.int64)

    # V-score gene filtering
    gene_filter = _filter_genes_streaming(
        mu_gene, var_gene, gene_cell_counts,
        min_cells, min_gene_variability_pctl,
    )
    n_genes_filtered = len(gene_filter)

    if _too_few_genes(n_genes_filtered, n_components):
        if verbose:
            print(f"    Warning: only {n_genes_filtered} genes selected "
                  f"(<= n_components={n_components}), skipping library")
        return None

    if verbose:
        print(f"    Selected {n_genes_filtered} genes "
              f"(streaming v-score pctl={min_gene_variability_pctl})")

    # Z-score params (TPM = 1e6): derive from normalize-to-1 stats
    # TPM gene mean/var = mean_norm * scale, var_norm * scale²
    tpm_scale = 1e6 / target
    gene_mean_tpm = mu_gene[gene_filter] * tpm_scale
    gene_var_tpm = var_gene[gene_filter] * tpm_scale ** 2
    gene_std_tpm = np.sqrt(np.maximum(gene_var_tpm, 0))
    gene_std_tpm[gene_std_tpm == 0] = 1.0

    # === PASS 2: IncrementalPCA fit ===
    from sklearn.decomposition import IncrementalPCA
    pca_batch = max(batch_size, 2 * n_components)
    pca = IncrementalPCA(n_components=n_components)

    for i in range(0, n_cells, pca_batch):
        end = min(i + pca_batch, n_cells)
        if end - i < n_components:
            break  # Skip too-small final batch
        batch_globals = sorted_globals[i:end]
        chunk = ml.rows(batch_globals)[:, gene_filter]
        tc = total_counts_sorted[i:end]
        chunk_z = _normalize_zscore_chunk(chunk, tc, 1e6,
                                          gene_mean_tpm, gene_std_tpm)
        pca.partial_fit(chunk_z)

    # === PASS 3: PCA transform observed ===
    manifold_obs_sorted = np.empty((n_cells, n_components), dtype=np.float64)
    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        batch_globals = sorted_globals[i:end]
        chunk = ml.rows(batch_globals)[:, gene_filter]
        tc = total_counts_sorted[i:end]
        chunk_z = _normalize_zscore_chunk(chunk, tc, 1e6,
                                          gene_mean_tpm, gene_std_tpm)
        manifold_obs_sorted[i:end] = pca.transform(chunk_z)

    # Unsort to original library order
    manifold_obs = np.empty_like(manifold_obs_sorted)
    manifold_obs[sort_order] = manifold_obs_sorted

    if verbose:
        mem_mb = manifold_obs.nbytes / 1e6
        print(f"    PCA done (streaming, 3 passes): "
              f"obs {manifold_obs.shape}, {mem_mb:.1f} MB")

    # === PASS 4: Doublet simulation via random access ===
    rng = np.random.RandomState(random_state)
    n_sim = int(n_cells * sim_doublet_ratio)
    pair_ix = rng.randint(0, n_cells, size=(n_sim, 2))

    manifold_sim = np.empty((n_sim, n_components), dtype=np.float64)

    for i in range(0, n_sim, batch_size):
        end = min(i + batch_size, n_sim)
        batch_pairs = pair_ix[i:end]

        # Gather unique cell indices needed for this batch
        unique_local = np.unique(batch_pairs.ravel())
        unique_global = lib_cell_indices[unique_local]

        # Sort for sequential chunk access
        gsort = np.argsort(unique_global)
        sorted_unique_global = unique_global[gsort]
        raw_data = ml.rows(sorted_unique_global)[:, gene_filter]

        # Build position map: local cell index → row in raw_data
        pos_map = np.empty(n_cells, dtype=np.int32)
        pos_map[unique_local[gsort]] = np.arange(len(unique_local))

        a_pos = pos_map[batch_pairs[:, 0]]
        b_pos = pos_map[batch_pairs[:, 1]]

        sim_raw = raw_data[a_pos] + raw_data[b_pos]
        sim_tc = total_counts[batch_pairs[:, 0]] + total_counts[batch_pairs[:, 1]]

        sim_z = _normalize_zscore_chunk(sim_raw, sim_tc, 1e6,
                                        gene_mean_tpm, gene_std_tpm)
        manifold_sim[i:end] = pca.transform(sim_z)

    if verbose:
        sim_mb = manifold_sim.nbytes / 1e6
        print(f"    Simulated {n_sim} doublets (streaming), {sim_mb:.1f} MB")

    # 7-9. KNN + scoring + threshold
    return _knn_score_threshold(
        manifold_obs, manifold_sim, n_cells, n_sim,
        n_neighbors, expected_doublet_rate, random_state,
        threshold, verbose,
    )


# ============================================================================
# Shared scoring (KNN + Bayesian + threshold)
# ============================================================================

def _knn_score_threshold(manifold_obs, manifold_sim, n_cells, n_sim,
                          n_neighbors, expected_doublet_rate, random_state,
                          threshold, verbose):
    """KNN, Bayesian scoring, and thresholding — shared by both paths."""
    _k = n_neighbors or int(round(0.5 * np.sqrt(n_cells)))
    k_adj = int(round(_k * (1 + n_sim / float(n_cells))))
    k_adj = min(k_adj, n_cells + n_sim - 1)

    manifold = np.vstack([manifold_obs, manifold_sim])
    doub_labels = np.concatenate([
        np.zeros(n_cells, dtype=int),
        np.ones(n_sim, dtype=int),
    ])

    # k_adj neighbours OTHER than the point itself, as scrublet takes them
    # (get_knn_graph: annoy's get_nns_by_item(i, k + 1)[1:], or scikit-learn's
    # kneighbors() on the fitted points, which leaves each point out).
    # pynndescent's neighbor_graph lists the point itself first, so it is asked
    # for one more and the point is dropped: otherwise every simulated doublet
    # counted itself as a doublet neighbour and every cell spent one of its k
    # places on itself.
    from pynndescent import NNDescent
    n_points = manifold.shape[0]
    index = NNDescent(manifold, n_neighbors=min(k_adj + 1, n_points),
                      metric='euclidean', random_state=random_state)
    knn_indices = _without_self(index.neighbor_graph[0], k_adj)

    if verbose:
        print(f"    KNN done: k={_k}, k_adj={k_adj}")

    # Bayesian doublet scoring (exact scrublet formula)
    doub_neigh_mask = doub_labels[knn_indices] == 1
    n_doub_neigh = doub_neigh_mask.sum(axis=1).astype(float)
    N = float(k_adj)
    rho = expected_doublet_rate
    r = n_sim / float(n_cells)

    q = (n_doub_neigh + 1) / (N + 2)
    Ld = q * rho / r / (1 - rho - q * (1 - rho - rho / r))
    Ld = np.clip(Ld, 0, None)

    doublet_score_obs = Ld[doub_labels == 0]
    doublet_score_sim = Ld[doub_labels == 1]

    if threshold is not None:
        lib_threshold = threshold
    else:
        lib_threshold = _auto_threshold(doublet_score_sim, expected_doublet_rate)

    lib_predicted = doublet_score_obs > lib_threshold

    if verbose:
        n_doublets = lib_predicted.sum()
        pct = 100 * n_doublets / n_cells
        print(f"    Threshold: {lib_threshold:.3f}, "
              f"doublets: {n_doublets}/{n_cells} ({pct:.1f}%)")

    return doublet_score_obs, lib_predicted


# ============================================================================
# Helpers
# ============================================================================

def _without_self(knn, k):
    """Each row's first ``k`` neighbours other than the row's own point.

    When the point is not among a row's neighbours (an approximate search can
    miss it, e.g. among exact duplicates), the row's first ``k`` are taken.
    """
    knn = np.asarray(knn)
    n, width = knn.shape
    is_self = knn == np.arange(n)[:, None]
    has_self = is_self.any(axis=1)
    first = np.argmax(is_self, axis=1)
    cols = np.arange(width)
    shift = (cols[None, :] >= first[:, None]) & has_self[:, None]
    gather = np.minimum(cols[None, :] + shift.astype(np.intp), width - 1)
    return np.take_along_axis(knn, gather, axis=1)[:, :k]


def _sparse_zscore(E, gene_mean, gene_std):
    """Z-score sparse matrix using precomputed mean/std. Returns dense."""
    if sparse.issparse(E):
        E_dense = E.toarray()
    else:
        E_dense = np.array(E)
    E_dense = E_dense.astype(np.float64)
    E_dense = (E_dense - gene_mean) / gene_std
    return E_dense


def _auto_threshold(doublet_scores_sim, expected_rate=0.06):
    """Auto-detect threshold from simulated doublet score distribution."""
    try:
        from skimage.filters import threshold_minimum
        t = threshold_minimum(doublet_scores_sim)
        return t
    except (ImportError, RuntimeError):
        pass

    from scipy.signal import find_peaks
    hist, bin_edges = np.histogram(doublet_scores_sim, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    hist_smooth = np.convolve(hist, kernel, mode='same')

    valleys, _ = find_peaks(-hist_smooth, distance=3)
    if len(valleys) > 0:
        main_peak = np.argmax(hist_smooth[:len(hist_smooth) // 2 + 1])
        valid = valleys[valleys > main_peak]
        if len(valid) > 0:
            return bin_centers[valid[0]]

    return np.percentile(doublet_scores_sim, 100 * (1 - expected_rate))


# ============================================================================
# Input handlers
# ============================================================================

from ..utils._cytome_compat import is_cytome_input as _is_cytome_input


def _holds_integers(matrix) -> bool:
    from ..tools._normalization import _sample_stored_values
    vals = np.asarray(_sample_stored_values(matrix), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return bool(vals.size == 0 or np.allclose(vals, np.round(vals)))


def _not_counts_error(what):
    return ValueError(
        f"scrublet: {what} holds non-integer values, so it is not raw counts. Scrublet simulates "
        f"doublets by adding two cells' raw counts, which means nothing on normalised or "
        f"log-transformed values. Pass layer= the layer that holds the raw counts.")


def _load_anndata_counts(data, layer=None):
    """Extract raw counts and obs from AnnData.

    ``layer`` names the counts; without it ``.X`` is used when it holds
    integers, else ``.raw.X``, and a matrix that is not integer counts is
    refused (a log1p ``.X`` with no ``.raw`` used to be scored as counts).
    """
    if layer is not None:
        if layer not in data.layers:
            raise KeyError(f"scrublet: layer {layer!r} not in adata.layers ({list(data.layers)}).")
        counts = data.layers[layer]
    elif hasattr(data, 'raw') and data.raw is not None:
        sample = data.X[:min(100, data.shape[0])]
        if sparse.issparse(sample):
            sample = sample.toarray()
        if np.allclose(sample, np.round(sample)):
            counts = data.X
        else:
            counts = data.raw.X
            if not _holds_integers(counts):
                raise _not_counts_error("neither adata.X nor adata.raw.X")
    else:
        counts = data.X
        if not _holds_integers(counts):
            raise _not_counts_error("adata.X (and there is no adata.raw)")

    if not sparse.issparse(counts):
        counts = sparse.csc_matrix(counts)
    elif not sparse.isspmatrix_csc(counts):
        counts = counts.tocsc()

    return counts, data.obs


def _resolve_cytome_layer(data, layer=None, modality="RNA"):
    """Open cytome Dataset and find the counts to score.

    ``layer`` names ``{modality}_{layer}``; without it the store's
    ``{modality}_raw_counts`` or ``{modality}_counts`` is used, and its values
    must be integers. Any other matrix is never guessed at (the old fallback
    took any matrix with 'RNA' or 'counts' in its name, then the first one).
    A column-major matrix is refused: each row block would decode the whole
    matrix. Returns (dataset, measurement_layer).
    """
    from ..utils._cytome_compat import open_cytome_sync, require_row_major
    if isinstance(data, str):
        ds = open_cytome_sync(data)
    else:
        ds = data

    from cytome.core.measurement import MeasurementLayer

    info = ds.to_info_dict()
    matrix_names = [m['name'] for m in info.get('matrices', [])]

    if layer is not None:
        candidates = [f"{modality}_{layer}"]
    else:
        candidates = [f"{modality}_raw_counts", f"{modality}_counts"]
    layer_name = next((c for c in candidates if c in matrix_names), None)
    if layer_name is None:
        raise ValueError(
            f"scrublet: no {' or '.join(candidates)} matrix in this cytome. "
            f"Available matrices: {matrix_names}. Pass layer= (and modality=) to name the raw counts.")
    require_row_major(ds, modality, layer_name[len(modality) + 1:], "scrublet",
                      "Scrublet reads each library's cells by row; re-import the counts row-major.")

    ml = MeasurementLayer(ds._conn, layer_name)
    if layer is None:
        first = ml.rows(np.arange(min(200, ml.shape[0]), dtype=np.int64))
        if not _holds_integers(first):
            raise _not_counts_error(f"the cytome's {layer_name}")
    return ds, ml


def _store_cytome_results(ds, scores, predicted, scored=None):
    """Write scrublet results to cytome cells table (NULL for an unscored cell).

    Uses ``executemany`` inside a ``with conn:`` block instead of an
    explicit ``BEGIN``. The cytome connection runs with Python's
    default ``isolation_level=""`` (deferred), so any DML/DDL
    (including ``ALTER TABLE``) implicitly opens a transaction —
    a subsequent explicit ``conn.execute("BEGIN")`` would then raise
    ``OperationalError: cannot start a transaction within a transaction``.
    """
    conn = ds._conn

    # Streaming passes upstream may leave an implicit transaction open;
    # commit it before schema changes so each phase is independent.
    if conn.in_transaction:
        conn.commit()

    existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()]
    if 'scrublet_score' not in existing_cols:
        conn.execute("ALTER TABLE cells ADD COLUMN scrublet_score REAL")
    if 'is_doublet' not in existing_cols:
        conn.execute("ALTER TABLE cells ADD COLUMN is_doublet INTEGER")
    conn.commit()  # finalize schema before the bulk UPDATE

    if scored is None:
        scored = np.isfinite(np.asarray(scores, dtype=np.float64))
    rows = [
        ((float(scores[i]), int(predicted[i])) if scored[i] else (None, None)) + (int(i),)
        for i in range(len(scores))
    ]
    # ``with conn:`` auto-commits on clean exit and rolls back on exception.
    # ``executemany`` issues one prepared statement in C — orders of
    # magnitude faster than a per-row Python loop on million-cell datasets.
    with conn:
        conn.executemany(
            "UPDATE cells SET scrublet_score=?, is_doublet=? WHERE cell_idx=?",
            rows,
        )
