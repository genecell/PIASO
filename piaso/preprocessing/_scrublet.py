"""Doublet detection matching the Scrublet algorithm (Wolock et al., 2019).

Reproduces the exact scrublet pipeline:
  1. Total-count normalize (to mean of total counts)
  2. Gene filtering via v-score (Poisson-noise model + percentile)
  3. Simulate doublets (random pairs, sum raw counts)
  4. Re-normalize (to 1e6 = TPM)
  5. Z-score (mean center + variance normalize, computed on observed only)
  6. PCA via sklearn.decomposition.PCA (or IncrementalPCA for streaming)
  7. KNN with adjusted k: k_adj = k * (1 + n_sim/n_obs)
  8. Bayesian doublet scoring
  9. Auto-threshold via skimage or histogram bimodality

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
):
    """Detect doublets using the Scrublet algorithm (Wolock et al., 2019).

    Processes each library independently. Supports both AnnData and
    cytome Dataset inputs. The cytome path is fully streaming with
    O(batch_size * n_genes) peak RAM per library.

    Parameters
    ----------
    data : AnnData or cytome.Dataset or str
        Input data. For AnnData, uses raw counts from ``X`` or ``raw``.
        For cytome, streams from the RNA measurement layer.
    library_key : str, optional
        Column in ``obs``/``cells`` identifying libraries.
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

    Returns
    -------
    None
        Adds ``scrublet_score`` and ``is_doublet`` to obs/cells.
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
            batch_size=batch_size, verbose=verbose,
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
            verbose=verbose,
        )


# ============================================================================
# AnnData path (in-memory, exact scrublet match)
# ============================================================================

def _scrublet_anndata(
    data, library_key, n_components, sim_doublet_ratio,
    expected_doublet_rate, n_neighbors, min_counts, min_cells,
    min_gene_variability_pctl, random_state, threshold, verbose,
):
    """In-memory scrublet for AnnData objects."""
    counts_full, obs_df = _load_anndata_counts(data)

    n_total = counts_full.shape[0]
    scores = np.zeros(n_total, dtype=np.float64)
    predicted = np.zeros(n_total, dtype=bool)

    if library_key is not None and library_key in obs_df.columns:
        libraries = obs_df[library_key].unique()
    else:
        libraries = [None]

    for lib_i, lib in enumerate(libraries):
        if lib is not None:
            mask = (obs_df[library_key] == lib).values
            lib_name = str(lib)
        else:
            mask = np.ones(n_total, dtype=bool)
            lib_name = "all"

        cell_indices = np.where(mask)[0]
        n_cells = len(cell_indices)

        if verbose:
            print(f"  Library {lib_i+1}/{len(libraries)}: "
                  f"{lib_name} ({n_cells} cells)")

        if n_cells < 50:
            if verbose:
                print(f"    Skipping — too few cells ({n_cells} < 50)")
            continue

        lib_scores, lib_predicted = _scrublet_library_inmemory(
            counts_full[cell_indices], n_components, sim_doublet_ratio,
            expected_doublet_rate, n_neighbors, min_counts, min_cells,
            min_gene_variability_pctl, random_state, threshold, verbose,
        )

        scores[cell_indices] = lib_scores
        predicted[cell_indices] = lib_predicted

    data.obs['scrublet_score'] = scores
    data.obs['is_doublet'] = predicted

    if verbose:
        total_doublets = predicted.sum()
        print(f"  Total doublets: {total_doublets}/{n_total} "
              f"({100*total_doublets/n_total:.1f}%)")


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
):
    """Streaming scrublet for cytome Dataset objects.

    Uses 4 streaming passes per library with O(batch_size * n_genes)
    peak RAM, independent of total cell count.
    """
    ds, ml = _resolve_cytome_layer(data)
    n_total = ml.shape[0]

    scores = np.zeros(n_total, dtype=np.float64)
    predicted = np.zeros(n_total, dtype=bool)

    # Determine libraries from cells table
    conn = ds._conn
    cells_cols = [r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()]

    if library_key and library_key in cells_cols:
        libs = [r[0] for r in conn.execute(
            f"SELECT DISTINCT [{library_key}] FROM cells ORDER BY [{library_key}]"
        ).fetchall()]
        lib_indices_map = {}
        for lib in libs:
            rows = conn.execute(
                f"SELECT cell_idx FROM cells WHERE [{library_key}] = ? "
                "ORDER BY cell_idx", (lib,)
            ).fetchall()
            lib_indices_map[lib] = np.array([r[0] for r in rows], dtype=np.int64)
    else:
        libs = [None]
        lib_indices_map = {None: np.arange(n_total, dtype=np.int64)}

    for lib_i, lib in enumerate(libs):
        lib_cell_indices = lib_indices_map[lib]
        n_cells = len(lib_cell_indices)
        lib_name = str(lib) if lib is not None else "all"

        if verbose:
            print(f"  Library {lib_i+1}/{len(libs)}: "
                  f"{lib_name} ({n_cells} cells) [streaming]")

        if n_cells < 50:
            if verbose:
                print(f"    Skipping — too few cells ({n_cells} < 50)")
            continue

        lib_scores, lib_predicted = _scrublet_library_streaming(
            ml, lib_cell_indices, n_components, sim_doublet_ratio,
            expected_doublet_rate, n_neighbors, min_counts, min_cells,
            min_gene_variability_pctl, random_state, threshold,
            batch_size, verbose,
        )

        scores[lib_cell_indices] = lib_scores
        predicted[lib_cell_indices] = lib_predicted

    _store_cytome_results(ds, scores, predicted)

    if verbose:
        total_doublets = predicted.sum()
        print(f"  Total doublets: {total_doublets}/{n_total} "
              f"({100*total_doublets/n_total:.1f}%)")


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

    if n_genes_filtered < n_components:
        if verbose:
            print(f"    Warning: only {n_genes_filtered} genes selected "
                  f"(< n_components={n_components}), skipping library")
        return np.zeros(n_cells), np.zeros(n_cells, dtype=bool)

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

    from pynndescent import NNDescent
    index = NNDescent(manifold, n_neighbors=k_adj,
                      metric='euclidean', random_state=random_state)
    knn_indices, _ = index.neighbor_graph

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


def _load_anndata_counts(data):
    """Extract raw counts and obs from AnnData."""
    if hasattr(data, 'raw') and data.raw is not None:
        sample = data.X[:min(100, data.shape[0])]
        if sparse.issparse(sample):
            sample = sample.toarray()
        if np.allclose(sample, np.round(sample)):
            counts = data.X
        else:
            counts = data.raw.X
    else:
        counts = data.X

    if not sparse.issparse(counts):
        counts = sparse.csc_matrix(counts)
    elif not sparse.isspmatrix_csc(counts):
        counts = counts.tocsc()

    return counts, data.obs


def _resolve_cytome_layer(data):
    """Open cytome Dataset and find the RNA measurement layer.

    Returns (dataset, measurement_layer).
    """
    from ..utils._cytome_compat import open_cytome_sync
    if isinstance(data, str):
        ds = open_cytome_sync(data)
    else:
        ds = data

    from cytome.core.measurement import MeasurementLayer

    # Try modality-based access first (ds.RNA.counts → RNA_counts)
    info = ds.to_info_dict()
    matrix_names = [m['name'] for m in info.get('matrices', [])]

    layer_name = None
    for candidate in ['RNA_raw_counts', 'RNA_counts', 'counts']:
        if candidate in matrix_names:
            layer_name = candidate
            break
    if layer_name is None:
        # Fallback: any matrix with "RNA" or "counts" in name
        for name in matrix_names:
            if 'RNA' in name or 'counts' in name.lower():
                layer_name = name
                break
    if layer_name is None and matrix_names:
        layer_name = matrix_names[0]

    if layer_name is None:
        raise ValueError("No count matrix found in cytome for scrublet. "
                         f"Available matrices: {matrix_names}")

    ml = MeasurementLayer(ds._conn, layer_name)
    return ds, ml


def _store_cytome_results(ds, scores, predicted):
    """Write scrublet results to cytome cells table.

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

    rows = [
        (float(scores[i]), int(predicted[i]), int(i))
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
