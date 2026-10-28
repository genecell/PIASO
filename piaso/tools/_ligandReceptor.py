import numpy as np
import pandas as pd
import anndata
from scipy import sparse
from tqdm.auto import tqdm
from statsmodels.stats.multitest import multipletests
from typing import List, Optional

def _calculate_interaction_scores(
    specificity_matrix: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    ligand_col: str,
    receptor_col: str,
    sender_cell_types: List[str],
    receiver_cell_types: List[str],
    annotation_col: Optional[str] = None,
    statistic: str = 'product',
) -> pd.DataFrame:
    """
    Internal helper to efficiently calculate ligand-receptor interaction scores.
    Assumes lr_pairs have already been validated.
    """
    # --- 1. Select relevant columns ---
    lr_cols_to_keep = [ligand_col, receptor_col]
    if annotation_col and annotation_col in lr_pairs.columns:
        lr_cols_to_keep.append(annotation_col)
    valid_lr_pairs = lr_pairs[lr_cols_to_keep]

    # --- 2. Prepare Ligand and Receptor Data ---
    ligands = valid_lr_pairs[ligand_col].unique()
    receptors = valid_lr_pairs[receptor_col].unique()

    # rename by POSITION, not by the literal name 'index': reset_index() names
    # the new column after the index's own name when it has one (any matrix
    # built from adata.var_names does), and rename(columns={'index': ...})
    # then silently misses — melt raised KeyError('ligand') two lines later.
    ligand_spec = specificity_matrix.loc[ligands, sender_cell_types].reset_index()
    ligand_spec.columns = [ligand_col] + list(sender_cell_types)
    ligand_long = ligand_spec.melt(id_vars=ligand_col, var_name='sender', value_name='ligand_score')

    receptor_spec = specificity_matrix.loc[receptors, receiver_cell_types].reset_index()
    receptor_spec.columns = [receptor_col] + list(receiver_cell_types)
    receptor_long = receptor_spec.melt(id_vars=receptor_col, var_name='receiver', value_name='receptor_score')

    # --- 3. Merge and Calculate Score ---
    merged_df = pd.merge(valid_lr_pairs, ligand_long, on=ligand_col)
    merged_df = pd.merge(merged_df, receptor_long, on=receptor_col)
    if statistic == 'product':
        merged_df['interaction_score'] = merged_df['ligand_score'] * merged_df['receptor_score']
    elif statistic == 'min':
        merged_df['interaction_score'] = np.minimum(merged_df['ligand_score'], merged_df['receptor_score'])
    else:
        raise ValueError(f"statistic must be 'product' or 'min', got {statistic!r}")
    # the geometric mean is a monotone function of the product, so it ranks
    # exactly as the product does and lives on the specificity scale
    merged_df['specificity_geomean'] = np.sqrt(merged_df['ligand_score'] * merged_df['receptor_score'])

    # --- 4. Clean Up ---
    final_cols = [ligand_col, receptor_col]
    if annotation_col and annotation_col in lr_pairs.columns:
        final_cols.append(annotation_col)
    final_cols.extend(['sender', 'receiver', 'ligand_score', 'receptor_score',
                       'interaction_score', 'specificity_geomean'])
    
    result_df = merged_df[final_cols].rename(columns={
        ligand_col: 'ligand', receptor_col: 'receptor',
        'ligand_score': 'ligand_specificity', 'receptor_score': 'receptor_specificity'})
    
    return result_df

def _scalar_var_names(data, modality="RNA"):
    """Feature names for either backend, without materialising the matrix."""
    from ..utils._cytome_compat import is_cytome_input, open_cytome

    if not is_cytome_input(data):
        return pd.Index(data.var_names)
    # open_cytome is the helper that knows whether it opened the Dataset and
    # therefore whether it may close it; open_cytome_sync rejects an already
    # open one on purpose ("caller owns the close").
    with open_cytome(data) as ds:
        from cytome import modality_feature_table_info
        ftab, idx_col, name_col = modality_feature_table_info(ds, modality)
        return pd.Index(pd.read_sql(
            f'SELECT "{name_col}" AS n FROM {ftab} ORDER BY "{idx_col}"',
            ds._conn).n.astype(str))


def _scalar_resolve_layer(data, layer, modality="RNA"):
    """The layer SCALAR reads, the same matrix on both backends.

    AnnData: ``None`` is ``.X``; a name must be a layer (a misspelt one used to
    fall back to ``.X`` without a word). Cytome: a name is
    ``{modality}_{name}``; ``None`` is the matrix that was the AnnData's ``.X``
    when the store was converted (``_anndata_X_layer``), else the counts, so a
    store read with ``layer=None`` gives what its AnnData gave (it always read
    the counts, while the AnnData read ``.X``).
    """
    from ..utils._cytome_compat import is_cytome_input, open_cytome

    if not is_cytome_input(data):
        if layer is not None and layer not in data.layers:
            raise KeyError(
                f"runSCALAR: layer={layer!r} is not a layer of the AnnData "
                f"(layers: {list(data.layers)}); pass None for .X")
        return layer
    if layer is not None:
        return layer
    with open_cytome(data, mode="r") as ds:
        try:
            recorded = ds.metadata.get("_anndata_X_layer")
        except Exception:
            recorded = None
        if (isinstance(recorded, str) and recorded.startswith(f"{modality}_")
                and ds.matrix_meta(recorded) is not None):
            return recorded[len(modality) + 1:]
    return "counts"


def _scalar_gene_stats(data, layer, universe_mask_names, modality="RNA",
                       batch_size=2048):
    """Per-gene (mean, variance) for the matched-gene null, from AnnData or a cytome.

    AnnData reads the matrix in memory; a cytome streams it chunk by chunk and
    accumulates column sums and sums of squares, so the null's matching space
    costs one pass and bounded RAM rather than a materialised matrix. Both
    paths compute the same two statistics on the same layer convention, which
    is what keeps a cytome run and an AnnData run of the same data comparable.

    ``universe_mask_names`` restricts to the gene universe (adata ∩ specificity
    matrix) *before* the stats, so a matched index can never point at a gene
    whose specificity score is NaN.
    """
    from ..utils._cytome_compat import (is_cytome_input as _is_cyt,
                                        require_row_major)
    if _is_cyt(data):
        require_row_major(
            data, modality, layer or "counts", "runSCALAR's gene statistics",
            "Ligand-receptor pairs are gene symbols, so this is normally an "
            "RNA or GA modality, both of which are row-major.")
    from ..utils._cytome_compat import is_cytome_input, open_cytome

    if not is_cytome_input(data):
        var_names = pd.Index(data.var_names)
        X = data.X if layer is None else data.layers[layer]
        keep = pd.Index([g for g in var_names if g in universe_mask_names])
        X = X[:, var_names.get_indexer(keep)]
        n = X.shape[1]
        if sparse.issparse(X):
            X = X.tocsr()
            # bincount, not scipy's .mean(): scipy accumulates a float32
            # matrix's column sums in float32, which costs ~1e-7 relative
            # precision and made the in-memory and streaming paths disagree
            # on the same data. bincount accumulates in float64, matching
            # the streaming accumulator exactly.
            d = X.data.astype(np.float64, copy=False)
            col_sum = np.bincount(X.indices, weights=d, minlength=n)
            col_sq = np.bincount(X.indices, weights=d * d, minlength=n)
        else:
            X = np.asarray(X, dtype=np.float64)
            col_sum = X.sum(axis=0)
            col_sq = (X ** 2).sum(axis=0)
        mean = col_sum / X.shape[0]
        return keep, mean, col_sq / X.shape[0] - mean ** 2

    with open_cytome(data) as ds:
        from cytome import modality_feature_table_info
        ftab, idx_col, name_col = modality_feature_table_info(ds, modality)
        feats = pd.read_sql(
            f'SELECT "{idx_col}" AS i, "{name_col}" AS n FROM {ftab} ORDER BY "{idx_col}"',
            ds._conn)
        var_names = pd.Index(feats.n.astype(str))
        keep_pos = np.flatnonzero(var_names.isin(universe_mask_names))
        n_kept = len(keep_pos)
        col_sum = np.zeros(n_kept, dtype=np.float64)
        col_sq = np.zeros(n_kept, dtype=np.float64)
        n_cells = 0
        for chunk in ds.iter_chunks(modality=modality,
                                    layer=(layer or "counts"),
                                    batch_size=batch_size):
            X = chunk[0] if isinstance(chunk, tuple) else chunk
            X = X.tocsr() if sparse.issparse(X) else np.asarray(X)
            sub = X[:, keep_pos]
            if sparse.issparse(sub):
                col_sum += np.asarray(sub.sum(axis=0)).ravel()
                col_sq += np.asarray(sub.power(2).sum(axis=0)).ravel()
            else:
                col_sum += sub.sum(axis=0)
                col_sq += (sub ** 2).sum(axis=0)
            n_cells += sub.shape[0]
        if n_cells == 0:
            raise ValueError("runSCALAR: the cytome yielded no cells for "
                             f"modality={modality!r}, layer={layer or 'counts'!r}")
        mean = col_sum / n_cells
        return var_names[keep_pos], mean, col_sq / n_cells - mean ** 2


def _scalar_labels(data, groupby, modality="RNA"):
    """Per-cell group labels as an object array, from AnnData ``.obs`` or a
    cytome's ``cells`` table (ordered by ``cell_idx``, the order
    ``iter_chunks`` yields rows in)."""
    from ..utils._cytome_compat import (is_cytome_input, open_cytome,
                                        require_cells_column)

    if not is_cytome_input(data):
        if groupby not in data.obs.columns:
            close = [c for c in data.obs.columns if c.lower() == str(groupby).lower()]
            raise KeyError(
                f"runSCALAR: groupby={groupby!r} is not a column of adata.obs."
                + (f" Did you mean {close[0]!r}?" if close else "")
                + f" Available: {sorted(data.obs.columns)}")
        return np.asarray(data.obs[groupby].astype(str).values, dtype=object)
    # Checked before the name is interpolated into the SQL below: SQLite treats
    # a double-quoted token that does not resolve to a column as a string
    # literal, so a typo would come back as that literal for every cell -- one
    # group, named after the typo, and every number downstream consistent with
    # it.
    require_cells_column(data, groupby, argument="runSCALAR: groupby")
    with open_cytome(data) as ds:
        rows = ds._conn.execute(
            f'SELECT "{groupby}" FROM cells ORDER BY cell_idx').fetchall()
    return np.asarray([str(r[0]) for r in rows], dtype=object)


def _scalar_detection_fraction(data, layer, groupby, universe_names,
                               modality="RNA", batch_size=2048):
    """Fraction of each group's cells in which each gene is non-zero.

    Returns a DataFrame (genes in ``universe_names`` order x groups). "Detected"
    is "non-zero in the cell", which every count-preserving normalisation
    (log1p, INFOG) leaves exactly where the raw counts put it, so the fraction
    is the same from any such ``layer`` — and it is the same statistic COSG's
    ``expressed_pct`` filter counts. A cytome is streamed; an AnnData is read
    in memory. Both accumulate per-group non-zero counts as
    ``onehot(labels).T @ (X != 0)``.
    """
    from ..utils._cytome_compat import is_cytome_input, open_cytome

    labels = _scalar_labels(data, groupby, modality=modality)
    groups = pd.Index(pd.unique(labels))
    grp_pos = groups.get_indexer(labels)

    def _accumulate(X, pos, counts, sizes):
        X = X.tocsr() if sparse.issparse(X) else np.asarray(X)
        ind = (X != 0)
        if not sparse.issparse(ind):
            ind = sparse.csr_matrix(ind)
        onehot = sparse.csr_matrix(
            (np.ones(len(pos)), (pos, np.arange(len(pos)))),
            shape=(len(groups), len(pos)))
        counts += np.asarray((onehot @ ind.astype(np.float64)).todense())
        sizes += np.bincount(pos, minlength=len(groups))

    if not is_cytome_input(data):
        var_names = pd.Index(data.var_names)
        X = data.X if layer is None else data.layers[layer]
        keep = pd.Index(universe_names)
        X = X[:, var_names.get_indexer(keep)]
        counts = np.zeros((len(groups), len(keep)))
        sizes = np.zeros(len(groups), dtype=np.int64)
        _accumulate(X, grp_pos, counts, sizes)
        genes = keep
    else:
        with open_cytome(data) as ds:
            from cytome import modality_feature_table_info
            ftab, idx_col, name_col = modality_feature_table_info(ds, modality)
            feats = pd.read_sql(
                f'SELECT "{idx_col}" AS i, "{name_col}" AS n FROM {ftab} ORDER BY "{idx_col}"',
                ds._conn)
            var_names = pd.Index(feats.n.astype(str))
            keep_pos = var_names.get_indexer(pd.Index(universe_names))
            counts = np.zeros((len(groups), len(keep_pos)))
            sizes = np.zeros(len(groups), dtype=np.int64)
            for chunk in ds.iter_chunks(modality=modality,
                                        layer=(layer or "counts"),
                                        batch_size=batch_size):
                X, rows = (chunk if isinstance(chunk, tuple)
                           else (chunk, None))
                if rows is None:
                    raise ValueError("runSCALAR: the cytome chunk iterator did "
                                     "not yield row indices; cannot align "
                                     "cells to groups")
                X = X.tocsr() if sparse.issparse(X) else np.asarray(X)
                _accumulate(X[:, keep_pos], grp_pos[np.asarray(rows)], counts, sizes)
            genes = pd.Index(universe_names)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = counts / np.maximum(sizes, 1)[:, None]
    return pd.DataFrame(frac.T, index=genes, columns=groups)


def _prepare_background_genes(
    adata: anndata.AnnData,
    layer: str = None,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 30,
    genes_universe=None,
    query_genes=None,
    modality: str = "RNA",
    stats=None,
):
    """Matched-gene sets for the null, built on piaso's shared matcher.

    Reuses :func:`piaso.tools._normalization._knn_from_mean_var` — the KDTree
    over per-gene (mean, variance) that ``piaso.tl.score`` uses — instead of a
    private copy. That buys three corrections at once:

    - self-exclusion **by identity**, not by position: thousands of
      low-expression genes share identical (mean, variance), the ordering
      among exact distance ties is arbitrary, and the old ``indices[:, 1:]``
      could keep a gene in its own matched set while dropping a twin;
    - ``query_rows``: only genes that appear in LR pairs need matched sets,
      so the tree is queried for those rather than the whole transcriptome;
    - stats follow score()'s convention — computed on the scored ``layer``
      when given — so the controls and the query see the same normalisation.

    ``genes_universe`` restricts the candidate space (and the stats) to a gene
    subset — the adata ∩ specificity-matrix intersection, or, under a
    detection floor, the genes detected in one cell type. ``stats`` passes
    (universe, mean, variance) already computed over a superset, so the
    per-cell-type universes reuse one pass over the matrix. Without it, matched
    indices could point at genes absent from the specificity matrix, whose
    NaN scores can never exceed anything but still count in the p-value
    denominator: silently anticonservative exactly when the universes differ.

    Returns
    -------
    (knn_df, frac_below) : matched-set indices as a DataFrame indexed by the
    universe's gene names (values are positions *within the universe*), and
    per queried gene the fraction of its matched set with strictly lower mean
    — the matchability diagnostic. 0.5 is ideal; 1.0 means the gene sits above
    its whole matched set and beats its null arithmetically (the matched-set
    ceiling), so its p-value overstates.
    """
    from ._normalization import _knn_from_mean_var
    from ..utils._cytome_compat import is_cytome_input

    if genes_universe is None:
        genes_universe = _scalar_var_names(adata, modality=modality)
    if stats is None:
        universe, mean, variance = _scalar_gene_stats(
            adata, layer, set(genes_universe), modality=modality)
    else:
        # ``stats`` = (universe, mean, variance) computed once over a superset;
        # subset rather than re-stream the matrix per cell type.
        u_all, m_all, v_all = stats
        keep = np.flatnonzero(pd.Index(u_all).isin(set(genes_universe)))
        universe, mean, variance = pd.Index(u_all)[keep], m_all[keep], v_all[keep]
    mean_var = np.column_stack([mean, variance])

    query_rows = None
    if query_genes is not None:
        query_rows = universe.get_indexer(pd.Index(pd.unique(pd.Series(list(query_genes)))))
        query_rows = query_rows[query_rows >= 0]

    knn_idx = _knn_from_mean_var(mean_var, n_nearest_neighbors, leaf_size,
                                 query_rows=query_rows)

    rows = np.arange(len(universe)) if query_rows is None else query_rows
    frac_below = pd.Series(0.5, index=universe, dtype=float)
    frac_below.iloc[rows] = (mean[knn_idx[rows]] < mean[rows, None]).mean(axis=1)

    knn_df = pd.DataFrame(knn_idx, index=universe)
    return knn_df, frac_below


def _exact_exceed_counts(lig_bg, rec_bg, observed):
    """#{(a, b): lig_bg[i, a] * rec_bg[i, b] >= observed[i]} per row, exactly.

    The k x k product support never has to be materialised: with non-negative
    scores, ``lig * rec >= obs``  ⟺  ``rec >= obs / lig`` (for lig > 0), so
    sorting each receptor row once and running one vectorised ``searchsorted``
    per interaction gives the exact count in O(k log k) per row instead of
    O(k²) memory. Specificity scores are non-negative by contract (raw COSG
    scores are; ``runSCALAR`` validates its matrix before it gets here), so a
    negative here is a programming error and raises rather than silently
    taking a different code path. ``_exact_exceed_counts_dense`` is the
    literal outer product the property test checks this against.
    """
    lig_bg = np.asarray(lig_bg, dtype=np.float64)
    rec_bg = np.asarray(rec_bg, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    n, k = lig_bg.shape

    if (lig_bg < 0).any() or (rec_bg < 0).any():
        raise ValueError("specificity scores must be non-negative")

    rec_sorted = np.sort(rec_bg, axis=1)
    exceed = np.empty(n, dtype=np.int64)
    for i in range(n):
        obs = observed[i]
        if obs <= 0:
            # every product is >= 0 >= obs
            exceed[i] = k * k
            continue
        lig = lig_bg[i]
        pos = lig > 0
        if not pos.any():
            exceed[i] = 0
            continue
        thresholds = obs / lig[pos]
        exceed[i] = int(
            (k - np.searchsorted(rec_sorted[i], thresholds, side='left')).sum())
    return exceed


def _exact_exceed_counts_min(lig_bg, rec_bg, observed):
    """#{(a, b): min(lig_bg[i, a], rec_bg[i, b]) >= observed[i]} per row, exactly.

    ``min(l, r) >= s`` iff both ``l >= s`` and ``r >= s``, so the count over
    the k x k grid factorises into a product of two one-sided counts.
    """
    lig_bg = np.asarray(lig_bg, dtype=np.float64)
    rec_bg = np.asarray(rec_bg, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if (lig_bg < 0).any() or (rec_bg < 0).any():
        raise ValueError("specificity scores must be non-negative")
    k = lig_bg.shape[1]
    n_l = (lig_bg >= observed[:, None]).sum(axis=1)
    n_r = (rec_bg >= observed[:, None]).sum(axis=1)
    out = (n_l * n_r).astype(np.int64)
    out[observed <= 0] = k * k
    return out


def _null_mean(lig_bg, rec_bg, statistic):
    """Mean of the k x k null support per row, without materialising it.

    product: mean(l) * mean(r). min: with both rows sorted, the mean of
    min(l_i, r_j) over the grid is
    (sum_i l_i * #{r_j >= l_i} + sum_j r_j * #{l_i > r_j}) / k**2.
    """
    lig_bg = np.asarray(lig_bg, dtype=np.float64)
    rec_bg = np.asarray(rec_bg, dtype=np.float64)
    if statistic == 'product':
        return lig_bg.mean(axis=1) * rec_bg.mean(axis=1)
    n, k = lig_bg.shape
    ls = np.sort(lig_bg, axis=1)
    rs = np.sort(rec_bg, axis=1)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        cnt_r_ge_l = k - np.searchsorted(rs[i], ls[i], side='left')
        cnt_l_gt_r = k - np.searchsorted(ls[i], rs[i], side='right')
        out[i] = ((ls[i] * cnt_r_ge_l).sum() + (rs[i] * cnt_l_gt_r).sum()) / (k * k)
    return out


def _exact_exceed_counts_dense(lig_bg, rec_bg, observed, max_elems=8_000_000):
    """The literal outer product, in row chunks bounded by ``max_elems``."""
    n, k = lig_bg.shape
    rows_per = max(1, int(max_elems // (k * k)))
    exceed = np.empty(n, dtype=np.int64)
    for lo in range(0, n, rows_per):
        hi = min(lo + rows_per, n)
        prod = lig_bg[lo:hi, :, None] * rec_bg[lo:hi, None, :]
        exceed[lo:hi] = (prod.reshape(hi - lo, -1)
                         >= observed[lo:hi, None]).sum(axis=1)
    return exceed


### The SCALAR function for the cell type-specific ligand-receptor anlayis
def runSCALAR(
    adata,
    specificity_matrix: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    ligand_col: str = 'ligand',
    receptor_col: str = 'receptor',
    annotation_col: Optional[str] = None,
    sender_cell_types: Optional[List[str]] = None,
    receiver_cell_types: Optional[List[str]] = None,
    n_permutations: int = 1000,
    n_nearest_neighbors: int = 100,
    layer: str = None,
    random_seed: int = 42,
    rank_by_score: bool = True,
    chunk_size: int = 50000,
    prefilter_fdr: bool = True,
    prefilter_threshold: float = 0.0,
    modality: str = 'RNA',
    expressed_pct: Optional[float] = 0.1,
    groupby: Optional[str] = None,
    statistic: str = 'product',
    enrichment_pseudocount: float = 1e-12,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculates ligand-receptor interaction scores, computes permutation-based p-values
    using a vectorized approach, and corrects for multiple testing using FDR for each
    cell type-cell type pair independently.

    Args:
        adata: AnnData object, or a ``.cytome`` path / open ``cytome.Dataset``.
            A cytome is streamed chunk by chunk to build the matched-gene
            null's (mean, variance) space, so the memory cost is set by the
            chunk size rather than the cell count; the statistics and the
            resulting p-values are the same either way.
        modality: Cytome modality supplying expression (default ``'RNA'``).
            Ignored for AnnData.
        specificity_matrix: DataFrame with genes as rows, cell types as columns,
                            and specificity scores as values.
        lr_pairs: DataFrame listing interacting gene pairs.
        ligand_col: Column name for ligands in lr_pairs.
        receptor_col: Column name for receptors in lr_pairs.
        annotation_col: Optional column in lr_pairs to carry over.
        sender_cell_types: List of cell types to use as senders. If None, all are used.
        receiver_cell_types: List of cell types to use as receivers. If None, all are used.
        n_permutations: Number of permutations for the null distribution.
        n_nearest_neighbors: Number of control genes to sample from.
        layer: Layer to use for expression. AnnData: ``None`` is ``.X``, a name
            must be a layer. Cytome: ``None`` is the matrix that was the
            AnnData's ``.X`` when the store was converted (else the counts), a
            name is ``{modality}_{name}``; so both backends read the same matrix.
        random_seed: Seed for reproducibility.
        rank_by_score: If True, sorts the final output by interaction_score.
        chunk_size: The number of interactions to process in each vectorized chunk
                    to manage memory usage.
        prefilter_fdr: If True, interactions with scores <= prefilter_threshold are
                       excluded from FDR calculation within each group and assigned an
                       FDR of 1.0.
        prefilter_threshold: The score threshold used for pre-filtering before FDR calculation.
        expressed_pct: Detection floor, as a fraction of a cell type's cells in
            which a gene is non-zero on ``layer`` (default 0.1, CellPhoneDB's
            convention). An interaction is tested only if the ligand clears
            the floor in the sender and the receptor in the receiver; its
            matched controls are drawn from genes that clear the floor in the
            same cell type, so the null never contains a control that is not
            expressed there. ``None`` disables the floor and reproduces the
            unfloored test exactly. Requires ``groupby``.
        groupby: Column of ``adata.obs`` (or of a cytome's ``cells`` table)
            holding the cell-type label of every cell. Its values must include
            every sender and receiver, i.e. the columns of
            ``specificity_matrix``. Required when ``expressed_pct`` is set.
        statistic: The test statistic. ``'product'`` (default) is
            ``spec[ligand, sender] * spec[receptor, receiver]``: one very
            specific gene can carry a pair. ``'min'`` is the smaller of the
            two, so both sides must beat the controls' weaker sides — the
            "are both specific" question. Its exact null factorises into two
            one-sided counts and costs nothing extra.
        verbose: Print the step messages and progress bars (default True).
        enrichment_pseudocount: Guard for ``log2_enrichment``: the null mean
            is floored at this value and it is added to the observation, so a
            zero never reaches the log. It is on the *product* scale, where
            called nulls sit at 1e-7 to 1e-3, so the default 1e-12 changes no
            reported value; a larger value caps the enrichment of pairs with
            the sparsest nulls (1e-6 halved the median on the tutorial data).

    Returns:
        A pandas DataFrame with interaction scores, exact p-values, FDR-corrected
        p-values, and two diagnostics: ``null_matchability`` (fraction of the
        matched sets sitting below the gene in mean space — >= 0.99 means the
        matched-set ceiling is active and the p-value overstates) and
        ``call_breadth`` (fraction of the sender-receiver pairs in which this
        ligand-receptor pair was tested, i.e. cleared the detection floor, that
        call it at FDR < 0.05; NaN where it was tested nowhere — results
        calling most of the grid carry no group-specific information). With a
        detection floor the table also carries ``testable`` (both genes clear
        the floor in their cell types) and ``ligand_expressed_pct`` /
        ``receptor_expressed_pct`` (the detection fractions the floor was
        applied to); untestable rows have ``p_value = 1`` and leave BH.

        Three columns give the score a readable magnitude. ``ligand_specificity``
        and ``receptor_specificity`` are the two inputs to the statistic.
        ``specificity_geomean`` is their geometric mean — a monotone function
        of the product, so it ranks exactly as ``interaction_score`` does
        under the default statistic, but a pair of two 0.02 specificities
        reads 0.02 rather than 0.0004; size plots by it. ``null_mean`` is the
        mean of the interaction's own k x k matched null, and
        ``log2_enrichment = log2((observed + eps) / max(null_mean, eps))`` is
        how far the observation sits above what expressed,
        expression-matched gene pairs achieve between these two cell types —
        the one number that is comparable *across* sender–receiver pairs,
        because every pair is referenced to its own null. Raw scores are
        not: specificity is calibrated within a cell type, so a sharp small
        subclass reaches high scores because its markers are exclusive.

    Notes:
        **Input contract.** ``specificity_matrix`` must be finite and
        non-negative: raw COSG scores over all genes
        (``cosg.cosg(..., n_genes_user=adata.n_vars)``) are the recommended
        input. NaN raises — it is what ``cosg.indexByGene`` leaves for genes
        outside a top-N list unless ``set_nan_to_zero=True``, and a NaN
        observation would otherwise score at the p-value floor. Negative
        values raise — COSG marks genes failing its own ``expressed_pct``
        filter with −1, and two such sentinels multiply to +1, the largest
        product in the table. Any per-column *linear* rescaling (dividing a
        cell type's column by a constant, e.g. its IQR) leaves every p-value
        and every within-pair ranking unchanged, because the observed product
        and its null scale together; a log or other concave transform does
        change them, and ``cosg.iqrLogNormalize`` (which applies log1p) is
        therefore not recommended as input. Scores compare *within* a
        sender–receiver pair, not across pairs.

        **The detection floor is applied in the test, not in the matrix.**
        Zeroing lowly-detected genes in the specificity matrix would leave
        the observed pair conditioned on passing the floor while its
        expression-matched controls are zeroed wherever they fall below it,
        so most of the null support becomes zeros and the p-value drifts
        toward "detected in more cells than its peers". Here each gene's
        controls are drawn among genes detected at ``expressed_pct`` in the
        same cell type, so the support stays a full ``k x k`` of real scores
        and the question asked is: given both genes are expressed here, is
        their specificity product unusual among expressed, expression-matched
        genes?

        **The p-values are exact, not sampled.** Each interaction's null is the
        full ``k x k`` support of matched-ligand x matched-receptor score
        products (``k = n_nearest_neighbors``), enumerated rather than drawn
        from: ``p = (#{support >= observed} + 1) / (k**2 + 1)``. Because a
        gene's matched set excludes the gene itself by identity, the observed
        product is genuinely outside its own support, and this is the exact
        p-value under exchangeability of the observed product with its matched
        products — the ``+1`` is that statement, not a Monte Carlo correction.
        The design follows the matched-control family of scDRS (Zhang et al.,
        Nature Genetics 2022; scDRS matches by mean/variance *bins*, here
        controls are the k nearest neighbours in (mean, variance) space).

        **The p-value floor is ``1/(k**2+1)``** (1/10001 at the default
        k=100; measured on MTG, the 100th-nearest neighbour is still matched
        to a median 0.016 dex in mean — barely looser than rank 30's 0.009 —
        and k=100 lowers the number of saturated matched sets).
        ``n_permutations`` never was and is not a way past it — only
        ``n_nearest_neighbors`` is. BH runs within each (sender, receiver)
        group deliberately, so each pair's threshold is driven by its own
        p-value distribution; the smallest attainable FDR for an isolated hit
        in a group testing ``m`` interactions is ``m/(k**2+1)``, and a warning
        fires when that bound exceeds 0.05.

        **Pre-filtering caveat**: ``prefilter_fdr`` excludes
        ``interaction_score <= prefilter_threshold`` rows from BH. At the
        default threshold of 0 this only removes never-rejectable tests
        (a non-positive specificity product has p ≈ 1). Any threshold above 0
        is selection on the test statistic itself and voids formal FDR control
        among the survivors — a warning says so if you set one.
    """
    _say = print if verbose else (lambda *a, **k: None)
    if n_permutations != 1000:
        import warnings
        warnings.warn(
            "runSCALAR: p-values are computed exactly over the "
            "n_nearest_neighbors**2 product support; n_permutations is "
            "ignored (and no longer sets a floor — only n_nearest_neighbors "
            "does).", stacklevel=2)
    # No RNG anywhere in the p-value path any more; random_seed is kept for
    # API stability but deliberately not fed to np.random.seed — mutating the
    # process-global RNG was a side effect callers never asked for.

    # --- Step 1: Validate inputs and filter LR pairs ---
    _say("--- Step 1: Validating inputs and filtering LR pairs ---")
    layer = _scalar_resolve_layer(adata, layer, modality=modality)
    _var_names = _scalar_var_names(adata, modality=modality)
    genes_in_adata = set(_var_names)
    genes_in_spec = set(specificity_matrix.index)
    common_genes = genes_in_adata.intersection(genes_in_spec)

    if not common_genes:
        _say("Error: No common genes found between AnnData object and specificity matrix.")
        return pd.DataFrame()

    # Input contract over the genes that can enter the test: finite and
    # non-negative. See Notes for why each is an error and not a warning.
    _spec_common = specificity_matrix.loc[specificity_matrix.index.isin(common_genes)]
    _vals = _spec_common.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isnan(_vals).any():
        _n = int(np.isnan(_vals).sum())
        raise ValueError(
            f"runSCALAR: specificity_matrix has {_n} NaN entries among the "
            f"{len(common_genes)} genes shared with the data. A NaN observation "
            "would be scored at the p-value floor. If the matrix came from "
            "cosg.indexByGene, pass set_nan_to_zero=True, or run COSG with "
            "n_genes_user=adata.n_vars so every gene has a score.")
    if (_vals < 0).any():
        _n = int((_vals < 0).sum())
        raise ValueError(
            f"runSCALAR: specificity_matrix has {_n} negative entries. COSG "
            "marks genes failing its expressed_pct filter with -1, and two "
            "such sentinels multiply to +1 - the largest product in the table. "
            "Build the matrix with the filter off: "
            "piaso.tl.specificity_matrix(adata, groupby=..., "
            "cosg_remove_lowly_expressed=False), or cosg.cosg(adata, groupby=..., "
            "n_genes_user=adata.n_vars, remove_lowly_expressed=False) as in the "
            "SCALAR tutorial. Or set the -1 entries to 0 (cosg.indexByGene does "
            "so by default). runSCALAR's own expressed_pct applies the "
            "detection floor.")
    del _vals, _spec_common

    if statistic not in ('product', 'min'):
        raise ValueError(f"runSCALAR: statistic must be 'product' or 'min', got {statistic!r}")
    if expressed_pct is not None:
        if not (0.0 <= expressed_pct <= 1.0):
            raise ValueError(f"runSCALAR: expressed_pct must be in [0, 1] or "
                             f"None, got {expressed_pct!r}")
        if groupby is None:
            raise ValueError(
                "runSCALAR: expressed_pct is set (default 0.1) but groupby is "
                "None. Pass groupby=<the obs column whose labels are the "
                "columns of specificity_matrix> so the detection floor can be "
                "computed per cell type, or expressed_pct=None to run without "
                "a floor.")

    original_pair_count = len(lr_pairs)
    lr_pairs_filtered = lr_pairs[
        lr_pairs[ligand_col].isin(common_genes) &
        lr_pairs[receptor_col].isin(common_genes)
    ].copy()

    if len(lr_pairs_filtered) < original_pair_count:
        _say(f"Filtered out {original_pair_count - len(lr_pairs_filtered)} LR pairs that were not found in both the AnnData object and the specificity matrix.")
    # a pair listed twice was scored, tested and BH-corrected twice
    _dup = lr_pairs_filtered.duplicated(subset=[ligand_col, receptor_col])
    if _dup.any():
        import warnings
        warnings.warn(
            f"runSCALAR: {int(_dup.sum())} duplicated ligand-receptor pair(s) in lr_pairs; the first "
            f"row of each is kept (a duplicate was tested, and counted in the FDR correction, twice).",
            stacklevel=2)
        lr_pairs_filtered = lr_pairs_filtered[~_dup]

    if lr_pairs_filtered.empty:
        _say("Warning: No valid ligand-receptor pairs found after filtering. Returning an empty DataFrame.")
        return pd.DataFrame()

    # --- Step 2: Calculate all observed interaction scores ---
    _say("\n--- Step 2: Calculating all observed interaction scores ---")
    all_cell_types = specificity_matrix.columns.tolist()
    senders = all_cell_types if sender_cell_types is None else [ct for ct in sender_cell_types if ct in all_cell_types]
    receivers = all_cell_types if receiver_cell_types is None else [ct for ct in receiver_cell_types if ct in all_cell_types]

    interaction_scores_df = _calculate_interaction_scores(
        specificity_matrix=specificity_matrix,
        lr_pairs=lr_pairs_filtered, # Use the filtered dataframe
        ligand_col=ligand_col,
        receptor_col=receptor_col,
        sender_cell_types=senders,
        receiver_cell_types=receivers,
        annotation_col=annotation_col,
        statistic=statistic,
    )

    if interaction_scores_df.empty:
        _say("Warning: No valid interactions found. Returning an empty DataFrame.")
        return pd.DataFrame()
    _say(f"Found {len(interaction_scores_df)} potential interactions to test.")

    # --- 3. Prepare background gene sets (one matcher per cell type under a floor) ---
    _say("\n--- Step 3: Preparing matched background gene sets ---")
    lr_genes = pd.unique(pd.concat([
        lr_pairs_filtered[ligand_col], lr_pairs_filtered[receptor_col]]))
    types_used = sorted(set(senders) | set(receivers))
    detection = None
    if expressed_pct is None:
        bg_global, fb_global = _prepare_background_genes(
            adata, layer=layer, n_nearest_neighbors=n_nearest_neighbors,
            genes_universe=common_genes, query_genes=lr_genes, modality=modality,
        )
        bg_for = {t: (bg_global, fb_global) for t in types_used}
    else:
        import warnings
        detection = _scalar_detection_fraction(
            adata, layer, groupby, sorted(common_genes), modality=modality)
        missing = [t for t in types_used if t not in detection.columns]
        if missing:
            raise ValueError(
                f"runSCALAR: groupby={groupby!r} has no cells labelled "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}; every "
                "sender and receiver (the columns of specificity_matrix) must "
                "be a label in that column.")
        # one pass over the matrix for (mean, variance); per-type universes
        # are subsets of it
        stats = _scalar_gene_stats(adata, layer, common_genes, modality=modality)
        bg_for = {}
        sizes = {}
        for t in types_used:
            universe_t = detection.index[detection[t].values >= expressed_pct]
            sizes[t] = len(universe_t)
            if len(universe_t) <= n_nearest_neighbors + 1:
                warnings.warn(
                    f"runSCALAR: only {len(universe_t)} genes are detected in "
                    f">= {expressed_pct:.0%} of {t!r} cells, fewer than the "
                    f"{n_nearest_neighbors + 1} a matched set needs; every "
                    f"interaction with {t!r} as sender or receiver is "
                    "untestable. Lower expressed_pct or n_nearest_neighbors.",
                    stacklevel=2)
                bg_for[t] = None
                continue
            bg_for[t] = _prepare_background_genes(
                adata, layer=layer, n_nearest_neighbors=n_nearest_neighbors,
                genes_universe=set(universe_t), query_genes=lr_genes,
                modality=modality, stats=stats,
            )
        _say(f"  - Detection floor {expressed_pct:.0%} on layer "
              f"{layer or 'X'!r}: {min(sizes.values()):,}-"
              f"{max(sizes.values()):,} genes per cell type form the matched "
              f"universes")

    # --- 4. Exact p-values over the enumerated product support ---
    n_support = n_nearest_neighbors ** 2
    _say(f"\n--- Step 4: Exact p-values over the {n_nearest_neighbors}x"
          f"{n_nearest_neighbors} matched product support "
          f"(floor 1/{n_support + 1}) ---")

    _exceed_fn = _exact_exceed_counts if statistic == 'product' else _exact_exceed_counts_min
    p_values_list = []
    testable_list = []
    matchability_list = []
    null_mean_list = []
    grouped_interactions = interaction_scores_df.groupby(['sender', 'receiver'])

    for (sender, receiver), group_df in tqdm(grouped_interactions, total=len(grouped_interactions), desc="Processing cell type pairs", disable=not verbose):
        bs, br = bg_for.get(sender), bg_for.get(receiver)
        p_grp = np.ones(len(group_df), dtype=np.float64)
        m_grp = np.full(len(group_df), np.nan, dtype=np.float64)
        nm_grp = np.full(len(group_df), np.nan, dtype=np.float64)
        if bs is None or br is None:
            ok = np.zeros(len(group_df), dtype=bool)
        else:
            (bg_s, fb_s), (bg_r, fb_r) = bs, br
            # membership in a type's matched universe IS clearing the floor
            # there (the universe is the common set without a floor)
            ok = (group_df['ligand'].isin(bg_s.index).values
                  & group_df['receptor'].isin(bg_r.index).values)
            if ok.any():
                # Hoisted: these depend only on the group, not the chunk.
                sender_scores_all = specificity_matrix[sender].reindex(bg_s.index).values
                receiver_scores_all = specificity_matrix[receiver].reindex(bg_r.index).values
                ok_pos = np.flatnonzero(ok)
                for i in range(0, len(ok_pos), chunk_size):
                    pos = ok_pos[i:i + chunk_size]
                    chunk_df = group_df.iloc[pos]
                    lig_idx = bg_s.loc[chunk_df['ligand'].values].values
                    rec_idx = bg_r.loc[chunk_df['receptor'].values].values
                    lig_bg = sender_scores_all[lig_idx]
                    rec_bg = receiver_scores_all[rec_idx]
                    exceed = _exceed_fn(lig_bg, rec_bg,
                                        chunk_df['interaction_score'].values)
                    p_grp[pos] = (exceed + 1) / (n_support + 1)
                    nm_grp[pos] = _null_mean(lig_bg, rec_bg, statistic)
                # Matchability: the larger of the ligand's and receptor's
                # frac-below. It measures whether the MATCHING ASSUMPTION
                # holds, not whether the result is strong - a gene at the top
                # of the abundance range has no peers above it, its matched
                # set lies entirely below it, and it beats its own null
                # arithmetically (LARIS measured the same design failing
                # exactly this way before fixing its pool; SCALAR matches
                # whole-transcriptome, so for the extreme gene there is
                # nothing better to match against and the honest response is
                # to say so).
                m_grp[ok] = np.maximum(
                    fb_s.reindex(group_df['ligand'].values[ok]).values,
                    fb_r.reindex(group_df['receptor'].values[ok]).values)
        p_values_list.append(pd.Series(p_grp, index=group_df.index))
        testable_list.append(pd.Series(ok, index=group_df.index))
        matchability_list.append(pd.Series(m_grp, index=group_df.index))
        null_mean_list.append(pd.Series(nm_grp, index=group_df.index))

    interaction_scores_df['p_value'] = pd.concat(p_values_list)
    interaction_scores_df['testable'] = pd.concat(testable_list)
    interaction_scores_df['null_matchability'] = pd.concat(matchability_list)
    interaction_scores_df['null_mean'] = pd.concat(null_mean_list)
    _eps = float(enrichment_pseudocount)
    interaction_scores_df['log2_enrichment'] = np.log2(
        (interaction_scores_df['interaction_score'] + _eps)
        / np.maximum(interaction_scores_df['null_mean'], _eps))
    if detection is not None:
        _li = detection.index.get_indexer(interaction_scores_df['ligand'])
        _ri = detection.index.get_indexer(interaction_scores_df['receptor'])
        _si = detection.columns.get_indexer(interaction_scores_df['sender'])
        _ci = detection.columns.get_indexer(interaction_scores_df['receiver'])
        interaction_scores_df['ligand_expressed_pct'] = detection.values[_li, _si]
        interaction_scores_df['receptor_expressed_pct'] = detection.values[_ri, _ci]
        _nt = int(interaction_scores_df['testable'].sum())
        _say(f"  - {_nt:,} of {len(interaction_scores_df):,} interactions "
              f"clear the detection floor in both cell types and were tested")

    # --- 5. FDR Correction per Cell-Type Pair ---
    _say("\n--- Step 5: Applying FDR (Benjamini/Hochberg) correction per cell-type pair ---")
    
    # Initialize the new column with NaNs
    interaction_scores_df['p_value_fdr'] = np.nan
    _floor_limited_groups = []

    # Iterate through each group to apply FDR and assign back correctly
    for _, group_df in tqdm(interaction_scores_df.groupby(['sender', 'receiver']), desc="FDR Correction", disable=not verbose):
        
        p_values = group_df['p_value']
        if p_values.empty or p_values.isnull().all():
            continue
            
            
        if prefilter_fdr:
            if prefilter_threshold > 0:
                import warnings
                warnings.warn(
                    f"runSCALAR: prefilter_threshold={prefilter_threshold} > 0 "
                    f"filters on the test statistic itself before BH — the "
                    f"q-values among survivors are not formally "
                    f"FDR-controlling. Threshold 0 (the default) only removes "
                    f"never-rejectable tests.", stacklevel=2)
            # Create a mask for interactions that pass the threshold (and,
            # under a detection floor, that were tested at all)
            filter_mask = ((group_df['interaction_score'] > prefilter_threshold)
                           & group_df['testable'])
            p_values_to_test = p_values[filter_mask]

            if not p_values_to_test.empty:
                # Apply FDR correction only to the filtered set
                _, p_values_corrected, _, _ = multipletests(p_values_to_test, method='fdr_bh')
                interaction_scores_df.loc[p_values_to_test.index, 'p_value_fdr'] = p_values_corrected
                # Minimum achievable FDR: an isolated hit at the exact floor
                # p = 1/(k^2+1), BH-adjusted in a group of m tests, gets
                # m/(k^2+1). Collected here and reported ONCE after the loop:
                # on a full LR panel every one of the 576 sender-receiver
                # groups tests ~2,500 interactions, and 576 near-identical
                # warnings are a wall of text no one reads.
                _m = len(p_values_to_test)
                if _m / (n_nearest_neighbors ** 2 + 1) > 0.05:
                    _floor_limited_groups.append(
                        (group_df['sender'].iloc[0],
                         group_df['receiver'].iloc[0], _m))
            
            # For interactions that were filtered out, set their FDR to 1.0
            interaction_scores_df.loc[p_values[~filter_mask].index, 'p_value_fdr'] = 1.0
        
        else: # Original behavior if pre-filtering is off
            _t = group_df['testable'].values
            if _t.any():
                _, p_values_corrected, _, _ = multipletests(p_values[_t], method='fdr_bh')
                interaction_scores_df.loc[group_df.index[_t], 'p_value_fdr'] = p_values_corrected
            interaction_scores_df.loc[group_df.index[~_t], 'p_value_fdr'] = 1.0

    # Clip FDR p-values to be at most 1.0 to handle potential floating point inaccuracies.
    interaction_scores_df['p_value_fdr'] = interaction_scores_df['p_value_fdr'].clip(upper=1.0)

    # Calculate the -log10 of the FDR p-value
    interaction_scores_df['nlog10_p_value_fdr'] = -np.log10(interaction_scores_df['p_value_fdr'] + 1e-10)
    
    # Correct any small negative values resulting from p-values of 1.0
    interaction_scores_df['nlog10_p_value_fdr'] = interaction_scores_df['nlog10_p_value_fdr'].clip(lower=0)


    if _floor_limited_groups:
        import warnings
        _worst = max(m for _, _, m in _floor_limited_groups)
        warnings.warn(
            f"runSCALAR: in {len(_floor_limited_groups)} of "
            f"{len(grouped_interactions)} sender-receiver groups the minimum "
            f"achievable FDR for an ISOLATED hit exceeds 0.05 (up to "
            f"{_worst}/{n_nearest_neighbors ** 2 + 1} = "
            f"{_worst / (n_nearest_neighbors ** 2 + 1):.2f}). Interactions "
            f"that share low p-values are unaffected; a lone novel hit in "
            f"those groups cannot clear FDR 0.05 at this floor — raise "
            f"n_nearest_neighbors to lower it.", stacklevel=2)

    # --- 5b. Diagnostics on the finished table ---
    # Breadth: fraction of tested (sender, receiver) pairs in which this LR
    # pair is called at FDR < 0.05. Real cell-type-specific results are
    # narrow; an interaction called across most of the grid is a ubiquity
    # signature, whatever its p-value says.
    # over the groups where the pair was tested (cleared the detection floor):
    # the denominator counted every group, tested or not
    tested = interaction_scores_df['testable'].astype(bool)
    called = (interaction_scores_df['p_value_fdr'] < 0.05) & tested
    _keys = [interaction_scores_df['ligand'], interaction_scores_df['receptor']]
    n_groups_tested = tested.groupby(_keys).transform('sum')
    n_groups_called = called.groupby(_keys).transform('sum')
    interaction_scores_df['call_breadth'] = (
        n_groups_called / n_groups_tested.where(n_groups_tested > 0)).astype(float)

    saturated = interaction_scores_df.loc[
        (interaction_scores_df['p_value'] < 0.05)
        & (interaction_scores_df['null_matchability'] >= 0.99)]
    if len(saturated):
        import warnings
        _genes = sorted(set(saturated['ligand']) | set(saturated['receptor']))
        warnings.warn(
            f"runSCALAR: {len(saturated)} significant interaction(s) have "
            f"null_matchability >= 0.99 — their matched sets lie entirely "
            f"below them in mean expression, so they beat their null "
            f"arithmetically rather than biologically and their p-values "
            f"overstate. Genes involved (up to 15): {_genes[:15]}. Treat "
            f"these calls as unquantified.", stacklevel=2)

    # --- 6. Final Sorting ---
    if rank_by_score:
        interaction_scores_df = interaction_scores_df.sort_values(by='interaction_score', ascending=False)
    else:
        # Sort by cell types first, then by score or p-value if desired
        sort_order = ['sender', 'receiver', 'interaction_score']
        interaction_scores_df = interaction_scores_df.sort_values(by=sort_order, ascending=[True, True, False])

    _say("\nAnalysis complete.")
    return interaction_scores_df.reset_index(drop=True)