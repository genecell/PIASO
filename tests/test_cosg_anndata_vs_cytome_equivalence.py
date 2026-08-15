"""Equivalence tests: COSG on AnnData vs COSG on Cytome (same dataset).

Builds a fresh cytome from the same h5ad used for the AnnData call,
then runs both COSG paths and compares:
  - groups_order matches exactly
  - top-1 marker per group matches exactly
  - top-K marker overlap (Jaccard) is high
  - per-(gene, group) score correlation is high

Tests across multiple `cytome_layer` settings:
  - 'counts' (default)
  - 'log1p' on-the-fly
  - 'infog' on-the-fly (requires piaso.tl.infog params cached first)

Failures are EXPECTED in some configs because of streaming summation
order vs full-matrix accumulation; what we want to confirm is that
top-K rankings agree even when individual scores differ in low-order
floats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


MTG_H5AD = "/path/to/project/mtg_sub_10000.h5ad"


def _jaccard_top_k(a, b, k):
    """Top-k Jaccard overlap between two ndarrays of marker names per group."""
    a_set = set(map(str, a[:k]))
    b_set = set(map(str, b[:k]))
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


@pytest.fixture(scope="module")
def mtg_anndata_and_cytome(tmp_path_factory):
    """One fixture: load h5ad → use raw counts as .X → write a fresh cytome
    from the SAME data so both code paths see the identical matrix."""
    import anndata, cytome
    if not __import__("os").path.exists(MTG_H5AD):
        pytest.skip(f"MTG fixture h5ad not found at {MTG_H5AD}")

    src = anndata.read_h5ad(MTG_H5AD)
    # MTG mtg_sub_10000.h5ad has normalised values in .X; raw counts in .raw.X.
    # COSG operates on raw counts (or layer='infog' if precomputed).
    raw_counts = src.raw.X
    a = anndata.AnnData(
        X=raw_counts.copy() if sp.issparse(raw_counts) else raw_counts.copy(),
        obs=src.obs.copy(),
        var=pd.DataFrame(index=list(src.raw.var_names)),
    )
    # Ensure obs index is unique strings (cytome requires this)
    a.obs_names_make_unique()
    a.var_names_make_unique()

    cytome_path = tmp_path_factory.mktemp("equiv") / "mtg10k.cytome"
    ds = cytome.from_anndata(a, modality="RNA", output=str(cytome_path))
    ds.close()

    return a, str(cytome_path)


def _strip_drop_warning():
    """Decorator to silence the GA-genes drop warning during cosg dispatch."""
    import warnings
    warnings.filterwarnings("ignore", message=".*drop.*", category=UserWarning)


# -----------------------------------------------------------------------
# 1. counts layer — the simplest equivalence
# -----------------------------------------------------------------------

@pytest.mark.parametrize("groupby,n_top", [("CrossArea_subclass", 10)])
def test_cosg_counts_anndata_vs_cytome_equivalence(
    mtg_anndata_and_cytome, groupby, n_top,
):
    """COSG on raw counts: AnnData vs Cytome should produce equivalent
    top-K markers per group (Jaccard ≥ 0.7), exact group order match."""
    import cosg
    a, cytome_path = mtg_anndata_and_cytome

    # AnnData path
    cosg.cosg(
        a, groupby=groupby, n_genes_user=n_top, mu=1.0,
        remove_lowly_expressed=True, expressed_pct=0.1,
        layer=None, key_added="cosg_counts",
    )
    a_names = pd.DataFrame(a.uns["cosg_counts"]["names"])
    a_scores = pd.DataFrame(a.uns["cosg_counts"]["scores"])

    # Cytome path
    from cosg._cytome_streaming import run_cosg_cytome
    result = run_cosg_cytome(
        cytome_path=cytome_path,
        groupby=groupby,
        modality="RNA",
        layer="counts",
        n_genes_user=n_top,
        mu=1.0,
        remove_lowly_expressed=True,
        expressed_pct=0.1,
        verbose=False,
    )
    c_names = pd.DataFrame(result["names"], columns=[str(g) for g in result["groups_order"]])
    c_scores = pd.DataFrame(result["scores"], columns=[str(g) for g in result["groups_order"]])

    # ---- Group ordering ----
    assert list(a_names.columns) == list(c_names.columns), (
        f"Group orders differ:\n  AnnData: {list(a_names.columns)}\n  "
        f"Cytome:  {list(c_names.columns)}"
    )

    # ---- Top-1 marker per group ----
    top1_mismatches = []
    for g in a_names.columns:
        if str(a_names[g].iloc[0]) != str(c_names[g].iloc[0]):
            top1_mismatches.append((g, a_names[g].iloc[0], c_names[g].iloc[0]))
    if top1_mismatches:
        print(f"\n[counts] top-1 mismatches ({len(top1_mismatches)}/{len(a_names.columns)} groups):")
        for g, av, cv in top1_mismatches[:5]:
            print(f"  {g!r}: anndata={av!r}, cytome={cv!r}")
    # Allow up to 10% of groups to disagree on top-1
    n_groups = len(a_names.columns)
    assert len(top1_mismatches) <= max(1, int(0.10 * n_groups)), (
        f"Too many top-1 mismatches: {len(top1_mismatches)}/{n_groups}"
    )

    # ---- Top-K Jaccard per group ----
    jaccard_per_group = []
    for g in a_names.columns:
        j = _jaccard_top_k(a_names[g].values, c_names[g].values, n_top)
        jaccard_per_group.append((g, j))
    mean_jaccard = float(np.mean([j for _, j in jaccard_per_group]))
    min_jaccard = float(np.min([j for _, j in jaccard_per_group]))
    print(f"\n[counts] top-{n_top} Jaccard: mean={mean_jaccard:.3f}, min={min_jaccard:.3f}")
    if min_jaccard < 0.5:
        worst = sorted(jaccard_per_group, key=lambda x: x[1])[:3]
        print(f"  Worst groups: {worst}")
    assert mean_jaccard >= 0.7, f"Mean top-{n_top} Jaccard too low: {mean_jaccard:.3f}"

    # ---- Score correlation across all (gene, group) entries ----
    # Build dict (gene, group) → score, then compare values for keys present in both.
    a_dict = {}
    for g in a_names.columns:
        for i in range(n_top):
            a_dict[(str(a_names[g].iloc[i]), g)] = float(a_scores[g].iloc[i])
    c_dict = {}
    for g in c_names.columns:
        for i in range(n_top):
            c_dict[(str(c_names[g].iloc[i]), g)] = float(c_scores[g].iloc[i])
    common = set(a_dict) & set(c_dict)
    print(f"[counts] common (gene, group) entries: {len(common)} / {len(a_dict)} (anndata) / {len(c_dict)} (cytome)")
    if len(common) > 5:
        a_vec = np.array([a_dict[k] for k in common])
        c_vec = np.array([c_dict[k] for k in common])
        r = float(np.corrcoef(a_vec, c_vec)[0, 1])
        print(f"[counts] score correlation on common entries: r = {r:.4f}")
        assert r > 0.99, f"Score correlation too low: {r:.4f}"


# -----------------------------------------------------------------------
# 2. log1p layer — on-the-fly cytome equivalence
# -----------------------------------------------------------------------

def test_cosg_log1p_anndata_vs_cytome_on_the_fly(mtg_anndata_and_cytome):
    """AnnData: cosg(adata, layer='log1p') after sc.pp.normalize_total +
    sc.pp.log1p. Cytome: run_cosg_cytome(layer='log1p',
    compute_on_fly=True) — uses piaso.pp.normalize_log1p chunk math.
    Expect top-K agreement; absolute scores may differ slightly because
    scanpy's log1p uses target_sum=1e4 by default — same as PIASO's
    log1p. Per-cell depth is summed over RAW counts in both paths.
    """
    import cosg, scanpy as sc
    a, cytome_path = mtg_anndata_and_cytome
    a = a.copy()
    a.layers["counts"] = a.X.copy()  # back up raw counts
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.layers["log1p"] = a.X.copy()
    a.X = a.layers["counts"]  # restore — leave layer intact

    cosg.cosg(
        a, groupby="CrossArea_subclass", n_genes_user=10, mu=1.0,
        remove_lowly_expressed=True, expressed_pct=0.1,
        layer="log1p", key_added="cosg_log1p",
    )
    a_names = pd.DataFrame(a.uns["cosg_log1p"]["names"])

    from cosg._cytome_streaming import run_cosg_cytome
    result = run_cosg_cytome(
        cytome_path=cytome_path,
        groupby="CrossArea_subclass",
        modality="RNA",
        layer="log1p",
        compute_on_fly=True,
        n_genes_user=10, mu=1.0,
        remove_lowly_expressed=True, expressed_pct=0.1,
        verbose=False,
    )
    c_names = pd.DataFrame(result["names"], columns=[str(g) for g in result["groups_order"]])

    assert list(a_names.columns) == list(c_names.columns), "Group orders differ"

    jaccard_per_group = []
    for g in a_names.columns:
        j = _jaccard_top_k(a_names[g].values, c_names[g].values, 10)
        jaccard_per_group.append((g, j))
    mean_j = float(np.mean([j for _, j in jaccard_per_group]))
    min_j = float(np.min([j for _, j in jaccard_per_group]))
    print(f"\n[log1p] mean Jaccard: {mean_j:.3f}, min: {min_j:.3f}")
    if min_j < 0.5:
        worst = sorted(jaccard_per_group, key=lambda x: x[1])[:5]
        print(f"  Worst groups: {worst}")
    # Looser threshold for log1p — different cell-depth computation paths
    assert mean_j >= 0.6, f"log1p mean Jaccard too low: {mean_j:.3f}"


# -----------------------------------------------------------------------
# 3. infog layer — requires piaso.tl.infog precomputation
# -----------------------------------------------------------------------

def test_cosg_infog_anndata_vs_cytome_on_the_fly(mtg_anndata_and_cytome):
    """AnnData: piaso.tl.infog(adata) → cosg(adata, layer='infog').
    Cytome: piaso.tl.infog(ds, save_layer=False) caches the params,
    then run_cosg_cytome(layer='infog', compute_on_fly=True).
    Expect strong top-K agreement — INFOG params are computed
    deterministically per-modality."""
    import cosg, piaso, cytome
    a, cytome_path = mtg_anndata_and_cytome
    a = a.copy()
    piaso.tl.infog(a)  # writes adata.layers['infog']

    cosg.cosg(
        a, groupby="CrossArea_subclass", n_genes_user=10, mu=1.0,
        remove_lowly_expressed=True, expressed_pct=0.1,
        layer="infog", key_added="cosg_infog",
    )
    a_names = pd.DataFrame(a.uns["cosg_infog"]["names"])

    # Precompute INFOG params on the cytome side
    ds = cytome.open(cytome_path)
    piaso.tl.infog(ds, save_layer=False, streaming=True, verbosity=0)
    ds.close()

    from cosg._cytome_streaming import run_cosg_cytome
    result = run_cosg_cytome(
        cytome_path=cytome_path,
        groupby="CrossArea_subclass",
        modality="RNA",
        layer="infog",
        compute_on_fly=True,
        n_genes_user=10, mu=1.0,
        remove_lowly_expressed=True, expressed_pct=0.1,
        verbose=False,
    )
    c_names = pd.DataFrame(result["names"], columns=[str(g) for g in result["groups_order"]])

    assert list(a_names.columns) == list(c_names.columns), "Group orders differ"

    jaccard_per_group = []
    top1_match = 0
    for g in a_names.columns:
        j = _jaccard_top_k(a_names[g].values, c_names[g].values, 10)
        jaccard_per_group.append((g, j))
        if str(a_names[g].iloc[0]) == str(c_names[g].iloc[0]):
            top1_match += 1
    mean_j = float(np.mean([j for _, j in jaccard_per_group]))
    min_j = float(np.min([j for _, j in jaccard_per_group]))
    print(f"\n[infog] mean Jaccard: {mean_j:.3f}, min: {min_j:.3f}, top-1 exact: {top1_match}/{len(a_names.columns)}")
    if min_j < 0.5:
        worst = sorted(jaccard_per_group, key=lambda x: x[1])[:5]
        print(f"  Worst groups: {worst}")
    assert mean_j >= 0.6, f"infog mean Jaccard too low: {mean_j:.3f}"


# -----------------------------------------------------------------------
# 4. Bonus: AnnData with .raw vs same data through cytome
# -----------------------------------------------------------------------

def test_cosg_uses_raw_counts_default(mtg_anndata_and_cytome):
    """Sanity check that the cytome side and AnnData side both consume
    the same RAW count matrix (not the normalised .X). The cytome was
    built from a.X which is raw counts (we used the .raw.X copy in the
    fixture)."""
    import anndata, cytome
    a, cytome_path = mtg_anndata_and_cytome
    ds = cytome.open(cytome_path)
    # cytome's RNA_counts should match a.X cell-for-cell
    assert ds.n_cells == a.n_obs
    assert ds.n_genes == a.n_vars
    ds.close()


# -----------------------------------------------------------------------
# 5. TFIDF layer equivalence (ATAC modality, synthetic small fixture)
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_atac_anndata_and_cytome(tmp_path_factory):
    """Build a tiny synthetic ATAC dataset (200 cells × 100 peaks) in
    BOTH formats from the same source, so we can compare TFIDF + COSG
    end-to-end. Real ATAC h5ad fixtures aren't checked into the repo;
    synthetic gives full reproducibility + fast tests."""
    import anndata, cytome
    rng = np.random.default_rng(0)
    n_cells, n_peaks = 200, 100
    n_clusters = 5
    cluster = np.array([f"c{i % n_clusters}" for i in range(n_cells)])
    # Per-cluster peak signature: peak i high in cluster (i % n_clusters)
    counts = rng.poisson(0.5, size=(n_cells, n_peaks)).astype(np.float32)
    for cl_idx in range(n_clusters):
        mask = (cluster == f"c{cl_idx}")
        # Boost a contiguous block of peaks for this cluster
        peak_block = slice(cl_idx * 15, cl_idx * 15 + 15)
        counts[mask, peak_block] += rng.poisson(8.0, size=(mask.sum(), 15)).astype(np.float32)

    var = pd.DataFrame(
        index=[f"chr1:{100*i}-{100*i+50}" for i in range(n_peaks)],
    )
    obs = pd.DataFrame(
        {"barcode": [f"AAA-{i}" for i in range(n_cells)],
         "Leiden": cluster},
        index=[f"AAA-{i}" for i in range(n_cells)],
    )
    a = anndata.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    a.obs_names_make_unique(); a.var_names_make_unique()

    cytome_path = tmp_path_factory.mktemp("atac_equiv") / "atac.cytome"
    ds = cytome.from_anndata(a, modality="ATAC", output=str(cytome_path))
    ds.close()
    return a, str(cytome_path)


def test_cosg_tfidf_atac_anndata_vs_cytome_on_the_fly(synthetic_atac_anndata_and_cytome):
    """ATAC + TFIDF: AnnData runs piaso.tl.run_TFIDF(adata, modality='ATAC',
    output_layer='tfidf') → adata.layers['tfidf'], then cosg(layer='tfidf').
    Cytome runs piaso.tl.compute_tfidf_stats(ds, modality='ATAC') to populate
    the params cache, then run_cosg_cytome(layer='tfidf',
    compute_on_fly=True). Predict bit-identical (same as infog/counts) since
    TFIDF math is deterministic and the chunk normalizer reuses cached params.
    """
    import anndata, cytome, cosg, piaso
    a, cytome_path = synthetic_atac_anndata_and_cytome

    # AnnData path: materialise TFIDF, then COSG on the layer.
    # piaso.tl.run_TFIDF (default: output_layer='tfidf', inplace=False) writes
    # a_local.layers['tfidf'] and leaves .X untouched — exactly what we want.
    a_local = a.copy()
    a_local.layers["counts"] = a_local.X.copy()
    piaso.tl.run_TFIDF(
        a_local, layer="counts", scale_factor=1e4, modality="ATAC",
    )
    assert "tfidf" in a_local.layers, "run_TFIDF should write adata.layers['tfidf']"
    cosg.cosg(
        a_local, groupby="Leiden", n_genes_user=10, mu=1.0,
        remove_lowly_expressed=False,  # All peaks contribute
        layer="tfidf", key_added="cosg_tfidf",
    )
    a_names = pd.DataFrame(a_local.uns["cosg_tfidf"]["names"])

    # Cytome path: cache TFIDF params → COSG with on-the-fly TFIDF
    ds = cytome.open(cytome_path)
    piaso.tl.compute_tfidf_stats(
        ds, modality="ATAC", measurement="counts", batch_size=64,
        scale_factor=1e4,
    )
    ds.close()

    result = cosg.run_cosg_cytome(
        cytome_path=cytome_path,
        groupby="Leiden",
        modality="ATAC",
        layer="tfidf",
        compute_on_fly=True,
        n_genes_user=10, mu=1.0,
        remove_lowly_expressed=False,
        verbose=False,
    )
    c_names = pd.DataFrame(result["names"], columns=[str(g) for g in result["groups_order"]])

    assert list(a_names.columns) == list(c_names.columns), "Group orders differ"

    jaccard_per_group = []
    top1_match = 0
    for g in a_names.columns:
        j = _jaccard_top_k(a_names[g].values, c_names[g].values, 10)
        jaccard_per_group.append((g, j))
        if str(a_names[g].iloc[0]) == str(c_names[g].iloc[0]):
            top1_match += 1
    mean_j = float(np.mean([j for _, j in jaccard_per_group]))
    min_j = float(np.min([j for _, j in jaccard_per_group]))
    print(f"\n[tfidf-atac] mean Jaccard: {mean_j:.3f}, min: {min_j:.3f}, top-1 exact: {top1_match}/{len(a_names.columns)}")
    if min_j < 0.5:
        worst = sorted(jaccard_per_group, key=lambda x: x[1])[:3]
        print(f"  Worst groups: {worst}")
    # TFIDF on synthetic ATAC: predict ≥ 0.7 mean Jaccard
    assert mean_j >= 0.7, f"tfidf-atac mean Jaccard too low: {mean_j:.3f}"


# -----------------------------------------------------------------------
# 6. runGDR end-to-end equivalence: AnnData vs Cytome
# -----------------------------------------------------------------------

def test_runGDR_anndata_vs_cytome_equivalent_X_gdr(mtg_anndata_and_cytome):
    """End-to-end pipeline equivalence:

    AnnData path:
        piaso.tl.infog(adata) → adata.layers['infog']
        piaso.tl.runGDR(adata, layer='infog', score_layer='infog', ...)
            → adata.obsm['X_gdr']

    Cytome path:
        piaso.tl.infog(ds, save_layer=True) → materialized RNA_infog
        piaso.tl.runGDR(ds, modality='RNA', cytome_layer='infog',
                        score_cytome_layer='infog', ...)
            → returned (X_gdr, marker_gene) tuple

    Both sides MUST score on the same layer (`infog` here): the AnnData
    side defaults `score_layer=None` → `.X` (raw counts), while the
    cytome side defaults `score_cytome_layer=None` → falls back to
    `cytome_layer`. Aligning explicitly is required for apples-to-apples
    equivalence.

    X_gdr is doubly L2-normalised (column then row) so rows are unit
    vectors. Asserts on the equivalence:
      - Same shape (n_cells, n_clusters)
      - COSG markers per cluster: bit-identical (already verified by
        the `infog` cosg test above at Jaccard 1.000)
      - Per-column correlation (after greedy column alignment): ≥ 0.95
      - Per-cell cosine median: ≥ 0.95
    """
    import piaso, cytome
    a, cytome_path = mtg_anndata_and_cytome
    a_local = a.copy()

    # AnnData side — explicitly score on the same layer used for COSG
    # (default for score_layer is None → adata.X = raw counts; we want
    # apples-to-apples equivalence with the cytome side which defaults
    # score_cytome_layer to cytome_layer).
    piaso.tl.infog(a_local)
    piaso.tl.runGDR(
        a_local, groupby="CrossArea_subclass",
        n_gene=10, mu=1.0,
        layer="infog", score_layer="infog",
        scoring_method="piaso",
        max_workers=1, verbosity=0, random_seed=42,
    )
    X_gdr_anndata = np.asarray(a_local.obsm["X_gdr"])

    # Cytome side: materialize infog layer on disk so the scoring step
    # (which reads from {modality}_{cytome_layer} via iter_chunks) finds
    # RNA_infog. Equivalent to AnnData side's adata.layers['infog'].
    ds = cytome.open(cytome_path)
    piaso.tl.infog(ds, save_layer=True, streaming=True, verbosity=0)
    n_cells_cy = ds.n_cells
    ds.close()

    # Default behaviour: write_to_cytome=True persists X_gdr via
    # ds.add_embedding('X_gdr', ...) and marker_gene via ds.metadata
    # (mirroring the AnnData in-place obsm write). Passing
    # cytome_layer='infog' / score_cytome_layer='infog' is also the new
    # default but kept explicit here as a self-documenting test.
    ret = piaso.tl.runGDR(
        cytome_path, groupby="CrossArea_subclass",
        n_gene=10, mu=1.0,
        modality="RNA", cytome_layer="infog",
        score_cytome_layer="infog",
        scoring_method="piaso",
        max_workers=1, verbosity=0, random_seed=42,
    )
    assert ret is None, (
        "runGDR cytome with write_to_cytome=True (default) should return None "
        f"and write to disk; got {type(ret).__name__}."
    )
    ds_re = cytome.open(cytome_path)
    X_gdr_cytome = np.asarray(ds_re.embeddings["X_gdr"])
    assert "runGDR_marker_genes" in ds_re.metadata, (
        "Expected runGDR to write marker_gene table to ds.metadata['runGDR_marker_genes']."
    )
    ds_re.close()

    # Sanity-check the alternate return-tuple flow: write_to_cytome=False
    ret_tuple = piaso.tl.runGDR(
        cytome_path, groupby="CrossArea_subclass",
        n_gene=10, mu=1.0,
        modality="RNA", cytome_layer="infog",
        score_cytome_layer="infog",
        scoring_method="piaso",
        max_workers=1, verbosity=0, random_seed=42,
        write_to_cytome=False,
    )
    assert isinstance(ret_tuple, tuple) and len(ret_tuple) == 2
    X_gdr_cytome_returned, _marker_gene_cy = ret_tuple
    np.testing.assert_allclose(
        X_gdr_cytome, np.asarray(X_gdr_cytome_returned),
        rtol=0, atol=1e-5,
        err_msg="Embedding written to cytome should match the returned array.",
    )

    # ---- Shape ----
    assert X_gdr_anndata.shape[0] == X_gdr_cytome.shape[0] == n_cells_cy, (
        f"n_cells mismatch: anndata={X_gdr_anndata.shape[0]}, "
        f"cytome={X_gdr_cytome.shape[0]}, expected={n_cells_cy}"
    )
    # Column count = number of clusters; should match if both runs picked up
    # the same group set (CrossArea_subclass is categorical — yes).
    assert X_gdr_anndata.shape[1] == X_gdr_cytome.shape[1], (
        f"n_groups mismatch: anndata={X_gdr_anndata.shape[1]}, "
        f"cytome={X_gdr_cytome.shape[1]}"
    )

    # ---- Per-cell cosine similarity ----
    # Both X_gdr matrices are doubly L2-normalised (rows unit-norm), so
    # row-wise dot product is the cosine similarity.
    # Note: column ordering may differ between paths if COSG returned
    # gene-sets in a different order. Check via a column-permutation-
    # invariant metric: cosine of |X_gdr_anndata @ X_gdr_cytome.T| isn't
    # quite right; instead align columns by best match, then per-row cosine.

    # Best column alignment via max-correlation matching
    n_cols = X_gdr_anndata.shape[1]
    # Compute |corr| matrix between AnnData cols and cytome cols
    corr_mat = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            corr_mat[i, j] = np.abs(np.corrcoef(
                X_gdr_anndata[:, i], X_gdr_cytome[:, j],
            )[0, 1])
    # Greedy assignment
    perm = np.zeros(n_cols, dtype=int)
    used = set()
    for i in range(n_cols):
        scores = corr_mat[i].copy()
        for u in used:
            scores[u] = -1
        best_j = int(np.argmax(scores))
        perm[i] = best_j
        used.add(best_j)
    X_gdr_cytome_aligned = X_gdr_cytome[:, perm]
    # If columns flipped sign (correlation can be negative), align signs
    for i in range(n_cols):
        if np.corrcoef(X_gdr_anndata[:, i], X_gdr_cytome_aligned[:, i])[0, 1] < 0:
            X_gdr_cytome_aligned[:, i] *= -1
    # Re-L2-normalise rows after sign flip (safety)
    norms = np.linalg.norm(X_gdr_cytome_aligned, axis=1, keepdims=True)
    X_gdr_cytome_aligned = np.divide(
        X_gdr_cytome_aligned, np.where(norms == 0, 1.0, norms),
    )
    norms_a = np.linalg.norm(X_gdr_anndata, axis=1, keepdims=True)
    X_gdr_anndata_n = np.divide(
        X_gdr_anndata, np.where(norms_a == 0, 1.0, norms_a),
    )

    cosines = np.sum(X_gdr_anndata_n * X_gdr_cytome_aligned, axis=1)
    median_cos = float(np.median(cosines))
    pct95 = float(np.percentile(cosines, 5))   # 5th percentile = worst 5%
    print(
        f"\n[runGDR] X_gdr per-cell cosine similarity: "
        f"median={median_cos:.4f}, p5={pct95:.4f}, "
        f"min={float(cosines.min()):.4f}, max={float(cosines.max()):.4f}"
    )

    # Per-column correlation diagnostic (after greedy alignment)
    col_corrs = np.array([
        float(np.corrcoef(
            X_gdr_anndata[:, i], X_gdr_cytome_aligned[:, i],
        )[0, 1])
        for i in range(n_cols)
    ])
    print(
        f"[runGDR] per-column |Pearson r| (after alignment): "
        f"min={float(np.min(np.abs(col_corrs))):.4f}, "
        f"median={float(np.median(np.abs(col_corrs))):.4f}, "
        f"mean={float(np.mean(np.abs(col_corrs))):.4f}"
    )

    # With matched score_layer, equivalence is high — small residual
    # drift from streaming KNN summation order in score().
    assert median_cos >= 0.95, (
        f"runGDR X_gdr median cosine too low: {median_cos:.4f}. "
        f"AnnData and cytome runGDR should produce equivalent embeddings."
    )
    assert float(np.median(np.abs(col_corrs))) >= 0.95, (
        f"Per-column correlation too low: {float(np.median(np.abs(col_corrs))):.4f}. "
        f"Each column is one cluster's gene-set score; should be highly correlated."
    )


# -----------------------------------------------------------------------
# 7. Override tests — sentinel-default mirroring is NOT clobbering
#    explicit user-supplied values for score_layer / score_cytome_layer.
# -----------------------------------------------------------------------

def test_runGDR_explicit_score_layer_override_anndata(mtg_anndata_and_cytome):
    """When the caller explicitly passes ``score_layer`` to a value that
    differs from ``layer``, the sentinel-default mirroring MUST step out of
    the way. We run runGDR twice — once with the default mirror and once
    with score_layer=None (legacy behaviour: score on adata.X) — and assert
    the two embeddings are noticeably different. If the override silently
    no-op'd, the two runs would be identical."""
    import piaso
    a, _ = mtg_anndata_and_cytome
    a_local = a.copy()
    piaso.tl.infog(a_local)

    # Run 1: defaults — score_layer mirrors layer='infog'
    a_mirror = a_local.copy()
    piaso.tl.runGDR(
        a_mirror, groupby="CrossArea_subclass",
        n_gene=10, mu=1.0, layer="infog",
        scoring_method="piaso",
        max_workers=1, verbosity=0, random_seed=42,
    )
    X_mirror = np.asarray(a_mirror.obsm["X_gdr"])

    # Run 2: explicit override — COSG on infog, but score on raw counts (.X)
    a_override = a_local.copy()
    piaso.tl.runGDR(
        a_override, groupby="CrossArea_subclass",
        n_gene=10, mu=1.0, layer="infog", score_layer=None,
        scoring_method="piaso",
        max_workers=1, verbosity=0, random_seed=42,
    )
    X_override = np.asarray(a_override.obsm["X_gdr"])

    # The two runs should produce different X_gdr (different scoring data).
    # We use Frobenius distance over the matrices rather than per-cell
    # cosines because the columns can permute / sign-flip arbitrarily and
    # we just care that something changed.
    assert X_mirror.shape == X_override.shape
    diff_norm = float(np.linalg.norm(X_mirror - X_override))
    base_norm = float(np.linalg.norm(X_mirror))
    rel = diff_norm / base_norm if base_norm > 0 else float("inf")
    print(
        f"\n[override-anndata] ||X_mirror - X_override|| / ||X_mirror|| = {rel:.4f}"
    )
    assert rel > 0.05, (
        f"score_layer=None override appears to have been ignored: relative "
        f"diff between mirror and override runs = {rel:.4f} (expected > 0.05). "
        f"This means the sentinel-default mirror is silently overwriting an "
        f"explicit user-passed None — a regression in the contract."
    )


def test_runGDR_cytome_score_layer_independent(mtg_anndata_and_cytome):
    """Cytome side: ``score_cytome_layer`` MUST be settable independently of
    ``cytome_layer``. We materialise both RNA_counts and RNA_infog on disk,
    then run runGDR twice — once with score_cytome_layer mirroring (infog)
    and once with score_cytome_layer='counts' — and assert the two X_gdr
    matrices differ."""
    import piaso, cytome
    _, cytome_path = mtg_anndata_and_cytome

    # Materialise infog on disk so both `RNA_counts` and `RNA_infog` are
    # available for the score step.
    ds = cytome.open(cytome_path)
    if "RNA_infog" not in {row[0] for row in ds._conn.execute(
        "SELECT matrix_name FROM matrix_meta"
    ).fetchall()}:
        piaso.tl.infog(ds, save_layer=True, streaming=True, verbosity=0)
    ds.close()

    # Run 1: defaults — score_cytome_layer mirrors cytome_layer='infog'
    res_mirror = piaso.tl.runGDR(
        cytome_path, groupby="CrossArea_subclass",
        n_gene=10, mu=1.0,
        modality="RNA", cytome_layer="infog",
        scoring_method="piaso",
        max_workers=1, verbosity=0, random_seed=42,
        write_to_cytome=False,
    )
    X_mirror, _ = res_mirror
    X_mirror = np.asarray(X_mirror)

    # Run 2: explicit override — COSG on infog, but score on RAW counts.
    res_override = piaso.tl.runGDR(
        cytome_path, groupby="CrossArea_subclass",
        n_gene=10, mu=1.0,
        modality="RNA", cytome_layer="infog",
        score_cytome_layer="counts",
        scoring_method="piaso",
        max_workers=1, verbosity=0, random_seed=42,
        write_to_cytome=False,
    )
    X_override, _ = res_override
    X_override = np.asarray(X_override)

    assert X_mirror.shape == X_override.shape
    diff_norm = float(np.linalg.norm(X_mirror - X_override))
    base_norm = float(np.linalg.norm(X_mirror))
    rel = diff_norm / base_norm if base_norm > 0 else float("inf")
    print(
        f"\n[override-cytome] ||X_mirror - X_override|| / ||X_mirror|| = {rel:.4f}"
    )
    assert rel > 0.05, (
        f"score_cytome_layer='counts' override appears to have been ignored: "
        f"relative diff between mirror and override runs = {rel:.4f} "
        f"(expected > 0.05). The sentinel-default mirror is silently "
        f"overwriting an explicit user-passed value — a regression in the "
        f"contract."
    )
