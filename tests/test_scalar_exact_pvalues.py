"""runSCALAR's exact null: enumeration, matching, and the two diagnostics.

The p-value is (#{matched-product support >= observed} + 1) / (k**2 + 1) —
exact under exchangeability of the observed product with its matched
products, which is only true because the matched sets exclude self by
IDENTITY (the shared piaso matcher) and are drawn from the adata ∩
specificity-matrix universe (so no NaN can pad the denominator).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from piaso.tools._ligandReceptor import (
    _exact_exceed_counts,
    _exact_exceed_counts_dense,
    _prepare_background_genes,
    runSCALAR,
)


def _fixture(n_genes=12, n_cells=40, seed=0, extreme=None):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    if extreme is not None:
        X[:, extreme] += 50            # an abundance ceiling, on purpose
    a = ad.AnnData(X=X)
    a.var_names = genes
    spec = pd.DataFrame(
        np.abs(rng.normal(size=(n_genes, 3))),
        index=genes, columns=["A", "B", "C"])
    lr = pd.DataFrame({"ligand": ["G0", "G2", "G4"],
                       "receptor": ["G1", "G3", "G5"]})
    return a, spec, lr


def _run(a, spec, lr, **kw):
    kw.setdefault("n_nearest_neighbors", 3)
    kw.setdefault("expressed_pct", None)      # the unfloored test, as before
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runSCALAR(a, spec, lr, **kw)


# ------------------------------------------------------------- exactness --


def test_pvalues_match_brute_force_enumeration():
    a, spec, lr = _fixture()
    k = 3
    out = _run(a, spec, lr, n_nearest_neighbors=k)

    knn_df, _ = _prepare_background_genes(
        a, n_nearest_neighbors=k,
        genes_universe=list(a.var_names),
        query_genes=list(lr["ligand"]) + list(lr["receptor"]))
    for _, row in out.iterrows():
        lig_scores = spec[row["sender"]].reindex(knn_df.index).values[
            knn_df.loc[row["ligand"]].values]
        rec_scores = spec[row["receiver"]].reindex(knn_df.index).values[
            knn_df.loc[row["receptor"]].values]
        support = np.multiply.outer(lig_scores, rec_scores).ravel()
        want = (int((support >= row["interaction_score"]).sum()) + 1) / (k * k + 1)
        assert row["p_value"] == pytest.approx(want), (
            f"{row['ligand']}~{row['receptor']} in "
            f"({row['sender']},{row['receiver']})")


@pytest.mark.parametrize("seed", range(5))
def test_searchsorted_path_equals_the_outer_product(seed):
    """The two implementations check each other — a property, not an example."""
    rng = np.random.default_rng(seed)
    n, k = 37, 8
    lig = np.abs(rng.normal(size=(n, k)))
    rec = np.abs(rng.normal(size=(n, k)))
    lig[rng.random((n, k)) < 0.2] = 0.0        # zeros are an edge case
    obs = np.abs(rng.normal(size=n))
    obs[:5] = 0.0                              # and so is obs <= 0
    assert (_exact_exceed_counts(lig, rec, obs)
            == _exact_exceed_counts_dense(lig, rec, obs)).all()


def test_negative_scores_are_an_error_not_a_slow_path():
    """COSG's -1 sentinel for filtered genes multiplies to +1 with another
    sentinel - the largest product in the table. Negatives never reach the
    counter legitimately, so it raises instead of silently enumerating."""
    rng = np.random.default_rng(1)
    lig = rng.normal(size=(9, 4))              # negatives present
    rec = rng.normal(size=(9, 4))
    obs = rng.normal(size=9)
    with pytest.raises(ValueError, match="non-negative"):
        _exact_exceed_counts(lig, rec, obs)


# -------------------------------------------------------------- matching --


def test_twin_genes_never_match_themselves():
    """Duplicate (mean, var) stats + arbitrary tie order was how a gene could
    slip into its own matched set under positional self-exclusion."""
    a = ad.AnnData(X=np.ones((10, 6), dtype=np.float32))   # all stats identical
    a.var_names = [f"G{i}" for i in range(6)]
    knn_df, _ = _prepare_background_genes(a, n_nearest_neighbors=3)
    for pos, gene in enumerate(knn_df.index):
        assert pos not in knn_df.loc[gene].values, f"{gene} matched itself"


def test_genes_outside_the_specificity_matrix_change_nothing():
    """The NaN-poisoning fix: adata-only genes must not enter the universe,
    where they would deflate every p-value they were matched into."""
    a, spec, lr = _fixture(n_genes=12)
    out1 = _run(a, spec, lr)

    rng = np.random.default_rng(7)
    extra = pd.DataFrame(
        rng.poisson(2.0, size=(a.n_obs, 4)).astype(np.float32),
        index=a.obs_names, columns=[f"X{i}" for i in range(4)])
    a2 = ad.AnnData(X=np.hstack([a.X, extra.values]))
    a2.var_names = list(a.var_names) + list(extra.columns)
    out2 = _run(a2, spec, lr)

    m1 = out1.set_index(["ligand", "receptor", "sender", "receiver"])["p_value"]
    m2 = out2.set_index(["ligand", "receptor", "sender", "receiver"])["p_value"]
    pd.testing.assert_series_equal(m1.sort_index(), m2.sort_index())


# ----------------------------------------------------------- diagnostics --


def test_extreme_gene_is_reported_saturated_and_warned_about():
    # k must be large enough that p < 0.05 is reachable at all: the floor is
    # 1/(k**2+1), so k=3 (floor 0.1) can never trigger a significance-gated
    # warning — k=8 gives 1/65.
    a, spec, lr = _fixture(n_genes=30, extreme=0)   # G0 towers over everything
    spec.loc["G0", :] = spec.values.max() * 10      # and its null cannot keep up
    with pytest.warns(UserWarning, match="null_matchability >= 0.99"):
        out = runSCALAR(a, spec, lr, n_nearest_neighbors=8, expressed_pct=None)
    g0 = out[out["ligand"] == "G0"]
    assert (g0["null_matchability"] >= 0.99).all()
    assert out["null_matchability"].between(0, 1).all()


def test_breadth_is_the_fraction_of_groups_called():
    a, spec, lr = _fixture()
    out = _run(a, spec, lr)
    assert out["call_breadth"].between(0, 1).all()
    for (lg, rc), grp in out.groupby(["ligand", "receptor"]):
        want = (grp["p_value_fdr"] < 0.05).mean()
        assert grp["call_breadth"].unique().tolist() == [pytest.approx(want)]


# ------------------------------------------------------------- contract --


def test_deterministic_across_seeds():
    a, spec, lr = _fixture()
    out1 = _run(a, spec, lr, random_seed=1)
    out2 = _run(a, spec, lr, random_seed=999)
    pd.testing.assert_frame_equal(out1, out2)


def test_n_permutations_warns_and_changes_nothing():
    a, spec, lr = _fixture()
    out1 = _run(a, spec, lr)
    with pytest.warns(UserWarning, match="n_permutations is ignored"):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            out2 = runSCALAR(a, spec, lr, n_nearest_neighbors=3,
                             n_permutations=5000, expressed_pct=None)
    pd.testing.assert_frame_equal(out1, out2)


def test_global_rng_is_left_alone():
    a, spec, lr = _fixture()
    np.random.seed(123)
    before = np.random.get_state()[1][:5].copy()
    _run(a, spec, lr, random_seed=42)
    after = np.random.get_state()[1][:5]
    assert (before == after).all(), "runSCALAR mutated the global NumPy RNG"


def test_floor_is_the_support_size():
    a, spec, lr = _fixture()
    k = 3
    out = _run(a, spec, lr, n_nearest_neighbors=k)
    assert (out["p_value"] >= 1 / (k * k + 1) - 1e-12).all()
    assert (out["p_value"] <= 1.0).all()


def test_a_named_specificity_index_works():
    """Any spec matrix built from adata.var_names carries an index NAME;
    reset_index() then names the column after it and the old
    rename(columns={'index': ...}) silently missed, crashing melt with
    KeyError('ligand') — found live on the MTG validation run."""
    a, spec, lr = _fixture()
    spec.index.name = "feature_name"
    out = _run(a, spec, lr)
    assert len(out) > 0 and "p_value" in out.columns


# ------------------------------------------------------------ cytome input --


def _as_cytome(a, path):
    """Write the fixture AnnData to a cytome with an RNA counts matrix."""
    import cytome
    import scipy.sparse as sp

    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(a.n_obs),
        "barcode": [f"c{i}" for i in range(a.n_obs)]}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(a.n_vars),
        "gene_id": list(a.var_names)}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.asarray(a.X, dtype=np.float32)))
    ds.flush()
    ds.close()
    return str(path)


def test_a_cytome_and_an_anndata_of_the_same_data_agree(tmp_path):
    """The matching space is built by streaming for a cytome and in memory for
    an AnnData; both must produce the same statistics, so the same p-values."""
    a, spec, lr = _fixture(n_genes=14, n_cells=60, seed=3)
    cy = _as_cytome(a, tmp_path / "s.cytome")

    out_mem = _run(a, spec, lr, n_nearest_neighbors=4)
    out_cy = _run(cy, spec, lr, n_nearest_neighbors=4)

    key = ["ligand", "receptor", "sender", "receiver"]
    m = out_mem.set_index(key)[["interaction_score", "p_value", "null_matchability"]].sort_index()
    c = out_cy.set_index(key)[["interaction_score", "p_value", "null_matchability"]].sort_index()
    pd.testing.assert_frame_equal(m, c, rtol=1e-9)


def test_cytome_streaming_is_chunk_size_invariant(tmp_path):
    """Column sums accumulated over chunks must not depend on the chunk size —
    the classic streaming-statistics failure."""
    from piaso.tools._ligandReceptor import _scalar_gene_stats

    a, spec, lr = _fixture(n_genes=14, n_cells=60, seed=4)
    cy = _as_cytome(a, tmp_path / "s.cytome")
    universe = set(a.var_names)
    ref = _scalar_gene_stats(a, None, universe)          # in-memory
    for bs in (7, 16, 1000):
        got = _scalar_gene_stats(cy, None, universe, batch_size=bs)
        assert list(got[0]) == list(ref[0])
        # Both paths accumulate in float64 (scipy's sparse .mean() would
        # accumulate a float32 matrix in float32 and lose ~1e-7 here), so the
        # streaming result matches the in-memory one to float64 precision at
        # every chunk size.
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(got[2], ref[2], rtol=1e-12, atol=1e-12)
