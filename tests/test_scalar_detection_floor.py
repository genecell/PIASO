"""runSCALAR's detection floor (``expressed_pct``) and its input contract.

The floor lives in the test, not in the matrix: an interaction is tested only
if the ligand is detected in >= expressed_pct of the sender's cells and the
receptor in the receiver's, and each gene's matched controls are drawn from
genes that clear the same floor in the same cell type. So the k x k support
is always made of real scores, and zeroing lowly-detected genes in the input
changes nothing - which is the property that lets a COSG matrix built with or
without COSG's own filter give the same answer.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from piaso.tools._ligandReceptor import (
    _prepare_background_genes,
    _scalar_detection_fraction,
    _scalar_gene_stats,
    runSCALAR,
)

TYPES = ["A", "B", "C"]


def _fixture(n_genes=16, n_cells=90, seed=0, sparse_gene="G6", sparse_in="A"):
    """Three labelled groups; ``sparse_gene`` is silenced in 95 % of
    ``sparse_in``'s cells so it fails a 10 % floor there and nowhere else."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    labels = np.array(TYPES * (n_cells // 3))
    a = ad.AnnData(X=X)
    a.var_names = genes
    a.obs["ct"] = pd.Categorical(labels)
    rows = np.flatnonzero(labels == sparse_in)
    X[rows[: int(0.95 * len(rows))], genes.index(sparse_gene)] = 0.0
    a.X = X
    spec = pd.DataFrame(np.abs(rng.normal(size=(n_genes, 3))),
                        index=genes, columns=TYPES)
    lr = pd.DataFrame({"ligand": ["G0", "G2", "G6", "G8"],
                       "receptor": ["G1", "G3", "G7", "G6"]})
    return a, spec, lr


def _run(a, spec, lr, **kw):
    kw.setdefault("n_nearest_neighbors", 3)
    kw.setdefault("groupby", "ct")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runSCALAR(a, spec, lr, **kw)


KEY = ["ligand", "receptor", "sender", "receiver"]


# ------------------------------------------------------------ the floor --


def test_detection_fraction_is_the_nonzero_fraction_per_group():
    a, _, _ = _fixture()
    det = _scalar_detection_fraction(a, None, "ct", list(a.var_names))
    for t in TYPES:
        want = (a.X[(a.obs["ct"] == t).values] != 0).mean(axis=0)
        np.testing.assert_allclose(det[t].values, want)
    assert det.loc["G6", "A"] < 0.1 and det.loc["G6", "B"] > 0.5


def test_sparse_gene_is_untestable_only_where_it_fails_the_floor():
    a, spec, lr = _fixture()
    out = _run(a, spec, lr, expressed_pct=0.1).set_index(KEY)
    g6_as_ligand_from_A = out.xs(("G6", "G7", "A"), level=KEY[:3])
    assert (~g6_as_ligand_from_A["testable"]).all()
    assert (g6_as_ligand_from_A["p_value"] == 1.0).all()
    assert (g6_as_ligand_from_A["p_value_fdr"] == 1.0).all()
    g6_as_ligand_from_B = out.xs(("G6", "G7", "B"), level=KEY[:3])
    assert g6_as_ligand_from_B["testable"].all()
    # and as a receptor, the same rule applies on the receiver side
    into_A = out.xs(("G8", "G6"), level=KEY[:2]).xs("A", level="receiver")
    assert (~into_A["testable"]).all()
    # the fractions the floor was applied to travel with the row
    assert (out.loc[("G6", "G7", "A", "B"), "ligand_expressed_pct"] < 0.1)
    assert (out["ligand_expressed_pct"].between(0, 1).all()
            and out["receptor_expressed_pct"].between(0, 1).all())


def test_floor_zero_reproduces_the_unfloored_test():
    """At expressed_pct=0 every gene clears the floor everywhere, so the
    per-type universes are the common universe and p-values are identical."""
    a, spec, lr = _fixture()
    ref = _run(a, spec, lr, expressed_pct=None).set_index(KEY)
    got = _run(a, spec, lr, expressed_pct=0.0).set_index(KEY)
    pd.testing.assert_series_equal(ref["p_value"].sort_index(),
                                   got["p_value"].sort_index())
    pd.testing.assert_series_equal(ref["p_value_fdr"].sort_index(),
                                   got["p_value_fdr"].sort_index())
    assert got["testable"].all()


def test_zeroing_lowly_detected_scores_in_the_input_changes_nothing():
    """The floor is applied in the test: a matrix with lowly-detected genes
    zeroed (what COSG's remove_lowly_expressed + indexByGene produce) must give
    the same result as the unfiltered matrix."""
    a, spec, lr = _fixture()
    det = _scalar_detection_fraction(a, None, "ct", list(a.var_names))
    spec_zeroed = spec.where(det.loc[spec.index, spec.columns] >= 0.1, 0.0)
    assert (spec_zeroed == 0).sum().sum() > (spec == 0).sum().sum()
    ref = _run(a, spec, lr, expressed_pct=0.1).set_index(KEY)
    got = _run(a, spec_zeroed, lr, expressed_pct=0.1).set_index(KEY)
    for col in ("p_value", "p_value_fdr", "testable"):
        pd.testing.assert_series_equal(ref[col].sort_index(), got[col].sort_index())


def test_pvalues_match_enumeration_over_the_per_type_universe():
    a, spec, lr = _fixture()
    k = 3
    out = _run(a, spec, lr, expressed_pct=0.1, n_nearest_neighbors=k)
    det = _scalar_detection_fraction(a, None, "ct", list(a.var_names))
    stats = _scalar_gene_stats(a, None, set(a.var_names))
    bg = {t: _prepare_background_genes(
        a, n_nearest_neighbors=k,
        genes_universe=set(det.index[det[t] >= 0.1]),
        query_genes=list(lr["ligand"]) + list(lr["receptor"]), stats=stats)[0]
        for t in TYPES}
    for _, row in out[out["testable"]].iterrows():
        bs, br = bg[row["sender"]], bg[row["receiver"]]
        lig = spec[row["sender"]].reindex(bs.index).values[bs.loc[row["ligand"]].values]
        rec = spec[row["receiver"]].reindex(br.index).values[br.loc[row["receptor"]].values]
        support = np.multiply.outer(lig, rec).ravel()
        want = (int((support >= row["interaction_score"]).sum()) + 1) / (k * k + 1)
        assert row["p_value"] == pytest.approx(want)


def test_untestable_rows_leave_bh():
    a, spec, lr = _fixture()
    out = _run(a, spec, lr, expressed_pct=0.1)
    for _, grp in out.groupby(["sender", "receiver"]):
        tested = grp[grp["testable"] & (grp["interaction_score"] > 0)]
        if tested.empty:
            continue
        # BH on the tested rows alone reproduces the reported FDR
        from statsmodels.stats.multitest import multipletests
        want = multipletests(tested["p_value"], method="fdr_bh")[1]
        np.testing.assert_allclose(tested["p_value_fdr"].values, np.minimum(want, 1.0))
        assert (grp.loc[~grp["testable"], "p_value_fdr"] == 1.0).all()


def test_too_few_detected_genes_makes_a_type_untestable_with_a_warning():
    a, spec, lr = _fixture()
    with pytest.warns(UserWarning, match="untestable"):
        out = runSCALAR(a, spec, lr, groupby="ct", expressed_pct=0.1,
                        n_nearest_neighbors=20)      # 16 genes < k + 1
    assert (~out["testable"]).all()
    assert (out["p_value"] == 1.0).all()


# ------------------------------------------------------- input contract --


def test_nan_in_the_matrix_raises():
    a, spec, lr = _fixture()
    spec.loc["G2", "B"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _run(a, spec, lr, expressed_pct=None)


def test_negative_sentinel_in_the_matrix_raises():
    a, spec, lr = _fixture()
    spec.loc["G2", "B"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        _run(a, spec, lr, expressed_pct=None)


def test_floor_without_groupby_raises():
    a, spec, lr = _fixture()
    with pytest.raises(ValueError, match="groupby"):
        runSCALAR(a, spec, lr, n_nearest_neighbors=3)      # default floor


def test_bad_groupby_or_labels_raise():
    a, spec, lr = _fixture()
    with pytest.raises(KeyError, match="groupby"):
        _run(a, spec, lr, groupby="nope")
    a.obs["ct2"] = a.obs["ct"].astype(str).replace({"C": "Z"})
    with pytest.raises(ValueError, match="no cells labelled"):
        _run(a, spec, lr, groupby="ct2")
    with pytest.raises(ValueError, match="expressed_pct"):
        _run(a, spec, lr, expressed_pct=1.5)


# ---------------------------------------------------------------- cytome --


def _as_cytome(a, path):
    import cytome
    import scipy.sparse as sp

    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(a.n_obs),
        "barcode": [f"c{i}" for i in range(a.n_obs)],
        "ct": a.obs["ct"].astype(str).values}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(a.n_vars),
        "gene_id": list(a.var_names)}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.asarray(a.X, dtype=np.float32)))
    ds.flush()
    ds.close()
    return str(path)


def test_cytome_and_anndata_agree_under_the_floor(tmp_path):
    a, spec, lr = _fixture(seed=3)
    cy = _as_cytome(a, tmp_path / "s.cytome")
    det_mem = _scalar_detection_fraction(a, None, "ct", list(a.var_names))
    det_cy = _scalar_detection_fraction(cy, None, "ct", list(a.var_names), batch_size=7)
    pd.testing.assert_frame_equal(det_mem, det_cy[det_mem.columns])
    out_mem = _run(a, spec, lr, expressed_pct=0.1, n_nearest_neighbors=4)
    out_cy = _run(cy, spec, lr, expressed_pct=0.1, n_nearest_neighbors=4)
    cols = ["interaction_score", "p_value", "testable",
            "ligand_expressed_pct", "receptor_expressed_pct"]
    pd.testing.assert_frame_equal(out_mem.set_index(KEY)[cols].sort_index(),
                                  out_cy.set_index(KEY)[cols].sort_index(),
                                  rtol=1e-9)
