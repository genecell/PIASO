"""runSCALAR's score columns and the 'min' statistic.

``specificity_geomean`` is a monotone function of the product and never
touches the test; ``null_mean`` is the mean of the interaction's own k x k
null, computed without materialising it; ``log2_enrichment`` references the
observation to that mean. ``statistic='min'`` asks whether both sides are
specific, with an exact null that factorises into two one-sided counts.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from piaso.tools._ligandReceptor import (
    _exact_exceed_counts_min,
    _null_mean,
    _prepare_background_genes,
    _scalar_detection_fraction,
    _scalar_gene_stats,
    runSCALAR,
)

TYPES = ["A", "B", "C"]
KEY = ["ligand", "receptor", "sender", "receiver"]


def _fixture(n_genes=16, n_cells=90, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = genes
    a.obs["ct"] = pd.Categorical(np.array(TYPES * (n_cells // 3)))
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


# ------------------------------------------------------------ columns --


def test_per_side_specificities_and_geomean_are_what_they_say():
    a, spec, lr = _fixture()
    out = _run(a, spec, lr)
    for r in out.itertuples():
        assert r.ligand_specificity == pytest.approx(spec.at[r.ligand, r.sender])
        assert r.receptor_specificity == pytest.approx(spec.at[r.receptor, r.receiver])
        assert r.interaction_score == pytest.approx(r.ligand_specificity * r.receptor_specificity)
        assert r.specificity_geomean == pytest.approx(np.sqrt(r.interaction_score))


def test_geomean_ranks_exactly_as_the_product_within_every_pair():
    a, spec, lr = _fixture()
    out = _run(a, spec, lr)
    for _, g in out.groupby(["sender", "receiver"]):
        a1 = g["interaction_score"].rank(method="first").values
        a2 = g["specificity_geomean"].rank(method="first").values
        assert (a1 == a2).all()


def test_null_mean_matches_the_materialised_support():
    rng = np.random.default_rng(3)
    lig = np.abs(rng.normal(size=(20, 7)))
    rec = np.abs(rng.normal(size=(20, 7)))
    lig[rng.random((20, 7)) < 0.2] = 0.0
    grid = lig[:, :, None] * rec[:, None, :]
    np.testing.assert_allclose(_null_mean(lig, rec, "product"), grid.reshape(20, -1).mean(1))
    grid_min = np.minimum(lig[:, :, None], rec[:, None, :])
    np.testing.assert_allclose(_null_mean(lig, rec, "min"), grid_min.reshape(20, -1).mean(1))


def test_enrichment_is_the_documented_ratio_and_nan_when_untestable():
    a, spec, lr = _fixture()
    eps = 1e-9
    out = _run(a, spec, lr, expressed_pct=0.1, enrichment_pseudocount=eps)
    t = out[out["testable"]]
    want = np.log2((t["interaction_score"] + eps) / np.maximum(t["null_mean"], eps))
    np.testing.assert_allclose(t["log2_enrichment"].values, want.values)
    u = out[~out["testable"]]
    if len(u):
        assert u["null_mean"].isna().all() and u["log2_enrichment"].isna().all()


def test_null_mean_is_the_mean_of_the_actual_matched_null():
    """Reconstruct one interaction's null from the same per-type universes
    runSCALAR builds and compare its mean to the reported column."""
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
        l = spec[row["sender"]].reindex(bs.index).values[bs.loc[row["ligand"]].values]
        r = spec[row["receiver"]].reindex(br.index).values[br.loc[row["receptor"]].values]
        assert row["null_mean"] == pytest.approx(np.multiply.outer(l, r).mean())


# ------------------------------------------------------------ 'min' --


@pytest.mark.parametrize("seed", range(3))
def test_min_exceed_counts_equal_the_grid(seed):
    rng = np.random.default_rng(seed)
    n, k = 25, 6
    lig = np.abs(rng.normal(size=(n, k)))
    rec = np.abs(rng.normal(size=(n, k)))
    obs = np.abs(rng.normal(size=n))
    obs[:3] = 0.0
    grid = np.minimum(lig[:, :, None], rec[:, None, :]).reshape(n, -1)
    want = (grid >= obs[:, None]).sum(1)
    want[obs <= 0] = k * k
    assert (_exact_exceed_counts_min(lig, rec, obs) == want).all()


def test_min_statistic_end_to_end_matches_enumeration():
    a, spec, lr = _fixture()
    k = 3
    out = _run(a, spec, lr, statistic="min", n_nearest_neighbors=k)
    knn_df, _ = _prepare_background_genes(
        a, n_nearest_neighbors=k, genes_universe=list(a.var_names),
        query_genes=list(lr["ligand"]) + list(lr["receptor"]))
    for _, row in out.iterrows():
        assert row["interaction_score"] == pytest.approx(
            min(spec.at[row["ligand"], row["sender"]], spec.at[row["receptor"], row["receiver"]]))
        l = spec[row["sender"]].reindex(knn_df.index).values[knn_df.loc[row["ligand"]].values]
        r = spec[row["receiver"]].reindex(knn_df.index).values[knn_df.loc[row["receptor"]].values]
        support = np.minimum.outer(l, r).ravel()
        want = (int((support >= row["interaction_score"]).sum()) + 1) / (k * k + 1)
        assert row["p_value"] == pytest.approx(want)


def test_min_and_product_differ_but_share_the_geomean():
    a, spec, lr = _fixture()
    p = _run(a, spec, lr).set_index(KEY)
    m = _run(a, spec, lr, statistic="min").set_index(KEY)
    pd.testing.assert_series_equal(p["specificity_geomean"].sort_index(),
                                   m["specificity_geomean"].sort_index())
    assert not np.allclose(p["interaction_score"].sort_index().values,
                           m["interaction_score"].sort_index().values)


def test_unknown_statistic_raises():
    a, spec, lr = _fixture()
    with pytest.raises(ValueError, match="statistic"):
        _run(a, spec, lr, statistic="harmonic")
