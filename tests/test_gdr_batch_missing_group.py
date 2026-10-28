"""runGDR with batch_key and groupby, where one batch has no cell of one group.

COSG before 1.2.1 scored a category without cells: the batch's COSG call returned a marker
set for the missing group, made of genes unrelated to it, and runGDR added it to the
reference as one more dimension of X_gdr ('3@b2' below: 20 genes, none of type 3's markers).
COSG 1.2.1 scores only the groups that have cells, so the reference holds one marker set per
batch-group pair that exists. PIASO's floor, cosg>=1.2.1, guarantees this; the test pins it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso


def _batches_with_a_missing_group(seed=5, n=1200, g=400, k=4):
    """Four types with 30 genes of their own each, in batches b1 and b2; b2 has no cell of type 3."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    ct = rng.integers(0, k, n)
    batch = np.where(np.arange(n) % 2 == 0, "b1", "b2")
    keep = ~((batch == "b2") & (ct == 3))
    ct, batch = ct[keep], batch[keep]
    n = int(keep.sum())
    lam = np.tile(rng.gamma(0.5, 1.0, g), (n, 1))
    for t in range(k):
        lam[np.ix_(ct == t, np.arange(t * 30, (t + 1) * 30))] *= 6.0
    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    a = ad.AnnData(X, obs=pd.DataFrame({"ct": pd.Categorical(ct.astype(str)), "batch": batch},
                                       index=[f"c{i}" for i in range(n)]))
    a.var_names = [f"g{i}" for i in range(g)]
    return a


def test_a_batch_without_a_group_adds_no_marker_set_for_it():
    a = _batches_with_a_missing_group()
    assert not ((a.obs["batch"] == "b2") & (a.obs["ct"] == "3")).any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piaso.tl.infog(a, verbosity=0)
        piaso.tl.runGDR(a, batch_key="batch", groupby="ct", layer="infog", n_gene=20, mu=10.0,
                        max_workers=1, verbosity=0)

    marker_gene = a.uns["gdr_reference"]["marker_gene"]
    sets = [str(c) for c in marker_gene.columns]
    assert "3@b2" not in sets
    # one marker set per batch-group pair with cells: four in b1, three in b2
    assert sorted(sets) == sorted([f"{t}@b1" for t in "0123"] + [f"{t}@b2" for t in "012"])
    assert a.obsm["X_gdr"].shape == (a.n_obs, 7)

    # every set is made of its own type's genes (each type's 30 markers are g{30t} .. g{30t+29})
    for name in sets:
        t = int(name.split("@")[0])
        genes = [x for x in marker_gene[name].astype(str) if x not in ("nan", "", "None")]
        assert genes, name
        own = sum(int(x[1:]) // 30 == t for x in genes if x[1:].isdigit())
        assert own >= len(genes) // 2, (name, genes)


def _typed_counts(rng, labels, genes_of, n_genes=300):
    """Poisson counts in which each label's cells raise their own 30 genes six-fold."""
    lam = np.tile(rng.gamma(0.5, 1.0, n_genes), (len(labels), 1))
    for lab, start in genes_of.items():
        lam[np.ix_(labels == lab, np.arange(start, start + 30))] *= 6.0
    return sp.csr_matrix(rng.poisson(lam).astype(np.float32))


def test_predictCellTypeByGDR_adds_no_set_for_a_label_absent_from_a_batch():
    """predictCellTypeByGDR joins the reference and the query as two batches of one runGDR, grouped by
    one column that holds the reference's types and the query's clusters, so every label is absent from
    one of the batches. With COSG before 1.2.1 each such (batch, label) pair added a marker set of
    unrelated genes: half of X_gdr was noise, aligned by Harmony and used for the label transfer. Now
    X_gdr has one dimension per (batch, label) pair that has cells."""
    import anndata as ad

    pytest.importorskip("harmonypy")
    rng = np.random.default_rng(11)
    ref_types = rng.choice(np.array(["A", "B", "C"]), 600)
    query_clusters = rng.choice(np.array(["0", "1", "2"]), 600)
    ref = ad.AnnData(_typed_counts(rng, ref_types, {"A": 0, "B": 30, "C": 60}),
                     obs=pd.DataFrame({"CellTypes": pd.Categorical(ref_types)},
                                      index=[f"r{i}" for i in range(600)]))
    query = ad.AnnData(_typed_counts(rng, query_clusters, {"0": 0, "1": 30, "2": 60}),
                       obs=pd.DataFrame({"Leiden": pd.Categorical(query_clusters)},
                                        index=[f"q{i}" for i in range(600)]))
    ref.var_names = query.var_names = [f"g{i}" for i in range(300)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joint = piaso.tl.predictCellTypeByGDR(query, ref, reference_groupby="CellTypes",
                                              query_groupby="Leiden", n_genes=15,
                                              return_integration=True)

    sets = sorted(str(c) for c in joint.uns["gdr_reference"]["marker_gene"].columns)
    assert sets == ["0@1", "1@1", "2@1", "A@0", "B@0", "C@0"]
    assert joint.obsm["X_gdr"].shape == (1200, 6)
    # the transfer itself: each query cluster is named after the reference type with its genes
    called = pd.crosstab(query.obs["Leiden"], query.obs["CellTypes_gdr"])
    assert (called.idxmax(axis=1).to_dict() == {"0": "A", "1": "B", "2": "C"})
