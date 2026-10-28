"""predictCellTypeByMarker(smooth_by='scores'): each cell's scores are averaged over its neighbors before the best
cell type is taken, instead of the default vote of the neighbors' own best types (smooth_by='labels', unchanged).

The neighbors are the vote's: the k nearest cells in ``use_rep`` (the cell itself included), or the existing graph's
k strongest connections when ``use_existing_adjacency_graph`` is set. The averaged matrix is kept next to the raw
one, and the confidence is the share of a cell's neighbors whose own best type is the one chosen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

K = 7
MARKERS = {"A": [f"G{i}" for i in range(10)],
           "B": [f"G{i}" for i in range(10, 20)],
           "C": [f"G{i}" for i in range(20, 30)]}


@pytest.fixture
def adata():
    import anndata as ad
    rng = np.random.default_rng(0)
    n, g = 300, 120
    X = rng.poisson(0.5, (n, g)).astype(np.float32)
    for t in range(3):                                  # three cell types, weak markers so single cells err
        X[100 * t:100 * (t + 1), 10 * t:10 * (t + 1)] += rng.poisson(0.4, (100, 10))
    a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    a.layers["infog"] = a.X.copy()
    centers = np.repeat(np.eye(3) * 5.0, 100, axis=0)
    a.obsm["X_svd"] = (centers + rng.normal(0, 1.0, (n, 3))).astype(np.float32)
    return a


def _run(a, key="p", markers=MARKERS, **kw):
    import piaso
    kw.setdefault("k_nearest_neighbors", K)
    piaso.tl.predictCellTypeByMarker(a, marker_gene_set=markers, use_rep="X_svd", key_added=key, verbosity=0, **kw)


def _column_names(a, key):
    """The score matrix's column names, read off the raw calls (each column's name is its argmax cells' call)."""
    S = np.asarray(a.obsm[f"{key}_score"])
    raw = a.obs[f"{key}_raw"].astype(str).to_numpy()
    best = S.argmax(axis=1)
    return {j: raw[best == j][0] for j in np.unique(best)}


def test_the_default_is_still_the_label_vote(adata):
    a, b = adata.copy(), adata.copy()
    _run(a)
    _run(b, smooth_by="labels")
    assert (a.obs["p"].astype(str).to_numpy() == b.obs["p"].astype(str).to_numpy()).all()
    assert np.allclose(a.obs["p_confidence_smoothed"], b.obs["p_confidence_smoothed"])
    assert "p_score_smoothed" not in a.obsm


def test_scores_are_averaged_over_the_votes_neighbors(adata):
    _run(adata, smooth_by="scores")
    S = np.asarray(adata.obsm["p_score"], dtype=float)
    idx = NearestNeighbors(n_neighbors=K).fit(adata.obsm["X_svd"]).kneighbors(adata.obsm["X_svd"],
                                                                                return_distance=False)
    avg = S[idx].mean(axis=1)
    assert np.allclose(adata.obsm["p_score_smoothed"], avg)
    names = _column_names(adata, "p")
    calls = adata.obs["p"].astype(str).to_numpy()
    assert (calls == np.array([names[j] for j in avg.argmax(axis=1)])).all()
    raw = adata.obs["p_raw"].astype(str).to_numpy()
    assert np.allclose(adata.obs["p_confidence_smoothed"], (raw[idx] == calls[:, None]).mean(axis=1))


def test_pooling_the_scores_recovers_cells_their_own_scores_miss(adata):
    _run(adata, smooth_by="scores")
    truth = np.repeat(["A", "B", "C"], 100)
    raw_acc = (adata.obs["p_raw"].astype(str).to_numpy() == truth).mean()
    acc = (adata.obs["p"].astype(str).to_numpy() == truth).mean()
    assert raw_acc < 0.95 and acc > raw_acc


def test_one_neighbor_keeps_each_cells_own_call(adata):
    _run(adata, smooth_by="scores", k_nearest_neighbors=1)
    assert (adata.obs["p"].astype(str).to_numpy() == adata.obs["p_raw"].astype(str).to_numpy()).all()
    assert np.allclose(adata.obs["p_confidence_smoothed"], 1.0)


def test_pvalues_are_averaged_with_use_score_false(adata):
    _run(adata, smooth_by="scores", use_score=False)
    P = np.asarray(adata.obsm["p_nlog10pvals"], dtype=float)
    idx = NearestNeighbors(n_neighbors=K).fit(adata.obsm["X_svd"]).kneighbors(adata.obsm["X_svd"],
                                                                                return_distance=False)
    assert np.allclose(adata.obsm["p_nlog10pvals_smoothed"], P[idx].mean(axis=1))


def test_an_unknown_value_is_refused(adata):
    with pytest.raises(ValueError, match="smooth_by"):
        _run(adata, smooth_by="label")


def test_the_existing_graph_is_used_when_asked(adata):
    # a graph that sends every cell to cells 0 to 6 (all type A), strongest first. Each cell pools itself and
    # its K - 1 strongest other connections (the K cells k_nearest_neighbors counts, the cell itself included,
    # as on the embedding): cells 0 to 5 for a cell outside them, and the six others of 0 to 6 for one inside.
    n = adata.n_obs
    rows = np.repeat(np.arange(n), K)
    cols = np.tile(np.arange(K), n)
    vals = np.tile(np.linspace(1.0, 0.5, K), n)
    adata.obsp["connectivities"] = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    adata.uns["neighbors"] = {"connectivities_key": "connectivities"}
    _run(adata, smooth_by="scores", use_existing_adjacency_graph=True)
    S = np.asarray(adata.obsm["p_score"], dtype=float)
    want = np.empty_like(S)
    for i in range(n):
        others = [j for j in range(K) if j != i][:K - 1]
        want[i] = S[[i] + others].mean(axis=0)
    assert np.allclose(adata.obsm["p_score_smoothed"], want)


def test_cell_type_extraction_works_on_the_average(adata):
    split = {"A-1": MARKERS["A"][:5], "A-2": MARKERS["A"][5:], "B-1": MARKERS["B"], "C-1": MARKERS["C"]}
    _run(adata, markers=split, smooth_by="scores", extract_cell_type=True, delimiter_cell_type="-")
    assert set(adata.obs["p"].astype(str)) <= {"A", "B", "C"}


def test_a_cytome_gets_the_pooled_calls(adata, tmp_path):
    import cytome
    path = str(tmp_path / "toy.cytome")
    a = adata.copy()
    a.X = sp.csr_matrix(np.round(a.X.toarray()))
    cytome.from_anndata(a, modality="RNA", output=path).close()
    import piaso
    piaso.tl.predictCellTypeByMarker(path, marker_gene_set=MARKERS, score_layer=None, use_rep="X_svd",
                                     k_nearest_neighbors=K, key_added="p", smooth_by="scores", verbosity=0)
    ds = cytome.open(path, mode="r")
    try:
        calls = np.asarray(ds.cells["p"]).astype(str)
    finally:
        ds.close()
    assert len(calls) == adata.n_obs and set(calls) <= {"A", "B", "C"}
