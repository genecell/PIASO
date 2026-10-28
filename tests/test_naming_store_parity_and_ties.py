"""Marker naming: p-value ties, the store's scoring layer, and the GDR route's defaults.

Regression tests for the code audit of 2026-10-27 (review I1, findings 3 and 4, and its minors):

- ``use_score=False`` took the first of the cell types tied at the best p-value. With 100 control sets
  -log10 p tops out at log10(101), so types that share most markers tie, and the marker table's order
  decided (60 % of type-A cells were called B on the toy below).
- on a cytome, ``score_layer='infog'`` was ignored and the raw counts were scored; ``cytome_layer='infog'``
  without ``modality`` raised ``no None_infog matrix``.
- ``score_method='scanpy'`` with ``use_score=False`` raised ``UnboundLocalError``.
- ``predictCellTypeByGDR(ds, ref)`` with its defaults raised ``KeyError: 'log1p'``, and a query and a
  reference sharing barcodes crashed in runGDR.
- smoothing on an existing graph left the cell itself out; NaN labels crashed the vote.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import piaso

cytome = pytest.importorskip("cytome")

N_PER, N_GENES = 120, 100
GENES = [f"g{i}" for i in range(N_GENES)]
# A and B share 8 of their 10 markers, and B is listed first
MARKERS = {"B": GENES[0:8] + GENES[10:12], "A": GENES[0:10], "C": GENES[20:30]}


def _counts(seed=0):
    rng = np.random.default_rng(seed)
    lam = np.full((3 * N_PER, N_GENES), 0.3)
    lam[:N_PER, :10] = 6.0                      # A: shared + A-only
    lam[N_PER:2 * N_PER, :8] = 6.0              # B: shared + B-only
    lam[N_PER:2 * N_PER, 10:12] = 6.0
    lam[2 * N_PER:, 20:30] = 6.0                # C
    X = rng.poisson(lam).astype(np.float32)
    truth = np.array(["A"] * N_PER + ["B"] * N_PER + ["C"] * N_PER)
    return X, truth


def _anndata(X):
    a = AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
                var=pd.DataFrame(index=GENES))
    a.layers["counts"] = a.X.copy()
    piaso.tl.infog(a, layer="counts", verbosity=0, n_top_genes=50)
    return a


def _store(path, X, embedding=None):
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(X.shape[0]),
                                         "barcode": [f"c{i}" for i in range(X.shape[0])]}))
    ds.set_entity("genes", pd.DataFrame({"gene_idx": np.arange(N_GENES), "gene_id": GENES}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    if embedding is not None:
        ds.add_embedding("X_test", np.asarray(embedding, dtype=np.float32))
    ds.flush()
    piaso.tl.infog(ds, verbosity=0, n_top_genes=50)
    return ds


def test_pvalue_ties_go_to_the_higher_score():
    X, truth = _counts()
    a = _anndata(X)
    piaso.tl.predictCellTypeByMarker(a, MARKERS, use_score=False, smooth_prediction=False, verbosity=0)
    pred = a.obs["CellTypes_predicted"].astype(str).to_numpy()
    nlp = np.asarray(a.obsm["CellTypes_predicted_nlog10pvals"])
    # the premise: A and B both saturate in most A cells
    assert np.mean(np.isclose(nlp[:N_PER, 0], nlp[:N_PER, 1])) > 0.5
    # before the fix 60 % of the A cells were called B, the type listed first
    assert np.mean(pred[:N_PER] == "A") > 0.95
    assert np.mean(pred == truth) > 0.95


def test_score_route_labels_are_unchanged_by_the_tie_break():
    X, truth = _counts()
    a = _anndata(X)
    piaso.tl.predictCellTypeByMarker(a, MARKERS, use_score=True, smooth_prediction=False, verbosity=0)
    S = np.asarray(a.obsm["CellTypes_predicted_score"])
    expected = pd.DataFrame(S, columns=list(MARKERS)).idxmax(axis=1).to_numpy()
    np.testing.assert_array_equal(a.obs["CellTypes_predicted"].to_numpy(), expected)


def test_n_ctrl_set_is_exposed_and_sets_the_pvalue_floor():
    X, _ = _counts()
    a = _anndata(X)
    piaso.tl.predictCellTypeByMarker(a, MARKERS, use_score=False, smooth_prediction=False, verbosity=0,
                                     n_ctrl_set=250)
    nlp = np.asarray(a.obsm["CellTypes_predicted_nlog10pvals"])
    assert nlp.max() == pytest.approx(np.log10(251))


def test_store_scores_the_same_layer_as_anndata(tmp_path):
    X, truth = _counts()
    a = _anndata(X)
    ds = _store(tmp_path / "s.cytome", X)
    try:
        piaso.tl.predictCellTypeByMarker(a, MARKERS, smooth_prediction=False, verbosity=0)
        piaso.tl.predictCellTypeByMarker(ds, MARKERS, smooth_prediction=False, verbosity=0)
        np.testing.assert_allclose(np.asarray(ds.cells["CellTypes_predicted_score"], dtype=float),
                                   a.obs["CellTypes_predicted_score"].to_numpy(), atol=1e-5)
        np.testing.assert_array_equal(np.asarray(ds.cells["CellTypes_predicted"]).astype(str),
                                      a.obs["CellTypes_predicted"].astype(str).to_numpy())
        # an explicit cytome_layer still overrides on the store: counts, as AnnData's score_layer='counts'
        b = a.copy()
        piaso.tl.predictCellTypeByMarker(b, MARKERS, score_layer="counts", smooth_prediction=False,
                                         verbosity=0)
        piaso.tl.predictCellTypeByMarker(ds, MARKERS, cytome_layer="counts", smooth_prediction=False,
                                         verbosity=0, key_added="on_counts")
        np.testing.assert_allclose(np.asarray(ds.cells["on_counts_score"], dtype=float),
                                   b.obs["CellTypes_predicted_score"].to_numpy(), atol=1e-5)
        assert not np.allclose(b.obs["CellTypes_predicted_score"].to_numpy(),
                               a.obs["CellTypes_predicted_score"].to_numpy(), atol=1e-3)
    finally:
        ds.close()


def test_store_cytome_layer_without_modality(tmp_path):
    X, _ = _counts()
    ds = _store(tmp_path / "s.cytome", X)
    try:
        # used to raise KeyError: no None_infog matrix
        piaso.tl.predictCellTypeByMarker(ds, MARKERS, cytome_layer="infog", smooth_prediction=False,
                                         verbosity=0)
        assert "CellTypes_predicted" in ds.cells
    finally:
        ds.close()


def test_store_writes_the_obs_columns_and_a_path_is_closed(tmp_path, monkeypatch):
    X, truth = _counts()
    emb = np.random.default_rng(1).normal(size=(X.shape[0], 5)) + np.repeat(np.eye(3, 5) * 8, N_PER, axis=0)
    ds = _store(tmp_path / "s.cytome", X, embedding=emb)
    ds.close()
    opened = []
    real_open = cytome.open

    def spy(*args, **kwargs):
        d = real_open(*args, **kwargs)
        opened.append(d)
        return d
    monkeypatch.setattr(cytome, "open", spy)
    piaso.tl.predictCellTypeByMarker(str(tmp_path / "s.cytome"), MARKERS, use_rep="X_test", verbosity=0)
    assert opened and all(d.is_closed for d in opened)
    monkeypatch.setattr(cytome, "open", real_open)
    with cytome.open(str(tmp_path / "s.cytome")) as d:
        for col in ("CellTypes_predicted", "CellTypes_predicted_score", "CellTypes_predicted_raw",
                    "CellTypes_predicted_confidence_smoothed"):
            assert col in d.cells, col
        assert np.mean(np.asarray(d.cells["CellTypes_predicted"]).astype(str) == truth) > 0.95


def test_scanpy_method_with_pvalues_raises_a_clear_error():
    X, _ = _counts()
    a = _anndata(X)
    with pytest.raises(ValueError, match="use_score=False"):
        piaso.tl.predictCellTypeByMarker(a, MARKERS, score_method="scanpy", use_score=False,
                                         smooth_prediction=False, verbosity=0)


def _reference(X, truth):
    ref = AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"CellTypes": truth},
                                                    index=[f"c{i}" for i in range(X.shape[0])]),
                  var=pd.DataFrame(index=GENES))
    return ref


def test_predict_by_gdr_defaults_on_a_store(tmp_path):
    pytest.importorskip("harmonypy")        # the label transfer aligns with Harmony (the optional harmony extra)
    X, truth = _counts()
    ds = _store(tmp_path / "q.cytome", X)
    try:
        ds.cells["Leiden"] = truth
        ds.flush()
        # neither the store nor the reference has a log1p layer; both hold counts
        piaso.tl.predictCellTypeByGDR(ds, _reference(X, truth), verbosity=0, n_highly_variable_genes=60)
        pred = np.asarray(ds.cells["CellTypes_gdr"]).astype(str)
        assert np.mean(pred == truth) > 0.9
    finally:
        ds.close()


def test_predict_by_gdr_with_shared_barcodes():
    pytest.importorskip("harmonypy")        # the label transfer aligns with Harmony (the optional harmony extra)
    X, truth = _counts()
    q = AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"Leiden": truth},
                                                  index=[f"c{i}" for i in range(X.shape[0])]),
                var=pd.DataFrame(index=GENES))
    ref = _reference(X, truth)                  # the same barcodes as the query
    piaso.tl.predictCellTypeByGDR(q, ref, verbosity=0, n_highly_variable_genes=60)
    assert np.mean(q.obs["CellTypes_gdr"].astype(str).to_numpy() == truth) > 0.9
    assert "log1p" not in q.layers              # built on a copy, never on the caller's object


def test_predict_by_gdr_refuses_to_log_normalised_values():
    X, truth = _counts()
    ref = _reference(np.log1p(X), truth)
    q = AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"Leiden": truth}), var=pd.DataFrame(index=GENES))
    with pytest.raises(KeyError, match="integer counts"):
        piaso.tl.predictCellTypeByGDR(q, ref, verbosity=0)


# ---------------------------------------------------------------- smoothing

def _old_loop_vote(labels, indices):
    """The per-cell vote smoothCellTypePrediction ran before it was vectorised."""
    from scipy import stats
    s = pd.Series(labels)
    out, conf = np.empty(len(s), dtype=object), np.empty(len(s))
    for i in range(len(s)):
        nt = s.iloc[indices[i]]
        try:
            maj = stats.mode(nt, keepdims=False).mode
        except (TypeError, AttributeError):
            u, c = np.unique(nt, return_counts=True)
            maj = u[np.argmax(c)]
        out[i] = maj
        conf[i] = np.mean(nt == maj)
    return out, conf


@pytest.mark.parametrize("k", [7, 15])
def test_vectorised_vote_matches_the_old_loop(k):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(3)
    n = 400
    emb = rng.normal(size=(n, 4))
    labels = rng.choice(np.array(["T", "B", "NK", "Mono"], dtype=object), size=n)
    a = AnnData(sp.csr_matrix((n, 1), dtype=np.float32), obs=pd.DataFrame({"raw": labels}))
    a.obsm["X_e"] = emb
    piaso.tl.smoothCellTypePrediction(a, groupby="raw", use_rep="X_e", k_nearest_neighbors=k,
                                      return_confidence=True, use_existing_adjacency_graph=False,
                                      key_added="sm", verbosity=0)
    idx = NearestNeighbors(n_neighbors=k).fit(emb).kneighbors(emb)[1]
    want, want_conf = _old_loop_vote(labels, idx)
    np.testing.assert_array_equal(a.obs["sm"].to_numpy(), want)
    np.testing.assert_array_equal(a.obs["sm_confidence"].to_numpy(), want_conf)


def test_graph_smoothing_counts_the_cell_itself():
    # cell 0 ('A') is connected to 1 ('B', 0.9), 2 ('A', 0.8) and 3 ('B', 0.1)
    conn = sp.csr_matrix(np.array([
        [0, .9, .8, .1],
        [.9, 0, .2, .3],
        [.8, .2, 0, .4],
        [.1, .3, .4, 0]]))
    a = AnnData(sp.csr_matrix((4, 1), dtype=np.float32), obs=pd.DataFrame({"raw": ["A", "B", "A", "B"]}))
    a.obsp["connectivities"] = conn
    a.uns["neighbors"] = {"params": {}}
    piaso.tl.smoothCellTypePrediction(a, groupby="raw", k_nearest_neighbors=3, return_confidence=True,
                                      use_existing_adjacency_graph=True, key_added="sm", verbosity=0)
    # itself + its two strongest (B 0.9, A 0.8): A wins 2 to 1. Without itself, B 0.9 + A 0.8 + B 0.1: B.
    assert a.obs["sm"].iloc[0] == "A"
    assert a.obs["sm_confidence"].iloc[0] == pytest.approx(2 / 3)


def test_graph_built_on_another_representation_warns():
    conn = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    a = AnnData(sp.csr_matrix((3, 1), dtype=np.float32), obs=pd.DataFrame({"raw": ["A", "A", "B"]}))
    a.obsp["connectivities"] = conn
    a.uns["neighbors"] = {"params": {"use_rep": "X_pca"}}
    with pytest.warns(UserWarning, match="built from 'X_pca'"):
        piaso.tl.smoothCellTypePrediction(a, groupby="raw", use_rep="X_gdr", k_nearest_neighbors=3,
                                          use_existing_adjacency_graph=True, verbosity=0)


def test_unlabelled_cells_cast_no_vote():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, .1, size=(20, 2)), rng.normal(5, .1, size=(20, 2))])
    labels = np.array(["A"] * 20 + ["B"] * 20, dtype=object)
    labels[[0, 1, 25]] = np.nan
    a = AnnData(sp.csr_matrix((40, 1), dtype=np.float32), obs=pd.DataFrame({"raw": labels}))
    a.obsm["X_e"] = emb
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        piaso.tl.smoothCellTypePrediction(a, groupby="raw", use_rep="X_e", k_nearest_neighbors=5,
                                          return_confidence=True, use_existing_adjacency_graph=False,
                                          key_added="sm", verbosity=0)
    out = a.obs["sm"].to_numpy()
    assert list(out[:20]) == ["A"] * 20 and list(out[20:]) == ["B"] * 20
