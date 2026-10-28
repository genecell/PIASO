"""Scrublet: libraries, unscored cells, the neighbour graph and the counts it reads.

Regression tests for the code audit of 2026-10-27 (review I1, finding 7):

- ``library_key`` on a NULL column (every pipeline store's ``sample_id``) ran ``WHERE col = NULL``,
  matched no cell and left every cell at score 0 and "not a doublet"; NaN labels on AnnData pooled or
  dropped their cells; a key that is not a column pooled every cell. All three now raise.
- libraries under 50 cells, and libraries with no more variable genes than ``n_components`` (a small
  panel, which crashed the AnnData path), got 0 / False; they now get NaN and a missing call, and one
  warning names them.
- the neighbour graph held each point itself (pynndescent's ``neighbor_graph`` lists it first), unlike
  scrublet, whose annoy and scikit-learn searches leave it out.
- a log1p ``.X`` without ``.raw`` was scored as counts; ``layer=`` names the counts; a column-major
  store is refused rather than decoded whole for every row block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import piaso
from piaso.preprocessing import _scrublet as S

cytome = pytest.importorskip("cytome")
pytest.importorskip("pynndescent")


def _counts(n=200, g=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(2, 0.3, size=(n, g)).astype(np.float32)


def _store(path, X, **columns):
    ds = cytome.create(str(path))
    cells = pd.DataFrame({"cell_idx": np.arange(X.shape[0]), "barcode": [f"b{i}" for i in range(X.shape[0])]})
    for k, v in columns.items():
        cells[k] = v
    ds.set_entity("cells", cells)
    ds.set_entity("genes", pd.DataFrame({"gene_idx": np.arange(X.shape[1]),
                                         "gene_id": [f"g{i}" for i in range(X.shape[1])]}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


# ------------------------------------------------------------------ library columns

def test_a_null_library_column_on_a_store_raises(tmp_path):
    X = _counts()
    ds = _store(tmp_path / "n.cytome", X, sample_id=[None] * X.shape[0])
    try:
        with pytest.raises(ValueError, match="missing .* for 200 of 200 cells"):
            piaso.pp.scrublet(ds, library_key="sample_id", verbose=False)
        assert "scrublet_score" not in ds.cells
    finally:
        ds.close()


@pytest.mark.parametrize("missing", [None, np.nan, ""])
def test_missing_library_labels_on_anndata_raise(missing):
    X = _counts()
    lib = np.array(["L1"] * 150 + [missing] * 50, dtype=object)
    a = AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"lib": lib}))
    with pytest.raises(ValueError, match="for 50 of 200 cells"):
        piaso.pp.scrublet(a, library_key="lib", verbose=False)


def test_a_library_key_that_is_not_a_column_raises(tmp_path):
    X = _counts()
    a = AnnData(sp.csr_matrix(X))
    with pytest.raises(KeyError, match="not a column"):
        piaso.pp.scrublet(a, library_key="nope", verbose=False)
    ds = _store(tmp_path / "k.cytome", X)
    try:
        with pytest.raises(KeyError, match="not a column"):
            piaso.pp.scrublet(ds, library_key="nope", verbose=False)
    finally:
        ds.close()


# ------------------------------------------------------------------ unscored cells

def test_a_small_library_is_unscored_on_both_inputs(tmp_path):
    X = _counts(n=230)
    lib = np.array(["big"] * 200 + ["small"] * 30, dtype=object)
    a = AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"lib": lib}))
    with pytest.warns(UserWarning, match=r"30 of 230 cells were not scored.*small \(30 cells"):
        piaso.pp.scrublet(a, library_key="lib", verbose=False)
    sc = a.obs["scrublet_score"].to_numpy()
    assert np.isnan(sc[200:]).all() and np.isfinite(sc[:200]).all()
    calls = a.obs["is_doublet"]
    assert str(calls.dtype) == "boolean"
    assert calls.isna().to_numpy()[200:].all() and not calls.isna().to_numpy()[:200].any()

    ds = _store(tmp_path / "s.cytome", X, lib=lib)
    try:
        with pytest.warns(UserWarning, match="30 of 230 cells were not scored"):
            piaso.pp.scrublet(ds, library_key="lib", verbose=False)
        df = ds.cells.to_pandas()
        assert df["scrublet_score"].isna().to_numpy()[200:].all()
        assert df["is_doublet"].isna().to_numpy()[200:].all()
        assert df["scrublet_score"].notna().to_numpy()[:200].all()
    finally:
        ds.close()


def test_a_fully_scored_run_keeps_a_plain_boolean():
    a = AnnData(sp.csr_matrix(_counts()))
    piaso.pp.scrublet(a, verbose=False)
    assert a.obs["is_doublet"].dtype == bool
    assert np.isfinite(a.obs["scrublet_score"].to_numpy()).all()


def test_a_small_panel_is_unscored_not_a_crash(tmp_path):
    X = _counts(g=60)
    a = AnnData(sp.csr_matrix(X))
    with pytest.warns(UserWarning, match="n_components=30"):
        piaso.pp.scrublet(a, verbose=False)             # crashed in PCA before
    assert np.isnan(a.obs["scrublet_score"].to_numpy()).all()
    ds = _store(tmp_path / "p.cytome", X)
    try:
        with pytest.warns(UserWarning, match="n_components=30"):
            piaso.pp.scrublet(ds, verbose=False)
        assert ds.cells.to_pandas()["scrublet_score"].isna().all()
    finally:
        ds.close()


# ------------------------------------------------------------------ the neighbour graph

def test_without_self_drops_the_point_wherever_it_is():
    knn = np.array([[0, 4, 5, 6],      # itself first
                    [3, 1, 7, 8],      # itself second
                    [9, 8, 7, 6]])     # itself absent
    out = S._without_self(knn, 3)
    np.testing.assert_array_equal(out, [[4, 5, 6], [3, 7, 8], [9, 8, 7]])


def test_doublet_scores_use_neighbours_other_than_the_point(monkeypatch):
    """With an exact search standing in for pynndescent, the scores are scrublet's formula on
    scikit-learn's kneighbors() of the fitted points (which leaves each point out)."""
    from sklearn.neighbors import NearestNeighbors
    import pynndescent

    class Exact:
        def __init__(self, data, n_neighbors, metric, random_state):
            idx = NearestNeighbors(n_neighbors=n_neighbors).fit(data).kneighbors(data)[1]
            self.neighbor_graph = (idx, None)            # each row starts with itself, as pynndescent's
    monkeypatch.setattr(pynndescent, "NNDescent", Exact)
    rng = np.random.default_rng(1)
    n_obs, n_sim = 120, 240
    obs, sim = rng.normal(size=(n_obs, 6)), rng.normal(0.5, 1.0, size=(n_sim, 6))
    got, _ = S._knn_score_threshold(obs, sim, n_obs, n_sim, None, 0.06, 0, 0.5, False)
    k = int(round(0.5 * np.sqrt(n_obs)))
    k_adj = int(round(k * (1 + n_sim / n_obs)))
    nb = NearestNeighbors(n_neighbors=k_adj).fit(np.vstack([obs, sim])).kneighbors()[1]
    labels = np.r_[np.zeros(n_obs), np.ones(n_sim)]
    nd = (labels[nb] == 1).sum(axis=1).astype(float)
    q = (nd + 1) / (k_adj + 2)
    r, rho = n_sim / n_obs, 0.06
    want = np.clip(q * rho / r / (1 - rho - q * (1 - rho - rho / r)), 0, None)[:n_obs]
    np.testing.assert_allclose(got, want)


# ------------------------------------------------------------------ which counts

def test_log_normalised_x_without_raw_is_refused_and_layer_names_the_counts():
    X = _counts()
    a = AnnData(sp.csr_matrix(np.log1p(X)))
    a.layers["counts"] = sp.csr_matrix(X)
    with pytest.raises(ValueError, match="not raw counts"):
        piaso.pp.scrublet(a, verbose=False)
    piaso.pp.scrublet(a, layer="counts", verbose=False)
    b = AnnData(sp.csr_matrix(X))
    piaso.pp.scrublet(b, verbose=False)
    np.testing.assert_array_equal(a.obs["scrublet_score"].to_numpy(), b.obs["scrublet_score"].to_numpy())


def test_a_column_major_store_is_refused(tmp_path):
    from conftest import make_column_major
    X = _counts()
    ds = _store(tmp_path / "c.cytome", X)
    try:
        make_column_major(ds, X, "RNA_counts")
        with pytest.raises(ValueError, match="column-major"):
            piaso.pp.scrublet(ds, verbose=False)
    finally:
        ds.close()
