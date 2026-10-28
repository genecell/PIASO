"""stitchSpace fixes of the 2026-10-27 audit (review H3 finding 7).

On a store, stitchSpace looked the embedding up as use_rep minus 'X_' (a name
PIASO never writes), ran COSG on a dummy one-column matrix, turned the error
into empty marker sets (so every edge between batches was pruned: no
integration), and wrote the result into an accessor that does not take
assignment. bbknn_computation was never honoured, and every failure printed a
message and returned None, which success returns too.

BBKNN is not a dependency, so a small stand-in builds its graph here: each
cell's nearest cells within every batch. What is tested is stitchSpace's own
plumbing.
"""
from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")


@pytest.fixture
def fake_bbknn(monkeypatch):
    from sklearn.neighbors import NearestNeighbors
    calls = []

    def bbknn_matrix(pca, batch_list, neighbors_within_batch=3, n_pcs=None, trim=None,
                     computation="annoy", annoy_n_trees=10, metric="euclidean", **kw):
        calls.append(computation)
        if computation == "explode":
            raise RuntimeError("boom")
        n = pca.shape[0]
        batch_list = np.asarray(batch_list)
        rows, cols, dist = [], [], []
        for b in np.unique(batch_list):
            idx = np.flatnonzero(batch_list == b)
            d, nb = NearestNeighbors(n_neighbors=min(neighbors_within_batch, len(idx))).fit(pca[idx]).kneighbors(pca)
            rows.append(np.repeat(np.arange(n), nb.shape[1]))
            cols.append(idx[nb].ravel())
            dist.append(d.ravel() + 1e-3)
        D = sp.csr_matrix((np.concatenate(dist), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        C = D.copy()
        C.data = np.exp(-C.data)
        return D, C.maximum(C.T).tocsr(), {}

    matrix = types.ModuleType("bbknn.matrix")
    matrix.bbknn = bbknn_matrix
    pkg = types.ModuleType("bbknn")
    pkg.matrix = matrix
    pkg.__version__ = "1.6.0"
    monkeypatch.setitem(sys.modules, "bbknn", pkg)
    monkeypatch.setitem(sys.modules, "bbknn.matrix", matrix)
    return calls


def _data(seed=0, n=600, g=200, k=3):
    """Two batches of the same three types; the second batch's embedding shifted."""
    from anndata import AnnData
    rng = np.random.default_rng(seed)
    ct = rng.integers(0, k, n)
    batch = np.where(np.arange(n) % 2 == 0, "b1", "b2")
    lam = np.full((n, g), 0.3)
    for t in range(k):
        lam[np.ix_(ct == t, np.arange(t * 20, (t + 1) * 20))] *= 12.0
    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    emb = np.zeros((n, 10))
    emb[np.arange(n), ct] = 5.0
    emb += rng.normal(scale=0.3, size=emb.shape)
    emb[batch == "b2", 5] += 4.0                                # the batch effect
    a = AnnData(X, obs=pd.DataFrame({"batch": batch, "ct": ct.astype(str)},
                                    index=[f"c{i}" for i in range(n)]),
                var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    a.obsm["X_svd"] = emb
    return a


_KW = dict(batch_key="batch", use_rep="X_svd", filter_leiden_resolution=0.5, filter_n_markers=20,
           filter_marker_overlap_threshold=0.1, filter_bbknn_neighbors_within_batch=5)


def test_a_store_is_integrated_as_its_anndata_is(tmp_path, fake_bbknn):
    import cytome
    import piaso
    a = _data()
    ds = cytome.from_anndata(a, modality="RNA", output=str(tmp_path / "s.cytome"))
    try:
        ds.cells["batch"] = a.obs["batch"].values
        ds.add_embedding("RNA_svd", a.obsm["X_svd"].astype(np.float64))
        ds.flush()
        piaso.tl.infog(ds, save_layer=True, verbosity=0)
        a.layers["infog"] = ds.RNA.layer("infog").to_memory().tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piaso.tl.stitchSpace(ds, filter_cosg_layer="infog", **_KW)
            piaso.tl.stitchSpace(a, filter_cosg_layer="infog", **_KW)
        got = np.asarray(ds.embeddings["RNA_stitch"])
        assert got.shape == a.obsm["X_stitch"].shape
        np.testing.assert_allclose(got, a.obsm["X_stitch"], atol=1e-5)
        labels = np.asarray(ds.cells["batch@leiden@res0.5"]).astype(str)
        np.testing.assert_array_equal(labels, a.obs["batch@leiden@res0.5"].astype(str).values)
        # it integrated: links between batches survived the pruning (with empty
        # marker sets, as before the fix, every one of them was pruned)
        assert any(a.uns["X_stitch_hypergraph_compatibility"].values())
        G = a.obsp["pruned_markers_connectivities"].tocoo()
        bt = a.obs["batch"].values
        assert (bt[G.row] != bt[G.col]).sum() > 0
        b2 = bt == "b2"
        gap_before = abs(a.obsm["X_svd"][b2, 5].mean() - a.obsm["X_svd"][~b2, 5].mean())
        gap_after = abs(got[b2, 5].mean() - got[~b2, 5].mean())
        assert gap_after < gap_before
    finally:
        ds.close()


def test_the_bbknn_computation_is_passed_on_and_failures_raise(fake_bbknn):
    import piaso
    a = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piaso.tl.stitchSpace(a, filter_bbknn_computation="pynndescent", **_KW)
    assert fake_bbknn[-1] == "pynndescent"
    with pytest.raises(RuntimeError, match="BBKNN failed"):
        piaso.tl.stitchSpace(a.copy(), filter_bbknn_computation="explode", **_KW)
