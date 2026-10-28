"""runGDR fixes of the 2026-10-27 audit (review H3 findings 4 and 10, and minors).

- Multi-batch runGDR on an AnnData found its markers on raw counts, not on the
  INFOG layer the single-batch path and the store use; the same data in a
  store gave other markers and another X_gdr.
- A marker set with no gene to score shifted every later block (AnnData) or
  broke the assembly (store).
- Repeated obs_names expanded the result by name.
- The store's de-novo path ignored n_highly_variable_genes, left its temporary
  embedding and graph in the store, and recorded a label column that did not
  exist; the multi-batch store path with save_reference=False raised NameError.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")


def _counts(seed=5, n=1200, g=400, k=4):
    """Types with their own genes, depth strongly varying and confounded with
    type 0 -- so markers on raw counts and on INFOG differ."""
    rng = np.random.default_rng(seed)
    ct = rng.integers(0, k, n)
    batch = np.where(np.arange(n) % 2 == 0, "b1", "b2")
    base = rng.gamma(0.5, 1.0, g)
    lam = np.tile(base, (n, 1))
    for t in range(k):
        lam[np.ix_(ct == t, np.arange(t * 30, (t + 1) * 30))] *= 6.0
    depth = rng.lognormal(0, 0.9, n)[:, None] * (1 + 2.0 * (ct[:, None] == 0))
    X = sp.csr_matrix(rng.poisson(lam * depth).astype(np.float32))
    return X, ct.astype(str), batch


def _adata(X, ct, batch, names=None):
    from anndata import AnnData
    n, g = X.shape
    obs = pd.DataFrame({"ct": pd.Categorical(ct), "batch": batch},
                       index=names if names is not None else [f"c{i}" for i in range(n)])
    return AnnData(X.copy(), obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))


def _store(path, X, ct, batch):
    import cytome
    n, g = X.shape
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": [f"c{i}" for i in range(n)],
                                         "ct": ct, "batch": batch}))
    ds.set_entity("genes", pd.DataFrame({"gene_idx": np.arange(g), "gene_id": [f"G{i}" for i in range(g)],
                                         "symbol": [f"G{i}" for i in range(g)]}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


def _markers_by_set(mg, store=False):
    """{(batch, group): set of genes} from either path's marker table."""
    out = {}
    for col in mg.columns:
        if store:
            b, grp = str(col).split("_", 1)
        else:
            grp, b = str(col).rsplit("@", 1)
        out[(b, grp)] = set(map(str, mg[col].dropna()))
    return out


def test_multibatch_markers_are_the_infog_layers_on_anndata_and_on_a_store(tmp_path):
    import cosg
    import piaso
    X, ct, batch = _counts()
    a = _adata(X, ct, batch)
    piaso.tl.infog(a, verbosity=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piaso.tl.runGDR(a, batch_key="batch", groupby="ct", layer="infog", n_gene=20, mu=10.0,
                        max_workers=1, verbosity=0)
    got = _markers_by_set(a.uns["gdr_reference"]["marker_gene"])
    # COSG on each batch's rows of the INFOG layer, by hand
    for b in ("b1", "b2"):
        m = (a.obs["batch"] == b).values
        sub = _adata(X[m], ct[m], batch[m])
        sub.layers["infog"] = a.layers["infog"][m]
        cosg.cosg(sub, groupby="ct", key_added="ref", mu=10.0, n_genes_user=20, layer="infog",
                  expressed_pct=0.1, remove_lowly_expressed=True)
        ref = pd.DataFrame(sub.uns["ref"]["names"])
        for grp in ref.columns:
            assert got[(b, str(grp))] == set(ref[grp]), (b, grp)
    # the same data in a store: the same markers, and X_gdr to rounding
    ds = _store(tmp_path / "s.cytome", X, ct, batch)
    try:
        piaso.tl.infog(ds, save_layer=True, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xs, mg_s = piaso.tl.runGDR(ds, batch_key="batch", groupby="ct", n_gene=20, mu=10.0,
                                       max_workers=1, write_to_cytome=False, verbosity=0)
        assert _markers_by_set(mg_s, store=True) == got
        np.testing.assert_allclose(Xs, a.obsm["X_gdr"], atol=1e-4)
    finally:
        ds.close()


def _with_an_empty_set(mg):
    """The marker table with one more set, in the first batch's block, that maps no gene."""
    mg = mg.copy()
    first = [c for c in mg.columns if str(c).endswith("@b1")][0]
    pos = list(mg.columns).index(first) + 1
    mg.insert(pos, "ghost@b1", [f"NOT_A_GENE_{i}" for i in range(mg.shape[0])])
    return mg


def test_a_marker_set_with_no_gene_to_score_keeps_the_blocks_aligned(monkeypatch):
    import piaso
    import piaso.tools._runGDR as rg
    X, ct, batch = _counts()
    a = _adata(X, ct, batch)
    piaso.tl.infog(a, verbosity=0)
    real = rg.runCOSGParallel

    def with_ghost(*args, **kw):
        mg, n_groups = real(*args, **kw)
        return _with_an_empty_set(mg), [n_groups[0] + 1] + list(n_groups[1:])

    monkeypatch.setattr(rg, "runCOSGParallel", with_ghost)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piaso.tl.runGDR(a, batch_key="batch", groupby="ct", layer="infog", n_gene=20, mu=10.0,
                        max_workers=1, verbosity=0)
    E = a.obsm["X_gdr"]
    st = a.uns["gdr_reference"]
    assert "ghost@b1" not in list(st["marker_gene"].columns)
    blocks = st["block_indices"]
    assert blocks[-1] == E.shape[1] == st["marker_gene"].shape[1]
    for s0, s1 in zip(blocks[:-1], blocks[1:]):          # each block of each row has unit length
        np.testing.assert_allclose(np.linalg.norm(E[:, s0:s1], axis=1), 1.0, atol=1e-6)


def test_a_marker_set_with_no_gene_to_score_on_a_store(tmp_path, monkeypatch):
    import cosg
    import piaso
    X, ct, batch = _counts()
    ds = _store(tmp_path / "s.cytome", X, ct, batch)
    real = cosg.run_cosg_cytome

    def with_ghost(*args, **kw):
        res = dict(real(*args, **kw))
        names = np.asarray(res["names"], dtype=object).copy()
        names[:, 0] = None                                  # the first group's markers all filtered out
        res["names"] = names
        return res

    try:
        piaso.tl.infog(ds, save_layer=True, verbosity=0)
        monkeypatch.setattr(cosg, "run_cosg_cytome", with_ghost)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E, mg = piaso.tl.runGDR(ds, batch_key="batch", groupby="ct", n_gene=20, mu=10.0,
                                    max_workers=1, write_to_cytome=False, verbosity=0)
        assert E.shape[1] == mg.shape[1] == 2 * 4 - 2
        assert np.isfinite(E).all()
    finally:
        ds.close()


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_repeated_obs_names(batch_key):
    import piaso
    X, ct, batch = _counts()
    a = _adata(X, ct, batch)
    names = np.array([f"bc{i // 2}" for i in range(a.n_obs)])      # every name twice, once per batch
    b = _adata(X, ct, batch, names=names)
    for obj in (a, b):
        piaso.tl.infog(obj, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piaso.tl.runGDR(obj, batch_key=batch_key, groupby="ct", layer="infog", n_gene=20, mu=10.0,
                            max_workers=1, verbosity=0)
    assert b.obsm["X_gdr"].shape == a.obsm["X_gdr"].shape
    np.testing.assert_array_equal(b.obsm["X_gdr"], a.obsm["X_gdr"])


def test_the_stores_de_novo_run_keeps_its_labels_and_drops_its_temporaries(tmp_path, monkeypatch):
    import piaso
    X, ct, batch = _counts()
    ds = _store(tmp_path / "s.cytome", X, ct, batch)
    seen = {}
    real = piaso.tl.runSVD

    def spy(*args, **kw):
        if kw.get("key_added") == "X_svd_TMP_GDR":
            seen["hvg"] = kw.get("hvg_indices_override")
        return real(*args, **kw)

    try:
        piaso.tl.infog(ds, save_layer=True, n_top_genes=50, verbosity=0)   # the store's own selection: 50
        monkeypatch.setattr(piaso.tl, "runSVD", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piaso.tl.runGDR(ds, groupby=None, n_gene=10, n_highly_variable_genes=120, n_svd_dims=10,
                            max_workers=1, verbosity=0)
        assert seen["hvg"] is not None and len(seen["hvg"]) == 120
        cols = [r[1] for r in ds._conn.execute("PRAGMA table_info(cells)")]
        assert "gdr_local" in cols and "gdr_local_TMP_GDR" not in cols
        assert not any("TMP_GDR" in e for e in ds.embeddings.keys())
        assert not any("TMP_GDR" in g for g in ds.graphs.keys())
        assert not any("TMP_GDR" in k for k in ds.metadata.keys())
        from piaso.tools._projectGDR import _get_state
        st = _get_state(ds)
        assert st["denovo_labels"] == "gdr_local"
        assert set(map(str, st["marker_gene"].columns)) == set(np.unique(np.asarray(ds.cells["gdr_local"])))
    finally:
        ds.close()


def test_a_multibatch_store_run_without_a_reference(tmp_path):
    import piaso
    X, ct, batch = _counts()
    ds = _store(tmp_path / "s.cytome", X, ct, batch)
    try:
        piaso.tl.infog(ds, save_layer=True, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piaso.tl.runGDR(ds, batch_key="batch", groupby="ct", n_gene=10, max_workers=1,
                            save_reference=False, verbosity=0)
        assert "X_gdr" in ds.embeddings.keys()
        assert ds.metadata["X_gdr_params"]["groupby"] == "ct"
    finally:
        ds.close()


def test_de_novo_runGDR_leaves_the_callers_infog_and_selection_alone():
    """infog_svd, run for the de-novo clustering, recomputed INFOG with
    n_highly_variable_genes and wrote it into the caller's object: their
    var['highly_variable'] and layers['infog'] were replaced, and the temporary
    kNN graphs stayed in obsp."""
    import piaso
    X, ct, batch = _counts()
    a = _adata(X, ct, batch)
    piaso.tl.infog(a, n_top_genes=100, trim=False, verbosity=0)
    hv = a.var["highly_variable"].values.copy()
    inf = a.layers["infog"].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piaso.tl.runGDR(a, groupby=None, n_highly_variable_genes=300, n_svd_dims=10, n_gene=10,
                        max_workers=1, verbosity=0)
    np.testing.assert_array_equal(a.var["highly_variable"].values, hv)
    assert abs(a.layers["infog"] - inf).max() == 0
    assert not any("TMP_GDR" in k for k in list(a.obsp.keys()) + list(a.obsm.keys()) + list(a.uns.keys()))
    assert "gdr_local" in a.obs and a.obsm["X_gdr"].shape[0] == a.n_obs
