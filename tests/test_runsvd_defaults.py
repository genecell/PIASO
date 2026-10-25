"""runSVD's defaults for its stopping rule and for unit length.

The solver stops on the residuals of its singular vectors for every input (``stop_on='residuals'``; under
``method='auto'`` at 1e-2): the vectors themselves converge, so the embedding no longer depends on
``random_state``. Unit length is on where it was measured to help, the TF-IDF of ATAC peaks or tiles, whose
embedding vector's length follows the cell's depth: a store's or an AnnData's ``modality='ATAC'`` or ``'tiles'``, or
TF-IDF applied in the call (``tfidf_params``, ``auto_tfidf``). It is off otherwise, as for INFOG of RNA, where it
gave no gain. Passing True or False always decides. ``leiden_local`` leaves the choice to each group's runSVD, which
decides by what it factorises.
"""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso
from piaso.tools._rsvd_core import AUTO_RESIDUAL_TOL


def slow_matrix(n=1500, m=400, seed=0, decay=0.995):
    """A matrix whose spectrum decays so slowly that the singular values settle before the vectors."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, m)))
    V, _ = np.linalg.qr(rng.standard_normal((m, m)))
    return (U * (10.0 * decay ** np.arange(m))) @ V.T


def counts(n_cells=300, n_genes=120, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.5, 1.0, size=(1, n_genes)) * rng.gamma(2.0, 0.5, size=(n_cells, 1))
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(lam).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


def norms(E):
    return np.linalg.norm(np.asarray(E, dtype=np.float64), axis=1)


def test_the_default_stop_is_the_residuals():
    X = slow_matrix()
    kw = dict(use_highly_variable=False, n_components=10, random_state=0, return_svd=True)
    e0, s0, v0 = piaso.tl.runSVD(ad.AnnData(X=sp.csr_matrix(X)), **kw)
    e1, s1, v1 = piaso.tl.runSVD(ad.AnnData(X=sp.csr_matrix(X)), stop_on="residuals", **kw)
    e2, _, _ = piaso.tl.runSVD(ad.AnnData(X=sp.csr_matrix(X)), stop_on="singular_values", **kw)
    assert np.array_equal(e0, e1) and np.array_equal(s0, s1) and np.array_equal(v0, v1)
    assert not np.array_equal(e0, e2)

    def resid(e, s):
        U = e / s
        return (np.linalg.norm(X @ (X.T @ U) - U * s ** 2, axis=0) / s ** 2).max()

    # the default's vectors are converged to its tolerance; the singular-value rule's here are not
    assert resid(e0, s0) < AUTO_RESIDUAL_TOL < resid(e2, piaso.tl.runSVD(ad.AnnData(X=sp.csr_matrix(X)),
                                                                          stop_on="singular_values", **kw)[1])


def test_unit_length_is_off_by_default_for_rna():
    a = counts()
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5)
    assert not np.allclose(norms(a.obsm["X_svd"]), 1.0)
    assert "X_svd_length" not in a.obs.columns and "X_svd_params" not in a.uns


@pytest.mark.parametrize("how", [dict(auto_tfidf=True), dict(modality="ATAC"), dict(modality="tiles")])
def test_unit_length_is_on_by_default_for_atac_and_tiles(how):
    a = counts()
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, **how)
    assert np.allclose(norms(a.obsm["X_svd"]), 1.0, atol=1e-12)
    assert a.uns["X_svd_params"]["unit_length"] is True and "X_svd_length" in a.obs.columns


def test_unit_length_is_on_by_default_when_tfidf_params_are_given():
    from piaso.tools._svd_memsource import tfidf_stats_matrix
    a = counts()
    tf = tfidf_stats_matrix(sp.csr_matrix(a.X))
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, tfidf_params=tf)
    assert np.allclose(norms(a.obsm["X_svd"]), 1.0, atol=1e-12)


def test_an_explicit_choice_always_decides():
    a, b = counts(), counts()
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, auto_tfidf=True, unit_length=False)
    assert not np.allclose(norms(a.obsm["X_svd"]), 1.0) and "X_svd_length" not in a.obs.columns
    piaso.tl.runSVD(b, use_highly_variable=False, n_components=5, unit_length=True)
    assert np.allclose(norms(b.obsm["X_svd"]), 1.0, atol=1e-12)


def _atac_store(path, n_obs=80, n_vars=40, seed=0):
    """Binary peak counts in two groups of cells, each with its own block of peaks, as an ATAC store."""
    import cytome
    rng = np.random.default_rng(seed)
    X = (rng.random((n_obs, n_vars)) < 0.05).astype(np.float32)
    half = n_obs // 2
    X[:half, : n_vars // 2] = np.maximum(X[:half, : n_vars // 2], rng.random((half, n_vars // 2)) < 0.5)
    X[half:, n_vars // 2:] = np.maximum(X[half:, n_vars // 2:], rng.random((n_obs - half, n_vars - n_vars // 2)) < 0.5)
    depth = rng.uniform(0.3, 1.0, size=(n_obs, 1))
    X = (X * (rng.random(X.shape) < depth)).astype(np.float32)
    obs = pd.DataFrame({"coarse": np.where(np.arange(n_obs) < half, "A", "B")},
                       index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame({"highly_variable": np.ones(n_vars, dtype=bool)},
                       index=[f"chr1:{i * 200}-{i * 200 + 150}" for i in range(n_vars)])
    a = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    cytome.from_anndata(a, modality="ATAC", output=str(path)).close()


def test_an_atac_store_gets_unit_length_by_default(tmp_path):
    import cytome
    path = str(tmp_path / "atac.cytome")
    _atac_store(path)
    piaso.tl.runSVD(path, modality="ATAC", measurement="counts", n_components=5, random_state=10)
    piaso.tl.runSVD(path, modality="ATAC", measurement="counts", n_components=5, random_state=10,
                    unit_length=False, key_added="X_svd_eucl")
    ds = cytome.open(path, mode="r")
    try:
        unit = np.asarray(ds.embeddings["ATAC_svd"], dtype=np.float64)
        eucl = np.asarray(ds.embeddings["ATAC_svd_eucl"], dtype=np.float64)
        lengths = np.asarray(ds.cells["ATAC_svd_length"], dtype=np.float64)
        prov = json.loads(ds.provenance.get_for_object("ATAC_svd")[-1]["parameters"])
    finally:
        ds.close()
    assert np.allclose(norms(unit), 1.0, atol=1e-6)
    assert not np.allclose(norms(eucl), 1.0)
    assert np.allclose(unit * lengths[:, None], eucl, rtol=1e-5, atol=1e-6)
    assert prov["unit_length"] is True and prov["length_column"] == "ATAC_svd_length"


def test_an_rna_store_keeps_the_euclidean_embedding_by_default(tmp_path):
    import cytome
    path = str(tmp_path / "rna.cytome")
    cytome.from_anndata(counts(), modality="RNA", output=path).close()
    piaso.tl.runSVD(path, modality="RNA", measurement="counts", use_highly_variable=False, n_components=5)
    ds = cytome.open(path, mode="r")
    try:
        E = np.asarray(ds.embeddings["RNA_svd"], dtype=np.float64)
        cols = set(ds.cells.columns)
    finally:
        ds.close()
    assert not np.allclose(norms(E), 1.0) and "RNA_svd_length" not in cols


def test_leiden_local_on_an_atac_store_scales_each_group_by_default(tmp_path, monkeypatch):
    """The per-group SVDs of the store path: unit length on (the groups are ATAC), the residual stop."""
    from piaso.tools import _clustering
    path = str(tmp_path / "atac.cytome")
    _atac_store(path, n_obs=200, n_vars=60)
    seen = []
    real = _clustering.runSVD

    def spy(*args, **kw):
        out = real(*args, **kw)
        sub = args[0]
        r = norms(np.asarray(sub.embeddings[f"{kw['modality']}_svd"], dtype=np.float64))
        # a cell with no signal in the group's selected peaks stays at the origin
        seen.append((kw.get("unit_length"), kw.get("stop_on"),
                     bool((r > 0).sum() > len(r) // 2 and np.allclose(r[r > 0], 1.0, atol=1e-5))))
        return out

    monkeypatch.setattr(_clustering, "runSVD", spy)
    piaso.tl.leiden_local(path, groupby="coarse", resolution=0.5, dr_method="X_svd", key_added="ll",
                          modality="ATAC", n_components=5, verbosity=0)
    assert seen and all(s == (None, "residuals", True) for s in seen), seen
