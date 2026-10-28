"""score() and INFOG on AnnData and on a cytome: one set of rules for both.

Regression tests for the code audit of 2026-10-27 (review I1, findings 9 and 10, and its minors):

- with ``cell_mask``, a single set's result (and a one-set dict with ``key_added``) was written to a
  copy and lost;
- a dense layer crashed the single-set path (``'ArrayView' has no attribute 'multiply'``);
- the multi-set kernel cast the whole matrix's row pointers to int32, which wraps past 2^31 non-zeros:
  the kernel is now called in row blocks that fit, with the same result;
- gene sets with no gene in the data were dropped without a word; a single such set gave an all-NaN
  column; ``score('x.cytome')`` crashed; a repeated var name scored its last column on AnnData and all of
  them on a cytome; integer ``cell_mask`` indices were cast to bool;
- one NaN count made the whole INFOG layer NaN; the in-memory and streaming INFOG broke HVG ties
  differently;
- ``ensure_infog_params(ds, modality)`` computed RNA INFOG whatever was asked and rewrote the highly
  variable genes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import piaso
from piaso.tools import _normalization as N

cytome = pytest.importorskip("cytome")

N_CELLS, N_GENES = 300, 80
GENES = [f"g{i}" for i in range(N_GENES)]
SETS = {"S1": GENES[0:8], "S2": GENES[10:18], "S3": GENES[30:40]}


def _counts(seed=0):
    rng = np.random.default_rng(seed)
    lam = np.full((N_CELLS, N_GENES), 0.5)
    lam[:100, 0:8] = 5.0
    lam[100:200, 10:18] = 5.0
    lam[200:, 30:40] = 5.0
    return rng.poisson(lam).astype(np.float32)


def _adata(X=None):
    X = _counts() if X is None else X
    a = AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
                var=pd.DataFrame(index=GENES[:X.shape[1]]))
    a.layers["counts"] = a.X.copy()
    return a


def _store(path, X=None, symbols=None):
    X = _counts() if X is None else X
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(X.shape[0]),
                                         "barcode": [f"c{i}" for i in range(X.shape[0])]}))
    genes = pd.DataFrame({"gene_idx": np.arange(X.shape[1]), "gene_id": [f"ENSG{i}" for i in range(X.shape[1])],
                          "symbol": symbols if symbols is not None else GENES[:X.shape[1]]})
    ds.set_entity("genes", genes)
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


# ------------------------------------------------------------------ cell_mask

def test_cell_mask_single_set_lands_on_the_callers_object():
    a = _adata()
    mask = np.zeros(N_CELLS, dtype=bool)
    mask[50:250] = True
    piaso.tl.score(a, SETS["S1"], layer="counts", key_added="s1", cell_mask=mask)
    assert "s1" in a.obs and "s1" in a.uns
    vals = a.obs["s1"].to_numpy()
    assert np.isnan(vals[~mask]).all() and np.isfinite(vals[mask]).all()
    # the masked cells score as they do when scored alone
    sub = a[mask].copy()
    piaso.tl.score(sub, SETS["S1"], layer="counts", key_added="s1")
    np.testing.assert_array_equal(vals[mask], sub.obs["s1"].to_numpy())
    table = a.uns["s1"]
    assert table.shape[0] == N_CELLS and table.index.equals(a.obs_names)
    np.testing.assert_array_equal(table.to_numpy()[mask], sub.uns["s1"].to_numpy())
    assert np.isnan(table.to_numpy()[~mask]).all()


def test_cell_mask_one_set_dict_lands_on_the_callers_object():
    a = _adata()
    mask = np.arange(N_CELLS) % 3 == 0
    sm, names, _ = piaso.tl.score(a, {"S2": SETS["S2"]}, layer="counts", key_added="s2", cell_mask=mask)
    assert sm.shape == (mask.sum(), 1)
    vals = a.obs["s2"].to_numpy()
    np.testing.assert_array_equal(vals[mask], sm[:, 0])
    assert np.isnan(vals[~mask]).all()


@pytest.mark.parametrize("backend", ["anndata", "cytome"])
def test_integer_cell_mask_means_indices(tmp_path, backend):
    idx = np.array([3, 40, 41, 120, 299])
    mask = np.zeros(N_CELLS, dtype=bool)
    mask[idx] = True
    data = _adata() if backend == "anndata" else _store(tmp_path / "m.cytome")
    try:
        by_idx, _, _ = piaso.tl.score(data, SETS, layer="counts", cell_mask=idx)
        by_mask, _, _ = piaso.tl.score(data, SETS, layer="counts", cell_mask=mask)
        assert by_idx.shape == (len(idx), len(SETS))
        np.testing.assert_array_equal(by_idx, by_mask)
    finally:
        if backend == "cytome":
            data.close()


# ------------------------------------------------------------------ dense layers

def test_dense_layer_single_set_matches_sparse():
    a = _adata()
    a.layers["dense"] = a.layers["counts"].toarray()
    piaso.tl.score(a, SETS["S3"], layer="counts", key_added="sparse")
    piaso.tl.score(a, SETS["S3"], layer="dense", key_added="dense")
    np.testing.assert_array_equal(a.obs["sparse"].to_numpy(), a.obs["dense"].to_numpy())
    np.testing.assert_array_equal(a.uns["sparse"].to_numpy(), a.uns["dense"].to_numpy())


# ------------------------------------------------------------------ int32 row pointers

@pytest.mark.parametrize("backend", ["anndata", "cytome"])
def test_kernel_called_in_row_blocks_gives_the_same_scores(tmp_path, monkeypatch, backend):
    pytest.importorskip("piaso._piaso")
    data = _adata() if backend == "anndata" else _store(tmp_path / "k.cytome")
    try:
        whole, _, p_whole = piaso.tl.score(data, SETS, layer="counts", compute_pvalues=True, max_workers=2)
        calls = []
        real = N._fused_matmul_reduce_dispatch

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        monkeypatch.setattr(N, "_fused_matmul_reduce_dispatch", spy)
        monkeypatch.setattr(N, "_KERNEL_MAX_NNZ", 997)        # a few hundred rows' worth
        split, _, p_split = piaso.tl.score(data, SETS, layer="counts", compute_pvalues=True, max_workers=2)
        assert len(calls) > 3
        np.testing.assert_array_equal(split, whole)
        np.testing.assert_array_equal(p_split, p_whole)
    finally:
        if backend == "cytome":
            data.close()


def test_kernel_row_blocks_respect_the_limit():
    pytest.importorskip("piaso._piaso")
    from piaso._piaso import fused_matmul_reduce
    rng = np.random.default_rng(0)
    A = sp.random(50, 20, density=0.4, format="csr", random_state=1)
    B = sp.random(20, 2 * 5, density=0.5, format="csr", random_state=2)
    q = rng.random((50, 2))
    seen = []

    def kernel(a_data, a_ind, a_ptr, n_rows, *rest):
        seen.append(int(a_ptr[-1]))
        return fused_matmul_reduce(a_data, a_ind, a_ptr, n_rows, *rest)
    old = N._KERNEL_MAX_NNZ
    try:
        N._KERNEL_MAX_NNZ = 37
        m, p = N._fused_matmul_reduce_rows(kernel, None, A, B.data, B.indices.astype(np.int32),
                                           B.indptr.astype(np.int32), B.shape[1], q,
                                           n_sets=2, n_ctrl_set=5, max_workers=1, compute_pvalues=True)
    finally:
        N._KERNEL_MAX_NNZ = old
    assert max(seen) <= 37 and len(seen) > 1
    m1, p1 = N._fused_matmul_reduce_rows(fused_matmul_reduce, None, A, B.data, B.indices.astype(np.int32),
                                         B.indptr.astype(np.int32), B.shape[1], q,
                                         n_sets=2, n_ctrl_set=5, max_workers=1, compute_pvalues=True)
    np.testing.assert_array_equal(m, m1)
    np.testing.assert_array_equal(p, p1)


# ------------------------------------------------------------------ unmatched sets

@pytest.mark.parametrize("backend", ["anndata", "cytome"])
def test_unmatched_sets_are_named_in_a_warning(tmp_path, backend):
    data = _adata() if backend == "anndata" else _store(tmp_path / "u.cytome")
    sets = dict(SETS, Missing=["nope1", "nope2"])
    try:
        with pytest.warns(UserWarning, match=r"1 of 4 gene set\(s\) match no feature.*'Missing'"):
            sm, names, _ = piaso.tl.score(data, sets, layer="counts")
        assert names == list(SETS) and sm.shape[1] == 3
    finally:
        if backend == "cytome":
            data.close()


def test_single_set_with_no_gene_raises():
    a = _adata()
    with pytest.raises(ValueError, match="none of the 2 gene"):
        piaso.tl.score(a, ["nope1", "nope2"], layer="counts", key_added="x")
    assert "x" not in a.obs


def test_score_accepts_a_cytome_path(tmp_path):
    ds = _store(tmp_path / "p.cytome")
    via_ds, _, _ = piaso.tl.score(ds, SETS, layer="counts")
    ds.close()
    via_path, names, _ = piaso.tl.score(str(tmp_path / "p.cytome"), SETS, layer="counts")
    np.testing.assert_array_equal(via_path, via_ds)


def test_repeated_var_names_score_all_their_columns_as_on_a_cytome(tmp_path):
    X = _counts()
    symbols = list(GENES)
    symbols[5] = "g3"                      # 'g3' names columns 3 and 5
    a = AnnData(sp.csr_matrix(X), var=pd.DataFrame(index=symbols))
    a.layers["counts"] = a.X.copy()
    ds = _store(tmp_path / "d.cytome", X, symbols=symbols)
    try:
        with pytest.warns(UserWarning, match="repeated in var_names"):
            s_ann, _, _ = piaso.tl.score(a, SETS, layer="counts")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_cy, _, _ = piaso.tl.score(ds, SETS, layer="counts", modality="RNA")
        np.testing.assert_allclose(s_ann, s_cy, atol=1e-5)
    finally:
        ds.close()


# ------------------------------------------------------------------ INFOG

def test_infog_refuses_a_nan_count(tmp_path):
    X = _counts()
    X[4, 7] = np.nan
    a = _adata(X)
    with pytest.raises(ValueError, match="NaN or infinite"):
        piaso.tl.infog(a, layer="counts", verbosity=0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        piaso.tl.infog(a, layer="counts", verbosity=0, streaming=True)


def test_infog_hvg_ties_are_broken_the_same_way_in_both_paths():
    rng = np.random.default_rng(5)
    X = np.zeros((200, 60), dtype=np.float32)
    X[:, :20] = rng.poisson(3.0, size=(200, 20))          # 20 variable genes, 40 all-zero
    a = _adata(X)
    b = a.copy()
    piaso.tl.infog(a, layer="counts", n_top_genes=35, verbosity=0)
    piaso.tl.infog(b, layer="counts", n_top_genes=35, verbosity=0, streaming=True)
    np.testing.assert_array_equal(a.var["highly_variable"].to_numpy(), b.var["highly_variable"].to_numpy())
    # equal (zero) variances are taken in gene order
    assert a.var["highly_variable"].to_numpy()[:35].all()


def _store_with_ga(path):
    rng = np.random.default_rng(2)
    ds = _store(path)
    ds.set_entity("GA_genes", pd.DataFrame({"gene_idx": np.arange(12), "gene_id": [f"ga{i}" for i in range(12)]}))
    ds.add_matrix("GA_counts", sp.csr_matrix(rng.poisson(2.0, size=(N_CELLS, 12)).astype(np.float32)))
    hv = np.zeros(N_GENES, dtype=bool)
    hv[[1, 2, 3]] = True
    ds.genes["highly_variable"] = hv
    ds.flush()
    return ds, hv


def test_ensure_infog_params_computes_the_modality_asked_for(tmp_path):
    from piaso.tools import ensure_infog_params
    ds, hv = _store_with_ga(tmp_path / "ga.cytome")
    try:
        params = ensure_infog_params(ds, "GA")
        assert params is not None
        assert len(params["inv_gene_depth"]) == 12          # GA's features, not RNA's 80
        assert ds.metadata.get("GA_infog_params") is not None
        assert ds.metadata.get("RNA_infog_params") is None
        np.testing.assert_array_equal(np.asarray(ds.genes["highly_variable"]).astype(bool), hv)
        # the same parameters infog(ds, modality='GA') stores
        ref = N._infog_streaming(ds, modality="GA", write=False)["infog_params"]
        np.testing.assert_array_equal(params["inv_gene_depth"], ref["inv_gene_depth"])
        np.testing.assert_array_equal(params["cell_depth"], ref["cell_depth"])
    finally:
        ds.close()


def test_ensure_infog_params_for_rna_keeps_the_users_hvgs(tmp_path):
    from piaso.tools import ensure_infog_params
    ds, hv = _store_with_ga(tmp_path / "rna.cytome")
    try:
        params = ensure_infog_params(ds, "RNA")
        assert len(params["inv_gene_depth"]) == N_GENES
        np.testing.assert_array_equal(np.asarray(ds.genes["highly_variable"]).astype(bool), hv)
    finally:
        ds.close()
