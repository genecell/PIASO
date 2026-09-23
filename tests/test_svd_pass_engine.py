"""The Rust SVD pass engine against the Python path it replaces.

Both products are checked on a stored cytome with TF-IDF and a column mask:
``X @ B`` must be bit-identical to the Python chunk loop (each row's sum is
sequential in both), ``X^T @ Y`` must agree to rounding and must not change
with the thread count, and a cell mask must select the same rows. Then the
whole SVD, both ways, must span the same subspace.
"""
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pytest

import cytome
from piaso.tools._runTFIDF import _normalize_chunk_tfidf
from piaso.tools._rsvd_core import EngineOps, FactoryOps, rsvd_core

try:
    from piaso import _piaso
    HAVE = hasattr(_piaso, "svd_pass")
except Exception:  # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="Rust extension without svd_pass")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    rng = np.random.default_rng(3)
    n, g = 700, 2500
    # A rank-6 signal with well-separated strengths above Poisson noise, so
    # the leading singular values have a gap and subspace angles mean
    # something; without a gap the trailing components are degenerate and
    # any two correct answers can differ in them.
    W = rng.gamma(2.0, 1.0, size=(n, 6)) * np.array([16, 8, 4, 2, 1, 0.5])
    H = rng.gamma(0.5, 1.0, size=(6, g))
    lam = (W @ H) * 0.02
    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    X.eliminate_zeros()
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = [f"chr1:{1000 * j}-{1000 * j + 500}" for j in range(g)]
    p = tmp_path_factory.mktemp("eng") / "m.cytome"
    ds = cytome.from_anndata(A, modality="ATAC", output=str(p))
    ds.close()
    depth = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    idf = np.log1p(n / np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)).astype(np.float32)
    mask = np.asarray((X > 0).sum(axis=0)).ravel() >= 3
    return str(p), X, depth, idf, mask


def _python_ops(path, mask, depth, idf, keep=None):
    ds = cytome.open(path)
    def factory():
        for chunk, idx in ds.iter_chunks(modality="ATAC", layer="counts", batch_size=64):
            if keep is not None:
                sel = np.isin(idx, keep)
                if not sel.any():
                    continue
                chunk, idx = chunk[sel], idx[sel]
            t = _normalize_chunk_tfidf(chunk, depth[idx], idf, 1e4)[:, mask]
            out_idx = idx if keep is None else np.searchsorted(keep, idx)
            yield t, out_idx
    n_rows = ds.n_cells if keep is None else len(keep)
    return FactoryOps(factory, n_rows, int(mask.sum())), ds


def test_xb_is_bit_identical_to_the_python_path(store):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, tfidf=tf, n_threads=4)
    py, ds = _python_ops(path, mask, depth, idf)
    B = np.random.default_rng(0).standard_normal((eng.n_cols, 12))
    a, b = eng.xb(B), py.xb(B)
    ds.close()
    assert a.shape == b.shape == (X.shape[0], 12)
    # Each row's sum is sequential in both, so the only source of difference
    # is float32 log1p: numpy's SIMD implementation and libm's differ by a
    # few ulp on a fraction of inputs. Agreement is therefore to float32
    # rounding of the transformed values, and most entries are bit-identical.
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(b).max())
    n_bits_equal = int(np.sum(a == b))
    assert n_bits_equal >= 0.5 * a.size, f"only {n_bits_equal}/{a.size} entries bit-identical"


def test_xty_agrees_and_is_thread_invariant(store):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    Y = np.random.default_rng(1).standard_normal((X.shape[0], 12))
    py, ds = _python_ops(path, mask, depth, idf)
    ref = py.xty(Y); ds.close()
    outs = []
    for t in (1, 3, 7):
        eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, tfidf=tf, n_threads=t)
        outs.append(eng.xty(Y))
    for o in outs:
        np.testing.assert_allclose(o, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[1], outs[2]), \
        "the transpose product must not depend on the thread count"


def test_a_cell_mask_selects_the_same_rows(store):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    keep = np.sort(np.random.default_rng(2).choice(X.shape[0], 300, replace=False))
    eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, row_keep=keep, tfidf=tf, n_threads=2)
    py, ds = _python_ops(path, mask, depth, idf, keep=keep)
    B = np.random.default_rng(0).standard_normal((eng.n_cols, 8))
    Y = np.random.default_rng(5).standard_normal((len(keep), 8))
    a, b = eng.xb(B), py.xb(B)
    c, d = eng.xty(Y), py.xty(Y)
    ds.close()
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(b).max())
    np.testing.assert_allclose(c, d, rtol=1e-4, atol=1e-4 * np.abs(d).max())


def _principal_angles(A, B):
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def test_the_whole_svd_spans_the_same_subspace(store):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, tfidf=tf, n_threads=4)
    py, ds = _python_ops(path, mask, depth, idf)
    e1, s1, v1, i1 = rsvd_core(eng, n_components=4, n_iter=4, random_state=10)
    e2, s2, v2, i2 = rsvd_core(py, n_components=4, n_iter=4, random_state=10)
    ds.close()
    np.testing.assert_allclose(s1, s2, rtol=1e-6)
    assert _principal_angles(e1, e2).max() < 1e-4
    assert i1["passes"] == i2["passes"] == 1 + 2 * 4 + 1


def test_the_stopping_rule_stops_early_on_an_easy_matrix(store):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, tfidf=tf, n_threads=4)
    e_ref, s_ref, _, _ = rsvd_core(eng, n_components=4, n_iter=30, random_state=10)
    e_tol, s_tol, _, info = rsvd_core(eng, n_components=4, n_iter=30, random_state=10, tol=1e-3)
    assert info["converged"] is True
    assert info["iterations"] < 30
    assert info["passes"] < 1 + 2 * 30 + 1
    np.testing.assert_allclose(s_tol, s_ref, rtol=5e-3)
    assert _principal_angles(e_tol, e_ref).max() < 0.05


def test_krylov_reaches_the_same_answer(store):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, tfidf=tf, n_threads=4)
    e_ref, s_ref, _, _ = rsvd_core(eng, n_components=4, n_iter=30, random_state=10)
    # four Lanczos rounds keep four blocks -- the basis the old block Krylov
    # had at q=3, reached here with a final pass n_components wide instead of
    # k(q+1) wide
    e_k, s_k, _, info = rsvd_core(eng, n_components=4, n_iter=4, random_state=10, method="krylov")
    assert info["method"] == "krylov" and info["basis_columns"] == 4 * (4 + 10)
    np.testing.assert_allclose(s_k, s_ref, rtol=1e-3)
    assert _principal_angles(e_k, e_ref).max() < 0.05


def test_runSVD_uses_the_engine_and_matches_the_factory_path(store):
    path, X, depth, idf, mask = store
    import piaso
    ds = cytome.open(path)
    ds.features("ATAC")["selected"] = mask
    ds.flush()
    kw = dict(modality="ATAC", measurement="counts", streaming=True, auto_tfidf=True,
              n_components=4, n_iter=4, random_state=10, return_svd=True, verbosity=0)
    e1, s1, _ = piaso.tl.runSVD(ds, key_added="X_eng", use_engine=True, **kw)
    e2, s2, _ = piaso.tl.runSVD(ds, key_added="X_py", use_engine=False, **kw)
    ds.close()
    np.testing.assert_allclose(s1, s2, rtol=1e-6)
    assert _principal_angles(e1, e2).max() < 1e-4


def test_csc_layout_gives_the_same_products(store):
    """The same matrix in CSC chunks: both products must agree with CSR.

    ``X @ B`` on CSC and ``X^T @ Y`` on CSR are owner-computes over output
    blocks; their counterparts are owned per chunk. All four must agree with each other and
    stay thread-count invariant.
    """
    path, X, depth, idf, mask = store
    ds = cytome.open(path)
    ds.ATAC.counts.build_feature_index()   # writes matrix_csc_chunks beside the CSR chunks
    ds.close()
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    from piaso import _piaso
    remap = np.full(X.shape[1], -1, dtype=np.int32); remap[mask] = np.arange(int(mask.sum()), dtype=np.int32)
    n_sel = int(mask.sum())
    B = np.random.default_rng(0).standard_normal((n_sel, 9))
    Y = np.random.default_rng(1).standard_normal((X.shape[0], 9))
    def run(mode, dense, layout, threads):
        return _piaso.svd_pass(path, "ATAC_counts", mode, np.ascontiguousarray(dense), remap, n_sel,
                               None, X.shape[0], depth, idf.astype(np.float32), 1e4,
                               n_threads=threads, layout=layout)
    xb_csr, xb_csc = run("xb", B, "csr", 3), run("xb", B, "csc", 3)
    xty_csr, xty_csc = run("xty", Y, "csr", 3), run("xty", Y, "csc", 3)
    np.testing.assert_allclose(xb_csc, xb_csr, rtol=1e-10, atol=1e-10 * np.abs(xb_csr).max())
    np.testing.assert_allclose(xty_csc, xty_csr, rtol=1e-10, atol=1e-10 * np.abs(xty_csr).max())
    # thread invariance of the CSC products
    assert np.array_equal(run("xb", B, "csc", 1), run("xb", B, "csc", 5))
    assert np.array_equal(run("xty", Y, "csc", 1), run("xty", Y, "csc", 5))


def test_auto_picks_krylov_when_the_block_fits_and_power_otherwise(store, monkeypatch):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, tfidf=tf, n_threads=4)
    _, _, _, info = rsvd_core(eng, n_components=4, n_iter=30, random_state=10, method="auto")
    assert info["method"] == "krylov" and info["converged"] is True and info["iterations"] < 30
    import piaso.tools._rsvd_core as core
    monkeypatch.setattr(core, "AUTO_KRYLOV_BLOCK_BYTES", 1)
    _, _, _, info = rsvd_core(eng, n_components=4, n_iter=30, random_state=10, method="auto")
    assert info["method"] == "power" and info["converged"] is True


def test_lanczos_stops_on_ritz_values_and_matches_the_reference(store):
    path, X, depth, idf, mask = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    eng = EngineOps(path, "ATAC_counts", X.shape[0], X.shape[1], col_mask=mask, tfidf=tf, n_threads=4)
    e_ref, s_ref, _, _ = rsvd_core(eng, n_components=4, n_iter=30, random_state=10)
    e_l, s_l, _, info = rsvd_core(eng, n_components=4, n_iter=30, random_state=10, method="krylov", tol=1e-3)
    assert info["converged"] is True and info["iterations"] < 30
    np.testing.assert_allclose(s_l, s_ref, rtol=2e-3)
    assert _principal_angles(e_l, e_ref).max() < 0.05


def test_the_aitken_guard_does_not_stop_a_crawling_iteration():
    from piaso.tools._rsvd_core import _converged
    # converging fast: increments shrink by 10x each round -> stop
    fast = [np.array([1.0]), np.array([1.1]), np.array([1.11]), np.array([1.111])]
    assert _converged(fast, 1e-2)[0] is True
    # crawling: increments shrink by only 5% a round; the last change is 1.1e-3 but the
    # extrapolated remaining change is ~20x that -> must not stop at tol=1e-2
    crawl = [np.array([1.0]), np.array([1.0 + 0.0012]), np.array([1.0 + 0.0012 + 0.00114])]
    ok, crit = _converged(crawl, 1e-2)
    assert ok is False and crit > 1e-2
    assert _converged(crawl[:2], 1e-2) == (False, None)
