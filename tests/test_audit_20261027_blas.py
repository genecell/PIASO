"""runSVD's bits do not depend on the BLAS thread count (audit 2026-10-27, review H3 finding 9).

The dense steps between the passes (QR, the Lanczos products, eigh, the final
SVD) ran on as many OpenBLAS threads as the machine had cores, and the same
SVD gave other bits at 1 and 2 threads (9e-12). They now run on one BLAS
thread, whatever the process is set to, and the limiter restores the count
when its last concurrent user leaves.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")
threadpoolctl = pytest.importorskip("threadpoolctl")


def _blas_threads():
    info = [i for i in threadpoolctl.threadpool_info() if i.get("user_api") == "blas"]
    return [i["num_threads"] for i in info]


def _matrix():
    rng = np.random.default_rng(1)
    n, m = 4000, 2000
    X = sp.random(n, m, density=0.03, format="csr", random_state=1,
                  data_rvs=lambda s: rng.poisson(2, s) + 1).astype(np.float32)
    labels = rng.integers(0, 6, n)
    rows, cols = [], []
    for g in range(6):
        r = np.flatnonzero(labels == g)
        c = rng.choice(m, 120, replace=False)
        rows.append(np.repeat(r, len(c)))
        cols.append(np.tile(c, len(r)))
    B = sp.csr_matrix((np.full(sum(len(x) for x in rows), 3.0, dtype=np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, m))
    keep = sp.random(n, m, density=0.3, format="csr", random_state=2) != 0
    X = (X + B.multiply(keep)).tocsr()
    X.data = np.round(X.data)
    X.eliminate_zeros()
    return X


@pytest.mark.parametrize("method", ["krylov", "power"])
def test_svd_bits_do_not_depend_on_the_blas_thread_count(method):
    import piaso
    from anndata import AnnData
    X = _matrix()
    out = {}
    for t in (1, 4):
        with threadpoolctl.threadpool_limits(limits=t, user_api="blas"):
            emb, S, Vt = piaso.tl.runSVD(AnnData(X.copy()), use_highly_variable=False, n_components=30,
                                         method=method, n_iter=6, tol=None, n_threads=2, modality="RNA",
                                         unit_length=False, streaming=True)
        out[t] = (emb, S, Vt)
    for a, b in zip(out[1], out[4]):
        np.testing.assert_array_equal(a, b)


def test_the_limit_is_restored_after_concurrent_users():
    from piaso.tools._blas import one_blas_thread
    before = _blas_threads()
    if not before:
        pytest.skip("no BLAS library visible to threadpoolctl")
    with threadpoolctl.threadpool_limits(limits=3, user_api="blas"):
        outer = _blas_threads()
        seen, errors = [], []

        def work(i):
            try:
                with one_blas_thread():
                    time.sleep(0.01 * (i % 3))
                    seen.append(max(_blas_threads()))
            except Exception as e:          # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert seen and max(seen) == 1
        assert _blas_threads() == outer          # restored once the last one left
    assert _blas_threads() == before


@pytest.mark.parametrize("method", ["krylov", "power"])
def test_svd_bits_do_not_depend_on_the_threads_on_a_matrix_of_several_row_blocks(method):
    """The dense steps run on row blocks of a fixed height side by side (_dense):
    1 and 4 threads give the same embedding on a matrix taller than one block."""
    import piaso
    from anndata import AnnData
    from piaso.tools._dense import ROWS
    rng = np.random.default_rng(3)
    n = ROWS + ROWS // 2 + 17
    X = sp.random(n, 400, density=0.02, format="csr", random_state=3,
                  data_rvs=lambda s: rng.poisson(2, s) + 1).astype(np.float32)
    out = []
    for t in (1, 4):
        emb, S, Vt = piaso.tl.runSVD(AnnData(X.copy()), use_highly_variable=False, n_components=12,
                                     method=method, n_iter=5, tol=None, n_threads=t, modality="RNA",
                                     unit_length=False, streaming=True)
        out.append((emb, S, Vt))
    for a, b in zip(*out):
        np.testing.assert_array_equal(a, b)
    # and it is the matrix's SVD: the leading singular value is a dense solver's
    # (the next ones of this flat random spectrum are not converged in 5 rounds)
    s_ref = np.linalg.svd(X.toarray(), compute_uv=False)[0]
    np.testing.assert_allclose(out[0][1][0], s_ref, rtol=1e-6)


def test_each_components_sign_follows_the_rule():
    """Each component's cell coordinate of largest magnitude is positive, and the
    loadings flip with it: U S Vt is unchanged."""
    import piaso
    from anndata import AnnData
    X = _matrix()
    emb, S, Vt = piaso.tl.runSVD(AnnData(X.copy()), use_highly_variable=False, n_components=10,
                                 method="krylov", n_iter=6, tol=None, n_threads=2, modality="RNA",
                                 unit_length=False, streaming=True)
    top = np.argmax(np.abs(emb), axis=0)
    assert (emb[top, np.arange(emb.shape[1])] > 0).all()
    # the loadings flipped with the coordinates: Vt = U^T X / S, as the solver makes them
    U = emb / S
    np.testing.assert_allclose(np.asarray(X.T @ U) / S, Vt.T, rtol=1e-6, atol=1e-8)
