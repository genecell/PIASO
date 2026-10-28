"""Randomized SVD over an operator: the two products, and nothing else.

``_rsvd_core`` runs the Halko-Martinsson-Tropp scheme against an object with
two methods, ``xb(B)`` returning ``X @ B`` and ``xty(Y)`` returning
``X^T @ Y``. Where those products come from is the operator's business:
``FactoryOps`` builds them from a chunk iterator in Python, the way the
streaming SVD always has; ``EngineOps`` gets them from the Rust pass engine,
which reads the stored matrix, applies TF-IDF and the column mask, and
multiplies, in one parallel sweep per pass.

Two things are new relative to the loop this replaces.

**A stopping rule.** After ``Z = X^T Q`` the singular values of ``Z`` are the
Rayleigh-Ritz estimates of X's on the current subspace; they rise
monotonically toward the truth, and their relative change between
iterations is free to compute (an SVD of an ``n_features x k`` matrix).
``tol`` stops the iteration when the largest relative change over the
requested components falls below it, with ``n_iter`` as the ceiling. Set
``tol=None`` for the fixed count.

**Block Krylov.** ``method="krylov"`` keeps every iterate's basis and
orthonormalises the concatenation (Musco & Musco, 2015). Same cost per pass,
faster convergence per iteration; the final basis is ``k * (n_iter + 1)``
wide, so it is the choice when passes are dear and cells are not too many.

The final step in both is ``B = (X^T Q)^T``, which is the same product as
the iteration's ``Z``; the last ``Y = X Z`` of the fixed-count loop is kept so
the fixed-count path is arithmetically the one it replaces.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["rsvd_core", "FactoryOps", "EngineOps"]


class FactoryOps:
    """Products from a chunk iterator: ``factory()`` yields ``(chunk, rows)``."""

    def __init__(self, factory, n_rows, n_cols):
        self.factory = factory
        self.n_rows = n_rows
        self.n_cols = n_cols

    def xb(self, B):
        Y = np.zeros((self.n_rows, B.shape[1]), dtype=np.float64)
        for chunk, idx in self.factory():
            Y[idx] = chunk.dot(B) if sparse.issparse(chunk) else chunk @ B
        return Y

    def xty(self, Y):
        Z = np.zeros((self.n_cols, Y.shape[1]), dtype=np.float64)
        for chunk, idx in self.factory():
            Z += chunk.T.dot(Y[idx]) if sparse.issparse(chunk) else chunk.T @ Y[idx]
        return Z


def snapshot_store(ds):
    """Make a cytome readable without file locks, and say whether it is.

    SQLite takes a file lock for every read transaction, and on a network
    file system every lock purges the client's cache of the file, so
    passes that should come from memory come over the network. An
    ``immutable`` reader takes no locks -- but it also cannot see the
    write-ahead log, so the log must be empty first. This checkpoints it
    through the dataset's own connection and then checks the files on
    disk, not a return code: the ``-wal`` must be absent or empty, and so
    must a rollback ``-journal`` (a non-empty one is a hot journal from an
    interrupted write, and the main file may be half-written).

    Returns ``(ok, reason)``; ``reason`` is empty when ``ok``.
    """
    import os
    path = str(getattr(ds, "path", ""))
    if not path:
        return False, "the dataset has no path"
    conn = ds._conn
    try:
        ds.flush()
        conn.commit()
        # Do not wait on a reader that pins the log: a fallback must be
        # immediate, not a minute of busy-waiting. The checkpoint reports
        # (busy, frames in the log, frames checkpointed); when every frame
        # is checkpointed the main file is complete whether or not the log
        # could also be truncated, and that is what an immutable reader needs.
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.execute("PRAGMA busy_timeout=0")
        try:
            busy, in_log, done = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        finally:
            conn.execute(f"PRAGMA busy_timeout={int(timeout)}")
    except Exception as e:  # read-only mount, closed connection
        return False, f"the write-ahead log could not be checkpointed ({e})"
    if in_log > 0 and done < in_log:
        return False, (f"{in_log - done} of {in_log} log frames could not be checkpointed"
                       + (" (another connection holds a read transaction)" if busy else ""))
    journal = path + "-journal"
    if os.path.exists(journal) and os.path.getsize(journal) > 0:
        return False, "a rollback journal is present and not empty"
    return True, ""


class EngineOps:
    """Products from the Rust pass engine over a stored cytome matrix.

    Parameters
    ----------
    path, matrix_name
        The cytome file and the matrix in it (``"ATAC_counts"``).
    col_mask
        Boolean over the matrix's columns, or ``None`` for all.
    row_keep
        Sorted global row indices to keep, or ``None`` for all. Output rows
        are in this order.
    tfidf
        ``{"cell_depth", "idf", "scale_factor"}`` or ``None`` for raw values.
    n_threads
        Rayon threads for the pass; ``0`` uses every core. The result does
        not depend on it: the engine sums every output element in one fixed
        order whatever the thread count.
    source
        A chunk source for a store this process holds open (``ds.chunk_source``):
        the engine computes on the blobs it hands over and never opens the
        file. ``path`` and ``snapshot`` are then unused. Without it the
        engine opens the store itself, which is right only when no
        connection of this process holds it (the command-line tools).
    snapshot
        Read the store with no file locks (SQLite ``immutable``). Only
        after :func:`snapshot_store` has emptied the write-ahead log and
        returned True; an immutable reader cannot see the log. The bytes
        read are the same either way, so the result is identical.
    batch_bytes
        Bucketed-chunk memory one batch of handed-over chunks may hold
        (``0``: the engine's 512 MB). Speed and memory only, never the result.
    """

    def __init__(self, path, matrix_name, n_rows_matrix, n_cols_matrix,
                 col_mask=None, row_keep=None, tfidf=None,
                 n_threads=0, snapshot=False, source=None, batch_bytes=0):
        from .. import _piaso
        self._pass = _piaso.svd_pass
        self._pass_source = getattr(_piaso, "svd_pass_source", None)
        self.path = str(path)
        self.snapshot = bool(snapshot)
        self.matrix_name = matrix_name
        self.source = source
        if source is not None and self._pass_source is None:
            raise RuntimeError("piaso._piaso has no svd_pass_source; rebuild the extension")
        # Which layout the pass will take, decided in Rust by the size of the
        # buffer the scattered product writes into (see svd_pass.rs). Kept
        # here so the caller can say so, and can tell a user with a row-major
        # peak matrix what a column-major store would buy.
        try:
            if source is not None:
                _nr, _nc, _dt, has_csr, has_csc = source.meta()
                lay = _piaso.svd_layout_rule(_nr, _nc, has_csr, has_csc)
            else:
                lay, has_csr, has_csc, _nr, _nc = _piaso.svd_layout(self.path, matrix_name, snapshot=self.snapshot)
            self.layout, self.has_csr, self.has_csc = lay, bool(has_csr), bool(has_csc)
        except Exception:
            self.layout, self.has_csr, self.has_csc = "auto", None, None
        if col_mask is None:
            self.remap = np.arange(n_cols_matrix, dtype=np.int32)
            self.n_cols = int(n_cols_matrix)
        else:
            m = np.asarray(col_mask, dtype=bool)
            if m.shape[0] != n_cols_matrix:
                raise ValueError(f"col_mask has {m.shape[0]} entries, matrix has {n_cols_matrix} columns")
            self.remap = np.full(n_cols_matrix, -1, dtype=np.int32)
            self.remap[m] = np.arange(int(m.sum()), dtype=np.int32)
            self.n_cols = int(m.sum())
        if row_keep is None:
            self.row_map = None
            self.n_rows = int(n_rows_matrix)
        else:
            keep = np.asarray(row_keep, dtype=np.int64)
            self.row_map = np.full(n_rows_matrix, -1, dtype=np.int32)
            self.row_map[keep] = np.arange(len(keep), dtype=np.int32)
            self.n_rows = int(len(keep))
        if tfidf is None:
            self.depth, self.idf, self.scale = None, None, 1.0
        else:
            self.depth = np.ascontiguousarray(tfidf["cell_depth"], dtype=np.float64)
            self.idf = np.ascontiguousarray(tfidf["idf"], dtype=np.float32)
            self.scale = float(tfidf.get("scale_factor", 1e4))
            # The Python transform (`_normalize_chunk_tfidf`, run_TFIDF) skips
            # the multiply for a scale of 0, i.e. reads it as 1; the engine
            # multiplied by 0 and factorised a matrix of zeros.
            if self.scale == 0.0:
                self.scale = 1.0
            if not (np.isfinite(self.depth).all() and np.isfinite(self.idf).all()):
                raise ValueError("tfidf cell_depth / idf hold values that are not finite")
            if self.depth.shape[0] != n_rows_matrix or self.idf.shape[0] != n_cols_matrix:
                raise ValueError("tfidf cell_depth / idf do not match the matrix shape")
        self.n_threads = int(n_threads)
        self.batch_bytes = int(batch_bytes)
        #: Wall seconds spent inside the engine's passes, summed, so a
        #: caller can say how much of a run was the store and how much
        #: the numpy between passes.
        self.seconds = 0.0
        self.passes = 0

    def _call(self, mode, dense):
        import time
        t = time.perf_counter()
        try:
            return self._pass_raw(mode, dense)
        finally:
            self.seconds += time.perf_counter() - t
            self.passes += 1

    def _pass_raw(self, mode, dense):
        if self.source is not None:
            return self._pass_source(
                self.source, mode,
                np.ascontiguousarray(dense, dtype=np.float64),
                self.remap, self.n_cols,
                self.row_map, self.n_rows,
                self.depth, self.idf, self.scale,
                self.n_threads, batch_bytes=self.batch_bytes,
            )
        return self._pass(
            self.path, self.matrix_name, mode,
            np.ascontiguousarray(dense, dtype=np.float64),
            self.remap, self.n_cols,
            self.row_map, self.n_rows,
            self.depth, self.idf, self.scale,
            self.n_threads, snapshot=self.snapshot,
        )

    def xb(self, B):
        return self._call("xb", B)

    def xty(self, Y):
        return self._call("xty", Y)


# Block Krylov keeps k*(q+1) columns of n_rows doubles. Under "auto" it is
# chosen when that block is small, and power iteration otherwise; both stop
# on the same rule. Measured on MC peaks against a 30-iteration reference:
# Krylov q=8 (18 passes) reaches 0.986 kNN-15 overlap; power with tol=1e-3
# reaches 0.979 in 44 passes.
AUTO_KRYLOV_QMAX = 30
AUTO_KRYLOV_BLOCK_BYTES = 1 << 30
AUTO_TOL = 1e-3
# The tolerance of stop_on='residuals': the largest relative residual
# ||X X^T u - s^2 u|| / s^2 of the components computed (benchmark/svd_stop/).
AUTO_RESIDUAL_TOL = 1e-2
STOP_RULES = ("singular_values", "residuals")


def _relative_residuals(R, theta):
    """Column norms of ``R`` over ``theta`` (the Ritz values of ``X X^T``)."""
    return np.linalg.norm(R, axis=0) / np.maximum(np.abs(theta), np.finfo(float).tiny)


def _converged(hist, tol):
    """Stopping test on a history of leading singular-value estimates.

    The estimates rise monotonically toward the truth, so the last relative
    change alone is a poor test: a slowly converging iteration also changes
    little per round. With three points the ratio of successive increments
    estimates the contraction factor ``rho`` per component, and Aitken's
    extrapolation puts the *remaining* change at about ``d * rho / (1 - rho)``.
    That is what is compared with ``tol`` -- so "crawling" is not mistaken
    for "done". Two points are never enough to stop.

    Returns ``(converged, criterion)``.
    """
    if tol is None or len(hist) < 3:
        return False, None
    tiny = np.finfo(float).tiny
    s2, s1, s0 = hist[-1], hist[-2], hist[-3]
    d1 = np.abs(s2 - s1) / np.maximum(np.abs(s2), tiny)
    d0 = np.abs(s1 - s0) / np.maximum(np.abs(s1), tiny)
    rho = np.where(d0 > 0, np.minimum(d1 / np.maximum(d0, tiny), 0.999), 0.0)
    remaining = d1 * rho / (1.0 - rho)
    crit = float(np.max(np.maximum(d1, remaining)))
    return crit < tol, crit


def _block_tridiagonal(A_blocks, B_blocks):
    """Assemble the Lanczos matrix T from its diagonal and sub-diagonal blocks."""
    k = A_blocks[0].shape[0]
    m = len(A_blocks)
    T = np.zeros((m * k, m * k))
    for i, A in enumerate(A_blocks):
        T[i * k:(i + 1) * k, i * k:(i + 1) * k] = A
    for i, B in enumerate(B_blocks):
        T[(i + 1) * k:(i + 2) * k, i * k:(i + 1) * k] = B
        T[i * k:(i + 1) * k, (i + 1) * k:(i + 2) * k] = B.T
    return T


def _block_lanczos(ops, n_components, k, q_max, rng, tol, log, verbosity,
                   stop_on="residuals", dense=None):
    """Randomized block Lanczos on ``X X^T``: the Krylov subspace, with structure.

    Each round applies ``X^T`` then ``X`` to the newest block only (two passes
    of ``k`` columns, the same as a power-iteration round), orthogonalises the
    result against every block kept, and records the ``k x k`` coefficient
    blocks of the block-tridiagonal ``T = Q^T X X^T Q``. The eigenvalues of
    ``T`` are the Ritz values -- the current ``sigma^2`` estimates on the whole
    subspace -- so the stopping test costs no pass. At the end the singular
    vectors come from ``T`` too, and the right ones from one pass of
    ``n_components`` columns.

    ``stop_on='residuals'`` stops instead when every computed Ritz pair's
    relative residual ``||X X^T u - s^2 u|| / s^2`` is below ``tol``. The
    residual costs no pass either: the block Lanczos relation
    ``X X^T Q = Q T + V_next B E^T`` makes it ``||B y_last||``, with ``y_last``
    the last block of the pair's eigenvector of ``T`` and ``B`` the triangular
    factor of the next block, which the next round needs anyway. Singular
    values converge before vectors do, so this is the stricter rule.
    """
    n_rows, n_cols = ops.n_rows, ops.n_cols
    Omega = rng.standard_normal((n_cols, k)).astype(np.float64)
    log("Y = X @ Omega")
    V, _ = dense.qr(ops.xb(Omega))
    del Omega
    blocks, A_blocks, B_blocks, hist = [V], [], [], []
    V_prev, B_prev = None, None
    converged, crit, rounds = False, None, 0
    for i in range(q_max):
        log(f"Z = X^T V (lanczos round {i + 1})")
        Z = ops.xty(V)
        log(f"W = X Z (lanczos round {i + 1})")
        W = ops.xb(Z)
        del Z
        A = dense.tmul(V, W)
        A = 0.5 * (A + A.T)
        dense.submul(W, V, A)
        if V_prev is not None:
            dense.submul(W, V_prev, B_prev.T)
        # Full reorthogonalisation, twice: in floating point the blocks
        # drift, and every Lanczos in production does this.
        for _ in range(2):
            for Vj in blocks:
                dense.submul(W, Vj, dense.tmul(Vj, W))
        A_blocks.append(A)
        rounds = i + 1
        V_next, B = dense.qr(W)
        if stop_on == "residuals":
            theta_all, Y_T = np.linalg.eigh(_block_tridiagonal(A_blocks, B_blocks))
            order = np.argsort(theta_all)[::-1][:n_components]
            theta = theta_all[order]
            hist.append(np.sqrt(np.maximum(theta, 0.0)))
            if tol is not None:
                rel = _relative_residuals(B @ Y_T[-k:, order], theta)
                crit = float(rel.max())
                converged = crit < tol
        else:
            theta = np.linalg.eigvalsh(_block_tridiagonal(A_blocks, B_blocks))[::-1][:n_components]
            hist.append(np.sqrt(np.maximum(theta, 0.0)))
            converged, crit = _converged(hist, tol)
        if verbosity > 1 and crit is not None:
            print(f"    round {i + 1}: convergence criterion {crit:.2e}")
        if converged or i == q_max - 1:
            break
        if np.linalg.norm(B) < 1e-12 * max(1.0, float(np.abs(hist[-1][0]))):
            converged = True     # invariant subspace found: nothing left to add
            break
        B_blocks.append(B)
        blocks.append(V_next)
        V_prev, B_prev, V = V, B, V_next
    theta, U_T = np.linalg.eigh(_block_tridiagonal(A_blocks, B_blocks))
    order = np.argsort(theta)[::-1][:n_components]
    S = np.sqrt(np.maximum(theta[order], 0.0))
    # U = [V_0 ... V_q] @ U_T[:, order], block by block, added in block order
    # (without the n_rows x (q + 1) k copy np.hstack made)
    U = dense.mul(blocks[0], U_T[:k, order])
    for j in range(1, len(blocks)):
        U += dense.mul(blocks[j], U_T[j * k:(j + 1) * k, order])
    basis_columns = sum(b.shape[1] for b in blocks)
    log("V = X^T U / S")
    Vt = (ops.xty(U) / np.maximum(S, np.finfo(float).tiny)).T
    info = {"iterations": rounds, "converged": converged if tol is not None else None,
            "method": "krylov", "criterion": crit, "basis_columns": int(basis_columns),
            "stop_on": stop_on}
    return U * S, S, Vt, info



def _fix_signs(emb, Vt):
    """Each component's sign fixed by a rule: its cell coordinate of largest
    magnitude positive (the first such cell on a tie), the loadings flipped with
    it. A singular pair is only defined up to its sign, and which one a solver
    returns follows the internals of its QR and SVD; with the rule it follows
    the data alone. (scikit-learn's ``svd_flip`` with ``u_based_decision``.)"""
    if emb.size == 0:
        return emb, Vt
    top = np.argmax(np.abs(emb), axis=0)
    signs = np.sign(emb[top, np.arange(emb.shape[1])])
    signs[signs == 0] = 1.0
    emb = emb * signs
    if Vt is not None:
        Vt = Vt * signs[:, None]
    return emb, Vt


def rsvd_core(ops, n_components, n_iter, random_state, oversampling=10,
              tol=None, method="power", verbosity=0, stop_on="residuals"):
    """Randomized SVD of the operator; returns ``(U*S, S, Vt, info)``.

    The dense steps between the passes (QR, the Lanczos products and
    reorthogonalisation, the eigen- and singular value decompositions) run on
    one BLAS thread (``_blas.one_blas_thread``), so the result does not depend
    on how many cores OpenBLAS sees: at one and at two threads they differed in
    the last bits (9e-12), enough to flip a kNN tie. The tall ones (QR, the
    products and the reorthogonalisation) run on row blocks of a fixed height
    side by side (``_dense.Dense``), so they keep their speed. The passes
    themselves are Rust (``EngineOps``) or scipy's sparse products, neither of
    which uses BLAS. Each component's sign follows a rule (``_fix_signs``): its
    cell coordinate of largest magnitude is positive.
    """
    from ._blas import one_blas_thread
    from ._dense import Dense
    with one_blas_thread(), Dense(getattr(ops, "n_threads", 0)) as dense:
        return _rsvd(ops, n_components, n_iter, random_state, oversampling=oversampling,
                     tol=tol, method=method, verbosity=verbosity, stop_on=stop_on, dense=dense)


def _rsvd(ops, n_components, n_iter, random_state, oversampling=10,
          tol=None, method="power", verbosity=0, stop_on="residuals", dense=None):
    """The body of :func:`rsvd_core`.

    ``info`` reports the passes taken and, under ``tol``, whether and when
    the iteration converged. ``method="auto"`` picks block Krylov with
    ``AUTO_KRYLOV_Q`` iterations when its block fits ``AUTO_KRYLOV_BLOCK_BYTES``,
    and power iteration with ``tol`` (default ``AUTO_POWER_TOL``) and
    ``n_iter`` as the ceiling otherwise.

    ``stop_on`` chooses what ``tol`` bounds: ``'singular_values'`` (the
    extrapolated change of the leading singular values between rounds) or
    ``'residuals'`` (every computed pair's relative residual
    ``||X X^T u - s^2 u|| / s^2``; under ``'auto'`` its default tolerance is
    ``AUTO_RESIDUAL_TOL``).
    """
    if stop_on not in STOP_RULES:
        raise ValueError(f"stop_on must be one of {STOP_RULES}, got {stop_on!r}")
    if dense is None:
        from ._dense import Dense
        dense = Dense(1)
    n_rows, n_cols = ops.n_rows, ops.n_cols
    k = n_components + oversampling
    if method == "auto":
        if tol is None:
            tol = AUTO_RESIDUAL_TOL if stop_on == "residuals" else AUTO_TOL
        block = n_rows * k * (AUTO_KRYLOV_QMAX + 1) * 8
        if block <= AUTO_KRYLOV_BLOCK_BYTES:
            # AUTO_KRYLOV_QMAX is a *default* ceiling for when the caller
            # expressed no opinion, not a licence to ignore one they did.
            # Overwriting n_iter unconditionally turned an explicit
            # `n_iter=12` into 30 rounds without saying so.
            method = "krylov"
            n_iter = AUTO_KRYLOV_QMAX if n_iter is None else min(int(n_iter),
                                                                AUTO_KRYLOV_QMAX)
        else:
            method = "power"
            if n_iter is None:
                n_iter = AUTO_KRYLOV_QMAX
        if verbosity > 0:
            label = ("block Lanczos" if method == "krylov" else "power iteration")
            print(f"  Solver: {label} chosen by method='auto' "
                  f"(tol={tol} on the {stop_on.replace('_', ' ')}, at most {n_iter} rounds = "
                  f"{2 + 2 * n_iter} passes)")
    elif verbosity > 0 and method == "krylov":
        # Said for every solver, so a log always names what ran. A pass is
        # one product over the store: the sketch, two per round, and the
        # final projection -- 2 + 2 * rounds.
        print(f"  Solver: block Lanczos"
              + (f" (tol={tol}, at most {n_iter} rounds = {2 + 2 * n_iter} passes)"
                 if tol is not None else f" ({n_iter} rounds = {2 + 2 * n_iter} passes)"))
    rng = np.random.RandomState(random_state)
    n_pass = 0

    def log(desc):
        nonlocal n_pass
        n_pass += 1
        if verbosity > 0:
            print(f"  SVD pass {n_pass}: {desc}")

    info = {"passes": 0, "iterations": 0, "converged": None, "method": method, "stop_on": stop_on}

    def _report(info):
        if verbosity > 0:
            label = ("block Lanczos" if info["method"] == "krylov"
                     else "power iteration")
            stopped = ("converged" if info.get("converged")
                       else "reached the round limit")
            print(f"  Solver done: {label}, {info['iterations']} rounds, "
                  f"{info['passes']} passes ({stopped})")

    if method == "krylov":
        emb, S, Vt, info = _block_lanczos(ops, n_components, k, n_iter, rng, tol, log, verbosity,
                                          stop_on=stop_on, dense=dense)
        info["passes"] = n_pass
        _report(info)
        emb, Vt = _fix_signs(emb, Vt)
        return emb, S, Vt, info
    elif method == "power":
        Omega = rng.standard_normal((n_cols, k)).astype(np.float64)
        log("Y = X @ Omega")
        Y = ops.xb(Omega)
        del Omega  # n_cols x k float64; never used again
        hist = []
        converged = False
        Q = None
        for i in range(n_iter):
            Q, _ = dense.qr(Y)
            log(f"Z = X^T Q (power iter {i + 1})")
            Z = ops.xty(Q)
            if tol is not None and stop_on == "singular_values":
                hist.append(np.linalg.svd(Z, compute_uv=False)[:n_components])
                converged, crit = _converged(hist, tol)
                if verbosity > 1 and crit is not None:
                    print(f"    iteration {i + 1}: convergence criterion {crit:.2e}")
                if converged:
                    info["iterations"] = i + 1
                    info["criterion"] = crit
                    break
            log(f"Y = X Z (power iter {i + 1})")
            Y = ops.xb(Z)
            if tol is not None and stop_on == "residuals":
                # The Ritz pairs of span(Q): Z^T = U_B S Vt, u = Q U_B. The pass
                # just made gives X X^T u = Y U_B, so their residuals cost no
                # pass: Y U_B - Q U_B S^2.
                U_B, S_r, _ = np.linalg.svd(Z.T, full_matrices=False)
                U_B, S_r = U_B[:, :n_components], S_r[:n_components]
                rel = _relative_residuals(dense.mul(Y, U_B) - dense.mul(Q, U_B * S_r ** 2), S_r ** 2)
                crit = float(rel.max())
                if verbosity > 1:
                    print(f"    iteration {i + 1}: largest relative residual {crit:.2e}")
                if crit < tol:
                    converged = True
                    info["iterations"] = i + 1
                    info["criterion"] = crit
                    break
        if not converged:
            Q, _ = dense.qr(Y)
            log("B = Q^T X")
            Z = ops.xty(Q)
            info["iterations"] = n_iter
        info["converged"] = converged if tol is not None else None
        B = Z.T
    else:
        raise ValueError(f"method must be 'power' or 'krylov', got {method!r}")

    U_B, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = dense.mul(Q, U_B[:, :n_components])
    info["passes"] = n_pass
    _report(info)
    emb, Vt = _fix_signs(U * S[:n_components], Vt[:n_components])
    return emb, S[:n_components], Vt, info
