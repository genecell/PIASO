"""The randomized SVD's dense steps, in parallel, with bits that do not depend on the threads.

Between the passes over the matrix the solver orthonormalises tall blocks
(``n_cells x k``), multiplies them by small ones and reorthogonalises them
against each other. Left to OpenBLAS these steps split their rows among as many
threads as the machine has cores, and the split moves the last bits (a row near
a partition boundary is summed in another order), so the embedding depended on
the core count. Run on one BLAS thread they are reproducible but slow: they
are half of an in-memory SVD's time, and one thread made it 27 to 42 % slower.

Here they run on row blocks of a fixed height (``ROWS``), each block on one BLAS
thread, the blocks on a pool of Python threads (numpy releases the GIL inside
BLAS and LAPACK). A block's arithmetic does not depend on the pool, and partial
sums over blocks are added in block order, so the result is the same at any
number of threads, and the blocks run side by side:

- ``mul`` (``A @ C``) and ``submul`` (``W -= A @ C``): block by block;
- ``tmul`` (``A.T @ B``): one partial per block, added in block order;
- ``qr``: tall-skinny QR (TSQR): Householder QR of each block, then of the
  stacked R factors; each block's Q times its rows of the second Q;
- ``svd_wide`` (the SVD of a ``k x n`` matrix): TSQR of its transpose, then the
  SVD of the small factor.

All of it runs inside ``one_blas_thread`` (see ``_blas``); the caller holds it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["Dense", "ROWS"]

#: Rows per block. Fixed, never derived from the thread count: the blocks are
#: what the arithmetic is cut into, the threads only who does which. 8,192 rows
#: give 7 blocks at 60,000 cells, enough to keep several threads busy; measured
#: on the Lanczos dense steps alone (6 threads, a loaded machine), as fast as
#: 32,768-row blocks at 60,000 rows and 9 % faster at 200,000, and both faster
#: than numpy on 6 BLAS threads (30.1 against 42.3 s; 58.4 against 96.5 s).
ROWS = 1 << 13


def _threads(n_threads):
    """As the Rust passes count them: ``n_threads`` when given, else
    ``RAYON_NUM_THREADS`` when set, else every core this process may use."""
    import os
    if n_threads and int(n_threads) > 0:
        return int(n_threads)
    env = os.environ.get("RAYON_NUM_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _blocks(n, rows=ROWS):
    if n <= 0:
        return [(0, 0)]
    return [(a, min(a + rows, n)) for a in range(0, n, rows)]


class Dense:
    """Row-block dense algebra on ``n_threads`` (0: every core the process may use)."""

    def __init__(self, n_threads=0):
        t = _threads(n_threads)
        self._pool = ThreadPoolExecutor(max_workers=t, thread_name_prefix="piaso-dense") if t > 1 else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _map(self, f, items):
        if self._pool is None or len(items) == 1:
            return [f(x) for x in items]
        return list(self._pool.map(f, items))

    def mul(self, A, C):
        """``A @ C`` for a tall ``A`` (n x k) and a small ``C`` (k x m)."""
        out = np.empty((A.shape[0], C.shape[1]), dtype=np.result_type(A, C))

        def f(ab):
            a, b = ab
            out[a:b] = A[a:b] @ C
        self._map(f, _blocks(A.shape[0]))
        return out

    def submul(self, W, A, C):
        """``W -= A @ C`` in place, ``W`` and ``A`` tall."""
        def f(ab):
            a, b = ab
            W[a:b] -= A[a:b] @ C
        self._map(f, _blocks(W.shape[0]))

    def tmul(self, A, B):
        """``A.T @ B`` for tall ``A`` (n x k) and ``B`` (n x m)."""
        parts = self._map(lambda ab: A[ab[0]:ab[1]].T @ B[ab[0]:ab[1]], _blocks(A.shape[0]))
        out = parts[0].copy()
        for p in parts[1:]:
            out += p
        return out

    def qr(self, Y):
        """Thin QR of a tall ``Y`` (n x k): ``(Q, R)``, ``Q`` n x min(n, k)."""
        blocks = _blocks(Y.shape[0])
        if len(blocks) == 1:
            return np.linalg.qr(Y)
        first = self._map(lambda ab: np.linalg.qr(Y[ab[0]:ab[1]]), blocks)
        Rs = [r for _, r in first]
        Q2, R = np.linalg.qr(np.vstack(Rs))
        offs = np.cumsum([0] + [r.shape[0] for r in Rs])
        Q = np.empty((Y.shape[0], Q2.shape[1]), dtype=Q2.dtype)

        def f(i):
            a, b = blocks[i]
            Q[a:b] = first[i][0] @ Q2[offs[i]:offs[i + 1]]
        self._map(f, list(range(len(blocks))))
        return Q, R
