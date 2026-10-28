"""One BLAS thread for the dense steps whose bits must not depend on the machine.

OpenBLAS splits a product, a QR or an eigendecomposition among its threads, and
the split changes the rounding: the same SVD gave other bits at one and at two
BLAS threads (max difference 9e-12), enough to flip a tie in a kNN graph and
with it a Leiden label. OpenBLAS starts as many threads as the machine has
cores, so without a limit the embedding depended on the core count (a cluster
job's allocation included). The SVD's passes and the UMAP's graph products run
in Rust, in a fixed order, and do not use BLAS; only the dense steps between
them do, and those are small (a few blocks of ``n_cells x k``).

``threadpoolctl``'s limit is process-wide, so two threads that each run ``with
threadpool_limits(1)`` race: the second saves the first one's 1, the first
restores the machine's count while the second is still inside, and the second
then leaves the process at one thread for good. ``one_blas_thread`` counts the
callers instead: the first in sets the limit, the last out restores it. PIASO
runs SVDs in threads (``runGDR``'s per-batch stage 1), so this matters.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

__all__ = ["one_blas_thread"]

_lock = threading.Lock()
_users = 0
_limiter = None


@contextmanager
def one_blas_thread():
    """Run the block with every BLAS library of the process on one thread."""
    global _users, _limiter
    with _lock:
        if _users == 0:
            try:
                from threadpoolctl import threadpool_limits
                _limiter = threadpool_limits(limits=1, user_api="blas")
            except Exception:          # threadpoolctl missing or unusable: run as is
                _limiter = None
        _users += 1
    try:
        yield
    finally:
        with _lock:
            _users -= 1
            if _users == 0 and _limiter is not None:
                try:
                    _limiter.restore_original_limits()
                finally:
                    _limiter = None
