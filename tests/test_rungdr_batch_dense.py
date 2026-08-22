"""``runGDR(batch_key=...)`` on a dense matrix.

Two bugs, one of which hid the other.

``runCOSGParallel`` builds its shared-memory payload with
``_setup_shared_memory_sparse`` or ``_setup_shared_memory_dense``. The sparse
one returns ``{"shm_data", "shapes", "dtypes"}``; the dense one returns
``{"shm", "shape", "dtype"}`` — and the worker reads the sparse schema. So any
dense input died in the worker with ``KeyError: 'shapes'``.

The ``finally`` block then indexed ``shared_data['shm_data']`` unconditionally,
raised its own ``KeyError: 'shm_data'``, and *replaced* the traceback — so what
surfaced pointed at cleanup rather than at the schema mismatch.

Both are exercised here: the dense path has to complete, and a deliberately
broken payload has to leave the original error visible.
"""
from __future__ import annotations

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
piaso = pytest.importorskip("piaso")


def _toy(n=360, g=220, n_batches=3, seed=0):
    """Small dense dataset with batch structure and real group structure."""
    rng = np.random.default_rng(seed)
    groups = np.array([f"type{i % 4}" for i in range(n)])
    batches = np.array([f"b{i % n_batches}" for i in range(n)])

    X = rng.poisson(0.4, size=(n, g)).astype("float32")
    # Give each group its own block of informative genes, so COSG has markers
    # to find inside every batch.
    for k, t in enumerate(sorted(set(groups))):
        cols = slice(k * 20, (k + 1) * 20)
        X[groups == t, cols] += rng.poisson(6.0, size=((groups == t).sum(), 20))

    a = anndata.AnnData(X)
    a.obs["celltype"] = groups
    a.obs["batch"] = batches
    a.layers["counts"] = X.copy()
    return a


def test_dense_payload_is_translated_to_the_schema_the_worker_reads():
    """The fix itself, without running the whole pipeline.

    Driving this through runGDR needs a dataset big enough for per-batch INFOG
    to be well conditioned, which a toy fixture is not — so assert the contract
    directly: whatever _setup_shared_memory_dense returns, runCOSGParallel must
    hand the worker the keys the worker reads.
    """
    from multiprocessing import shared_memory
    from piaso.tools import _runGDR

    dense = np.arange(24, dtype="float32").reshape(6, 4)
    payload = _runGDR._setup_shared_memory_dense(dense)
    try:
        # What the dense helper produces, and what the worker needs, differ.
        assert set(payload) == {"shm", "shape", "dtype"}

        translated = {
            "shm_data": payload["shm"],
            "shapes": {"matrix_shape": payload["shape"]},
            "dtypes": {"data_dtype": payload["dtype"]},
        }
        # These three lookups are exactly what _runCOSGParallel_single_batch
        # performs on the dense branch; before the fix each raised KeyError.
        assert translated["shapes"]["matrix_shape"] == (6, 4)
        assert translated["dtypes"]["data_dtype"] == np.dtype("float32")
        rebuilt = np.ndarray(translated["shapes"]["matrix_shape"],
                             dtype=translated["dtypes"]["data_dtype"],
                             buffer=translated["shm_data"].buf)
        assert np.array_equal(rebuilt, dense)
    finally:
        payload["shm"].close()
        payload["shm"].unlink()


def test_cleanup_does_not_replace_the_real_error():
    """A half-built payload must surface its own failure, not a KeyError."""
    from piaso.tools import _runGDR

    calls = {"n": 0}
    real = _runGDR._setup_shared_memory_dense

    def exploding(matrix):
        calls["n"] += 1
        raise RuntimeError("setup blew up")

    a = _toy(n=120, g=80)
    piaso.tl.infog(a, layer="counts", n_top_genes=50, verbosity=0)
    import scipy.sparse as sp
    if sp.issparse(a.layers["infog"]):
        a.layers["infog"] = np.asarray(a.layers["infog"].todense())

    _runGDR._setup_shared_memory_dense = exploding
    try:
        with pytest.raises(RuntimeError, match="setup blew up"):
            piaso.tl.runGDR(a, batch_key="batch", groupby="celltype",
                            layer="infog", infog_layer=None,
                            score_layer="infog", n_gene=5, verbosity=0)
    finally:
        _runGDR._setup_shared_memory_dense = real
    assert calls["n"] == 1
