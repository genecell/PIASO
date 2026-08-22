"""Leiden must give the same labels whether or not callers use threads.

igraph's set_random_number_generator installs a PROCESS-GLOBAL generator, so
concurrent Leiden calls reset each other's RNG mid-run. This shipped in the
GDR stage1_workers path and only showed up at scale: a 38-batch cytome came
back at ARI 0.976 against the serial result, while the 5-batch dataset the
original test used happened to agree and hid the bug entirely.
"""
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor

import piaso

N, K = 1500, 15
N_GRAPHS = 8


@pytest.fixture(scope="module")
def knns():
    rng = np.random.default_rng(0)
    return [piaso.tl.neighbors(rng.random((N, 50)), n_neighbors=K, random_state=1)
            for _ in range(N_GRAPHS)]


def _leiden(knn):
    return np.asarray(piaso.tl.leiden(
        None, knn_result=knn, resolution=1.0, random_state=1,
        key_added="t", n_iterations=10,
        cell_mask=np.ones(knn["knn_indices"].shape[0], dtype=bool),
    ))


def test_threaded_leiden_matches_serial(knns):
    """The regression: eight graphs clustered concurrently vs one at a time."""
    serial = [_leiden(k) for k in knns]
    with ThreadPoolExecutor(max_workers=8) as pool:
        threaded = list(pool.map(_leiden, knns))
    for i, (a, b) in enumerate(zip(serial, threaded)):
        assert np.array_equal(a, b), f"graph {i} diverged under threads"


def test_repeated_serial_calls_are_stable(knns):
    """Guards against blaming threads for ordinary non-determinism."""
    assert np.array_equal(_leiden(knns[0]), _leiden(knns[0]))
