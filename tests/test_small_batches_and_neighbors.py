"""Small inputs to the kNN graph and to batch-aware GDR.

``runGDR(batch_key=...)`` clusters each batch on its own: SVD, a 15-neighbor
graph, Leiden, COSG. A batch of a few cells made the graph fail, because
pynndescent fills the neighbor slots it cannot fill with -1 and the sparse
matrix built from them raised ``negative axis 1 index: -1`` (found on the
intestine multiome, where libraries hold 0 to 38 cells of one compartment).

- ``neighbors`` now caps the neighbors at the number of cells, and replaces a
  graph with -1 entries by the exact one; an input without -1 is unchanged.
- ``runCOSGParallel`` gives a batch too small to cluster no markers of its own
  and says so; its cells are still embedded on the other batches' markers.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
piaso = pytest.importorskip("piaso")
pytest.importorskip("pynndescent")


def test_neighbors_caps_at_the_number_of_cells():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5)).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = piaso.tl.neighbors(X, n_neighbors=15, random_state=0)
    assert any("fewer than n_neighbors" in str(x.message) for x in w)
    idx = np.asarray(res["knn_indices"])
    assert idx.shape == (10, 10) and (idx >= 0).all()
    assert res["distances"].shape == (10, 10)
    # every cell is its own first neighbor and every other cell is listed
    assert all(set(row) == set(range(10)) for row in idx)


def test_neighbors_replaces_a_graph_with_unfilled_slots_by_the_exact_one(monkeypatch):
    import pynndescent
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    k = 8

    class Partial:
        """An NNDescent whose search left some slots unfilled."""
        def __init__(self, data, n_neighbors, **kw):
            nn = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
            d, i = nn.kneighbors(data)
            i[::7, -1] = -1
            d[::7, -1] = np.inf
            self.neighbor_graph = (i, d.astype(np.float32))

    monkeypatch.setattr(pynndescent, "NNDescent", Partial)
    res = piaso.tl.neighbors(X, n_neighbors=k, random_state=0)
    exact_d, exact_i = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(X).kneighbors(X)
    assert (np.asarray(res["knn_indices"]) >= 0).all()
    assert np.array_equal(np.sort(res["knn_indices"], axis=1), np.sort(exact_i, axis=1))
    assert np.allclose(np.sort(res["knn_dists"], axis=1), np.sort(exact_d, axis=1), atol=1e-5)


def test_neighbors_without_unfilled_slots_is_unchanged():
    """The fallback must not touch a graph NNDescent filled."""
    from pynndescent import NNDescent

    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 6)).astype(np.float32)
    res = piaso.tl.neighbors(X, n_neighbors=15, random_state=3)
    ref_i, ref_d = NNDescent(X, n_neighbors=15, metric="euclidean", random_state=3).neighbor_graph
    assert (ref_i >= 0).all()
    assert np.array_equal(res["knn_indices"], ref_i)
    assert np.array_equal(res["knn_dists"], ref_d)


def test_neighbors_refuses_a_single_cell():
    with pytest.raises(ValueError, match="at least 2 cells"):
        piaso.tl.neighbors(np.zeros((1, 3), np.float32), n_neighbors=15)


def test_a_batch_too_small_to_cluster_gives_no_markers_and_no_error():
    from scipy.sparse import csr_matrix
    from piaso.tools import _runGDR

    rng = np.random.default_rng(4)
    X = csr_matrix(rng.poisson(1.0, size=(30, 40)).astype(np.float32))
    obs = pd.DataFrame({"batch": ["big"] * 22 + ["tiny"] * 8})
    payload = _runGDR._setup_shared_memory_sparse(X)
    try:
        payload["obs"] = obs
        b, mg = _runGDR._runCOSGParallel_single_batch(
            "batch", payload, "tiny", None, 10, 2, 30, 0, 1.0, 1.0, 10, True, None, 0)
        assert b == "tiny" and isinstance(mg, pd.DataFrame) and mg.shape[1] == 0
    finally:
        for key in ("shm_data", "shm_indices", "shm_indptr"):
            if key in payload:
                payload[key].close()
                payload[key].unlink()
