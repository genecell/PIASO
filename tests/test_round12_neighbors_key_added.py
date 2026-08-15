"""Round 12 follow-up (2026-05-27): per-variant key plumbing.

cluster.py + piaso.tl.neighbors now honor a non-default ``key_added``
so concurrent sweep_cluster jobs writing to a SHARED cytome don't
race on hardcoded 'connectivities' / 'distances' graph names.

Pre-fix: ``piaso.tl.neighbors(ds, ...)`` always wrote
``ds.add_graph('connectivities', ...)`` regardless of key_added.
Concurrent jobs would overwrite each other's KNN graphs and
downstream Leiden output was contaminated.

Round 12 follow-up: cytome path now uses ``key_added`` as a
prefix the same way the AnnData path did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_cytome(tmp_path, n_obs=40, seed=0):
    import cytome

    p = tmp_path / "test.cytome"
    ds = cytome.create(p)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(8),
        "gene_id": [f"g{i}" for i in range(8)],
    }))
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.standard_normal((n_obs, 8)).astype(np.float32))
    ds.add_matrix("RNA_counts", X)
    # Synthetic SVD embeddings (just need different shapes for the test)
    ds.add_embedding("X_svd_v1", rng.standard_normal((n_obs, 10)).astype(np.float32))
    ds.add_embedding("X_svd_v2", rng.standard_normal((n_obs, 10)).astype(np.float32))
    ds.flush()
    ds.close()
    return p


def test_neighbors_default_key_writes_hardcoded_graph_names(tmp_path):
    """Backward-compat: default key_added='neighbors' still produces
    ``ds.graphs['connectivities']`` and ``ds.graphs['distances']``."""
    import cytome
    import piaso

    p = _build_cytome(tmp_path)
    ds = cytome.open(p)
    piaso.tl.neighbors(ds, use_rep="X_svd_v1", n_neighbors=5,
                       random_state=10)
    keys = list(ds.graphs.keys())
    assert "connectivities" in keys, f"Default key_added must produce 'connectivities'. Got: {keys}"
    assert "distances" in keys
    ds.close()


def test_neighbors_custom_key_writes_prefixed_graph_names(tmp_path):
    """Round 12 follow-up: non-default key_added produces
    ``{key_added}_connectivities`` and ``{key_added}_distances`` in
    the cytome graphs table — mirroring the AnnData path."""
    import cytome
    import piaso

    p = _build_cytome(tmp_path)
    ds = cytome.open(p)
    piaso.tl.neighbors(ds, use_rep="X_svd_v1", n_neighbors=5,
                       random_state=10, key_added="my_variant")
    keys = list(ds.graphs.keys())
    assert "my_variant_connectivities" in keys, (
        f"Custom key_added must produce 'my_variant_connectivities'. "
        f"Got: {keys}"
    )
    assert "my_variant_distances" in keys
    # Default keys must NOT exist (no contamination)
    assert "connectivities" not in keys
    assert "distances" not in keys
    ds.close()


def test_neighbors_two_variants_keep_separate_graphs(tmp_path):
    """Two sequential neighbors() calls with different key_added must
    produce TWO sets of (connectivities, distances) graphs in the
    cytome — not overwrite each other. This pins the race-condition
    fix: concurrent sweep_cluster jobs writing per-variant keys
    don't collide."""
    import cytome
    import piaso

    p = _build_cytome(tmp_path)
    ds = cytome.open(p)
    piaso.tl.neighbors(ds, use_rep="X_svd_v1", n_neighbors=5,
                       random_state=10, key_added="variant_a")
    piaso.tl.neighbors(ds, use_rep="X_svd_v2", n_neighbors=5,
                       random_state=10, key_added="variant_b")
    keys = list(ds.graphs.keys())
    for needed in ("variant_a_connectivities", "variant_a_distances",
                    "variant_b_connectivities", "variant_b_distances"):
        assert needed in keys, f"Missing key {needed!r}. Got: {keys}"
    # And the two variants' connectivities should differ (different SVDs).
    conn_a = ds.graphs["variant_a_connectivities"].to_sparse()
    conn_b = ds.graphs["variant_b_connectivities"].to_sparse()
    assert conn_a.shape == conn_b.shape
    # Different SVDs → some difference in the connectivity pattern.
    same_data = (conn_a.nnz == conn_b.nnz and
                 np.array_equal(conn_a.indices, conn_b.indices) and
                 np.allclose(conn_a.data, conn_b.data))
    assert not same_data, (
        "Different SVDs should produce different KNN graphs. "
        "If identical, the per-variant key plumbing is broken."
    )
    ds.close()
