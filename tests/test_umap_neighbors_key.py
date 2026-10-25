"""umap(neighbors_key=...) must select the graph it names, on AnnData too.

The AnnData branch hard-coded '_neighbors_knn_indices', so a second graph
stored under a different key was unreachable: neighbors(key_added='gdr')
followed by umap(neighbors_key='gdr') silently reused the un-prefixed graph
and, worse, overrode an explicit use_rep with the one that graph was built
on. Two different representations then produced bit-identical embeddings,
which reads as "the method changed nothing" rather than as a wiring bug.
"""
import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
import piaso


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    n = 240
    a = anndata.AnnData(X=rng.random((n, 12)).astype(np.float32))
    # two clearly different representations
    a.obsm["X_svd"] = rng.random((n, 8))
    lab = np.repeat(np.arange(4), n // 4)
    a.obsm["X_gdr"] = (np.eye(4)[lab] * 10 + rng.random((n, 4)) * 0.1)
    a.obs["group"] = lab.astype(str)
    return a


def test_neighbors_key_selects_the_named_graph(adata):
    piaso.tl.neighbors(adata, use_rep="X_svd", n_neighbors=10)
    piaso.tl.neighbors(adata, use_rep="X_gdr", n_neighbors=10, key_added="gdr")
    assert "_neighbors_knn_indices" in adata.uns
    assert "_gdr_knn_indices" in adata.uns

    piaso.tl.umap(adata, use_rep="X_svd", key_added="U_svd", random_state=0)
    piaso.tl.umap(adata, use_rep="X_gdr", key_added="U_gdr",
                  neighbors_key="gdr", random_state=0)

    u1, u2 = adata.obsm["U_svd"], adata.obsm["U_gdr"]
    assert u1.shape == u2.shape
    assert not np.allclose(u1, u2), (
        "two different representations produced the same embedding: "
        "neighbors_key was ignored")


def _rust_layout():
    from piaso.tools import _umap_rust
    return _umap_rust.available()


@pytest.mark.skipif(not _rust_layout(), reason="the extension has no umap_layout_csr: tl.umap uses umap-learn")
def test_explicit_use_rep_is_not_overridden(adata, monkeypatch):
    """The stored graph says X_svd; the call says X_gdr. The call wins."""
    from piaso.tools import _umap_rust
    piaso.tl.neighbors(adata, use_rep="X_svd", n_neighbors=10)
    seen = {}
    real = _umap_rust.embed

    def spy(data, *args, **kwargs):
        seen["X"] = np.asarray(data)
        return real(data, *args, **kwargs)

    monkeypatch.setattr(_umap_rust, "embed", spy)
    piaso.tl.umap(adata, use_rep="X_gdr", key_added="U_a", random_state=0)
    assert np.array_equal(seen["X"], adata.obsm["X_gdr"])
    # umap-learn, given a stored kNN with fewer columns than its n_neighbors
    # (15), computes its own neighbors from the array, so there the
    # representation shows in the picture
    piaso.tl.umap(adata, use_rep="X_gdr", key_added="U_c", random_state=0, backend="umap-learn")
    piaso.tl.umap(adata, use_rep="X_svd", key_added="U_d", random_state=0, backend="umap-learn")
    assert not np.allclose(adata.obsm["U_c"], adata.obsm["U_d"])


def test_unknown_neighbors_key_raises(adata):
    piaso.tl.neighbors(adata, use_rep="X_svd", n_neighbors=10)
    with pytest.raises(KeyError, match="no kNN stored"):
        piaso.tl.umap(adata, use_rep="X_svd", neighbors_key="nope")
