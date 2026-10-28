"""Clustering minors of the 2026-10-27 audit (review H3 minors and improvements 4 and 6).

- leiden and neighbors take cell_mask as cell indices too, as documented (it
  was cast with astype(bool)).
- igraph's graph is read straight from the matrix (same edges, same weights,
  same partition).
- leiden_local labels a group's cells by position: by barcode, cells sharing
  one took the label of the last of them.
- leiden, umap and runSVD say when they fall back for lack of the extension.
- leiden_local passes drop_first_component to each group's SVD.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")


def _blobs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(4, 8))
    lab = rng.integers(0, 4, n)
    return centers[lab] + rng.normal(size=(n, 8)), lab


def test_cell_mask_as_indices_equals_the_boolean_mask():
    import piaso
    X, _ = _blobs()
    keep = np.zeros(len(X), dtype=bool)
    keep[::3] = True
    idx = np.flatnonzero(keep)
    a = piaso.tl.neighbors(X, n_neighbors=10, cell_mask=keep, random_state=0)
    b = piaso.tl.neighbors(X, n_neighbors=10, cell_mask=idx, random_state=0)
    np.testing.assert_array_equal(a["knn_indices"], b["knn_indices"])
    full = piaso.tl.neighbors(X, n_neighbors=10, random_state=0)
    la = piaso.tl.leiden(None, knn_result=full, cell_mask=keep, random_state=0)
    lb = piaso.tl.leiden(None, knn_result=full, cell_mask=idx, random_state=0)
    np.testing.assert_array_equal(la, lb)
    with pytest.raises(ValueError, match="more than once"):
        piaso.tl.leiden(None, knn_result=full, cell_mask=np.array([0, 0, 1]))


def test_the_igraph_graph_is_the_one_built_before():
    import igraph as ig
    import piaso
    from piaso.tools._leiden import _igraph_from_adjacency, _SeededRNG
    X, _ = _blobs()
    C = piaso.tl.neighbors(X, n_neighbors=15, random_state=0)["connectivities"]
    # the construction of PIASO 1.2.7
    s, t = C.nonzero()
    m = s <= t
    s, t = s[m], t[m]
    w = np.array(C[s, t]).flatten()
    g_old = ig.Graph(n=C.shape[0], edges=list(zip(s.tolist(), t.tolist())), directed=False)
    g_old.es["weight"] = w.tolist()
    g_new = _igraph_from_adjacency(C)
    assert g_new.get_edgelist() == g_old.get_edgelist()
    assert g_new.es["weight"] == g_old.es["weight"]
    out = []
    for g in (g_old, g_new):
        ig.set_random_number_generator(_SeededRNG(7))
        out.append(g.community_leiden(objective_function="modularity", weights="weight",
                                      resolution=1.0, n_iterations=10).membership)
    assert out[0] == out[1]


def test_leiden_local_labels_cells_that_share_a_barcode(tmp_path):
    """Two latent types in each coarse group; within group A every barcode is
    used twice. Labels must be those of the same store with unique barcodes."""
    import cytome
    import piaso
    rng = np.random.default_rng(1)
    n_per, m = 60, 300

    def two_types():
        Y = (rng.random((2 * n_per, m)) < 0.03).astype(np.float32)
        Y[:n_per, :30] += (rng.random((n_per, 30)) < 0.85) * 4
        Y[n_per:, 30:60] += (rng.random((n_per, 30)) < 0.85) * 4
        Y[:, -30:] += (rng.random((2 * n_per, 30)) < 0.98) * 9
        return Y

    X = np.vstack([two_types(), two_types()])
    n = X.shape[0]
    coarse = np.array(["A"] * (2 * n_per) + ["B"] * (2 * n_per))
    labels = {}
    for name, bcs in (("unique", [f"b{i}" for i in range(n)]),
                      ("shared", [f"a{i % n_per}" for i in range(2 * n_per)] + [f"b{i}" for i in range(2 * n_per)])):
        ds = cytome.create(str(tmp_path / f"{name}.cytome"))
        ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": bcs, "coarse": coarse}))
        ds.set_entity("peaks", pd.DataFrame({"peak_idx": np.arange(m),
                                             "peak_id": [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(m)],
                                             "chr": ["chr1"] * m, "start": np.arange(m) * 1000,
                                             "end_": np.arange(m) * 1000 + 500}))
        ds.add_matrix("ATAC_counts", sp.csr_matrix(X))
        ds.flush()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piaso.tl.leiden_local(ds, groupby="coarse", modality="ATAC", resolution=1.0,
                                  n_components=10, min_cells=10, verbosity=0, tmp_dir=str(tmp_path))
        labels[name] = np.asarray(ds.cells["Leiden_local"]).astype(str)
        ds.close()
    np.testing.assert_array_equal(labels["shared"], labels["unique"])
    assert len(set(labels["unique"][coarse == "A"])) >= 2


def test_leiden_local_passes_drop_first_component(monkeypatch):
    import piaso
    import piaso.tools._clustering as cl
    from anndata import AnnData
    X, lab = _blobs(n=200)
    a = AnnData(np.abs(X), obs=pd.DataFrame({"coarse": np.where(lab < 2, "A", "B")},
                                            index=[f"c{i}" for i in range(len(X))]))
    seen = []
    real = cl.runSVD

    def spy(*args, **kw):
        seen.append(kw.get("drop_first_component"))
        return real(*args, **kw)

    monkeypatch.setattr(cl, "runSVD", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piaso.tl.leiden_local(a, groupby="coarse", n_components=5, verbosity=0,
                              svd_drop_first_component=True)
    assert seen and all(v is True for v in seen)


def test_falling_back_for_lack_of_the_extension_warns(monkeypatch):
    import piaso
    from piaso import _piaso
    import piaso.tools._umap_rust as ur
    X, _ = _blobs()
    knn = piaso.tl.neighbors(X, n_neighbors=10, random_state=0)
    monkeypatch.delattr(_piaso, "leiden_csr")
    with pytest.warns(RuntimeWarning, match="no leiden_csr"):
        piaso.tl.leiden(None, knn_result=knn, random_state=0)
    monkeypatch.setattr(ur, "available", lambda: False)
    from anndata import AnnData
    a = AnnData(np.zeros((len(X), 1), dtype=np.float32))
    a.obsm["X_svd"] = X
    piaso.tl.neighbors(a, use_rep="X_svd", n_neighbors=15, random_state=0)
    with pytest.warns(RuntimeWarning, match="no umap_layout_csr"):
        try:
            piaso.tl.umap(a)
        except ImportError:          # umap-learn too old for this scikit-learn: the warning came first
            pass
