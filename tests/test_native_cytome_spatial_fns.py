"""Native cytome support for the spatial/label helpers, and the audit's guards.

The rule under test: a function either works natively on a cytome (no
convert-to-AnnData round trip of the matrix), or it fails immediately with an
error naming the entry point that does.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso

cytome = pytest.importorskip("cytome")


def _line(tmp_path, name="r.cytome"):
    import anndata as ad

    rs = np.random.RandomState(0)
    a = ad.AnnData(X=sp.csr_matrix(rs.poisson(1.0, (10, 4)).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(4)]
    a.obsm["spatial"] = np.column_stack(
        [np.arange(10, dtype=float), np.zeros(10)])
    ds = cytome.from_anndata(a, output=str(tmp_path / name))
    return a, ds


# --------------------------------------------------------------- rotation

def test_rotate_cytome_matches_anndata_bit_for_bit(tmp_path):
    a, ds = _line(tmp_path)
    try:
        piaso.pp.rotateSpatialCoordinates(ds, 37.5, spatial_key="spatial")
        piaso.pp.rotateSpatialCoordinates(a, 37.5, spatial_key="spatial")
        assert np.allclose(np.asarray(ds.embeddings["RNA_spatial"]),
                           a.obsm["spatial"])
    finally:
        ds.close()


def test_rotate_cytome_backup_and_index_stay_in_sync(tmp_path):
    a, ds = _line(tmp_path)
    try:
        before = np.asarray(ds.embeddings["RNA_spatial"]).copy()
        piaso.pp.rotateSpatialCoordinates(
            ds, 90, spatial_key="spatial",
            backup_spatial_key="spatial_original")
        after = np.asarray(ds.embeddings["RNA_spatial"])
        # a horizontal line becomes vertical about the centroid x=4.5
        assert np.allclose(after[:, 0], 4.5)
        assert np.array_equal(
            np.asarray(ds.embeddings["spatial_original"]), before)
        # the R*-tree must answer for the ROTATED coordinates
        assert len(ds.cells_in_region(x=(4.0, 5.0), y=(-10, 10))) == 10
        assert len(ds.cells_in_region(x=(6.0, 10.0), y=(-1, 1))) == 0
    finally:
        ds.close()


def test_rotate_accepts_a_cytome_path(tmp_path):
    a, ds = _line(tmp_path)
    p = str(ds.path)
    before = np.asarray(ds.embeddings["RNA_spatial"]).copy()
    ds.close()
    piaso.pp.rotateSpatialCoordinates(p, 90, spatial_key="spatial")
    piaso.pp.rotateSpatialCoordinates(p, -90, spatial_key="spatial")
    ds = cytome.open(p)
    try:
        assert np.allclose(np.asarray(ds.embeddings["RNA_spatial"]), before)
    finally:
        ds.close()


def test_rotate_cytome_rejects_inplace_false(tmp_path):
    _, ds = _line(tmp_path)
    try:
        with pytest.raises(ValueError, match="AnnData-only"):
            piaso.pp.rotateSpatialCoordinates(ds, 10, spatial_key="spatial",
                                              inplace=False)
    finally:
        ds.close()


def test_rotate_clockwise_inverts_ccw(tmp_path):
    a, _ds = _line(tmp_path)
    _ds.close()
    ref = a.obsm["spatial"].copy()
    piaso.pp.rotateSpatialCoordinates(a, 30, spatial_key="spatial")
    piaso.pp.rotateSpatialCoordinates(a, 30, spatial_key="spatial",
                                      clockwise=True)
    assert np.allclose(a.obsm["spatial"], ref)


def test_rotate_anndata_behaviour_unchanged():
    """The pre-existing AnnData contract survives the dispatch: inplace=False
    returns a copy and leaves the input untouched; a bad key names the
    available ones."""
    import anndata as ad

    rs = np.random.RandomState(1)
    a = ad.AnnData(X=sp.csr_matrix(rs.rand(10, 3)))
    coords = rs.rand(10, 2) * 10
    a.obsm["X_spatial"] = coords.copy()
    out = piaso.pp.rotateSpatialCoordinates(a, 30, inplace=False,
                                            backup_spatial_key="orig")
    assert out is not None and np.allclose(a.obsm["X_spatial"], coords)
    assert np.allclose(out.obsm["orig"], coords)
    with pytest.raises(KeyError, match="Available keys"):
        piaso.pp.rotateSpatialCoordinates(a, 30, spatial_key="nope")


# --------------------------------------------------------------- smoothing

def _labeled(tmp_path):
    import anndata as ad

    rs = np.random.RandomState(1)
    n = 60
    emb = np.zeros((n, 2))
    emb[30:, 0] = 10.0
    emb += rs.rand(n, 2) * 0.1
    labels = np.array(["A"] * 30 + ["B"] * 30, dtype=object)
    labels[3], labels[45] = "B", "A"          # outliers the smoother must fix
    a = ad.AnnData(X=sp.csr_matrix(rs.poisson(1.0, (n, 4)).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(4)]
    a.obs["ct"] = pd.Categorical(labels)
    a.obsm["X_svd"] = emb
    ds = cytome.from_anndata(a, output=str(tmp_path / "s.cytome"))
    return a, ds


def test_smooth_cytome_equals_anndata_and_writes_to_cells(tmp_path):
    a, ds = _labeled(tmp_path)
    try:
        kw = dict(groupby="ct", use_rep="X_svd", k_nearest_neighbors=7,
                  use_existing_adjacency_graph=False, verbosity=0)
        piaso.tl.smoothCellTypePrediction(a, **kw)
        piaso.tl.smoothCellTypePrediction(ds, **kw)
        sm_a = np.asarray(a.obs["ct_smoothed"]).astype(str)
        sm_c = np.asarray(ds.cells["ct_smoothed"]).astype(str)
        assert np.array_equal(sm_a, sm_c)
        assert sm_a[3] == "A" and sm_a[45] == "B"
    finally:
        ds.close()


def test_smooth_cytome_confidence_column(tmp_path):
    a, ds = _labeled(tmp_path)
    try:
        piaso.tl.smoothCellTypePrediction(
            ds, groupby="ct", use_rep="X_svd", k_nearest_neighbors=7,
            return_confidence=True, use_existing_adjacency_graph=False,
            verbosity=0)
        conf = np.asarray(ds.cells["ct_smoothed_confidence"], dtype=float)
        assert conf.shape == (60,) and (conf >= 0).all() and (conf <= 1).all()
    finally:
        ds.close()


def test_smooth_cytome_rejects_inplace_false_and_bad_key(tmp_path):
    _, ds = _labeled(tmp_path)
    try:
        with pytest.raises(ValueError, match="AnnData-only"):
            piaso.tl.smoothCellTypePrediction(ds, groupby="ct",
                                              use_rep="X_svd", inplace=False)
        with pytest.raises(ValueError, match="not found"):
            piaso.tl.smoothCellTypePrediction(ds, groupby="nope",
                                              use_rep="X_svd", verbosity=0)
    finally:
        ds.close()


# ---------------------------------------------------- the audit's guards

def test_anndata_only_internals_reject_a_cytome_immediately(tmp_path):
    """The shared-memory GDR internals are the AnnData path by design —
    runGDR dispatches a cytome to the streaming implementation. Handing them
    a cytome directly must fail at the door, naming runGDR, not twenty frames
    deep in shared-memory setup."""
    _, ds = _line(tmp_path, "g.cytome")
    try:
        with pytest.raises(TypeError, match="runGDR"):
            piaso.tl.runCOSGParallel(ds, batch_key="sample_id")
        with pytest.raises(TypeError, match="runGDR"):
            piaso.tl.calculateScoreParallel_multiBatch(
                ds, batch_key="sample_id", marker_gene=pd.DataFrame(),
                marker_gene_n_groups_indices=[], score_method="piaso")
    finally:
        ds.close()


# ------------------------------------------------ plotEmbedding(cell_mask=)

def test_plot_embedding_cell_mask_plots_a_subset(tmp_path):
    """ROI views without writing a subset file — the counterpart to
    cells_in_region. Boolean mask or integer indices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, ds = _labeled(tmp_path)
    try:
        n = ds.n_cells
        mask = np.zeros(n, dtype=bool)
        mask[:10] = True

        fig, ax = plt.subplots()
        piaso.pl.plotEmbedding(ds, color="ct", basis="X_svd", cell_mask=mask,
                               ax=ax, show=False)
        drawn = sum(c.get_offsets().shape[0] for c in ax.collections)
        assert drawn == 10, f"expected 10 points, drew {drawn}"
        plt.close(fig)

        # integer indices select the same cells
        fig, ax = plt.subplots()
        piaso.pl.plotEmbedding(ds, color="ct", basis="X_svd",
                               cell_mask=np.arange(10), ax=ax, show=False)
        assert sum(c.get_offsets().shape[0] for c in ax.collections) == 10
        plt.close(fig)

        # full mask == no mask
        fig, ax = plt.subplots()
        piaso.pl.plotEmbedding(ds, color="ct", basis="X_svd",
                               cell_mask=np.ones(n, dtype=bool),
                               ax=ax, show=False)
        full = sum(c.get_offsets().shape[0] for c in ax.collections)
        plt.close(fig)
        fig, ax = plt.subplots()
        piaso.pl.plotEmbedding(ds, color="ct", basis="X_svd", ax=ax, show=False)
        assert sum(c.get_offsets().shape[0] for c in ax.collections) == full
        plt.close(fig)
    finally:
        ds.close()


def test_plot_embedding_cell_mask_rejects_bad_input(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    a, ds = _labeled(tmp_path)
    try:
        with pytest.raises(ValueError, match="length"):
            piaso.pl.plotEmbedding(ds, color="ct", basis="X_svd",
                                   cell_mask=np.zeros(3, dtype=bool),
                                   show=False)
        with pytest.raises(ValueError, match="no cells"):
            piaso.pl.plotEmbedding(ds, color="ct", basis="X_svd",
                                   cell_mask=np.zeros(ds.n_cells, dtype=bool),
                                   show=False)
    finally:
        ds.close()
