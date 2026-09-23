"""score() on a column-major matrix gives the row-major answer."""
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pytest

import cytome
import piaso
from cytome.core.measurement import MeasurementLayer


@pytest.fixture
def two_stores(tmp_path):
    rng = np.random.default_rng(4)
    n, g = 400, 900
    X = sp.random(n, g, density=0.06, format="csr", dtype=np.float32, random_state=rng)
    X.data = np.ceil(X.data * 8).astype(np.float32)
    A = ad.AnnData(X=X); A.obs_names = [f"c{i}" for i in range(n)]; A.var_names = [f"g{j}" for j in range(g)]
    csr = str(tmp_path / "csr.cytome"); csc = str(tmp_path / "csc.cytome")
    ds = cytome.from_anndata(A, modality="RNA", output=csr); ds.close()
    ds = cytome.from_anndata(A, modality="RNA", output=csc)
    MeasurementLayer(ds._conn, "RNA_counts").build_feature_index(chunk_size=200)
    ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name='RNA_counts'")
    ds._conn.execute("UPDATE matrix_meta SET n_chunks = 0 WHERE matrix_name='RNA_counts'")
    ds._conn.commit(); ds.close()
    sets = {"a": [f"g{j}" for j in range(0, 40)], "b": [f"g{j}" for j in range(500, 560)]}
    return csr, csc, sets


def _score(path, sets, **kw):
    from piaso.tools import _normalization as N
    ds = cytome.open(path)
    out = N.score(ds, gene_list=sets, layer="counts", modality="RNA", random_seed=1,
                  n_ctrl_set=20, n_nearest_neighbors=10, key_added="s", **kw)
    ds.close()
    return out


def test_scores_agree_between_layouts(two_stores):
    csr, csc, sets = two_stores
    a = _score(csr, sets); b = _score(csc, sets)
    sa = a[0] if isinstance(a, tuple) else a; sb = b[0] if isinstance(b, tuple) else b
    np.testing.assert_allclose(np.asarray(sa), np.asarray(sb), rtol=1e-9, atol=1e-9)


def test_scores_agree_under_a_cell_mask(two_stores):
    csr, csc, sets = two_stores
    mask = np.zeros(400, dtype=bool); mask[::2] = True
    a = _score(csr, sets, cell_mask=mask); b = _score(csc, sets, cell_mask=mask)
    sa = a[0] if isinstance(a, tuple) else a; sb = b[0] if isinstance(b, tuple) else b
    np.testing.assert_allclose(np.asarray(sa), np.asarray(sb), rtol=1e-9, atol=1e-9)


def test_filter_cells_thresholds_agree_between_layouts(two_stores):
    csr, csc, _ = two_stores
    outs = []
    for p in (csr, csc):
        ds = cytome.open(p)
        keep, descs = piaso.preprocessing._filtering._qc_threshold_mask_cytome(
            ds, min_counts=200, max_counts=None, min_features=30, max_features=None,
            modality="RNA", batch_size=64)
        outs.append(np.asarray(keep)); ds.close()
    assert outs[0].sum() > 0 and np.array_equal(outs[0], outs[1])


def test_calculate_metrics_with_feature_masks_agrees_between_layouts(two_stores):
    csr, csc, _ = two_stores
    outs = []
    for p in (csr, csc):
        ds = cytome.open(p)
        piaso.pp.calculateCellMetrics(ds, modality="RNA", measurement="counts", prefix_vars={"g1_": "g1"}, verbose=False)
        cols = [c for c in ds.cells.columns if c.startswith("pct_") or c.startswith("n_") or c.endswith("_counts") or "g1_" in c]
        outs.append({c: np.asarray(ds.cells[c], dtype=float) for c in cols}); ds.close()
    assert outs[0].keys() == outs[1].keys() and any("g1_" in c for c in outs[0])
    for c in outs[0]:
        np.testing.assert_allclose(outs[0][c], outs[1][c], rtol=1e-9, atol=1e-9)


def test_precomputed_knn_takes_the_column_path(two_stores):
    """A precomputed KNN skips pass 1; the layout must still be resolved for pass 2."""
    csr, csc, sets = two_stores
    import cytome
    ds = cytome.open(csr); n_features = ds.RNA.counts.shape[1]; ds.close()
    knn = np.random.default_rng(3).integers(0, n_features, size=(n_features, 10))
    a = _score(csr, sets, precomputed_knn=knn); b = _score(csc, sets, precomputed_knn=knn)
    sa = a[0] if isinstance(a, tuple) else a; sb = b[0] if isinstance(b, tuple) else b
    np.testing.assert_allclose(np.asarray(sa), np.asarray(sb), rtol=1e-9, atol=1e-9)
