"""The cytome GDR path must parallelise across batches, and must not reorder.

`_runGDRParallel_cytome` ran its per-batch COSG loop serially while the
AnnData path used a ProcessPoolExecutor across batches. A 35-batch run sat at
106% CPU for over two hours on a 20-core machine.

The ordering test is the one that matters: marker columns are concatenated and
then indexed by cumulative per-batch group counts, so results arriving in
completion order rather than batch order would mis-assign every block without
raising anything.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

cytome = pytest.importorskip("cytome")
anndata = pytest.importorskip("anndata")
import piaso  # noqa: E402


@pytest.fixture
def cyt(tmp_path):
    rng = np.random.default_rng(0)
    n, g = 360, 80
    X = sp.csr_matrix(rng.poisson(0.8, size=(n, g)).astype(np.float32))
    a = anndata.AnnData(X=X)
    a.obs["batch"] = pd.Categorical([f"b{i % 4}" for i in range(n)])
    a.obs["ct"] = pd.Categorical([f"t{i % 3}" for i in range(n)])
    p = tmp_path / "t.cytome"
    ds = cytome.from_anndata(a, output=str(p))
    ds.close()
    piaso.tl.infog(str(p), n_top_genes=40, save_layer=True, verbosity=0)
    return str(p)


def _run(path, workers):
    piaso.tl.runGDR(path, batch_key="batch", groupby="ct", n_gene=5,
                    layer="infog", key_added="X_gdr", max_workers=workers,
                    verbosity=0, write_to_cytome=True)
    ds = cytome.open(path)
    try:
        # GDR stores "X_gdr" un-prefixed while runSVD stores "RNA_svd";
        # accept whichever this build writes rather than pinning the quirk.
        names = list(ds.list_embeddings())
        key = "X_gdr" if "X_gdr" in names else "RNA_gdr"
        return np.asarray(ds.embeddings[key])
    finally:
        ds.close()


def test_parallel_and_serial_agree(cyt, tmp_path):
    """Same input, different worker counts, same embedding. If the batch
    results were collected out of order this is what would catch it."""
    import shutil
    other = str(tmp_path / "serial.cytome")
    shutil.copy2(cyt, other)

    par = _run(cyt, workers=4)
    ser = _run(other, workers=1)
    assert par.shape == ser.shape
    assert np.allclose(par, ser, atol=1e-6), np.abs(par - ser).max()


def test_batch_order_is_preserved_not_completion_order(cyt):
    """Marker column names carry their batch prefix, so the concatenated
    frame's column order is a direct readout of the collection order."""
    ds = cytome.open(cyt)
    try:
        batches = sorted(set(
            r[0] for r in ds._conn.execute("SELECT batch FROM cells")))
    finally:
        ds.close()

    piaso.tl.runGDR(cyt, batch_key="batch", groupby="ct", n_gene=5,
                    layer="infog", key_added="X_gdr", max_workers=4,
                    verbosity=0, write_to_cytome=True)
    ds = cytome.open(cyt)
    try:
        markers = ds.metadata.get("runGDR_marker_genes")
    finally:
        ds.close()
    assert markers is not None
    cols = list(markers) if isinstance(markers, dict) else list(markers.keys())
    seen = [c.split("_")[0] for c in cols]
    first_seen = list(dict.fromkeys(seen))
    assert first_seen == batches, (first_seen, batches)
