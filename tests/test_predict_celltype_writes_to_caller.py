"""`predictCellTypeByGDR` must write its result to the object it was given.

The function copies the query internally to avoid mutating `.X`
(`adata = adata.copy()`), and every write after that -- including the
prediction itself -- went to the copy. It printed

    All finished. The predicted cell types are saved as `CellTypes_gdr` in adata.obs.

and the caller's AnnData came back without that column, so the prediction was
computed, announced, and discarded. The cytome branch was unaffected because
it writes through a separate handle.

This runs against a real reference because the GDR pipeline inside needs
genuine marker structure; a toy matrix produces a malformed embedding and
tests the fixture instead of the function. It skips where the dataset is not
cached rather than pretending to cover it.
"""
import pathlib

import numpy as np
import pytest

CACHED = pathlib.Path(
    "/data2/rukshina/spatial_tutorials/fixups/adult_cortex_gdr.h5ad")


@pytest.fixture(scope="module")
def query_reference():
    if not CACHED.exists():
        pytest.skip(f"reference not cached at {CACHED}")
    ad = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    ref = ad.read_h5ad(CACHED)
    rs = np.random.RandomState(0)
    m = rs.rand(ref.n_obs) < 0.3
    q, r = ref[m].copy(), ref[~m].copy()
    for a in (q, r):
        a.layers["log1p"] = a.layers.get("infog", a.X)
    sc.pp.neighbors(q, use_rep="X_gdr")
    sc.tl.leiden(q, key_added="Leiden", flavor="igraph", n_iterations=2)
    return q, r


def test_prediction_lands_on_the_caller_s_object(query_reference):
    import piaso
    q, r = query_reference
    q = q.copy()
    before = set(q.obs.columns)

    piaso.tl.predictCellTypeByGDR(
        q, r, layer="log1p", layer_reference="log1p",
        reference_groupby="CellTypes", query_groupby="Leiden")

    assert "CellTypes_gdr" in q.obs.columns, (
        "the function announced success, so the caller's object must carry "
        f"the result; new columns were {sorted(set(q.obs.columns) - before)}")
    pred = q.obs["CellTypes_gdr"].astype(str)
    assert len(pred) == q.n_obs and pred.notna().all()

    # and it must be a real prediction, not a copy of the labels
    agree = float((pred == q.obs["CellTypes"].astype(str)).mean())
    assert 0.5 < agree < 1.0, f"implausible agreement {agree:.3f}"


def test_key_added_is_honoured(query_reference):
    import piaso
    q, r = query_reference
    q = q.copy()
    piaso.tl.predictCellTypeByGDR(
        q, r, layer="log1p", layer_reference="log1p",
        reference_groupby="CellTypes", query_groupby="Leiden",
        key_added="my_pred")
    assert "my_pred" in q.obs.columns
    assert "CellTypes_gdr" not in q.obs.columns
