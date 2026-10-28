"""projectGDR fixes of the 2026-10-27 audit (review H3 findings 1, 2, 3 and 11).

1. The novelty null was the reference's leave-one-out distances within a
   recomputed embedding; for a reference built with batch_key that is another
   space than the stored one the queries are measured against, and held-out
   cells of the reference's own distribution came out novel (MWU p 9e-72).
   The null is now the reference's cells projected as queries are, measured
   against the stored embedding without their own coordinate.
2. The frozen control-gene neighbours were applied by gene position.
3. A store reference never froze its control genes.
11. The frozen-INFOG layer of a store query is written chunk by chunk.

Single-batch self-projection reproduces X_gdr exactly, as before.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")


def _data(seed=3, n=1600, g=800, k=4, batches=False):
    from anndata import AnnData
    rng = np.random.default_rng(seed)
    ct = rng.integers(0, k, n)
    base = rng.gamma(0.6, 1.0, g)
    lam = np.tile(base, (n, 1))
    for t in range(k):
        lam[np.ix_(ct == t, np.arange(t * 40, (t + 1) * 40))] *= 8.0
    batch = np.where(rng.random(n) < 0.5, "b1", "b2")
    if batches:                                   # b2: deeper, and short of type 3
        keep = ~((batch == "b2") & (ct == 3) & (rng.random(n) < 0.8))
        lam = lam * np.where(batch == "b2", 1.8, 1.0)[:, None]
    else:
        keep = np.ones(n, dtype=bool)
    X = sp.csr_matrix(rng.poisson(lam * rng.lognormal(0, 0.4, n)[:, None]).astype(np.float32))[keep]
    m = int(keep.sum())
    obs = pd.DataFrame({"ct": pd.Categorical(ct[keep].astype(str)), "batch": batch[keep]},
                       index=[f"c{i}" for i in range(m)])
    a = AnnData(X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    a.layers["counts"] = a.X.copy()
    return a


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def test_genes_in_another_order_give_the_same_projection():
    import piaso
    full = _data()
    ref, q = full[:1100].copy(), full[1100:].copy()
    piaso.tl.infog(ref, verbosity=0)
    _quiet(piaso.tl.runGDR, ref, groupby="ct", max_workers=1, verbosity=0)
    piaso.tl.infog(q, verbosity=0)
    perm = np.random.default_rng(0).permutation(q.n_vars)
    q2 = q[:, perm].copy()
    _quiet(piaso.tl.projectGDR, q, ref, verbosity=0)
    _quiet(piaso.tl.projectGDR, q2, ref, verbosity=0)
    assert q.uns["X_gdr_projection"]["control_genes_frozen"]
    assert q2.uns["X_gdr_projection"]["control_genes_frozen"]
    np.testing.assert_allclose(q2.obsm["X_gdr"], q.obsm["X_gdr"], atol=1e-10)
    # a query lacking reference genes is scored with its own controls, and says so
    q3 = q[:, 10:].copy()
    _quiet(piaso.tl.projectGDR, q3, ref, verbosity=0)
    assert not q3.uns["X_gdr_projection"]["control_genes_frozen"]


def test_single_batch_self_projection_and_null_are_as_before():
    """Self-projection reproduces X_gdr bit for bit, and the null is the
    leave-one-out distance within the stored embedding, as it was."""
    import piaso
    from sklearn.neighbors import NearestNeighbors
    ref = _data()
    piaso.tl.infog(ref, verbosity=0)
    _quiet(piaso.tl.runGDR, ref, groupby="ct", max_workers=1, verbosity=0)
    q = ref.copy()
    _quiet(piaso.tl.projectGDR, q, ref, verbosity=0)
    np.testing.assert_array_equal(q.obsm["X_gdr"], ref.obsm["X_gdr"])
    X = ref.obsm["X_gdr"]
    d, _ = NearestNeighbors(n_neighbors=16).fit(X).kneighbors(X)
    np.testing.assert_array_equal(ref.uns["gdr_reference"]["novelty_reference"], d[:, 1:].mean(axis=1))


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_held_out_cells_of_the_references_distribution_are_not_novel(batch_key):
    import piaso
    full = _data(seed=11, n=3600, g=500, batches=True)
    piaso.tl.infog(full, verbosity=0)
    half = np.random.default_rng(1).random(full.n_obs) < 0.5
    ref, q = full[half].copy(), full[~half].copy()
    _quiet(piaso.tl.runGDR, ref, groupby="ct", batch_key=batch_key, max_workers=1, verbosity=0)
    _quiet(piaso.tl.projectGDR, q, ref, verbosity=0)
    t = q.uns["X_gdr_projection"]["novelty_test"]
    assert t["mannwhitney_p"] > 1e-3, t
    assert t["ks_stat"] < 0.1, t
    assert q.obs["X_gdr_is_novel"].mean() < 0.03


def test_a_reference_completed_before_the_new_null_is_completed_again():
    import piaso
    ref = _data()
    piaso.tl.infog(ref, verbosity=0)
    _quiet(piaso.tl.runGDR, ref, groupby="ct", max_workers=1, verbosity=0)
    q = ref[:300].copy()
    _quiet(piaso.tl.projectGDR, q, ref, verbosity=0)
    good = np.array(ref.uns["gdr_reference"]["novelty_reference"])
    st = ref.uns["gdr_reference"]
    st.pop("novelty_null")                          # as an older cached state has it
    st["novelty_reference"] = np.zeros_like(good)
    _quiet(piaso.tl.projectGDR, q, ref, verbosity=0)
    np.testing.assert_array_equal(ref.uns["gdr_reference"]["novelty_reference"], good)


@pytest.fixture
def store_reference(tmp_path):
    import cytome
    import piaso
    a = _data()
    ref = a[:1100].copy()
    ds = cytome.from_anndata(ref, modality="RNA", output=str(tmp_path / "ref.cytome"))
    ds.cells["ct"] = ref.obs["ct"].astype(str).values
    ds.flush()
    piaso.tl.infog(ds, modality="RNA", layer="counts", key_added="infog", save_layer=True, verbosity=0)
    _quiet(piaso.tl.runGDR, ds, groupby="ct", layer="infog", n_gene=20, max_workers=1, verbosity=0)
    yield ds, a[1100:].copy()
    ds.close()


def test_a_store_reference_freezes_its_control_genes(store_reference):
    import piaso
    ds, q = store_reference
    # the same 200 cells projected alone and inside a 500-cell query land on the same coordinates
    small, big = q[:200].copy(), q.copy()
    _quiet(piaso.tl.projectGDR, small, ds, counts_layer="counts", verbosity=0)
    _quiet(piaso.tl.projectGDR, big, ds, counts_layer="counts", verbosity=0)
    assert small.uns["X_gdr_projection"]["control_genes_frozen"]
    np.testing.assert_allclose(small.obsm["X_gdr"], big.obsm["X_gdr"][:200], atol=1e-10)
    # and the frozen neighbours are the ones runGDR drew: the store projected onto itself is its X_gdr
    X_self = _quiet(piaso.tl.projectGDR, ds, ds, write_to_cytome=False, verbosity=0)
    np.testing.assert_allclose(X_self, np.asarray(ds.embeddings["X_gdr"]), atol=1e-5)


def test_a_store_query_with_frozen_infog_equals_the_anndata_one(store_reference, tmp_path):
    import cytome
    import piaso
    ds, q = store_reference
    qa = q.copy()
    _quiet(piaso.tl.projectGDR, qa, ds, counts_layer="counts", verbosity=0)
    qs = cytome.from_anndata(q, modality="RNA", output=str(tmp_path / "q.cytome"))
    try:
        X = _quiet(piaso.tl.projectGDR, qs, ds, counts_layer="counts", write_to_cytome=False, verbosity=0)
        assert "RNA_infog_frozen" in [r[0] for r in qs._conn.execute("SELECT matrix_name FROM matrix_meta")]
        np.testing.assert_allclose(X, qa.obsm["X_gdr"], atol=1e-6)
    finally:
        qs.close()
