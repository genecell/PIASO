"""runGDR's reference can be written to an h5ad file when cell type labels contain "/".

anndata >= 0.13 refuses a "/" in an h5ad key (``anndata.settings.disallow_forward_slash_in_h5ad``),
and the reference kept its marker sets in ``uns`` as a DataFrame named by cell type, so a label
such as "L2/3 IT" left the whole AnnData unwritable after runGDR. Such names are now stored with
"/" written as "%2F" and the exact names beside them; projectGDR reads them back unchanged.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import piaso
from piaso.tools import _projectGDR as P

GROUPS = ["L2/3 IT", "L5/6 NP", "Astro", "Micro/PVM"]


def _slash_keys(adata):
    """Every key or DataFrame column with "/" that an h5ad file would hold."""
    bad = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "/" in str(k):
                    bad.append(f"{path}/{k}")
                walk(v, f"{path}/{k}")
        elif isinstance(obj, pd.DataFrame):
            bad.extend(f"{path}:{c}" for c in obj.columns if "/" in str(c))
    walk(dict(adata.uns), "uns")
    return bad


def _data(groups, seed=0, batches=None):
    rng = np.random.default_rng(seed)
    n_per, n_genes = 120, 300
    X = rng.negative_binomial(2, 0.6, size=(n_per * len(groups), n_genes)).astype(np.float32)
    for i in range(len(groups)):
        X[i * n_per:(i + 1) * n_per, i * 40:(i + 1) * 40] += rng.poisson(6, size=(n_per, 40))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"G{j}" for j in range(n_genes)]
    a.obs["ct"] = pd.Categorical(np.repeat(groups, n_per), categories=groups)
    if batches is not None:
        a.obs["batch"] = pd.Categorical(np.tile(batches, a.n_obs // len(batches)))
    a.layers["counts"] = a.X.copy()
    piaso.tl.infog(a, layer="counts", n_top_genes=200, verbosity=0)
    return a


@pytest.fixture(scope="module")
def ref():
    a = _data(GROUPS)
    piaso.tl.runGDR(a, groupby="ct", layer="infog", n_svd_dims=10, max_workers=1, save_reference=True)
    return a


def _query(a):
    q = a.copy()
    q.uns.clear()
    for k in list(q.obsm.keys()):
        del q.obsm[k]
    return q


def test_no_key_of_the_reference_holds_a_slash(ref):
    assert _slash_keys(ref) == []
    assert "marker_gene_names" in ref.uns["gdr_reference"]


def test_the_state_reads_back_under_the_exact_labels(ref):
    st = P._get_state(ref)
    assert list(st["marker_gene"].columns) == GROUPS
    assert "marker_gene_names" not in st
    stored = ref.uns["gdr_reference"]["marker_gene"]
    assert list(stored.columns) == [g.replace("/", "%2F") for g in GROUPS]
    # the same genes under each name
    for g, s in zip(GROUPS, stored.columns):
        assert list(st["marker_gene"][g]) == list(stored[s])


def test_an_h5ad_round_trip_projects_as_the_reference_in_memory(ref, tmp_path):
    p = tmp_path / "ref.h5ad"
    ref.copy().write_h5ad(p)
    back = ad.read_h5ad(p)
    assert list(P._get_state(back)["marker_gene"].columns) == GROUPS
    q1, q2 = _query(ref), _query(ref)
    piaso.tl.projectGDR(q1, reference=ref.copy(), layer="infog", max_workers=1)
    piaso.tl.projectGDR(q2, reference=back, layer="infog", max_workers=1)
    np.testing.assert_allclose(q1.obsm["X_gdr"], q2.obsm["X_gdr"], rtol=1e-6, atol=1e-6)
    q2.write_h5ad(tmp_path / "query.h5ad")                 # the query is writable as well


def test_the_completed_state_is_cached_back_without_a_slash(ref, tmp_path):
    r = ref.copy()
    piaso.tl.projectGDR(_query(r), reference=r, layer="infog", max_workers=1)
    assert _slash_keys(r) == []
    assert list(P._get_state(r)["marker_gene"].columns) == GROUPS
    r.write_h5ad(tmp_path / "completed.h5ad")


def test_labels_without_a_slash_are_stored_as_before():
    groups = ["L23 IT", "L56 NP", "Astro", "Micro"]
    a = _data(groups, seed=1)
    piaso.tl.runGDR(a, groupby="ct", layer="infog", n_svd_dims=10, max_workers=1, save_reference=True)
    stored = a.uns["gdr_reference"]
    assert "marker_gene_names" not in stored
    assert list(stored["marker_gene"].columns) == groups
    assert P._get_state(a) is stored                       # read as it is, no copy


def test_batch_labels_with_a_slash(tmp_path):
    a = _data(GROUPS, seed=2, batches=["b1", "b2"])
    piaso.tl.runGDR(a, groupby="ct", batch_key="batch", layer="infog", n_svd_dims=10, max_workers=1,
                    save_reference=True)
    assert _slash_keys(a) == []
    cols = list(P._get_state(a)["marker_gene"].columns)
    assert any("/" in c for c in cols) and all("%2F" not in c for c in cols)
    a.write_h5ad(tmp_path / "batch.h5ad")
    back = ad.read_h5ad(tmp_path / "batch.h5ad")
    assert list(P._get_state(back)["marker_gene"].columns) == cols


def test_an_earlier_state_with_slash_columns_still_reads():
    mg = pd.DataFrame({"L2/3 IT": ["G1", "G2"], "Astro": ["G3", "G4"]})
    a = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    a.uns["gdr_reference"] = {"marker_gene": mg, "fitted": False}   # as runGDR stored it before
    st = P._get_state(a)
    assert list(st["marker_gene"].columns) == ["L2/3 IT", "Astro"]


def test_inconsistent_names_are_refused():
    st = P._uns_state({"marker_gene": pd.DataFrame({"L2/3 IT": ["G1"], "Astro": ["G2"]})})
    st["marker_gene_names"] = np.asarray(["L2/3 IT"], dtype=object)
    with pytest.raises(ValueError, match="marker-set names"):
        P._from_uns_state(st)


def test_writes_where_anndata_refuses_slashes(ref, tmp_path):
    settings = getattr(ad, "settings", None)
    if settings is None or not hasattr(settings, "disallow_forward_slash_in_h5ad"):
        pytest.skip("this anndata has no switch for slashes in h5ad keys")
    old = settings.disallow_forward_slash_in_h5ad
    settings.disallow_forward_slash_in_h5ad = True
    try:
        ref.copy().write_h5ad(tmp_path / "strict.h5ad")
    finally:
        settings.disallow_forward_slash_in_h5ad = old
