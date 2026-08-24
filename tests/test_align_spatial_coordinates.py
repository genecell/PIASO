"""Sections are placed on their chips independently, so raw coordinates put
two samples far apart. A split plot then renders each panel at a different
offset and the tissues look displaced. Centring per group fixes the display
without touching within-sample geometry — which is the property tested here.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import piaso


def _adata(n_per=50, offsets=((0., 0.), (500., -300.))):
    import anndata as ad
    import pandas as pd
    rs = np.random.RandomState(0)
    xs, ys, samp = [], [], []
    for i, (ox, oy) in enumerate(offsets):
        xs.append(rs.uniform(0, 10, n_per) + ox)
        ys.append(rs.uniform(0, 10, n_per) + oy)
        samp += [f"S{i}"] * n_per
    a = ad.AnnData(X=sp.csr_matrix(
        rs.poisson(1.0, (n_per * len(offsets), 5)).astype(np.float32)))
    a.obs["Sample"] = pd.Categorical(samp)
    a.obsm["spatial"] = np.column_stack([np.concatenate(xs),
                                         np.concatenate(ys)])
    return a


def test_each_sample_is_centred_on_its_own_centroid():
    a = _adata()
    piaso.pp.alignSpatialCoordinates(a, groupby="Sample")
    out = a.obsm["spatial_aligned"]
    for s in a.obs["Sample"].unique():
        m = (a.obs["Sample"] == s).values
        assert np.allclose(out[m].mean(axis=0), 0, atol=1e-9), s


def test_within_sample_geometry_is_preserved():
    """Centring is a translation: every pairwise distance inside a sample
    must survive it exactly. This is what separates it from rescaling."""
    a = _adata()
    piaso.pp.alignSpatialCoordinates(a, groupby="Sample")
    raw, out = a.obsm["spatial"], a.obsm["spatial_aligned"]
    m = (a.obs["Sample"] == "S0").values
    d_raw = np.linalg.norm(raw[m][:20, None] - raw[m][None, :20], axis=-1)
    d_out = np.linalg.norm(out[m][:20, None] - out[m][None, :20], axis=-1)
    assert np.allclose(d_raw, d_out)


def test_with_std_equalises_spread_and_is_not_the_default():
    a = _adata(offsets=((0., 0.), (500., -300.)))
    a.obsm["spatial"][a.obs["Sample"].values == "S1"] *= 4.0   # bigger section
    piaso.pp.alignSpatialCoordinates(a, groupby="Sample", key_added="plain")
    piaso.pp.alignSpatialCoordinates(a, groupby="Sample", key_added="scaled",
                                     with_std=True)
    sd = lambda arr, s: arr[(a.obs["Sample"] == s).values].std(axis=0)
    assert not np.allclose(sd(a.obsm["plain"], "S0"), sd(a.obsm["plain"], "S1"),
                           rtol=0.2), "plain centring must preserve size"
    assert np.allclose(sd(a.obsm["scaled"], "S0"), sd(a.obsm["scaled"], "S1"))


def test_backup_keeps_the_original_frame():
    a = _adata()
    before = a.obsm["spatial"].copy()
    piaso.pp.alignSpatialCoordinates(a, groupby="Sample",
                                     backup_spatial_key="spatial_original")
    assert np.array_equal(a.obsm["spatial_original"], before)
    assert np.array_equal(a.obsm["spatial"], before)   # source untouched


def test_copy_returns_and_leaves_the_input_alone():
    a = _adata()
    out = piaso.pp.alignSpatialCoordinates(a, groupby="Sample", copy=True)
    assert out is not None and "spatial_aligned" in out.obsm
    assert "spatial_aligned" not in a.obsm


def test_missing_keys_say_which():
    a = _adata()
    with pytest.raises(KeyError, match="nope"):
        piaso.pp.alignSpatialCoordinates(a, groupby="nope")
    with pytest.raises(KeyError, match="available"):
        piaso.pp.alignSpatialCoordinates(a, groupby="Sample",
                                         spatial_key="nope")


def test_cytome_backend_round_trip(tmp_path):
    cytome = pytest.importorskip("cytome")
    a = _adata()
    p = str(tmp_path / "s.cytome")
    cytome.from_anndata(a, output=p).close()

    piaso.pp.alignSpatialCoordinates(p, groupby="Sample",
                                     key_added="spatial_aligned")
    ds = cytome.open(p)
    try:
        out = np.asarray(ds.embeddings["spatial_aligned"])
        groups = np.asarray(ds.cells["Sample"]).astype(str)
        for s in np.unique(groups):
            assert np.allclose(out[groups == s].mean(axis=0), 0, atol=1e-4), s
    finally:
        ds.close()


def test_cytome_rejects_inplace_false(tmp_path):
    cytome = pytest.importorskip("cytome")
    a = _adata()
    p = str(tmp_path / "s.cytome")
    cytome.from_anndata(a, output=p).close()
    with pytest.raises(ValueError, match="in place"):
        piaso.pp.alignSpatialCoordinates(p, groupby="Sample", inplace=False)


def test_batch_key_is_an_accepted_alias_for_groupby():
    """GDR calls this column a batch and plotting calls it a group; a reader
    arriving from either should not have to look up which word won."""
    a, b = _adata(), _adata()
    piaso.pp.alignSpatialCoordinates(a, groupby="Sample")
    piaso.pp.alignSpatialCoordinates(b, batch_key="Sample")
    assert np.allclose(a.obsm["spatial_aligned"], b.obsm["spatial_aligned"])


def test_neither_or_conflicting_is_an_error():
    a = _adata()
    with pytest.raises(TypeError, match="groupby"):
        piaso.pp.alignSpatialCoordinates(a)
    with pytest.raises(ValueError, match="disagree"):
        piaso.pp.alignSpatialCoordinates(a, groupby="Sample", batch_key="other")
