"""Tissue-image overlay on spatial embeddings — both backends, one resolver.

The three classic overlay bugs each have a test: orientation (the image's row
0 is the TOP of the tissue), units (spot coords are full-res; the image is
scalef smaller — the image moves into coordinate space, not the other way),
and library ambiguity (never silently pick one of several tissues).
"""
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso
from piaso.plotting._spatial_image import (
    _draw_image_overlay,
    _image_axis_limits,
    _resolve_spatial_image,
    _spatial_uns_from,
)


def _tissue_img(h=40, w=60):
    """Top half red, bottom half blue — the orientation witness."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[: h // 2, :, 0] = 1.0
    img[h // 2:, :, 2] = 1.0
    return img


def _spatial_adata(n=30, libs=("libA",), seed=0):
    import anndata as ad

    rs = np.random.RandomState(seed)
    a = ad.AnnData(X=sp.csr_matrix(rs.poisson(1.0, (n, 6)).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(6)]
    a.obs["grp"] = pd.Categorical([f"c{i % 2}" for i in range(n)])
    a.obs["library_id"] = pd.Categorical(
        [libs[i % len(libs)] for i in range(n)])
    # full-res coords inside a 120x80 fullres frame (image 60x40 at sf 0.5)
    a.obsm["spatial"] = np.column_stack([
        rs.uniform(5, 115, n), rs.uniform(5, 75, n)]).astype(np.float64)
    a.uns["spatial"] = {
        lib: {
            "images": {"hires": _tissue_img()},
            "scalefactors": {"tissue_hires_scalef": 0.5,
                             "spot_diameter_fullres": 6.0},
        } for lib in libs
    }
    return a


def _spatial_cytome(tmp_path, libs=("libA",)):
    import cytome

    a = _spatial_adata(libs=libs)
    ds = cytome.from_anndata(a, output=str(tmp_path / "sp.cytome"))
    return a, ds


# ------------------------------------------------------------- the resolver

def test_resolver_reads_both_backends_identically(tmp_path):
    pytest.importorskip("cytome")
    a, ds = _spatial_cytome(tmp_path)
    try:
        ca = _resolve_spatial_image(a, True, "hires")
        cd = _resolve_spatial_image(ds, True, "hires")
        assert np.array_equal(ca["img"], cd["img"])
        assert ca["scalef"] == cd["scalef"] == 0.5
        assert ca["extent"] == cd["extent"] == (0.0, 120.0, 80.0, 0.0)
        assert ca["spot_diameter"] == 6.0
    finally:
        ds.close()


def test_resolver_auto_needs_a_unique_library():
    a = _spatial_adata(libs=("libA", "libB"))
    with pytest.raises(ValueError, match="libA.*libB|ambiguous"):
        _resolve_spatial_image(a, True, "hires")
    ctx = _resolve_spatial_image(a, "libB", "hires")
    assert ctx["library"] == "libB"
    with pytest.raises(KeyError, match="available"):
        _resolve_spatial_image(a, "nope", "hires")
    with pytest.raises(KeyError, match="available"):
        _resolve_spatial_image(a, "libA", "fullres")


def test_resolver_narrows_by_the_cells_libraries():
    """Two libraries in the store, but the plotted cells span only one."""
    a = _spatial_adata(libs=("libA", "libB"))
    ctx = _resolve_spatial_image(a, True, "hires",
                                 library_values=["libA"] * 5)
    assert ctx["library"] == "libA"


def test_resolver_none_when_no_image_and_warns():
    a = _spatial_adata()
    del a.uns["spatial"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _resolve_spatial_image(a, True, "hires") is None
    assert any("no spatial images" in str(x.message) for x in w)
    assert _resolve_spatial_image(a, False, "hires") is None


def test_uns_source_degrades_on_undecodable_store():
    class _Raises:
        def as_uns(self):
            raise ImportError("pillow missing")

    class _Fake:
        spatial_images = _Raises()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _spatial_uns_from(_Fake()) == {}
    assert any("not decodable" in str(x.message) for x in w)


# ------------------------------------------------------ orientation + units

def test_orientation_a_top_spot_sits_on_red(tmp_path):
    """The pixel behind a spot near y=0 (fullres top) must be RED."""
    a = _spatial_adata(n=1)
    a.obsm["spatial"] = np.array([[60.0, 10.0]])       # near the top
    fig, ax = plt.subplots()
    ctx = _resolve_spatial_image(a, True, "hires")
    _draw_image_overlay(ax, ctx, 1.0)
    _image_axis_limits(ax, a.obsm["spatial"], ctx)
    fig.canvas.draw()
    # sample the BACKGROUND pixel at the spot's data position (no scatter
    # drawn, so what we read is the tissue, not the marker)
    xpix, ypix = ax.transData.transform((60.0, 10.0))
    buf = np.asarray(fig.canvas.buffer_rgba())
    h = buf.shape[0]
    r, g, b, _ = buf[int(h - ypix), int(xpix)]
    assert r > 150 and b < 100, f"expected red under a top spot, got {(r, g, b)}"
    plt.close(fig)


def test_image_lives_in_fullres_units():
    ctx = _resolve_spatial_image(_spatial_adata(), True, "hires")
    # image is 60 wide at scalef 0.5 -> spans 120 fullres units
    assert ctx["extent"][1] == 120.0 and ctx["extent"][2] == 80.0


def test_axis_limits_are_inverted_and_padded():
    fig, ax = plt.subplots()
    coords = np.array([[10.0, 20.0], [110.0, 70.0]])
    ctx = {"spot_diameter": 6.0}
    _image_axis_limits(ax, coords, ctx)
    x0, x1 = ax.get_xlim()
    ytop, ybot = ax.get_ylim()
    assert x0 < 10 and x1 > 110
    assert ytop > ybot, "y must be inverted (image convention)"
    plt.close(fig)


# --------------------------------------------------------------- end to end

def test_plot_embedding_draws_the_image_under_the_spots():
    a = _spatial_adata()
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="grp", basis="spatial", image=True,
                           ax=ax, show=False)
    assert len(ax.images) == 1
    ytop, ybot = ax.get_ylim()
    assert ytop > ybot
    plt.close(fig)


def test_plot_embedding_without_image_is_unchanged():
    a = _spatial_adata()
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="grp", basis="spatial", ax=ax, show=False)
    assert len(ax.images) == 0
    y0, y1 = ax.get_ylim()
    assert y0 < y1, "no image: keep the y-up default"
    plt.close(fig)


def test_plot_embedding_cytome_backend(tmp_path):
    pytest.importorskip("cytome")
    a, ds = _spatial_cytome(tmp_path)
    try:
        fig, ax = plt.subplots()
        piaso.pl.plotEmbedding(ds, color="grp", basis="spatial", image=True,
                               ax=ax, show=False)
        assert len(ax.images) == 1
        plt.close(fig)
    finally:
        ds.close()


def test_split_by_library_gets_one_image_per_panel():
    a = _spatial_adata(n=40, libs=("libA", "libB"))
    piaso.pl.plot_embeddings_split(
        a, color="grp", splitby="library_id", basis="spatial", image=True,
        show_figure=False)
    fig = plt.gcf()
    axes = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    imaged = [ax for ax in axes if len(ax.images) == 1]
    assert len(imaged) == 2, "each library panel draws its own tissue"
    for ax in imaged:
        ytop, ybot = ax.get_ylim()
        assert ytop > ybot
    plt.close(fig)


def test_split_by_other_column_with_many_libraries_warns_and_skips():
    a = _spatial_adata(n=40, libs=("libA", "libB"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        piaso.pl.plot_embeddings_split(
            a, color="library_id", splitby="grp", basis="spatial", image=True,
            show_figure=False)
        fig = plt.gcf()
    assert any("single library" in str(x.message) for x in w)
    assert all(len(ax.images) == 0 for ax in fig.axes)
    plt.close(fig)


def test_split_single_library_draws_on_every_panel():
    a = _spatial_adata(n=40, libs=("libA",))
    piaso.pl.plot_embeddings_split(
        a, color="grp", splitby="grp", basis="spatial", image=True,
        show_figure=False)
    fig = plt.gcf()
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert all(len(ax.images) == 1 for ax in panels)
    plt.close(fig)


def test_split_explicit_library_string(tmp_path):
    a = _spatial_adata(n=40, libs=("libA", "libB"))
    piaso.pl.plot_embeddings_split(
        a, color="grp", splitby="grp", basis="spatial", image="libA",
        show_figure=False)
    fig = plt.gcf()
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert all(len(ax.images) == 1 for ax in panels)
    plt.close(fig)


def test_resolver_accepts_a_cytome_path(tmp_path):
    """plotEmbedding accepts a .cytome path; the image resolver must too."""
    pytest.importorskip("cytome")
    a, ds = _spatial_cytome(tmp_path)
    p = str(ds.path)
    ds.close()
    ctx = _resolve_spatial_image(p, True, "hires")
    assert ctx is not None and ctx["scalef"] == 0.5

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(p, color="grp", basis="spatial", image=True,
                           ax=ax, show=False)
    assert len(ax.images) == 1
    plt.close(fig)
