"""Dendrogram and heatmap on stores and AnnData agree, and name their groups.

Regressions for the audit of 2026-10-27:

* part 3 critical 3 (review I2 finding 3): on a cytome, ``use_rep='auto'``
  looked for AnnData's ``X_`` names only, found nothing, and clustered the
  groups on the cells table's numeric columns -- cell index, fragment counts --
  without a warning; ``use_rep='X_pca'`` did not resolve either;
* finding 4: the dendrogram's leaf colours were positional;
* finding 7: ``heatmap(cell_level=True)`` raised NameError for every input,
  and its sidebar colours were positional;
* finding 15: the group means were a per-cell Python loop.
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso

CATS = ["B", "A", "C"]                       # categorical order, not natural
COLS = {"B": "#00ff00", "A": "#ff0000", "C": "#0000ff"}


def _adata(n=300, seed=0):
    rng = np.random.default_rng(seed)
    grp = np.repeat(["A", "B", "C"], n // 3)
    cent = {"A": [0, 0, 0], "B": [0.5, 0, 0], "C": [10, 10, 10]}
    pca = np.vstack([np.asarray(cent[g]) + rng.normal(scale=0.1, size=3) for g in grp])
    qc = np.where(grp == "B", 1000.0, 10.0) + rng.normal(size=n)   # pairs A with C
    X = rng.poisson(1.0, (n, 5)).astype(np.float32)
    X[:, 0] += np.where(grp == "C", 5, 0)
    a = ad.AnnData(sp.csr_matrix(X),
                   obs=pd.DataFrame({"leiden": pd.Categorical(grp, categories=CATS),
                                     "n_counts": qc}, index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(5)]))
    a.obsm["X_pca"] = pca
    a.uns["leiden_colors"] = [COLS[c] for c in CATS]
    return a


def _store(tmp_path, a, name="d.cytome"):
    import cytome
    path = str(tmp_path / name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = cytome.from_anndata(a, output=path)
        ds.close()
    return path


def _first_merge(Z, groups):
    i, j = (int(v) for v in np.asarray(Z)[0, :2])
    return {groups[i], groups[j]}


def test_store_dendrogram_uses_the_store_embedding(tmp_path):
    a = _adata()
    path = _store(tmp_path, a)
    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no silent fallback, no warning
        (fig, ax), Z = piaso.pl.plot_dendrogram(path, groupby="leiden", show=False,
                                                return_fig=True, return_linkage=True)
    leaves = [t.get_text() for t in ax.get_xticklabels()]
    assert {"A", "B"} in ({leaves[0], leaves[1]}, {leaves[1], leaves[2]})
    groups = [t for t in CATS]                   # mean_df index = stored order
    assert _first_merge(Z, groups) == {"A", "B"}
    plt.close("all")


def test_store_dendrogram_explicit_anndata_name_resolves(tmp_path):
    a = _adata()
    path = _store(tmp_path, a)
    Z = piaso.pl.plot_dendrogram(path, groupby="leiden", use_rep="X_pca",
                                 show=False, return_linkage=True)
    Z_ad = piaso.pl.plot_dendrogram(a, groupby="leiden", use_rep="X_pca",
                                    show=False, return_linkage=True)
    np.testing.assert_allclose(np.asarray(Z), np.asarray(Z_ad), rtol=1e-6)
    plt.close("all")


def test_store_without_embedding_or_variable_genes_refuses(tmp_path):
    import cytome
    p = str(tmp_path / "bare.cytome")
    ds = cytome.create(p)
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"b{i}" for i in range(30)],
                                         "leiden": ["A", "B", "C"] * 10,
                                         "n_fragments": np.arange(30)}))
    ds.flush()
    ds.close()
    with pytest.raises(ValueError, match="use_rep"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piaso.pl.plot_dendrogram(p, groupby="leiden", show=False)


def test_dendrogram_leaf_colours_follow_their_names(tmp_path):
    a = _adata()
    for data in (a, _store(tmp_path, a)):
        plt.close("all")
        fig, ax = piaso.pl.plot_dendrogram(data, groupby="leiden", show=False,
                                           return_fig=True)
        got = {t.get_text(): mcolors.to_hex(t.get_color()) for t in ax.get_xticklabels()}
        assert got == COLS
        plt.close("all")


# --- heatmap --------------------------------------------------------------------

def test_cell_level_heatmap_draws_on_anndata_and_store(tmp_path):
    a = _adata()
    for data in (a, _store(tmp_path, a)):
        plt.close("all")
        fig, ax = piaso.pl.heatmap(data, ["g0", "g1"], groupby="leiden", cell_level=True,
                                   max_cells_per_group=20, show=False, return_fig=True)
        side = [x for x in fig.axes if x is not ax and x.images and x.get_yticklabels()]
        assert side, "no group sidebar"
        labels = [t.get_text() for t in side[0].get_yticklabels()]
        assert labels == CATS                    # the stored order, not 'A','B','C'
        # sidebar colour of each group, sampled at its tick
        img = side[0].images[0].get_array()
        for t, g in zip(side[0].get_yticks(), labels):
            rgb = np.asarray(img[int(t), 0, :3], dtype=float)
            np.testing.assert_allclose(rgb, mcolors.to_rgb(COLS[g]), atol=1e-6)
        plt.close("all")


def test_group_means_vectorised_match_nanmean():
    from piaso.plotting._plotHeatmap import _get_mean_expression
    a = _adata()
    a.obs["score"] = np.random.default_rng(3).normal(size=a.n_obs)
    a.obs.loc[a.obs.index[:20], "score"] = np.nan
    got = _get_mean_expression(a, ["g0", "g3", "score"], "leiden")
    X = a.X.toarray()
    for g in CATS:
        m = (a.obs["leiden"] == g).to_numpy()
        assert got.loc[g, "g0"] == pytest.approx(X[m, 0].mean())
        assert got.loc[g, "g3"] == pytest.approx(X[m, 3].mean())
        assert got.loc[g, "score"] == pytest.approx(np.nanmean(a.obs["score"].to_numpy()[m]))
    assert list(got.index) == CATS


def test_standardised_heatmap_says_z_score():
    a = _adata()
    plt.close("all")
    fig, ax = piaso.pl.heatmap(a, ["g0", "g1"], groupby="leiden", standard_scale="var",
                               show=False, return_fig=True)
    cbar = [x for x in fig.axes if hasattr(x, "_colorbar")][0]
    assert "z-score" in cbar.get_ylabel()
    plt.close("all")
