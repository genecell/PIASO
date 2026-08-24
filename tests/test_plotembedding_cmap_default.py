"""The continuous colour map follows where the values came from.

A gene or peak value is an expression level: sequential, zero meaningful.
A numeric cell-metadata column -- a regulon score, a QC metric -- is a
quantity with no privileged zero and reads better diverging. Explicit
``cmap=`` always wins over both.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso


@pytest.fixture
def cytome_path(tmp_path):
    cytome = pytest.importorskip("cytome")
    ad = pytest.importorskip("anndata")
    rs = np.random.RandomState(0)
    a = ad.AnnData(X=sp.csr_matrix(rs.poisson(2.0, (40, 5)).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(5)]
    a.obs["score"] = rs.rand(40)
    a.obs["grp"] = pd.Categorical(["a", "b"] * 20)
    a.obsm["X_umap"] = rs.rand(40, 2)
    p = str(tmp_path / "t.cytome")
    cytome.from_anndata(a, output=p).close()
    return p


def _cmap_of(path, color, **kw):
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(path, color=color, basis="umap", ax=ax, show=False, **kw)
    name = ax.collections[0].get_cmap().name
    plt.close(fig)
    return name


def test_obs_numeric_gets_spectral(cytome_path):
    assert _cmap_of(cytome_path, "score") == "Spectral_r"


def test_feature_keeps_the_expression_map(cytome_path):
    assert _cmap_of(cytome_path, "g0") != "Spectral_r"


def test_explicit_cmap_wins_for_both(cytome_path):
    assert _cmap_of(cytome_path, "score", cmap="viridis") == "viridis"
    assert _cmap_of(cytome_path, "g0", cmap="viridis") == "viridis"


def test_categorical_is_unaffected(cytome_path):
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(cytome_path, color="grp", basis="umap", ax=ax, show=False)
    assert len(ax.collections) >= 1        # palette path, no cmap applied
    plt.close(fig)
