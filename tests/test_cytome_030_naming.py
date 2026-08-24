"""PIASO on a cytome 0.3.0 file that has no `{modality}_counts`.

0.3.0 reserves that name for raw integer counts, so a file converted from
normalised data carries `{modality}_data` instead. Two different behaviours
are correct here, and the difference is the point:

- **Plotting** wants the values the user asked to see. Falling back to the
  main matrix is right, with a warning naming what it read.
- **INFOG** models count dispersion. On non-counts it returns numbers that
  look fine and mean nothing, so it must refuse.
"""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import piaso


@pytest.fixture
def normalised_cytome(tmp_path):
    anndata = pytest.importorskip("anndata")
    cytome = pytest.importorskip("cytome")
    import pandas as pd

    rs = np.random.RandomState(0)
    counts = rs.poisson(3.0, (120, 25)).astype(np.float32)
    a = anndata.AnnData(X=sp.csr_matrix(np.log1p(counts)))
    a.var_names = [f"g{i}" for i in range(25)]
    a.obs["grp"] = pd.Categorical(rs.choice(list("ab"), 120))
    a.obsm["X_umap"] = rs.rand(120, 2)
    p = str(tmp_path / "norm.cytome")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cytome.from_anndata(a, output=p).close()
    return p


def test_no_counts_matrix_is_written(normalised_cytome):
    import sqlite3
    con = sqlite3.connect(normalised_cytome)
    try:
        names = {r[0] for r in con.execute(
            "SELECT matrix_name FROM matrix_meta")}
    finally:
        con.close()
    assert "RNA_counts" not in names
    assert "RNA_data" in names


def test_plotting_falls_back_and_says_so(normalised_cytome):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cytome

    ds = cytome.open(normalised_cytome)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            piaso.pl.plotEmbedding(ds, color="g0", basis="X_umap", show=False)
        assert any("reserves that name" in str(x.message) for x in w), \
            "the fallback must be announced, not silent"
    finally:
        plt.close("all")
        ds.close()


def test_infog_refuses_and_explains(normalised_cytome):
    import cytome
    ds = cytome.open(normalised_cytome)
    try:
        with pytest.raises(ValueError) as e:
            piaso.tl.infog(ds, modality="RNA", layer="counts",
                           key_added="infog", save_layer=True, inplace=True)
    finally:
        ds.close()
    msg = str(e.value)
    assert "RNA_data" in msg, "the message must name what is actually there"
    assert "counts_layer" in msg, "and how to fix the conversion"
