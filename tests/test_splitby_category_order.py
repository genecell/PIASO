"""`splitby` panels follow the order stored on the cytome.

A cytome column comes back as plain objects, so an ordering set with
``ds.set_categories`` was lost and the panels fell back to alphabetical --
which puts E10.5 before E9.5 and reads as a bug in the figure rather than in
the sort.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso

STAGES = ["E9.5", "E10.5", "E11.5", "E12.5"]


@pytest.fixture
def ds_path(tmp_path):
    cytome = pytest.importorskip("cytome")
    ad = pytest.importorskip("anndata")
    rs = np.random.RandomState(0)
    n = len(STAGES) * 12
    a = ad.AnnData(X=sp.csr_matrix(rs.poisson(2.0, (n, 5)).astype(np.float32)))
    a.obs["stage"] = pd.Categorical([s for s in STAGES for _ in range(12)])
    a.obs["grp"] = pd.Categorical(["x", "y"] * (n // 2))
    a.obsm["X_umap"] = rs.rand(n, 2)
    p = str(tmp_path / "t.cytome")
    cytome.from_anndata(a, output=p).close()
    return p


def _panel_stages(path):
    piaso.pl.plot_embeddings_split(path, color="grp", splitby="stage",
                                   basis="umap", ncols=4, show_figure=False)
    fig = plt.gcf()
    out = [a.get_title().replace("\n", " ").split()[-1]
           for a in fig.axes if a.get_title()]
    plt.close(fig)
    return out


def test_alphabetical_without_stored_order(ds_path):
    """Baseline: 'E10.5' sorts before 'E9.5' as a string."""
    got = _panel_stages(ds_path)
    assert got[0] == "E10.5", got


def test_stored_order_is_honoured(ds_path):
    cytome = pytest.importorskip("cytome")
    ds = cytome.open(ds_path)
    ds.set_categories("stage", STAGES)
    ds.flush(); ds.close()
    assert _panel_stages(ds_path) == STAGES


def test_values_absent_from_the_data_are_skipped(ds_path):
    cytome = pytest.importorskip("cytome")
    ds = cytome.open(ds_path)
    ds.set_categories("stage", ["E8.5"] + STAGES)     # E8.5 has no cells
    ds.flush(); ds.close()
    assert _panel_stages(ds_path) == STAGES
