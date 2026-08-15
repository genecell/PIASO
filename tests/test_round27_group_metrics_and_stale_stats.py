"""Round 27: stale cached stats after filter_cells + calculateGroupMetrics.

A — filter_cells/subset must invalidate cell-dependent cached normalization stats
    (cell_depth, infog/tfidf/log1p params); the resolver self-heals a wrong-length
    cached cell_depth.
B — piaso.pp.calculateGroupMetrics returns per-group detection/count metrics
    (modality-aware, honoring the set_categories order/colors), and
    piaso.pl.plotGroupMetrics renders both layouts.
"""
import os
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cytome
import piaso


def _make_multimodal(path, n=240, ng=40, npk=300, seed=0):
    rs = np.random.RandomState(seed)
    cts = ["Exc", "Inh", "Astro", "Oligo"]
    lab = [cts[i % 4] for i in range(n)]
    Xr = np.zeros((n, ng), np.float32)
    for i in range(n):
        ci = cts.index(lab[i])
        Xr[i, ci * 10:(ci + 1) * 10] = rs.poisson(5, 10)
        Xr[i] += rs.poisson(0.2, ng)
    Xa = (rs.random((n, npk)) < 0.04).astype(np.float32)
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)],
        "CellType": lab, "n_fragments": rs.randint(1000, 8000, n)}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(ng), "gene_id": [f"G{i}" for i in range(ng)]}))
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": np.arange(npk),
        "peak_id": [f"chr1:{i*100}-{i*100+50}" for i in range(npk)],
        "chr": ["chr1"] * npk, "start": np.arange(npk) * 100,
        "end_": np.arange(npk) * 100 + 50}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(Xr))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(Xa))
    ds.add_embedding("X_umap_RNA", rs.randn(n, 2).astype(np.float32))
    ds.flush()
    return ds


# --------------------------------------------------------------------------
# A — stale cached stats
# --------------------------------------------------------------------------

def test_filter_cells_invalidates_cell_depth(tmp_path):
    ds = _make_multimodal(str(tmp_path / "a.cytome"))
    # populate the RNA_cell_depth cache via an on-the-fly log1p plot
    piaso.pl.embedding(ds, basis="X_umap_RNA", color="G3", modality="RNA",
                       layer="log1p", show=False)
    plt.close("all")
    assert np.asarray(ds.metadata.get("RNA_cell_depth")).shape[0] == 240

    n_after = ds.filter_cells(np.arange(240) % 2 == 0)
    assert n_after == 120
    # cache dropped on subset (self-healing)
    assert ds.metadata.get("RNA_cell_depth") is None
    # re-plot must not raise (recomputes fresh at the new length)
    piaso.pl.embedding(ds, basis="X_umap_RNA", color="G3", modality="RNA",
                       layer="log1p", show=False)
    plt.close("all")
    assert np.asarray(ds.metadata.get("RNA_cell_depth")).shape[0] == 120
    ds.close()


def test_cell_depth_length_guard_self_heals(tmp_path):
    ds = _make_multimodal(str(tmp_path / "b.cytome"), n=120)
    # inject a stale wrong-length cache (simulates a pre-fix filtered cytome)
    ds.metadata["RNA_cell_depth"] = np.ones(9999, dtype=np.float64)
    ds.flush()
    from cytome.utils.modality import modality_cell_depth
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cd = modality_cell_depth(ds, "RNA")
        assert cd.shape[0] == 120
        assert any("stale" in str(x.message) for x in w)
    ds.close()


# --------------------------------------------------------------------------
# B — calculateGroupMetrics + plotGroupMetrics
# --------------------------------------------------------------------------

def test_group_metrics_cytome_multimodal(tmp_path):
    ds = _make_multimodal(str(tmp_path / "c.cytome"))
    ds.set_categories("CellType", order=["Inh", "Exc", "Oligo", "Astro"],
                      colors={"Inh": "#4E79A7", "Exc": "#E69F00",
                              "Oligo": "#009E73", "Astro": "#CC79A7"})
    df = piaso.pp.calculateGroupMetrics(ds, groupby="CellType", verbose=False)
    # rows follow the set_categories order
    assert list(df.index) == ["Inh", "Exc", "Oligo", "Astro"]
    # both modalities reported
    for col in ["n_cells", "RNA_n_features_detected", "ATAC_n_features_detected",
                "RNA_counts_mean", "ATAC_n_fragments_median"]:
        assert col in df.columns
    assert (df["n_cells"] == 60).all()
    # detection counts are within feature bounds
    assert df["RNA_n_features_detected"].max() <= 40
    assert df["ATAC_n_features_detected"].max() <= 300
    # colors + detection thresholds stashed for the plotter
    assert df.attrs["colors"]["Inh"] == "#4E79A7"
    assert df.attrs["detection_pct"] == {"RNA": 0.10, "ATAC": 0.05}
    ds.close()


def test_group_metrics_threshold_dict(tmp_path):
    ds = _make_multimodal(str(tmp_path / "d.cytome"))
    lo = piaso.pp.calculateGroupMetrics(
        ds, "CellType", detection_pct={"RNA": 0.01, "ATAC": 0.01}, verbose=False)
    hi = piaso.pp.calculateGroupMetrics(
        ds, "CellType", detection_pct={"RNA": 0.9, "ATAC": 0.9}, verbose=False)
    # a stricter threshold can only detect fewer-or-equal features
    assert (hi["RNA_n_features_detected"] <= lo["RNA_n_features_detected"]).all()
    assert (hi["ATAC_n_features_detected"] <= lo["ATAC_n_features_detected"]).all()
    ds.close()


def test_plot_group_metrics_both_kinds(tmp_path):
    ds = _make_multimodal(str(tmp_path / "e.cytome"))
    ds.set_categories("CellType", order=["Inh", "Exc", "Oligo", "Astro"],
                      colors={"Inh": "#4E79A7", "Exc": "#E69F00",
                              "Oligo": "#009E73", "Astro": "#CC79A7"})
    df = piaso.pp.calculateGroupMetrics(ds, "CellType", verbose=False)
    fig = piaso.pl.plotGroupMetrics(df, kind="bar", show=False, return_fig=True)
    assert fig is not None
    plt.close("all")
    fig = piaso.pl.plotGroupMetrics(df, kind="heatmap", show=False, return_fig=True)
    assert fig is not None
    plt.close("all")
    ds.close()


def test_group_metrics_anndata(tmp_path):
    import anndata as ad
    rs = np.random.RandomState(1)
    n, g = 80, 20
    X = sp.csr_matrix(rs.poisson(1.0, (n, g)).astype(np.float32))
    obs = pd.DataFrame({"CellType": pd.Categorical(
        [["A", "B"][i % 2] for i in range(n)])})
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    df = piaso.pp.calculateGroupMetrics(a, "CellType", verbose=False)
    assert set(df.index) == {"A", "B"}
    assert "RNA_n_features_detected" in df.columns
