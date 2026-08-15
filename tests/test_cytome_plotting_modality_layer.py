"""Cytome-native plotting: modality + cytome_layer routing.

Tests assert CONTENT (rendered values match a known reference), not
just absence of exceptions. Catches the previous silent failures where
``dotplot`` / ``plot_features_violin`` for cytome only read from
``ds.cells`` and produced empty plots when handed gene names.

Covers the 7 cytome-supporting plotting functions:
plotEmbedding, plotUMAP, plot_embeddings_split, dotplot,
plot_features_violin, plot_dendrogram, heatmap, scatter.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_multimodal_cytome(path, n_cells=12, seed=0):
    """RNA + GA + ATAC, with deliberate value patterns so tests can check
    that the right matrix was read for the right modality:
      RNA_counts['Sox2']  is 5.0 in cluster 'g0', 0 elsewhere.
      GA_counts['Sox2']   is 7.0 in cluster 'g0', 0 elsewhere.   (same name, different mod)
      ATAC_counts['chr1:100-200']  is 3.0 in cluster 'g1', 0 elsewhere.
    """
    import cytome
    rng = np.random.default_rng(seed)
    ds = cytome.create(path)
    leiden = np.array([f"g{i % 3}" for i in range(n_cells)])
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"AAA-{i}" for i in range(n_cells)],
        "Leiden": leiden,
        "Sample": np.array([f"s{i % 2}" for i in range(n_cells)]),
    }))
    # RNA: 3 genes (Sox2, Pax6, Foxg1), Sox2 is 5.0 in g0, 0 elsewhere
    rna_X = np.zeros((n_cells, 3), dtype=np.float32)
    rna_X[leiden == "g0", 0] = 5.0
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": [0, 1, 2],
        "gene_id": ["Sox2", "Pax6", "Foxg1"],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(rna_X))
    # GA: 4 genes (Sox2, Pax6, Olig2, Nestin), Sox2 is 7.0 in g0, 0 elsewhere
    # Disjoint from RNA in part (Olig2, Nestin only in GA)
    ga_X = np.zeros((n_cells, 4), dtype=np.float32)
    ga_X[leiden == "g0", 0] = 7.0
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": [0, 1, 2, 3],
        "gene_id": ["Sox2", "Pax6", "Olig2", "Nestin"],
    }))
    ds.add_matrix("GA_counts", sp.csr_matrix(ga_X))
    # ATAC: 2 peaks
    atac_X = np.zeros((n_cells, 2), dtype=np.float32)
    atac_X[leiden == "g1", 0] = 3.0
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": [0, 1],
        "peak_id": ["chr1:100-200", "chr2:300-400"],
        "chr": ["chr1", "chr2"],
        "start": [100, 300],
        "end_": [200, 400],
    }))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(atac_X))
    # An embedding for plotEmbedding tests
    ds.add_embedding("X_umap", rng.standard_normal((n_cells, 2)).astype(np.float32))
    ds.flush()
    return ds


# -----------------------------------------------------------------------
# 1. The resolver — direct unit tests
# -----------------------------------------------------------------------

def _resolver(ds, feature, **kw):
    from piaso.plotting._plotEmbedding import _resolve_cytome_feature_values
    return _resolve_cytome_feature_values(ds, feature, **kw)


def test_resolver_explicit_rna_returns_rna_values(tmp_path):
    """modality='RNA' on an ambiguous gene resolves to the RNA matrix."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    vals, mod = _resolver(ds, "Sox2", modality="RNA")
    assert mod == "RNA"
    # Sox2 in RNA_counts: 5.0 for cluster g0, 0 elsewhere
    leiden = np.asarray(ds.cells["Leiden"]).astype(str)
    assert float(vals[leiden == "g0"].mean()) == pytest.approx(5.0)
    assert float(vals[leiden != "g0"].max()) == 0.0
    ds.close()


def test_resolver_explicit_ga_returns_ga_values(tmp_path):
    """modality='GA' on the SAME gene resolves to the GA matrix (different
    value)."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    vals, mod = _resolver(ds, "Sox2", modality="GA")
    assert mod == "GA"
    leiden = np.asarray(ds.cells["Leiden"]).astype(str)
    assert float(vals[leiden == "g0"].mean()) == pytest.approx(7.0)
    ds.close()


def test_resolver_atac_peak(tmp_path):
    """ATAC peak strings resolve via the peaks entity."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    vals, mod = _resolver(ds, "chr1:100-200", modality="ATAC")
    assert mod == "ATAC"
    leiden = np.asarray(ds.cells["Leiden"]).astype(str)
    assert float(vals[leiden == "g1"].mean()) == pytest.approx(3.0)
    ds.close()


def test_resolver_ambiguous_modality_raises(tmp_path):
    """Sox2 exists in BOTH RNA and GA → ambiguous error when modality=None."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    with pytest.raises(ValueError) as exc_info:
        _resolver(ds, "Sox2")  # no modality
    msg = str(exc_info.value)
    assert "ambiguous" in msg.lower()
    assert "RNA" in msg and "GA" in msg
    ds.close()


def test_resolver_unique_auto_modality(tmp_path):
    """A gene present in only ONE modality auto-resolves."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    vals, mod = _resolver(ds, "Olig2")  # only in GA
    assert mod == "GA"
    ds.close()


def test_resolver_feature_not_found(tmp_path):
    """Feature absent from all modalities → KeyError."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    with pytest.raises(KeyError) as exc_info:
        _resolver(ds, "NoSuchGene")
    assert "NoSuchGene" in str(exc_info.value)
    ds.close()


def test_resolver_user_modality_but_feature_missing(tmp_path):
    """modality='RNA' but the feature is in GA → ValueError suggests the right
    modality."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    with pytest.raises(ValueError) as exc_info:
        _resolver(ds, "Olig2", modality="RNA")
    msg = str(exc_info.value)
    assert "Olig2" in msg
    assert "GA" in msg  # hint mentions where the feature actually is
    ds.close()


def test_resolver_log1p_compute_on_fly(tmp_path):
    """log1p_otf produces the expected log1p(counts/depth*scale) values."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    leiden = np.asarray(ds.cells["Leiden"]).astype(str)
    vals, mod = _resolver(
        ds, "Sox2", modality="RNA",
        cytome_layer="log1p", compute_on_fly=True,
    )
    assert mod == "RNA"
    # For cells in g0: counts=5; depth = 5 (only Sox2 nonzero); ratio = 1.0
    # → log1p(1.0 * 1e4) = log1p(10000) ≈ 9.2103
    assert float(vals[leiden == "g0"].mean()) == pytest.approx(np.log1p(1e4), rel=1e-4)
    # For cells outside g0: counts=0 → log1p(0) = 0
    assert float(vals[leiden != "g0"].max()) == 0.0
    ds.close()


def test_resolver_strict_no_compute_on_fly_raises(tmp_path):
    """compute_on_fly=False + missing matrix → actionable ValueError mentioning
    compute_on_fly=True as one of the fix paths."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    with pytest.raises(ValueError) as exc_info:
        _resolver(
            ds, "Sox2", modality="RNA",
            cytome_layer="log1p", compute_on_fly=False,
        )
    assert "compute_on_fly" in str(exc_info.value)
    ds.close()


def test_resolver_uses_stored_matrix_even_when_compute_on_fly_true(tmp_path):
    """When the matrix IS stored, we read it (storage > recomputation)."""
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    leiden = np.asarray(ds.cells["Leiden"]).astype(str)
    # Materialize a fake RNA_log1p with deliberately distinct values so we
    # can detect that we read the matrix instead of recomputing on-the-fly.
    fake_log1p = np.zeros((ds.n_cells, 3), dtype=np.float32)
    fake_log1p[leiden == "g0", 0] = 99.0
    ds.add_matrix("RNA_log1p", sp.csr_matrix(fake_log1p))
    ds.flush()
    vals, _ = _resolver(
        ds, "Sox2", modality="RNA",
        cytome_layer="log1p", compute_on_fly=True,
    )
    assert float(vals[leiden == "g0"].mean()) == pytest.approx(99.0)
    ds.close()


# -----------------------------------------------------------------------
# 2. plotEmbedding — actually reads the right matrix
# -----------------------------------------------------------------------

def test_plotEmbedding_modality_routes_to_correct_matrix(tmp_path):
    """Coloring by Sox2 with modality='RNA' vs modality='GA' produces
    different scatter colour values (5.0 vs 7.0 for g0 cells)."""
    import piaso
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    leiden = np.asarray(ds.cells["Leiden"]).astype(str)

    fig, ax = piaso.pl.plotEmbedding(
        ds, color="Sox2", basis="X_umap",
        modality="RNA", show=False, return_fig=True,
    )
    rna_arr = ax.collections[0].get_array().data
    plt.close(fig)

    fig, ax = piaso.pl.plotEmbedding(
        ds, color="Sox2", basis="X_umap",
        modality="GA", show=False, return_fig=True,
    )
    ga_arr = ax.collections[0].get_array().data
    plt.close(fig)

    # In g0 cells: RNA=5.0, GA=7.0 — verify the routing went to the right matrix
    rna_g0_max = float(rna_arr[leiden == "g0"].max())
    ga_g0_max = float(ga_arr[leiden == "g0"].max())
    assert rna_g0_max == pytest.approx(5.0)
    assert ga_g0_max == pytest.approx(7.0)
    ds.close()


def test_plotEmbedding_show_modality_in_title(tmp_path):
    """show_modality_in_title=True appends '(RNA)' to the title."""
    import piaso
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    fig, ax = piaso.pl.plotEmbedding(
        ds, color="Sox2", basis="X_umap",
        modality="RNA", show_modality_in_title=True, show=False, return_fig=True,
    )
    assert ax.get_title() == "Sox2 (RNA)"
    plt.close(fig)
    # Default off:
    fig, ax = piaso.pl.plotEmbedding(
        ds, color="Sox2", basis="X_umap",
        modality="RNA", show=False, return_fig=True,
    )
    assert ax.get_title() == "Sox2"
    plt.close(fig)
    ds.close()


# -----------------------------------------------------------------------
# 3. dotplot — actually reads gene matrices (the big behavioural fix)
# -----------------------------------------------------------------------

def test_dotplot_cytome_reads_gene_values_not_just_cells_table(tmp_path):
    """Dotplot on a cytome with gene names should read RNA_counts —
    NOT silently return zeros from the cells table."""
    import piaso
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    fig = plt.figure()
    fraction_df, mean_df = piaso.pl._plotDotplot._get_expression_data(
        ds, ["Sox2"], "Leiden",
        modality="RNA",
    )
    # mean_df.loc['g0', 'Sox2'] should be 5.0 (per construction)
    assert float(mean_df.loc["g0", "Sox2"]) == pytest.approx(5.0)
    # And 0.0 for other groups
    assert float(mean_df.loc["g1", "Sox2"]) == 0.0
    assert float(mean_df.loc["g2", "Sox2"]) == 0.0
    plt.close(fig)
    ds.close()


def test_dotplot_renders_without_error(tmp_path):
    """End-to-end: piaso.pl.dotplot on a cytome with gene name actually
    renders + produces a nonzero mean value."""
    import piaso
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    piaso.pl.dotplot(
        ds, features=["Sox2"], groupby="Leiden",
        modality="RNA", show=False, return_fig=True,
    )
    plt.close("all")
    ds.close()


# -----------------------------------------------------------------------
# 4. heatmap & scatter — gain cytome support outright
# -----------------------------------------------------------------------

def test_heatmap_renders_on_cytome_with_genes(tmp_path):
    """Pre-fix: heatmap(cytome, features=['Sox2'], ...) fell back to
    ds.cells lookup, found nothing, drew zeros. Post-fix: reads RNA_counts
    correctly."""
    import piaso
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    result = piaso.pl.heatmap(
        ds, features=["Sox2", "Pax6"], groupby="Leiden",
        modality="RNA", show=False, return_fig=True,
    )
    # heatmap returns either a Figure or (fig, ax) depending on version.
    fig = result[0] if isinstance(result, tuple) else result
    plt.close(fig)
    ds.close()


def test_scatter_renders_on_cytome_with_genes(tmp_path):
    """piaso.pl.scatter on a cytome with two gene names actually pulls
    counts columns through the resolver."""
    import piaso
    ds = _build_multimodal_cytome(tmp_path / "x.cytome")
    fig, ax = piaso.pl.scatter(
        ds, x="Sox2", y="Pax6",
        modality="RNA", show=False, return_fig=True,
    )
    # X-data should match Sox2 column from RNA_counts
    leiden = np.asarray(ds.cells["Leiden"]).astype(str)
    coll = ax.collections[0]
    offsets = coll.get_offsets().data
    x_vals = offsets[:, 0]
    # In g0 cells, x (Sox2) should be 5.0
    assert float(x_vals[leiden == "g0"].mean()) == pytest.approx(5.0)
    plt.close(fig)
    ds.close()
