"""Round 12 regression: runSVD streaming path uses the modality registry.

The pre-Round-12 streaming path had:

    if modality == 'ATAC':
        hv_col = ds.peaks['selected']
    else:
        hv_col = ds.genes['highly_variable']

For ``modality='GA'`` (gene activity) on a cytome that has BOTH ``genes``
(RNA) AND ``GA_genes`` (gene activity) tables, this silently read
**RNA's** highly_variable mask instead of GA's. Same class of bug as
Round 11's ``cytome.from_anndata`` ATAC-vs-else hardcoding.

Round 12 replaces the dichotomy with ``modality_var_entity`` lookup:

    var_entity, _ = modality_var_entity(modality)
    hv_col = getattr(ds, var_entity)[selected_feature_col_name]

This file pins that:
  1. GA modality on a cytome with both RNA+GA tables reads from
     ``ds.GA_genes['highly_variable']`` (NOT ``ds.genes['highly_variable']``).
  2. ATAC modality reads from ``ds.peaks['highly_variable']`` (or legacy
     ``'selected'`` with the deprecation warning).
  3. Tiles modality reads from ``ds.tiles[...]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def test_runSVD_cytome_GA_reads_GA_genes_not_genes(tmp_path):
    """If RNA and GA tables both have a 'highly_variable' column but
    with OPPOSITE masks, runSVD(modality='GA') must read from GA_genes
    (not genes). Bit-tests the registry-lookup fix."""
    import cytome
    import piaso

    p = tmp_path / "rna_plus_ga.cytome"
    ds = cytome.create(p)

    n_obs = 50
    rna_vars = 20
    ga_vars = 15
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))

    # RNA table — 'highly_variable' picks indices 0..9 of 20.
    rna_hv = np.zeros(rna_vars, dtype=bool)
    rna_hv[:10] = True
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(rna_vars),
        "gene_id": [f"r{i}" for i in range(rna_vars)],
        "highly_variable": rna_hv,
    }))

    # GA table — 'highly_variable' picks indices 5..14 of 15 (DIFFERENT mask
    # from RNA so we can tell which entity was read).
    ga_hv = np.zeros(ga_vars, dtype=bool)
    ga_hv[5:15] = True
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": np.arange(ga_vars),
        "gene_id": [f"ga{i}" for i in range(ga_vars)],
        "highly_variable": ga_hv,
    }))

    rng = np.random.default_rng(0)
    rna_X = sp.csr_matrix(rng.standard_normal((n_obs, rna_vars)).astype(np.float32))
    ga_X = sp.csr_matrix(rng.standard_normal((n_obs, ga_vars)).astype(np.float32))
    ds.add_matrix("RNA_counts", rna_X)
    ds.add_matrix("GA_counts", ga_X)
    ds.flush()
    ds.close()

    # Call runSVD with modality='GA' — registry lookup MUST resolve to
    # ds.GA_genes (the GA mask has 10 True entries), not ds.genes.
    piaso.tl.runSVD(str(p), modality="GA", n_components=3,
                    measurement="counts", verbosity=0)

    ds = cytome.open(p)
    assert "GA_svd" in ds.list_embeddings(), (
        f"Expected embedding 'GA_svd' to be written; got {ds.list_embeddings()}"
    )
    emb = ds.embeddings["GA_svd"]
    # If the registry lookup is correct, GA's 10-of-15 mask drives SVD —
    # embedding shape is (n_obs, 3). If the old bug applied and it read
    # ds.genes['highly_variable'] (10-of-20), SVD would have used 10 RNA
    # columns on the GA matrix shape mismatch — either crash or wrong
    # embedding. Shape sanity-check below.
    assert emb.shape == (n_obs, 3), f"GA_svd shape {emb.shape}, expected (50, 3)"
    ds.close()


def test_runSVD_cytome_ATAC_reads_peaks_table(tmp_path):
    """ATAC modality must read from ds.peaks (per registry), with
    'highly_variable' column when present."""
    import cytome
    import piaso

    p = tmp_path / "atac.cytome"
    ds = cytome.create(p)
    n_obs, n_vars = 40, 25
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    hv = np.zeros(n_vars, dtype=bool)
    hv[:15] = True
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": np.arange(n_vars),
        "peak_id": [f"p{i}" for i in range(n_vars)],
        "chr": ["chr1"] * n_vars,
        "start": np.arange(n_vars) * 1000,
        "end_": np.arange(n_vars) * 1000 + 500,
        "highly_variable": hv,
    }))
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.standard_normal((n_obs, n_vars)).astype(np.float32))
    ds.add_matrix("ATAC_counts", X)
    ds.flush()
    ds.close()

    piaso.tl.runSVD(str(p), modality="ATAC", n_components=4,
                    measurement="counts", verbosity=0)

    ds = cytome.open(p)
    assert "ATAC_svd" in ds.list_embeddings()
    assert ds.embeddings["ATAC_svd"].shape == (n_obs, 4)
    ds.close()


def test_runSVD_cytome_tiles_reads_tiles_table(tmp_path):
    """Tiles modality must read from ds.tiles (per registry). This was
    silently broken pre-Round-12: hardcoded fallback to ds.genes."""
    import cytome
    import piaso

    p = tmp_path / "tiles.cytome"
    ds = cytome.create(p)
    n_obs, n_vars = 30, 18
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    hv = np.zeros(n_vars, dtype=bool)
    hv[3:13] = True  # 10 tiles
    ds.set_entity("tiles", pd.DataFrame({
        "tile_idx": np.arange(n_vars),
        "tile_id": [f"t{i}" for i in range(n_vars)],
        "chr": ["chr1"] * n_vars,
        "start": np.arange(n_vars) * 500,
        "end_": np.arange(n_vars) * 500 + 250,
        "highly_variable": hv,
    }))
    rng = np.random.default_rng(2)
    X = sp.csr_matrix(rng.standard_normal((n_obs, n_vars)).astype(np.float32))
    ds.add_matrix("tiles_counts", X)
    ds.flush()
    ds.close()

    piaso.tl.runSVD(str(p), modality="tiles", n_components=3,
                    measurement="counts", verbosity=0)

    ds = cytome.open(p)
    assert "tiles_svd" in ds.list_embeddings()
    assert ds.embeddings["tiles_svd"].shape == (n_obs, 3)
    ds.close()
