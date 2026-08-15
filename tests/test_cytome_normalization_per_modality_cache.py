"""Per-modality params cache for normalization helpers.

Verifies that:
  - piaso.tl.infog writes ds.metadata['RNA_infog_params'] (new, namespaced)
    AND legacy ds.metadata['infog_params'].
  - piaso.tl.compute_tfidf_stats writes ds.metadata['{modality}_tfidf_params']
    when write_to_metadata=True (default).
  - The plotting on-the-fly path falls through to the legacy unprefixed
    key with a DeprecationWarning when only the legacy key is present.
  - use_cached_stats=False forces a fresh recompute and overwrites the
    cache (round-trips bit-identical).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_rna_cytome(path, n_cells=20, n_genes=8, seed=0):
    """Build an RNA-only cytome with a deterministic counts matrix so the
    INFOG params and TF-IDF stats are reproducible."""
    import cytome
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"AAA-{i}" for i in range(n_cells)],
        "Leiden": [f"g{i % 3}" for i in range(n_cells)],
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "gene_id": [f"Gene{i}" for i in range(n_genes)],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(counts))
    ds.flush()
    return ds


def test_infog_writes_prefixed_key_only(tmp_path):
    """piaso.tl.infog writes the MODALITY-PREFIXED 'RNA_infog_params' and NO
    LONGER the modality-blind un-prefixed 'infog_params' legacy alias (which was
    dropped because it clobbered / leaked across modalities on read)."""
    import piaso
    ds = _build_rna_cytome(tmp_path / "rna.cytome")
    piaso.tl.infog(ds, save_layer=False, streaming=True, verbosity=0)

    keys = {k for k in ds.metadata.keys()}
    assert "RNA_infog_params" in keys
    assert "infog_params" not in keys, "legacy un-prefixed alias must not be written"

    new = ds.metadata.get("RNA_infog_params")
    assert new is not None and "inv_gene_depth" in new and "scale" in new
    ds.close()


def test_compute_tfidf_stats_caches_to_metadata(tmp_path):
    """compute_tfidf_stats(write_to_metadata=True) populates the
    per-modality cache key. Default flips to True per the design."""
    import piaso
    import cytome
    # An ATAC-flavored cytome
    n = 10
    ds = cytome.create(tmp_path / "atac.cytome")
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"AAA-{i}" for i in range(n)],
    }))
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": np.arange(3),
        "peak_id": ["chr1:0-100", "chr1:200-300", "chr2:0-100"],
        "chr": ["chr1", "chr1", "chr2"],
        "start": [0, 200, 0],
        "end_": [100, 300, 100],
    }))
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.5, size=(n, 3)).astype(np.float32)
    ds.add_matrix("ATAC_counts", sp.csr_matrix(counts))
    ds.flush()

    params = piaso.tl.compute_tfidf_stats(
        ds, modality="ATAC", measurement="counts", batch_size=8,
    )
    assert "ATAC_tfidf_params" in {k for k in ds.metadata.keys()}
    # legacy modality-blind un-prefixed alias is no longer written
    assert "tfidf_params" not in {k for k in ds.metadata.keys()}
    cached = ds.metadata.get("ATAC_tfidf_params")
    assert cached is not None
    assert "cell_depth" in cached and "idf" in cached
    np.testing.assert_allclose(
        np.asarray(cached["cell_depth"]),
        np.asarray(params["cell_depth"]),
    )
    ds.close()


def test_legacy_infog_params_fall_through_with_deprecation(tmp_path):
    """A cytome with ONLY the legacy 'infog_params' key (no new one)
    should produce a DeprecationWarning when read via the resolver path,
    and still produce the right values."""
    import piaso
    from piaso.plotting._plotEmbedding import _ensure_infog_params

    ds = _build_rna_cytome(tmp_path / "rna.cytome")
    # infog now writes ONLY the prefixed key. To simulate a *legacy* cytome,
    # copy the prefixed payload to the un-prefixed 'infog_params' key (a real,
    # correctly-sized RNA payload so the read-side feature-count guard passes),
    # then delete the prefixed key.
    piaso.tl.infog(ds, save_layer=False, streaming=True, verbosity=0)
    ds.metadata["infog_params"] = ds.metadata.get("RNA_infog_params")
    ds.flush()  # persist the planted legacy key before the raw-SQL purge below
    ds._conn.execute("DELETE FROM _metadata WHERE key='RNA_infog_params'")
    ds._conn.commit()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = _ensure_infog_params(ds, "RNA", use_cached_stats=True)
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "Expected DeprecationWarning when falling through to legacy key."
    assert params is not None
    ds.close()


def test_use_cached_stats_false_forces_recompute(tmp_path):
    """use_cached_stats=False on the on-the-fly resolver path triggers a
    fresh compute (the cache is overwritten with the same values for a
    deterministic cytome — assert the result is bit-identical)."""
    import piaso
    from piaso.plotting._plotEmbedding import _resolve_cytome_feature_values

    ds = _build_rna_cytome(tmp_path / "rna.cytome")
    # First call: compute_on_fly=True populates the cache
    vals_1, _ = _resolve_cytome_feature_values(
        ds, "Gene0", modality="RNA",
        cytome_layer="infog", compute_on_fly=True, use_cached_stats=True,
    )
    # Second call with use_cached_stats=False: same values (deterministic)
    vals_2, _ = _resolve_cytome_feature_values(
        ds, "Gene0", modality="RNA",
        cytome_layer="infog", compute_on_fly=True, use_cached_stats=False,
    )
    np.testing.assert_allclose(vals_1, vals_2, rtol=1e-5)
    ds.close()


def test_log1p_cell_depth_caches_per_modality(tmp_path):
    """The per-modality `_modality_cell_depth` helper caches under
    'RNA_cell_depth' (not a global key) so RNA + GA can coexist."""
    from piaso.plotting._plotEmbedding import _modality_cell_depth
    ds = _build_rna_cytome(tmp_path / "rna.cytome")
    # First call computes and caches
    depth1 = _modality_cell_depth(ds, "RNA", use_cached_stats=True)
    assert "RNA_cell_depth" in {k for k in ds.metadata.keys()}
    # Second call should read cache (returns same array)
    depth2 = _modality_cell_depth(ds, "RNA", use_cached_stats=True)
    np.testing.assert_allclose(depth1, depth2)
    ds.close()
