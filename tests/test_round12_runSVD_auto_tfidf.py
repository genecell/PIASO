"""Round 12 regression: runSVD ``auto_tfidf=True`` parity with manual
``tfidf_params``.

Pre-Round-12, the workflow ``select_tfidf_svd.py`` had to:
  1. Compute tfidf_stats externally via ``compute_tfidf_stats(ds)``
  2. Build ``selected_mask = np.asarray(ds.peaks['selected']).astype(bool)``
  3. Stuff into ``tfidf_params={..., 'col_mask': selected_mask}``
  4. Call ``runSVD(..., tfidf_params=tfidf_params)``

Round 12 adds ``auto_tfidf=True`` which does steps 1-3 internally,
using ``_load_or_compute_tfidf_stats`` for the cache-or-compute and
``modality_var_entity`` + ``selected_feature_col_name`` to build
``col_mask``.

This test verifies the two paths produce IDENTICAL embeddings (up to
sign flips that are unobservable in absolute SVD column norms).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_atac_cytome(tmp_path, n_obs=60, n_vars=30):
    import cytome

    p = tmp_path / "atac_auto_tfidf.cytome"
    ds = cytome.create(p)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    hv = np.zeros(n_vars, dtype=bool)
    hv[:20] = True
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": np.arange(n_vars),
        "peak_id": [f"p{i}" for i in range(n_vars)],
        "chr": ["chr1"] * n_vars,
        "start": np.arange(n_vars) * 1000,
        "end_": np.arange(n_vars) * 1000 + 500,
        "highly_variable": hv,
    }))
    rng = np.random.default_rng(123)
    counts = (rng.random((n_obs, n_vars)) > 0.6).astype(np.float32)
    X = sp.csr_matrix(counts)
    ds.add_matrix("ATAC_counts", X)
    ds.flush()
    ds.close()
    return p, n_obs, n_vars


def test_runSVD_auto_tfidf_parity_with_manual_tfidf_params(tmp_path):
    """auto_tfidf=True must match the manual tfidf_params workflow path
    bit-identically (modulo SVD sign flips)."""
    import cytome
    import piaso
    from piaso.tools._runTFIDF import compute_tfidf_stats

    p, n_obs, n_vars = _build_atac_cytome(tmp_path)

    # Path A — manual (workflow's pre-Round-12 pattern)
    dsA = cytome.open(p)
    tfidf_stats = compute_tfidf_stats(dsA, modality="ATAC", measurement="counts",
                                       write_to_metadata=False)
    tfidf_stats["col_mask"] = np.asarray(
        dsA.peaks["highly_variable"]
    ).astype(bool)
    piaso.tl.runSVD(
        dsA, modality="ATAC", measurement="counts", streaming=True,
        tfidf_params=tfidf_stats, key_added="X_svd_manual",
        n_components=5, n_iter=7, random_state=10, verbosity=0,
    )
    emb_manual = np.asarray(dsA.embeddings["ATAC_svd_manual"]).copy()
    dsA.close()

    # Path B — auto_tfidf=True
    dsB = cytome.open(p)
    piaso.tl.runSVD(
        dsB, modality="ATAC", measurement="counts", streaming=True,
        auto_tfidf=True, key_added="X_svd_auto",
        n_components=5, n_iter=7, random_state=10, verbosity=0,
    )
    emb_auto = np.asarray(dsB.embeddings["ATAC_svd_auto"]).copy()
    dsB.close()

    # SVD has sign ambiguity per column; compare absolute values
    np.testing.assert_allclose(
        np.abs(emb_manual), np.abs(emb_auto), rtol=1e-4, atol=1e-5,
    )


def test_runSVD_auto_tfidf_on_anndata_raises_clear_error():
    """auto_tfidf=True is cytome-only — using it on AnnData should raise
    a ValueError that names auto_tfidf."""
    import piaso
    from anndata import AnnData

    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((20, 8)).astype(np.float32))
    adata = AnnData(X=X)
    hv = np.zeros(8, dtype=bool); hv[:5] = True
    adata.var["highly_variable"] = hv

    with pytest.raises(ValueError) as exc:
        piaso.tl.runSVD(adata, n_components=3, modality="ATAC",
                        auto_tfidf=True)
    assert "auto_tfidf" in str(exc.value)


def test_runSVD_auto_tfidf_populates_metadata_cache(tmp_path):
    """auto_tfidf=True with empty cache must populate
    ds.metadata['ATAC_tfidf_params'] as a side effect (for downstream
    COSG / plotting calls)."""
    import cytome
    import piaso

    p, _, _ = _build_atac_cytome(tmp_path)
    ds = cytome.open(p)
    assert "ATAC_tfidf_params" not in ds.metadata

    piaso.tl.runSVD(
        ds, modality="ATAC", measurement="counts", streaming=True,
        auto_tfidf=True, n_components=3, n_iter=5, verbosity=0,
    )
    assert "ATAC_tfidf_params" in ds.metadata, (
        "auto_tfidf=True must write tfidf params to cytome metadata "
        "so downstream COSG / plotting calls don't recompute."
    )
    ds.close()
