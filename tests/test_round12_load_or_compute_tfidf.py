"""Round 12 regression tests for ``_load_or_compute_tfidf_stats``.

The new helper (in ``piaso.tools._runTFIDF``) returns
``{modality}_tfidf_params`` from cache when present, otherwise computes
via one streaming pass and caches. It is shared by:

  - ``runSVD(auto_tfidf=True)``
  - ``cosg.run_cosg_cytome(layer='tfidf')`` (via ``_ensure_tfidf_params``)
  - ``piaso.pl.plotEmbedding(layer='tfidf')`` (via ``_ensure_tfidf_params``)

Round 12 also makes ``compute_tfidf_stats`` write a ``tfidf_idf`` column
to the modality's var entity (ds.peaks for ATAC, ds.tiles for tiles)
mirroring how ``piaso.tl.infog`` writes per-feature columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _make_atac_cytome(tmp_path):
    import cytome

    p = tmp_path / "atac.cytome"
    ds = cytome.create(p)
    n_obs, n_vars = 35, 22
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": np.arange(n_vars),
        "peak_id": [f"p{i}" for i in range(n_vars)],
        "chr": ["chr1"] * n_vars,
        "start": np.arange(n_vars) * 1000,
        "end_": np.arange(n_vars) * 1000 + 500,
    }))
    rng = np.random.default_rng(42)
    X = sp.csr_matrix((rng.random((n_obs, n_vars)) > 0.5).astype(np.float32))
    ds.add_matrix("ATAC_counts", X)
    ds.flush()
    ds.close()
    return p, n_obs, n_vars


def test_load_or_compute_tfidf_cache_miss_computes_and_writes_metadata(tmp_path):
    import cytome
    from piaso.tools._runTFIDF import _load_or_compute_tfidf_stats

    p, n_obs, n_vars = _make_atac_cytome(tmp_path)
    ds = cytome.open(p)
    assert "ATAC_tfidf_params" not in ds.metadata

    params = _load_or_compute_tfidf_stats(ds, modality="ATAC", layer="counts")
    assert set(params) >= {"cell_depth", "idf", "scale_factor"}
    assert params["idf"].shape == (n_vars,)
    assert "ATAC_tfidf_params" in ds.metadata
    ds.close()


def test_load_or_compute_tfidf_cache_hit_returns_cached_unchanged(tmp_path):
    import cytome
    from piaso.tools._runTFIDF import _load_or_compute_tfidf_stats

    p, _, _ = _make_atac_cytome(tmp_path)
    ds = cytome.open(p)
    p1 = _load_or_compute_tfidf_stats(ds, modality="ATAC", layer="counts")
    p2 = _load_or_compute_tfidf_stats(ds, modality="ATAC", layer="counts")
    np.testing.assert_array_equal(p1["idf"], p2["idf"])
    np.testing.assert_array_equal(p1["cell_depth"], p2["cell_depth"])
    assert p1["scale_factor"] == p2["scale_factor"]
    ds.close()


def test_load_or_compute_tfidf_force_recompute_overwrites_cache(tmp_path):
    import cytome
    from piaso.tools._runTFIDF import _load_or_compute_tfidf_stats

    p, _, _ = _make_atac_cytome(tmp_path)
    ds = cytome.open(p)
    p1 = _load_or_compute_tfidf_stats(ds, modality="ATAC", layer="counts",
                                       scale_factor=1e4)
    p2 = _load_or_compute_tfidf_stats(ds, modality="ATAC", layer="counts",
                                       scale_factor=2e4, force_recompute=True)
    assert p2["scale_factor"] == 2e4
    # And the cached entry should reflect the recompute
    assert ds.metadata["ATAC_tfidf_params"]["scale_factor"] == 2e4
    ds.close()


def test_compute_tfidf_stats_writes_tfidf_idf_to_peaks_entity(tmp_path):
    import cytome
    from piaso.tools._runTFIDF import compute_tfidf_stats

    p, _, n_vars = _make_atac_cytome(tmp_path)
    ds = cytome.open(p)
    params = compute_tfidf_stats(ds, modality="ATAC", measurement="counts")
    # Reopen to verify the column is persisted.
    ds.flush()
    ds.close()

    ds = cytome.open(p)
    cols = list(ds.peaks.columns)
    assert "tfidf_idf" in cols, (
        f"compute_tfidf_stats must write per-peak 'tfidf_idf' column "
        f"(Round 12, mirror of infog's per-feature column writes). "
        f"Got peaks.columns = {cols}"
    )
    persisted_idf = np.asarray(ds.peaks["tfidf_idf"], dtype=np.float32)
    np.testing.assert_allclose(persisted_idf, params["idf"].astype(np.float32),
                                rtol=1e-5)
    ds.close()
