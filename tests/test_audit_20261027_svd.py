"""runSVD fixes of the 2026-10-27 audit (review H3, findings 5 and 8, and the minors).

- A modality other than RNA counts its own features: on an ATAC-only store
  ``ds.n_genes`` is 0, and runSVD(use_highly_variable=False) refused to run.
- ``tfidf_params`` without a ``col_mask`` and a selected-feature column: the
  columns were sliced before the TF-IDF, which then read the IDF of the first
  n_selected features. Now the selection becomes the TF-IDF's column mask.
- The Python TF-IDF path under ``cell_mask`` read each cell's depth by its
  renumbered row, i.e. the depths of the first n_masked cells.
- ``_mask_to_indices`` refuses negative, repeated and out-of-range indices.
- TF-IDF statistics with a non-finite entry are refused before any pass.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")


def _two_type_counts(rng, n_per, n_feat, n_marker):
    X = (rng.random((2 * n_per, n_feat)) < 0.03).astype(np.float32) * rng.integers(1, 3, (2 * n_per, n_feat))
    X[:n_per, :n_marker] += (rng.random((n_per, n_marker)) < 0.85) * 4
    X[n_per:, n_marker:2 * n_marker] += (rng.random((n_per, n_marker)) < 0.85) * 4
    X[:, -n_marker:] += (rng.random((2 * n_per, n_marker)) < 0.98) * 9
    return X.astype(np.float32)


def _atac_store(path, X):
    import cytome
    n, m = X.shape
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)]}))
    ds.set_entity("peaks", pd.DataFrame({"peak_idx": np.arange(m),
                                         "peak_id": [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(m)],
                                         "chr": ["chr1"] * m, "start": np.arange(m) * 1000,
                                         "end_": np.arange(m) * 1000 + 500}))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


@pytest.fixture
def atac(tmp_path):
    rng = np.random.default_rng(0)
    X = np.vstack([_two_type_counts(rng, 80, 400, 40), _two_type_counts(rng, 80, 400, 40)])
    # an IDF that varies strongly across features, so a mis-indexed one shows
    X[:, 200:] *= (rng.random((X.shape[0], 200)) < 0.3)
    X[:100] *= 3                                   # and depths that vary along the rows
    ds = _atac_store(tmp_path / "a.cytome", X)
    yield ds
    ds.close()


_KW = dict(modality="ATAC", n_components=6, return_svd=True, random_state=0, method="krylov",
           n_iter=10, tol=None, unit_length=False, n_threads=2)


def test_an_atac_only_store_counts_its_peaks(atac):
    import piaso
    assert atac.n_genes == 0
    e_all, S_all, _ = piaso.tl.runSVD(atac, use_highly_variable=False, **_KW)
    atac.peaks["highly_variable"] = np.ones(atac.n_peaks, dtype=bool)
    atac.flush()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # "every entry is selected"
        e_sel, S_sel, _ = piaso.tl.runSVD(atac, use_highly_variable=True, **_KW)
    assert e_all.shape == (atac.n_cells, 6)
    np.testing.assert_array_equal(S_all, S_sel)
    np.testing.assert_array_equal(e_all, e_sel)


def test_tfidf_params_without_col_mask_take_the_selection_as_their_mask(atac):
    import piaso
    hv = np.zeros(atac.n_peaks, dtype=bool)
    hv[150:] = True                                 # not the first features
    atac.peaks["highly_variable"] = hv
    atac.flush()
    stats = piaso.tl.compute_tfidf_stats(atac, measurement="counts", modality="ATAC")
    e_doc, S_doc, _ = piaso.tl.runSVD(atac, tfidf_params={k: v for k, v in stats.items()
                                                          if k != "col_mask"}, **_KW)
    e_ok, S_ok, _ = piaso.tl.runSVD(atac, tfidf_params={**stats, "col_mask": hv}, **_KW)
    np.testing.assert_array_equal(S_doc, S_ok)
    np.testing.assert_array_equal(e_doc, e_ok)
    # explicit indices are the selection too, and replace a col_mask
    e_idx, S_idx, _ = piaso.tl.runSVD(atac, tfidf_params={**stats, "col_mask": ~hv},
                                       hvg_indices_override=np.flatnonzero(hv), **_KW)
    np.testing.assert_array_equal(S_idx, S_ok)


def test_masked_tfidf_reads_each_cells_own_depth(atac):
    """Python and engine agree under cell_mask; before the fix the Python path
    read the depths of the first n_masked cells (S0 93.26 against 103.84)."""
    import piaso
    atac.peaks["highly_variable"] = np.ones(atac.n_peaks, dtype=bool)
    atac.flush()
    stats = piaso.tl.compute_tfidf_stats(atac, measurement="counts", modality="ATAC")
    mask = np.zeros(atac.n_cells, dtype=bool)
    mask[60:260] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eE, SE, _ = piaso.tl.runSVD(atac, tfidf_params=dict(stats), use_engine=True, cell_mask=mask, **_KW)
        eP, SP, _ = piaso.tl.runSVD(atac, tfidf_params=dict(stats), use_engine=False, cell_mask=mask, **_KW)
    np.testing.assert_allclose(SP, SE, rtol=1e-8)
    # and both equal the SVD of the masked cells' own TF-IDF, computed by hand
    from piaso.tools._runTFIDF import _normalize_chunk_tfidf
    X = atac.ATAC.counts.to_memory().tocsr()[mask]
    T = _normalize_chunk_tfidf(X, stats["cell_depth"][mask], stats["idf"], stats["scale_factor"])
    s_ref = np.linalg.svd(T.toarray(), compute_uv=False)[:6]
    np.testing.assert_allclose(SE, s_ref, rtol=1e-6)


def test_mask_indices_are_checked():
    from piaso.tools._runSVD import _mask_to_indices
    np.testing.assert_array_equal(_mask_to_indices(np.array([3, 1, 2]), 5), [1, 2, 3])
    np.testing.assert_array_equal(_mask_to_indices(np.array([True, False, True]), 5), [0, 2])
    with pytest.raises(IndexError):
        _mask_to_indices(np.array([-1, 2]), 5)
    np.testing.assert_array_equal(_mask_to_indices(np.array([1, 5]), 5), [1])   # dropped, as before
    with pytest.raises(ValueError, match="more than once"):
        _mask_to_indices(np.array([0, 1, 1, 0]), 5)


def test_a_scale_factor_of_zero_means_the_same_on_both_paths(atac):
    """Python reads scale_factor=0 as no scaling; the engine multiplied by 0."""
    import piaso
    stats = dict(piaso.tl.compute_tfidf_stats(atac, measurement="counts", modality="ATAC"))
    stats["scale_factor"] = 0
    eE, SE, _ = piaso.tl.runSVD(atac, tfidf_params=dict(stats), use_highly_variable=False,
                                use_engine=True, **_KW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eP, SP, _ = piaso.tl.runSVD(atac, tfidf_params=dict(stats), use_highly_variable=False,
                                    use_engine=False, **_KW)
    assert SE[0] > 0
    np.testing.assert_allclose(SE, SP, rtol=1e-8)


def test_non_finite_tfidf_statistics_are_refused(atac):
    import piaso
    from anndata import AnnData
    stats = dict(piaso.tl.compute_tfidf_stats(atac, measurement="counts", modality="ATAC"))
    bad = {**stats, "idf": np.full_like(stats["idf"], np.nan)}
    with pytest.raises(ValueError, match="IDF values are not finite"):
        piaso.tl.runSVD(atac, tfidf_params=bad, use_highly_variable=False, **_KW)
    a = AnnData(atac.ATAC.counts.to_memory().tocsr())
    with pytest.raises(ValueError, match="IDF values are not finite"):
        piaso.tl.runSVD(a, tfidf_params=bad, use_highly_variable=False, **_KW)
    with pytest.raises(ValueError, match="cell depths"):
        piaso.tl.runSVD(a, tfidf_params={**stats, "cell_depth": -stats["cell_depth"]},
                        use_highly_variable=False, **_KW)


def test_the_engine_checks_its_maps_and_statistics():
    """svd_pass assumed its maps numbered the kept rows and columns in order, and
    carried non-finite TF-IDF statistics into the products; it now refuses both,
    and reads a scale of 0 as no scaling, as the Python transform does."""
    from piaso import _piaso
    from piaso.tools._svd_memsource import MatrixChunkSource, engine_ready
    if not hasattr(_piaso, "svd_pass_source"):
        pytest.skip("extension without svd_pass_source")
    rng = np.random.default_rng(0)
    X = engine_ready(sp.random(50, 30, density=0.2, format="csr", random_state=0,
                               data_rvs=lambda s: rng.integers(1, 5, s)).astype(np.float32))
    depth = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    idf = np.log1p(50 / np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)).astype(np.float32)
    B = rng.standard_normal((30, 3))
    remap = np.arange(30, dtype=np.int32)

    def run(remap=remap, row_map=None, n_out=50, idf=idf, scale=1e4, n_sel=30):
        src = MatrixChunkSource(X, n_threads=2)
        try:
            return _piaso.svd_pass_source(src, "xb", B[:n_sel], remap, n_sel, row_map, n_out,
                                          depth, idf, scale, 2)
        finally:
            src.close()

    ok = run()
    with pytest.raises(ValueError, match="col_remap"):
        run(remap=remap[::-1].copy())
    bad_rows = np.full(50, -1, dtype=np.int32)
    bad_rows[[3, 1]] = [0, 1]                      # row 1 numbered after row 3
    with pytest.raises(ValueError, match="row_map"):
        run(row_map=bad_rows, n_out=2)
    with pytest.raises(ValueError, match="not finite"):
        run(idf=np.where(np.arange(30) == 4, np.nan, idf).astype(np.float32))
    np.testing.assert_array_equal(run(scale=0.0), run(scale=1.0))
    assert np.isfinite(ok).all()
