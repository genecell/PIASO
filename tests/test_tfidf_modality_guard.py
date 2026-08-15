"""Regression test: TF-IDF cached-params must not leak across modalities.

The modality-blind legacy 'tfidf_params' (and any stale '{modality}_tfidf_params')
is ATAC-peak-scale; returning it for a 'tiles' request broadcast a wrong-length
idf and crashed COSG (`idf[chunk.indices]` IndexError). The feature-count guard
`_tfidf_idf_matches_modality` routes a mismatched payload to recompute.
"""
import numpy as np

from piaso.plotting._plotEmbedding import (
    _tfidf_idf_matches_modality,
    _params_feature_len_matches_modality,
)


class _FakeDS:
    """Minimal cytome stand-in exposing matrix_meta for the guard."""
    def __init__(self, n_cols_by_matrix):
        self._n = n_cols_by_matrix

    def matrix_meta(self, name):
        n = self._n.get(name)
        return None if n is None else {"n_cols": n, "n_rows": 100}


def test_atac_params_rejected_for_tiles():
    ds = _FakeDS({"ATAC_counts": 386360, "tiles_counts": 5451055})
    atac_params = {"idf": np.zeros(386360), "cell_depth": np.zeros(100)}
    # correct for ATAC, wrong for tiles
    assert _tfidf_idf_matches_modality(atac_params, ds, "ATAC") is True
    assert _tfidf_idf_matches_modality(atac_params, ds, "tiles") is False


def test_tiles_params_accepted_for_tiles():
    ds = _FakeDS({"tiles_counts": 5451055})
    tiles_params = {"idf": np.zeros(5451055), "cell_depth": np.zeros(100)}
    assert _tfidf_idf_matches_modality(tiles_params, ds, "tiles") is True


def test_permissive_when_meta_missing():
    ds = _FakeDS({})  # no matrix_meta → cannot verify → permissive (unchanged behaviour)
    params = {"idf": np.zeros(123), "cell_depth": np.zeros(100)}
    assert _tfidf_idf_matches_modality(params, ds, "ATAC") is True


def test_permissive_when_no_idf():
    ds = _FakeDS({"ATAC_counts": 386360})
    assert _tfidf_idf_matches_modality({"cell_depth": np.zeros(100)}, ds, "ATAC") is True


def test_infog_inv_gene_depth_guard():
    """Generalized guard: a legacy RNA infog payload (inv_gene_depth ~n_genes)
    must be rejected for an ATAC request (peaks ~n_peaks)."""
    ds = _FakeDS({"RNA_counts": 32285, "ATAC_counts": 386360})
    rna_infog = {"inv_gene_depth": np.zeros(32285), "cell_depth": np.zeros(100)}
    assert _params_feature_len_matches_modality(rna_infog, ds, "RNA", "inv_gene_depth") is True
    assert _params_feature_len_matches_modality(rna_infog, ds, "ATAC", "inv_gene_depth") is False


def test_chunk_tfidf_accepts_integer_counts():
    """Raw counts are commonly integer -- a Matrix Market import keeps int64.

    The in-place ops write float results into chunk.data, so an int chunk used
    to raise UFuncOutputCastingError ("Cannot cast ufunc 'divide' output from
    dtype('float64') to dtype('int64')") on the very first divide. This is the
    HDMA RNA atlas failure, reduced.
    """
    import numpy as np
    import scipy.sparse as sp
    from piaso.tools._runTFIDF import _normalize_chunk_tfidf

    counts = sp.csr_matrix(np.array([[0, 3, 0, 1],
                                     [2, 0, 5, 0]], dtype=np.int64))
    depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    idf = np.array([1.0, 2.0, 0.5, 1.5])

    out = _normalize_chunk_tfidf(counts, depth, idf, 1e4)

    assert np.issubdtype(out.dtype, np.floating)
    assert out.shape == counts.shape
    assert np.all(np.isfinite(out.data))
    # Same result as feeding the float version of the same matrix.
    ref = _normalize_chunk_tfidf(counts.astype(np.float32), depth, idf, 1e4)
    np.testing.assert_allclose(out.toarray(), ref.toarray(), rtol=1e-5)


def test_chunk_tfidf_does_not_mutate_the_input():
    """The int path must copy, not cast in place under the caller's feet."""
    import numpy as np
    import scipy.sparse as sp
    from piaso.tools._runTFIDF import _normalize_chunk_tfidf

    counts = sp.csr_matrix(np.array([[0, 3, 0, 1]], dtype=np.int64))
    before = counts.data.copy()
    _normalize_chunk_tfidf(counts, np.array([4.0]), np.ones(4), 1e4)
    assert counts.dtype == np.int64
    np.testing.assert_array_equal(counts.data, before)
