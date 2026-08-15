"""2026-06-15: plotEmbedding must treat INTEGER obs/cell columns as continuous.

Pre-fix `is_cat = not np.issubdtype(dtype, np.floating)` flagged integer metrics
(n_fragments, n_counts, n_peaks, raw counts) as categorical → discrete palette.
Fix: `not np.issubdtype(dtype, np.number)` (int + float continuous; object/str/
bool/category categorical).
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import anndata as ad
import scipy.sparse as sp

from piaso.plotting._plotEmbedding import _get_embedding_and_color


def _adata():
    a = ad.AnnData(X=sp.csr_matrix(np.random.poisson(1, (60, 4)).astype("f4")))
    a.obsm["X_umap"] = np.random.randn(60, 2)
    a.obs["n_fragments"] = np.random.randint(100, 9000, 60)        # int64
    a.obs["tss_score"] = np.random.rand(60) * 10                   # float64
    a.obs["Leiden"] = [str(i % 4) for i in range(60)]              # object/str
    a.obs["flag"] = (np.arange(60) % 2 == 0)                       # bool
    return a


def test_integer_column_is_continuous():
    a = _adata()
    _, _, is_cat, _ = _get_embedding_and_color(a, "X_umap", "n_fragments")
    assert is_cat is False           # the bug: int was flagged categorical


def test_float_column_is_continuous():
    a = _adata()
    _, _, is_cat, _ = _get_embedding_and_color(a, "X_umap", "tss_score")
    assert is_cat is False


def test_string_column_is_categorical():
    a = _adata()
    _, _, is_cat, _ = _get_embedding_and_color(a, "X_umap", "Leiden")
    assert is_cat is True


def test_bool_column_is_categorical():
    a = _adata()
    _, _, is_cat, _ = _get_embedding_and_color(a, "X_umap", "flag")
    assert is_cat is True            # bool is NOT a numeric subdtype → categorical
