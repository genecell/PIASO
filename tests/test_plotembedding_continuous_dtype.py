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
    _, _, is_cat, _, _from_obs = _get_embedding_and_color(a, "X_umap", "n_fragments")
    assert is_cat is False           # the bug: int was flagged categorical


def test_float_column_is_continuous():
    a = _adata()
    _, _, is_cat, _, _from_obs = _get_embedding_and_color(a, "X_umap", "tss_score")
    assert is_cat is False


def test_string_column_is_categorical():
    a = _adata()
    _, _, is_cat, _, _from_obs = _get_embedding_and_color(a, "X_umap", "Leiden")
    assert is_cat is True


def test_bool_column_is_categorical():
    a = _adata()
    _, _, is_cat, _, _from_obs = _get_embedding_and_color(a, "X_umap", "flag")
    assert is_cat is True            # bool is NOT a numeric subdtype → categorical


def test_continuous_cmap_defaults_by_source(tmp_path):
    """A numeric CELL column and a FEATURE are both continuous, but they get
    different ramps: metadata reads as `Spectral_r`, expression as color_1.

    Colouring a regulon-activity column with the expression ramp made a
    metadata panel look like a gene panel, which is the confusion this
    prevents.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import piaso
    from piaso.plotting import _plotEmbedding as pe

    a = _adata()                       # has obs 'n_fragments' and var genes
    gene = str(a.var_names[0])

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="n_fragments", basis="X_umap",
                           ax=ax, show=False)
    meta_cmap = ax.collections[-1].get_cmap().name
    plt.close(fig)

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color=gene, basis="X_umap", ax=ax, show=False)
    feat_cmap = ax.collections[-1].get_cmap().name
    plt.close(fig)

    assert meta_cmap == "Spectral_r", meta_cmap
    assert meta_cmap != feat_cmap

    # an explicit cmap= still wins over both defaults
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="n_fragments", basis="X_umap",
                           cmap="viridis", ax=ax, show=False)
    assert ax.collections[-1].get_cmap().name == "viridis"
    plt.close(fig)
