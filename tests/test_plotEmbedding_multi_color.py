"""Multi-color list support for piaso.pl.plotEmbedding / plotUMAP.

Pre-fix, ``plotEmbedding(data, color=['gene1', 'gene2'])`` crashed with
``TypeError: unhashable type: 'list'`` deep in pandas Index lookup
(``if color in obs_df.columns``). Post-fix, a list / tuple of colors
builds an N-panel subplot grid (mirroring scanpy's
``sc.pl.umap(adata, color=[...])`` semantics).

Tests assert per-panel CONTENT (the right colour values landed on the
right axes), not just absence of exceptions — same principle as the
cytome plotting modality+layer tests.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_anndata(n_cells=40, n_genes=5, seed=0):
    """AnnData with a deterministic gene-expression pattern so we can
    assert per-panel colour values match the input."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    leiden = pd.Categorical([f'g{i % 3}' for i in range(n_cells)])
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    # Sox2: 5.0 in g0, 0 elsewhere; Pax6: 7.0 in g1, 0 elsewhere; etc.
    X[leiden == 'g0', 0] = 5.0
    X[leiden == 'g1', 1] = 7.0
    a = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({'leiden': leiden}),
        obsm={'X_umap': rng.standard_normal((n_cells, 2)).astype(np.float32)},
    )
    a.var_names = ['Sox2', 'Pax6', 'Foxg1', 'Olig2', 'Nestin'][:n_genes]
    a.obs_names = [f'AAA-{i}' for i in range(n_cells)]
    return a


def _build_multimodal_cytome(path, n_cells=24, seed=0):
    """RNA + GA + ATAC cytome with deterministic patterns per modality."""
    import cytome
    rng = np.random.default_rng(seed)
    leiden = np.array([f'g{i % 3}' for i in range(n_cells)])
    ds = cytome.create(path)
    ds.set_entity('cells', pd.DataFrame({
        'cell_idx': np.arange(n_cells),
        'barcode': [f'AAA-{i}' for i in range(n_cells)],
        'Leiden': leiden,
    }))
    # RNA: Sox2=5 in g0
    rna_X = np.zeros((n_cells, 3), dtype=np.float32)
    rna_X[leiden == 'g0', 0] = 5.0
    ds.set_entity('genes', pd.DataFrame({
        'gene_idx': [0, 1, 2],
        'gene_id': ['Sox2', 'Pax6', 'Foxg1'],
    }))
    ds.add_matrix('RNA_counts', sp.csr_matrix(rna_X))
    # GA: Pax6=9 in g1, Olig2=11 in g2 (different genes from RNA's Pax6 to test
    # explicit modality routing)
    ga_X = np.zeros((n_cells, 3), dtype=np.float32)
    ga_X[leiden == 'g1', 0] = 9.0
    ga_X[leiden == 'g2', 1] = 11.0
    ds.set_entity('GA_genes', pd.DataFrame({
        'gene_idx': [0, 1, 2],
        'gene_id': ['GA1', 'Olig2', 'Nestin'],
    }))
    ds.add_matrix('GA_counts', sp.csr_matrix(ga_X))
    ds.add_embedding('X_umap', rng.standard_normal((n_cells, 2)).astype(np.float32))
    ds.flush()
    return ds


# -----------------------------------------------------------------------
# 1. Backward compatibility — single-color path unchanged
# -----------------------------------------------------------------------

def test_single_color_str_unchanged():
    """color='leiden' (existing scalar form) still produces one Axes."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(a, color='leiden', show=False, return_fig=True
)
    assert ax.get_title() == 'leiden'
    plt.close(fig)


# -----------------------------------------------------------------------
# 2. Multi-color list path
# -----------------------------------------------------------------------

def test_multi_color_list_builds_panel_per_color():
    """color=[a, b, c] returns (fig, [ax0, ax1, ax2]) with the right titles."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['leiden', 'Sox2', 'Pax6'], show=False, ncol=3, return_fig=True,
)
    assert isinstance(axs, list) and len(axs) == 3
    assert [ax.get_title() for ax in axs] == ['leiden', 'Sox2', 'Pax6']
    plt.close(fig)


def test_multi_color_panels_carry_correct_values():
    """Sox2 panel's scatter colour values should contain {0.0, 5.0}; Pax6
    panel's should contain {0.0, 7.0}. Continuous-color plotEmbedding
    sorts points by value before drawing, so we check the SET of values
    (sorted unique), not per-cell ordering."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['Sox2', 'Pax6'], show=False, ncol=2, return_fig=True,
)
    sox2_uniq = sorted(set(np.round(axs[0].collections[0].get_array().data, 5)))
    assert sox2_uniq == [0.0, 5.0]
    pax6_uniq = sorted(set(np.round(axs[1].collections[0].get_array().data, 5)))
    assert pax6_uniq == [0.0, 7.0]
    plt.close(fig)


def test_multi_color_single_element_list_falls_through_to_scalar():
    """A 1-element list should give the SAME shape as a scalar str — single
    Axes, not a 1-panel grid (avoids subplot layout overhead)."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(a, color=['leiden'], show=False, return_fig=True
)
    # The scalar path returns a single Axes, not a list.
    assert hasattr(ax, 'get_title')  # an Axes
    assert not isinstance(ax, list)
    assert ax.get_title() == 'leiden'
    plt.close(fig)


def test_empty_color_list_raises():
    import piaso
    a = _build_anndata()
    with pytest.raises(ValueError, match="empty list"):
        piaso.pl.plotEmbedding(a, color=[], show=False, return_fig=True
)


def test_color_list_plus_ax_raises():
    """A list of colors needs its own grid, can't be drawn into an existing Axes."""
    import piaso
    a = _build_anndata()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="`color` as a list together with `ax`"):
        piaso.pl.plotEmbedding(a, color=['leiden', 'Sox2'], ax=ax, return_fig=True
)
    plt.close(fig)


def test_ncols_alias_for_ncol():
    """scanpy users write `ncols`, PIASO conventional is `ncol`. Both should work.
    Verify via figure size: ncol*col_size wide, nrow*row_size tall."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['Sox2', 'Pax6', 'Foxg1', 'Olig2'], show=False,
        ncols=2, col_size=4.0, row_size=4.0, return_fig=True,
)
    # 4 panels in a 2-col grid → 2 rows; figure 8 wide × 8 tall.
    assert len(axs) == 4
    w, h = fig.get_size_inches()
    assert w == pytest.approx(2 * 4.0)  # ncol * col_size
    assert h == pytest.approx(2 * 4.0)  # nrow * row_size
    plt.close(fig)


def test_ncol_default_is_ceil_sqrt():
    """No `ncol` given → defaults to ceil(sqrt(n)) per _build_subplots.
    9 panels → 3×3 grid → 12 wide × 12 tall (col_size/row_size=4)."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['leiden'] * 9, show=False, col_size=4.0, row_size=4.0, return_fig=True,
)
    assert len(axs) == 9
    w, h = fig.get_size_inches()
    assert w == pytest.approx(3 * 4.0)
    assert h == pytest.approx(3 * 4.0)
    plt.close(fig)


def test_unknown_kwarg_raises_typeerror():
    """An accidental misspelling should still raise — we accept ncols as alias
    but unknown kwargs must surface, not be silently absorbed."""
    import piaso
    a = _build_anndata()
    with pytest.raises(TypeError, match="unexpected keyword"):
        piaso.pl.plotEmbedding(
            a, color=['leiden', 'Sox2'], show=False, totally_made_up_kwarg=True, return_fig=True,
)


# -----------------------------------------------------------------------
# 3. Multi-color × cytome × explicit modality
# -----------------------------------------------------------------------

def test_multi_color_cytome_explicit_modality(tmp_path):
    """plotEmbedding(ds, color=['Sox2', 'Olig2'], modality='RNA' | 'GA') —
    each panel honours its own modality dispatch via the resolver.

    Here we use a single modality across all colours; mixed-modality is
    a follow-up feature.
    """
    import piaso, cytome
    ds = _build_multimodal_cytome(tmp_path / 'multi.cytome')
    leiden = np.asarray(ds.cells['Leiden']).astype(str)
    fig, axs = piaso.pl.plotEmbedding(
        ds, color=['Olig2', 'Nestin'],
        basis='X_umap', modality='GA', show=False, return_fig=True,
)
    assert len(axs) == 2
    # Olig2 panel: from GA matrix — values are {0.0, 11.0} (continuous color
    # path sorts by value before drawing, so check the unique set).
    olig2_uniq = sorted(set(np.round(axs[0].collections[0].get_array().data, 5)))
    assert olig2_uniq == [0.0, 11.0]
    plt.close(fig)
    ds.close()


# -----------------------------------------------------------------------
# 4. plotUMAP — wraps plotEmbedding via **kwargs, inherits multi-color
# -----------------------------------------------------------------------

def test_plotUMAP_inherits_multi_color():
    """plotUMAP is a `**kwargs` passthrough — list-color should work too."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotUMAP(a, color=['leiden', 'Sox2'], show=False, ncol=2, return_fig=True
)
    assert isinstance(axs, list) and len(axs) == 2
    assert [ax.get_title() for ax in axs] == ['leiden', 'Sox2']
    plt.close(fig)


# -----------------------------------------------------------------------
# 5. Trailing empty axes are hidden when n_colors < nrow*ncol
# -----------------------------------------------------------------------

def test_trailing_empty_axes_hidden():
    """3 colours into a 2×2 grid (ncol=2) → the 4th cell should be invisible."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['Sox2', 'Pax6', 'Foxg1'], show=False, ncol=2, return_fig=True,
)
    # 3 panels rendered, 1 hidden
    visible = [ax for ax in axs if ax.get_visible()]
    hidden = [ax for ax in axs if not ax.get_visible()]
    assert len(visible) == 3
    assert len(hidden) == 1
    plt.close(fig)


# -----------------------------------------------------------------------
# 6. Original bug regression: cytome + list of genes + modality
# -----------------------------------------------------------------------

def test_user_reported_bug_cytome_list_modality(tmp_path):
    """The exact call from the user report:
        piaso.pl.plotUMAP(ds, color=['Satb2', 'Lhx6', 'Sst'], modality='GA', return_fig=True
)
    should no longer crash with TypeError: unhashable type: 'list'.
    Use synthetic GA gene names matching the test fixture."""
    import piaso
    ds = _build_multimodal_cytome(tmp_path / 'reported.cytome')
    fig, axs = piaso.pl.plotUMAP(
        ds, color=['GA1', 'Olig2', 'Nestin'],
        modality='GA', show=False, return_fig=True,
)
    assert len(axs) == 3
    plt.close(fig)
    ds.close()
