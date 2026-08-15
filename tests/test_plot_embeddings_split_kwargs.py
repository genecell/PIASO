"""``plot_embeddings_split`` regression: per-panel scatter delegates to
``plotEmbedding`` so user kwargs (cmap, palette, frameon, point_size,
legend_fontsize) flow through. The five drops the user reported
(cmap/palette/frameon ignored, ``ncols`` vs ``ncol`` mismatch,
``size`` vs ``point_size`` mismatch) are checked here.

Tests use ``matplotlib.use('Agg')`` so they run headless in CI.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def _make_adata(n_cells=80, n_groups=3, n_samples=2, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        'Leiden': pd.Categorical(
            [f'g{i % n_groups}' for i in range(n_cells)],
            categories=[f'g{i}' for i in range(n_groups)],
        ),
        'Sample': pd.Categorical(
            [f's{i % n_samples}' for i in range(n_cells)],
            categories=[f's{i}' for i in range(n_samples)],
        ),
    })
    obsm = {'X_umap': rng.standard_normal((n_cells, 2))}
    a = ad.AnnData(
        X=np.zeros((n_cells, 1), dtype=np.float32),
        obs=obs, obsm=obsm,
    )
    return a


def _scatter_facecolors(ax):
    """Return list of unique RGBA tuples seen in the scatter PathCollections
    of an axes."""
    seen = []
    for c in ax.collections:
        fc = c.get_facecolors()
        if fc is None or len(fc) == 0:
            continue
        seen.append(tuple(np.round(fc[0], 5)))
    return seen


def test_palette_kwarg_takes_effect():
    """User-supplied palette is forwarded to per-panel plotEmbedding."""
    import piaso
    a = _make_adata()
    custom = ['#ff0000', '#00ff00', '#0000ff']  # R, G, B
    piaso.pl.plot_embeddings_split(
        a, color='Leiden', splitby='Sample',
        palette=custom, show_figure=False,
    )
    fig = plt.gcf()
    panel_axes = [ax for ax in fig.axes if ax.collections]
    assert len(panel_axes) >= 1, "At least one panel must be drawn"
    # Each panel should use one of the user palette colors. Convert palette
    # to RGBA at alpha=1.0 (matplotlib normalises to 0-1 floats).
    import matplotlib.colors as mcolors
    expected_rgba = {tuple(np.round(np.array(mcolors.to_rgba(c)), 5)) for c in custom}
    for ax in panel_axes:
        seen = _scatter_facecolors(ax)
        assert seen, f"Panel has no scatter colors: {ax}"
        for sc in seen:
            assert sc in expected_rgba, (
                f"Scatter color {sc} not in user palette {expected_rgba}"
            )
    plt.close('all')


def test_frameon_kwarg_takes_effect():
    """frameon=True shows axis spines on each panel; frameon=False hides them."""
    import piaso
    a = _make_adata()

    # frameon=True
    piaso.pl.plot_embeddings_split(
        a, color='Leiden', splitby='Sample',
        frameon=True, show_figure=False,
    )
    fig_on = plt.gcf()
    panel_axes_on = [ax for ax in fig_on.axes if ax.collections]
    for ax in panel_axes_on:
        visible = [s.get_visible() for s in ax.spines.values()]
        assert any(visible), (
            f"frameon=True should leave at least one spine visible; "
            f"got all hidden on {ax}"
        )
    plt.close('all')

    # frameon=False
    piaso.pl.plot_embeddings_split(
        a, color='Leiden', splitby='Sample',
        frameon=False, show_figure=False,
    )
    fig_off = plt.gcf()
    panel_axes_off = [ax for ax in fig_off.axes if ax.collections]
    for ax in panel_axes_off:
        visible = [s.get_visible() for s in ax.spines.values()]
        assert not any(visible), (
            f"frameon=False should hide all spines; got at least one visible on {ax}"
        )
    plt.close('all')


def test_cmap_kwarg_takes_effect_for_numeric():
    """For numeric color, the user's cmap is forwarded to plotEmbedding's
    scatter call (so the panel uses that colormap, not the hardcoded
    default)."""
    import piaso
    a = _make_adata()
    rng = np.random.default_rng(1)
    a.obs['expr'] = rng.uniform(0, 1, size=a.n_obs)
    piaso.pl.plot_embeddings_split(
        a, color='expr', splitby='Sample',
        cmap='viridis', show_figure=False,
    )
    fig = plt.gcf()
    panel_axes = [ax for ax in fig.axes if ax.collections]
    assert panel_axes
    for ax in panel_axes:
        sc = ax.collections[0]
        assert sc.cmap.name == 'viridis', (
            f"Expected cmap=viridis, got {sc.cmap.name} on {ax}"
        )
    plt.close('all')


def test_size_alias_for_point_size():
    """`size=10` flows through as point_size, just like in scanpy/plotEmbedding."""
    import piaso
    a = _make_adata()
    piaso.pl.plot_embeddings_split(
        a, color='Leiden', splitby='Sample',
        size=10.0, show_figure=False,
    )
    fig = plt.gcf()
    panel_axes = [ax for ax in fig.axes if ax.collections]
    assert panel_axes
    for ax in panel_axes:
        for sc in ax.collections:
            sizes = sc.get_sizes()
            if len(sizes) == 0:
                continue
            assert float(sizes[0]) == pytest.approx(10.0), (
                f"Expected point size 10.0, got {sizes[0]} on {ax}"
            )
    plt.close('all')


def test_ncols_alias_for_ncol():
    """`ncols=2` is accepted (scanpy-style) and used as the column count."""
    import piaso
    a = _make_adata(n_groups=4, n_samples=4)  # 4 panels
    piaso.pl.plot_embeddings_split(
        a, color='Leiden', splitby='Sample',
        ncols=2, show_figure=False,
    )
    fig = plt.gcf()
    panel_axes = [ax for ax in fig.axes if ax.collections]
    # 4 panels in a 2-column grid → 2 rows. Inspect grid spec for the grid.
    if panel_axes:
        gs = panel_axes[0].get_gridspec()
        assert gs.ncols == 2, f"Expected ncols=2 grid, got {gs.ncols}"
    plt.close('all')


def test_legend_fontsize_propagates_to_global_legend():
    """legend_fontsize controls the figure-level legend's font."""
    import piaso
    a = _make_adata()
    piaso.pl.plot_embeddings_split(
        a, color='Leiden', splitby='Sample',
        legend_fontsize=15, show_figure=False,
    )
    fig = plt.gcf()
    legends = [child for child in fig.get_children()
               if isinstance(child, matplotlib.legend.Legend)]
    assert legends, "Expected a figure-level legend for categorical data"
    for leg in legends:
        for txt in leg.get_texts():
            assert int(txt.get_fontsize()) == 15, (
                f"Expected legend font size 15, got {txt.get_fontsize()}"
            )
    plt.close('all')


def test_global_legend_colors_match_palette_across_panels():
    """The figure-level legend must use the same color-per-category mapping
    as the per-panel scatters — same category gets same colour everywhere."""
    import piaso
    import matplotlib.colors as mcolors
    a = _make_adata()
    custom = ['#ff0000', '#00ff00', '#0000ff']  # R G B
    piaso.pl.plot_embeddings_split(
        a, color='Leiden', splitby='Sample',
        palette=custom, show_figure=False,
    )
    fig = plt.gcf()
    legends = [child for child in fig.get_children()
               if isinstance(child, matplotlib.legend.Legend)]
    assert legends
    leg = legends[0]
    label_to_color = {}
    for handle, txt in zip(leg.legendHandles if hasattr(leg, 'legendHandles')
                           else leg.legend_handles, leg.get_texts()):
        label = txt.get_text()
        face = handle.get_markerfacecolor()
        rgba = tuple(np.round(mcolors.to_rgba(face), 5))
        label_to_color[label] = rgba

    # Every legend label must show a colour from our palette
    expected_rgba = {tuple(np.round(np.array(mcolors.to_rgba(c)), 5)) for c in custom}
    for label, rgba in label_to_color.items():
        assert rgba in expected_rgba, (
            f"Legend label {label!r} has color {rgba}, "
            f"not in user palette {expected_rgba}"
        )
    plt.close('all')


def test_cytome_input_works_end_to_end(tmp_path):
    """The refactor must keep working for cytome inputs (the user's split
    plot is typically against a cytome). End-to-end: build a tiny cytome
    with cells + Leiden + Sample obs columns + an embedding, call the
    function, expect it to render without raising."""
    import cytome
    import piaso
    import scipy.sparse as sp

    work = tmp_path / "tiny.cytome"
    ds = cytome.create(work)
    n = 60
    rng = np.random.default_rng(0)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n),
        "barcode": [f"AAA-{i}" for i in range(n)],
        "Leiden": [f"g{i % 3}" for i in range(n)],
        "Sample": [f"s{i % 2}" for i in range(n)],
    }))
    ds.add_embedding('X_umap', rng.standard_normal((n, 2)).astype(np.float32))
    ds.flush()

    piaso.pl.plot_embeddings_split(
        ds, color='Leiden', splitby='Sample',
        basis='X_umap', show_figure=False,
    )
    fig = plt.gcf()
    panels = [ax for ax in fig.axes if ax.collections]
    assert panels, "Cytome split plot produced no panels"
    plt.close('all')
    ds.close()
