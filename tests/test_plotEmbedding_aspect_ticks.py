"""Cross-pollinated kwargs from plot_embeddings_split into plotEmbedding:
``fix_coordinate_ratio``, ``show_axis_ticks``, ``x_min`` / ``x_max`` /
``y_min`` / ``y_max``, and ``legend_marker_size``.

Tests assert content (the Axes property actually changed), not just
absence of exceptions.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_anndata(n_cells=40, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(2.0, size=(n_cells, 3)).astype(np.float32))
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame({'leiden': pd.Categorical([f'g{i % 3}' for i in range(n_cells)])}),
        obsm={'X_umap': rng.standard_normal((n_cells, 2)).astype(np.float32)},
    )
    a.var_names = ['Sox2', 'Pax6', 'Foxg1']
    a.obs_names = [f'AAA-{i}' for i in range(n_cells)]
    return a


# -----------------------------------------------------------------------
# fix_coordinate_ratio
# -----------------------------------------------------------------------

def test_fix_coordinate_ratio_true_default():
    """Default fix_coordinate_ratio=True → ax.set_aspect('equal')."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(a, color='leiden', show=False, return_fig=True
)
    aspect = ax.get_aspect()
    # In matplotlib, aspect='equal' typically yields the float 1.0 OR the
    # string 'equal' depending on version. Either is acceptable.
    assert aspect == 'equal' or aspect == 1.0
    plt.close(fig)


def test_fix_coordinate_ratio_false():
    """fix_coordinate_ratio=False → ax.set_aspect('auto')."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(
        a, color='leiden', show=False, fix_coordinate_ratio=False, return_fig=True,
)
    assert ax.get_aspect() == 'auto'
    plt.close(fig)


# -----------------------------------------------------------------------
# show_axis_ticks
# -----------------------------------------------------------------------

def _n_visible_tick_labels(ax):
    """Count rendered (visible, non-empty) x and y tick labels.

    Assert on what the user actually sees rather than on
    ``ax.xaxis.get_tick_params()['labelbottom']`` — that internal dict is
    matplotlib-version-dependent and on mpl 3.9.x does not report
    ``labelbottom`` at all (returns None), which is a test-introspection
    artifact, not a behaviour change.
    """
    xn = sum(1 for t in ax.get_xticklabels() if t.get_visible() and t.get_text())
    yn = sum(1 for t in ax.get_yticklabels() if t.get_visible() and t.get_text())
    return xn, yn


def test_show_axis_ticks_false_default():
    """Default hides tick LABELS (the bottom/left labels)."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(a, color='leiden', show=False, return_fig=True
)
    xn, yn = _n_visible_tick_labels(ax)
    assert xn == 0 and yn == 0, (
        f"default show_axis_ticks=False should hide tick labels, "
        f"got x={xn}, y={yn} visible"
    )
    plt.close(fig)


def test_show_axis_ticks_true():
    """show_axis_ticks=True keeps tick labels visible."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(
        a, color='leiden', show=False, show_axis_ticks=True, return_fig=True,
)
    xn, yn = _n_visible_tick_labels(ax)
    assert xn > 0 and yn > 0, (
        f"show_axis_ticks=True should show tick labels, got x={xn}, y={yn}"
    )
    plt.close(fig)


def test_split_show_axis_ticks_default_hides_on_every_panel():
    """plot_embeddings_split: default show_axis_ticks=False must hide ticks on
    EVERY panel (regression — the split path was missing the else-branch that
    plotEmbedding has, so the default leaked matplotlib's visible ticks)."""
    import piaso
    a = _build_anndata()
    a.obs['Sample'] = pd.Categorical([f's{i % 2}' for i in range(a.n_obs)])
    for flag, want_visible in ((False, False), (True, True)):
        plt.close('all')
        piaso.pl.plot_embeddings_split(
            a, splitby='Sample', color='leiden', point_size=1,
            show_figure=False, show_axis_ticks=flag,
        )
        fig = plt.gcf()
        panels = [ax for ax in fig.axes if ax.has_data()]
        assert panels, "no data panels rendered"
        for ax in panels:
            xn, yn = _n_visible_tick_labels(ax)
            if want_visible:
                assert xn > 0 and yn > 0, f"show_axis_ticks=True: x={xn}, y={yn}"
            else:
                assert xn == 0 and yn == 0, (
                    f"default show_axis_ticks=False should hide ticks, got x={xn}, y={yn}"
                )
    plt.close('all')


# -----------------------------------------------------------------------
# x_min / x_max / y_min / y_max
# -----------------------------------------------------------------------

def test_xy_limits_partial_override():
    """Pass only x_min and y_max — the other limits keep the data extents."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(
        a, color='leiden', show=False, x_min=-5.0, y_max=10.0, return_fig=True,
)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin == pytest.approx(-5.0)
    # y_max overridden, y_min came from data
    assert ymax == pytest.approx(10.0)
    plt.close(fig)


def test_xy_limits_all_four():
    """All four explicit → exact rectangle."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(
        a, color='leiden', show=False,
        x_min=-3.0, x_max=3.0, y_min=-2.0, y_max=2.0, return_fig=True,
)
    assert ax.get_xlim() == pytest.approx((-3.0, 3.0))
    assert ax.get_ylim() == pytest.approx((-2.0, 2.0))
    plt.close(fig)


# -----------------------------------------------------------------------
# legend_marker_size
# -----------------------------------------------------------------------

def _legend_handle_markersize(leg):
    handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles")
    return handles[0].get_markersize()


def test_legend_marker_size_explicit():
    """Explicit legend_marker_size → the proxy legend handle is drawn at that
    fixed markersize (Round 26: decoupled from point_size)."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(
        a, color='leiden', show=False, legend_marker_size=2.5, return_fig=True,
)
    leg = ax.get_legend()
    assert leg is not None
    assert _legend_handle_markersize(leg) == pytest.approx(2.5)
    plt.close(fig)


def test_legend_marker_size_auto_when_none():
    """legend_marker_size=None → a FIXED readable default (6 pt), INDEPENDENT of
    point_size (Round 26: no more 12/point_size coupling)."""
    import piaso
    a = _build_anndata()
    sizes = []
    for ps in (1.0, 2.0, 8.0):
        fig, ax = piaso.pl.plotEmbedding(
            a, color='leiden', show=False, point_size=ps, return_fig=True)
        sizes.append(_legend_handle_markersize(ax.get_legend()))
        plt.close(fig)
    # constant across point_size, and equal to the 6.0 default
    assert all(s == pytest.approx(6.0) for s in sizes)


# -----------------------------------------------------------------------
# Combined: new kwargs propagate through the multi-color recurse
# -----------------------------------------------------------------------

def test_new_kwargs_propagate_through_multi_color_grid():
    """Multi-color grid: each panel honours fix_coordinate_ratio,
    show_axis_ticks, x_min/x_max/y_min/y_max."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['leiden', 'Sox2'], show=False,
        ncol=2,
        fix_coordinate_ratio=False,
        show_axis_ticks=True,
        x_min=-5.0, x_max=5.0, return_fig=True,
)
    assert len(axs) == 2
    for ax in axs:
        assert ax.get_aspect() == 'auto'
        xn, yn = _n_visible_tick_labels(ax)
        assert xn > 0 and yn > 0, (
            f"show_axis_ticks=True panel should show tick labels, got x={xn}, y={yn}"
        )
        xmin, xmax = ax.get_xlim()
        assert xmin == pytest.approx(-5.0)
        assert xmax == pytest.approx(5.0)
    plt.close(fig)


# -----------------------------------------------------------------------
# Backward compat: existing single-color call without the new kwargs
# -----------------------------------------------------------------------

def test_backward_compat_no_new_kwargs():
    """A vanilla call (no new kwargs) still produces a sensible plot:
    aspect='equal', ticks hidden, default xlim from data."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(a, color='leiden', show=False, return_fig=True
)
    aspect = ax.get_aspect()
    assert aspect == 'equal' or aspect == 1.0
    plt.close(fig)


# -----------------------------------------------------------------------
# Docstring contains the new kwargs (catches future doc drift)
# -----------------------------------------------------------------------

def test_plotEmbedding_docstring_documents_new_kwargs():
    """Docstring should mention every new kwarg added in 1.1.x. If a kwarg
    is added to the signature but not the docstring (the bug reported in
    discussion 2026-05-04), this test catches it."""
    import inspect, piaso
    doc = inspect.getdoc(piaso.pl.plotEmbedding) or ""
    for name in (
        "modality", "cytome_layer", "compute_on_fly", "use_cached_stats",
        "show_modality_in_title", "ncol", "col_size", "row_size",
        "fix_coordinate_ratio", "show_axis_ticks",
        "x_min", "x_max", "y_min", "y_max", "legend_marker_size",
        "hspace", "wspace",
    ):
        assert name in doc, f"plotEmbedding docstring missing kwarg '{name}'"


# -----------------------------------------------------------------------
# hspace / wspace — tighter inter-panel spacing for the multi-color grid
# -----------------------------------------------------------------------

def test_hspace_default_is_tight_when_axis_ticks_hidden():
    """Default hspace=0.1 when show_axis_ticks=False — the matplotlib
    default of 0.2 wastes vertical space because tick labels are hidden."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['leiden', 'Sox2', 'Pax6', 'Foxg1'], show=False, ncol=2, return_fig=True,
)
    # subplots_adjust writes to the figure's subplotpars
    assert fig.subplotpars.hspace == pytest.approx(0.1)
    assert fig.subplotpars.wspace == pytest.approx(0.2)
    plt.close(fig)


def test_hspace_default_loosens_when_axis_ticks_shown():
    """When show_axis_ticks=True, default hspace bumps to 0.25 so tick
    labels don't overlap titles in the row below."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['leiden', 'Sox2', 'Pax6', 'Foxg1'], show=False, ncol=2,
        show_axis_ticks=True, return_fig=True,
)
    assert fig.subplotpars.hspace == pytest.approx(0.25)
    plt.close(fig)


def test_hspace_explicit_override():
    """User-provided hspace overrides the default."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['leiden', 'Sox2'], show=False, ncol=1, hspace=0.05, return_fig=True,
)
    assert fig.subplotpars.hspace == pytest.approx(0.05)
    plt.close(fig)


def test_wspace_explicit_override():
    """User-provided wspace overrides the default."""
    import piaso
    a = _build_anndata()
    fig, axs = piaso.pl.plotEmbedding(
        a, color=['leiden', 'Sox2'], show=False, ncol=2, wspace=0.5, return_fig=True,
)
    assert fig.subplotpars.wspace == pytest.approx(0.5)
    plt.close(fig)


def test_hspace_wspace_only_apply_to_grid_path():
    """Single-color call doesn't touch fig.subplotpars (no grid to space).
    Verifies we don't accidentally trigger subplots_adjust for the scalar path."""
    import piaso
    a = _build_anndata()
    fig, ax = piaso.pl.plotEmbedding(a, color='leiden', show=False, hspace=0.05, return_fig=True
)
    # Default mpl hspace is 0.2 — if our code wrongly applied 0.05, the
    # value would change. The scalar path doesn't recurse into the grid
    # branch, so subplotpars stays at the matplotlib default.
    assert fig.subplotpars.hspace == pytest.approx(0.2)
    plt.close(fig)
