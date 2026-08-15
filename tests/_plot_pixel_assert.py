"""Pixel-coverage assertions for plotting tests.

Use these to catch the "all unit tests pass but the panel is empty
on screen" failure mode. Example failure that triggered the helper:
the Affine2D pyramid had a PolyCollection with N values but the
rotated bounding box landed outside ``xlim``, producing a blank
panel that AxesArtist-level checks missed.
"""
from __future__ import annotations

import numpy as np


def panel_non_white_fraction(fig, ax, *, white_tol=10):
    """Fraction of pixels in ``ax``'s bounding box that are not white.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure (must be drawn before calling — caller should
        ``fig.canvas.draw()`` first or rely on ``savefig``).
    ax : matplotlib.axes.Axes
        The axes whose bounding box we sample.
    white_tol : int
        A pixel counts as "white" if all of R, G, B are >= 255 - tol.

    Returns
    -------
    float
        Non-white fraction in ``[0.0, 1.0]``.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
    # rgba is (h, w, 4); ax.get_window_extent() is in display pixels
    bbox = ax.get_window_extent()
    h, w = rgba.shape[:2]
    # bbox.y0 is from bottom-left; matplotlib's image array is top-left
    x0 = int(max(0, np.floor(bbox.x0)))
    x1 = int(min(w, np.ceil(bbox.x1)))
    y0 = int(max(0, np.floor(h - bbox.y1)))
    y1 = int(min(h, np.ceil(h - bbox.y0)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = rgba[y0:y1, x0:x1, :3]
    if sub.size == 0:
        return 0.0
    is_white = np.all(sub >= (255 - white_tol), axis=-1)
    return float((~is_white).sum() / is_white.size)


def assert_panel_has_content(fig, ax, *, min_fraction=0.05,
                              name="panel"):
    """Assert that ``ax`` is not visually empty.

    Empty axes (only spines, no plotted content) have <1% non-white
    pixels in their bbox. Real content easily exceeds 5%.
    """
    frac = panel_non_white_fraction(fig, ax)
    assert frac >= min_fraction, (
        f"{name} appears empty: non-white pixel fraction = "
        f"{frac:.4f} (threshold {min_fraction:.2f}). Likely render "
        f"bug — panel exists but nothing drawn."
    )
    return frac
