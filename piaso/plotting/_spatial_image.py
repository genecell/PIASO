"""Tissue-image resolution and overlay for spatial embeddings.

One resolver serves both backends — AnnData's ``uns['spatial']`` and a
cytome's ``spatial_images`` store — so ``pl.embedding``,
``pl.plot_embeddings_split`` and downstream packages share a single
implementation of the three details every spatial overlay gets wrong once:

- **Orientation.** Image row 0 is the *top* of the tissue. The image is drawn
  with ``extent=(0, W/sf, H/sf, 0)`` — full-resolution coordinate units with
  the y-axis increasing downward — so spot coordinates overlay with no
  scaling and no extra ``invert_yaxis()`` call. Panels that draw an image
  must keep y inverted; panels without one keep the existing y-up behaviour.
- **Units.** Spot coordinates are full-resolution pixels; the stored image is
  ``scalef`` times smaller. Putting the *image* into coordinate space (via
  ``extent``) beats scaling the coordinates, because axis limits, crops and
  spot sizes then all live in one unit system.
- **Spot size.** ``spot_diameter_fullres`` is a diameter in coordinate
  units. Scatter's ``s=`` is in points²; an ``EllipseCollection`` with
  ``units='xy'`` takes data units directly and needs no dpi arithmetic.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def _spatial_uns_from(data) -> dict:
    """A scanpy-style ``uns['spatial']`` dict from either backend, or ``{}``.

    Cytome first (``spatial_images`` accessor, cytome ≥ 0.2.6), then AnnData
    ``uns``. A file-backed cytome row whose decoder is missing degrades to
    no-image with a warning rather than failing a plot.
    """
    if isinstance(data, str):
        # a .cytome path — plotEmbedding accepts these, so the resolver must too
        try:
            from ..utils._cytome_compat import open_cytome_sync as _open
            with _open(data) as _ds:
                return _spatial_uns_from(_ds)
        except Exception as e:
            warnings.warn(f"could not read spatial images from {data!r}: {e}")
            return {}
    acc = getattr(data, "spatial_images", None)
    if acc is not None:
        try:
            return acc.as_uns() or {}
        except ImportError as e:
            warnings.warn(f"spatial image present but not decodable: {e}")
            return {}
    uns = getattr(data, "uns", None)
    if uns is not None and "spatial" in uns and isinstance(uns["spatial"], dict):
        return uns["spatial"]
    return {}


def _resolve_spatial_image(data, image, img_key: str = "hires",
                           library_values=None) -> Optional[dict]:
    """Resolve the ``image=`` argument to one drawable image, or ``None``.

    ``image`` is ``True`` (auto: the only library, else raise naming them) or
    a library id string. ``library_values`` (optional, the plotted cells'
    library labels) narrows auto-resolution: if the cells span exactly one
    library that has an image, that one wins even when the store has several.

    Returns ``{'img', 'scalef', 'spot_diameter', 'library', 'extent'}`` where
    ``extent`` places the image in full-resolution coordinate units with y
    increasing downward, or ``None`` when there is nothing to draw.
    """
    if not image:
        return None
    spatial = _spatial_uns_from(data)
    if not spatial:
        warnings.warn("image= requested but no spatial images found "
                      "(no uns['spatial'] / ds.spatial_images)")
        return None

    if isinstance(image, str):
        library = image
        if library not in spatial:
            raise KeyError(
                f"library {library!r} has no stored image; available: "
                f"{sorted(spatial)}")
    else:
        libs = sorted(spatial)
        if len(libs) == 1:
            library = libs[0]
        else:
            narrowed = None
            if library_values is not None:
                present = {str(v) for v in np.asarray(library_values)}
                candidates = [l for l in libs if l in present]
                if len(candidates) == 1:
                    narrowed = candidates[0]
            if narrowed is None:
                raise ValueError(
                    f"image=True is ambiguous: {len(libs)} libraries have "
                    f"images ({libs}). Pass image='<library_id>' or subset "
                    f"to one library.")
            library = narrowed

    entry = spatial[library]
    images = entry.get("images", {})
    if img_key not in images:
        raise KeyError(
            f"library {library!r} has no image {img_key!r}; available: "
            f"{sorted(images)}")
    img = np.asarray(images[img_key])
    sfs = entry.get("scalefactors", {})
    scalef = float(sfs.get(f"tissue_{img_key}_scalef", 1.0))
    spot_d = sfs.get("spot_diameter_fullres")
    h, w = img.shape[0], img.shape[1]
    return {
        "img": img,
        "scalef": scalef,
        "spot_diameter": (float(spot_d) if spot_d is not None else None),
        "library": library,
        # full-res units, y-down: (left, right, bottom, top)
        "extent": (0.0, w / scalef, h / scalef, 0.0),
    }


def _draw_image_overlay(ax, ctx: dict, image_alpha: float = 1.0) -> None:
    """Draw the resolved image under the scatter, in coordinate units."""
    img = ctx["img"]
    kwargs = {}
    if img.ndim == 2:
        kwargs["cmap"] = "gray"
    ax.imshow(img, extent=ctx["extent"], origin="upper", zorder=0,
              alpha=image_alpha, aspect="equal",
              interpolation="antialiased", **kwargs)


def _image_axis_limits(ax, coords, ctx: dict, pad_frac: float = 0.02) -> None:
    """Crop to the plotted cells: coord bounding box padded by a spot
    diameter (or ``pad_frac`` of the span), with y inverted to match the
    image's top-down orientation."""
    coords = np.asarray(coords, dtype=float)
    x0, x1 = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    y0, y1 = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    pad = ctx.get("spot_diameter") or 0.0
    pad = max(pad, pad_frac * max(x1 - x0, y1 - y0, 1.0))
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)      # inverted: image convention
