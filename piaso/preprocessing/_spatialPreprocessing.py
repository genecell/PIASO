"""Spatial-coordinate preprocessing, on AnnData or directly on a cytome.

The rotation is pure geometry on the (n, 2) coordinate array; the two
backends differ only in where the array lives and what must stay in sync:

- AnnData: ``adata.obsm[spatial_key]`` (unchanged behaviour).
- Cytome: the spatial embedding (``RNA_spatial`` etc., resolved from a short
  name like ``'spatial'``), rotated **on disk** — and when the rotated
  embedding is the spatial one, the ``spatial_coords`` R*-tree index is
  rebuilt in the same call, so ``cells_in_region`` keeps matching what is
  plotted.
"""
from typing import Optional

import anndata
import numpy as np


def _rotate_xy(coords: np.ndarray, angle_degrees: float,
               clockwise: bool) -> np.ndarray:
    """Rotate the first two columns about their centroid; extra columns pass
    through untouched. Returns a new array."""
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coordinates must be (n, >=2) for rotation; got {coords.shape}")
    center = np.mean(coords[:, :2], axis=0)
    centered = coords[:, :2] - center
    angle_rad = np.deg2rad(-angle_degrees if clockwise else angle_degrees)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotated = centered @ np.array([[c, s], [-s, c]]) + center
    out = np.array(coords, dtype=np.float64, copy=True)
    out[:, :2] = rotated
    return out


def _is_cytome(data) -> bool:
    try:
        from cytome.core.dataset import CytomeDataset
    except ImportError:
        return False
    return isinstance(data, CytomeDataset)


def rotateSpatialCoordinates(
    adata,
    angle_degrees: float,
    spatial_key: str = 'X_spatial',
    clockwise: bool = False,
    inplace: bool = True,
    backup_spatial_key: Optional[str] = None
):
    """Rotate spatial coordinates around their centroid.

    Works on an AnnData, an open cytome Dataset, or a path to a ``.cytome``
    file. On AnnData the coordinates live in ``adata.obsm[spatial_key]``; on a
    cytome, ``spatial_key`` resolves against the stored embeddings the same
    way plotting's ``basis=`` does (``'spatial'``, ``'X_spatial'`` and the
    full stored name all work), the rotation is written back to the file, and
    — when the rotated embedding is the spatial one — the ``spatial_coords``
    R*-tree index is rebuilt so ``cells_in_region`` stays consistent with the
    plot.

    Parameters
    ----------
    adata : AnnData, cytome.Dataset, or str
        The data whose coordinates to rotate.
    angle_degrees : float
        Rotation angle in degrees.
    spatial_key : str, default 'X_spatial'
        ``obsm`` key (AnnData) or embedding name / short name (cytome).
        Tip: converted cytomes store ``obsm['spatial']`` as
        ``{modality}_spatial`` — ``spatial_key='spatial'`` finds it.
    clockwise : bool, default False
        Rotate clockwise instead of the mathematical counter-clockwise.
    inplace : bool, default True
        AnnData only: modify in place (True) or return a rotated copy
        (False). A cytome is a file — the rotation is always written to it,
        and ``inplace=False`` raises rather than pretending to copy a
        dataset; use ``backup_spatial_key`` to keep the original.
    backup_spatial_key : str, optional
        Store the pre-rotation coordinates under this ``obsm`` key /
        embedding name first, so the original is one assignment away.
    """
    # ---------------- cytome (Dataset or path) ----------------
    if isinstance(adata, str) or _is_cytome(adata):
        if not inplace:
            raise ValueError(
                "inplace=False is AnnData-only: a cytome is a file and the "
                "rotation is written into it. Keep the original with "
                "backup_spatial_key=..., or copy the file first "
                "(ds.copy(path)).")
        from ..utils._cytome_compat import open_cytome_sync as _open

        def _run(ds):
            from ..plotting._plotEmbedding import _resolve_cytome_basis
            emb_name = _resolve_cytome_basis(ds, spatial_key)
            coords = np.asarray(ds.embeddings[emb_name])
            if backup_spatial_key is not None:
                ds.add_embedding(backup_spatial_key, coords)
                print(f"Original coordinates backed up in "
                      f"embeddings['{backup_spatial_key}'].")
            rotated = _rotate_xy(coords, angle_degrees, clockwise)
            ds.add_embedding(emb_name, rotated)
            # keep the ROI index consistent with what will be plotted
            if emb_name.endswith("_spatial") or emb_name in ("spatial", "X_spatial"):
                try:
                    ds.set_spatial_coords(rotated[:, :2])
                except Exception:      # cytome < 0.2.6: no index to rebuild
                    pass
            ds.flush()
            print(f"Coordinates in embeddings['{emb_name}'] rotated by "
                  f"{angle_degrees} degrees "
                  f"{'clockwise' if clockwise else 'counter-clockwise'}.")

        if isinstance(adata, str):
            with _open(adata) as ds:
                _run(ds)
        else:
            _run(adata)
        return None

    # ---------------- AnnData (behaviour unchanged) ----------------
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata.obsm. "
            f"Available keys are: {list(adata.obsm.keys())}"
        )

    adata_to_modify = adata if inplace else adata.copy()
    coords = adata_to_modify.obsm[spatial_key]

    if backup_spatial_key is not None:
        adata_to_modify.obsm[backup_spatial_key] = np.array(coords, copy=True)
        print(f"Original coordinates backed up in "
              f"`adata.obsm['{backup_spatial_key}']`.")

    adata_to_modify.obsm[spatial_key] = _rotate_xy(
        coords, angle_degrees, clockwise)
    print(f"Coordinates in `adata.obsm['{spatial_key}']` rotated by "
          f"{angle_degrees} degrees "
          f"{'clockwise' if clockwise else 'counter-clockwise'}.")

    return None if inplace else adata_to_modify
