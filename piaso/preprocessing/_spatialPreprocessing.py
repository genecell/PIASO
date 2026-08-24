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


rotate_spatial_coordinates = rotateSpatialCoordinates


def _center_per_group(coords, groups, with_std=False):
    """Centre each group's coordinates on its own centroid.

    Pure geometry, shared by both backends. ``with_std=True`` also divides by
    the per-group standard deviation, which makes sections of different
    physical size comparable at the cost of no longer preserving scale.
    """
    coords = np.asarray(coords, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    out = coords.copy()
    for g in np.unique(groups):
        m = groups == g
        block = coords[m, :2]
        out[m, :2] = block - block.mean(axis=0)
        if with_std:
            sd = block.std(axis=0)
            sd[sd == 0] = 1.0
            out[m, :2] = out[m, :2] / sd
    return out


def alignSpatialCoordinates(
    data,
    groupby: Optional[str] = None,
    spatial_key: str = "spatial",
    key_added: str = "spatial_aligned",
    backup_spatial_key: Optional[str] = None,
    with_std: bool = False,
    inplace: bool = True,
    copy: bool = False,
    batch_key: Optional[str] = None,
):
    """Put every sample's spatial coordinates in a common frame.

    Sections are placed on their chips independently, so two samples can sit
    hundreds of microns apart in raw coordinates. Nothing is wrong with the
    data, but a split plot then renders each panel at a different offset and
    the tissues look displaced relative to one another. Centring each group on
    its own centroid fixes the display without touching within-sample geometry.

    The default is centring only (subtract the mean). ``with_std=True`` also
    divides by the per-group standard deviation, which equalises apparent
    section size — useful when samples differ a lot in extent, wrong when the
    relative sizes are part of what you are showing.

    Parameters
    ----------
    data
        AnnData, an open cytome dataset, or a path to a ``.cytome`` file.
    groupby
        Column naming the sample/section each cell belongs to
        (``adata.obs`` or ``ds.cells``).
    spatial_key
        Source coordinates. For AnnData an ``obsm`` key (``'spatial'`` and
        ``'X_spatial'`` both accepted); for a cytome a short embedding name
        resolved the same way ``basis=`` is in plotting.
    key_added
        Where the aligned coordinates are written.
    backup_spatial_key
        If given, the *original* coordinates are copied here first, so the
        un-aligned frame stays available.
    with_std
        Also scale each group to unit standard deviation.
    inplace
        AnnData only; ``False`` returns a modified copy. A cytome is always
        written in place.
    copy
        AnnData only; alias for ``not inplace``.

    Returns
    -------
    The modified AnnData when ``inplace=False``/``copy=True``, else ``None``.

    Examples
    --------
    >>> piaso.pp.alignSpatialCoordinates(adata, groupby="Sample")
    >>> piaso.pl.plot_embeddings_split(adata, color="cell_type",
    ...                                splitby="Sample",
    ...                                basis="spatial_aligned")
    """
    # `batch_key` and `groupby` name the same column here. GDR calls it a batch,
    # plotting calls it a group, and a reader coming from either should not
    # have to look it up.
    if groupby is None:
        groupby = batch_key
    if groupby is None:
        raise TypeError(
            "alignSpatialCoordinates needs the column naming each sample: "
            "pass groupby= (or batch_key=, same thing).")
    if batch_key is not None and batch_key != groupby:
        raise ValueError(
            f"groupby={groupby!r} and batch_key={batch_key!r} disagree; "
            "they name the same column, so pass only one.")

    if isinstance(data, anndata.AnnData):
        adata = data.copy() if (copy or not inplace) else data
        key = spatial_key if spatial_key in adata.obsm else f"X_{spatial_key}"
        if key not in adata.obsm:
            raise KeyError(
                f"'{spatial_key}' not in adata.obsm; available: "
                f"{list(adata.obsm)}")
        if groupby not in adata.obs.columns:
            raise KeyError(f"'{groupby}' not in adata.obs")
        coords = np.asarray(adata.obsm[key], dtype=np.float64)
        if backup_spatial_key:
            adata.obsm[backup_spatial_key] = coords.copy()
        adata.obsm[key_added] = _center_per_group(
            coords, adata.obs[groupby].values, with_std=with_std)
        return adata if (copy or not inplace) else None

    # cytome (open dataset or path)
    from ._spatial_cytome_io import _resolve_and_open
    ds, opened = _resolve_and_open(data)
    try:
        if not inplace:
            raise ValueError(
                "alignSpatialCoordinates writes to the cytome in place; "
                "inplace=False is not supported for a cytome input.")
        from ..plotting._plotEmbedding import _resolve_cytome_basis
        emb_name = _resolve_cytome_basis(ds, spatial_key)
        coords = np.asarray(ds.embeddings[emb_name], dtype=np.float64)
        groups = np.asarray(ds.cells[groupby]).astype(str)
        if backup_spatial_key:
            ds.add_embedding(backup_spatial_key, coords.astype(np.float32))
        aligned = _center_per_group(coords, groups, with_std=with_std)
        ds.add_embedding(key_added, aligned.astype(np.float32))
        ds.flush()
        return None
    finally:
        if opened:
            ds.close()


# camelCase / snake_case aliases, matching the rest of piaso.pp
align_spatial_coordinates = alignSpatialCoordinates
