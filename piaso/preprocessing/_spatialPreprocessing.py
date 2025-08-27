import numpy as np
import anndata
from typing import Optional, Union

### To rotate the spatial coordinates
def rotateSpatialCoordinates(
    adata: anndata.AnnData,
    angle_degrees: float,
    spatial_key: str = 'X_spatial',
    clockwise: bool = False,
    inplace: bool = True,
    backup_spatial_key: Optional[str] = None
) -> Optional[anndata.AnnData]:
    """
    Rotates the spatial coordinates in an AnnData object around their center.

    This function performs a 2D rotation on the coordinates stored in
    adata.obsm[spatial_key]. It first calculates the centroid of the
    coordinates, translates the data to center it at the origin, performs
    the rotation, and then translates it back.

    Args:
        adata:
            The annotated data matrix of shape `(n_obs, n_vars)`.
        angle_degrees:
            The angle of rotation in degrees.
        spatial_key:
            The key in `adata.obsm` where the spatial coordinates are stored.
            Defaults to 'X_spatial'.
        clockwise:
            If True, performs a clockwise rotation. If False (default),
            performs a counter-clockwise rotation (standard mathematical convention).
        inplace:
            If True (default), modifies the input AnnData object in place and returns None.
            If False, returns a new AnnData object with rotated coordinates.
        backup_spatial_key:
            If specified, the original spatial coordinates will be backed up in 
            `adata.obsm[backup_spatial_key]` before rotation. If None (default), no backup is created.

    Returns:
        If inplace=True, returns None and modifies the input adata object.
        If inplace=False, returns a new AnnData object with the rotated spatial coordinates.

    Raises:
        KeyError: If `spatial_key` is not found in `adata.obsm`.
        ValueError: If the coordinates in `adata.obsm[spatial_key]` are not 2D or 3D.

    Example:
        # Rotate coordinates in place with backup
        piaso.pp.rotateSpatialCoordinates(adata, 45, backup_spatial_key='X_spatial_original')
        
        # Rotate and create new object without backup
        adata_rotated = piaso.pp.rotateSpatialCoordinates(adata, 90, inplace=False)
    """
    # --- 1. Input Validation ---
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata.obsm. "
            f"Available keys are: {list(adata.obsm.keys())}"
        )

    # Choose whether to work on original or copy
    if inplace:
        adata_to_modify = adata
    else:
        adata_to_modify = adata.copy()
    
    coords = adata_to_modify.obsm[spatial_key]

    # --- 2. Backup original coordinates if requested ---
    if backup_spatial_key is not None:
        adata_to_modify.obsm[backup_spatial_key] = coords.copy()
        print(f"Original coordinates backed up in `adata.obsm['{backup_spatial_key}']`.")

    if coords.shape[1] < 2:
        raise ValueError(
            f"Coordinates in `adata.obsm['{spatial_key}']` must have at least 2 dimensions "
            f"for rotation, but found {coords.shape[1]}."
        )

    # --- 3. Center the coordinates ---
    center = np.mean(coords[:, :2], axis=0)
    coords_centered = coords[:, :2] - center

    # --- 4. Perform the rotation ---
    # Adjust angle for clockwise rotation if requested
    effective_angle = -angle_degrees if clockwise else angle_degrees
    
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(effective_angle)
    
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[c, -s],
                                [s,  c]])

    coords_rotated_centered = (rotation_matrix @ coords_centered.T).T

    # --- 5. Translate the coordinates back ---
    coords_rotated = coords_rotated_centered + center
    
    # --- 6. Update the AnnData object ---
    new_coords = coords.copy()
    new_coords[:, :2] = coords_rotated
    
    adata_to_modify.obsm[spatial_key] = new_coords
    
    direction = "clockwise" if clockwise else "counter-clockwise"
    print(f"Coordinates in `adata.obsm['{spatial_key}']` rotated by {angle_degrees} degrees {direction}.")
    
    # Return based on inplace parameter
    if inplace:
        return None
    else:
        return adata_to_modify