"""Harmony batch correction wrapper using harmonypy directly.

Supports both AnnData and cytome.Dataset as input.
"""

import numpy as np
import pandas as pd

from .._neighbors import _is_cytome_dataset, _load_embedding_from_cytome


def runHarmony(
    data,
    batch_key,
    use_rep='X_pca',
    key_added=None,
    random_state=0,
):
    """Run Harmony batch correction on an embedding.

    Corrects batch effects in a low-dimensional embedding using the
    Harmony algorithm (Korsunsky et al., 2019). Uses ``harmonypy``
    directly without scanpy dependency.

    Parameters
    ----------
    data : AnnData or cytome.Dataset
        Input data. For AnnData: reads embedding from ``obsm``, stores
        corrected embedding in ``obsm``. For cytome: reads/stores embeddings.
    batch_key : str
        Column in ``obs`` (AnnData) or ``cells`` (cytome) containing batch
        labels.
    use_rep : str, optional (default: 'X_pca')
        Key for the embedding to correct.
    key_added : str or None, optional (default: None)
        Key for the corrected embedding. If None, defaults to
        ``'{use_rep}_harmony'``.
    random_state : int, optional (default: 0)
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Corrected embedding matrix (n_cells, n_components).
    """
    import harmonypy

    if key_added is None:
        key_added = f'{use_rep}_harmony'

    # Load embedding and batch labels
    if _is_cytome_dataset(data):
        embedding = _load_embedding_from_cytome(data, use_rep)
        batch_labels = np.array(data.cells[batch_key])
    else:
        embedding = np.array(data.obsm[use_rep])
        batch_labels = data.obs[batch_key].values

    # Build metadata DataFrame for harmonypy
    meta_df = pd.DataFrame({batch_key: batch_labels})

    # Run Harmony
    ho = harmonypy.run_harmony(
        embedding, meta_df, batch_key,
        random_state=random_state,
    )
    Z = np.array(ho.Z_corr)
    # harmonypy >= 0.0.10 returns Z_corr as (n_cells, n_components);
    # older versions returned (n_components, n_cells). Normalize.
    if Z.shape[0] == embedding.shape[1] and Z.shape[1] == embedding.shape[0]:
        Z = Z.T
    corrected = Z

    # Store results
    if _is_cytome_dataset(data):
        data.add_embedding(key_added, corrected)
        data.flush()
    else:
        data.obsm[key_added] = corrected

    return corrected
