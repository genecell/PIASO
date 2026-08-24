"""`basis='spatial'` must keep meaning the spatial embedding after an
aligned or backup copy is added beside it.

The resolver took the last substring match, so writing `spatial_aligned`
silently redirected every existing `basis='spatial'` call -- including
`rotateSpatialCoordinates(spatial_key='spatial')`, which then rotated the
aligned copy instead of the source.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from piaso.plotting._plotEmbedding import _resolve_cytome_basis


class _DS:
    def __init__(self, names): self._n = list(names)
    def list_embeddings(self): return self._n


def test_exact_name_wins():
    assert _resolve_cytome_basis(_DS(["spatial", "spatial_aligned"]), "spatial") == "spatial"


def test_modality_prefixed_name_beats_a_suffixed_sibling():
    ds = _DS(["RNA_spatial", "spatial_aligned", "spatial_chip"])
    assert _resolve_cytome_basis(ds, "spatial") == "RNA_spatial"


def test_legacy_obsm_name_still_resolves():
    ds = _DS(["RNA_obsm_spatial", "spatial_aligned"])
    assert _resolve_cytome_basis(ds, "spatial") == "RNA_obsm_spatial"


def test_derived_copies_are_still_reachable_by_their_own_name():
    ds = _DS(["RNA_spatial", "spatial_aligned"])
    assert _resolve_cytome_basis(ds, "spatial_aligned") == "spatial_aligned"


def test_umap_prefix_handling():
    ds = _DS(["RNA_umap", "X_umap_harmony"])
    assert _resolve_cytome_basis(ds, "X_umap") == "RNA_umap"


def test_missing_lists_what_exists():
    with pytest.raises(KeyError, match="Available"):
        _resolve_cytome_basis(_DS(["RNA_pca"]), "tsne")
