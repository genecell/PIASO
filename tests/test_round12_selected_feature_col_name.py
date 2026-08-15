"""Round 12 (2026-05-26) regression tests for ``runSVD`` cross-modality
``selected_feature_col_name`` kwarg.

What changed in Round 12:

- ``runSVD`` gained a single ``selected_feature_col_name`` kwarg that
  replaces the historical hardcoded ATAC-vs-else dichotomy.
- Default is ``'highly_variable'`` (matches ``piaso.tl.infog`` /
  scanpy's ``highly_variable_genes`` output).
- ATAC and tiles callers with the legacy ``'selected'`` column (and
  no ``'highly_variable'``) get a ``DeprecationWarning`` and the
  fallback still works.
- AnnData and cytome paths share the same resolver semantics.

These tests pin:
  1. Default for the kwarg is ``'highly_variable'``.
  2. RNA (default modality) finds ``'highly_variable'`` without warning.
  3. ATAC with only legacy ``'selected'`` emits DeprecationWarning + works.
  4. ATAC with ``selected_feature_col_name='selected'`` explicit is silent.
  5. Unknown column raises ``KeyError`` with the column name in the message.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData


def test_runSVD_signature_has_selected_feature_col_name_default_highly_variable():
    import piaso
    sig = inspect.signature(piaso.tl.runSVD)
    assert "selected_feature_col_name" in sig.parameters
    assert sig.parameters["selected_feature_col_name"].default == "highly_variable"


def _rna_adata(n_obs=40, n_vars=20, seed=0, hvg_col="highly_variable"):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.standard_normal((n_obs, n_vars)).astype(np.float32))
    adata = AnnData(X=X)
    hv = np.zeros(n_vars, dtype=bool)
    hv[:10] = True
    adata.var[hvg_col] = hv
    return adata


def test_runSVD_rna_default_finds_highly_variable_silently():
    import piaso
    adata = _rna_adata()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        piaso.tl.runSVD(adata, n_components=5, modality="RNA")
    assert "X_svd" in adata.obsm
    assert adata.obsm["X_svd"].shape == (40, 5)


def test_runSVD_atac_legacy_selected_emits_deprecation_and_works():
    import piaso
    adata = _rna_adata(hvg_col="selected")  # legacy ATAC column name only
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        piaso.tl.runSVD(adata, n_components=5, modality="ATAC")
    deprecation_warnings = [
        wi for wi in w if issubclass(wi.category, DeprecationWarning)
        and "selected" in str(wi.message).lower()
    ]
    assert deprecation_warnings, (
        f"runSVD(modality='ATAC', legacy 'selected' col) must emit "
        f"DeprecationWarning. Got warnings: {[str(wi.message) for wi in w]}"
    )
    assert "X_svd" in adata.obsm


def test_runSVD_atac_explicit_selected_is_silent():
    import piaso
    adata = _rna_adata(hvg_col="selected")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Passing the column explicitly silences the deprecation fallback path.
        piaso.tl.runSVD(adata, n_components=5, modality="ATAC",
                        selected_feature_col_name="selected")
    assert "X_svd" in adata.obsm


def test_runSVD_unknown_column_raises_keyerror_with_column_name():
    import piaso
    adata = _rna_adata(hvg_col="highly_variable")
    with pytest.raises(KeyError) as exc:
        piaso.tl.runSVD(adata, n_components=5, modality="RNA",
                        selected_feature_col_name="nope_column_xyz")
    assert "nope_column_xyz" in str(exc.value)


def test_runSVD_default_kwarg_raises_when_neither_col_present():
    """If neither 'highly_variable' nor 'selected' exists, the resolver
    raises (rather than silently using all features)."""
    import piaso
    adata = _rna_adata(hvg_col="some_other_marker")
    with pytest.raises(KeyError):
        piaso.tl.runSVD(adata, n_components=5, modality="RNA")
