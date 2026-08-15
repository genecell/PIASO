"""Round 23 regressions:

1. ``runGDR(cytome, groupby=None)`` de-novo clustering must not crash. The bug:
   ``neighbors()`` on a cytome returns ``None`` (it writes the graph to disk under
   ``neighbors_TMP_GDR_*``), so the subsequent ``leiden()`` call defaulted its
   ``adjacency_key`` to ``'connectivities'`` — which doesn't exist — and raised
   ``KeyError: 'Unknown graph: connectivities'``. Fix: pass ``neighbors_key=``.

2. ``infog(cytome)`` returns ``None`` by default (results live in ds.metadata /
   ds.genes), with an opt-in ``return_info=True`` for the params dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_rna_cytome(path, n_cells=60, n_genes=40, seed=0):
    """RNA cytome with 3 latent groups + an RNA_infog matrix (for SVD)."""
    import cytome
    rng = np.random.default_rng(seed)
    grp = np.array([i % 3 for i in range(n_cells)])
    X = rng.poisson(0.3, size=(n_cells, n_genes)).astype(np.float32)
    # plant block structure so SVD/leiden find 3 clusters
    for g in range(3):
        cols = slice(g * 10, g * 10 + 10)
        X[grp == g, cols] += 6.0
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"AAA-{i}" for i in range(n_cells)],
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "gene_id": [f"G{i}" for i in range(n_genes)],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.add_matrix("RNA_infog", sp.csr_matrix(X))      # de-novo path reads this for SVD
    ds.flush()
    return ds


def test_runGDR_cytome_groupby_none_does_not_crash(tmp_path, monkeypatch):
    """The §6 regression: groupby=None must run SVD→neighbors→leiden on the
    cytome without the 'Unknown graph: connectivities' KeyError, and produce
    the temp cluster column the rest of GDR consumes."""
    import piaso
    from cosg import _cytome_streaming as _cosg_mod

    # Mock COSG + scoring so the test targets ONLY the de-novo clustering chain
    # (which is where the bug lived) and stays fast / data-light.
    def _mock_cosg(*args, **kwargs):
        n_top = kwargs.get("n_genes_user", 20)
        names = np.empty((n_top, 3), dtype=object); names[:] = "G0"
        return {"names": names, "scores": np.zeros((n_top, 3), np.float32),
                "groups_order": ["0", "1", "2"]}

    def _mock_score(*args, **kwargs):
        n = args[0].n_cells
        return np.zeros((n, 3), np.float32), ["0", "1", "2"]

    monkeypatch.setattr(_cosg_mod, "run_cosg_cytome", _mock_cosg)
    monkeypatch.setattr(piaso.tools._runGDR, "calculateScoreParallel", _mock_score)

    ds = _build_rna_cytome(tmp_path / "x.cytome")
    # Realistic flow: infog first (writes highly_variable + RNA_infog), then GDR.
    piaso.tl.infog(ds, modality="RNA")
    # groupby=None → de-novo SVD + neighbors + leiden (the path that crashed)
    piaso.tl.runGDR(
        ds, groupby=None, modality="RNA",
        n_svd_dims=10, n_svd_iter=5, resolution=1.0,
        max_workers=1, verbosity=0,
    )
    # The derived labels must have been written (no KeyError thrown above).
    assert "gdr_local_TMP_GDR" in list(ds.cells.columns)
    labels = np.asarray(ds.cells["gdr_local_TMP_GDR"])
    assert len(labels) == ds.n_cells and len(np.unique(labels)) >= 1
    ds.close()


def test_infog_cytome_returns_none_by_default(tmp_path):
    """§2: infog(cytome) returns None by default (info persisted to metadata);
    return_info=True yields the params dict."""
    import piaso
    ds = _build_rna_cytome(tmp_path / "y.cytome")
    out = piaso.tl.infog(ds, modality="RNA")
    assert out is None, "infog(cytome) should return None by default"
    # infog now persists the MODALITY-PREFIXED key (the un-prefixed legacy
    # 'infog_params' alias was dropped — it was modality-blind on read).
    assert "RNA_infog_params" in ds.metadata, "infog must persist prefixed params"
    assert "infog_params" not in ds.metadata, "legacy un-prefixed alias must NOT be written"

    out2 = piaso.tl.infog(ds, modality="RNA", return_info=True)
    assert isinstance(out2, dict) and "infog_params" in out2  # return-dict key, unaffected
    ds.close()


def test_infog_source_and_adata_aliases(tmp_path):
    """§3: `source=` and `adata=` keep working as deprecated aliases for `data=`."""
    import piaso
    ds = _build_rna_cytome(tmp_path / "z.cytome")
    with pytest.warns(FutureWarning):
        piaso.tl.infog(source=ds, modality="RNA")
    ds.close()
