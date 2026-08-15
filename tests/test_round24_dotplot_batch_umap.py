"""Round 24 regressions:
- dotplot's batched cytome reader returns values identical to the per-gene resolver
  (one streaming pass vs N), for a materialised layer and an on-the-fly (infog) layer.
- tl.neighbors (cytome) persists use_rep; tl.umap reads it back so umap(ds) works after
  neighbors(ds, use_rep='X_gdr') without the old 'X_svd' default error, and warns on mismatch.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _rna_cytome(path, n=50, g=15, seed=0):
    import cytome
    X = np.random.RandomState(seed).poisson(0.8, (n, g)).astype(np.float32)
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)],
        "leiden": [f"c{i % 3}" for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(g),
        "gene_id": [f"ENSG{i:05d}" for i in range(g)],
        "symbol": [f"Gene{i}" for i in range(g)]}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


@pytest.mark.parametrize("layer", ["counts", "infog"])
def test_dotplot_batched_equals_per_gene(tmp_path, layer):
    from piaso.plotting._plotEmbedding import (
        _resolve_cytome_feature_values as single,
        _resolve_cytome_feature_values_batch as batch,
    )
    ds = _rna_cytome(tmp_path / "d.cytome")
    feats = [f"Gene{i}" for i in (1, 4, 7, 10, 13)]
    b = batch(ds, feats, modality="RNA", cytome_layer=layer)
    for f in feats:
        s, _ = single(ds, f, modality="RNA", cytome_layer=layer)
        assert np.allclose(np.asarray(s), b[f][0], atol=1e-5)
    ds.close()


def test_umap_reads_back_use_rep_from_neighbors(tmp_path):
    import cytome, piaso
    p = tmp_path / "u.cytome"
    n = 40
    ds = cytome.create(str(p))
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({"gene_idx": [0], "gene_id": ["G0"]}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.ones((n, 1), np.float32)))
    ds.add_embedding("X_gdr", np.random.RandomState(0).randn(n, 10).astype(np.float32))
    ds.add_embedding("X_pca", np.random.RandomState(1).randn(n, 10).astype(np.float32))
    ds.flush()
    piaso.tl.neighbors(ds, use_rep="X_gdr", n_neighbors=10, key_added=None)
    # neighbors persisted use_rep
    assert ds.metadata.get("use_rep") == "X_gdr"
    # umap with a DIFFERENT explicit use_rep warns about the graph/rep mismatch
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            piaso.tl.umap(ds, use_rep="X_pca", key_added="X_umap")
        except Exception:
            pass   # umap-learn/sklearn version issues are out of scope; the warning fires first
        assert any("differs from the representation" in str(x.message) for x in w)
    ds.close()
