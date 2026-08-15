"""Round 25 regression: COSG→score gene-name handshake on cellranger cytomes.

Round 24 made COSG emit marker genes as SYMBOLS (via modality_feature_table_info),
but the score step's `_resolve_gene_names` still hard-coded gene_id-first → returned
Ensembl ids → the symbols never matched → runGDR/score raised "No valid gene sets
found" on any cytome with gene_id != symbol (cellranger). This pins the fix:
`_resolve_gene_names` now uses the single source of truth, and the lookup indexes
BOTH symbol and gene_id so a gene list in either vocabulary resolves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _cellranger_style_cytome(path, n=120, g=80, seed=0):
    """gene_id = Ensembl, symbol = gene name (the gene_id != symbol case)."""
    import cytome
    grp = np.array([i % 3 for i in range(n)])
    X = np.random.RandomState(seed).poisson(0.5, (n, g)).astype(np.float32)
    for k in range(3):
        X[grp == k, k * 25:k * 25 + 25] += 8
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)],
        "leiden": [f"c{x}" for x in grp]}))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(g),
        "gene_id": [f"ENSG{i:05d}" for i in range(g)],   # Ensembl ids
        "symbol": [f"Gene{i}" for i in range(g)]}))       # symbols (!= gene_id)
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


def test_resolve_gene_names_uses_symbol_first(tmp_path):
    from piaso.tools._normalization import _resolve_gene_names
    ds = _cellranger_style_cytome(tmp_path / "n.cytome")
    names = _resolve_gene_names(ds, "RNA")
    assert names[0] == "Gene0"                  # symbol, not 'ENSG00000'
    ds.close()


def test_score_resolves_symbol_and_ensembl_gene_sets(tmp_path):
    import piaso
    ds = _cellranger_style_cytome(tmp_path / "s.cytome")
    # SYMBOL gene sets (what COSG emits post-Round-24) — used to raise "No valid gene sets found"
    r_sym = piaso.tl.score(
        ds, {"setA": ["Gene0", "Gene1", "Gene2"], "setB": ["Gene25", "Gene26"]},
        modality="RNA", layer="counts")
    assert np.asarray(r_sym[0]).shape == (120, 2)
    # ENSEMBL gene set — alias-both lookup must also resolve it
    r_ens = piaso.tl.score(
        ds, {"setA": ["ENSG00000", "ENSG00001", "ENSG00002"]},
        modality="RNA", layer="counts")
    assert np.asarray(r_ens[0]).shape == (120, 1)
    ds.close()


def test_runGDR_end_to_end_on_gene_id_ne_symbol(tmp_path):
    """The user's failing case: runGDR on a cellranger cytome must complete
    (COSG symbols → score) instead of 'No valid gene sets found'."""
    import piaso
    ds = _cellranger_style_cytome(tmp_path / "g.cytome")
    piaso.tl.infog(ds, modality="RNA", save_layer=True)   # materialise RNA_infog
    piaso.tl.runGDR(ds, groupby="leiden", modality="RNA",
                    layer="infog", score_layer="infog", n_gene=10,
                    max_workers=1, verbosity=0)
    # X_gdr written to the cytome → runGDR completed past the scoring step.
    assert "X_gdr" in ds.list_embeddings()
    ds.close()
