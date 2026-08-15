"""Round 26: score() default modality='RNA', layer consolidation, duplicate-symbol union, pointed error."""
import warnings
import numpy as np, pandas as pd, pytest, scipy.sparse as sp


def _multi_cytome(path, n=80, g=40, dup_idx=20):
    import cytome
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)]}))
    sym = [f"Gene{i}" for i in range(g)]
    if dup_idx is not None:
        sym[dup_idx] = "Gene1"                        # duplicate symbol
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(g), "gene_id": [f"ENSG{i}" for i in range(g)], "symbol": sym}))
    ds.add_matrix("RNA_counts", sp.csr_matrix(np.random.RandomState(0).poisson(1, (n, g)).astype(np.float32)))
    # also an ATAC peaks matrix, so modality='ATAC' resolves a (wrong) feature table
    ds.set_entity("peaks", pd.DataFrame({
        "peak_idx": np.arange(5), "peak_id": [f"chr1:{i}-{i+1}" for i in range(5)],
        "chr": ["chr1"]*5, "start": np.arange(5)*100, "end_": np.arange(5)*100+50}))
    ds.add_matrix("ATAC_counts", sp.csr_matrix(np.ones((n, 5), np.float32)))
    ds.flush()
    return ds


def test_score_default_modality_is_rna(tmp_path):
    import piaso
    ds = _multi_cytome(tmp_path / "a.cytome", dup_idx=None)
    r = piaso.tl.score(ds, gene_list=["Gene1", "Gene2", "Gene5"], layer="counts")  # no modality=
    assert np.asarray(r[0]).shape[0] == ds.n_cells
    ds.close()


def test_score_cytome_layer_alias_is_removed():
    """1.2.0 removed the alias; `layer=` is canonical on both backends.

    Previously this asserted the alias warned and forwarded. It never shipped
    in a released version, so it was removed outright rather than given a
    deprecation cycle -- see tests/test_runGDR_cytome_layer_flow.py.
    """
    import inspect
    import piaso
    assert "cytome_layer" not in inspect.signature(piaso.tl.score).parameters
    with pytest.raises(TypeError, match="cytome_layer"):
        piaso.tl.score(None, cytome_layer="infog")


def test_score_duplicate_symbol_union_warns(tmp_path):
    import piaso
    ds = _multi_cytome(tmp_path / "c.cytome", dup_idx=20)
    with pytest.warns(UserWarning, match="duplicated"):
        piaso.tl.score(ds, gene_list=["Gene1", "Gene2"], modality="RNA", layer="counts")
    ds.close()


def test_score_wrong_modality_pointed_error(tmp_path):
    import piaso
    ds = _multi_cytome(tmp_path / "d.cytome", dup_idx=None)
    with pytest.raises(ValueError, match="modality"):
        piaso.tl.score(ds, gene_list=["Gene1", "Gene2"], modality="ATAC", layer="counts")
    ds.close()
