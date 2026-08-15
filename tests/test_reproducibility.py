"""
Reproducibility tests: verify that streaming score/GDR produce
bit-identical results on repeated runs with the same random_seed.
"""
import sys, os
import numpy as np
import pytest
from conftest import E18_CYTOME

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/COSG")

# E18_CYTOME comes from conftest.py; @pytest.mark.requires_e18 applies the skip


def _skip_if_no_e18():
    if not os.path.exists(E18_CYTOME):
        pytest.skip("E18 cytome not available")


def _get_gene_sets(ds, n_sets=5, n_genes=20, seed=123):
    """Build small random gene sets from the cytome gene table."""
    gene_cols = ds.genes.columns
    for col in ["gene_id", "gene_name", "symbol"]:
        if col in gene_cols:
            vals = np.array(ds.genes[col])
            if vals[0] is not None:
                gene_names = vals
                break
    else:
        raise ValueError("Cannot resolve gene names")

    rng = np.random.RandomState(seed)
    sets = {}
    for i in range(n_sets):
        idx = rng.choice(len(gene_names), size=n_genes, replace=False)
        sets[f"set_{i}"] = list(gene_names[idx])
    return sets


def test_score_reproducibility():
    """score() with same seed produces bit-identical results on two runs."""
    _skip_if_no_e18()
    import cytome
    from piaso.tools._normalization import score

    ds = cytome.open(E18_CYTOME)

    gene_sets = _get_gene_sets(ds)

    sm1, names1, _ = score(
        ds, gene_sets,
        modality="RNA", cytome_layer="counts",
        batch_size=1024, layer=None,
        random_seed=42, use_rust=False, verbosity=0,
    )
    sm2, names2, _ = score(
        ds, gene_sets,
        modality="RNA", cytome_layer="counts",
        batch_size=1024, layer=None,
        random_seed=42, use_rust=False, verbosity=0,
    )

    ds.close()

    assert names1 == names2, f"Gene set names differ: {names1} vs {names2}"
    assert np.array_equal(sm1, sm2), (
        f"Score matrices not identical. Max diff: {np.max(np.abs(sm1 - sm2))}"
    )
    print(f"PASS: score() reproducibility — arrays bit-identical, shape {sm1.shape}")


def test_gdr_reproducibility():
    """runGDRParallel() with same seed produces bit-identical results on two runs."""
    _skip_if_no_e18()
    import cytome
    import piaso

    ds = cytome.open(E18_CYTOME)

    result1, markers1 = piaso.tl.runGDRParallel(
        ds,
        groupby="leiden_0.8",
        n_gene=30,
        mu=1.0,
        scoring_method="piaso",
        random_seed=42,
        modality="RNA",
        cytome_layer="counts",
        batch_size_cytome=1024,
        verbosity=0,
    )
    result2, markers2 = piaso.tl.runGDRParallel(
        ds,
        groupby="leiden_0.8",
        n_gene=30,
        mu=1.0,
        scoring_method="piaso",
        random_seed=42,
        modality="RNA",
        cytome_layer="counts",
        batch_size_cytome=1024,
        verbosity=0,
    )

    ds.close()

    # Check marker genes identical
    assert np.array_equal(markers1.values, markers2.values), (
        "Marker gene DataFrames differ"
    )

    # Check GDR scores identical
    assert np.array_equal(result1, result2), (
        f"GDR matrices not identical. Max diff: {np.max(np.abs(result1 - result2))}"
    )
    print(f"PASS: runGDRParallel() reproducibility — arrays bit-identical, shape {result1.shape}")


if __name__ == "__main__":
    test_score_reproducibility()
    test_gdr_reproducibility()
    print("\nAll reproducibility tests passed!")
