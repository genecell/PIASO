"""
MSSM 4.1M score() benchmark — million-cell scale validation.

Run as script:
    python tests/test_mssm_score_benchmark.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/COSG")

MSSM_CYTOME = "/path/to/project/results/brain_7M/MSSM_4.1M.cytome"


def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0


def main():
    if not os.path.exists(MSSM_CYTOME):
        print(f"ERROR: MSSM cytome not found at {MSSM_CYTOME}")
        sys.exit(1)

    import cytome
    from piaso.tools._normalization import score

    print(f"Opening {MSSM_CYTOME}...")
    ds = cytome.open(MSSM_CYTOME)
    n_cells = ds.n_cells
    print(f"  {n_cells:,} cells")

    # Build gene sets
    gene_cols = ds.genes.columns
    for col in ["gene_id", "gene_name", "symbol"]:
        if col in gene_cols:
            vals = np.array(ds.genes[col])
            if vals[0] is not None:
                gene_names = vals
                break

    rng = np.random.RandomState(123)
    gene_sets = {}
    for i in range(10):
        idx = rng.choice(len(gene_names), size=30, replace=False)
        gene_sets[f"set_{i}"] = list(gene_names[idx])
    print(f"  {len(gene_sets)} gene sets, 30 genes each")

    # Score benchmark
    print(f"\n{'='*60}")
    print(f"score() on MSSM 4.1M ({n_cells:,} cells × {len(gene_names):,} genes)")
    print(f"{'='*60}")

    rss_before = get_rss_mb()
    t0 = time.time()

    sm, names, _ = score(
        ds, gene_sets,
        modality="RNA", cytome_layer="counts",
        batch_size=1024, layer=None,
        random_seed=42, use_rust=False, verbosity=1,
    )

    elapsed = time.time() - t0
    rss_after = get_rss_mb()

    print(f"\n=== MSSM 4.1M score() results ===")
    print(f"  Shape: {sm.shape}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  RSS: {rss_before:.0f} → {rss_after:.0f} MB (delta: {rss_after - rss_before:.0f} MB)")

    # Reproducibility check
    print(f"\nReproducibility check...")
    sm2, names2, _ = score(
        ds, gene_sets,
        modality="RNA", cytome_layer="counts",
        batch_size=1024, layer=None,
        random_seed=42, use_rust=False, verbosity=0,
    )
    identical = np.array_equal(sm, sm2)
    print(f"  Bit-identical: {identical}")

    ds.close()

    print(f"\n{'='*60}")
    print(f"SUMMARY: MSSM 4.1M")
    print(f"{'='*60}")
    print(f"  Cells: {n_cells:,}")
    print(f"  Features: {len(gene_names):,}")
    print(f"  score() time: {elapsed:.1f}s")
    print(f"  score() RSS delta: {rss_after - rss_before:.0f} MB")
    print(f"  Reproducibility: {'PASS' if identical else 'FAIL'}")


if __name__ == "__main__":
    main()
