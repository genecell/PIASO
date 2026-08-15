"""
Large-scale benchmark: streaming score + GDR on RADC 694K cells.

Run as script (too slow for pytest):
    python tests/test_large_scale_benchmark.py

Tests:
  A) score() on RADC 694K — time + RSS
  B) runGDRParallel() with streaming COSG — time + RSS
  C) score() reproducibility on RADC — two runs, verify identical
"""
import sys, os, time
import numpy as np

# Not a pytest module: this is a standalone benchmark driven by main(), and its
# helpers take real cytome handles rather than fixtures. Marked so pytest does
# not try (and fail) to collect them.
__test__ = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/COSG")

RADC_CYTOME = "/path/to/project/results/brain_7M/RADC_694K.cytome"


def get_rss_mb():
    """Read current RSS from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        return 0.0


def get_gene_sets(ds, n_sets=10, n_genes=30, seed=123):
    """Build random gene sets from RNA genes."""
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


def benchmark_score(ds, gene_sets):
    """Benchmark streaming score() on cytome."""
    from piaso.tools._normalization import score

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

    print(f"\n=== score() benchmark ===")
    print(f"  Shape: {sm.shape}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  RSS: {rss_before:.0f} → {rss_after:.0f} MB (delta: {rss_after - rss_before:.0f} MB)")
    return sm, names, elapsed, rss_after - rss_before


def benchmark_gdr(ds):
    """Benchmark runGDRParallel() with streaming COSG on cytome."""
    import piaso

    rss_before = get_rss_mb()
    t0 = time.time()

    result, markers = piaso.tl.runGDRParallel(
        ds,
        groupby="cell_type",
        n_gene=30,
        mu=1.0,
        scoring_method="piaso",
        random_seed=42,
        modality="RNA",
        cytome_layer="counts",
        batch_size_cytome=1024,
        verbosity=1,
    )

    elapsed = time.time() - t0
    rss_after = get_rss_mb()

    print(f"\n=== runGDRParallel() benchmark ===")
    print(f"  GDR shape: {result.shape}")
    print(f"  Marker genes: {markers.shape}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  RSS: {rss_before:.0f} → {rss_after:.0f} MB (delta: {rss_after - rss_before:.0f} MB)")
    return result, markers, elapsed, rss_after - rss_before


def check_score_reproducibility(ds, gene_sets):
    """Verify score() produces identical results on two runs."""
    from piaso.tools._normalization import score

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

    assert names1 == names2
    identical = np.array_equal(sm1, sm2)
    if identical:
        print(f"\n=== score() reproducibility: PASS (bit-identical) ===")
    else:
        max_diff = np.max(np.abs(sm1 - sm2))
        print(f"\n=== score() reproducibility: FAIL (max diff: {max_diff}) ===")
    return identical


def main():
    if not os.path.exists(RADC_CYTOME):
        print(f"ERROR: RADC cytome not found at {RADC_CYTOME}")
        sys.exit(1)

    import cytome

    print(f"Opening {RADC_CYTOME}...")
    ds = cytome.open(RADC_CYTOME)
    n_cells = ds.n_cells
    print(f"  {n_cells:,} cells")

    gene_sets = get_gene_sets(ds)
    print(f"  {len(gene_sets)} gene sets, {len(next(iter(gene_sets.values())))} genes each")

    # Test A: score benchmark
    print("\n" + "=" * 60)
    print("TEST A: score() benchmark")
    print("=" * 60)
    sm, names, score_time, score_rss = benchmark_score(ds, gene_sets)

    # Test B: GDR benchmark (streaming COSG)
    print("\n" + "=" * 60)
    print("TEST B: runGDRParallel() benchmark (streaming COSG)")
    print("=" * 60)
    try:
        gdr, markers, gdr_time, gdr_rss = benchmark_gdr(ds)
        gdr_ok = True
    except Exception as e:
        print(f"  runGDRParallel() FAILED: {e}")
        import traceback; traceback.print_exc()
        gdr_ok = False
        gdr_time = gdr_rss = 0

    # Test C: score reproducibility
    print("\n" + "=" * 60)
    print("TEST C: score() reproducibility")
    print("=" * 60)
    repro_ok = check_score_reproducibility(ds, gene_sets)

    ds.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: RADC 694K ({n_cells:,} cells × 61,497 genes)")
    print(f"score():  {score_time:.1f}s, {score_rss:.0f} MB RSS delta, shape {sm.shape}")
    if gdr_ok:
        print(f"GDR:      {gdr_time:.1f}s, {gdr_rss:.0f} MB RSS delta, shape {gdr.shape}")
    else:
        print(f"GDR:      FAILED (see error above)")
    print(f"Reproducibility: {'PASS' if repro_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
