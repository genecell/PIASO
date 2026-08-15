"""
ADVIS-scale memory benchmark: measure RSS for score() and runGDRParallel()
on 5-lib (~28K cells) and 35-lib (~200K cells) cytome datasets.

Not a pytest test — run directly:
  python tests/test_advis_memory_benchmark.py [--5lib-only] [--35lib-only]
"""
import sys, os, time, gc, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/path/to/project")

# ADVIS 5-lib has old schema (no min_start) — skip it
# ADVIS_5LIB = "/path/to/project/results/snakemake/advis_5lib_dim30/pipeline.cytome"
ADVIS_35LIB = "/path/to/project/results/snakemake/advis_35/pipeline.cytome"
# Standalone benchmark script: import the canonical E18 path from conftest
from conftest import E18_CYTOME  # noqa: E402


def get_rss_mb():
    """Get current RSS in MB from /proc/self/status."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024  # kB → MB
    return 0.0


def benchmark_score(cytome_path, label, modality="RNA"):
    """Run score() on cytome and measure memory + time."""
    import cytome
    from piaso.tools._normalization import score as piaso_score

    gc.collect()
    rss_before = get_rss_mb()
    print(f"\n{'='*60}")
    print(f"BENCHMARK: score() on {label} (modality={modality})")
    print(f"{'='*60}")

    ds = cytome.open(cytome_path)
    n_cells = ds.n_cells
    print(f"  Cells: {n_cells}")

    rss_after_open = get_rss_mb()
    print(f"  RSS after open: {rss_after_open:.0f} MB (delta: {rss_after_open - rss_before:.0f} MB)")

    # Get feature names based on modality
    import sqlite3
    cursor = ds._conn.execute(
        "SELECT col_entity FROM matrix_meta WHERE matrix_name=?",
        (f"{modality}_counts",)
    )
    row = cursor.fetchone()
    entity = row[0] if row else "genes"

    if entity == "peaks":
        feat_names = np.array(ds.peaks["peak_id"])
    elif entity == "tiles":
        feat_names = np.array(ds.tiles["tile_id"])
    else:
        from piaso.tools._normalization import _resolve_gene_names
        feat_names = _resolve_gene_names(ds)

    print(f"  Features: {len(feat_names)} ({entity})")

    rng = np.random.RandomState(42)
    n_sets = 10
    gene_sets = {}
    for i in range(n_sets):
        idx = rng.choice(len(feat_names), size=min(30, len(feat_names)), replace=False)
        gene_sets[f"set_{i}"] = list(feat_names[idx])

    gc.collect()
    rss_before_score = get_rss_mb()
    t0 = time.time()

    score_result, _, _ = piaso_score(
        ds,
        gene_sets,
        modality=modality,
        cytome_layer="counts",
        batch_size=1024,
        layer=None,
        use_rust=False,
        verbosity=1,
    )

    t_score = time.time() - t0
    rss_after_score = get_rss_mb()

    print(f"  Score shape: {score_result.shape}")
    print(f"  Score time: {t_score:.1f}s")
    print(f"  RSS after score: {rss_after_score:.0f} MB (delta: {rss_after_score - rss_before_score:.0f} MB)")
    print(f"  RSS peak estimate: {rss_after_score:.0f} MB")

    ds.close()
    del score_result
    gc.collect()

    return {
        "label": label,
        "function": "score",
        "n_cells": n_cells,
        "n_sets": n_sets,
        "time_s": round(t_score, 1),
        "rss_before_mb": round(rss_before_score),
        "rss_after_mb": round(rss_after_score),
        "rss_delta_mb": round(rss_after_score - rss_before_score),
    }


def benchmark_gdr(cytome_path, label, groupby, modality="RNA"):
    """Run runGDRParallel() on cytome and measure memory + time."""
    import cytome
    import piaso

    gc.collect()
    rss_before = get_rss_mb()
    print(f"\n{'='*60}")
    print(f"BENCHMARK: runGDRParallel() on {label}")
    print(f"{'='*60}")

    ds = cytome.open(cytome_path)
    n_cells = ds.n_cells

    # Check available cluster columns
    cell_cols = ds.cells.columns
    if groupby not in cell_cols:
        print(f"  WARNING: {groupby} not found in cells columns: {cell_cols}")
        # Try to find a suitable column
        for col in ["leiden_0.8", "leiden_1.0", "leiden_0.6", "Leiden", "celltype", "cell_type"]:
            if col in cell_cols:
                groupby = col
                print(f"  Using {groupby} instead")
                break

    cluster_labels = np.array(ds.cells[groupby])
    n_clusters = len(np.unique(cluster_labels))
    print(f"  Cells: {n_cells}, Clusters: {n_clusters} (groupby={groupby})")

    rss_after_open = get_rss_mb()
    print(f"  RSS after open: {rss_after_open:.0f} MB (delta: {rss_after_open - rss_before:.0f} MB)")

    gc.collect()
    rss_before_gdr = get_rss_mb()
    t0 = time.time()

    result = piaso.tl.runGDRParallel(
        ds,
        groupby=groupby,
        modality=modality,
        cytome_layer="counts",
        batch_size_cytome=1024,
        verbosity=1,
    )

    t_gdr = time.time() - t0
    rss_after_gdr = get_rss_mb()

    gdr_matrix = result[0]
    print(f"  GDR shape: {gdr_matrix.shape}")
    print(f"  GDR time: {t_gdr:.1f}s")
    print(f"  RSS after GDR: {rss_after_gdr:.0f} MB (delta: {rss_after_gdr - rss_before_gdr:.0f} MB)")

    ds.close()
    del result, gdr_matrix
    gc.collect()

    return {
        "label": label,
        "function": "runGDRParallel",
        "n_cells": n_cells,
        "n_clusters": n_clusters,
        "time_s": round(t_gdr, 1),
        "rss_before_mb": round(rss_before_gdr),
        "rss_after_mb": round(rss_after_gdr),
        "rss_delta_mb": round(rss_after_gdr - rss_before_gdr),
    }


def main():
    results = []

    # E18 (small, quick baseline — has RNA data)
    if os.path.exists(E18_CYTOME):
        results.append(benchmark_score(E18_CYTOME, "E18 (4K cells)", modality="RNA"))
        results.append(benchmark_gdr(E18_CYTOME, "E18 (4K cells)", "leiden_0.8", modality="RNA"))

    # ADVIS 35-lib (200K cells — ATAC-only, main scale test)
    if os.path.exists(ADVIS_35LIB):
        results.append(benchmark_score(ADVIS_35LIB, "ADVIS 35-lib (200K cells)", modality="ATAC"))
        results.append(benchmark_gdr(ADVIS_35LIB, "ADVIS 35-lib (200K cells)", "leiden_0.8", modality="ATAC"))

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Function':<20} {'Dataset':<25} {'Cells':>8} {'Time':>8} {'RSS delta':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['function']:<20} {r['label']:<25} {r['n_cells']:>8} {r['time_s']:>7.1f}s {r['rss_delta_mb']:>9} MB")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "tests", "advis_memory_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
