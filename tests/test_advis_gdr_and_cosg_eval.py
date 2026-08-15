"""
ADVIS 35-lib GDR benchmark + COSG streaming vs in-memory comparison + UMAP plots.

Run as script:
    python tests/test_advis_gdr_and_cosg_eval.py

Tasks:
  A) ADVIS GDR with streaming COSG — verify no OOM, measure time + RSS
  B) COSG comparison: streaming vs in-memory marker overlap + score correlation
  C) UMAP visualization on GDR embeddings with cell type legends
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/COSG")

ADVIS_CYTOME = "/path/to/project/results/snakemake/advis_35/pipeline.cytome"
# Standalone benchmark script: import the canonical E18 path from conftest
# (this file is also runnable directly with `python tests/test_advis_gdr_and_cosg_eval.py`,
# in which case Python finds conftest.py via the sys.path.insert above).
from conftest import E18_CYTOME  # noqa: E402
RADC_CYTOME = "/path/to/project/results/brain_7M/RADC_694K.cytome"
FIG_DIR = "/path/to/project/results/figures/streaming_cosg_eval"


def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0


# ===================================================================
#  COSG comparison: streaming vs in-memory
# ===================================================================

def compare_cosg(ds, groupby, modality, cytome_layer, n_gene=50):
    """Compare streaming COSG vs in-memory COSG marker genes + scores."""
    import cytome
    import cosg
    from cosg._cytome_streaming import run_cosg_cytome
    from scipy import sparse as sp_sparse
    import anndata

    print(f"\n--- COSG comparison: {modality}, groupby={groupby}, n_gene={n_gene} ---")

    # 1. Streaming COSG
    t0 = time.time()
    stream_result = run_cosg_cytome(
        cytome_path=str(ds.path),
        groupby=groupby,
        n_genes_user=n_gene,
        mu=1.0,
        remove_lowly_expressed=True,
        expressed_pct=0.1,
        modality=modality,
        batch_size=2048,
        verbose=False,
        feature_batching="auto",
    )
    t_stream = time.time() - t0
    stream_names = pd.DataFrame(
        stream_result['names'],
        columns=[str(g) for g in stream_result['groups_order']],
    )
    stream_scores = pd.DataFrame(
        stream_result['scores'],
        columns=[str(g) for g in stream_result['groups_order']],
    )
    print(f"  Streaming COSG: {t_stream:.1f}s, {stream_names.shape}")

    # 2. In-memory COSG (load full matrix)
    matrix_name = f"{modality}_{cytome_layer}"
    t0 = time.time()
    chunks = []
    for chunk_csr, _ri in ds.iter_chunks(modality=modality, layer=cytome_layer, batch_size=4096):
        chunks.append(chunk_csr)
    full_matrix = sp_sparse.vstack(chunks, format='csr')

    # Get feature names
    cursor = ds._conn.execute(
        "SELECT col_entity FROM matrix_meta WHERE matrix_name=?", (matrix_name,)
    )
    row = cursor.fetchone()
    entity_table = row[0] if row else "genes"
    if entity_table == "peaks":
        var_names = np.array(ds.peaks["peak_id"])
    elif entity_table == "tiles":
        var_names = np.array(ds.tiles["tile_id"])
    else:
        from piaso.tools._normalization import _resolve_gene_names
        var_names = _resolve_gene_names(ds)

    cluster_labels = np.array(ds.cells[groupby])
    adata_tmp = anndata.AnnData(X=full_matrix)
    adata_tmp.var_names = pd.Index(var_names)
    adata_tmp.obs[groupby] = pd.Categorical(cluster_labels)

    cosg.cosg(
        adata_tmp,
        key_added='cosg',
        mu=1.0,
        expressed_pct=0.1,
        remove_lowly_expressed=True,
        n_genes_user=n_gene,
        groupby=groupby,
    )
    inmem_names = pd.DataFrame(adata_tmp.uns['cosg']['names'])
    inmem_scores = pd.DataFrame(adata_tmp.uns['cosg']['scores'])
    t_inmem = time.time() - t0
    print(f"  In-memory COSG: {t_inmem:.1f}s, {inmem_names.shape}")

    del adata_tmp, full_matrix, chunks
    import gc; gc.collect()

    # 3. Compare markers
    results = []
    for col_s, col_m in zip(stream_names.columns, inmem_names.columns):
        s_genes = list(stream_names[col_s])
        m_genes = list(inmem_names[col_m])
        s_scores_arr = np.array(stream_scores[col_s], dtype=float)
        m_scores_arr = np.array(inmem_scores[col_m], dtype=float)

        # Top-K overlap at various K
        for k in [5, 10, 20, 50]:
            s_set = set(s_genes[:k])
            m_set = set(m_genes[:k])
            overlap = len(s_set & m_set) / k
            results.append({
                'cluster': col_s,
                'top_k': k,
                'overlap_frac': overlap,
                'n_overlap': len(s_set & m_set),
            })

        # Score correlation for shared genes
        shared = set(s_genes) & set(m_genes)
        if len(shared) > 5:
            s_idx = [s_genes.index(g) for g in shared]
            m_idx = [m_genes.index(g) for g in shared]
            s_sc = s_scores_arr[s_idx]
            m_sc = m_scores_arr[m_idx]
            from scipy.stats import pearsonr
            r, _ = pearsonr(s_sc, m_sc)
            results.append({
                'cluster': col_s,
                'top_k': 'score_r',
                'overlap_frac': r,
                'n_overlap': len(shared),
            })

    df = pd.DataFrame(results)
    return df, stream_names, inmem_names, stream_scores, inmem_scores


def plot_cosg_comparison(df, stream_names, inmem_names, stream_scores, inmem_scores, tag):
    """Generate comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)

    clusters = stream_names.columns.tolist()
    n_clusters = len(clusters)

    # Figure 1: Top-K overlap heatmap
    fig, axes = plt.subplots(1, 4, figsize=(20, max(4, n_clusters * 0.3 + 1)))
    for ax_idx, k in enumerate([5, 10, 20, 50]):
        subset = df[df['top_k'] == k].set_index('cluster')
        overlap_vals = [subset.loc[c, 'overlap_frac'] if c in subset.index else 0 for c in clusters]

        ax = axes[ax_idx]
        bars = ax.barh(range(n_clusters), overlap_vals, color='steelblue', edgecolor='white')
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels(clusters, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Overlap fraction')
        ax.set_title(f'Top-{k} marker overlap')
        ax.axvline(1.0, color='gray', ls='--', lw=0.5)

        # Annotate with count
        for i, (v, bar) in enumerate(zip(overlap_vals, bars)):
            ax.text(v + 0.02, i, f'{v:.0%}', va='center', fontsize=6)

    plt.suptitle(f'Streaming vs In-memory COSG: Marker Gene Overlap ({tag})', fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'cosg_marker_overlap_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Figure 2: Per-cluster score correlation (scatter per cluster)
    n_show = min(n_clusters, 12)
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx in range(n_show):
        ax = axes[idx // ncols, idx % ncols]
        col_s = stream_names.columns[idx]
        col_m = inmem_names.columns[idx]

        s_genes = list(stream_names[col_s])
        m_genes = list(inmem_names[col_m])
        s_sc = np.array(stream_scores[col_s], dtype=float)
        m_sc = np.array(inmem_scores[col_m], dtype=float)

        # All genes from both
        all_genes = list(dict.fromkeys(s_genes + m_genes))  # preserve order, unique
        s_dict = {g: s_sc[i] for i, g in enumerate(s_genes)}
        m_dict = {g: m_sc[i] for i, g in enumerate(m_genes)}

        x_vals, y_vals, colors = [], [], []
        for g in all_genes:
            sv = s_dict.get(g, 0)
            mv = m_dict.get(g, 0)
            x_vals.append(sv)
            y_vals.append(mv)
            if g in s_dict and g in m_dict:
                colors.append('steelblue')
            else:
                colors.append('salmon')

        ax.scatter(x_vals, y_vals, c=colors, s=8, alpha=0.6, edgecolors='none')
        ax.set_xlabel('Streaming score', fontsize=8)
        ax.set_ylabel('In-memory score', fontsize=8)
        ax.set_title(col_s[:30], fontsize=8)

        # Add diagonal
        lim = max(max(x_vals), max(y_vals)) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=0.5, alpha=0.3)

        # Overlap annotation
        shared = set(s_genes) & set(m_genes)
        ax.text(0.05, 0.95, f'{len(shared)}/{len(s_genes)} shared',
                transform=ax.transAxes, fontsize=7, va='top')

    # Hide unused axes
    for idx in range(n_show, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.suptitle(f'COSG Score Comparison: Streaming vs In-memory ({tag})', fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'cosg_score_scatter_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Figure 3: Summary bar chart — mean overlap per K
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in [5, 10, 20, 50]:
        subset = df[df['top_k'] == k]
        mean_ov = subset['overlap_frac'].mean()
        ax.bar(f'Top-{k}', mean_ov, color='steelblue', edgecolor='white')
        ax.text(f'Top-{k}', mean_ov + 0.01, f'{mean_ov:.1%}', ha='center', fontsize=10)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Mean overlap fraction')
    ax.set_title(f'Mean Marker Overlap Across Clusters ({tag})')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'cosg_mean_overlap_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
#  GDR + UMAP
# ===================================================================

def run_gdr_and_umap(ds, groupby, modality, cytome_layer, cell_type_col, tag):
    """Run runGDRParallel, then UMAP on the GDR embedding, plot with cell type."""
    import piaso
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)

    n_cells = ds.n_cells
    print(f"\n--- GDR + UMAP: {tag} ({n_cells:,} cells) ---")

    # Run GDR
    rss_before = get_rss_mb()
    t0 = time.time()
    result, markers = piaso.tl.runGDRParallel(
        ds,
        groupby=groupby,
        n_gene=30,
        mu=1.0,
        scoring_method="piaso",
        random_seed=42,
        modality=modality,
        cytome_layer=cytome_layer,
        batch_size_cytome=1024,
        verbosity=1,
    )
    gdr_time = time.time() - t0
    rss_after = get_rss_mb()
    print(f"  GDR: {gdr_time:.1f}s, RSS delta {rss_after - rss_before:.0f} MB, shape {result.shape}")

    # Get cell types
    cell_types = np.array(ds.cells[cell_type_col])
    unique_types = np.unique(cell_types)
    print(f"  Cell types: {len(unique_types)} unique ({cell_type_col})")

    # Run UMAP on GDR embedding
    print(f"  Computing UMAP on GDR ({result.shape[0]:,} × {result.shape[1]})...")
    try:
        import umap
        t0 = time.time()
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=8)
        embedding = reducer.fit_transform(result)
        umap_time = time.time() - t0
        print(f"  UMAP: {umap_time:.1f}s")
    except ImportError:
        print("  umap-learn not installed, skipping UMAP")
        return result, markers, gdr_time, rss_after - rss_before, None

    # Plot UMAP with cell type
    fig, ax = plt.subplots(figsize=(12, 10))

    # Assign colors
    type_counts = pd.Series(cell_types).value_counts()
    sorted_types = type_counts.index.tolist()

    # Use a good colormap
    if len(sorted_types) <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    colors = {t: cmap(i / max(len(sorted_types) - 1, 1)) for i, t in enumerate(sorted_types)}

    # Plot in random order to avoid overplotting bias
    rng = np.random.RandomState(42)
    order = rng.permutation(len(cell_types))

    for ct in sorted_types:
        mask = cell_types[order] == ct
        ax.scatter(
            embedding[order][mask, 0],
            embedding[order][mask, 1],
            c=[colors[ct]],
            s=0.5,
            alpha=0.3,
            label=f'{ct} ({type_counts[ct]:,})',
            rasterized=True,
        )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'GDR UMAP — {tag} ({n_cells:,} cells, {len(sorted_types)} types)')

    # Legend outside plot
    leg = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        markerscale=6,
        frameon=True,
        ncol=1 if len(sorted_types) <= 25 else 2,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'gdr_umap_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return result, markers, gdr_time, rss_after - rss_before, umap_time


# ===================================================================
#  Main
# ===================================================================

def main():
    import cytome

    # ── Part 1: COSG comparison on E18 (small, fast) ────────────
    print("=" * 70)
    print("PART 1: COSG comparison on E18 (4K cells, RNA)")
    print("=" * 70)

    ds = cytome.open(E18_CYTOME)
    df_e18, sn_e18, mn_e18, ss_e18, ms_e18 = compare_cosg(
        ds, groupby="leiden_0.8", modality="RNA", cytome_layer="counts", n_gene=50
    )
    plot_cosg_comparison(df_e18, sn_e18, mn_e18, ss_e18, ms_e18, tag="E18_RNA")

    # Print summary
    for k in [5, 10, 20, 50]:
        subset = df_e18[df_e18['top_k'] == k]
        print(f"  Top-{k}: mean overlap = {subset['overlap_frac'].mean():.1%}, "
              f"min = {subset['overlap_frac'].min():.1%}")

    # GDR + UMAP on E18
    result_e18, markers_e18, gdr_time_e18, gdr_rss_e18, umap_time_e18 = run_gdr_and_umap(
        ds, groupby="leiden_0.8", modality="RNA", cytome_layer="counts",
        cell_type_col="leiden_0.8", tag="E18_RNA"
    )
    ds.close()

    # ── Part 2: COSG comparison on RADC (694K cells, RNA) ────────
    print("\n" + "=" * 70)
    print("PART 2: COSG comparison on RADC 694K (RNA)")
    print("=" * 70)

    if os.path.exists(RADC_CYTOME):
        ds = cytome.open(RADC_CYTOME)
        df_radc, sn_radc, mn_radc, ss_radc, ms_radc = compare_cosg(
            ds, groupby="cell_type", modality="RNA", cytome_layer="counts", n_gene=50
        )
        plot_cosg_comparison(df_radc, sn_radc, mn_radc, ss_radc, ms_radc, tag="RADC_694K")

        for k in [5, 10, 20, 50]:
            subset = df_radc[df_radc['top_k'] == k]
            print(f"  Top-{k}: mean overlap = {subset['overlap_frac'].mean():.1%}, "
                  f"min = {subset['overlap_frac'].min():.1%}")

        # GDR + UMAP on RADC
        result_radc, markers_radc, gdr_time_radc, gdr_rss_radc, umap_time_radc = run_gdr_and_umap(
            ds, groupby="cell_type", modality="RNA", cytome_layer="counts",
            cell_type_col="cell_type", tag="RADC_694K"
        )
        ds.close()
    else:
        print("  RADC cytome not found, skipping")

    # ── Part 3: ADVIS GDR (200K cells, ATAC) ────────────────────
    print("\n" + "=" * 70)
    print("PART 3: ADVIS 35-lib GDR (200K cells, ATAC peaks)")
    print("=" * 70)

    if os.path.exists(ADVIS_CYTOME):
        ds = cytome.open(ADVIS_CYTOME)
        result_advis, markers_advis, gdr_time_advis, gdr_rss_advis, umap_time_advis = run_gdr_and_umap(
            ds, groupby="leiden_0.8", modality="ATAC", cytome_layer="counts",
            cell_type_col="cell_type", tag="ADVIS_35lib_ATAC"
        )
        ds.close()
    else:
        print("  ADVIS cytome not found, skipping")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"E18:   GDR {gdr_time_e18:.1f}s, {gdr_rss_e18:.0f} MB, "
          f"UMAP {umap_time_e18:.1f}s" if umap_time_e18 else "")
    if os.path.exists(RADC_CYTOME):
        print(f"RADC:  GDR {gdr_time_radc:.1f}s, {gdr_rss_radc:.0f} MB, "
              f"UMAP {umap_time_radc:.1f}s" if umap_time_radc else "")
    if os.path.exists(ADVIS_CYTOME):
        print(f"ADVIS: GDR {gdr_time_advis:.1f}s, {gdr_rss_advis:.0f} MB, "
              f"UMAP {umap_time_advis:.1f}s" if umap_time_advis else "")
    print(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
