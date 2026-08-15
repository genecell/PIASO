"""Unit test for the generic loop-contact scorer (HiChIP/PLAC-seq), synthetic — no data."""
import os, sys

import pytest
import numpy as np, pandas as pd
# The benchmark helpers live in the *parent* analysis repo (PIASO/benchmark/grn),
# not inside the package — they were moved there and these tests were never
# re-pointed. Skip cleanly when the package is checked out on its own.
_BENCH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmark", "grn"))
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_BENCH),
    reason=f"benchmark harness not present at {_BENCH} (parent analysis repo only)")
sys.path.insert(0, _BENCH)


def test_loop_contact_benchmark_synthetic():
    import orthogonal_gold as O
    # gene G1 TSS at chr1:100000 (+); its true enhancer peak at chr1:300000-300500.
    gene_tss = {"G1": ("chr1", 100000), "G2": ("chr1", 2000000)}
    # a loop: anchor A over G1 promoter (chr1:99-101k), anchor B over the enhancer (chr1:300-301k)
    loops = pd.DataFrame({"cA": ["chr1"], "sA": [99000], "eA": [101000],
                          "cB": ["chr1"], "sB": [300000], "eB": [301000]})
    tri = pd.DataFrame({
        "gene": ["G1", "G1", "G2"],
        "peak": ["chr1:300100-300400", "chr1:900000-900500", "chr1:300100-300400"],
        "chrom": ["chr1", "chr1", "chr1"],
        "ps": [300100, 900000, 300100], "pe": [300400, 900500, 300400],
        "contribution": [5.0, 1.0, 3.0]})     # the supported link has the top contribution
    r = O.loop_contact_benchmark(tri, loops, gene_tss)
    # only G1's enhancer link is loop-supported (peak overlaps the distal anchor);
    # G2 shares the peak coords but its promoter isn't on any anchor → not supported.
    assert r["n_links"] == 3 and r["n_pos"] == 1
    assert r["aupr"] == 1.0                          # supported link ranks top by contribution
    # OR separation needs more links (small-sample control rounds); logic shown by AUPR + frac
    assert r["odds_ratio"] >= 1.0 and r["frac_supported"] > r["frac_control"]


def test_hichip_placseq_data_pending_graceful():
    import os
    import orthogonal_gold as O
    tri = pd.DataFrame({"gene": ["G1"], "peak": ["chr1:1-2"], "chrom": ["chr1"],
                        "ps": [1], "pe": [2], "contribution": [1.0]})
    # both scorers: if the BEDPE is registered+present they score (no data-pending note),
    # else they degrade gracefully to 'data-pending' (never crash).
    for fn, key, files in [(O.hichip_benchmark, "CD4 T", O.HICHIP_FILES),
                           (O.placseq_benchmark, "Microglia", O.PLACSEQ_FILES)]:
        r = fn(tri, key); rel = files.get(key)
        if rel and os.path.exists(f"{O.GS}/{rel}"):
            assert "data-pending" not in r.get("note", "")          # real scorer ran
        else:
            assert "data-pending" in r.get("note", "")
