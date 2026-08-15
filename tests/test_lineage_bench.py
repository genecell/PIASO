"""Unit tests for the pure lineage-assignment helpers (no data/gold needed)."""
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


def test_rss_assign_and_byct_and_links():
    import lineage_bench as L
    lineage_map = {"CD14 monocyte": "Mono/DC", "Naive B cell": "B/Plasma",
                   "Naive CD4 T cell": "CD4 T", "NotMapped": None}
    # 3 regulons × 4 cell types RSS (rows=regulons, cols=celltypes)
    spec = {"regulons": ["SPI1", "PAX5", "TCF7"],
            "celltypes": ["CD14 monocyte", "Naive B cell", "Naive CD4 T cell", "NotMapped"],
            "matrix": [[5.0, 0.1, 0.1, 9.0],    # SPI1 top in mono (+NotMapped, dropped)
                       [0.1, 5.0, 0.1, 0.1],    # PAX5 top in B
                       [0.1, 0.1, 5.0, 0.1]]}   # TCF7 top in CD4
    lin = L.rss_assign(spec, lineage_map, top_k=1)
    assert lin["Mono/DC"] == {"SPI1"} and lin["B/Plasma"] == {"PAX5"} and lin["CD4 T"] == {"TCF7"}
    assert "NotMapped" not in lin and None not in lin            # unmapped cts dropped

    rbc = {"CD14 monocyte": {"SPI1": ["A", "B"]}, "CD16 monocyte": {"SPI1": ["B", "C"]},
           "Naive B cell": {"PAX5": ["D"]}, "NotMapped": {"X": ["Z"]}}
    lmap2 = {**lineage_map, "CD16 monocyte": "Mono/DC"}
    by = L.byct_assign(rbc, lmap2)
    assert by["Mono/DC"]["SPI1"] == {"A", "B", "C"}              # union over the two monos
    assert by["B/Plasma"]["PAX5"] == {"D"} and "Mono/DC" in by and "NotMapped" not in str(by.keys())

    # lineage_links_from_tfs restricts + de-dups + adds coords
    tri = pd.DataFrame({"TF": ["SPI1", "SPI1", "PAX5"],
                        "gene": ["A", "A", "D"],
                        "peak": ["chr1:100-200", "chr1:100-200", "chr2:5-9"],
                        "contribution": [1.0, 2.0, 3.0]})
    lk = L.lineage_links_from_tfs(tri, {"SPI1"})
    assert list(lk["gene"]) == ["A"] and lk.iloc[0]["contribution"] == 2.0    # max, de-duped
    assert lk.iloc[0]["chrom"] == "chr1" and int(lk.iloc[0]["ps"]) == 100
    ed = L.lineage_edges_from_tfs(tri, {"SPI1", "PAX5"})
    assert set(zip(ed.TF, ed.gene)) == {("SPI1", "A"), ("PAX5", "D")}


def test_bmmc_lineage_map_covers_all():
    import celltype_mapping as M
    bmmc = ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "CD8+ T", "NK", "CD4+ T naive",
            "Naive CD20+ B", "Normoblast", "B1 B", "G/M prog", "ILC", "pDC", "Transitional B",
            "ID2-hi myeloid prog", "Proerythroblast", "CD16+ Mono", "MK/E prog", "Plasma cell",
            "Lymph prog", "HSC", "cDC2"]
    assert all(c in M.BMMC_LINEAGE for c in bmmc), [c for c in bmmc if c not in M.BMMC_LINEAGE]
    assert set(M.BMMC_LINEAGE.values()) <= {"Mono/DC", "CD4 T", "CD8/other T", "B/Plasma",
                                            "NK", "Ery", "Prog"}
    assert M.lineage_map("bmmc") is M.BMMC_LINEAGE and M.lineage_map("san2") is M.SAN2_LINEAGE
