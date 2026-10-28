"""Which TFs a motif is given to (code audit of 2026-10-27, part 3; review I1 finding 12 and its minors).

- a JASPAR heterodimer ``A::B`` is given to each partner (taken as one symbol it matched no gene);
- ``load_cistarget_motifs(max_per_tf=k)`` keeps each TF's k motifs with the best motif-similarity q-value
  (it kept the first k in the table's order).
"""
from __future__ import annotations

import numpy as np

from piaso.data._motifs import build_tf_motif_map, load_cistarget_motifs
from piaso.data._pwm import PWM


def _pwm(motif_id, tf_name):
    return PWM(motif_id, tf_name, np.full((4, 6), 0.25, dtype=np.float32))


def test_heterodimer_is_given_to_each_partner():
    pwms = [_pwm("MA0099.3", "FOS::JUN"), _pwm("MA0142.1", "Pou5f1::Sox2"), _pwm("MA0004.1", "Arnt"),
            _pwm("MA0259.1", "ARNT::HIF1A")]
    out = build_tf_motif_map(pwms, gene_universe=["Fos", "Jun", "Pou5f1", "Sox2", "Arnt", "Actb"])
    assert sorted(out) == ["Arnt", "Fos", "Jun", "Pou5f1", "Sox2"]
    assert [p.motif_id for p in out["Fos"]] == ["MA0099.3"] and [p.motif_id for p in out["Jun"]] == ["MA0099.3"]
    assert out["Fos"][0].tf_name == "Fos" and out["Fos"][0].meta["complex"] == "FOS::JUN"
    # Arnt keeps its own motif and gains the complex's; HIF1A is not in the universe
    assert [p.motif_id for p in out["Arnt"]] == ["MA0004.1", "MA0259.1"]
    assert "complex" not in out["Arnt"][0].meta
    # the input PWMs are not modified
    assert pwms[0].tf_name == "FOS::JUN" and "complex" not in pwms[0].meta


def test_heterodimer_partners_pass_the_tf_list_one_by_one():
    out = build_tf_motif_map([_pwm("MA0099.3", "FOS::JUN")], tf_list={"JUN"})
    assert list(out) == ["JUN"] and out["JUN"][0].meta["complex"] == "FOS::JUN"
    # no universe: each partner under its own name
    out = build_tf_motif_map([_pwm("MA0099.3", "FOS::JUN")])
    assert sorted(out) == ["FOS", "JUN"]


def test_a_complex_is_given_once_to_a_gene_both_partners_name():
    out = build_tf_motif_map([_pwm("M1", "STAT3::Stat3")], gene_universe=["Stat3"])
    assert list(out) == ["Stat3"] and len(out["Stat3"]) == 1


def _cistarget(tmp_path, rows):
    cb = tmp_path / "cb"
    cb.mkdir()
    for mid in {r[0] for r in rows}:
        (cb / f"{mid}.cb").write_text(f">{mid}\n10 0 0 0\n0 10 0 0\n0 0 10 0\n0 0 0 10\n")
    tbl = tmp_path / "m2tf.tbl"
    tbl.write_text("#motif_id\tgene_name\tmotif_similarity_qvalue\tdescription\n"
                   + "".join(f"{m}\t{tf}\t{q}\t{d}\n" for m, tf, q, d in rows))
    return str(cb), str(tbl)


def test_max_per_tf_keeps_the_best_q_values(tmp_path):
    cb, tbl = _cistarget(tmp_path, [
        ("bergman__m1", "Sox9", 0.04, "motif similar to X"),     # first in the table, worst q
        ("jaspar__m2", "Sox9", 0.0, "gene is directly annotated"),
        ("homer__m3", "Sox9", 0.001, "motif similar to Y"),
        ("cisbp__m4", "Pax6", 0.2, "motif similar to Z"),
        ("cisbp__m5", "Pax6", 0.2, "motif similar to Z"),        # a tie: the table's order
    ])
    one = load_cistarget_motifs(cb, tbl, max_per_tf=1)
    assert sorted((p.tf_name, p.motif_id) for p in one) == [("PAX6", "cisbp__m4"), ("SOX9", "jaspar__m2")]
    two = load_cistarget_motifs(cb, tbl, max_per_tf=2)
    assert [p.motif_id for p in two if p.tf_name == "SOX9"] == ["jaspar__m2", "homer__m3"]
    assert [p.motif_id for p in two if p.tf_name == "PAX6"] == ["cisbp__m4", "cisbp__m5"]
    assert all(p.meta["motif_similarity_qvalue"] in (0.0, 0.001, 0.2) for p in two)
    # without a cap: every motif, in the table's order
    assert [p.motif_id for p in load_cistarget_motifs(cb, tbl)] == [
        "bergman__m1", "jaspar__m2", "homer__m3", "cisbp__m4", "cisbp__m5"]
