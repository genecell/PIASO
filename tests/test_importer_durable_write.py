"""How a fresh import reaches disk, and what it costs per cell.

Three changes are pinned here:

* every per-cell QC column is written in one transaction, so the values must
  be exactly what the three separate autocommit writers produced;
* a fresh import writes to `<output>.partial` and renames on success, so a
  failed one leaves nothing at the output path and a finished one leaves no
  side files;
* the finished store is in WAL, which is what every reader expects, even
  though the import itself ran with the journal in memory.
"""
import random
import sqlite3
import subprocess
from pathlib import Path

import pytest

pysam = pytest.importorskip("pysam")

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [REPO / "target" / "release" / "cytome-import-fragments",
              REPO / "bin" / "cytome-import-fragments"]
IMPORTER = next((p for p in CANDIDATES if p.exists()), None)
pytestmark = pytest.mark.skipif(IMPORTER is None, reason="importer binary not built")

TSS = [("chr1", 120_000), ("chr1", 180_000), ("chr1", 240_000)]


def _write(path, rows):
    plain = Path(str(path).replace(".gz", ""))
    plain.write_text("\n".join(rows) + "\n")
    pysam.tabix_compress(str(plain), str(path), force=True)
    plain.unlink()


@pytest.fixture
def fragments(tmp_path):
    """One library whose fragments straddle the TSS sites and every length bin."""
    rng = random.Random(11)
    rows, pos = [], 100_000
    for i in range(1200):
        L = rng.choice([50, 100, 146, 147, 200, 293, 294, 400, 1500])
        rows.append(f"chr1\t{pos}\t{pos + L}\tBC{i % 8}\t1")
        pos += rng.randint(30, 400)
    p = tmp_path / "F.tsv.gz"
    _write(p, rows)
    return p


@pytest.fixture
def tss_bed(tmp_path):
    p = tmp_path / "tss.bed"
    p.write_text("".join(f"{c}\t{s}\t{s + 1}\tT{i}\t0\t+\n"
                         for i, (c, s) in enumerate(TSS)))
    return p


def _import(frag, out, tss=None, extra=()):
    args = [str(IMPORTER), "--fragments", str(frag), "--output", str(out),
            "--genome", "mm10", "--min-fragments", "1", "--threads", "2",
            "--chunk-size", "500", *map(str, extra)]
    if tss is not None:
        args += ["--tss-bed", str(tss)]
    return subprocess.run(args, capture_output=True, text=True, timeout=600)


def _cells(path):
    con = sqlite3.connect(str(path))
    cols = [r[1] for r in con.execute("PRAGMA table_info(cells)")]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM cells ORDER BY cell_idx").fetchall()
    con.close()
    return cols, rows


def test_every_qc_column_survives_the_single_transaction(fragments, tss_bed, tmp_path):
    """The merged writer must produce the values the three writers did."""
    out = tmp_path / "qc.cytome"
    r = _import(fragments, out, tss=tss_bed)
    assert r.returncode == 0, r.stderr[-3000:]
    cols, rows = _cells(out)
    for c in ["n_fragments", "tss_score", "tss_core_count", "tss_flank_count",
              "nucleosome_signal", "nucleosome_free_count", "mononucleosomal_count"]:
        assert c in cols, f"{c} missing from {cols}"
    assert rows, "no cells written"
    i_frag = cols.index("n_fragments")
    assert sum(r[i_frag] for r in rows) == 1200
    # TSS is the arithmetic the separate writer used: core / mean(flanks).
    i_score, i_core, i_flank = (cols.index(c) for c in
                                ["tss_score", "tss_core_count", "tss_flank_count"])
    for row in rows:
        if row[i_core] is None:
            continue
        if row[i_flank] > 0:
            assert row[i_score] == pytest.approx(row[i_core] / row[i_flank])
        else:
            assert row[i_score] == 0.0
    # Nucleosome signal is mono / nfr, NULL when there is no nucleosome-free read.
    i_sig, i_nfr, i_mono = (cols.index(c) for c in
                            ["nucleosome_signal", "nucleosome_free_count",
                             "mononucleosomal_count"])
    for row in rows:
        if row[i_nfr] is None:
            continue
        if row[i_nfr] > 0:
            assert row[i_sig] == pytest.approx(row[i_mono] / row[i_nfr])
        else:
            assert row[i_sig] is None


def test_no_tss_bed_means_no_tss_columns(fragments, tmp_path):
    """The merged writer only adds the columns its accumulators produced."""
    out = tmp_path / "notss.cytome"
    r = _import(fragments, out)
    assert r.returncode == 0, r.stderr[-3000:]
    cols, _ = _cells(out)
    assert "tss_score" not in cols
    assert "nucleosome_signal" in cols


def test_a_finished_import_leaves_one_file_in_wal(fragments, tss_bed, tmp_path):
    out = tmp_path / "pub.cytome"
    r = _import(fragments, out, tss=tss_bed)
    assert r.returncode == 0, r.stderr[-3000:]
    assert out.exists()
    assert not Path(str(out) + ".partial").exists()
    assert not Path(str(out) + ".partial-wal").exists()
    con = sqlite3.connect(str(out))
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    assert mode.lower() == "wal", mode


def test_a_failed_import_leaves_nothing_at_the_output_path(fragments, tmp_path):
    """The guard removes the partial store, so no half-written cytome appears.

    A whitelist that shares no barcode with the fragment files trips the
    barcode-join guard, which fires after the store has been created and the
    merge has run -- exactly the case where the old behaviour left a
    plausible-looking file at the output path.
    """
    bc = tmp_path / "whitelist.txt"
    bc.write_text("".join(f"NOMATCH{i}\n" for i in range(4)))
    out = tmp_path / "bad.cytome"
    r = subprocess.run(
        [str(IMPORTER), "--fragments", str(fragments), "--barcodes", str(bc),
         "--output", str(out), "--genome", "mm10", "--threads", "2"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode != 0, r.stdout[-2000:]
    assert not out.exists(), "a failed import left a file where a cytome is expected"
    assert not Path(str(out) + ".partial").exists()


def test_an_existing_output_is_replaced_not_merged(fragments, tmp_path):
    """Re-importing over a path publishes the new store, not a mixture."""
    out = tmp_path / "twice.cytome"
    assert _import(fragments, out).returncode == 0
    first = out.stat().st_size
    assert _import(fragments, out).returncode == 0
    _, rows = _cells(out)
    assert len(rows) == 8
    assert out.stat().st_size == pytest.approx(first, rel=0.5)


def test_many_libraries_still_merge_in_parallel(tmp_path):
    """The reader cap follows the descriptor limit, not a fixed 128.

    The old cap was `128 / n_files`, so 65 or more libraries merged one
    chromosome at a time however `--merge-parallelism` was set -- the largest
    datasets ran the slowest per fragment. Seventy small files fit in any
    normal descriptor budget and must keep the parallelism they asked for.
    """
    paths = []
    for f in range(70):
        rows = [f"chr1\t{100_000 + i * 900}\t{100_000 + i * 900 + 120}\tBC{i % 3}\t1"
                for i in range(40)]
        p = tmp_path / f"M{f:02d}.tsv.gz"
        _write(p, rows)
        paths.append(str(p))
    out = tmp_path / "many.cytome"
    r = subprocess.run(
        [str(IMPORTER), "--fragments", *paths,
         "--fragment-prefixes", ",".join(f"S{f:02d}" for f in range(70)),
         "--output", str(out), "--genome", "mm10", "--min-fragments", "1",
         "--threads", "2", "--merge-parallelism", "4"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(l for l in r.stderr.splitlines() if "Merging" in l and "at a time" in l)
    at_a_time = int(line.split("chromosomes,")[1].split("at a time")[0].strip())
    assert at_a_time > 1, line
    _, rows = _cells(out)
    assert len(rows) == 3 * 70
