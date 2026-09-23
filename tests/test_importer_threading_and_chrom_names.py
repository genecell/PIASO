"""The multi-file importer's use of the thread budget, and the naming guard.

Two passes read the fragment files. Pass 1 counts barcodes, once per file; pass
2 merges the files in position order. Both were leaving most of the machine
idle: pass 1 walked the files in a plain ``for`` loop, and every merge reader
was opened with a hard-coded single decompression thread regardless of
``--threads``.

Both now share the budget. The property that matters is that sharing it changes
nothing about the result, so the tests here import the same inputs at one thread
and at many and compare the cytomes byte for byte -- including ``cell_idx``,
which is assigned by descending fragment count and so is the thing most exposed
to a change in the order files are counted.

The third test covers a separate hole the threading work sat next to: the
genome-mismatch guard divided by the number of fragments on a *known*
chromosome, so a file whose chromosomes are spelled ``1`` rather than ``chr1``
had nothing to divide by, skipped the guard entirely, and imported an all-zero
matrix without a word.
"""
import random
import sqlite3
import subprocess
from pathlib import Path

import pytest

pysam = pytest.importorskip("pysam")

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [
    REPO / "target" / "release" / "cytome-import-fragments",
    REPO / "bin" / "cytome-import-fragments",
]
IMPORTER = next((p for p in CANDIDATES if p.exists()), None)

pytestmark = pytest.mark.skipif(
    IMPORTER is None,
    reason="cytome-import-fragments binary not built (cargo build --release)",
)

CHROMS = ["chr1", "chr2", "chr3"]
N_FILES = 4


def _write_fragments(path, barcodes, seed, chroms=CHROMS):
    """One bgzipped fragment file. Barcode frequencies are deliberately uneven
    (and deliberately tied for some pairs) so that cell_idx ordering is a real
    test rather than an accident of a uniform draw."""
    rng = random.Random(seed)
    weights = [1 + (i % 5) for i in range(len(barcodes))]
    rows = []
    for chrom in chroms:
        for start in sorted(rng.sample(range(1000, 400_000), 400)):
            bc = rng.choices(barcodes, weights=weights)[0]
            rows.append(f"{chrom}\t{start}\t{start + 120}\t{bc}\t1")
    plain = Path(str(path).replace(".gz", ""))
    plain.write_text("\n".join(rows) + "\n")
    pysam.tabix_compress(str(plain), str(path), force=True)
    plain.unlink()


def _fixture(tmp_path, n_files=N_FILES, chroms=CHROMS):
    paths, prefixes = [], []
    for f in range(n_files):
        bcs = [f"{chr(65 + f)}{i:04d}" for i in range(30)]
        p = tmp_path / f"F{f}.tsv.gz"
        _write_fragments(p, bcs, seed=f, chroms=chroms)
        paths.append(p)
        prefixes.append(f"S{f}")
    return paths, ",".join(prefixes)


def _run(*args):
    return subprocess.run([str(IMPORTER), *map(str, args)],
                          capture_output=True, text=True, timeout=900)


def _import(paths, prefixes, out, threads, extra=()):
    """Auto-discover mode: no --barcodes, so pass 1 runs."""
    return subprocess.run(
        [str(IMPORTER), "--fragments", *map(str, paths),
         "--fragment-prefixes", prefixes, "--output", str(out),
         "--genome", "hg38", "--layout", "csr", "--min-fragments", "5",
         "--threads", str(threads), "--chunk-size", "500", *map(str, extra)],
        capture_output=True, text=True, timeout=900)


def _cells(cytome):
    con = sqlite3.connect(str(cytome))
    try:
        return con.execute(
            "SELECT cell_idx, barcode, n_fragments FROM cells ORDER BY cell_idx"
        ).fetchall()
    finally:
        con.close()


def _chunk_blobs(cytome):
    con = sqlite3.connect(str(cytome))
    try:
        return con.execute(
            "SELECT chrom, chunk_idx, n_fragments, starts_blob, ends_blob, "
            "cell_idx_blob FROM fragment_chunks ORDER BY chrom, chunk_idx"
        ).fetchall()
    finally:
        con.close()


# ── 1. the budget is shared, and sharing it changes nothing ───────────────────

def test_thread_count_does_not_change_the_cytome(tmp_path):
    """One thread and many must produce the same cells and the same fragments.

    cell_idx is assigned by descending fragment count, so if concurrent counting
    perturbed either the counts or the tie order, this is where it would show.
    """
    paths, prefixes = _fixture(tmp_path)
    serial, parallel = tmp_path / "serial.cytome", tmp_path / "parallel.cytome"

    r1 = _import(paths, prefixes, serial, threads=1)
    assert r1.returncode == 0, f"serial import failed:\n{r1.stderr[-3000:]}"
    r8 = _import(paths, prefixes, parallel, threads=8)
    assert r8.returncode == 0, f"parallel import failed:\n{r8.stderr[-3000:]}"

    cells_serial, cells_parallel = _cells(serial), _cells(parallel)
    assert cells_serial, "no cells were imported -- the fixture is not exercising anything"
    assert cells_serial == cells_parallel

    assert _chunk_blobs(serial) == _chunk_blobs(parallel)


def test_pass_one_scans_files_concurrently(tmp_path):
    """With threads to spare, pass 1 should take more than one file at a time."""
    paths, prefixes = _fixture(tmp_path)
    r = _import(paths, prefixes, tmp_path / "p.cytome", threads=8)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next((l for l in r.stderr.splitlines() if l.startswith("Pass 1: Counting")), None)
    assert line is not None, f"pass 1 banner missing:\n{r.stderr[-2000:]}"
    assert f"{N_FILES} at a time" in line, line


def test_pass_one_respects_a_small_budget(tmp_path):
    """--threads 1 must still mean one file at a time, not four."""
    paths, prefixes = _fixture(tmp_path)
    r = _import(paths, prefixes, tmp_path / "q.cytome", threads=1)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(l for l in r.stderr.splitlines() if l.startswith("Pass 1: Counting"))
    assert "1 at a time" in line, line


def test_merge_readers_open_with_one_decoder_and_a_bounded_batch(tmp_path):
    """The merge's parallelism is the files being merged, not decoder threads:
    each reader's consumer is the merge loop, one line at a time, so extra
    inflate workers were measured at zero benefit and are gone. The batch
    buffer is a budget divided by the number of readers, so a 91-file import
    holds the same total as a 4-file one."""
    paths, prefixes = _fixture(tmp_path)
    r = _import(paths, prefixes, tmp_path / "m.cytome", threads=8)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next((l for l in r.stderr.splitlines() if l.strip().startswith("Merging")), None)
    assert line is not None, f"merge banner missing:\n{r.stderr[-2000:]}"
    assert "entries per batch" in line, line
    assert "decompression thread" not in r.stderr


# ── 2. the naming guard ───────────────────────────────────────────────────────

def test_chromosome_naming_mismatch_is_refused(tmp_path):
    """`1` against a `chr1` grid used to import an all-zero matrix, silently."""
    paths, prefixes = _fixture(tmp_path, n_files=2, chroms=["1", "2", "3"])
    r = _import(paths, prefixes, tmp_path / "bad.cytome", threads=4)
    assert r.returncode != 0, (
        "an import where no fragment is on a known chromosome must fail:\n"
        + r.stderr[-3000:])
    assert "recognises" in r.stderr, r.stderr[-2000:]
    # the message has to name what it saw, and point at the likely cause,
    # or the user is sent back to the barcode whitelist for a second time
    assert "1, 2, 3" in r.stderr, r.stderr[-2000:]
    assert "chr" in r.stderr


def test_correctly_named_chromosomes_still_import(tmp_path):
    """The guard must not fire on ordinary data (it divides by zero either side)."""
    paths, prefixes = _fixture(tmp_path, n_files=2)
    out = tmp_path / "good.cytome"
    r = _import(paths, prefixes, out, threads=4)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "recognises, so nothing" not in r.stderr
    assert len(_cells(out)) > 0


# ── 3. chromosome-parallel merge ─────────────────────────────────────────────

def _hashes(cytome):
    import hashlib
    con = sqlite3.connect(str(cytome))
    def h(q):
        m = hashlib.sha256()
        for r in con.execute(q):
            m.update(repr(r).encode())
        return m.hexdigest()
    out = (h("SELECT cell_idx, barcode, n_fragments, nucleosome_free_count, mononucleosomal_count FROM cells ORDER BY cell_idx"),
           h("SELECT chrom, chunk_idx, n_fragments, starts_blob, ends_blob, cell_idx_blob FROM fragment_chunks ORDER BY chrom, chunk_idx"),
           h("SELECT chunk_idx, data_blob, indices_blob, indptr_blob FROM matrix_chunks WHERE matrix_name='tiles_counts' ORDER BY chunk_idx"),
           h("SELECT chrom, n_fragments, min_start, max_end FROM fragment_meta ORDER BY chrom"))
    con.close()
    return out


def test_merge_parallelism_does_not_change_the_cytome(tmp_path):
    """Chromosomes merged one at a time and four at a time must hash the same:
    cells, fragment chunks, the tile matrix and the per-chromosome metadata.
    Every input to the output is fixed before the merge starts, and the heap
    order is total, so this holds by construction -- and is pinned anyway."""
    paths, prefixes = _fixture(tmp_path, chroms=["chr1", "chr2", "chr3", "chr4", "chr5"])
    one, four = tmp_path / "t1.cytome", tmp_path / "t4.cytome"
    r1 = _import(paths, prefixes, one, threads=4, extra=["--merge-parallelism", "1"])
    assert r1.returncode == 0, r1.stderr[-3000:]
    r4 = _import(paths, prefixes, four, threads=4, extra=["--merge-parallelism", "4"])
    assert r4.returncode == 0, r4.stderr[-3000:]
    assert _hashes(one) == _hashes(four)


def test_offsets_come_from_the_index_when_pass_one_did_not_run(tmp_path):
    """A whitelist import has no pass 1. With a .tbi beside the file the
    offsets come from it; without one, from an index pass. Either way the
    cytome is the one auto-discovery produces."""
    paths, prefixes = _fixture(tmp_path, n_files=2)
    # auto-discover: offsets recorded by pass 1
    auto = tmp_path / "auto.cytome"
    r = _import(paths, prefixes, auto, threads=2)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "2 from pass 1" in r.stderr
    con = sqlite3.connect(str(auto))
    wl = tmp_path / "wl.txt"
    wl.write_text("\n".join(r[0] for r in con.execute("SELECT barcode FROM cells ORDER BY cell_idx")) + "\n")
    con.close()

    def _whitelist(out):
        return _run("--fragments", *map(str, paths), "--fragment-prefixes", prefixes,
                    "--barcodes", wl, "--output", out, "--genome", "hg38", "--layout", "csr",
                    "--threads", "2", "--chunk-size", "500")

    # no index: an index pass
    scan = tmp_path / "scan.cytome"
    r = _whitelist(scan)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "2 from an index pass" in r.stderr
    assert _hashes(scan) == _hashes(auto)

    # with tabix indexes: read from them
    for p in paths:
        pysam.tabix_index(str(p), preset="bed", force=True)
    tbi = tmp_path / "tbi.cytome"
    r = _whitelist(tbi)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "2 from .tbi" in r.stderr
    assert _hashes(tbi) == _hashes(auto)


def test_a_stale_index_is_refused(tmp_path):
    """An offset from a .tbi that no longer matches the file is caught on the
    first line rather than trusted."""
    paths, prefixes = _fixture(tmp_path, n_files=1)
    p = paths[0]
    pysam.tabix_index(str(p), preset="bed", force=True)
    # rewrite the file with a different chromosome layout, keep the old index
    rng = random.Random(3)
    rows = [f"chr9\t{s}\t{s + 120}\tA0000\t1" for s in sorted(rng.sample(range(1000, 300_000), 300))]
    rows += [f"chr1\t{s}\t{s + 120}\tA0000\t1" for s in sorted(rng.sample(range(1000, 300_000), 300))]
    plain = Path(str(p).replace(".gz", ""))
    plain.write_text("\n".join(rows) + "\n")
    pysam.tabix_compress(str(plain), str(p), force=True)
    plain.unlink()
    wl = tmp_path / "wl.txt"; wl.write_text("S0:A0000\n")
    r = _run("--fragments", p, "--fragment-prefixes", "S0", "--barcodes", wl,
             "--output", tmp_path / "stale.cytome", "--genome", "hg38", "--layout", "csr",
             "--threads", "2", "--chunk-size", "500")
    assert r.returncode != 0
    assert "stale" in r.stderr, r.stderr[-2000:]
