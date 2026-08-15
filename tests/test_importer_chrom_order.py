"""Regression tests for the fragment importer's k-way merge chromosome ordering.

The merge heap is ordered by `(chrom_id, start)`. `chrom_id` used to come from a
hard-coded *lexicographic* table, which only agrees with 10x/CellRanger output.
On karyotypically-ordered files (`chr1, chr2, …, chr22` — ArchR, SHARE-seq,
HDMA) the readers desynchronised: one would advance to `chr10` (low fixed rank)
while another still held `chr9` (high fixed rank), the merge would close `chr9`,
and `chr9` came back after `chr22` with `chunk_idx` restarting at 0 →
``UNIQUE constraint failed: fragment_chunks.chrom, fragment_chunks.chunk_idx``.

Ranks are now minted at runtime in first-appearance order, so any order the files
*agree* on merges correctly. Files that disagree with each other cannot be merged
without buffering; those hit an explicit guard instead of a SQLite error.

Covered here:

1. karyotypic two-file merge completes, every chromosome written exactly once;
2. lexicographic two-file merge is **byte-identical** to the pre-fix binary
   (skipped unless a pre-fix binary is supplied — see PIASO_IMPORTER_PREFIX_BIN);
3. mutually inconsistent orders hit the guard, with an actionable message;
4. appending into a cytome that already holds fragments is refused;
5. a whitelist that matches nothing is a hard error, not a silent empty cytome.
"""
import os
import random
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

pysam = pytest.importorskip("pysam")

REPO = Path(__file__).resolve().parents[1]
# The freshly built binary; fall back to the shipped one.
CANDIDATES = [
    REPO / "target" / "release" / "cytome-import-fragments",
    REPO / "bin" / "cytome-import-fragments",
]
IMPORTER = next((p for p in CANDIDATES if p.exists()), None)
# Optional: a copy of the binary from before the fix, for the byte-identity check.
PREFIX_BIN = os.environ.get("PIASO_IMPORTER_PREFIX_BIN")

KARYOTYPIC = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
LEXICOGRAPHIC = sorted(KARYOTYPIC)
N_PER_CHROM = 300
N_BARCODES = 20

pytestmark = pytest.mark.skipif(
    IMPORTER is None,
    reason="cytome-import-fragments binary not built (cargo build --release)",
)


def _write_fragments(path: Path, chrom_order, barcodes, seed):
    """One bgzipped fragment file, chromosome blocks in `chrom_order`."""
    rng = random.Random(seed)
    rows = []
    for chrom in chrom_order:
        for start in sorted(rng.sample(range(1000, 500_000), N_PER_CHROM)):
            rows.append(f"{chrom}\t{start}\t{start + 120}\t{rng.choice(barcodes)}\t1")
    plain = path.with_suffix("")
    plain.write_text("\n".join(rows) + "\n")
    pysam.tabix_compress(str(plain), str(path), force=True)
    plain.unlink()


def _fixture(tmp_path, order_a, order_b, seed_a=0, seed_b=1):
    bcs_a = [f"AAA{i}" for i in range(N_BARCODES)]
    bcs_b = [f"BBB{i}" for i in range(N_BARCODES)]
    a, b = tmp_path / "A.tsv.gz", tmp_path / "B.tsv.gz"
    _write_fragments(a, order_a, bcs_a, seed_a)
    _write_fragments(b, order_b, bcs_b, seed_b)
    wl = tmp_path / "barcodes.txt"
    wl.write_text("\n".join([f"S1:{x}" for x in bcs_a] + [f"S2:{x}" for x in bcs_b]) + "\n")
    return a, b, wl


def _run(binary, *args):
    return subprocess.run([str(binary), *map(str, args)],
                          capture_output=True, text=True, timeout=600)


def _import_pair(binary, a, b, wl, out, extra=()):
    if Path(out).exists():
        shutil.rmtree(out) if Path(out).is_dir() else Path(out).unlink()
    return _run(binary, "--fragments", a, b, "--fragment-prefixes", "S1,S2",
                "--barcodes", wl, "--output", out, "--genome", "hg38",
                "--threads", "2", "--chunk-size", "500", *extra)


def _chunks(cytome):
    """(chrom, chunk_idx, n_fragments, min_start) for every chunk, in write order."""
    con = sqlite3.connect(str(cytome))
    try:
        return con.execute(
            "SELECT chrom, chunk_idx, n_fragments, min_start FROM fragment_chunks "
            "ORDER BY rowid").fetchall()
    finally:
        con.close()


def _blobs(cytome):
    con = sqlite3.connect(str(cytome))
    try:
        return con.execute(
            "SELECT chrom, chunk_idx, row_start, row_end, n_fragments, min_start, "
            "starts_blob, ends_blob, cell_idx_blob "
            "FROM fragment_chunks ORDER BY chrom, chunk_idx").fetchall()
    finally:
        con.close()


# ── 1. the bug ────────────────────────────────────────────────────────────────

def test_karyotypic_two_file_merge_completes(tmp_path):
    """Karyotypic files (HDMA, ArchR) merge without a chunk_idx collision."""
    a, b, wl = _fixture(tmp_path, KARYOTYPIC, KARYOTYPIC)
    out = tmp_path / "kary.cytome"
    r = _import_pair(IMPORTER, a, b, wl, out)
    assert r.returncode == 0, f"import failed:\n{r.stderr[-3000:]}"
    assert "UNIQUE constraint failed" not in r.stderr

    rows = _chunks(out)
    written = [c for c, _, _, _ in rows]
    # every chromosome appears as one contiguous run — i.e. never re-opened
    runs = [c for i, c in enumerate(written) if i == 0 or written[i - 1] != c]
    assert len(runs) == len(set(runs)), f"a chromosome was re-opened: {runs}"
    assert set(runs) == set(KARYOTYPIC)
    # and per chromosome the counter is 0..n-1 with ascending min_start
    for chrom in KARYOTYPIC:
        sub = [(i, m) for c, i, _, m in rows if c == chrom]
        assert [i for i, _ in sub] == list(range(len(sub)))
        starts = [m for _, m in sub]
        assert starts == sorted(starts), f"{chrom} chunks are not position-ordered"
    # nothing lost: 2 files x 24 chroms x 300
    assert sum(n for _, _, n, _ in rows) == 2 * len(KARYOTYPIC) * N_PER_CHROM


def test_lexicographic_still_works(tmp_path):
    """The path that always worked keeps working."""
    a, b, wl = _fixture(tmp_path, LEXICOGRAPHIC, LEXICOGRAPHIC)
    out = tmp_path / "lex.cytome"
    r = _import_pair(IMPORTER, a, b, wl, out)
    assert r.returncode == 0, f"import failed:\n{r.stderr[-3000:]}"
    rows = _chunks(out)
    assert sum(n for _, _, n, _ in rows) == 2 * len(LEXICOGRAPHIC) * N_PER_CHROM
    # chromosomes are visited in the files' own (lexicographic) order
    runs = [c for i, c in enumerate(r_[0] for r_ in rows)
            if i == 0 or rows[i - 1][0] != rows[i][0]]
    assert runs == LEXICOGRAPHIC


# ── 2. output invariance ──────────────────────────────────────────────────────

@pytest.mark.skipif(not PREFIX_BIN or not Path(PREFIX_BIN or "").exists(),
                    reason="set PIASO_IMPORTER_PREFIX_BIN to a pre-fix binary")
def test_lexicographic_output_byte_identical_to_prefix_binary(tmp_path):
    """Runtime ranks == the old fixed ranks for lexicographic input.

    `chrom_id` is in-memory only (every table stores `chrom TEXT`), so this must
    reproduce the pre-fix output exactly, blobs included.
    """
    a, b, wl = _fixture(tmp_path, LEXICOGRAPHIC, LEXICOGRAPHIC)
    old_out, new_out = tmp_path / "old.cytome", tmp_path / "new.cytome"
    r_old = _import_pair(PREFIX_BIN, a, b, wl, old_out)
    r_new = _import_pair(IMPORTER, a, b, wl, new_out)
    assert r_old.returncode == 0, r_old.stderr[-2000:]
    assert r_new.returncode == 0, r_new.stderr[-2000:]
    assert _blobs(old_out) == _blobs(new_out), "fragment_chunks differ from the pre-fix build"


# ── 3. the guard ──────────────────────────────────────────────────────────────

def test_inconsistent_orders_hit_the_guard(tmp_path):
    """Files that disagree get a named cause, not a SQLite constraint error."""
    a, b, wl = _fixture(tmp_path, KARYOTYPIC, LEXICOGRAPHIC)
    out = tmp_path / "mixed.cytome"
    r = _import_pair(IMPORTER, a, b, wl, out)
    assert r.returncode != 0, "an unmergeable pair imported silently"
    msg = r.stderr + r.stdout
    assert "disagree on chromosome order" in msg, msg[-3000:]
    assert "resort_fragments.sh" in msg
    assert "UNIQUE constraint failed" not in msg


# ── 4 + 5. the other two silent failures ──────────────────────────────────────

def test_append_into_populated_cytome_is_refused(tmp_path):
    """--cytome append restarts chunk_idx at 0; say so instead of colliding."""
    a, b, wl = _fixture(tmp_path, LEXICOGRAPHIC, LEXICOGRAPHIC)
    out = tmp_path / "append.cytome"
    r1 = _run(IMPORTER, "--fragments", a, "--fragment-prefixes", "S1",
              "--barcodes", wl, "--output", out, "--genome", "hg38", "--threads", "2")
    assert r1.returncode == 0, r1.stderr[-2000:]
    r2 = _run(IMPORTER, "--fragments", b, "--fragment-prefixes", "S2",
              "--cytome", out, "--genome", "hg38", "--threads", "2")
    assert r2.returncode != 0
    msg = r2.stderr + r2.stdout
    assert "already contains" in msg and "not supported" in msg, msg[-3000:]
    assert "UNIQUE constraint failed" not in msg


def test_zero_imported_fragments_is_an_error(tmp_path):
    """A whitelist that matches nothing must not exit 0 with an empty cytome."""
    a, _, _ = _fixture(tmp_path, LEXICOGRAPHIC, LEXICOGRAPHIC)
    # '#'-joined whitelist against the default ':' delimiter — the HDMA failure
    wl = tmp_path / "hash.txt"
    wl.write_text("\n".join(f"S1#AAA{i}" for i in range(N_BARCODES)) + "\n")
    out = tmp_path / "empty.cytome"
    r = _run(IMPORTER, "--fragments", a, "--fragment-prefixes", "S1",
             "--barcodes", wl, "--output", out, "--genome", "hg38", "--threads", "2")
    assert r.returncode != 0, "a 0-fragment import reported success"
    assert "imported 0" in (r.stderr + r.stdout)


def test_single_file_auto_discover_applies_the_prefix(tmp_path):
    """Auto-discover used to register BARE barcodes for a single file.

    The import loop then looked up '{prefix}{delim}{barcode}' and matched none of
    the cells it had just discovered — silently, before the imported==0 guard.
    """
    a, _, _ = _fixture(tmp_path, LEXICOGRAPHIC, LEXICOGRAPHIC)
    out = tmp_path / "discover.cytome"
    r = _run(IMPORTER, "--fragments", a, "--fragment-prefixes", "S1",
             "--prefix-delimiter", "#", "--min-fragments", "1",
             "--output", out, "--genome", "hg38", "--threads", "2")
    assert r.returncode == 0, r.stderr[-3000:]
    assert sum(n for _, _, n, _ in _chunks(out)) == len(LEXICOGRAPHIC) * N_PER_CHROM
    con = sqlite3.connect(str(out))
    try:
        ids = [x[0] for x in con.execute("SELECT barcode FROM cells LIMIT 5")]
    finally:
        con.close()
    assert all(x.startswith("S1#") for x in ids), ids


def test_matching_delimiter_imports_everything(tmp_path):
    """…and the same whitelist works once the delimiter matches."""
    a, _, _ = _fixture(tmp_path, LEXICOGRAPHIC, LEXICOGRAPHIC)
    wl = tmp_path / "hash.txt"
    wl.write_text("\n".join(f"S1#AAA{i}" for i in range(N_BARCODES)) + "\n")
    out = tmp_path / "hash.cytome"
    r = _run(IMPORTER, "--fragments", a, "--fragment-prefixes", "S1",
             "--barcodes", wl, "--prefix-delimiter", "#", "--output", out,
             "--genome", "hg38", "--threads", "2")
    assert r.returncode == 0, r.stderr[-2000:]
    assert sum(n for _, _, n, _ in _chunks(out)) == len(LEXICOGRAPHIC) * N_PER_CHROM


def test_whitelist_beats_auto_discover(tmp_path):
    """`--barcodes` must pin the cell set, even alongside --auto-discover.

    The pipeline's ATAC-only rule always passes --auto-discover; a dataset-level
    `barcodes:` is what restricts the import to a curated annotation. If the
    whitelist were ignored the cytome would silently hold every barcode above
    --min-fragments instead, and the label join would become an intersection.
    """
    a, _, _ = _fixture(tmp_path, LEXICOGRAPHIC, LEXICOGRAPHIC)
    wl = tmp_path / "subset.txt"
    keep = [f"S1#AAA{i}" for i in range(5)]          # 5 of the 20 barcodes
    wl.write_text("\n".join(keep) + "\n")
    out = tmp_path / "wl.cytome"
    r = _run(IMPORTER, "--fragments", a, "--fragment-prefixes", "S1",
             "--prefix-delimiter", "#", "--barcodes", wl, "--min-fragments", "1",
             "--output", out, "--genome", "hg38", "--threads", "2")
    assert r.returncode == 0, r.stderr[-3000:]
    con = sqlite3.connect(str(out))
    try:
        ids = sorted(x[0] for x in con.execute("SELECT barcode FROM cells"))
    finally:
        con.close()
    assert ids == sorted(keep), ids
