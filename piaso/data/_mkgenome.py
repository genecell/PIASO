"""Build PIASO's genome BED files from a cellranger reference.

PIASO reads four small files per genome: chromosome sizes, gene bodies,
transcript start sites, and promoters. For the two genomes it ships these
were derived from GENCODE. For any other species the natural source is the
cellranger (or cellranger-arc) reference that the data was mapped against,
because that reference carries whatever chromosome renaming was done when it
was built. Deriving the BEDs from the original annotation instead risks a
silent mismatch between the names in the BED files and the names in the
fragments -- which does not raise, it just finds nothing.

The layout this reads is common to cellranger and cellranger-arc::

    <ref>/fasta/genome.fa.fai     chromosome sizes
    <ref>/genes/genes.gtf.gz      the filtered annotation

The promoter window matches what the shipped mm10 and hg38 files use, which
was measured rather than assumed: 400 bp upstream and 200 bp downstream of
each transcript start, strand-aware, and overlapping promoters of one gene
merged.
"""
from __future__ import annotations

import gzip
import os
import re
from typing import Dict, List, Optional, Tuple

__all__ = ["makeGenomeFiles"]

#: Upstream / downstream promoter extent, in bp, matching the shipped genomes.
PROMOTER_UPSTREAM = 400
PROMOTER_DOWNSTREAM = 200

_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')


def _attrs(field: str) -> Dict[str, str]:
    return {k: v for k, v in _ATTR.findall(field)}


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _find(reference_dir: str, *candidates: str) -> Optional[str]:
    for rel in candidates:
        p = os.path.join(reference_dir, rel)
        if os.path.exists(p):
            return p
    return None


def _chrom_key(name: str):
    """Karyotypic order: chr1..chrN, then chrX/Y/M, then everything else.

    The shipped mm10 and hg38 BEDs are in this order, and a file that opens
    with an unplaced scaffold reads as though something went wrong even when
    nothing did. Nothing depends on it -- every reader builds a per-chromosome
    index -- so this is legibility, not correctness.
    """
    core = name[3:] if name.startswith("chr") else name
    if core.isdigit():
        return (0, int(core), "")
    if core in ("X", "Y", "M", "MT"):
        return (1, {"X": 0, "Y": 1, "M": 2, "MT": 2}[core], "")
    return (2, 0, name)


def _merge(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[List[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def makeGenomeFiles(
    reference_dir: str,
    genome: str,
    outdir: str,
    chromosomes: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Write ``<genome>.chrom.sizes`` and three BED files from a reference.

    Parameters
    ----------
    reference_dir : str
        A cellranger or cellranger-arc reference directory, holding
        ``fasta/genome.fa.fai`` and ``genes/genes.gtf.gz``.
    genome : str
        Name to write the files under, e.g. ``'Mmul_10'``. Use the name of
        the reference folder, so the provenance of the files is visible.
    outdir : str
        Directory to write into, normally ``PIASO-data/<genome>/``.
    chromosomes : list of str, optional
        Restrict to these sequence names. The default keeps everything,
        including unplaced scaffolds, which matches what the shipped genomes
        do and keeps the sizes file a faithful description of the reference.
    verbose : bool, default True
        Report what was written.

    Returns
    -------
    dict
        ``{kind: path}`` for the four files written.

    Notes
    -----
    No cCRE file is produced: SCREEN publishes candidate cis-regulatory
    elements for human and mouse only, so ``splitby='ccre'`` is unavailable
    for other species. That is a gap in the source data, not in this
    function.
    """
    fai = _find(reference_dir, "fasta/genome.fa.fai", "fasta/genome.fa.fai.gz")
    gtf = _find(reference_dir, "genes/genes.gtf.gz", "genes/genes.gtf")
    if fai is None or gtf is None:
        raise FileNotFoundError(
            f"{reference_dir!r} does not look like a cellranger reference: "
            f"expected fasta/genome.fa.fai and genes/genes.gtf(.gz). "
            f"Found fai={fai!r}, gtf={gtf!r}."
        )
    os.makedirs(outdir, exist_ok=True)
    keep = set(chromosomes) if chromosomes else None

    # --- chromosome sizes -------------------------------------------------
    sizes_path = os.path.join(outdir, f"{genome}.chrom.sizes")
    n_chrom = 0
    rows = []
    with _open(fai) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2 or (keep and f[0] not in keep):
                continue
            rows.append((f[0], f[1]))
    rows.sort(key=lambda r: _chrom_key(r[0]))
    with open(sizes_path, "w") as out:
        for name, length in rows:
            out.write(f"{name}\t{length}\n")
            n_chrom += 1

    # --- one pass over the annotation ------------------------------------
    genes: Dict[Tuple[str, str], List] = {}
    tss_rows: List[Tuple] = []
    with _open(gtf) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            chrom, feature, start, end, strand = f[0], f[2], int(f[3]), int(f[4]), f[6]
            if keep and chrom not in keep:
                continue
            if feature not in ("gene", "transcript"):
                continue
            a = _attrs(f[8])
            name = a.get("gene") or a.get("gene_name") or a.get("gene_id") or ""
            if not name:
                continue
            # Gene bodies come from BOTH `gene` and `transcript` features,
            # unioned. A `gene` line is not reliably present: this macaque
            # annotation carries 2,799 of them against 34,204 distinct genes
            # named on its transcripts, so trusting `gene` alone would have
            # shipped a file with 8% of the genes in it. Aggregating
            # transcripts is what cellranger does for its own
            # regions/transcripts.bed, and it degrades gracefully.
            # BED is half-open and 0-based; GTF is closed and 1-based.
            key = (chrom, name)
            span = genes.get(key)
            if span is None:
                genes[key] = [start - 1, end, strand]
            else:
                span[0] = min(span[0], start - 1)
                span[1] = max(span[1], end)
            if feature == "transcript":
                tss = start - 1 if strand != "-" else end - 1
                tss_rows.append((
                    chrom, tss, name, strand,
                    a.get("transcript_biotype") or a.get("gene_biotype") or "",
                    a.get("db_xref") or a.get("gene_id") or "",
                    a.get("transcript_id") or "",
                ))

    # --- gene bodies ------------------------------------------------------
    genes_path = os.path.join(outdir, f"{genome}_genes.bed")
    with open(genes_path, "w") as out:
        for (chrom, name), (s, e, strand) in sorted(
                genes.items(), key=lambda kv: (_chrom_key(kv[0][0]), kv[1][0])):
            out.write(f"{chrom}\t{s}\t{e}\t{name}\t.\t{strand}\n")

    # --- transcript start sites ------------------------------------------
    tss_rows.sort(key=lambda r: (_chrom_key(r[0]), r[1]))
    tss_path = os.path.join(outdir, f"{genome}_transcript_tss.bed")
    with open(tss_path, "w") as out:
        for chrom, tss, name, strand, biotype, gene_id, tx_id in tss_rows:
            out.write(f"{chrom}\t{tss}\t{tss + 1}\t{name}\t.\t{strand}\t"
                      f"{biotype}\t{gene_id}\t{tx_id}\n")

    # --- promoters --------------------------------------------------------
    # Per gene, because the shipped files merge a gene's overlapping promoters
    # into one interval but keep two genes' promoters separate even where they
    # overlap.
    by_gene: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for chrom, tss, name, strand, _b, _g, _t in tss_rows:
        if strand == "-":
            s, e = tss - PROMOTER_DOWNSTREAM, tss + PROMOTER_UPSTREAM + 1
        else:
            s, e = tss - PROMOTER_UPSTREAM, tss + PROMOTER_DOWNSTREAM + 1
        by_gene.setdefault((chrom, name), []).append((max(0, s), e))
    prom_path = os.path.join(outdir, f"{genome}_promoterSet.bed")
    prom_rows = []
    for (chrom, name), iv in by_gene.items():
        for s, e in _merge(iv):
            prom_rows.append((chrom, s, e, name))
    prom_rows.sort(key=lambda r: (_chrom_key(r[0]), r[1]))
    with open(prom_path, "w") as out:
        for chrom, s, e, name in prom_rows:
            out.write(f"{chrom}\t{s}\t{e}\t{name}\n")

    if verbose:
        print(f"  {genome}: {n_chrom:,} sequences, {len(genes):,} genes, "
              f"{len(tss_rows):,} transcript TSSes, {len(prom_rows):,} promoters")
        print(f"  written to {outdir}")
    return {"chrom_sizes": sizes_path, "gene_boundary": genes_path,
            "tss_bed": tss_path, "promoter": prom_path}
