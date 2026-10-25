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

The annotation is read by cytome's GTF reader (``cytome.read_gtf``), the one
behind ``cytome.import_gtf``, so a genome built here and a store's imported
annotation see the same genes: one per ``gene_id``, from its ``gene`` line or
spanning its transcripts. Genes used to be keyed by name, which merged every
gene sharing a name on one chromosome into a single span: in the Cell Ranger
ARC Mmul_10 reference the copies of TRNAN-GUU became one 118.6 Mb "gene"
across chr1, and 19 such spans exceeded 10 Mb.
"""
from __future__ import annotations

import gzip
import os
from typing import Dict, List, Optional, Tuple

__all__ = ["makeGenomeFiles"]

#: Upstream / downstream promoter extent, in bp, matching the shipped genomes.
PROMOTER_UPSTREAM = 400
PROMOTER_DOWNSTREAM = 200


def _read_annotation(gtf: str, keep):
    """The genes and transcripts of ``gtf``, read the way cytome reads every GTF."""
    try:
        from cytome.io.gtf_import import read_gtf
    except ImportError as exc:                       # cytome < 0.3.7
        raise ImportError(
            "makeGenomeFiles reads the annotation with cytome.read_gtf, new in "
            "cytome 0.3.7. Upgrade cytome: pip install -U cytome") from exc
    genes, feats = read_gtf(gtf, feature_filter=("gene", "transcript"))
    tx = feats[feats["feature"] == "transcript"]
    if keep:
        genes = genes[genes["chrom"].isin(keep)]
        tx = tx[tx["chrom"].isin(keep)]
    return genes, tx


def _text(value) -> str:
    """A GTF attribute as read, or '' where the file had none (None or NaN)."""
    return value if isinstance(value, str) else ""


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
    The GTF is read by ``cytome.read_gtf`` (cytome 0.3.7 or later): one gene
    per ``gene_id``, from its ``gene`` line or spanning its transcripts, named
    by ``gene_name``, else ``gene``, else ``Name``, else written under its ID.
    Genes that share a name stay separate rows, as in the shipped files.

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

    # --- the annotation ---------------------------------------------------
    # One gene per gene_id, from its `gene` line or spanning its transcripts:
    # this macaque annotation has `gene` lines for 2,799 of its 34,643 genes,
    # so trusting them alone would ship 8% of the genes. Coordinates are
    # already BED's 0-based half-open. A BED row needs a name, so a gene
    # without one (common in Ensembl annotations of non-model species) is
    # written under its ID.
    genes_df, tx_df = _read_annotation(gtf, keep)
    name_of: Dict[str, Tuple[str, str]] = {}
    gene_rows: List[Tuple] = []
    for gid, chrom, s, e, strand, gname, gtype in zip(
            genes_df["gene_id"], genes_df["chrom"], genes_df["start"],
            genes_df["end"], genes_df["strand"], genes_df["gene_name"],
            genes_df["gene_type"]):
        name = _text(gname) or gid
        name_of[gid] = (name, _text(gtype))
        gene_rows.append((chrom, int(s), int(e), name, strand))

    # --- gene bodies ------------------------------------------------------
    gene_rows.sort(key=lambda r: (_chrom_key(r[0]), r[1]))
    genes_path = os.path.join(outdir, f"{genome}_genes.bed")
    with open(genes_path, "w") as out:
        for chrom, s, e, name, strand in gene_rows:
            out.write(f"{chrom}\t{s}\t{e}\t{name}\t.\t{strand}\n")

    # --- transcript start sites ------------------------------------------
    # Column 8 is the gene ID, as in the shipped hg38 and mm10 files.
    tss_rows: List[Tuple] = []
    for gid, tid, chrom, s, e, strand, ttype in zip(
            tx_df["gene_id"], tx_df["transcript_id"], tx_df["chrom"],
            tx_df["start"], tx_df["end"], tx_df["strand"],
            tx_df["transcript_type"]):
        name, gtype = name_of.get(gid, (gid, ""))
        tss = int(s) if strand != "-" else int(e) - 1
        tss_rows.append((chrom, tss, name, strand, _text(ttype) or gtype,
                         gid, _text(tid)))
    tss_rows.sort(key=lambda r: (_chrom_key(r[0]), r[1]))
    tss_path = os.path.join(outdir, f"{genome}_transcript_tss.bed")
    with open(tss_path, "w") as out:
        for chrom, tss, name, strand, biotype, gene_id, tx_id in tss_rows:
            out.write(f"{chrom}\t{tss}\t{tss + 1}\t{name}\t.\t{strand}\t"
                      f"{biotype}\t{gene_id}\t{tx_id}\n")

    # --- promoters --------------------------------------------------------
    # Per gene, because the shipped files merge a gene's overlapping promoters
    # into one interval but keep two genes' promoters separate even where they
    # overlap; two genes that share a name are two genes.
    by_gene: Dict[Tuple[str, str], Tuple[str, List[Tuple[int, int]]]] = {}
    for chrom, tss, name, strand, _b, gid, _t in tss_rows:
        if strand == "-":
            s, e = tss - PROMOTER_DOWNSTREAM, tss + PROMOTER_UPSTREAM + 1
        else:
            s, e = tss - PROMOTER_UPSTREAM, tss + PROMOTER_DOWNSTREAM + 1
        by_gene.setdefault((chrom, gid), (name, []))[1].append((max(0, s), e))
    prom_path = os.path.join(outdir, f"{genome}_promoterSet.bed")
    prom_rows = []
    for (chrom, _gid), (name, iv) in by_gene.items():
        for s, e in _merge(iv):
            prom_rows.append((chrom, s, e, name))
    prom_rows.sort(key=lambda r: (_chrom_key(r[0]), r[1]))
    with open(prom_path, "w") as out:
        for chrom, s, e, name in prom_rows:
            out.write(f"{chrom}\t{s}\t{e}\t{name}\n")

    if verbose:
        print(f"  {genome}: {n_chrom:,} sequences, {len(gene_rows):,} genes, "
              f"{len(tss_rows):,} transcript TSSes, {len(prom_rows):,} promoters")
        print(f"  written to {outdir}")
    return {"chrom_sizes": sizes_path, "gene_boundary": genes_path,
            "tss_bed": tss_path, "promoter": prom_path}
