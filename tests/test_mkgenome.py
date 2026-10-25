"""Building PIASO's genome BEDs from a cellranger reference.

The source has to be the reference the data was mapped against, not the
original annotation: the reference carries whatever chromosome renaming was
done when it was built, and BED files whose names disagree with the fragments
do not raise, they find nothing.
"""
from __future__ import annotations

import gzip
import os

import pytest

from piaso.data._mkgenome import (PROMOTER_DOWNSTREAM, PROMOTER_UPSTREAM,
                                  makeGenomeFiles)

GTF = """\
#gtf-version 2.2
chr1\tSrc\tgene\t1001\t5000\t.\t+\t.\tgene_id "AAA"; gene "AAA"; gene_biotype "protein_coding";
chr1\tSrc\ttranscript\t1001\t5000\t.\t+\t.\tgene_id "AAA"; transcript_id "T1"; gene "AAA"; transcript_biotype "mRNA"; db_xref "GeneID:1";
chr1\tSrc\ttranscript\t1201\t5000\t.\t+\t.\tgene_id "AAA"; transcript_id "T2"; gene "AAA"; transcript_biotype "mRNA"; db_xref "GeneID:1";
chr1\tSrc\texon\t1001\t2000\t.\t+\t.\tgene_id "AAA"; transcript_id "T1"; gene "AAA";
chr2\tSrc\ttranscript\t9000\t9500\t.\t-\t.\tgene_id "BBB"; transcript_id "T3"; gene "BBB"; transcript_biotype "lncRNA"; db_xref "GeneID:2";
scaf_9\tSrc\ttranscript\t100\t400\t.\t+\t.\tgene_id "CCC"; transcript_id "T4"; gene "CCC"; transcript_biotype "mRNA";
"""

FAI = "chr2\t20000\t0\t80\t81\nscaf_9\t500\t0\t80\t81\nchr1\t10000\t0\t80\t81\n"


@pytest.fixture
def reference(tmp_path):
    d = tmp_path / "ref"
    (d / "fasta").mkdir(parents=True)
    (d / "genes").mkdir(parents=True)
    (d / "fasta" / "genome.fa.fai").write_text(FAI)
    with gzip.open(d / "genes" / "genes.gtf.gz", "wt") as fh:
        fh.write(GTF)
    return str(d)


def _rows(path):
    return [l.rstrip("\n").split("\t") for l in open(path)]


def test_the_four_files_are_written_with_the_shipped_column_counts(reference, tmp_path):
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    assert set(out) == {"chrom_sizes", "gene_boundary", "tss_bed", "promoter"}
    assert all(os.path.exists(p) for p in out.values())
    assert len(_rows(out["chrom_sizes"])[0]) == 2
    assert len(_rows(out["gene_boundary"])[0]) == 6
    assert len(_rows(out["tss_bed"])[0]) == 9
    assert len(_rows(out["promoter"])[0]) == 4


def test_gene_bodies_come_from_transcripts_not_only_gene_lines(reference, tmp_path):
    """The macaque annotation has 2,799 gene lines against 34,204 genes.

    Trusting `gene` features alone would have shipped 8% of the genes.
    """
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    names = {r[3] for r in _rows(out["gene_boundary"])}
    assert names == {"AAA", "BBB", "CCC"}, names
    aaa = next(r for r in _rows(out["gene_boundary"]) if r[3] == "AAA")
    # Union of the gene line and both transcripts, 0-based half-open.
    assert (int(aaa[1]), int(aaa[2])) == (1000, 5000)


def test_the_tss_is_the_strand_aware_start(reference, tmp_path):
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    tss = {(r[3], r[8]): int(r[1]) for r in _rows(out["tss_bed"])}
    assert tss[("AAA", "T1")] == 1000          # + strand: the left edge
    assert tss[("BBB", "T3")] == 9499          # - strand: the right edge


def test_the_promoter_window_matches_the_shipped_genomes(reference, tmp_path):
    """400 up, 200 down, strand-aware, measured off mm10 and hg38."""
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    prom = {r[3]: (int(r[1]), int(r[2])) for r in _rows(out["promoter"])}
    # BBB is on the minus strand, TSS 9499: upstream is the HIGHER coordinate.
    s, e = prom["BBB"]
    assert 9499 - s == PROMOTER_DOWNSTREAM
    assert e - 9499 == PROMOTER_UPSTREAM + 1
    assert e - s == PROMOTER_UPSTREAM + PROMOTER_DOWNSTREAM + 1 == 601


def test_a_genes_overlapping_promoters_are_merged(reference, tmp_path):
    """AAA has two TSSes 200 bp apart; their promoters overlap."""
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    aaa = [r for r in _rows(out["promoter"]) if r[3] == "AAA"]
    assert len(aaa) == 1, aaa
    assert int(aaa[0][2]) - int(aaa[0][1]) == 801      # 601 + the 200 bp offset


def test_chromosomes_are_written_in_karyotypic_order(reference, tmp_path):
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    order = [r[0] for r in _rows(out["chrom_sizes"])]
    assert order == ["chr1", "chr2", "scaf_9"], order
    tss_order = [r[0] for r in _rows(out["tss_bed"])]
    assert tss_order.index("chr1") < tss_order.index("scaf_9")


def test_every_bed_chromosome_is_in_the_sizes_file(reference, tmp_path):
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    sizes = {r[0] for r in _rows(out["chrom_sizes"])}
    for kind in ("gene_boundary", "tss_bed", "promoter"):
        assert {r[0] for r in _rows(out[kind])} <= sizes, kind


def test_a_chromosome_restriction_is_honoured(reference, tmp_path):
    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"),
                          chromosomes=["chr1", "chr2"], verbose=False)
    assert {r[0] for r in _rows(out["chrom_sizes"])} == {"chr1", "chr2"}
    assert "scaf_9" not in {r[0] for r in _rows(out["tss_bed"])}


def test_a_directory_that_is_not_a_reference_says_what_was_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="genes.gtf"):
        makeGenomeFiles(str(tmp_path), "X", str(tmp_path / "o"), verbose=False)


def test_the_two_primate_genomes_are_registered():
    from piaso.data._genome import GENOME_FEATURES, GENOME_FILES

    for g in ("Mmul_10", "mCalJac1"):
        assert g in GENOME_FILES
        assert "ctcf" not in GENOME_FILES[g], "SCREEN has no primate beyond human"
        assert {"gene_boundary", "promoter", "chrom_sizes", "tss_bed"} <= set(GENOME_FILES[g])
    # The marmoset assembly has no mitochondrion; a QC that reports 0% there
    # is describing the assembly, not the cells.
    assert GENOME_FEATURES["mCalJac1"]["has_chrM"] is False
    assert GENOME_FEATURES["Mmul_10"]["has_chrM"] is True


def _reference_from(tmp_path, gtf_text, fai):
    d = tmp_path / "ref2"
    (d / "fasta").mkdir(parents=True)
    (d / "genes").mkdir(parents=True)
    (d / "fasta" / "genome.fa.fai").write_text(fai)
    with gzip.open(d / "genes" / "genes.gtf.gz", "wt") as fh:
        fh.write(gtf_text)
    return str(d)


# NCBI style: a repeated symbol is suffixed in gene_id and kept whole in `gene`.
REPEATED = """\
chr1\tSrc\ttranscript\t1001\t1072\t.\t+\t.\tgene_id "TRNAN-GUU"; transcript_id "R1"; gene "TRNAN-GUU"; transcript_biotype "tRNA"; db_xref "GeneID:11";
chr1\tSrc\ttranscript\t5000001\t5000072\t.\t-\t.\tgene_id "TRNAN-GUU_1"; transcript_id "R2"; gene "TRNAN-GUU"; transcript_biotype "tRNA"; db_xref "GeneID:12";
chr1\tSrc\ttranscript\t9000001\t9000072\t.\t+\t.\tgene_id "TRNAN-GUU_2"; transcript_id "R3"; gene "TRNAN-GUU"; transcript_biotype "tRNA"; db_xref "GeneID:13";
chr1\tSrc\ttranscript\t9000201\t9000272\t.\t+\t.\tgene_id "TRNAN-GUU_3"; transcript_id "R4"; gene "TRNAN-GUU"; transcript_biotype "tRNA"; db_xref "GeneID:14";
"""


def test_genes_that_share_a_name_stay_separate(tmp_path):
    """Keyed by name, the Mmul_10 reference's copies of TRNAN-GUU became one 118.6 Mb gene.

    Each gene_id is its own row, and so are its promoters: TRNAN-GUU_2 and _3 sit 200 bp apart,
    their promoters overlap, and they are still two genes.
    """
    ref = _reference_from(tmp_path, REPEATED, "chr1\t10000000\t0\t80\t81\n")
    out = makeGenomeFiles(ref, "T", str(tmp_path / "o"), verbose=False)
    genes = _rows(out["gene_boundary"])
    assert [(r[1], r[2], r[3]) for r in genes] == [
        ("1000", "1072", "TRNAN-GUU"), ("5000000", "5000072", "TRNAN-GUU"),
        ("9000000", "9000072", "TRNAN-GUU"), ("9000200", "9000272", "TRNAN-GUU")]
    assert max(int(r[2]) - int(r[1]) for r in genes) == 72
    prom = _rows(out["promoter"])
    assert len(prom) == 4, prom
    assert int(prom[3][1]) < int(prom[2][2]), "the last two promoters overlap and stay two rows"


def test_the_tss_file_carries_the_gene_id_in_column_8(tmp_path):
    """As in the shipped hg38 and mm10 files; NCBI's db_xref was there before."""
    ref = _reference_from(tmp_path, REPEATED, "chr1\t10000000\t0\t80\t81\n")
    out = makeGenomeFiles(ref, "T", str(tmp_path / "o"), verbose=False)
    assert [(r[3], r[6], r[7], r[8]) for r in _rows(out["tss_bed"])][:2] == [
        ("TRNAN-GUU", "tRNA", "TRNAN-GUU", "R1"), ("TRNAN-GUU", "tRNA", "TRNAN-GUU_1", "R2")]


# GENCODE / Ensembl style: gene lines, gene_name, gene_type / transcript_type, an unnamed gene.
ENSEMBL = """\
8\tEns\tgene\t1001\t3000\t.\t+\t.\tgene_id "ENSG1"; gene_name "CHRNA2"; gene_type "protein_coding";
8\tEns\ttranscript\t1001\t3000\t.\t+\t.\tgene_id "ENSG1"; transcript_id "ENST1"; gene_name "CHRNA2"; transcript_type "protein_coding";
8\tEns\texon\t1001\t1500\t.\t+\t.\tgene_id "ENSG1"; transcript_id "ENST1"; gene_name "CHRNA2";
8\tEns\tgene\t7001\t9000\t.\t-\t.\tgene_id "ENSMMUG00000023296"; gene_biotype "protein_coding";
8\tEns\ttranscript\t7001\t9000\t.\t-\t.\tgene_id "ENSMMUG00000023296"; transcript_id "ENSMMUT9"; transcript_biotype "protein_coding";
"""


def test_gencode_and_ensembl_attributes_are_read(tmp_path):
    """Names from gene_name, biotypes from transcript_type (GENCODE spells it so), and a gene
    without a name written under its ID."""
    ref = _reference_from(tmp_path, ENSEMBL, "8\t20000\t0\t80\t81\n")
    out = makeGenomeFiles(ref, "T", str(tmp_path / "o"), verbose=False)
    assert [(r[3], r[5]) for r in _rows(out["gene_boundary"])] == [
        ("CHRNA2", "+"), ("ENSMMUG00000023296", "-")]
    tss = _rows(out["tss_bed"])
    assert [(r[3], r[6], r[7]) for r in tss] == [
        ("CHRNA2", "protein_coding", "ENSG1"),
        ("ENSMMUG00000023296", "protein_coding", "ENSMMUG00000023296")]
    assert int(tss[1][1]) == 8999           # minus strand: the right edge, 0-based


def test_the_genes_are_those_of_cytomes_reader(reference, tmp_path):
    """One reader: the gene BED is cytome.read_gtf's gene table, row for row."""
    import cytome

    out = makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
    genes, _ = cytome.read_gtf(os.path.join(reference, "genes", "genes.gtf.gz"))
    want = {(g.chrom, g.start, g.end, g.gene_name or g.gene_id, g.strand)
            for g in genes.itertuples()}
    got = {(r[0], int(r[1]), int(r[2]), r[3], r[5]) for r in _rows(out["gene_boundary"])}
    assert got == want


def test_an_older_cytome_is_named_in_the_error(reference, tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_reader(name, *a, **k):
        if name == "cytome.io.gtf_import":
            raise ImportError("cannot import name 'read_gtf'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_reader)
    with pytest.raises(ImportError, match="cytome 0.3.7"):
        makeGenomeFiles(reference, "TestG", str(tmp_path / "o"), verbose=False)
