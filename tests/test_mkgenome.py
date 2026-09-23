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
