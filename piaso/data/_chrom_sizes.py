"""Chromosome sizes: the genomes PIASO knows by name, and any chrom.sizes file.

The table is the one compiled into the Rust extension and the ``cytome-import-
fragments`` / ``piaso-atac`` binaries, so ``chrom_sizes("Mmul_10")`` here, the
tile grid the importer builds and the chromosomes the peak caller walks are all
the same rows. When the extension is not built, the copy below answers.

A file is reduced to its primary chromosomes by name: sequences with an
underscore (``chr1_KI270706v1_random``, ``chrUn_...``), ``chrUn*`` and the
mitochondrion are left out; everything else is kept, so a reference that
spells a chromosome ``chr2A`` keeps it.
"""

import os
import re

#: Chromosome sizes of the genomes PIASO knows by name: the same table the
#: Rust extension carries (``_piaso.chrom_sizes``); this copy answers when the
#: extension is not built. ``tests/test_genome_data.py`` checks the two agree.
_TABLES = {
    "hg38": (
        ("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559), ("chr4", 190214555),
        ("chr5", 181538259), ("chr6", 170805979), ("chr7", 159345973), ("chr8", 145138636),
        ("chr9", 138394717), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309),
        ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345),
        ("chr17", 83257441), ("chr18", 80373285), ("chr19", 58617616), ("chr20", 64444167),
        ("chr21", 46709983), ("chr22", 50818468), ("chrX", 156040895), ("chrY", 57227415),
    ),
    "mm10": (
        ("chr1", 195471971), ("chr2", 182113224), ("chr3", 160039680), ("chr4", 156508116),
        ("chr5", 151834684), ("chr6", 149736546), ("chr7", 145441459), ("chr8", 129401213),
        ("chr9", 124595110), ("chr10", 130694993), ("chr11", 122082543), ("chr12", 120129022),
        ("chr13", 120421639), ("chr14", 124902244), ("chr15", 104043685), ("chr16", 98207768),
        ("chr17", 94987271), ("chr18", 90702639), ("chr19", 61431566), ("chrX", 171031299),
        ("chrY", 91744698),
    ),
    "Mmul_10": (
        ("chr1", 223616942), ("chr2", 196197964), ("chr3", 185288947), ("chr4", 169963040),
        ("chr5", 187317192), ("chr6", 179085566), ("chr7", 169868564), ("chr8", 145679320),
        ("chr9", 134124166), ("chr10", 99517758), ("chr11", 133066086), ("chr12", 130043856),
        ("chr13", 108737130), ("chr14", 128056306), ("chr15", 113283604), ("chr16", 79627064),
        ("chr17", 95433459), ("chr18", 74474043), ("chr19", 58315233), ("chr20", 77137495),
        ("chrX", 153388924), ("chrY", 11753682),
    ),
    "mCalJac1": (
        ("chr1", 216975769), ("chr2", 202808461), ("chr3", 189455076), ("chr4", 173414057),
        ("chr5", 161716902), ("chr6", 159674559), ("chr7", 156129252), ("chr8", 126104592),
        ("chr9", 132900640), ("chr10", 136971485), ("chr11", 128529401), ("chr12", 123697827),
        ("chr13", 117638787), ("chr14", 112969529), ("chr15", 98621277), ("chr16", 98110366),
        ("chr17", 74700100), ("chr18", 47063576), ("chr19", 50540917), ("chr20", 44412365),
        ("chr21", 50614742), ("chr22", 49900457), ("chrX", 146897247), ("chrY", 4974773),
    ),
    "hg19": (
        ("chr1", 249250621), ("chr2", 243199373), ("chr3", 198022430), ("chr4", 191154276),
        ("chr5", 180915260), ("chr6", 171115067), ("chr7", 159138663), ("chr8", 146364022),
        ("chr9", 141213431), ("chr10", 135534747), ("chr11", 135006516), ("chr12", 133851895),
        ("chr13", 115169878), ("chr14", 107349540), ("chr15", 102531392), ("chr16", 90354753),
        ("chr17", 81195210), ("chr18", 78077248), ("chr19", 59128983), ("chr20", 63025520),
        ("chr21", 48129895), ("chr22", 51304566), ("chrX", 155270560), ("chrY", 59373566),
    ),
    "mm39": (
        ("chr1", 195154279), ("chr2", 181755017), ("chr3", 159745316), ("chr4", 156860686),
        ("chr5", 151758149), ("chr6", 149588044), ("chr7", 144995196), ("chr8", 130127694),
        ("chr9", 124359700), ("chr10", 130530862), ("chr11", 121973369), ("chr12", 120092757),
        ("chr13", 120883175), ("chr14", 125139656), ("chr15", 104073951), ("chr16", 98008968),
        ("chr17", 95294699), ("chr18", 90720763), ("chr19", 61420004), ("chrX", 169476592),
        ("chrY", 91455967),
    ),
}

KNOWN_GENOMES = tuple(_TABLES)

_MITO = {"chrM", "chrMT", "MT", "M"}


def is_primary_chrom(name: str) -> bool:
    """Whether a sequence name from a chrom.sizes file is a primary chromosome."""
    return not ("_" in name or name.startswith("chrUn") or name in _MITO)


def chrom_sort_key(name: str):
    """Numbered chromosomes by number (``2A`` after ``2``), then X, Y, then the rest by name."""
    s = name[3:] if name.startswith("chr") else name
    m = re.match(r"(\d+)(.*)", s)
    if m:
        return (0, int(m.group(1)), m.group(2))
    if s in ("X", "Y"):
        return (1, 0 if s == "X" else 1, "")
    return (2, 0, s)


def _read_file(path: str) -> dict:
    sizes = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2 or not is_primary_chrom(parts[0]):
                continue
            sizes[parts[0]] = int(parts[1])
    if not sizes:
        raise ValueError(
            f"No primary chromosomes found in '{path}' (names with an underscore, chrUn* and "
            f"the mitochondrion are left out; the rest must have a tab-separated size).")
    return dict(sorted(sizes.items(), key=lambda kv: chrom_sort_key(kv[0])))


def _python_chrom_sizes(genome: str) -> dict:
    g = str(genome)
    if g in _TABLES:
        return dict(_TABLES[g])
    if os.path.isfile(g):
        return _read_file(g)
    raise ValueError(
        f"'{g}' is not a genome known by name ({', '.join(KNOWN_GENOMES)}) and is not a "
        f"chrom.sizes file. Pass one of those names, or the path to a two-column "
        f"<chrom>\\t<size> file (a .fai works too).")


def chrom_sizes(genome: str) -> dict:
    """Chromosome sizes of a genome, ``{chrom: size}`` in canonical order.

    Parameters
    ----------
    genome
        A name PIASO knows (``hg38``, ``mm10``, ``Mmul_10``, ``mCalJac1``,
        ``hg19``, ``mm39``) or the path to a ``chrom.sizes`` / ``.fai`` file.

    Returns
    -------
    dict
        Primary chromosomes only, numbered first, then X and Y. For a name,
        exactly the rows the fragment importer builds its tile grid from and the
        peak caller walks; for a file, the same reduction those tools apply.

    Examples
    --------
    >>> piaso.data.chrom_sizes("Mmul_10")["chr1"]
    223616942
    >>> len(piaso.data.chrom_sizes("/path/to/panTro6.chrom.sizes"))
    """
    try:
        from .._piaso import chrom_sizes as _rust
    except ImportError:
        return _python_chrom_sizes(genome)
    return dict(_rust(str(genome)))
