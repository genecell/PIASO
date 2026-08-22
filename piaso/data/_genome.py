"""
Genome reference data management for PIASO.

Downloads and caches genome reference files (gene bodies, promoters, CTCF sites)
from the PIASO-data GitHub repository, plus GTF gene-annotation files from
GENCODE / RefSeq / Ensembl upstream sources.
"""

import gzip
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

PIASO_DATA_DIR = Path.home() / ".piaso" / "data"


def resolve_data_dir(data_dir=None) -> Path:
    """Resolve the PIASO data root. Most specific wins:

    1. explicit ``data_dir`` argument (per call)
    2. ``piaso.settings.data_dir`` (per session)
    3. the ``PIASO_DATA_DIR`` environment variable (per machine)
    4. ``~/.piaso/data`` (the historical default)

    Datasets, genomes, and the registry cache all share this root, so the
    stores cannot end up configured differently by accident.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser()
    try:
        from .. import settings as _settings
        session_dir = getattr(_settings, "data_dir", None)
    except ImportError:  # pragma: no cover - settings always importable
        session_dir = None
    if session_dir is not None:
        return Path(session_dir).expanduser()
    env_dir = os.environ.get("PIASO_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return PIASO_DATA_DIR

# Per-file raw URL on the PIASO-data repo's master branch.
# We download each file directly rather than a release tarball — the repo's
# layout already places files in {genome}/ subfolders, and the per-file
# approach works without needing GitHub releases to be cut.
PIASO_DATA_RAW_URL = "https://raw.githubusercontent.com/genecell/PIASO-data/master"

GENOME_FILES = {
    "hg38": {
        "gene_boundary": "hg38_genes.bed",
        "promoter": "hg38_promoterSet.bed",
        "ctcf": "GRCh38-cCREs.CTCF-only.bed",
        "chrom_sizes": "hg38.chrom.sizes",
        "tss_bed": "hg38_transcript_tss.bed",
        # Optional: gene structure annotation for plotCoverage /
        # plotBigWig gene-track rendering. Larger than the BEDs
        # (~30-50 MB compressed) so download is optional.
        "gtf": "gencode.v44.basic.annotation.gtf.gz",
    },
    "mm10": {
        "gene_boundary": "mm10_genes.bed",
        "promoter": "mm10_promoterSet.bed",
        "ctcf": "mm10-cCREs.CTCF-only.bed",
        "chrom_sizes": "mm10.chrom.sizes",
        "tss_bed": "mm10_transcript_tss.bed",
        "gtf": "gencode.vM25.basic.annotation.gtf.gz",
    },
}

# Nested GTF source table: {genome → source → release → spec}.
# `spec` keys:
#   filename                — basename of the cached .gtf.gz
#   url                     — upstream URL to fetch from
#   assembly_report_url     — required when needs_chrom_remap='refseq'
#   needs_chrom_remap       — None | 'refseq' | 'ensembl'
#
# 'refseq'  : RefSeq ships `NC_000001.11`-style chrom names; we
#             download _assembly_report.txt and remap to UCSC
#             style (`chr1`).
# 'ensembl' : Ensembl ships `1`, `2`, ..., `X`, `MT` (no `chr`
#             prefix); we prepend `chr`, with `MT` → `chrM`.
GTF_SOURCES: Dict[str, Dict[str, Dict[str, dict]]] = {
    "hg38": {
        "gencode": {
            "v44": {
                "filename": "gencode.v44.basic.annotation.gtf.gz",
                "url": (
                    "https://ftp.ebi.ac.uk/pub/databases/gencode/"
                    "Gencode_human/release_44/"
                    "gencode.v44.basic.annotation.gtf.gz"
                ),
                "needs_chrom_remap": None,
            },
        },
        "refseq": {
            "110": {
                "filename": (
                    "refseq_GRCh38.p14_110.gtf.gz"
                ),
                "url": (
                    "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
                    "annotation_releases/vertebrate_mammalian/"
                    "Homo_sapiens/110/"
                    "GCF_000001405.40_GRCh38.p14/"
                    "GCF_000001405.40_GRCh38.p14_genomic.gtf.gz"
                ),
                "assembly_report_url": (
                    "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
                    "annotation_releases/vertebrate_mammalian/"
                    "Homo_sapiens/110/"
                    "GCF_000001405.40_GRCh38.p14/"
                    "GCF_000001405.40_GRCh38.p14_assembly_report.txt"
                ),
                "needs_chrom_remap": "refseq",
            },
        },
        "ensembl": {
            "110": {
                "filename": "ensembl_Homo_sapiens.GRCh38.110.gtf.gz",
                "url": (
                    "https://ftp.ensembl.org/pub/release-110/gtf/"
                    "homo_sapiens/Homo_sapiens.GRCh38.110.chr.gtf.gz"
                ),
                "needs_chrom_remap": "ensembl",
            },
        },
    },
    "mm10": {
        "gencode": {
            "vM25": {
                "filename": "gencode.vM25.basic.annotation.gtf.gz",
                "url": (
                    "https://ftp.ebi.ac.uk/pub/databases/gencode/"
                    "Gencode_mouse/release_M25/"
                    "gencode.vM25.basic.annotation.gtf.gz"
                ),
                "needs_chrom_remap": None,
            },
        },
        "refseq": {
            "106": {
                "filename": "refseq_GRCm38.p6_106.gtf.gz",
                "url": (
                    "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
                    "annotation_releases/vertebrate_mammalian/"
                    "Mus_musculus/106/"
                    "GCF_000001635.26_GRCm38.p6/"
                    "GCF_000001635.26_GRCm38.p6_genomic.gtf.gz"
                ),
                "assembly_report_url": (
                    "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
                    "annotation_releases/vertebrate_mammalian/"
                    "Mus_musculus/106/"
                    "GCF_000001635.26_GRCm38.p6/"
                    "GCF_000001635.26_GRCm38.p6_assembly_report.txt"
                ),
                "needs_chrom_remap": "refseq",
            },
        },
        "ensembl": {
            "102": {
                "filename": "ensembl_Mus_musculus.GRCm38.102.gtf.gz",
                "url": (
                    "https://ftp.ensembl.org/pub/release-102/gtf/"
                    "mus_musculus/Mus_musculus.GRCm38.102.chr.gtf.gz"
                ),
                "needs_chrom_remap": "ensembl",
            },
        },
    },
}

# Per-source default release used when `release=None` in fetch_genome.
DEFAULT_GTF_RELEASE = {
    ("hg38",  "gencode"): "v44",
    ("hg38",  "refseq"):  "110",
    ("hg38",  "ensembl"): "110",
    ("mm10",  "gencode"): "vM25",
    ("mm10",  "refseq"):  "106",
    ("mm10",  "ensembl"): "102",
}

# Friendly preset bundles for the `transcript_id_pattern` /
# `transcript_id_prefixes` / `transcript_tags` / `gene_biotypes`
# kwargs accepted by cytome.import_gtf and plotCoverage. Pass these
# via `**piaso.data.GTF_PRESETS[name]` for the common combos.
GTF_PRESETS: Dict[str, dict] = {
    "gencode_basic":       {"transcript_tags": ["basic"]},
    "ensembl_canonical":   {"transcript_tags": ["Ensembl_canonical"]},
    "mane_select":         {"transcript_tags": ["MANE_Select"]},
    "appris_principal":    {"transcript_tags": ["appris_principal_1"]},
    "refseq_curated":      {"transcript_id_prefixes": ["NM_", "NR_"]},
    "protein_coding_only": {"gene_biotypes": ["protein_coding"]},
}


# Back-compat shim: the old flat URL table is still referenced by
# downstream callers. Keep it in sync with the gencode default
# release per genome.
GTF_UPSTREAM_URLS = {
    g: GTF_SOURCES[g]["gencode"][DEFAULT_GTF_RELEASE[(g, "gencode")]]["url"]
    for g in GTF_SOURCES
}


def list_available_genomes() -> List[str]:
    """Return list of supported genome names."""
    return list(GENOME_FILES.keys())


def list_available_gtf_sources(genome: str) -> Dict[str, List[str]]:
    """Return ``{source: [release, ...]}`` for the given genome.

    Example: ``list_available_gtf_sources('mm10')`` →
    ``{'gencode': ['vM25'], 'refseq': ['106'], 'ensembl': ['102']}``.
    """
    if genome not in GTF_SOURCES:
        raise ValueError(
            f"Unsupported genome {genome!r}. Available: "
            f"{list(GTF_SOURCES.keys())}"
        )
    return {src: list(rel.keys()) for src, rel in GTF_SOURCES[genome].items()}


def _resolve_gtf_spec(genome: str, source: str = "gencode",
                       release: Optional[str] = None) -> dict:
    """Resolve ``(genome, source, release)`` to the URL spec dict.

    ``release=None`` picks the default release for that
    ``(genome, source)`` from :data:`DEFAULT_GTF_RELEASE`.
    Returns a dict containing keys: ``filename``, ``url``,
    ``needs_chrom_remap``, and optionally ``assembly_report_url``.
    """
    if genome not in GTF_SOURCES:
        raise ValueError(
            f"Unsupported genome {genome!r}. Available: "
            f"{list(GTF_SOURCES.keys())}"
        )
    src_table = GTF_SOURCES[genome]
    if source not in src_table:
        raise ValueError(
            f"GTF source {source!r} not available for genome "
            f"{genome!r}. Available: {list(src_table.keys())}"
        )
    if release is None:
        release = DEFAULT_GTF_RELEASE.get((genome, source))
        if release is None:
            raise ValueError(
                f"No default release for genome={genome!r}, "
                f"source={source!r}. Pass release= explicitly. "
                f"Available: {list(src_table[source].keys())}"
            )
    rel_table = src_table[source]
    if release not in rel_table:
        raise ValueError(
            f"Release {release!r} not available for genome={genome!r}, "
            f"source={source!r}. Available: {list(rel_table.keys())}"
        )
    return rel_table[release]


def _parse_refseq_assembly_report(path) -> Dict[str, str]:
    """Parse an NCBI ``*_assembly_report.txt`` into ``{refseq_accn → ucsc_name}``.

    Skips header lines (starting with ``#``) and entries without a
    UCSC-style name (the assembly_report uses ``na`` for those).
    """
    remap: Dict[str, str] = {}
    with open(path, "rt") as f:
        header = None
        for line in f:
            if line.startswith("# Sequence-Name"):
                # Header row identifying columns
                header = [c.strip() for c in line.lstrip("#").rstrip("\n").split("\t")]
                continue
            if line.startswith("#") or not line.strip():
                continue
            if header is None:
                # Fall back to documented column order
                header = [
                    "Sequence-Name", "Sequence-Role", "Assigned-Molecule",
                    "Assigned-Molecule-Location/Type", "GenBank-Accn",
                    "Relationship", "RefSeq-Accn", "Assembly-Unit",
                    "Sequence-Length", "UCSC-style-name",
                ]
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            refseq = row.get("RefSeq-Accn", "").strip()
            ucsc = row.get("UCSC-style-name", "").strip()
            if not refseq or not ucsc or ucsc.lower() == "na":
                continue
            remap[refseq] = ucsc
    return remap


def _remap_gtf_chroms(in_path, out_path, chrom_map: Dict[str, str]) -> Dict[str, int]:
    """Stream-rewrite ``in_path`` → ``out_path`` replacing column 1
    chrom names via ``chrom_map``. Both files are gzipped.

    Lines whose chrom isn't in the map are SKIPPED (they're typically
    unplaced scaffolds with no UCSC mapping). Header lines (starting
    with ``#``) are preserved verbatim.

    Returns ``{'kept': n_kept, 'dropped': n_dropped}``.
    """
    stats = {"kept": 0, "dropped": 0}
    with gzip.open(in_path, "rt") as fin, \
            gzip.open(out_path, "wt") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            t1 = line.find("\t")
            if t1 == -1:
                continue
            chrom = line[:t1]
            new_chrom = chrom_map.get(chrom)
            if new_chrom is None:
                stats["dropped"] += 1
                continue
            fout.write(new_chrom + line[t1:])
            stats["kept"] += 1
    return stats


def _remap_ensembl_chrom(name: str) -> str:
    """Ensembl uses ``1``, ``2``, …, ``X``, ``MT``. UCSC convention is
    ``chr1``, ``chr2``, …, ``chrX``, ``chrM``."""
    if name == "MT":
        return "chrM"
    if name.startswith("chr"):
        return name
    return f"chr{name}"


def _remap_ensembl_gtf(in_path, out_path) -> Dict[str, int]:
    """Like ``_remap_gtf_chroms`` but for Ensembl conventions."""
    stats = {"kept": 0, "dropped": 0}
    with gzip.open(in_path, "rt") as fin, \
            gzip.open(out_path, "wt") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            t1 = line.find("\t")
            if t1 == -1:
                continue
            chrom = line[:t1]
            new_chrom = _remap_ensembl_chrom(chrom)
            fout.write(new_chrom + line[t1:])
            stats["kept"] += 1
    return stats


def list_downloaded_genomes() -> List[str]:
    """Return list of genome names present under the PIASO data root."""
    root = resolve_data_dir()
    if not root.exists():
        return []
    return [
        d.name for d in root.iterdir()
        if d.is_dir() and d.name in GENOME_FILES
    ]


def resolve_genome_files(
    genome: str,
    source: str = "gencode",
    release: Optional[str] = None,
) -> Dict[str, str]:
    """Return dict with resolved paths to BED files for a genome.

    Parameters
    ----------
    genome : str
        Genome name (e.g., 'hg38', 'mm10').
    source : str, default ``'gencode'``
        GTF source. One of ``'gencode'``, ``'refseq'``, ``'ensembl'``.
        Selects which cached GTF to return under the ``'gtf'`` key.
    release : str, optional
        GTF release (e.g. ``'vM25'``, ``'v44'``, ``'110'``). ``None``
        picks the per-source default from :data:`DEFAULT_GTF_RELEASE`.

    Returns
    -------
    paths : dict
        Keys: 'gene_boundary', 'promoter', 'ctcf', 'chrom_sizes',
        'tss_bed', and optionally 'gtf' (if the requested GTF was
        downloaded).

    Raises
    ------
    ValueError
        If genome / source / release is not supported.
    FileNotFoundError
        If required files are not downloaded yet.
    """
    if genome not in GENOME_FILES:
        raise ValueError(
            f"Unsupported genome '{genome}'. "
            f"Available: {list_available_genomes()}"
        )

    _OPTIONAL_KEYS = {"tss_bed", "gtf"}
    genome_dir = resolve_data_dir() / genome
    paths = {}
    missing_required = []

    # Non-GTF files (BEDs + chrom.sizes): source/release-independent
    for key, filename in GENOME_FILES[genome].items():
        if key == "gtf":
            continue  # GTF resolved via GTF_SOURCES table below
        path = genome_dir / filename
        if path.exists():
            paths[key] = str(path)
        elif key in _OPTIONAL_KEYS:
            continue
        else:
            missing_required.append(filename)

    # GTF: resolve via (source, release) → spec
    try:
        spec = _resolve_gtf_spec(genome, source, release)
        gtf_path = genome_dir / spec["filename"]
        if gtf_path.exists():
            paths["gtf"] = str(gtf_path)
    except ValueError:
        # Unknown source/release for the GTF lookup — silently skip
        # (BED-only callers don't need this to succeed).
        pass

    if missing_required:
        raise FileNotFoundError(
            f"Genome data for '{genome}' not found at {genome_dir}.\n"
            f"Missing: {', '.join(missing_required)}\n"
            f"Download with: piaso.data.fetch_genome('{genome}')\n"
            f"Or provide BED file paths directly."
        )

    return paths


def fetch_genome(
    genome: str,
    source_dir: Optional[str] = None,
    dest_dir: Optional[str] = None,
    force: bool = False,
    download_gtf: bool = True,
    source: str = "gencode",
    release: Optional[str] = None,
) -> str:
    """Download or copy genome reference files.

    Parameters
    ----------
    genome : str
        Genome name (e.g., 'hg38', 'mm10').
    source_dir : str, optional
        Local directory containing the genome files. If provided, copies
        from this directory instead of downloading.
    dest_dir : str, optional
        Destination directory for the genome files. If None (default),
        uses ``~/.piaso/data/{genome}/``. Files always end up in a
        ``{genome}/`` subdirectory under whichever destination is used.
    force : bool
        If True, re-download even if files exist.
    download_gtf : bool, default True
        Whether to fetch the GTF file. The GTF is large (~30-50 MB)
        relative to the BED files, so users can opt out if they don't
        need the gene-structure overlay in plotCoverage / plotBigWig.
    source : str, default ``'gencode'``
        GTF upstream source. One of ``'gencode'``, ``'refseq'``,
        ``'ensembl'``. RefSeq + Ensembl are auto-remapped to UCSC
        ``chr1``-style chrom names; GENCODE already uses them.
    release : str, optional
        GTF release identifier. ``None`` picks the per-source default
        from :data:`DEFAULT_GTF_RELEASE`. Use
        :func:`list_available_gtf_sources` to enumerate.

    Returns
    -------
    genome_dir : str
        Path to the genome data directory (as a string, so it drops
        straight into ``open()`` / ``os.path.join`` / f-strings; wrap in
        ``pathlib.Path(...)`` if you need Path methods). Matches the
        string paths returned by :func:`resolve_genome_files`.
    """
    if genome not in GENOME_FILES:
        raise ValueError(
            f"Unsupported genome '{genome}'. "
            f"Available: {list_available_genomes()}"
        )

    if dest_dir is not None:
        genome_dir = Path(dest_dir) / genome
    else:
        genome_dir = resolve_data_dir() / genome

    # Non-GTF files (BEDs + chrom.sizes)
    file_specs = {
        k: v for k, v in GENOME_FILES[genome].items() if k != "gtf"
    }
    # GTF spec (resolved via source/release table)
    gtf_spec = None
    if download_gtf:
        try:
            gtf_spec = _resolve_gtf_spec(genome, source, release)
        except ValueError as e:
            print(f"  WARNING: GTF not fetched — {e}")
            gtf_spec = None

    # Idempotency check
    if not force and genome_dir.exists():
        bed_present = all(
            (genome_dir / fn).exists() for fn in file_specs.values()
        )
        gtf_present = (gtf_spec is None
                       or (genome_dir / gtf_spec["filename"]).exists())
        if bed_present and gtf_present:
            print(f"Genome '{genome}' already available at {genome_dir}")
            return str(genome_dir)

    genome_dir.mkdir(parents=True, exist_ok=True)

    if source_dir is not None:
        # Copy from local directory
        local = Path(source_dir)
        for key, filename in file_specs.items():
            src = local / filename
            dst = genome_dir / filename
            if src.exists():
                shutil.copy2(str(src), str(dst))
                print(f"  Copied {filename}")
            else:
                print(f"  WARNING: {filename} not found in {source_dir}")
        if gtf_spec is not None:
            src = local / gtf_spec["filename"]
            dst = genome_dir / gtf_spec["filename"]
            if src.exists():
                shutil.copy2(str(src), str(dst))
                print(f"  Copied {gtf_spec['filename']}")
    else:
        # Download per-file from the PIASO-data master branch on GitHub
        import urllib.request

        print(
            f"Downloading {genome} reference data "
            f"(GTF: {source}{f' release {release}' if release else ''})..."
        )
        failed = []
        for key, filename in file_specs.items():
            url = f"{PIASO_DATA_RAW_URL}/{genome}/{filename}"
            dst = genome_dir / filename
            try:
                urllib.request.urlretrieve(url, str(dst))
                size_kb = dst.stat().st_size / 1024
                if size_kb > 1024:
                    print(f"  {filename}  ({size_kb / 1024:.1f} MB)")
                else:
                    print(f"  {filename}  ({size_kb:.1f} KB)")
            except Exception as e:
                failed.append((filename, url, e))
                print(f"  FAILED: {filename}  →  {e}")

        if gtf_spec is not None:
            try:
                _fetch_and_normalize_gtf(gtf_spec, genome_dir)
            except Exception as e:
                failed.append((gtf_spec["filename"], gtf_spec["url"], e))
                print(f"  FAILED: {gtf_spec['filename']}  →  {e}")

        if failed:
            required_failed = [
                (fn, url, e) for (fn, url, e) in failed
                if not fn.endswith("_transcript_tss.bed")
                and not fn.endswith(".gtf.gz")   # GTF is optional too
            ]
            if required_failed:
                print(
                    f"\n  {len(required_failed)} required file(s) failed to "
                    f"download. You can:\n"
                    f"    1. Retry — transient network errors are common.\n"
                    f"    2. Manually place files in {genome_dir} and "
                    f"re-run resolve_genome_files('{genome}').\n"
                    f"    3. Use piaso.data.fetch_genome('{genome}', "
                    f"source_dir='/path/to/local/{genome}/').\n"
                )
                raise RuntimeError(
                    f"Failed to download {len(required_failed)} required file(s) "
                    f"for genome '{genome}': "
                    f"{', '.join(fn for fn, _, _ in required_failed)}"
                )

    # Verify
    try:
        paths = resolve_genome_files(genome, source=source, release=release)
        print(f"Genome '{genome}' ready at {genome_dir}")
        for key, path in paths.items():
            print(f"  {key}: {path}")
    except FileNotFoundError as e:
        print(f"  WARNING: Some files missing after download: {e}")

    return str(genome_dir)


def _fetch_and_normalize_gtf(gtf_spec: dict, genome_dir: Path) -> Path:
    """Download a GTF from ``gtf_spec`` into ``genome_dir`` and, when
    the source needs it, remap chromosome names to UCSC ``chr1`` style.

    For ``needs_chrom_remap='refseq'``: also download the
    ``_assembly_report.txt`` from the same FTP folder, parse the
    ``RefSeq-Accn ↔ UCSC-style-name`` mapping, and stream-rewrite
    the downloaded GTF in-place (preserving the gzip wrapping).

    For ``needs_chrom_remap='ensembl'``: stream-rewrite prepending
    ``chr`` (with ``MT → chrM``).

    For ``needs_chrom_remap=None`` (GENCODE): no remap.

    Returns the path to the FINAL gtf file (chrom-remapped if
    applicable).
    """
    import urllib.request

    final_path = genome_dir / gtf_spec["filename"]
    raw_path = (
        genome_dir / (gtf_spec["filename"] + ".raw")
        if gtf_spec.get("needs_chrom_remap")
        else final_path
    )
    print(f"  fetching GTF: {gtf_spec['filename']}")
    urllib.request.urlretrieve(gtf_spec["url"], str(raw_path))
    size_mb = raw_path.stat().st_size / 1024 / 1024
    print(f"    downloaded ({size_mb:.1f} MB)")

    remap_kind = gtf_spec.get("needs_chrom_remap")
    if remap_kind == "refseq":
        report_url = gtf_spec.get("assembly_report_url")
        if not report_url:
            raise RuntimeError(
                "RefSeq spec is missing 'assembly_report_url' — "
                "cannot remap chromosome names."
            )
        report_path = genome_dir / "_assembly_report.txt"
        print(f"  fetching assembly report for chrom remap")
        urllib.request.urlretrieve(report_url, str(report_path))
        chrom_map = _parse_refseq_assembly_report(str(report_path))
        print(f"    parsed {len(chrom_map)} chrom mappings")
        stats = _remap_gtf_chroms(str(raw_path), str(final_path), chrom_map)
        print(
            f"  chrom remap: kept {stats['kept']:,} lines, "
            f"dropped {stats['dropped']:,} (unmapped scaffolds)"
        )
        raw_path.unlink()   # remove the un-remapped original
    elif remap_kind == "ensembl":
        stats = _remap_ensembl_gtf(str(raw_path), str(final_path))
        print(
            f"  ensembl chrom remap (prepend 'chr'): "
            f"kept {stats['kept']:,} lines"
        )
        raw_path.unlink()
    # else: GENCODE — already in UCSC style, no remap needed

    return final_path
