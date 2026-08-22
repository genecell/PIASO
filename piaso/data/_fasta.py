"""Genome sequence access for PIASO-GRN — UCSC ``.2bit`` via the optional
``py2bit`` dependency.

Design notes
------------
- **Opt-in / optional.** ``py2bit`` is NOT a core PIASO dependency (most users
  never run the GRN module). It is imported lazily with an actionable error, and
  the genome ``.2bit`` is only downloaded when the user explicitly asks
  (``fetch_2bit`` / ``fetch_genome(..., download_fasta=True)``) — never on import.
- ``.2bit`` (≈780 MB hg38) is preferred over a bgzipped FASTA: smaller, with
  O(1) random access to any interval, which is all the promoter step needs.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

# UCSC golden-path 2bit URLs (used only by the opt-in fetcher).
TWOBIT_URLS = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit",
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.2bit",
}

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _require_py2bit():
    """Lazy import with an install hint (py2bit is an optional GRN dependency)."""
    try:
        import py2bit  # noqa: F401
        return py2bit
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(
            "piaso.data needs the optional 'py2bit' package to read genome "
            "sequence from a .2bit file. Install it with `pip install py2bit` "
            "(or `conda install -c bioconda py2bit`). It is NOT required for the "
            "rest of PIASO."
        ) from exc


def revcomp(seq: str) -> str:
    """Reverse-complement a DNA string (IUPAC ACGTN; case preserved)."""
    return seq.translate(_COMPLEMENT)[::-1]


def _default_cache_dir(dest_dir: Optional[str]) -> str:
    d = dest_dir or os.path.join(os.path.expanduser("~"), ".piaso", "data")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_2bit_path(
    genome: str,
    twobit_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[str]:
    """Return a local ``.2bit`` path for ``genome`` if one exists, else None.

    Search order: explicit ``twobit_path`` → ``<data_dir>/<genome>.2bit`` →
    ``~/.piaso/data/<genome>/<genome>.2bit`` → ``~/.piaso/data/<genome>.2bit``.
    Never downloads (use :func:`fetch_2bit` for that).
    """
    if twobit_path:
        return twobit_path if os.path.exists(twobit_path) else None
    cands = []
    if data_dir:
        cands += [os.path.join(data_dir, f"{genome}.2bit"),
                  os.path.join(data_dir, genome, f"{genome}.2bit")]
    home = os.path.join(os.path.expanduser("~"), ".piaso", "data")
    cands += [os.path.join(home, genome, f"{genome}.2bit"),
              os.path.join(home, f"{genome}.2bit")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def fetch_2bit(genome: str, dest_dir: Optional[str] = None,
               force: bool = False) -> str:
    """Download the UCSC ``.2bit`` for ``genome`` (OPT-IN, ~700-800 MB).

    Returns the local path. No-op if already present (unless ``force``).
    """
    if genome not in TWOBIT_URLS:
        raise ValueError(f"genome {genome!r} not in {sorted(TWOBIT_URLS)}; "
                         "pass an explicit twobit_path instead.")
    d = _default_cache_dir(dest_dir)
    out = os.path.join(d, f"{genome}.2bit")
    if os.path.exists(out) and not force:
        return out
    import urllib.request
    url = TWOBIT_URLS[genome]
    tmp = out + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, out)
    return out


def open_2bit(path: str):
    """Open a ``.2bit`` file (returns a py2bit handle). Caller closes it."""
    py2bit = _require_py2bit()
    if not os.path.exists(path):
        raise FileNotFoundError(f".2bit file not found: {path}")
    return py2bit.open(path)


def extract_sequences(
    twobit_path: str,
    intervals: Sequence[Tuple[str, int, int, str]],
    uppercase: bool = True,
) -> List[str]:
    """Extract sequences for ``intervals = [(chrom, start, end, strand), ...]``.

    0-based half-open coordinates. ``strand == '-'`` returns the reverse
    complement. Out-of-range / missing-chrom intervals yield ``""`` (the caller
    can drop them). Opens the ``.2bit`` once and reuses the handle (RAM = one
    sequence at a time).
    """
    tb = open_2bit(twobit_path)
    try:
        chrom_sizes = tb.chroms()
        out: List[str] = []
        for chrom, start, end, strand in intervals:
            size = chrom_sizes.get(chrom)
            if size is None:
                out.append("")
                continue
            s = max(0, int(start))
            e = min(int(size), int(end))
            if e <= s:
                out.append("")
                continue
            seq = tb.sequence(chrom, s, e)
            if uppercase:
                seq = seq.upper()
            if strand == "-":
                seq = revcomp(seq)
            out.append(seq)
        return out
    finally:
        tb.close()
