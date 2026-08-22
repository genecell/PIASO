"""Motif-database and TF-list loaders for the PIASO-GRN feature.

All loaders produce ``PWM`` objects (shared type from ``_pwm``); no third-party
DB-specific libraries are required — only NumPy and the stdlib.

Public API
----------
load_jaspar_meme        Parse a JASPAR MEME-format file → list[PWM]
load_cisbp              Parse a CIS-BP PWM directory (+ optional TF_Information.txt) → list[PWM]
load_tf_list            Return a set of TF gene symbols from a file or a supported DB
fetch_animaltfdb_tf_list  Download AnimalTFDB 4.0 TF list for human/mouse (opt-in)
build_tf_motif_map      Group PWMs by TF, optionally restricting to a TF/gene universe
harmonize_symbol        Case-insensitive symbol lookup against a precomputed universe dict
"""
from __future__ import annotations

import csv
import io
import os
import pathlib
import re


def _clean_meme_tf(token: str) -> str:
    """Extract the TF symbol from a MEME ``MOTIF`` name token.

    Handles CIS-BP's homology-inferred naming ``(TF)_(species)_(DBD_x)`` — e.g.
    ``(Ascl1)_(Homo_sapiens)_(DBD_1.00)`` → ``Ascl1`` — which would otherwise be
    taken verbatim as ``tf_name`` and never match a gene symbol (hiding ~40% of
    mouse CIS-BP motifs from the GRN). Plain names (JASPAR ``PAX6``, CIS-BP
    ``Tfap2a``) pass through unchanged.
    """
    if token.startswith("("):
        m = re.match(r"\(([A-Za-z0-9._/-]+)\)", token)
        if m:
            return m.group(1)
    return token
from typing import Optional

import numpy as np

from ._pwm import PWM, BASES

__all__ = [
    "load_meme",
    "load_jaspar_meme",
    "load_cisbp_meme",
    "load_cisbp",
    "load_tf_list",
    "fetch_jaspar",
    "resolve_jaspar_path",
    "fetch_cisbp",
    "resolve_cisbp_meme_path",
    "fetch_cistarget_motifs",
    "load_cistarget_motifs",
    "resolve_cistarget_paths",
    "write_meme",
    "fetch_animaltfdb_tf_list",
    "build_tf_motif_map",
    "harmonize_symbol",
]

# ---------------------------------------------------------------------------
# AnimalTFDB 4.0 download URLs (opt-in; used only inside fetch_animaltfdb_tf_list)
# ---------------------------------------------------------------------------
_ANIMALTFDB_URLS: dict[str, list[str]] = {
    "human": [
        "https://guolab.wchscu.cn/AnimalTFDB4_static/download/TF_list_final/Homo_sapiens_TF",
        "https://guolab.wchscu.cn/AnimalTFDB4/static/download/TF_list_final/Homo_sapiens_TF",
    ],
    "mouse": [
        "https://guolab.wchscu.cn/AnimalTFDB4_static/download/TF_list_final/Mus_musculus_TF",
        "https://guolab.wchscu.cn/AnimalTFDB4/static/download/TF_list_final/Mus_musculus_TF",
    ],
}

_DEFAULT_CACHE_DIR = pathlib.Path.home() / ".piaso" / "grn"


# ---------------------------------------------------------------------------
# 1. JASPAR MEME loader
# ---------------------------------------------------------------------------

def load_meme(path: str, source: str = "meme") -> list[PWM]:
    """Parse a MEME-format motif file (JASPAR, CIS-BP-from-MEME, …) → list[PWM].

    Pure-Python parser (NumPy + stdlib only — no MEME package required). Used by
    both :func:`load_jaspar_meme` (``source='jaspar'``) and
    :func:`load_cisbp_meme` (``source='cisbp'``).

    Parameters
    ----------
    path : str
        Path to a MEME 4 file (e.g. ``JASPAR2024_CORE_vertebrates.meme`` or
        ``CIS-BP_2.00/Homo_sapiens.meme`` from the MEME Suite bundle).
    source : str
        Provenance label stored on each returned ``PWM`` (``'jaspar'`` / ``'cisbp'``).

    Returns
    -------
    list[PWM]
        One ``PWM`` per ``MOTIF`` block. The parser is robust to blank lines,
        ``URL`` lines and varying whitespace; ``MOTIF <id>`` with only two tokens
        sets ``tf_name = motif_id``, ``MOTIF <id> <name>`` uses the name.
    """
    path = str(path)
    pwms: list[PWM] = []

    motif_id: Optional[str] = None
    tf_name: Optional[str] = None
    width: Optional[int] = None
    rows: list[list[float]] = []
    inside_matrix = False

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # ---- New motif header ----------------------------------------
            if line.startswith("MOTIF"):
                # Flush previous motif if we were collecting one
                if motif_id is not None and rows:
                    pwms.append(_build_pwm_from_rows(motif_id, tf_name, rows, source))
                tokens = line.split()
                motif_id = tokens[1]
                tf_name = _clean_meme_tf(tokens[2]) if len(tokens) >= 3 else motif_id
                width = None
                rows = []
                inside_matrix = False
                continue

            # ---- Matrix header (letter-probability matrix: alength= 4 w= N ...) -
            if line.startswith("letter-probability matrix"):
                inside_matrix = True
                # Extract w= value for validation (optional — we trust row count)
                width = _parse_meme_width(line)
                continue

            # ---- Skip URL / comment / empty lines --------------------------
            if not line or line.startswith("URL") or line.startswith("#"):
                if inside_matrix:
                    # A blank line signals the end of the probability block
                    inside_matrix = False
                continue

            # ---- Skip global header blocks (ALPHABET, strands, Background) -
            if (
                line.startswith("MEME version")
                or line.startswith("ALPHABET")
                or line.startswith("strands")
                or line.startswith("Background")
            ):
                inside_matrix = False
                continue

            # ---- Probability row (inside a matrix block) -------------------
            if inside_matrix:
                parts = line.split()
                # A valid probability row has exactly 4 floats
                try:
                    vals = [float(x) for x in parts]
                    if len(vals) == 4:
                        rows.append(vals)
                    else:
                        # Might be a non-row line inside the block; stop collecting
                        inside_matrix = False
                except ValueError:
                    # Non-numeric line — exit matrix mode
                    inside_matrix = False

    # Flush last motif
    if motif_id is not None and rows:
        pwms.append(_build_pwm_from_rows(motif_id, tf_name, rows, source))

    return pwms


def _parse_meme_width(line: str) -> Optional[int]:
    """Extract ``w=N`` from a MEME letter-probability matrix header line."""
    for token in line.split():
        if token.startswith("w="):
            try:
                return int(token[2:])
            except ValueError:
                pass
    # Sometimes formatted as "w= N" (two tokens)
    parts = line.split()
    for i, tok in enumerate(parts):
        if tok == "w=" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _build_pwm_from_rows(
    motif_id: str,
    tf_name: Optional[str],
    rows: list[list[float]],
    source: str,
) -> PWM:
    """Build a PWM from a list of probability rows (each row = one position, A C G T)."""
    arr = np.array(rows, dtype=np.float32)  # shape (w, 4)
    probs = arr.T  # shape (4, w)
    return PWM(
        motif_id=motif_id,
        tf_name=tf_name if tf_name else motif_id,
        probs=probs,
        source=source,
    )


def load_jaspar_meme(path: str) -> list[PWM]:
    """Parse a JASPAR MEME-format file → list[PWM] (``source='jaspar'``).

    Thin wrapper over :func:`load_meme`. Pair with :func:`fetch_jaspar`.
    """
    return load_meme(path, source="jaspar")


def load_cisbp_meme(path: str) -> list[PWM]:
    """Parse a CIS-BP MEME file (MEME Suite bundle) → list[PWM] (``source='cisbp'``).

    For the MEME-format CIS-BP file produced by :func:`fetch_cisbp`
    (e.g. ``CIS-BP_2.00/Homo_sapiens.meme``). The raw per-motif CIS-BP layout
    (``<motif_id>.txt`` + ``TF_Information.txt``) is handled by :func:`load_cisbp`.
    """
    return load_meme(path, source="cisbp")


# ---------------------------------------------------------------------------
# Motif-database fetchers (OPT-IN downloads → ~/.piaso/data/motifs cache)
# ---------------------------------------------------------------------------

# JASPAR MEME files (jaspar.genereg.net redirects → use the elixir.no mirror).
_JASPAR_BASE = "https://jaspar.elixir.no/download/data"
# MEME Suite motif database bundle (contains CIS-BP_<ver>/<Species>.meme, among others).
MEME_MOTIF_DB_VERSION = "12.27"
_MEME_MOTIF_DB_URL = (
    "https://meme-suite.org/meme/meme-software/Databases/motifs/"
    f"motif_databases.{MEME_MOTIF_DB_VERSION}.tgz"
)
# genome → CIS-BP/MEME species name convenience
_GENOME_TO_SPECIES = {"hg38": "Homo_sapiens", "hg19": "Homo_sapiens",
                      "mm10": "Mus_musculus", "mm9": "Mus_musculus"}


def _motif_cache_dir(dest_dir=None) -> str:
    d = dest_dir or os.path.join(os.path.expanduser("~"), ".piaso", "data", "motifs")
    os.makedirs(d, exist_ok=True)
    return d


def _jaspar_filename(release, collection, taxon) -> str:
    return f"{release}_{collection}_{taxon}.meme"


def resolve_jaspar_path(release="JASPAR2024", collection="CORE", taxon="vertebrates",
                        jaspar_path=None, dest_dir=None):
    """Return a usable JASPAR MEME path (explicit ``jaspar_path`` or cached), else None."""
    if jaspar_path:
        return jaspar_path if os.path.exists(jaspar_path) else None
    cand = os.path.join(_motif_cache_dir(dest_dir),
                        _jaspar_filename(release, collection, taxon))
    return cand if os.path.exists(cand) else None


def fetch_jaspar(release="JASPAR2024", collection="CORE", taxon="vertebrates",
                 dest_dir=None, force=False) -> str:
    """Download a JASPAR MEME motif file (OPT-IN). Returns the cached path.

    Parameters
    ----------
    release : str
        e.g. ``'JASPAR2024'`` (the year ``2024`` is parsed from the trailing digits).
    collection : str
        e.g. ``'CORE'``, ``'UNVALIDATED'``.
    taxon : str
        e.g. ``'vertebrates'``, ``'plants'``, ``'insects'``, ``'fungi'``,
        ``'nematodes'``, ``'urochordates'``.
    """
    year = "".join(ch for ch in release if ch.isdigit()) or "2024"
    url = (f"{_JASPAR_BASE}/{year}/{collection}/"
           f"{release}_{collection}_{taxon}_non-redundant_pfms_meme.txt")
    out = os.path.join(_motif_cache_dir(dest_dir),
                       _jaspar_filename(release, collection, taxon))
    if os.path.exists(out) and not force:
        return out
    import urllib.request
    tmp = out + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, out)
    return out


def _cisbp_meme_filename(species, version) -> str:
    return f"CIS-BP_{version}_{species}.meme"


def resolve_cisbp_meme_path(species="Homo_sapiens", version="2.00",
                            cisbp_meme_path=None, dest_dir=None, genome=None):
    """Return a usable CIS-BP MEME path (explicit, or cached), else None."""
    if cisbp_meme_path:
        return cisbp_meme_path if os.path.exists(cisbp_meme_path) else None
    if genome:
        species = _GENOME_TO_SPECIES.get(genome, species)
    cand = os.path.join(_motif_cache_dir(dest_dir),
                        _cisbp_meme_filename(species, version))
    return cand if os.path.exists(cand) else None


def fetch_cisbp(species="Homo_sapiens", version="2.00", dest_dir=None,
                force=False, genome=None) -> str:
    """Download CIS-BP motifs (MEME format) from the MEME Suite bundle (OPT-IN).

    The MEME Suite motif-database bundle ships per-species single-MEME CIS-BP files
    (``CIS-BP_<version>/<Species>.meme``; one motif per TF, redundancy-reduced).
    This streams the ``.tgz`` and extracts only the requested file (the CIS-BP
    entries sit near the start of the archive, so the whole bundle is not read).

    Parameters
    ----------
    species : str
        ``'Homo_sapiens'`` / ``'Mus_musculus'`` (MEME naming). Or pass ``genome``
        (``'hg38'`` → Homo_sapiens, ``'mm10'`` → Mus_musculus).
    version : str
        CIS-BP version inside the bundle (default ``'2.00'``; ``'1.02'`` available).
    """
    if genome:
        species = _GENOME_TO_SPECIES.get(genome, species)
    out = os.path.join(_motif_cache_dir(dest_dir),
                       _cisbp_meme_filename(species, version))
    if os.path.exists(out) and not force:
        return out
    member = f"motif_databases/CIS-BP_{version}/{species}.meme"
    import urllib.request
    import tarfile
    tmp = out + ".part"
    # Stream the gzip tar; iterate members and extract only the target, then stop.
    with urllib.request.urlopen(_MEME_MOTIF_DB_URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for m in tf:
                if m.name == member:
                    src = tf.extractfile(m)
                    if src is None:
                        raise IOError(f"could not read {member} from MEME bundle")
                    with open(tmp, "wb") as fh:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            fh.write(chunk)
                    break
            else:
                raise ValueError(
                    f"{member!r} not found in MEME bundle {_MEME_MOTIF_DB_URL}. "
                    f"Check species/version (have CIS-BP 2.00 / 1.02; "
                    f"Homo_sapiens / Mus_musculus / …).")
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------
# 2. CIS-BP loader
# ---------------------------------------------------------------------------

def load_cisbp(
    pwm_dir: str,
    tf_info_path: Optional[str] = None,
) -> list[PWM]:
    """Load motifs from a CIS-BP PWM directory.

    Parameters
    ----------
    pwm_dir : str
        Directory containing per-motif files named ``<motif_id>.txt``.  Each
        file has a tab-separated header row ``Pos\\tA\\tC\\tG\\tT`` followed by
        one row per position.
    tf_info_path : str or None
        Optional path to ``TF_Information.txt`` (tab-separated, columns include
        ``Motif_ID`` and ``TF_Name``).  When None or the file is missing, the
        motif filename stem is used as ``tf_name``.

    Returns
    -------
    list[PWM]
        One ``PWM`` per successfully parsed file.  ``source`` is ``"cisbp"``.

    Raises
    ------
    FileNotFoundError
        If ``pwm_dir`` does not exist.
    """
    pwm_dir_path = pathlib.Path(pwm_dir)
    if not pwm_dir_path.exists():
        raise FileNotFoundError(
            f"CIS-BP PWM directory not found: {pwm_dir!r}. "
            "Please download CIS-BP motifs and pass the directory path."
        )

    # Build motif_id → TF_name map from TF_Information.txt if provided
    tf_map: dict[str, str] = {}
    if tf_info_path is not None:
        tf_map = _load_cisbp_tf_info(tf_info_path)

    pwms: list[PWM] = []
    for txt_file in sorted(pwm_dir_path.glob("*.txt")):
        motif_id = txt_file.stem
        tf_name = tf_map.get(motif_id, motif_id)
        pwm = _parse_cisbp_pwm_file(txt_file, motif_id, tf_name)
        if pwm is not None:
            pwms.append(pwm)

    return pwms


def _load_cisbp_tf_info(path: str) -> dict[str, str]:
    """Parse ``TF_Information.txt`` and return ``{Motif_ID: TF_Name}``."""
    mapping: dict[str, str] = {}
    path = str(path)
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        # Normalise header names (CIS-BP versions differ in capitalisation)
        for row in reader:
            norm = {k.strip(): v.strip() for k, v in row.items() if k}
            motif_id = norm.get("Motif_ID") or norm.get("motif_id") or norm.get("MotifID", "")
            tf_name = (
                norm.get("TF_Name")
                or norm.get("tf_name")
                or norm.get("TFName")
                or norm.get("Gene_Name")
                or ""
            )
            if motif_id and tf_name:
                mapping[motif_id] = tf_name
    return mapping


def _parse_cisbp_pwm_file(
    path: pathlib.Path,
    motif_id: str,
    tf_name: str,
) -> Optional[PWM]:
    """Parse a single CIS-BP PWM file (best-effort; returns None on failure)."""
    rows: list[list[float]] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header_seen = False
            for row in reader:
                # Skip blank rows
                if not any(cell.strip() for cell in row):
                    continue
                # The first non-blank row is the header (Pos A C G T ...)
                if not header_seen:
                    header_seen = True
                    # Validate it looks like a header
                    upper = [c.strip().upper() for c in row]
                    if "A" in upper and "C" in upper and "G" in upper and "T" in upper:
                        continue  # header confirmed; skip
                    # Otherwise treat as a data row (some files omit headers)
                # Data row: first col = Pos (int), next 4 = A C G T
                cells = [c.strip() for c in row]
                try:
                    vals = [float(c) for c in cells if c]
                    # Accept either 4 (no Pos) or 5 (Pos + 4) values
                    if len(vals) == 5:
                        rows.append(vals[1:5])
                    elif len(vals) == 4:
                        rows.append(vals[:4])
                except ValueError:
                    continue  # malformed row — skip silently
    except OSError:
        return None

    if not rows:
        return None

    arr = np.array(rows, dtype=np.float32)  # (w, 4)
    probs = arr.T  # (4, w)
    return PWM(motif_id=motif_id, tf_name=tf_name, probs=probs, source="cisbp")


# ---------------------------------------------------------------------------
# 3. TF-list loader
# ---------------------------------------------------------------------------

def load_tf_list(
    species: str = "human",
    source: str = "animaltfdb",
    path: Optional[str] = None,
) -> Optional[set[str]]:
    """Return a set of TF gene symbols.

    Parameters
    ----------
    species : str
        Species identifier used with ``source="animaltfdb"`` (``"human"`` or
        ``"mouse"``).  Ignored when ``path`` is given.
    source : str
        ``"animaltfdb"`` (default) — AnimalTFDB 4.0 TF catalogue.
        ``"motifdb"`` — sentinel meaning "use the motif DB's own TF names";
        returns ``None`` and ``build_tf_motif_map`` interprets that as no
        restriction.
    path : str or None
        Path to a local TF list.  Accepts either a one-symbol-per-line plain
        text file OR a TSV with a ``Symbol`` / ``TF`` / ``gene`` header column.
        When provided, ``species`` and ``source`` are ignored.

    Returns
    -------
    set[str] or None
        Set of gene symbols, or ``None`` when ``source="motifdb"``.

    Raises
    ------
    ValueError
        For an unsupported ``source``.
    FileNotFoundError
        If a local ``path`` is specified but does not exist.
    RuntimeError
        When ``source="animaltfdb"``, no ``path`` is given, and no cached file
        exists — instructs the caller to use :func:`fetch_animaltfdb_tf_list`.
    """
    # sentinel — caller wants to use motif-DB TF names without restriction
    if source == "motifdb":
        return None

    # --- User-supplied file takes precedence ---------------------------------
    if path is not None:
        return _read_tf_list_file(path)

    # --- AnimalTFDB: require explicit opt-in (no auto-download) --------------
    if source == "animaltfdb":
        species_lower = species.lower()
        if species_lower not in _ANIMALTFDB_URLS:
            raise ValueError(
                f"Unsupported species {species!r} for AnimalTFDB. "
                f"Supported: {sorted(_ANIMALTFDB_URLS.keys())}"
            )
        cached = _animaltfdb_cache_path(species_lower)
        if cached.exists():
            return _read_tf_list_file(str(cached))
        raise RuntimeError(
            f"No cached AnimalTFDB TF list found for species={species!r}.\n"
            "To download it, call:\n"
            f"    piaso.data.fetch_animaltfdb_tf_list({species!r})\n"
            "Then pass the returned path to load_tf_list(path=...).\n"
            "Alternatively, supply any symbol-per-line or TSV file via path=."
        )

    raise ValueError(
        f"Unknown source {source!r}. Supported: 'animaltfdb', 'motifdb'."
    )


def _read_tf_list_file(path: str) -> set[str]:
    """Read a TF symbol list from a plain-text or TSV file."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"TF list file not found: {path!r}")

    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline()

    # Sniff: is it a TSV with a recognised symbol column?
    sep = "\t" if "\t" in first_line else None
    if sep is not None:
        headers = [h.strip() for h in first_line.split(sep)]
        _known_cols = {"Symbol", "TF", "gene", "Gene", "TF_Name", "symbol", "gene_name"}
        col_name = next((h for h in headers if h in _known_cols), None)
        if col_name is not None:
            symbols: set[str] = set()
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    sym = (row.get(col_name) or "").strip()
                    if sym:
                        symbols.add(sym)
            return symbols

    # Otherwise treat as one-symbol-per-line (skip blank lines / comment lines)
    symbols = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            sym = line.strip()
            if sym and not sym.startswith("#"):
                symbols.add(sym)
    return symbols


# ---------------------------------------------------------------------------
# 4. AnimalTFDB fetcher (opt-in)
# ---------------------------------------------------------------------------

def fetch_animaltfdb_tf_list(
    species: str,
    dest_dir: Optional[str] = None,
) -> str:
    """Download the AnimalTFDB 4.0 TF list for *species* and cache it locally.

    Parameters
    ----------
    species : str
        ``"human"`` or ``"mouse"``.
    dest_dir : str or None
        Directory to save the file.  Defaults to ``~/.piaso/grn/``.

    Returns
    -------
    str
        Absolute path to the downloaded file.

    Raises
    ------
    ValueError
        Unsupported species.
    RuntimeError
        Download failed from all candidate URLs.

    Notes
    -----
    This function makes a network request.  It is intentionally **not** called
    automatically by :func:`load_tf_list` — the caller must invoke it
    explicitly so that downstream pipelines have reproducible, network-free
    behaviour after the first run.
    """
    import urllib.request
    import urllib.error

    species_lower = species.lower()
    if species_lower not in _ANIMALTFDB_URLS:
        raise ValueError(
            f"Unsupported species {species!r}. "
            f"Supported: {sorted(_ANIMALTFDB_URLS.keys())}"
        )

    dest_path = _animaltfdb_cache_path(species_lower, dest_dir=dest_dir)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for url in _ANIMALTFDB_URLS[species_lower]:
        try:
            urllib.request.urlretrieve(url, str(dest_path))
            return str(dest_path)
        except urllib.error.URLError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to download AnimalTFDB TF list for species={species!r}.\n"
        f"Last error: {last_error}\n"
        "Please download the file manually and pass its path to "
        "load_tf_list(path=...).\n"
        "Expected format: one TF symbol per line, or a TSV with a 'Symbol' column."
    )


def _animaltfdb_cache_path(species: str, dest_dir: Optional[str] = None) -> pathlib.Path:
    base = pathlib.Path(dest_dir) if dest_dir else _DEFAULT_CACHE_DIR
    stem_map = {"human": "Homo_sapiens_TF", "mouse": "Mus_musculus_TF"}
    return base / stem_map[species]


# ---------------------------------------------------------------------------
# 5. build_tf_motif_map
# ---------------------------------------------------------------------------

def build_tf_motif_map(
    pwms: list[PWM],
    tf_list: Optional[set[str]] = None,
    gene_universe: Optional[list[str]] = None,
) -> dict[str, list[PWM]]:
    """Group PWMs by TF symbol, optionally restricting to known TFs and the RNA universe.

    Parameters
    ----------
    pwms : list[PWM]
        Source PWMs (from :func:`load_jaspar_meme` or :func:`load_cisbp`).
    tf_list : set[str] or None
        If given, only PWMs whose ``tf_name`` matches a symbol in *tf_list*
        (case-insensitive) are retained.
    gene_universe : list[str] or None
        The RNA ``var_names`` (exact-case gene symbols in the expression matrix).
        When provided, further restricts to TFs present in the universe **and**
        remaps each retained PWM's ``tf_name`` to the exact-case symbol used in
        the expression matrix.  Handles human (ALL-CAPS) and mouse (Title-case)
        convention differences via case-insensitive matching.

    Returns
    -------
    dict[str, list[PWM]]
        ``{tf_symbol: [PWM, ...]}`` where ``tf_symbol`` is the exact-case key
        from *gene_universe* (when provided) or the original ``tf_name`` / the
        *tf_list* symbol.

    Notes
    -----
    PWMs with ``tf_name`` matching multiple *gene_universe* entries (e.g. a DB
    uses ``"STAT3"`` but the universe has both ``"Stat3"`` and ``"STAT3"``) are
    mapped to the first (alphabetically sorted) match to remain deterministic.
    """
    # Build case-insensitive lookup dicts
    tf_upper: dict[str, str] = {}  # {TF_NAME_UPPER: exact_case_in_tf_list}
    if tf_list is not None:
        for sym in tf_list:
            tf_upper[sym.upper()] = sym

    gene_upper: dict[str, str] = {}  # {GENE_UPPER: exact_case_in_universe}
    if gene_universe is not None:
        for sym in gene_universe:
            key = sym.upper()
            # Keep alphabetically-first in case of collision (e.g. Stat3 vs STAT3)
            if key not in gene_upper or sym < gene_upper[key]:
                gene_upper[key] = sym

    result: dict[str, list[PWM]] = {}

    for pwm in pwms:
        raw = pwm.tf_name
        raw_upper = raw.upper()

        # --- tf_list filter (case-insensitive) --------------------------------
        if tf_upper:
            if raw_upper not in tf_upper:
                continue
            canonical = tf_upper[raw_upper]  # use the tf_list casing
        else:
            canonical = raw

        # --- gene_universe filter + remap ------------------------------------
        if gene_upper:
            matched = harmonize_symbol(canonical, gene_upper)
            if matched is None:
                continue
            canonical = matched  # remap to expression-matrix casing

        # --- Build output map ------------------------------------------------
        # Attach a shallow-copy of the PWM with remapped tf_name so downstream
        # lookups (e.g. COSG score by gene symbol) match without further effort.
        if canonical != pwm.tf_name:
            remapped = PWM(
                motif_id=pwm.motif_id,
                tf_name=canonical,
                probs=pwm.probs,
                source=pwm.source,
                meta=dict(pwm.meta),
            )
        else:
            remapped = pwm

        result.setdefault(canonical, []).append(remapped)

    return result


# ---------------------------------------------------------------------------
# 6. harmonize_symbol (helper)
# ---------------------------------------------------------------------------

def harmonize_symbol(sym: str, universe_lookup: dict) -> Optional[str]:
    """Case-insensitive lookup of *sym* against a precomputed universe dict.

    Parameters
    ----------
    sym : str
        Query symbol (any casing).
    universe_lookup : dict
        Mapping ``{SYM_UPPER: exact_case_sym}`` — typically the output of
        ``{s.upper(): s for s in gene_universe}``.

    Returns
    -------
    str or None
        The exact-case symbol from *universe_lookup*, or ``None`` if not found.

    Examples
    --------
    >>> lu = {s.upper(): s for s in ["Ctcf", "Gata1", "Actb"]}
    >>> harmonize_symbol("CTCF", lu)
    'Ctcf'
    >>> harmonize_symbol("ctcf", lu)
    'Ctcf'
    >>> harmonize_symbol("TP53", lu)
    None
    """
    return universe_lookup.get(sym.upper())


# ---------------------------------------------------------------------------
# cisTarget / aertslab motif collection (the motifs SCENIC+ uses)
# ---------------------------------------------------------------------------
# The aertslab cisTarget motif collection (e.g. ``v10nr_clust``) is a clustered,
# non-redundant union of motifs from many source DBs, distributed as cluster-buster
# ``.cb`` singletons + a motif->TF annotation table. SCENIC+ pairs these PWMs with
# precomputed cisTarget *ranking* databases (multi-GB); inferGRN does its own PWM scan,
# so we need ONLY the PWMs + the motif2TF table — far lighter.
_CISTARGET_BASE = "https://resources.aertslab.org/cistarget"
_CISTARGET_MOTIF2TF = {  # (species) -> motif2tf filename
    "Homo_sapiens": "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
    "Mus_musculus": "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl",
}


def _parse_cb(path: str):
    """Parse a cluster-buster ``.cb`` motif file -> (motif_id, probs (4, w))."""
    motif_id = None
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                motif_id = line[1:].strip()
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    rows.append([float(x) for x in parts[:4]])
                except ValueError:
                    continue
    if not rows:
        return motif_id, None
    arr = np.asarray(rows, dtype=np.float32) + 1e-3        # (w, 4) counts + pseudocount
    arr = arr / arr.sum(1, keepdims=True)                  # row-normalize to probabilities
    return motif_id, arr.T                                 # (4, w)


def resolve_cistarget_paths(species="Homo_sapiens", version="v10nr_clust",
                            dest_dir=None, genome=None):
    """Return (cb_singletons_dir, motif2tf_tbl_path) if cached, else (None, None)."""
    if genome:
        species = _GENOME_TO_SPECIES.get(genome, species)
    d = os.path.join(dest_dir or _motif_cache_dir(), "cistarget")
    cb = os.path.join(d, f"{version}_public", "singletons")
    tbl = os.path.join(d, _CISTARGET_MOTIF2TF.get(species, ""))
    return (cb if os.path.isdir(cb) else None, tbl if os.path.exists(tbl) else None)


def fetch_cistarget_motifs(species="Homo_sapiens", version="v10nr_clust",
                           dest_dir=None, force=False, genome=None):
    """Download the aertslab cisTarget motif collection (PWMs + motif2TF) — OPT-IN.

    Downloads the ``<version>_public.zip`` cluster-buster singletons (~85 MB) and the
    species motif->TF annotation table into ``~/.piaso/data/motifs/cistarget``. Returns
    ``(cb_singletons_dir, motif2tf_tbl_path)``.
    """
    import urllib.request
    import zipfile
    if genome:
        species = _GENOME_TO_SPECIES.get(genome, species)
    d = os.path.join(dest_dir or _motif_cache_dir(), "cistarget")
    os.makedirs(d, exist_ok=True)
    cb_root = os.path.join(d, f"{version}_public", "singletons")
    if force or not os.path.isdir(cb_root):
        zpath = os.path.join(d, f"{version}_public.zip")
        if force or not os.path.exists(zpath):
            url = f"{_CISTARGET_BASE}/motif_collections/{version}_public/{version}_public.zip"
            urllib.request.urlretrieve(url, zpath + ".part"); os.replace(zpath + ".part", zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(d)
    tbl_name = _CISTARGET_MOTIF2TF.get(species)
    if tbl_name is None:
        raise ValueError(f"no cisTarget motif2tf table for species {species!r}")
    tbl = os.path.join(d, tbl_name)
    if force or not os.path.exists(tbl):
        url = f"{_CISTARGET_BASE}/motif2tf/{tbl_name}"
        urllib.request.urlretrieve(url, tbl + ".part"); os.replace(tbl + ".part", tbl)
    return cb_root, tbl


def load_cistarget_motifs(cb_dir, motif2tf_path, tf_universe=None,
                          max_per_tf=None, direct_only=False):
    """Load the cisTarget collection -> list[PWM] with ``tf_name`` from the motif2TF table.

    Parameters
    ----------
    cb_dir : str
        Directory of cluster-buster ``<motif_id>.cb`` singletons.
    motif2tf_path : str
        Path to a ``motifs-...-nr.<species>-m...-o....tbl`` annotation (maps motif_id ->
        gene_name/TF, with similarity q-value + a 'directly annotated' vs orthology note).
    tf_universe : set[str] or None
        Restrict to these TF symbols (upper-cased) — pass the dataset's expressed TFs to
        keep the scan small.
    max_per_tf : int or None
        Keep at most this many motifs per TF (best motif-similarity q-value first). ``1``
        gives one representative motif per TF (comparable in size to JASPAR/CIS-BP).
    direct_only : bool
        Keep only motifs *directly* annotated to the TF (drop orthology-extended rows).
    """
    tfu = {t.upper() for t in tf_universe} if tf_universe else None
    # motif_id -> list of (tf, qvalue)
    m2tf: dict[str, list] = {}
    with open(motif2tf_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        idcol = "#motif_id" if "#motif_id" in (reader.fieldnames or []) else "motif_id"
        for row in reader:
            tf = (row.get("gene_name") or "").strip()
            if not tf or tf == "None":
                continue
            if direct_only and "directly annotated" not in (row.get("description") or ""):
                continue
            if tfu and tf.upper() not in tfu:
                continue
            try:
                q = float(row.get("motif_similarity_qvalue") or 0.0)
            except ValueError:
                q = 0.0
            m2tf.setdefault(row[idcol], []).append((tf.upper(), q))
    pwms: list[PWM] = []
    cache: dict[str, object] = {}
    for mid, tfs in m2tf.items():
        cb = os.path.join(cb_dir, mid + ".cb")
        if not os.path.exists(cb):
            continue
        if mid not in cache:
            _, probs = _parse_cb(cb)
            cache[mid] = probs
        probs = cache[mid]
        if probs is None:
            continue
        best: dict[str, float] = {}
        for tf, q in tfs:
            if tf not in best or q < best[tf]:
                best[tf] = q
        for tf in best:
            pwms.append(PWM(motif_id=mid, tf_name=tf, probs=probs, source="cistarget"))
    if max_per_tf:
        from collections import defaultdict
        bytf: dict[str, list] = defaultdict(list)
        for p in pwms:
            bytf[p.tf_name].append(p)
        kept = []
        for tf, ps in bytf.items():
            kept.extend(ps[:max_per_tf])     # already best-q-first within a motif; cap count
        pwms = kept
    return pwms


def write_meme(pwms, path: str, bg=(0.25, 0.25, 0.25, 0.25)):
    """Write a list[PWM] to a minimal MEME-format file (``MOTIF <id>_<tf> <tf>`` headers,
    so :func:`load_meme` recovers ``tf_name`` from the 3rd token). Lets any PWM collection
    (e.g. cisTarget) drive inferGRN's existing ``jaspar_path=`` (MEME) interface."""
    with open(path, "w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        fh.write("Background letter frequencies\n")
        fh.write("A %.3f C %.3f G %.3f T %.3f\n\n" % tuple(bg))
        for i, p in enumerate(pwms):
            probs = np.asarray(p.probs)                 # (4, w)
            w = probs.shape[1]
            name = f"{p.motif_id}__{p.tf_name}_{i}"
            fh.write(f"MOTIF {name} {p.tf_name}\n")
            fh.write(f"letter-probability matrix: alength= 4 w= {w}\n")
            for j in range(w):
                col = probs[:, j]
                fh.write(" %.6f %.6f %.6f %.6f\n" % (col[0], col[1], col[2], col[3]))
            fh.write("\n")
    return path
