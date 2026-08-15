"""ENCODE SCREEN cCRE registry (candidate cis-regulatory elements) for GRN.

Optional wider regulatory search space: in addition to promoters, a gene can
include nearby SCREEN cCREs (enhancer-/promoter-like). cCREs are
**cell-type-agnostic** (the registry, not per-cell-type activity) — this raises
recall at some precision cost and is most powerful with multiome ATAC. Opt-in:
the BED is only downloaded when the user calls :func:`fetch_screen`.

cCRE BED (6 col): ``chrom, start, end, accession, rDHS, class`` where class ∈
{PLS, pELS, dELS, CA-CTCF, CA-TF, CA, CA-H3K4me3, TF, ...}.
"""
from __future__ import annotations

import gzip
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

SCREEN_URLS = {
    "hg38": "https://downloads.wenglab.org/V4/GRCh38-cCREs.bed",
    "mm10": "https://downloads.wenglab.org/V4/mm10-cCREs.bed",
}
# enhancer-/promoter-like classes (good default for distal regulatory search)
ENHANCER_LIKE = ("PLS", "pELS", "dELS")


def _cache_dir(dest_dir):
    d = dest_dir or os.path.join(os.path.expanduser("~"), ".piaso", "data")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_screen_path(genome, screen_bed=None, data_dir=None):
    if screen_bed:
        return screen_bed if os.path.exists(screen_bed) else None
    cands = []
    if data_dir:
        cands.append(os.path.join(data_dir, f"{genome}_cCREs.bed"))
    home = os.path.join(os.path.expanduser("~"), ".piaso", "data")
    cands += [os.path.join(home, genome, f"{genome}_cCREs.bed"),
              os.path.join(home, f"{genome}_cCREs.bed")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def fetch_screen(genome, dest_dir=None, force=False) -> str:
    """Download the SCREEN cCRE BED for ``genome`` (OPT-IN). Returns the path."""
    if genome not in SCREEN_URLS:
        raise ValueError(f"genome {genome!r} not in {sorted(SCREEN_URLS)}.")
    d = _cache_dir(dest_dir)
    out = os.path.join(d, f"{genome}_cCREs.bed")
    if os.path.exists(out) and not force:
        return out
    import urllib.request
    tmp = out + ".part"
    urllib.request.urlretrieve(SCREEN_URLS[genome], tmp)
    os.replace(tmp, out)
    return out


def load_screen_ccres(path: str, classes: Optional[Sequence[str]] = ENHANCER_LIKE) -> Dict[str, dict]:
    """Load cCREs into ``{chrom: {'starts': sorted ndarray, 'ends': ndarray}}``.

    ``classes`` filters the 6th-column class (substring match, e.g. ``'pELS'``
    matches; ``None`` keeps all)."""
    opn = gzip.open if path.endswith(".gz") else open
    byc: Dict[str, List] = {}
    cls = tuple(classes) if classes else None
    with opn(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            if cls is not None and len(f) >= 6 and not any(c in f[5] for c in cls):
                continue
            byc.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    out = {}
    for c, ivs in byc.items():
        ivs.sort()
        out[c] = {"starts": np.array([s for s, _ in ivs]),
                  "ends": np.array([e for _, e in ivs])}
    return out


def ccres_near_tss(ccres: Dict[str, dict], chrom: str, tss: int, window: int):
    """Return [(start, end)] cCREs on ``chrom`` within ``±window`` of ``tss``
    (binary search on sorted starts)."""
    c = ccres.get(chrom)
    if c is None:
        return []
    lo, hi = tss - window, tss + window
    starts = c["starts"]
    # cCREs overlapping [lo, hi]: start <= hi AND end >= lo
    import bisect
    j = bisect.bisect_right(starts, hi)
    res = []
    for i in range(j - 1, -1, -1):
        if c["ends"][i] < lo:
            if starts[i] < lo - 1_000_000:  # far past the window; stop
                break
            continue
        res.append((int(starts[i]), int(c["ends"][i])))
    return res
