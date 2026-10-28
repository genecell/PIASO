"""Pure-NumPy PWM motif scanner — reference / fallback implementation.

This module is the correctness reference that the Rust backend (``_piaso.scan_motifs``)
will later mirror. It is designed to be RAM-efficient: only one PSSM and the current
sequence's sliding window view are live at a time inside the hot loops.

Design choices
--------------
N-handling
    A code == 4 means "ambiguous / unknown base" (N, or any non-ACGT character).
    For each window position j where code == 4 we contribute the **column minimum**
    of the PSSM.  This is the most conservative choice: N-containing windows almost
    never reach the score threshold (same philosophy as FIMO's hard "N = skip column"
    but retains a defined numeric score so the DP below stays simple).  A window
    that is entirely non-degenerate sees no penalty.

DP quantization (pvalue_to_threshold)
    We follow the classic MOODS / FIMO algorithm: score columns are quantized to
    integer bin indices, then per-column distributions are convolved by shifting
    and adding probability vectors.  The grid resolution is
    ``(total_score_range) / n_bins`` bits per bin; each column's score is rounded
    to the nearest bin, so a word's quantized score is within ``w / 2`` bins of
    its exact score and the threshold is approximate to that resolution. The
    distribution is sized to hold every quantized score (rounding can carry the
    best words past ``n_bins``), so no probability is lost at the top.

Unreachable p-values
    A motif whose best-scoring word alone is more likely than ``pvalue`` under
    the background (a short or degenerate motif: a 6-mer at a uniform background
    has p = 2.4e-4) has no threshold: :func:`pvalue_to_threshold` returns
    ``+inf`` and the scanners report no hits for it, with one warning naming the
    motifs. This is decided exactly, from the product over columns of the
    background mass of each column's best bases, not from the quantized DP.

One precision (numpy and Rust)
    A window's score is the float64 sum of its columns' scores added left to
    right from 0.0, and it is compared with the threshold in float64. The numpy
    reference and the Rust kernel (``src/motif_scan.rs``) add the same terms in
    the same order, so they agree bit for bit; the extreme thresholds (the best
    word's score, ``relative_frac`` 0 or 1) are computed in that order too, so a
    window that ties them is a hit on both backends.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional
import numpy as np

from ...data._pwm import PWM, BASES, BASE_INDEX

__all__ = [
    "MotifThresholdWarning",
    "encode_seq",
    "estimate_background",
    "pvalue_to_threshold",
    "relative_threshold",
    "scan_sequence",
    "scan_motifs",
    "scan_motifs_numpy",
]


class MotifThresholdWarning(UserWarning):
    """A motif cannot reach the requested p-value on the background, so it reports no hits."""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Sequence encoding
# ──────────────────────────────────────────────────────────────────────────────

# Pre-build a 256-element lookup table: ord(char) → 0-4
_ENCODE_TABLE = np.full(256, 4, dtype=np.uint8)
for _b, _i in BASE_INDEX.items():
    _ENCODE_TABLE[ord(_b)] = _i
    _ENCODE_TABLE[ord(_b.lower())] = _i  # soft-masked lowercase → valid


def encode_seq(seq: str) -> np.ndarray:
    """Encode a DNA string to uint8 codes (A=0, C=1, G=2, T=3, other=4).

    Parameters
    ----------
    seq
        DNA string; may contain uppercase or lowercase ACGT, N, or any IUPAC
        / gap characters.  Lowercase a/c/g/t (soft-masked) are treated as
        their canonical base (code 0-3).

    Returns
    -------
    np.ndarray
        ``uint8`` array of length ``len(seq)``.  Code 4 means ambiguous.
    """
    # view the string as a byte array, index into the lookup table
    arr = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    return _ENCODE_TABLE[arr]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Background nucleotide composition
# ──────────────────────────────────────────────────────────────────────────────

def estimate_background(
    sequences: list[str],
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Estimate A/C/G/T composition from a set of sequences.

    Ambiguous bases (N, etc.) are ignored.  A pseudocount is added before
    normalisation so that even a monomer sequence does not produce zeros.

    Parameters
    ----------
    sequences
        List of DNA strings (any case; soft-masked allowed).
    pseudocount
        Added to each base count before normalisation (default 1.0).

    Returns
    -------
    np.ndarray
        ``float64`` array of length 4 (A, C, G, T) summing to 1.0.
        Returns uniform ``[0.25]*4`` when no valid bases are found.
    """
    counts = np.zeros(4, dtype=np.float64)
    for seq in sequences:
        codes = encode_seq(seq)
        valid = codes[codes < 4]
        if valid.size > 0:
            counts += np.bincount(valid, minlength=4).astype(np.float64)

    total = counts.sum()
    if total == 0.0:
        return np.full(4, 0.25, dtype=np.float64)

    return (counts + pseudocount) / (total + 4.0 * pseudocount)


# ──────────────────────────────────────────────────────────────────────────────
# 3. P-value → score threshold via DP convolution
# ──────────────────────────────────────────────────────────────────────────────

def _seq_sum(values) -> float:
    """The float64 sum of ``values`` added left to right from 0.0.

    The order the scanners add a window's column scores in (numpy here, Rust in
    ``src/motif_scan.rs``), so a threshold computed this way from the column
    extremes is exactly the score of the window that reaches them. Python's own
    ``sum`` is compensated since 3.12 and ``ndarray.sum`` is pairwise from eight
    terms, so neither is used.
    """
    s = 0.0
    for v in np.asarray(values, dtype=np.float64).tolist():
        s += v
    return s


def _best_word_pvalue(pssm: np.ndarray, bg: np.ndarray) -> float:
    """Exact null probability of the best score: the product over columns of the
    background mass of the bases that reach the column's maximum."""
    col_max = pssm.max(axis=0)
    p = 1.0
    for j in range(pssm.shape[1]):
        p *= float(bg[pssm[:, j] == col_max[j]].sum())
    return p


def _pvalue_threshold(pssm, background, pvalue: float = 1e-4, n_bins: int = 1000):
    """``(threshold, p_best)``: :func:`pvalue_to_threshold`'s value without its
    warning, and the exact null probability of the motif's best score. The
    threshold is ``+inf`` exactly when ``p_best > pvalue``."""
    pssm = np.asarray(pssm, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    bg = bg / bg.sum()
    w = pssm.shape[1]

    col_min = pssm.min(axis=0)
    col_max = pssm.max(axis=0)
    # in the scanners' summation order, so a window holding the best word scores
    # exactly score_max
    score_min = _seq_sum(col_min)
    score_max = _seq_sum(col_max)

    p_best = _best_word_pvalue(pssm, bg)
    if p_best > pvalue:
        return math.inf, p_best
    if score_max == score_min:
        return float(score_max), p_best

    bin_width = (score_max - score_min) / n_bins
    # each base's score above its column's minimum, in bins (round half to even,
    # as Python's round did here before)
    deltas = np.rint((pssm - col_min) / bin_width).astype(np.int64)    # (4, w)
    # Rounding can carry the best words past n_bins. Size the distribution for the
    # largest quantized score, so no mass is shifted off the end: the earlier
    # n_bins + 1 grid dropped it and renormalised the rest, which under-counted the
    # top of the tail: on short motifs the exact tail at the threshold reached 24
    # times the target (random motifs of 5 to 8 columns, checked by enumeration).
    top = int(deltas.max(axis=0).sum())
    dist = np.zeros(top + 1, dtype=np.float64)
    dist[0] = 1.0
    for j in range(w):
        new_dist = np.zeros(top + 1, dtype=np.float64)
        for b in range(4):
            d = int(deltas[b, j])
            new_dist[d:] += bg[b] * dist[:top + 1 - d]
        dist = new_dist
    total = dist.sum()
    if total > 0:
        dist /= total                                    # floating-point drift only

    # tail[k] = P(quantized score >= k); the smallest k with tail[k] <= pvalue
    tail = np.cumsum(dist[::-1])[::-1]
    passing = np.flatnonzero(tail <= pvalue)
    if passing.size == 0:
        # The grid lumps the best words with near-ties, but the best words alone are
        # rare enough (p_best <= pvalue): they are the threshold.
        return float(score_max), p_best
    threshold = score_min + int(passing[0]) * bin_width
    # past score_max no window could reach it; the best words still qualify
    return float(min(threshold, score_max)), p_best


def pvalue_to_threshold(
    pssm: np.ndarray,
    background: np.ndarray,
    pvalue: float = 1e-4,
    n_bins: int = 1000,
) -> float:
    """Convert a right-tail p-value to a PSSM score cutoff via DP convolution.

    Implements the standard FIMO/MOODS algorithm:
      1. Find the global min and max achievable score (sum of column mins/maxs).
      2. Quantize the score range to ``n_bins`` integer bins.
      3. For each PSSM column, build a probability distribution over quantized
         per-column scores; convolve (shift-add) across all columns.
      4. Compute the right-tail CDF; return the smallest score whose right-tail
         probability ≤ ``pvalue``.

    Complexity: O(w × n_bins) time and space.

    Parameters
    ----------
    pssm
        ``(4, w)`` log2-odds matrix as returned by ``PWM.pssm()``.
    background
        Length-4 float64 background frequencies (normalised here).
    pvalue
        Target right-tail probability (default 1e-4).
    n_bins
        Number of quantization bins over the score range (default 1000).
        Larger = more accurate threshold but linearly more compute. A word's
        quantized score is within ``w / 2`` bins of its exact score.

    Returns
    -------
    float
        Score threshold ``t`` such that P(score ≥ t | null) ≤ pvalue, to the
        bin resolution. ``+inf`` when no threshold exists: the motif's
        best-scoring word alone is more likely than ``pvalue`` (decided
        exactly), so no window can be called at that p-value; a
        :class:`MotifThresholdWarning` says so. A window's score is compared
        with ``t`` in float64 (see the module notes on one precision).
    """
    threshold, _p_best = _pvalue_threshold(pssm, background, pvalue, n_bins)
    if not math.isfinite(threshold):
        warnings.warn(
            "pvalue_to_threshold: no window of this motif can reach the requested "
            "p-value on this background (its best-scoring word alone is more "
            "likely), so the threshold is +inf and the motif reports no hits. Use "
            "a larger pvalue, or relative_threshold, to keep short or degenerate "
            "motifs.", MotifThresholdWarning, stacklevel=2)
    return threshold


# ──────────────────────────────────────────────────────────────────────────────
# 4. Relative (fraction-of-max) threshold
# ──────────────────────────────────────────────────────────────────────────────

def relative_threshold(pssm: np.ndarray, frac: float = 0.8) -> float:
    """Score cutoff as a linear interpolation between min and max PSSM score.

    Parameters
    ----------
    pssm
        ``(4, w)`` log2-odds matrix.
    frac
        Fraction of the score range above the minimum (default 0.8, meaning
        80% of the way from the worst to the best possible score).

    Returns
    -------
    float
        ``frac * max_score + (1 - frac) * min_score``, with the two extremes
        summed in the scanners' order, so ``frac=1`` is exactly the best word's
        score and ``frac=0`` the worst's: both are hits on either backend.
    """
    pssm = np.asarray(pssm, dtype=np.float64)
    min_score = _seq_sum(pssm.min(axis=0))
    max_score = _seq_sum(pssm.max(axis=0))
    return float(frac * max_score + (1.0 - frac) * min_score)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Single-sequence scanner
# ──────────────────────────────────────────────────────────────────────────────

def _build_rc_pssm(pssm: np.ndarray) -> np.ndarray:
    """Reverse-complement of a (4, w) PSSM.

    The row order A,C,G,T becomes T,G,C,A (complement) and columns are
    reversed (reverse) — equivalent to transposing the strand while
    preserving the 5'→3' reading direction.
    """
    return pssm[::-1, ::-1].copy()


def _score_windows(
    pssm: np.ndarray,
    codes: np.ndarray,
    n_min: int = 0,
) -> np.ndarray:
    """Return per-window PSSM scores as a float64 array of length n_windows.

    N-handling: positions with code==4 contribute ``col_min`` — the worst
    possible score for that column.  This makes N-containing windows almost
    always fall below threshold (conservative / FIMO-compatible).

    Each window's score is its column scores added left to right from 0.0 in
    float64, the order the Rust kernel adds them in, so the two backends give
    the same scores to the bit (a pairwise ``sum(axis=1)`` did not, from eight
    columns on, and a hit at a threshold equal to an achievable score then
    depended on the backend).

    Parameters
    ----------
    pssm
        ``(4, w)`` log2-odds matrix.
    codes
        uint8 encoded sequence of length L (codes in 0..4).
    n_min
        Unused (kept for signature compatibility; N positions take the
        column minimum).

    Returns
    -------
    np.ndarray
        ``float64`` array of length ``max(0, L - w + 1)``.
    """
    pssm = np.asarray(pssm, dtype=np.float64)
    w = pssm.shape[1]
    L = codes.shape[0]
    n_windows = L - w + 1
    if n_windows <= 0:
        return np.empty(0, dtype=np.float64)

    # Augmented PSSM: row 4 = column minimums (for N bases)
    pssm_aug = np.vstack([pssm, pssm.min(axis=0)[np.newaxis, :]])   # (5, w)
    codes = np.asarray(codes, dtype=np.intp)
    scores = np.zeros(n_windows, dtype=np.float64)
    for j in range(w):
        scores += pssm_aug[codes[j:j + n_windows], j]
    return scores


def scan_sequence(
    pssm: np.ndarray,
    codes: np.ndarray,
    threshold: float,
    both_strands: bool = True,
) -> list[tuple[int, str, float]]:
    """Slide a PSSM over an encoded sequence and return hits above threshold.

    Parameters
    ----------
    pssm
        ``(4, w)`` log2-odds matrix (forward strand).
    codes
        uint8 encoded sequence (from ``encode_seq``).
    threshold
        Minimum score to report a hit.
    both_strands
        If True, also scan the reverse-complement strand.

    Returns
    -------
    list of (position, strand, score)
        ``position`` is the 0-based start, on the **forward** strand, of the
        window the match occupies (the same convention for both strands).
        ``strand`` is ``'+'`` or ``'-'``.
        ``score`` is the PSSM log2-odds score.
    """
    pssm = np.asarray(pssm, dtype=np.float64)
    threshold = float(threshold)
    hits: list[tuple[int, str, float]] = []

    # Forward strand
    fwd_scores = _score_windows(pssm, codes)
    for i in np.flatnonzero(fwd_scores >= threshold):
        hits.append((int(i), "+", float(fwd_scores[i])))

    # Reverse strand: the reverse-complement PSSM is slid along the forward
    # sequence, so window i of it is the minus-strand match occupying forward
    # positions i .. i + w - 1, and its forward start is i itself. (It used to be
    # reported as L - w - i, the coordinate it would have on the reversed
    # sequence: every minus-strand position was mirrored.)
    if both_strands:
        rev_scores = _score_windows(_build_rc_pssm(pssm), codes)
        for i in np.flatnonzero(rev_scores >= threshold):
            hits.append((int(i), "-", float(rev_scores[i])))

    return hits


def _count_and_best(pssm, rc_pssm, codes, threshold):
    """``(hit count, best window score)`` of one PSSM on one sequence, over the
    forward and (when ``rc_pssm`` is given) reverse strands; scored and compared
    in float64 as :func:`scan_sequence` does."""
    scores = _score_windows(pssm, codes)
    count = int(np.count_nonzero(scores >= threshold))
    best = float(scores.max()) if scores.size else -math.inf
    if rc_pssm is not None:
        rev = _score_windows(rc_pssm, codes)
        count += int(np.count_nonzero(rev >= threshold))
        if rev.size:
            best = max(best, float(rev.max()))
    return count, best


def _motif_thresholds(pwms, bg, pvalue, relative_frac, pseudocount):
    """The scanners' shared prologue: ``(pssms, thresholds)`` for every motif, and
    one :class:`MotifThresholdWarning` naming the motifs that cannot reach
    ``pvalue`` (threshold ``+inf``: they report no hits)."""
    pssms, thresholds, unreachable = [], [], []
    for pwm in pwms:
        pssm = pwm.pssm(bg, pseudocount=pseudocount)
        if relative_frac is not None:
            thr = relative_threshold(pssm, relative_frac)
        else:
            thr, p_best = _pvalue_threshold(pssm, bg, pvalue)
            if not math.isfinite(thr):
                unreachable.append((pwm, p_best))
        pssms.append(np.asarray(pssm, dtype=np.float64))
        thresholds.append(float(thr))
    if unreachable:
        names = ", ".join(f"{p.motif_id} ({p.tf_name}, best word p={pb:.1e})" for p, pb in unreachable)
        warnings.warn(
            f"scan_motifs: {len(unreachable)} of {len(pwms)} motifs cannot reach p={pvalue:g} on this "
            f"background: the best-scoring word of each is more likely than that alone, so they report no "
            f"hits. Scan them at a larger pvalue, or with relative_frac, to keep them: {names}",
            MotifThresholdWarning, stacklevel=3)
    return pssms, thresholds


# ──────────────────────────────────────────────────────────────────────────────
# 6. Top-level multi-motif scanner
# ──────────────────────────────────────────────────────────────────────────────

def _scan_motifs_numpy(
    pwms: list[PWM],
    sequences: list[str],
    *,
    background: Optional[np.ndarray] = None,
    pvalue: float = 1e-4,
    relative_frac: Optional[float] = None,
    both_strands: bool = True,
    pseudocount: float = 0.01,
) -> dict:
    """Reference (pure-numpy) motif scanner: all PWMs × all sequences.

    RAM behaviour
    -------------
    Peak resident memory is approximately::

        n_motifs × (4 × w × 8 bytes)       # all PSSMs (tiny)
        + max_seq_len × 1 byte              # encoded sequence
        + n_windows × 8 bytes               # sliding window scores (one seq)

    PSSMs and thresholds are precomputed once per motif.  Only one sequence's
    window score array is live at a time (each is released before the next).

    Parameters
    ----------
    pwms
        List of ``PWM`` objects to scan.
    sequences
        List of DNA strings (promoter / peak sequences).
    background
        Length-4 float64 background frequencies.  If None, estimated from
        ``sequences`` via ``estimate_background``.
    pvalue
        Right-tail p-value threshold (used when ``relative_frac`` is None).
    relative_frac
        If given, use ``relative_threshold(pssm, relative_frac)`` instead of
        the DP p-value method.
    both_strands
        Scan both forward and reverse-complement strands.
    pseudocount
        Pseudocount passed to ``PWM.pssm()`` (default 0.01).

    Returns
    -------
    dict with keys:

    ``"motif_ids"``
        List[str] of length n_motifs.
    ``"tf_names"``
        List[str] of length n_motifs.
    ``"best_score"``
        ``float32`` array of shape ``(n_motifs, n_seqs)``.
        ``NaN`` where no hit was found.
    ``"hit_count"``
        ``int32`` array of shape ``(n_motifs, n_seqs)``.
        0 where no hit was found.

    A motif that cannot reach ``pvalue`` on the background (threshold ``+inf``,
    see :func:`pvalue_to_threshold`) is not scanned and reports no hits; one
    :class:`MotifThresholdWarning` names those motifs.
    """
    n_motifs = len(pwms)
    n_seqs = len(sequences)

    if background is None:
        background = estimate_background(sequences)
    bg = np.asarray(background, dtype=np.float64)

    # One PSSM + threshold per motif (done once, not per sequence)
    pssms, thresholds = _motif_thresholds(pwms, bg, pvalue, relative_frac, pseudocount)
    rc_pssms = [_build_rc_pssm(p) if both_strands else None for p in pssms]
    scanned = [m for m in range(n_motifs) if math.isfinite(thresholds[m])]

    # Output arrays
    best_score = np.full((n_motifs, n_seqs), np.nan, dtype=np.float32)
    hit_count = np.zeros((n_motifs, n_seqs), dtype=np.int32)

    # Motif_ids / tf_names
    motif_ids = [pwm.motif_id for pwm in pwms]
    tf_names = [pwm.tf_name for pwm in pwms]

    # Main scan: one (motif, sequence) pair at a time
    for s_idx, seq in enumerate(sequences):
        codes = encode_seq(seq)
        for m_idx in scanned:
            if codes.shape[0] < pssms[m_idx].shape[1]:
                continue           # sequence shorter than the motif: no window
            count, best = _count_and_best(pssms[m_idx], rc_pssms[m_idx], codes, thresholds[m_idx])
            if count:
                hit_count[m_idx, s_idx] = count
                best_score[m_idx, s_idx] = np.float32(best)

    return {
        "motif_ids": motif_ids,
        "tf_names": tf_names,
        "best_score": best_score,
        "hit_count": hit_count,
    }


# Back-compat / explicit-backend alias for the numpy reference scanner.
scan_motifs_numpy = _scan_motifs_numpy


def _rust_ext_available() -> bool:
    """True if the compiled ``_piaso`` extension (Rust PWM scanner) is importable."""
    try:
        from ... import _piaso  # noqa: F401  (piaso.preprocessing.grn._scan → piaso._piaso)
    except ImportError:
        return False
    return True


def scan_motifs(
    pwms: list[PWM],
    sequences: list[str],
    *,
    background: Optional[np.ndarray] = None,
    pvalue: float = 1e-4,
    relative_frac: Optional[float] = None,
    both_strands: bool = True,
    pseudocount: float = 0.01,
    backend: str = "auto",
) -> dict:
    """Motif scanner: all PWMs × all sequences (Rust-accelerated when available).

    This is the public entry point (``piaso.pp.scan_motifs``). It dispatches
    between the Rust backend (``_piaso.scan_motifs_fwd`` via
    :func:`~piaso.preprocessing.grn._scan_rust.scan_motifs_rust`) and the
    pure-numpy reference (:func:`_scan_motifs_numpy`); the two give identical
    results to the bit (same log-odds PSSM, N-augmentation, p-value/relative
    threshold, reverse-complement handling, and float64 window scores added in
    the same order), so ``backend`` only trades speed for the no-compiler
    fallback. A motif that cannot reach ``pvalue`` reports no hits on either
    backend, with one warning naming it (see :func:`pvalue_to_threshold`).

    Parameters
    ----------
    pwms, sequences, background, pvalue, relative_frac, both_strands, pseudocount
        See :func:`_scan_motifs_numpy` — forwarded unchanged to the selected
        backend.
    backend
        Which scanner to use:

        ``"auto"`` (default)
            Use Rust if the ``_piaso`` extension is built, else fall back to numpy.
        ``"rust"``
            Force the Rust backend; raise :class:`ImportError` if ``_piaso`` is
            not built.
        ``"numpy"``
            Force the pure-numpy reference scanner.

    Returns
    -------
    dict
        Same schema as :func:`_scan_motifs_numpy` (``motif_ids``, ``tf_names``,
        ``best_score``, ``hit_count``).
    """
    if backend not in ("auto", "rust", "numpy"):
        raise ValueError(
            f"backend must be one of 'auto', 'rust', 'numpy'; got {backend!r}"
        )

    use_rust = backend == "rust" or (backend == "auto" and _rust_ext_available())

    if use_rust:
        from ._scan_rust import scan_motifs_rust  # deferred → no import cycle
        return scan_motifs_rust(
            pwms, sequences, background=background, pvalue=pvalue,
            relative_frac=relative_frac, both_strands=both_strands,
            pseudocount=pseudocount,
        )

    return _scan_motifs_numpy(
        pwms, sequences, background=background, pvalue=pvalue,
        relative_frac=relative_frac, both_strands=both_strands,
        pseudocount=pseudocount,
    )
