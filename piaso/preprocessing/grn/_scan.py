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
    ``(total_score_range) / n_bins`` bits per bin, so the returned threshold is
    approximate to within one bin width.  Rounding is toward the nearest bin, which
    slightly underestimates the threshold (conservative: more hits may pass).
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...data._pwm import PWM, BASES, BASE_INDEX

__all__ = [
    "encode_seq",
    "estimate_background",
    "pvalue_to_threshold",
    "relative_threshold",
    "scan_sequence",
    "scan_motifs",
    "scan_motifs_numpy",
]

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
        Length-4 float64 background frequencies (sum to 1).
    pvalue
        Target right-tail probability (default 1e-4).
    n_bins
        Number of quantization bins (default 1000).  Larger = more accurate
        threshold but linearly more compute.  Approximation error is at most
        one bin width.

    Returns
    -------
    float
        Score threshold ``t`` such that P(score ≥ t | null) ≤ pvalue.
        Approximate to within the bin resolution.
    """
    pssm = np.asarray(pssm, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    w = pssm.shape[1]

    # Per-column min and max achievable score
    col_min = pssm.min(axis=0)   # shape (w,)
    col_max = pssm.max(axis=0)   # shape (w,)

    score_min = col_min.sum()
    score_max = col_max.sum()

    if score_max == score_min:
        # Constant PSSM (degenerate) — return score_max + epsilon
        return float(score_max)

    # Map real score → bin index: bin 0 = score_min, bin n_bins-1 ~ score_max
    score_range = score_max - score_min
    bin_width = score_range / n_bins  # resolution of the grid

    # DP: dist[i] = probability that the cumulative score falls in bin i
    # Start with a point mass at bin 0 (score = 0 = score_min offset)
    dist = np.zeros(n_bins + 1, dtype=np.float64)
    dist[0] = 1.0

    for j in range(w):
        new_dist = np.zeros(n_bins + 1, dtype=np.float64)
        for b in range(4):
            s = pssm[b, j]
            # Offset relative to column minimum to keep bin indices ≥ 0
            # Cumulative offset from previous columns is already accounted for
            # by starting at bin 0 = score_min.
            # Bin index for score s in column j relative to col_min[j]:
            s_offset = s - col_min[j]
            bin_delta = int(round(s_offset / bin_width)) if bin_width > 0 else 0
            # Shift the current distribution by bin_delta and add bg[b] weight
            if bin_delta == 0:
                new_dist += bg[b] * dist
            else:
                new_dist[bin_delta:] += bg[b] * dist[:-bin_delta]
        dist = new_dist

    # Normalise (floating-point drift)
    dist_sum = dist.sum()
    if dist_sum > 0:
        dist /= dist_sum

    # Right-tail CDF: P(score >= bin k) = sum(dist[k:])
    # We want the smallest k such that tail[k] <= pvalue
    tail = np.cumsum(dist[::-1])[::-1]  # tail[k] = P(score >= bin-k)

    # Find the smallest bin k with tail[k] <= pvalue
    passing = np.where(tail <= pvalue)[0]
    if len(passing) == 0:
        # No threshold achieves the target p-value; return the maximum score
        return float(score_max)

    k = int(passing[0])
    # Convert bin k back to a real score:
    # total score = score_min + k * bin_width
    threshold = score_min + k * bin_width
    return float(threshold)


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
        ``frac * max_score + (1 - frac) * min_score``.
    """
    pssm = np.asarray(pssm, dtype=np.float64)
    min_score = pssm.min(axis=0).sum()
    max_score = pssm.max(axis=0).sum()
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
    n_min: int,
) -> np.ndarray:
    """Return per-window PSSM scores as a float32 array of length n_windows.

    N-handling: positions with code==4 contribute ``col_min`` — the worst
    possible score for that column.  This makes N-containing windows almost
    always fall below threshold (conservative / FIMO-compatible).

    Parameters
    ----------
    pssm
        ``(4, w)`` log2-odds matrix.
    codes
        uint8 encoded sequence of length L (codes in 0..4).
    n_min
        Minimum score index for N positions (unused here; we use pssm.min).

    Returns
    -------
    np.ndarray
        ``float32`` array of length ``max(0, L - w + 1)``.
    """
    w = pssm.shape[1]
    L = codes.shape[0]
    n_windows = L - w + 1
    if n_windows <= 0:
        return np.empty(0, dtype=np.float32)

    # Build augmented PSSM: row 4 = column minimums (for N bases)
    col_mins = pssm.min(axis=0)  # shape (w,)
    pssm_aug = np.vstack([pssm, col_mins[np.newaxis, :]])  # shape (5, w)

    # sliding_window_view: shape (n_windows, w) — zero-copy view
    windows = sliding_window_view(codes, w)  # (n_windows, w), dtype uint8

    # Gather: for each window i and position j, fetch pssm_aug[code, j]
    # codes in windows are 0..4, all valid indices into pssm_aug rows
    # scores[i, j] = pssm_aug[windows[i, j], j]
    j_idx = np.arange(w, dtype=np.intp)
    # Use advanced indexing: windows shape (n_windows, w), j_idx shape (w,)
    scores = pssm_aug[windows, j_idx]  # (n_windows, w), float64
    return scores.sum(axis=1).astype(np.float32)


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
        ``position`` is 0-based start on the **forward** strand.
        ``strand`` is ``'+'`` or ``'-'``.
        ``score`` is the PSSM log2-odds score.
    """
    pssm = np.asarray(pssm, dtype=np.float64)
    w = pssm.shape[1]
    hits: list[tuple[int, str, float]] = []

    # Forward strand
    fwd_scores = _score_windows(pssm, codes, 0)
    if fwd_scores.size > 0:
        hit_idx = np.where(fwd_scores >= threshold)[0]
        for i in hit_idx:
            hits.append((int(i), "+", float(fwd_scores[i])))

    # Reverse strand
    if both_strands:
        rc_pssm = _build_rc_pssm(pssm)
        rev_scores = _score_windows(rc_pssm, codes, 0)
        L = codes.shape[0]
        if rev_scores.size > 0:
            hit_idx = np.where(rev_scores >= threshold)[0]
            for i in hit_idx:
                # Report position as the start on the forward strand (i.e., the
                # 3'-end of the rc match mapped back to forward coords)
                fwd_pos = L - w - int(i)
                hits.append((fwd_pos, "-", float(rev_scores[i])))

    return hits


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
    """
    n_motifs = len(pwms)
    n_seqs = len(sequences)

    if background is None:
        background = estimate_background(sequences)
    bg = np.asarray(background, dtype=np.float64)

    # Precompute one PSSM + threshold per motif (done once, not per sequence)
    pssms: list[np.ndarray] = []
    thresholds: list[float] = []
    for pwm in pwms:
        pssm = pwm.pssm(bg, pseudocount=pseudocount)
        if relative_frac is not None:
            thr = relative_threshold(pssm, relative_frac)
        else:
            thr = pvalue_to_threshold(pssm, bg, pvalue)
        pssms.append(pssm)
        thresholds.append(thr)

    # Output arrays
    best_score = np.full((n_motifs, n_seqs), np.nan, dtype=np.float32)
    hit_count = np.zeros((n_motifs, n_seqs), dtype=np.int32)

    # Motif_ids / tf_names
    motif_ids = [pwm.motif_id for pwm in pwms]
    tf_names = [pwm.tf_name for pwm in pwms]

    # Main scan: one (motif, sequence) pair at a time
    for s_idx, seq in enumerate(sequences):
        codes = encode_seq(seq)

        for m_idx in range(n_motifs):
            pssm = pssms[m_idx]
            thr = thresholds[m_idx]
            w = pssm.shape[1]
            L = codes.shape[0]

            if L < w:
                # Sequence shorter than motif — no possible hit
                continue

            hits = scan_sequence(pssm, codes, thr, both_strands=both_strands)

            if hits:
                hit_count[m_idx, s_idx] = len(hits)
                best = max(h[2] for h in hits)
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
    pure-numpy reference (:func:`_scan_motifs_numpy`); the two are numerically
    identical (same log-odds PSSM, N-augmentation, p-value/relative threshold and
    reverse-complement handling), so ``backend`` only trades speed for the
    no-compiler fallback.

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
