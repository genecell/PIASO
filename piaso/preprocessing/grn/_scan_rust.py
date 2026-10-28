"""Rust-accelerated PWM scan — same contract as :func:`piaso.pp.scan_motifs`.

The numeric definitions (log-odds PSSM, N-augmentation, p-value/relative
threshold, reverse complement) live in the tested numpy module ``_scan``; this
wrapper only offloads the O(motifs × sequences × windows) inner loop to
``_piaso.scan_motifs_fwd`` (rayon). Forward + reverse strands are passed as two
forward PSSMs and merged here. The kernel adds a window's column scores left to
right in float64 and compares them with the threshold in float64, as the numpy
reference does, so the two agree to the bit (tests/test_grn_scan_parity.py
checks it against ``_scan_motifs_numpy``).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ._scan import _build_rc_pssm, _motif_thresholds, encode_seq, estimate_background


def _augmented_pssm(pssm: np.ndarray) -> np.ndarray:
    """(4, w) PSSM → (5, w) with an N-fill row = per-column minimum (matches the
    numpy scanner's N handling)."""
    col_min = pssm.min(axis=0, keepdims=True)
    return np.vstack([pssm, col_min]).astype(np.float64)


def scan_motifs_rust(
    pwms: List,
    sequences: List[str],
    *,
    background: Optional[np.ndarray] = None,
    pvalue: float = 1e-4,
    relative_frac: Optional[float] = None,
    both_strands: bool = True,
    pseudocount: float = 0.01,
) -> dict:
    """Rust-backed equivalent of :func:`piaso.pp.scan_motifs`."""
    from ... import _piaso  # raises ImportError if the ext isn't built

    if background is None:
        background = estimate_background(sequences)
    background = np.asarray(background, dtype=np.float64)

    # One PSSM and threshold per motif, as the numpy reference computes them (and
    # one warning naming the motifs that cannot reach pvalue: threshold +inf, not
    # scanned). Forward (+ rc) augmented blocks; remember each block's motif.
    pssms, thresholds = _motif_thresholds(pwms, background, pvalue, relative_frac, pseudocount)
    blocks_pssm: List[np.ndarray] = []
    widths: List[int] = []
    block_thr: List[float] = []
    block_motif: List[int] = []
    for mi, (pssm, thr) in enumerate(zip(pssms, thresholds)):
        if not math.isfinite(thr):
            continue
        strands = [pssm, _build_rc_pssm(pssm)] if both_strands else [pssm]
        for p in strands:
            blocks_pssm.append(_augmented_pssm(p))
            widths.append(p.shape[1])
            block_thr.append(thr)
            block_motif.append(mi)

    n_motifs = len(pwms)
    n_seq = len(sequences)
    out_best = np.full((n_motifs, n_seq), -np.inf)
    out_count = np.zeros((n_motifs, n_seq), dtype=np.int32)
    if blocks_pssm and n_seq:
        pssms_flat = np.concatenate([b.reshape(-1) for b in blocks_pssm]).astype(np.float64)
        # Flat-buffer (#116): concatenate all encoded sequences into ONE uint8 buffer + CSR offsets,
        # instead of a Python list-of-lists (N allocations + N FFI conversions). One bytes copy crosses
        # the PyO3 boundary; the Rust scanner indexes seq i as seq_codes[offsets[i]:offsets[i+1]].
        enc = [np.asarray(encode_seq(s), dtype=np.uint8) for s in sequences]
        seq_codes = np.concatenate(enc) if len(enc) > 1 else enc[0]
        offsets = np.empty(len(enc) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([len(e) for e in enc], out=offsets[1:])

        best, count = _piaso.scan_motifs_fwd(
            pssms_flat, [int(w) for w in widths], [float(t) for t in block_thr],
            seq_codes.tobytes(), [int(o) for o in offsets])
        best = np.asarray(best, dtype=np.float64)   # (n_blocks, n_seq)
        count = np.asarray(count, dtype=np.int64)
        # merge fwd/rc blocks back to their motif: max score, summed hit count
        for b in range(best.shape[0]):
            mi = block_motif[b]
            np.maximum(out_best[mi], np.where(np.isnan(best[b]), -np.inf, best[b]), out=out_best[mi])
            out_count[mi] += count[b].astype(np.int32)

    # best_score: NaN where no window cleared threshold (match numpy contract:
    # NaN/None means "no hit"). We mark no-hit where hit_count==0.
    best_out = np.where(out_count > 0, out_best, np.nan).astype(np.float32)
    return {
        "motif_ids": [p.motif_id for p in pwms],
        "tf_names": [p.tf_name for p in pwms],
        "best_score": best_out,
        "hit_count": out_count,
    }
