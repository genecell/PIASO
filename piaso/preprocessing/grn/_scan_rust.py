"""Rust-accelerated PWM scan — same contract as :func:`piaso.pp.scan_motifs`.

The numeric definitions (log-odds PSSM, N-augmentation, p-value/relative
threshold, reverse complement) live in the tested numpy module ``_scan``; this
wrapper only offloads the O(motifs × sequences × windows) inner loop to
``_piaso.scan_motifs_fwd`` (rayon). Forward + reverse strands are passed as two
forward PSSMs and merged here, so results match the numpy reference exactly.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ._scan import (
    encode_seq, estimate_background, pvalue_to_threshold, relative_threshold,
)


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

    # Build forward (+ optional rc) augmented PSSMs and thresholds; remember which
    # original motif each block belongs to.
    blocks_pssm: List[np.ndarray] = []
    widths: List[int] = []
    thresholds: List[float] = []
    block_motif: List[int] = []
    for mi, pwm in enumerate(pwms):
        pssm = pwm.pssm(background, pseudocount=pseudocount)  # (4, w)
        thr = (relative_threshold(pssm, relative_frac) if relative_frac is not None
               else pvalue_to_threshold(pssm, background, pvalue))
        blocks_pssm.append(_augmented_pssm(pssm))
        widths.append(pssm.shape[1])
        thresholds.append(float(thr))
        block_motif.append(mi)
        if both_strands:
            rc = pssm[::-1, ::-1]  # reverse-complement PSSM (matches numpy)
            blocks_pssm.append(_augmented_pssm(rc))
            widths.append(rc.shape[1])
            thresholds.append(float(thr))
            block_motif.append(mi)

    pssms_flat = np.concatenate([b.reshape(-1) for b in blocks_pssm]).astype(np.float64)
    # Flat-buffer (#116): concatenate all encoded sequences into ONE uint8 buffer + CSR offsets,
    # instead of a Python list-of-lists (N allocations + N FFI conversions). One bytes copy crosses
    # the PyO3 boundary; the Rust scanner indexes seq i as seq_codes[offsets[i]:offsets[i+1]].
    enc = [np.asarray(encode_seq(s), dtype=np.uint8) for s in sequences]
    if enc:
        seq_codes = np.concatenate(enc) if len(enc) > 1 else enc[0]
        offsets = np.empty(len(enc) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([len(e) for e in enc], out=offsets[1:])
    else:
        seq_codes = np.zeros(0, dtype=np.uint8)
        offsets = np.zeros(1, dtype=np.int64)

    best, count = _piaso.scan_motifs_fwd(
        pssms_flat, [int(w) for w in widths], [float(t) for t in thresholds],
        seq_codes.tobytes(), [int(o) for o in offsets])
    best = np.asarray(best, dtype=np.float64)   # (n_blocks, n_seq)
    count = np.asarray(count, dtype=np.int64)

    n_motifs = len(pwms)
    n_seq = len(sequences)
    out_best = np.full((n_motifs, n_seq), -np.inf)
    out_count = np.zeros((n_motifs, n_seq), dtype=np.int32)
    block_motif = np.asarray(block_motif)
    # merge fwd/rc blocks back to their motif: max score, summed hit count, but
    # a "hit" only counts where the score cleared threshold → use count>0 mask.
    for b in range(best.shape[0]):
        mi = block_motif[b]
        np.maximum(out_best[mi], best[b], out=out_best[mi])
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
