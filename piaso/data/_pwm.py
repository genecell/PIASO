"""Shared PWM (position weight matrix) type for the PIASO-GRN motif machinery.

A ``PWM`` is the common currency between the motif-DB loaders (``_motifs.py``,
which produce them) and the scanner (``_scan.py`` / the Rust backend, which
consume them). Kept deliberately tiny + dependency-free so both sides can import
it without cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# Canonical base order used everywhere downstream (matrix rows / 2-bit codes).
BASES = "ACGT"
BASE_INDEX = {b: i for i, b in enumerate(BASES)}


@dataclass
class PWM:
    """A single transcription-factor motif.

    Attributes
    ----------
    motif_id
        Stable DB identifier (e.g. JASPAR ``MA0004.1``, CIS-BP ``M00123``).
    tf_name
        TF gene symbol the motif is associated with (e.g. ``Arnt``). One TF may
        own several PWMs; the cistrome step aggregates per TF (N2: union).
    probs
        ``(4, w)`` float32 probability matrix (columns sum to 1), rows = A,C,G,T.
    source
        ``"jaspar"`` / ``"cisbp"`` — provenance.
    """

    motif_id: str
    tf_name: str
    probs: np.ndarray
    source: str = "jaspar"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        p = np.asarray(self.probs, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] != 4:
            raise ValueError(
                f"PWM {self.motif_id!r}: probs must be (4, w), got {p.shape}"
            )
        self.probs = p

    @property
    def width(self) -> int:
        return self.probs.shape[1]

    def pssm(self, background: np.ndarray, pseudocount: float = 0.01) -> np.ndarray:
        """Log2-odds position-specific scoring matrix.

        ``PSSM[b, j] = log2((p[b, j] + pc * bg[b]) / bg[b])`` — the standard
        FIMO/MOODS log-odds with a background-proportional pseudocount so columns
        with a 0 entry don't blow up to -inf.
        """
        bg = np.asarray(background, dtype=np.float64).reshape(4, 1)
        bg = np.maximum(bg, 1e-9)
        p = self.probs.astype(np.float64)
        # renormalise columns defensively (some DBs store counts / rounded probs)
        col = p.sum(axis=0, keepdims=True)
        col = np.where(col > 0, col, 1.0)
        p = p / col
        adj = (p + pseudocount * bg) / (1.0 + pseudocount)
        return np.log2(adj / bg).astype(np.float64)

    def reverse_complement(self) -> "PWM":
        """A,C,G,T → T,G,C,A and reverse the columns."""
        rc = self.probs[::-1, ::-1].copy()
        return PWM(self.motif_id + "_rc", self.tf_name, rc, self.source, dict(self.meta))
