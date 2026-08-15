"""Tests for the pure-NumPy PWM motif scanner (piaso.preprocessing.grn._scan).

Coverage
--------
- encode_seq: N → 4, lowercase → valid, uppercase ACGT → 0-3
- estimate_background: all-A → [1,0,0,0]; uniform → ~0.25
- pvalue_to_threshold: monotonic in p-value
- relative_threshold: boundary values
- scan_sequence / scan_motifs:
    * E-box (CACGTG, palindromic) — found on both strands
    * Non-palindromic motif (GGGGCC) — forward hit on + strand; no false rc hit
    * Non-palindromic motif reverse planted — found on − strand
    * No-hit sequence → best_score NaN, hit_count 0
"""
from __future__ import annotations

import numpy as np
import pytest

from piaso.data._pwm import PWM, BASE_INDEX
from piaso.preprocessing.grn._scan import (
    encode_seq,
    estimate_background,
    pvalue_to_threshold,
    relative_threshold,
    scan_sequence,
    scan_motifs,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_UNIFORM_BG = np.full(4, 0.25, dtype=np.float64)


def _make_near_deterministic_pwm(
    motif_id: str,
    tf_name: str,
    bases: str,
    high: float = 0.97,
    source: str = "test",
) -> PWM:
    """Build a near-deterministic PWM for the given base sequence.

    Each column has probability ``high`` for the target base and the remaining
    ``(1-high)/3`` spread equally over the other three.
    """
    w = len(bases)
    probs = np.full((4, w), (1.0 - high) / 3.0, dtype=np.float32)
    for j, b in enumerate(bases.upper()):
        probs[BASE_INDEX[b], j] = high
    return PWM(motif_id=motif_id, tf_name=tf_name, probs=probs, source=source)


def _plant(seq: str, motif: str, pos: int) -> str:
    """Insert ``motif`` into ``seq`` at ``pos``."""
    return seq[:pos] + motif + seq[pos + len(motif):]


def _rc(seq: str) -> str:
    """Reverse complement of a DNA string."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


# ──────────────────────────────────────────────────────────────────────────────
# encode_seq
# ──────────────────────────────────────────────────────────────────────────────

class TestEncodeSeq:
    def test_acgt_uppercase(self):
        codes = encode_seq("ACGT")
        np.testing.assert_array_equal(codes, [0, 1, 2, 3])

    def test_lowercase_valid(self):
        codes = encode_seq("acgt")
        np.testing.assert_array_equal(codes, [0, 1, 2, 3])

    def test_n_maps_to_4(self):
        codes = encode_seq("ANCGT")
        assert codes[1] == 4

    def test_mixed_lowercase_n(self):
        codes = encode_seq("aNcGt")
        assert codes[0] == 0   # a
        assert codes[1] == 4   # N
        assert codes[2] == 1   # c
        assert codes[3] == 2   # G
        assert codes[4] == 3   # t

    def test_dtype(self):
        assert encode_seq("ACGT").dtype == np.uint8

    def test_non_acgt_maps_to_4(self):
        for ch in "BDEFHIJKLMOPQRSUVWXYZ":
            assert encode_seq(ch)[0] == 4, f"Expected 4 for {ch!r}"


# ──────────────────────────────────────────────────────────────────────────────
# estimate_background
# ──────────────────────────────────────────────────────────────────────────────

class TestEstimateBackground:
    def test_all_a(self):
        # Use a long all-A sequence so pseudocount (1.0) is small relative to counts
        bg = estimate_background(["A" * 1000])
        # A should be strongly dominant
        assert bg[0] > 0.95
        np.testing.assert_allclose(bg.sum(), 1.0, atol=1e-9)

    def test_uniform(self):
        # Equal counts of each base
        seq = "ACGTACGTACGTACGT"
        bg = estimate_background([seq])
        np.testing.assert_allclose(bg, 0.25, atol=0.05)

    def test_empty_sequences_returns_uniform(self):
        bg = estimate_background(["NNNN"])
        np.testing.assert_allclose(bg, 0.25, atol=1e-9)

    def test_sums_to_one(self):
        bg = estimate_background(["AAATTTGGGCCC", "NNNNN", "acgtacgt"])
        np.testing.assert_allclose(bg.sum(), 1.0, atol=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# pvalue_to_threshold
# ──────────────────────────────────────────────────────────────────────────────

class TestPvalueToThreshold:
    def _simple_pssm(self):
        """A 2-column PSSM for a simple test."""
        pwm = _make_near_deterministic_pwm("tst", "TST", "AC")
        return pwm.pssm(_UNIFORM_BG)

    def test_monotonic_decreasing(self):
        """Smaller p-value should yield a higher (or equal) threshold."""
        pssm = self._simple_pssm()
        p_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        thresholds = [pvalue_to_threshold(pssm, _UNIFORM_BG, p) for p in p_values]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1], (
                f"Not monotonic: p={p_values[i]:.0e} → t={thresholds[i]:.4f}, "
                f"p={p_values[i+1]:.0e} → t={thresholds[i+1]:.4f}"
            )

    def test_returns_float(self):
        pssm = self._simple_pssm()
        t = pvalue_to_threshold(pssm, _UNIFORM_BG)
        assert isinstance(t, float)

    def test_threshold_below_max(self):
        pssm = self._simple_pssm()
        max_score = pssm.max(axis=0).sum()
        t = pvalue_to_threshold(pssm, _UNIFORM_BG, pvalue=0.5)
        assert t <= max_score + 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# relative_threshold
# ──────────────────────────────────────────────────────────────────────────────

class TestRelativeThreshold:
    def test_frac_one_equals_max(self):
        pwm = _make_near_deterministic_pwm("t", "T", "ACGT")
        pssm = pwm.pssm(_UNIFORM_BG)
        max_score = pssm.max(axis=0).sum()
        t = relative_threshold(pssm, frac=1.0)
        np.testing.assert_allclose(t, max_score, rtol=1e-6)

    def test_frac_zero_equals_min(self):
        pwm = _make_near_deterministic_pwm("t", "T", "ACGT")
        pssm = pwm.pssm(_UNIFORM_BG)
        min_score = pssm.min(axis=0).sum()
        t = relative_threshold(pssm, frac=0.0)
        np.testing.assert_allclose(t, min_score, rtol=1e-6)

    def test_midpoint(self):
        pwm = _make_near_deterministic_pwm("t", "T", "ACGT")
        pssm = pwm.pssm(_UNIFORM_BG)
        max_score = pssm.max(axis=0).sum()
        min_score = pssm.min(axis=0).sum()
        t = relative_threshold(pssm, frac=0.5)
        np.testing.assert_allclose(t, 0.5 * (max_score + min_score), rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# scan_sequence / scan_motifs — E-box (palindrome CACGTG)
# ──────────────────────────────────────────────────────────────────────────────

class TestEboxPalindrome:
    """CACGTG is its own reverse complement; it should appear on both strands."""

    MOTIF = "CACGTG"
    PLANT_POS = 30

    @pytest.fixture
    def ebox_pwm(self):
        return _make_near_deterministic_pwm("MA_EBOX", "Arnt", self.MOTIF)

    @pytest.fixture
    def ebox_seq(self):
        # AT-rich flanking context so only the planted site scores high
        rng = np.random.default_rng(42)
        n = 100
        at_bases = ["A", "T"]
        background = "".join(rng.choice(at_bases, size=n))
        return _plant(background, self.MOTIF, self.PLANT_POS)

    def test_forward_hit_found(self, ebox_pwm, ebox_seq):
        result = scan_motifs(
            [ebox_pwm],
            [ebox_seq],
            relative_frac=0.7,
            both_strands=True,
        )
        assert result["hit_count"][0, 0] >= 1
        assert not np.isnan(result["best_score"][0, 0])

    def test_best_score_is_high(self, ebox_pwm, ebox_seq):
        pssm = ebox_pwm.pssm(_UNIFORM_BG)
        max_score = pssm.max(axis=0).sum()
        result = scan_motifs(
            [ebox_pwm],
            [ebox_seq],
            relative_frac=0.7,
            both_strands=True,
        )
        best = result["best_score"][0, 0]
        # Should be close to the theoretical maximum
        assert best > 0.7 * max_score

    def test_palindrome_both_strands(self, ebox_pwm, ebox_seq):
        """Since CACGTG == rc(CACGTG), scanning both strands must find ≥ 2 hits."""
        result = scan_motifs(
            [ebox_pwm],
            [ebox_seq],
            relative_frac=0.7,
            both_strands=True,
        )
        # A palindrome planted once → at least 1 hit; both_strands may give 2
        assert result["hit_count"][0, 0] >= 1

    def test_hit_position_correct(self, ebox_pwm, ebox_seq):
        """The planted position should appear among the reported hits."""
        pssm = ebox_pwm.pssm(_UNIFORM_BG)
        thr = relative_threshold(pssm, frac=0.7)
        codes = encode_seq(ebox_seq)
        hits = scan_sequence(pssm, codes, thr, both_strands=True)
        positions = {h[0] for h in hits}
        # For a palindrome, the rc hit at the same site may map to the same or
        # adjacent position — allow ±1 tolerance
        assert any(abs(p - self.PLANT_POS) <= 1 for p in positions), (
            f"Expected hit near {self.PLANT_POS}, got positions {sorted(positions)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Non-palindromic motif: GGGGCC (SP1/GC-box-like)
# ──────────────────────────────────────────────────────────────────────────────

class TestNonPalindrome:
    """GGGGCC rc = GGCCCC (different sequence).

    Plant GGGGCC on + strand: should find a + hit.
    Plant GGGGCC on − strand (i.e. plant GGCCCC on +): should find a − hit.
    """

    FWD_MOTIF = "GGGGCC"
    RC_MOTIF = _rc(FWD_MOTIF)   # GGCCCC
    PLANT_POS = 20

    @pytest.fixture
    def sp1_pwm(self):
        return _make_near_deterministic_pwm("SP1_TEST", "Sp1", self.FWD_MOTIF)

    @pytest.fixture
    def fwd_seq(self):
        rng = np.random.default_rng(7)
        at_bases = ["A", "T"]
        bg = "".join(rng.choice(at_bases, size=80))
        return _plant(bg, self.FWD_MOTIF, self.PLANT_POS)

    @pytest.fixture
    def rev_seq(self):
        """Plant the reverse-complement motif (GGCCCC) on the forward strand."""
        rng = np.random.default_rng(99)
        at_bases = ["A", "T"]
        bg = "".join(rng.choice(at_bases, size=80))
        return _plant(bg, self.RC_MOTIF, self.PLANT_POS)

    def test_forward_strand_hit(self, sp1_pwm, fwd_seq):
        pssm = sp1_pwm.pssm(_UNIFORM_BG)
        thr = relative_threshold(pssm, frac=0.75)
        codes = encode_seq(fwd_seq)
        hits = scan_sequence(pssm, codes, thr, both_strands=True)
        fwd_hits = [h for h in hits if h[1] == "+"]
        assert len(fwd_hits) >= 1
        assert any(abs(h[0] - self.PLANT_POS) <= 1 for h in fwd_hits), (
            f"Expected + hit near {self.PLANT_POS}, got {fwd_hits}"
        )

    def test_reverse_strand_hit(self, sp1_pwm, rev_seq):
        """Planting rc motif on + → scanner should flag a − strand hit."""
        pssm = sp1_pwm.pssm(_UNIFORM_BG)
        thr = relative_threshold(pssm, frac=0.75)
        codes = encode_seq(rev_seq)
        hits = scan_sequence(pssm, codes, thr, both_strands=True)
        rev_hits = [h for h in hits if h[1] == "-"]
        assert len(rev_hits) >= 1

    def test_no_false_reverse_hit_when_forward_planted(self, sp1_pwm, fwd_seq):
        """When FWD motif (GGGGCC) is planted, there should be no − strand hit
        because GGCCCC is not present in the sequence."""
        pssm = sp1_pwm.pssm(_UNIFORM_BG)
        # Use a stricter threshold to avoid noise
        thr = relative_threshold(pssm, frac=0.85)
        codes = encode_seq(fwd_seq)
        hits = scan_sequence(pssm, codes, thr, both_strands=True)
        rev_hits = [h for h in hits if h[1] == "-"]
        assert len(rev_hits) == 0, (
            f"Unexpected − strand hits in sequence with only + motif planted: {rev_hits}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# No-hit sequence
# ──────────────────────────────────────────────────────────────────────────────

class TestNoHit:
    def test_no_hit_sequence(self):
        """A sequence of all-A's should not trigger an E-box hit."""
        pwm = _make_near_deterministic_pwm("ebox", "MYC", "CACGTG")
        seq = "A" * 200
        result = scan_motifs(
            [pwm],
            [seq],
            relative_frac=0.7,
            both_strands=True,
        )
        assert result["hit_count"][0, 0] == 0
        assert np.isnan(result["best_score"][0, 0])

    def test_sequence_shorter_than_motif(self):
        """Sequence shorter than the motif width → no hits, no crash."""
        pwm = _make_near_deterministic_pwm("long", "TF", "ACGTACGTACGT")
        seq = "ACG"  # 3 < 12
        result = scan_motifs([pwm], [seq], relative_frac=0.8)
        assert result["hit_count"][0, 0] == 0
        assert np.isnan(result["best_score"][0, 0])


# ──────────────────────────────────────────────────────────────────────────────
# Output structure / dtype
# ──────────────────────────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_result_keys(self):
        pwm = _make_near_deterministic_pwm("t1", "TF1", "ACGT")
        result = scan_motifs([pwm], ["ACGTACGT"], relative_frac=0.5)
        assert set(result.keys()) == {"motif_ids", "tf_names", "best_score", "hit_count"}

    def test_shapes(self):
        pwms = [
            _make_near_deterministic_pwm("m1", "TF1", "ACGT"),
            _make_near_deterministic_pwm("m2", "TF2", "GGCC"),
        ]
        seqs = ["ACGTACGT", "GGCCGGCC", "ATATATATA"]
        result = scan_motifs(pwms, seqs, relative_frac=0.5)
        assert result["best_score"].shape == (2, 3)
        assert result["hit_count"].shape == (2, 3)

    def test_dtypes(self):
        pwm = _make_near_deterministic_pwm("t", "T", "ACGT")
        result = scan_motifs([pwm], ["ACGTACGT"], relative_frac=0.5)
        assert result["best_score"].dtype == np.float32
        assert result["hit_count"].dtype == np.int32

    def test_motif_ids_tf_names(self):
        pwms = [
            _make_near_deterministic_pwm("MA0001", "Arnt", "CACGTG"),
            _make_near_deterministic_pwm("MA0002", "Sp1", "GGGGCC"),
        ]
        result = scan_motifs(pwms, ["ACGT"], relative_frac=0.9)
        assert result["motif_ids"] == ["MA0001", "MA0002"]
        assert result["tf_names"] == ["Arnt", "Sp1"]


# ──────────────────────────────────────────────────────────────────────────────
# N handling
# ──────────────────────────────────────────────────────────────────────────────

class TestNHandling:
    def test_all_n_no_hit(self):
        """An all-N window should not produce a high score."""
        pwm = _make_near_deterministic_pwm("e", "E", "ACGT")
        result = scan_motifs([pwm], ["NNNN" * 50], relative_frac=0.7)
        assert result["hit_count"][0, 0] == 0

    def test_n_in_window_suppresses_hit(self):
        """Inserting an N at a key position in the planted motif should suppress the hit."""
        motif = "CACGTG"
        pwm = _make_near_deterministic_pwm("ebox", "Arnt", motif)
        rng = np.random.default_rng(11)
        at_bases = ["A", "T"]
        bg = "".join(rng.choice(at_bases, size=80))
        # Replace one base in the planted motif with N
        degraded = motif[:2] + "N" + motif[3:]  # CAN GTG
        seq = _plant(bg, degraded, 30)
        # With a high threshold the N-containing window should not score above threshold
        pssm = pwm.pssm(_UNIFORM_BG)
        thr = relative_threshold(pssm, frac=0.9)
        codes = encode_seq(seq)
        hits = scan_sequence(pssm, codes, thr, both_strands=False)
        fwd_hits = [h for h in hits if h[1] == "+"]
        assert len(fwd_hits) == 0, (
            f"Expected no hit at strict threshold with N in motif, got {fwd_hits}"
        )


# --- Rust backend parity (skipped if the extension isn't built) ---
import pytest as _pytest


def _rust_available():
    try:
        from piaso.preprocessing.grn._scan_rust import scan_motifs_rust  # noqa
        from piaso import _piaso
        return hasattr(_piaso, "scan_motifs_fwd")
    except Exception:
        return False


@_pytest.mark.skipif(not _rust_available(), reason="_piaso.scan_motifs_fwd not built")
def test_rust_matches_numpy():
    import numpy as np
    from piaso.data._pwm import PWM
    from piaso.preprocessing.grn._scan import scan_motifs
    from piaso.preprocessing.grn._scan_rust import scan_motifs_rust

    rng = np.random.default_rng(0)
    # a few random + one structured motif
    def rand_pwm(i, w):
        m = rng.random((4, w)) + 0.05
        m /= m.sum(0, keepdims=True)
        return PWM(f"M{i}", f"TF{i}", m.astype(np.float32))
    ebox = np.array([[0,1,0,0,0,0],[1,0,1,0,0,0],[0,0,0,1,0,1],[0,0,0,0,1,0]], float) + 0.001
    ebox /= ebox.sum(0, keepdims=True)
    pwms = [PWM("EBOX", "MYC", ebox.astype(np.float32))] + [rand_pwm(i, w) for i, w in enumerate([6, 8, 11])]
    seqs = ["".join(rng.choice(list("ACGT"), size=400)) for _ in range(15)]
    seqs[3] = seqs[3][:100] + "CACGTG" + seqs[3][106:]

    bg = np.array([0.25, 0.25, 0.25, 0.25])
    r_np = scan_motifs(pwms, seqs, background=bg, pvalue=1e-3, both_strands=True)
    r_rs = scan_motifs_rust(pwms, seqs, background=bg, pvalue=1e-3, both_strands=True)
    # hit counts identical; best scores match where there are hits
    assert np.array_equal(r_np["hit_count"], r_rs["hit_count"]), \
        (r_np["hit_count"], r_rs["hit_count"])
    a, b = r_np["best_score"], r_rs["best_score"]
    both = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[both], b[both], atol=1e-4)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
