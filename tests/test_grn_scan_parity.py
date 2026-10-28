"""Motif scanning: the numpy reference against the Rust kernel, and the p-value threshold at its edges.

From the code audit of 2026-10-27 (part 3, "Motif scanning"; review I1, finding 2):

- a motif whose best word alone is more likely than the p-value has no threshold: ``pvalue_to_threshold``
  returns +inf (it used to return the maximum score, or, when rounding pushed the best words off the DP's
  grid, a finite threshold below it) and both backends report no hits, with one warning naming the motifs;
- the DP keeps the mass that rounding carries past ``n_bins`` (it was dropped and the rest renormalised,
  which made thresholds up to twenty times too lenient on short motifs);
- numpy and Rust score windows in float64 in the same order, so they agree to the bit; this is tested on
  ``_scan_motifs_numpy`` itself (the earlier parity test compared Rust with Rust through ``scan_motifs``);
- a minus-strand hit is reported at the forward start of the window it occupies (it was mirrored).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from piaso.data._pwm import PWM, BASE_INDEX
from piaso.preprocessing.grn import _scan as S

UNIFORM = np.full(4, 0.25)
AT_RICH = np.array([0.3, 0.2, 0.2, 0.3])


def _rust_available():
    try:
        from piaso import _piaso
        return hasattr(_piaso, "scan_motifs_fwd")
    except Exception:
        return False


needs_rust = pytest.mark.skipif(not _rust_available(), reason="_piaso.scan_motifs_fwd not built")


def _det(motif_id, bases, high=0.97):
    probs = np.full((4, len(bases)), (1.0 - high) / 3.0)
    for j, b in enumerate(bases):
        probs[BASE_INDEX[b], j] = high
    return PWM(motif_id, motif_id, probs.astype(np.float32))


def _best_word(pwm, bg):
    pssm = pwm.pssm(bg, pseudocount=0.01)
    return "".join("ACGT"[i] for i in pssm.argmax(axis=0))


def _rc(s):
    return s.translate(str.maketrans("ACGTacgtN", "TGCAtgcaN"))[::-1]


def _exact_tail(pssm, bg, t):
    """P(score >= t) under the background, by enumerating every word (4**w)."""
    if not math.isfinite(t):
        return 0.0
    s, p = np.zeros(1), np.ones(1)
    for j in range(pssm.shape[1]):
        s = (s[:, None] + pssm[:, j][None, :]).ravel()
        p = (p[:, None] * bg[None, :]).ravel()
    return float(p[s >= t - 1e-12].sum())


def _motifs(rng):
    """Short consensus motifs (no threshold at 1e-4), wide ones whose flanks are uniform (the best word ties
    across bases in those columns, and from eight columns on a pairwise sum is not the kernel's), and random
    ones of every width."""
    pwms = [_det(f"DET{i}", s) for i, s in enumerate(["CACGTG", "GATAA", "TAATTA", "CACGTGAC", "TGACTCAT"])]
    for i in range(6):
        core = rng.dirichlet(np.full(4, 0.3), size=6).T
        n_flank = int(rng.integers(2, 9))
        flank = np.full((4, n_flank), 0.25)
        k = int(rng.integers(0, n_flank + 1))
        pwms.append(PWM(f"FLANK{i}", f"FLANK{i}",
                        np.concatenate([flank[:, :k], core, flank[:, k:]], axis=1).astype(np.float32)))
    for i in range(12):
        w = int(rng.integers(4, 21))
        m = rng.random((4, w)) + 0.02
        pwms.append(PWM(f"RND{i}", f"RND{i}", (m / m.sum(0, keepdims=True)).astype(np.float32)))
    return pwms


def _sequences(rng, pwms, bg):
    """Random sequences with N, soft-masked bases and very short ones; half carry some motif's best word,
    on either strand, so windows tie the extreme thresholds."""
    alpha = np.array(list("ACGTNacgt"))
    p = np.array([0.23, 0.23, 0.23, 0.23, 0.02, 0.015, 0.015, 0.015, 0.015])
    seqs = []
    for i in range(70):
        n = int(rng.integers(3, 500))
        s = "".join(rng.choice(alpha, size=n, p=p / p.sum()))
        if i % 2 == 0:
            word = _best_word(pwms[i % len(pwms)], bg)
            word = word if i % 4 == 0 else _rc(word)
            at = int(rng.integers(0, max(1, len(s) - len(word) + 1)))
            s = s[:at] + word + s[at + len(word):]
        seqs.append(s)
    return seqs


@needs_rust
@pytest.mark.parametrize("kw", [dict(pvalue=1e-4), dict(pvalue=1e-3), dict(pvalue=0.05),
                                dict(relative_frac=0.0), dict(relative_frac=0.8), dict(relative_frac=1.0)],
                         ids=["p1e-4", "p1e-3", "p0.05", "rel0", "rel0.8", "rel1"])
@pytest.mark.parametrize("bg_name", ["uniform", "at_rich", "estimated"])
def test_numpy_reference_matches_rust_to_the_bit(kw, bg_name):
    from piaso.preprocessing.grn._scan_rust import scan_motifs_rust

    rng = np.random.default_rng(20261027)
    pwms = _motifs(rng)
    bg = {"uniform": UNIFORM, "at_rich": AT_RICH, "estimated": None}[bg_name]
    seqs = _sequences(rng, pwms, UNIFORM if bg is None else bg)
    for both in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", S.MotifThresholdWarning)
            a = S._scan_motifs_numpy(pwms, seqs, background=bg, both_strands=both, **kw)
            r = scan_motifs_rust(pwms, seqs, background=bg, both_strands=both, **kw)
        diff = np.argwhere(a["hit_count"] != r["hit_count"])
        assert diff.size == 0, [(pwms[m].motif_id, int(s), int(a["hit_count"][m, s]), int(r["hit_count"][m, s]))
                                for m, s in diff[:10]]
        assert np.array_equal(a["best_score"], r["best_score"], equal_nan=True)
    if kw.get("relative_frac") == 1.0:
        # the planted best words tie the threshold exactly: hits on both backends
        assert (a["hit_count"] > 0).sum() >= 10


@pytest.mark.parametrize("bases,p_best", [("CACGTG", 0.25 ** 6), ("GATAA", 0.25 ** 5)])
def test_unreachable_pvalue_threshold_is_inf(bases, p_best):
    """CACGTG: rounding carried its best word past the grid, and the old DP called a threshold below the
    maximum (the best word, p = 2.4e-4, was a hit at 1e-4). GATAA: the old DP fell back to the maximum."""
    pwm = _det("M", bases)
    pssm = pwm.pssm(UNIFORM)
    with pytest.warns(S.MotifThresholdWarning):
        assert S.pvalue_to_threshold(pssm, UNIFORM, 1e-4) == math.inf
    thr, pb = S._pvalue_threshold(pssm, UNIFORM, 1e-4)
    assert thr == math.inf and pb == pytest.approx(p_best, rel=0.25)
    # just above the best word's probability the threshold admits the best word and nothing else
    thr = S.pvalue_to_threshold(pssm, UNIFORM, 1.001 * pb)
    assert math.isfinite(thr) and _exact_tail(pssm, UNIFORM, thr) == pytest.approx(pb)
    word = _best_word(pwm, UNIFORM)
    assert S.scan_sequence(pssm, S.encode_seq("A" * 10 + word + "A" * 10), thr, both_strands=False)


@pytest.mark.parametrize("scanner", ["numpy", pytest.param("rust", marks=needs_rust)])
def test_unreachable_motif_reports_no_hits_and_is_named(scanner):
    from piaso.preprocessing.grn._scan_rust import scan_motifs_rust
    scan = S._scan_motifs_numpy if scanner == "numpy" else scan_motifs_rust
    short = _det("MA_SHORT", "CACGTG")
    long_ = _det("MA_LONG", "CACGTGACGTTA")       # best word p = 6e-8: reachable
    seq = "AT" * 15 + "CACGTGACGTTA" + "TA" * 15
    with pytest.warns(S.MotifThresholdWarning, match=r"1 of 2 motifs.*MA_SHORT \(MA_SHORT, best word p=2\.4e-04\)"):
        r = scan([short, long_], [seq], background=UNIFORM, pvalue=1e-4)
    assert r["hit_count"][0, 0] == 0 and np.isnan(r["best_score"][0, 0])
    assert r["hit_count"][1, 0] >= 1 and np.isfinite(r["best_score"][1, 0])


def test_threshold_tail_stays_near_the_target():
    """The exact tail at the threshold, by enumeration, for random motifs of 5 to 8 columns: a finite
    threshold keeps it within the grid's resolution (under 2.6 times the target on 800 motifs; the old DP,
    which dropped the top mass, reached 24 times), and +inf comes exactly when the best word's own
    probability exceeds the target."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for bg in (UNIFORM, AT_RICH):
        for _ in range(60):
            w = int(rng.integers(5, 9))
            pssm = PWM("m", "m", rng.dirichlet(np.full(4, 0.3), size=w).T.astype(np.float32)).pssm(bg, 0.01)
            thr, p_best = S._pvalue_threshold(pssm, bg, 1e-4)
            assert (thr == math.inf) == (p_best > 1e-4)
            if math.isfinite(thr):
                worst = max(worst, _exact_tail(pssm, bg, thr))
    assert 0 < worst <= 3e-4


def test_minus_strand_hit_is_reported_at_its_forward_start():
    pwm = _det("SP1", "GGGGCC")
    pssm = pwm.pssm(UNIFORM)
    seq = "A" * 20 + "GGCCCC" + "T" * 54           # the motif's reverse complement at 20 (L = 80)
    hits = S.scan_sequence(pssm, S.encode_seq(seq), S.relative_threshold(pssm, 0.85), both_strands=True)
    assert hits == [(20, "-", pytest.approx(S._seq_sum(pssm.max(axis=0))))]
