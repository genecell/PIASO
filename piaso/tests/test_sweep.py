"""Test sweep-line intersection algorithm correctness."""

import array

import numpy as np
import pytest

from piaso.preprocessing._sweep import load_peaks, sweep_intersect


def _make_arrays(tuples):
    """Convert list of (cell_idx, start, end) to parallel array.array('i')."""
    if not tuples:
        return array.array("i"), array.array("i"), array.array("i")
    cells = array.array("i", [t[0] for t in tuples])
    starts = array.array("i", [t[1] for t in tuples])
    ends = array.array("i", [t[2] for t in tuples])
    return cells, starts, ends


class TestSweepIntersect:
    """17 test cases covering all edge conditions."""

    def test_basic_overlap(self):
        c, s, e = _make_arrays([(0, 150, 250)])
        peaks = [(100, 200, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == [(0, 0)]

    def test_no_overlap_before(self):
        c, s, e = _make_arrays([(0, 50, 100)])
        peaks = [(200, 300, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == []

    def test_no_overlap_after(self):
        c, s, e = _make_arrays([(0, 300, 400)])
        peaks = [(100, 200, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == []

    def test_adjacent_peak_end_eq_frag_start(self):
        c, s, e = _make_arrays([(0, 200, 300)])
        peaks = [(100, 200, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == []

    def test_adjacent_frag_end_eq_peak_start(self):
        c, s, e = _make_arrays([(0, 100, 200)])
        peaks = [(200, 300, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == []

    def test_fragment_spans_multiple_peaks(self):
        c, s, e = _make_arrays([(0, 150, 500)])
        peaks = [(100, 200, 0), (250, 350, 1), (300, 400, 2)]
        assert set(sweep_intersect(c, s, e, peaks)) == {(0, 0), (0, 1), (0, 2)}

    def test_overlapping_peaks(self):
        c, s, e = _make_arrays([(0, 250, 350)])
        peaks = [(100, 300, 0), (200, 400, 1)]
        assert set(sweep_intersect(c, s, e, peaks)) == {(0, 0), (0, 1)}

    def test_fragment_contained_in_peak(self):
        c, s, e = _make_arrays([(0, 200, 300)])
        peaks = [(100, 500, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == [(0, 0)]

    def test_peak_contained_in_fragment(self):
        c, s, e = _make_arrays([(0, 100, 500)])
        peaks = [(200, 300, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == [(0, 0)]

    def test_multiple_cells_same_peak(self):
        c, s, e = _make_arrays([(0, 150, 250), (1, 160, 260), (2, 170, 270)])
        peaks = [(100, 300, 0)]
        assert set(sweep_intersect(c, s, e, peaks)) == {(0, 0), (1, 0), (2, 0)}

    def test_duplicate_hits_same_cell(self):
        c, s, e = _make_arrays([(5, 120, 180), (5, 200, 250)])
        peaks = [(100, 300, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == [(5, 0), (5, 0)]

    def test_empty_peaks(self):
        c, s, e = _make_arrays([(0, 100, 200)])
        assert list(sweep_intersect(c, s, e, [])) == []

    def test_empty_fragments(self):
        c, s, e = _make_arrays([])
        peaks = [(100, 200, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == []

    def test_single_base_peak(self):
        c, s, e = _make_arrays([(0, 100, 200)])
        peaks = [(150, 151, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == [(0, 0)]

    def test_exact_overlap(self):
        c, s, e = _make_arrays([(0, 100, 200)])
        peaks = [(100, 200, 0)]
        assert list(sweep_intersect(c, s, e, peaks)) == [(0, 0)]

    def test_global_idx_preserved(self):
        c, s, e = _make_arrays([(0, 150, 250)])
        peaks = [(100, 200, 42)]
        assert list(sweep_intersect(c, s, e, peaks)) == [(0, 42)]

    def test_brute_force_comparison(self):
        rng = np.random.RandomState(42)
        n_frags = 5000
        starts_np = np.sort(rng.randint(0, 10000, size=n_frags))
        lengths = rng.randint(100, 500, size=n_frags)
        ends_np = starts_np + lengths
        cell_np = np.arange(n_frags) % 50

        c = array.array("i", cell_np.tolist())
        s = array.array("i", starts_np.tolist())
        e = array.array("i", ends_np.tolist())

        peaks = sorted(
            [
                (int(ps), int(ps) + rng.randint(200, 2000), i)
                for i, ps in enumerate(rng.randint(0, 10000, size=500))
            ],
            key=lambda x: x[0],
        )

        hits = list(sweep_intersect(c, s, e, peaks))
        expected = []
        for i in range(n_frags):
            for ps, pe, pi in peaks:
                if ps < int(ends_np[i]) and pe > int(starts_np[i]):
                    expected.append((int(cell_np[i]), pi))
        assert sorted(hits) == sorted(expected)


class TestLoadPeaks:

    def test_bed_file(self, tmp_path):
        bed = tmp_path / "peaks.bed"
        bed.write_text("chr1\t100\t200\nchr1\t300\t400\nchr2\t50\t150\n")
        peaks_by_chr, names, n = load_peaks(str(bed))
        assert n == 3
        assert len(peaks_by_chr["chr1"]) == 2
        assert len(peaks_by_chr["chr2"]) == 1
        assert names == ["chr1:100-200", "chr1:300-400", "chr2:50-150"]

    def test_unsorted_peaks_get_sorted(self, tmp_path):
        bed = tmp_path / "peaks.bed"
        bed.write_text("chr1\t300\t400\nchr1\t100\t200\n")
        peaks_by_chr, _, _ = load_peaks(str(bed))
        starts = [p[0] for p in peaks_by_chr["chr1"]]
        assert starts == [100, 300]

    def test_comment_and_track_lines_skipped(self, tmp_path):
        bed = tmp_path / "peaks.bed"
        bed.write_text("# comment\ntrack name=peaks\nchr1\t100\t200\n")
        _, _, n = load_peaks(str(bed))
        assert n == 1

    def test_malformed_start_ge_end_skipped(self, tmp_path):
        bed = tmp_path / "peaks.bed"
        bed.write_text(
            "chr1\t200\t100\n"
            "chr1\t500\t500\n"
            "chr1\t300\t400\n"
        )
        peaks_by_chr, names, n = load_peaks(str(bed))
        assert n == 1
        assert names == ["chr1:300-400"]
