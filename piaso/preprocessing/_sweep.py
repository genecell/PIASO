"""
Sweep-line intersection for fragment x peak overlap detection.

Same O(n + m + k) algorithm as ``bedtools intersect -sorted``.

The sweep function is a **generator** — it yields hits one at a time and
never materialises a list.  This keeps memory at O(1) for the hit stream
regardless of dataset size.

The function accepts **parallel arrays** (``array.array`` or numpy) instead
of Python tuples, keeping per-fragment memory at 12 bytes instead of ~140.
"""

from collections import defaultdict
from typing import Dict, Iterator, List, Tuple


# ===================================================================
#  Peak loading
# ===================================================================

def load_peaks(peak_file: str) -> Tuple[Dict[str, list], list, int]:
    """
    Load peaks from a BED / narrowPeak / broadPeak file.

    Blank, ``#``, ``track`` and ``browser`` lines are skipped; a line with fewer
    than three columns or with non-integer coordinates raises ``ValueError``;
    intervals with end <= start are left out (they can count nothing), as in
    the Rust quantifier.

    Parameters
    ----------
    peak_file : str
        Path to peak file.

    Returns
    -------
    peaks_by_chr : dict
        ``{chrom: [(start, end, global_idx), ...]}`` sorted by *start*
        within each chromosome.
    peak_names : list of str
        ``"{chrom}:{start}-{end}"`` for each peak, in file order.
    n_peaks : int
        Total peak count.
    """
    peaks_by_chr: Dict[str, list] = defaultdict(list)
    peak_names: List[str] = []
    global_idx = 0

    with open(peak_file, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            if (not line.strip() or line.startswith("#") or line.startswith("track")
                    or line.startswith("browser")):
                continue
            parts = line.rstrip("\r\n").split("\t")
            # A short or non-numeric line is an error, as in the Rust reader:
            # skipped, or read as 0, it changed the peak set without a word.
            if len(parts) < 3:
                raise ValueError(f"{peak_file}: line {lineno} has fewer than three "
                                 f"tab-separated columns: {line[:80]!r}")
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                raise ValueError(f"{peak_file}: line {lineno}: start and end must be "
                                 f"integers: {line[:80]!r}") from None

            if start >= end:
                continue

            peak_names.append(f"{chrom}:{start}-{end}")
            peaks_by_chr[chrom].append((start, end, global_idx))
            global_idx += 1

    for chrom in peaks_by_chr:
        peaks_by_chr[chrom].sort(key=lambda x: x[0])

    return dict(peaks_by_chr), peak_names, global_idx


# ===================================================================
#  Sweep-line intersection (generator, parallel-array input)
# ===================================================================

def sweep_intersect(
    cell_indices,
    starts,
    ends,
    peaks_sorted: list,
    counting: str = "overlaps",
) -> Iterator[Tuple[int, int]]:
    """
    Generator sweep-line intersection.  Yields hits one at a time — O(1)
    memory for the hit stream.

    Parameters
    ----------
    cell_indices : array-like of int
        Cell index per fragment.
    starts : array-like of int
        Fragment start positions (**sorted ascending** within one chromosome).
    ends : array-like of int
        Fragment end positions.
    peaks_sorted : list of (int, int, int)
        ``(peak_start, peak_end, peak_global_idx)`` for **one** chromosome,
        sorted by *peak_start*.

    Yields
    ------
    (cell_idx, peak_global_idx) : tuple of int
        One pair per fragment-peak overlap.

    Overlap condition (half-open intervals)::

        peak_start < frag_end   AND   peak_end > frag_start

    Adjacent intervals (``peak_end == frag_start``) are **not** overlaps.

    Complexity: O(n + m + k) with n fragments, m peaks, k overlaps.
    """
    n_peaks = len(peaks_sorted)
    if n_peaks == 0:
        return
    if counting not in ("overlaps", "insertions"):
        raise ValueError(f"counting must be 'overlaps' or 'insertions', got {counting!r}")

    n_frags = len(starts)
    active_start = 0

    if counting == "insertions":
        # The two Tn5 cut sites, each a point: a peak holds a point p when
        # peak_start <= p < peak_end. One hit per cut site per peak.
        for i in range(n_frags):
            frag_start = int(starts[i])
            frag_end = int(ends[i])
            cell_idx = int(cell_indices[i])
            cut_b = max(frag_end - 1, frag_start)
            while (active_start < n_peaks
                   and peaks_sorted[active_start][1] <= frag_start):
                active_start += 1
            for p in range(active_start, n_peaks):
                peak_start, peak_end, peak_global_idx = peaks_sorted[p]
                if peak_start > cut_b:
                    break
                if peak_start <= frag_start < peak_end:
                    yield cell_idx, peak_global_idx
                if peak_start <= cut_b < peak_end:
                    yield cell_idx, peak_global_idx
        return

    for i in range(n_frags):
        frag_start = int(starts[i])
        frag_end = int(ends[i])
        cell_idx = int(cell_indices[i])

        while (
            active_start < n_peaks
            and peaks_sorted[active_start][1] <= frag_start
        ):
            active_start += 1

        for p in range(active_start, n_peaks):
            peak_start, peak_end, peak_global_idx = peaks_sorted[p]
            if peak_start >= frag_end:
                break
            # Must also verify peak_end > frag_start. The active_start
            # pointer only guarantees this for the *first* active peak;
            # subsequent peaks sorted by start may have shorter end
            # positions that don't actually reach frag_start.
            if peak_end > frag_start:
                yield cell_idx, peak_global_idx
