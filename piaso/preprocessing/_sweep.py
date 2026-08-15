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
        for line in fh:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

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

    n_frags = len(starts)
    active_start = 0

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
