//! PWM motif scanning hot loop (parallel over motifs, on a pool of its own).
//!
//! Deliberately minimal: the Python wrapper (`piaso/grn/_scan_rust.py`) computes
//! the log-odds PSSMs, the per-motif score thresholds (DP p-value or relative),
//! the 5-row N-augmented layout, AND the reverse-complement PSSM — then calls
//! this for FORWARD-only scanning. Reverse strand = a second PSSM passed in by
//! the wrapper. That keeps the numeric definitions in one (tested) place and
//! makes the Rust result bit-match the numpy reference, while the O(motifs ×
//! sequences × windows) inner loop runs in parallel native code.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Scan `n_pssm` PSSMs against `seqs` (forward only), returning best score and
/// hit count per (pssm, seq).
///
/// * `pssms` — concatenated row-major 5×w blocks (rows A,C,G,T,N-fill), one per
///   PSSM, in `widths` order. Offset of pssm p = sum(5*widths[0..p]).
/// * `widths` — width w of each PSSM.
/// * `thresholds` — score cutoff per PSSM (hit iff window score >= threshold).
/// * `seq_codes` — ALL sequences encoded as codes 0..=4 (A,C,G,T,N), concatenated
///   into one flat buffer (CSR-style). Sequence `i` is
///   `seq_codes[seq_offsets[i]..seq_offsets[i+1]]`.
/// * `seq_offsets` — `n_seq + 1` cumulative offsets into `seq_codes`
///   (`offsets[0] = 0`, `offsets[n_seq] = seq_codes.len()`).
///
/// Flat-buffer layout (vs the old `Vec<Vec<u8>>`): one contiguous allocation and
/// one PyO3 copy instead of N per-sequence allocations + N FFI conversions —
/// faster (cache-friendly, fewer allocs) and lower peak RAM; output is identical.
///
/// Returns `(best_score [n_pssm, n_seq] f32, hit_count [n_pssm, n_seq] i32)`.
/// `best_score` is `f32::NAN` where a sequence is shorter than the motif, else the
/// max window score seen — the wrapper gates the cistrome via `hit_count`.
/// `n_threads` — 0 (default): RAYON_NUM_THREADS, else the cores this process
/// may use. The scan runs on a pool of its own with Python's lock released,
/// so it is safe after a fork, other Python threads run meanwhile, and
/// Ctrl-C stops it within about 1,024 sequences of one motif.
#[pyfunction]
#[pyo3(signature = (pssms, widths, thresholds, seq_codes, seq_offsets, n_threads=0))]
pub fn scan_motifs_fwd<'py>(
    py: Python<'py>,
    pssms: Vec<f64>,
    widths: Vec<usize>,
    thresholds: Vec<f64>,
    seq_codes: Vec<u8>,
    seq_offsets: Vec<usize>,
    n_threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
    let n_pssm = widths.len();
    let n_seq = seq_offsets.len().saturating_sub(1);

    // Offset (in f64 elements) of each PSSM block: 5 rows * w each.
    let mut offsets = Vec::with_capacity(n_pssm);
    let mut acc = 0usize;
    for &w in &widths {
        offsets.push(acc);
        acc += 5 * w;
    }
    if acc > pssms.len() || thresholds.len() < n_pssm
        || seq_offsets.last().copied().unwrap_or(0) > seq_codes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scan_motifs_fwd: pssms, thresholds or seq_offsets do not match widths and seq_codes"));
    }
    if seq_codes.iter().any(|&c| c > 4) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scan_motifs_fwd: seq_codes must be 0..=4 (A, C, G, T, N)"));
    }

    // The output is written in place, one row per PSSM: the rows are never
    // held a second time (8 bytes per motif and sequence saved at the end).
    let mut best = vec![f32::NAN; n_pssm * n_seq];
    let mut count = vec![0i32; n_pssm * n_seq];
    let (pssms, thresholds, seq_codes, seq_offsets) = (&pssms, &thresholds, &seq_codes, &seq_offsets);
    let (widths_r, offsets_r) = (&widths, &offsets);
    let (best_r, count_r) = (&mut best, &mut count);
    crate::threads::run_released(py, "scan_motifs_fwd", n_threads, move |cancel| {
        best_r
            .par_chunks_mut(n_seq.max(1))
            .zip(count_r.par_chunks_mut(n_seq.max(1)))
            .enumerate()
            .for_each(|(p, (best_row, cnt_row))| {
                let w = widths_r[p];
                scan_row(&pssms[offsets_r[p]..offsets_r[p] + 5 * w], w, thresholds[p],
                         seq_codes, seq_offsets, best_row, cnt_row, cancel);
            });
    })?;

    let best = Array2::from_shape_vec((n_pssm, n_seq), best)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let count = Array2::from_shape_vec((n_pssm, n_seq), count)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((best.into_pyarray(py), count.into_pyarray(py)))
}

/// One PSSM against every sequence, into its output row.
///
/// The PSSM is copied position-major (5 scores per position) so a window is
/// scored by walking it and the columns together, with no index arithmetic
/// or bounds checks in the inner loop. The terms are added in the same order
/// as before, so the scores are identical to the bit. The indexed loop it
/// replaces ran at 10 s or 13.6 s on the same input depending on unrelated
/// code elsewhere in the library.
#[allow(clippy::too_many_arguments)]
fn scan_row(pssm: &[f64], w: usize, thr: f64, seq_codes: &[u8], seq_offsets: &[usize],
            best_row: &mut [f32], cnt_row: &mut [i32], cancel: &std::sync::atomic::AtomicBool) {
    // pssm is row-major 5 x w (row = base A,C,G,T,N; column = position)
    let mut cols = vec![[0.0f64; 5]; w];
    for (j, col) in cols.iter_mut().enumerate() {
        for (c, v) in col.iter_mut().enumerate() {
            *v = pssm[c * w + j];
        }
    }
    for si in 0..best_row.len() {
        if si % 1024 == 0 && cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let seq = &seq_codes[seq_offsets[si]..seq_offsets[si + 1]];
        if seq.len() < w {
            continue;
        }
        let mut best = f64::NEG_INFINITY;
        let mut cnt = 0i32;
        for window in seq.windows(w) {
            let mut score = 0.0f64;
            for (col, &c) in cols.iter().zip(window) {
                score += col[(c as usize).min(4)];
            }
            if score > best {
                best = score;
            }
            if score >= thr {
                cnt += 1;
            }
        }
        if best > f64::NEG_INFINITY {
            best_row[si] = best as f32;
        }
        cnt_row[si] = cnt;
    }
}
