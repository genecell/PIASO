//! PWM motif scanning hot loop (rayon-parallel over motifs).
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
#[pyfunction]
pub fn scan_motifs_fwd<'py>(
    py: Python<'py>,
    pssms: Vec<f64>,
    widths: Vec<usize>,
    thresholds: Vec<f64>,
    seq_codes: Vec<u8>,
    seq_offsets: Vec<usize>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
    let n_pssm = widths.len();
    let n_seq = seq_offsets.len().saturating_sub(1);

    // Byte offset (in f64 elements) of each PSSM block: 5 rows * w each.
    let mut offsets = Vec::with_capacity(n_pssm);
    let mut acc = 0usize;
    for &w in &widths {
        offsets.push(acc);
        acc += 5 * w;
    }

    // Parallel over PSSMs; each produces a row of length n_seq for best + count.
    let rows: Vec<(Vec<f32>, Vec<i32>)> = (0..n_pssm)
        .into_par_iter()
        .map(|p| {
            let w = widths[p];
            let off = offsets[p];
            let thr = thresholds[p];
            let pssm = &pssms[off..off + 5 * w];
            let mut best_row = vec![f32::NAN; n_seq];
            let mut cnt_row = vec![0i32; n_seq];
            for si in 0..n_seq {
                let seq = &seq_codes[seq_offsets[si]..seq_offsets[si + 1]];
                let n = seq.len();
                if n < w {
                    continue;
                }
                let mut best = f64::NEG_INFINITY;
                let mut cnt = 0i32;
                for pos in 0..=(n - w) {
                    let mut score = 0.0f64;
                    for j in 0..w {
                        let c = seq[pos + j] as usize; // 0..=4
                        // pssm row-major: row c, column j  -> c*w + j
                        score += pssm[c * w + j];
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
            (best_row, cnt_row)
        })
        .collect();

    let mut best = Array2::<f32>::from_elem((n_pssm, n_seq), f32::NAN);
    let mut count = Array2::<i32>::zeros((n_pssm, n_seq));
    for (p, (br, cr)) in rows.into_iter().enumerate() {
        for s in 0..n_seq {
            best[[p, s]] = br[s];
            count[[p, s]] = cr[s];
        }
    }
    Ok((best.into_pyarray(py), count.into_pyarray(py)))
}
