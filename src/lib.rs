// --- Shared modules used by both PyO3 extension and piaso-atac CLI ---
pub mod encoder;
pub mod cytome_reader;
pub mod motif_scan;

// --- PyO3 gene set scoring ---
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Fused sparse matmul-reduce: computes ctrl_means and optionally pval_matrix
/// without materializing the dense intermediate.
///
/// Algorithm: row-scatter approach (same as scipy CSR @ CSR), fused with reduction.
/// For each row i of A:
///   1. Scatter-multiply: acc[k] += A[i,j] * B[j,k] for all nonzeros
///   2. Reduce: means[i][s] = mean(acc[s*n_ctrl_set .. (s+1)*n_ctrl_set])
///   3. Pval: count(acc[s*ctrl..] >= query[i][s])
///
/// The accumulator (n_ctrl_total * 8 bytes = ~19 KB) fits in L1 cache.
/// Total work: O(nnz_A * avg_nnz_per_B_row) — same as scipy but without
/// materializing the full (n_cells, n_ctrl_total) dense intermediate.
///
/// Per-thread memory: ~19 KB accumulator vs 192 MB dense chunk in Python.
/// Generic core for the fused matmul-reduce.
///
/// Generic over the two value slices so the f32 and f64 entry points share ONE
/// body. Products are formed in f64 (`Into<f64>` on each operand), and the
/// accumulator is f64, so an f32 call whose inputs came from f32 data is
/// bit-identical to the f64 call Python used to make by upcasting first — it
/// just moves half as many bytes.
#[allow(clippy::too_many_arguments)]
fn fused_matmul_reduce_core<TA, TB>(
    ad: &[TA],
    ai: &[i32],
    ap: &[i32],
    n_rows: usize,
    bd: &[TB],
    bi: &[i32],
    bp: &[i32],
    qs: &[f64],
    n_sets: usize,
    n_ctrl_set: usize,
    chunk_size: usize,
    n_threads: usize,
    compute_pvalues: bool,
) -> Result<(Vec<f64>, Option<Vec<f64>>), String>
where
    TA: Copy + Sync + Into<f64>,
    TB: Copy + Sync + Into<f64>,
{
    let n_ctrl_total = n_sets * n_ctrl_set;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| format!("rayon: {}", e))?;

    // chunk_size is the TILE: the unit of parallel work, not a memory block.
    let tile = chunk_size.max(1);
    let n_chunks = (n_rows + tile - 1) / tile;
    let chunk_ranges: Vec<(usize, usize)> = (0..n_chunks)
        .map(|c| {
            let start = c * tile;
            (start, (start + tile).min(n_rows))
        })
        .collect();

    let results: Vec<(Vec<f64>, Option<Vec<f64>>)> = pool.install(|| {
        chunk_ranges
            .par_iter()
            .map(|&(c_start, c_end)| {
                let chunk_rows = c_end - c_start;
                let mut means = vec![0.0f64; chunk_rows * n_sets];
                let mut pvals = if compute_pvalues {
                    Some(vec![0.0f64; chunk_rows * n_sets])
                } else {
                    None
                };
                let mut acc = vec![0.0f64; n_ctrl_total];

                for local_i in 0..chunk_rows {
                    let global_i = c_start + local_i;
                    acc.iter_mut().for_each(|x| *x = 0.0);

                    let a_row_start = ap[global_i] as usize;
                    let a_row_end = ap[global_i + 1] as usize;

                    for a_idx in a_row_start..a_row_end {
                        let j = ai[a_idx] as usize;
                        let a_val: f64 = ad[a_idx].into();
                        let b_row_start = bp[j] as usize;
                        let b_row_end = bp[j + 1] as usize;
                        for b_idx in b_row_start..b_row_end {
                            let k = bi[b_idx] as usize;
                            // SAFETY: k < n_ctrl_total guaranteed by B's construction
                            unsafe {
                                *acc.get_unchecked_mut(k) += a_val * bd[b_idx].into();
                            }
                        }
                    }

                    for s in 0..n_sets {
                        let base = s * n_ctrl_set;
                        let mut sum = 0.0f64;
                        for c in 0..n_ctrl_set {
                            sum += acc[base + c];
                        }
                        means[local_i * n_sets + s] = sum / n_ctrl_set as f64;
                    }

                    if let Some(ref mut pv) = pvals {
                        for s in 0..n_sets {
                            let q = qs[global_i * n_sets + s];
                            let base = s * n_ctrl_set;
                            let mut n_greater = 0u32;
                            for c in 0..n_ctrl_set {
                                if acc[base + c] >= q {
                                    n_greater += 1;
                                }
                            }
                            pv[local_i * n_sets + s] =
                                (n_greater as f64 + 1.0) / (n_ctrl_set as f64 + 1.0);
                        }
                    }
                }

                (means, pvals)
            })
            .collect()
    });

    let mut means_flat = vec![0.0f64; n_rows * n_sets];
    let mut pval_flat = if compute_pvalues {
        Some(vec![0.0f64; n_rows * n_sets])
    } else {
        None
    };
    for (chunk_idx, (means, pvals)) in results.into_iter().enumerate() {
        let c_start = chunk_idx * tile;
        let c_end = (c_start + tile).min(n_rows);
        let n = (c_end - c_start) * n_sets;
        means_flat[c_start * n_sets..c_start * n_sets + n].copy_from_slice(&means[..n]);
        if let (Some(ref mut pf), Some(pv)) = (&mut pval_flat, pvals) {
            pf[c_start * n_sets..c_start * n_sets + n].copy_from_slice(&pv[..n]);
        }
    }

    Ok((means_flat, pval_flat))
}


macro_rules! fused_matmul_reduce_entry {
    ($name:ident, $ta:ty, $tb:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
            a_data, a_indices, a_indptr, n_rows, _n_cols,
            b_data, b_indices, b_indptr, b_n_cols,
            query_scores,
            n_sets, n_ctrl_set, chunk_size, n_threads, compute_pvalues
        ))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            a_data: PyReadonlyArray1<'py, $ta>,
            a_indices: PyReadonlyArray1<'py, i32>,
            a_indptr: PyReadonlyArray1<'py, i32>,
            n_rows: usize,
            _n_cols: usize,
            b_data: PyReadonlyArray1<'py, $tb>,
            b_indices: PyReadonlyArray1<'py, i32>,
            b_indptr: PyReadonlyArray1<'py, i32>,
            b_n_cols: usize,
            query_scores: PyReadonlyArray1<'py, f64>,
            n_sets: usize,
            n_ctrl_set: usize,
            chunk_size: usize,
            n_threads: usize,
            compute_pvalues: bool,
        ) -> PyResult<(Bound<'py, PyArray1<f64>>, Option<Bound<'py, PyArray1<f64>>>)> {
            let ad = a_data.as_slice()?;
            let ai = a_indices.as_slice()?;
            let ap = a_indptr.as_slice()?;
            let bd = b_data.as_slice()?;
            let bi = b_indices.as_slice()?;
            let bp = b_indptr.as_slice()?;
            let qs = query_scores.as_slice()?;
            assert_eq!(b_n_cols, n_sets * n_ctrl_set, "b_n_cols != n_sets * n_ctrl_set");

            let (means_flat, pval_flat) = py
                .detach(|| {
                    fused_matmul_reduce_core(
                        ad, ai, ap, n_rows, bd, bi, bp, qs, n_sets, n_ctrl_set,
                        chunk_size, n_threads, compute_pvalues,
                    )
                })
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            Ok((
                Array1::from_vec(means_flat).into_pyarray(py),
                pval_flat.map(|v| Array1::from_vec(v).into_pyarray(py)),
            ))
        }
    };
}

// Fused sparse matmul-reduce: ctrl_means (and optionally pvals) without
// materializing the dense intermediate.
//
// `chunk_size` is the TILE -- the rows in one unit of parallel work. It is NOT
// a memory budget: every tile's result is collected before assembly, so peak
// memory is the same whether there are 2 tiles or 12,000. Passing the whole row
// count leaves ONE tile, and therefore one working thread.
fused_matmul_reduce_entry!(fused_matmul_reduce, f64, f64);

// f32 twin. Products and the accumulator are still f64, so results are
// identical when the caller's data was f32 to begin with -- the normal case,
// since cytome layers are stored float32.
fused_matmul_reduce_entry!(fused_matmul_reduce_f32, f32, f32);


/// Simple LCG PRNG for deterministic, fast random number generation.
/// Same sequence regardless of thread count.
#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

/// Fused score_complete: control gene sampling + matmul + reduce in one call.
///
/// Eliminates all Python intermediate allocations by:
/// 1. Sampling control genes directly from KNN indices (no sparse B matrix)
/// 2. For each cell row, scatter-accumulate into a per-set accumulator
/// 3. Reduce to means/pvalues inline
///
/// Per-thread memory: n_ctrl_set * 8 bytes per gene set (~800 bytes for 100 ctrl)
/// Total per-thread: n_sets * n_ctrl_set * 8 ≈ 19 KB (same as fused_matmul_reduce)
#[pyfunction]
#[pyo3(signature = (
    a_data, a_indices, a_indptr, n_rows, n_cols,
    knn_idx, knn_k,
    gene_sets_flat, gene_sets_offsets,
    weights_flat, weights_offsets,
    n_ctrl_set, random_seed, chunk_size, n_threads, compute_pvalues
))]
fn score_complete<'py>(
    py: Python<'py>,
    // CSR expression matrix A (n_cells × n_genes)
    a_data: PyReadonlyArray1<'py, f64>,
    a_indices: PyReadonlyArray1<'py, i32>,
    a_indptr: PyReadonlyArray1<'py, i32>,
    n_rows: usize,
    n_cols: usize,
    // KNN indices: flat array (n_genes × knn_k), row-major
    knn_idx: PyReadonlyArray1<'py, i64>,
    knn_k: usize,
    // Gene sets: flattened indices + offsets (CSR-style)
    // gene_sets_flat[offsets[s]..offsets[s+1]] = gene indices for set s
    gene_sets_flat: PyReadonlyArray1<'py, i32>,
    gene_sets_offsets: PyReadonlyArray1<'py, i32>,
    // Weights: flattened + offsets (same layout as gene_sets)
    weights_flat: PyReadonlyArray1<'py, f64>,
    weights_offsets: PyReadonlyArray1<'py, i32>,
    // Parameters
    n_ctrl_set: usize,
    random_seed: u64,
    chunk_size: usize,
    n_threads: usize,
    compute_pvalues: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,  // score_matrix flat (n_rows * n_sets)
    Bound<'py, PyArray1<f64>>,  // query_scores flat (n_rows * n_sets)
    Bound<'py, PyArray1<f64>>,  // scaling_factors (n_sets)
    Option<Bound<'py, PyArray1<f64>>>,  // pval_matrix flat (n_rows * n_sets)
)> {
    // Extract slices from numpy arrays (requires GIL)
    let ad = a_data.as_slice()?;
    let ai = a_indices.as_slice()?;
    let ap = a_indptr.as_slice()?;
    let knn = knn_idx.as_slice()?;
    let gs_flat = gene_sets_flat.as_slice()?;
    let gs_off = gene_sets_offsets.as_slice()?;
    let w_flat = weights_flat.as_slice()?;
    let _w_off = weights_offsets.as_slice()?;

    let n_sets = gs_off.len() - 1;

    // Reborrow slices with independent lifetimes for use inside py.detach.
    // SAFETY: The numpy array buffers remain valid for the function's duration.
    // We only release the GIL for computation, the Python objects are not freed.
    let (ad, ai, ap, knn, gs_flat, gs_off, w_flat) = unsafe {
        (
            std::slice::from_raw_parts(ad.as_ptr(), ad.len()),
            std::slice::from_raw_parts(ai.as_ptr(), ai.len()),
            std::slice::from_raw_parts(ap.as_ptr(), ap.len()),
            std::slice::from_raw_parts(knn.as_ptr(), knn.len()),
            std::slice::from_raw_parts(gs_flat.as_ptr(), gs_flat.len()),
            std::slice::from_raw_parts(gs_off.as_ptr(), gs_off.len()),
            std::slice::from_raw_parts(w_flat.as_ptr(), w_flat.len()),
        )
    };

    // Release GIL for all heavy computation (control gene sampling + matmul + reduce)
    let (score_out, query_scores_vec, scaling_factors, pval_out) = py.detach(|| {

    // --- Pre-compute control gene indices for all sets ---
    let mut ctrl_genes: Vec<Vec<(usize, usize, f64)>> = Vec::with_capacity(n_sets);
    let mut scaling_factors = vec![0.0f64; n_sets];

    for s in 0..n_sets {
        let gs_start = gs_off[s] as usize;
        let gs_end = gs_off[s + 1] as usize;
        let n_gs = gs_end - gs_start;

        // Compute scaling factor (median weight * n_genes_in_set)
        let mut sorted_w: Vec<f64> = (gs_start..gs_end).map(|g| w_flat[g]).collect();
        sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_w = if sorted_w.len() % 2 == 0 {
            (sorted_w[sorted_w.len() / 2 - 1] + sorted_w[sorted_w.len() / 2]) / 2.0
        } else {
            sorted_w[sorted_w.len() / 2]
        };
        scaling_factors[s] = median_w * n_gs as f64;

        // Sample control genes using deterministic PRNG
        let mut rng_state = random_seed;
        let mut triples = Vec::with_capacity(n_ctrl_set * n_gs);

        for c in 0..n_ctrl_set {
            for g_local in 0..n_gs {
                let gene_global = gs_flat[gs_start + g_local] as usize;
                let knn_base = gene_global * knn_k;
                let rand_val = lcg_next(&mut rng_state);
                let neighbor_idx = (rand_val >> 33) as usize % knn_k;
                let ctrl_gene = knn[knn_base + neighbor_idx] as usize;
                let weight = w_flat[gs_start + g_local];
                triples.push((c, ctrl_gene, weight));
            }
        }
        ctrl_genes.push(triples);
    }

    // --- Build virtual B matrix as CSR ---
    let mut b_entries: Vec<Vec<(i32, f64)>> = vec![Vec::new(); n_cols];
    for s in 0..n_sets {
        for &(c, ctrl_gene, weight) in &ctrl_genes[s] {
            let col = (s * n_ctrl_set + c) as i32;
            b_entries[ctrl_gene].push((col, weight));
        }
    }
    for row in b_entries.iter_mut() {
        row.sort_by_key(|&(col, _)| col);
    }

    let mut b_indptr_vec: Vec<i32> = Vec::with_capacity(n_cols + 1);
    let mut b_indices_vec: Vec<i32> = Vec::new();
    let mut b_data_vec: Vec<f64> = Vec::new();
    b_indptr_vec.push(0);
    for row in &b_entries {
        for &(col, val) in row {
            b_indices_vec.push(col);
            b_data_vec.push(val);
        }
        b_indptr_vec.push(b_indices_vec.len() as i32);
    }
    drop(b_entries);

    let bp = &b_indptr_vec;
    let bi = &b_indices_vec;

    // The control side is the dominant memory traffic in the scatter loop:
    // every nonzero of A walks a row of B. Halving its element width is the
    // cheapest win available here. Only taken when every value round-trips
    // exactly through f32, so the products -- formed in f64 either way -- are
    // unchanged. Gene-set weights default to 1.0, so this is the usual case.
    let b_exact_in_f32 = b_data_vec.iter().all(|&v| (v as f32) as f64 == v);
    let b_data_f32: Vec<f32> = if b_exact_in_f32 {
        b_data_vec.iter().map(|&v| v as f32).collect()
    } else {
        Vec::new()
    };

    // --- Compute query scores in single pass ---
    let mut gene_membership: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_cols];
    for s in 0..n_sets {
        let gs_start = gs_off[s] as usize;
        let gs_end = gs_off[s + 1] as usize;
        for g_local in 0..(gs_end - gs_start) {
            let gene_global = gs_flat[gs_start + g_local] as usize;
            let weight = w_flat[gs_start + g_local];
            gene_membership[gene_global].push((s, weight));
        }
    }

    let mut query_scores_vec = vec![0.0f64; n_rows * n_sets];
    for cell in 0..n_rows {
        let row_start = ap[cell] as usize;
        let row_end = ap[cell + 1] as usize;
        for idx in row_start..row_end {
            let gene = ai[idx] as usize;
            let val = ad[idx];
            for &(s, w) in &gene_membership[gene] {
                query_scores_vec[cell * n_sets + s] += val * w;
            }
        }
    }

    // --- Parallel matmul-reduce ---
    // The same scatter/reduce/pvalue loop the streaming kernel runs, so it uses
    // the same core rather than a second copy that could drift. `chunk_size` is
    // the TILE here, not a memory block -- see fused_matmul_reduce_core.
    let qs = &query_scores_vec;
    let (ctrl_means, pval_out) = if b_exact_in_f32 {
        fused_matmul_reduce_core(
            ad, ai, ap, n_rows, &b_data_f32, bi, bp, qs, n_sets, n_ctrl_set,
            chunk_size, n_threads, compute_pvalues,
        )
    } else {
        fused_matmul_reduce_core(
            ad, ai, ap, n_rows, &b_data_vec, bi, bp, qs, n_sets, n_ctrl_set,
            chunk_size, n_threads, compute_pvalues,
        )
    }
    // Inside py.detach, so there is no `?` here; the only error the core can
    // return is a rayon pool failure, which this code already unwrapped.
    .expect("rayon thread pool");

    // Final scores: (query / scaling) - (ctrl_means / scaling)
    let mut score_out = vec![0.0f64; n_rows * n_sets];
    for cell in 0..n_rows {
        for s in 0..n_sets {
            let idx = cell * n_sets + s;
            let sf = scaling_factors[s];
            score_out[idx] = (qs[idx] / sf) - (ctrl_means[idx] / sf);
        }
    }

    (score_out, query_scores_vec, scaling_factors, pval_out)

    }); // end py.detach

    // Convert to numpy arrays (requires GIL)
    let scores_arr = Array1::from_vec(score_out).into_pyarray(py);
    let query_arr = Array1::from_vec(query_scores_vec).into_pyarray(py);
    let sf_arr = Array1::from_vec(scaling_factors).into_pyarray(py);
    let pval_arr = pval_out.map(|v| Array1::from_vec(v).into_pyarray(py));

    Ok((scores_arr, query_arr, sf_arr, pval_arr))
}


// --- PyO3 wrappers for PICCO peak calling + quantification ---

/// Run a Rust closure with GIL released, catching panics as PyRuntimeError.
/// Converts serde_json::Value result to a native Python dict via pythonize.
fn run_catching_panic<F>(py: Python<'_>, context: &str, f: F) -> PyResult<Py<PyAny>>
where
    F: FnOnce() -> serde_json::Value + Send,
{
    let result = py.detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)));
    match result {
        Ok(value) => pythonize::pythonize(py, &value)
            .map(|bound| bound.unbind())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("{context}: pythonize failed: {e}")
            )),
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("{context} panicked: {msg}")
            ))
        }
    }
}


/// Call peaks using PICCO (Rust backend, in-process via PyO3).
///
/// Equivalent to `piaso-atac call-peaks` CLI but runs in-process,
/// avoiding subprocess overhead and enabling direct Python dict return.


/// Quantify peak activity per cell (Rust backend, in-process via PyO3).
///
/// Equivalent to `piaso-atac quantify` CLI but runs in-process.


/// Python module
#[pymodule]
fn _piaso(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_matmul_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(fused_matmul_reduce_f32, m)?)?;
    m.add_function(wrap_pyfunction!(score_complete, m)?)?;
    m.add_function(wrap_pyfunction!(motif_scan::scan_motifs_fwd, m)?)?;
    Ok(())
}
