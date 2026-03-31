use numpy::ndarray::{Array1, Array2};
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
#[pyfunction]
#[pyo3(signature = (
    a_data, a_indices, a_indptr, n_rows, _n_cols,
    b_data, b_indices, b_indptr, b_n_cols,
    query_scores,
    n_sets, n_ctrl_set, chunk_size, n_threads, compute_pvalues
))]
fn fused_matmul_reduce<'py>(
    py: Python<'py>,
    // CSR matrix A (cellxgene)
    a_data: PyReadonlyArray1<'py, f64>,
    a_indices: PyReadonlyArray1<'py, i32>,
    a_indptr: PyReadonlyArray1<'py, i32>,
    n_rows: usize,
    _n_cols: usize,
    // CSR matrix B (big_ctrl — must be CSR, convert in Python if needed)
    b_data: PyReadonlyArray1<'py, f64>,
    b_indices: PyReadonlyArray1<'py, i32>,
    b_indptr: PyReadonlyArray1<'py, i32>,
    b_n_cols: usize,
    // Query scores for pvalue computation, flattened (n_rows * n_sets)
    query_scores: PyReadonlyArray1<'py, f64>,
    // Parameters
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

    let n_ctrl_total = n_sets * n_ctrl_set;
    assert_eq!(b_n_cols, n_ctrl_total, "b_n_cols != n_sets * n_ctrl_set");

    // Configure thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("rayon: {}", e)))?;

    // Determine chunks
    let n_chunks = (n_rows + chunk_size - 1) / chunk_size;
    let chunk_ranges: Vec<(usize, usize)> = (0..n_chunks)
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n_rows);
            (start, end)
        })
        .collect();

    // Parallel computation using row-scatter approach
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

                // Thread-local accumulator: fits in L1 cache (~19 KB for 2400 cols)
                let mut acc = vec![0.0f64; n_ctrl_total];

                for local_i in 0..chunk_rows {
                    let global_i = c_start + local_i;

                    // Phase 1: Scatter-multiply (CSR row of A × CSR rows of B)
                    // acc[k] += A[i,j] * B[j,k] for all nonzeros in row i of A
                    acc.iter_mut().for_each(|x| *x = 0.0);

                    let a_row_start = ap[global_i] as usize;
                    let a_row_end = ap[global_i + 1] as usize;

                    for a_idx in a_row_start..a_row_end {
                        let j = ai[a_idx] as usize; // column in A = row in B
                        let a_val = ad[a_idx];

                        let b_row_start = bp[j] as usize;
                        let b_row_end = bp[j + 1] as usize;

                        for b_idx in b_row_start..b_row_end {
                            let k = bi[b_idx] as usize;
                            // SAFETY: k < n_ctrl_total guaranteed by B's construction
                            unsafe {
                                *acc.get_unchecked_mut(k) += a_val * bd[b_idx];
                            }
                        }
                    }

                    // Phase 2: Reduce accumulator to means per gene set
                    for s in 0..n_sets {
                        let base = s * n_ctrl_set;
                        let mut sum = 0.0f64;
                        for c in 0..n_ctrl_set {
                            sum += acc[base + c];
                        }
                        means[local_i * n_sets + s] = sum / n_ctrl_set as f64;
                    }

                    // Phase 3: Compute p-values (optional)
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

    // Assemble results into contiguous arrays
    let mut ctrl_means = Array2::<f64>::zeros((n_rows, n_sets));
    let mut pval_matrix = if compute_pvalues {
        Some(Array2::<f64>::zeros((n_rows, n_sets)))
    } else {
        None
    };

    for (chunk_idx, (means, pvals)) in results.into_iter().enumerate() {
        let c_start = chunk_idx * chunk_size;
        let c_end = (c_start + chunk_size).min(n_rows);
        let chunk_rows = c_end - c_start;

        for i in 0..chunk_rows {
            for s in 0..n_sets {
                ctrl_means[[c_start + i, s]] = means[i * n_sets + s];
            }
        }

        if let (Some(ref mut pm), Some(pv)) = (&mut pval_matrix, pvals) {
            for i in 0..chunk_rows {
                for s in 0..n_sets {
                    pm[[c_start + i, s]] = pv[i * n_sets + s];
                }
            }
        }
    }

    // Convert to numpy arrays
    let (means_flat, _) = ctrl_means.into_raw_vec_and_offset();
    let pval_flat = pval_matrix.map(|pm| {
        let (v, _) = pm.into_raw_vec_and_offset();
        v
    });

    let means_array = Array1::from_vec(means_flat).into_pyarray(py);
    let pval_array = pval_flat.map(|v| Array1::from_vec(v).into_pyarray(py));

    Ok((means_array, pval_array))
}


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
    let n_ctrl_total = n_sets * n_ctrl_set;

    // Reborrow slices with independent lifetimes for use inside allow_threads.
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
    let (score_out, query_scores_vec, scaling_factors, pval_out) = py.allow_threads(|| {

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
    let bd = &b_data_vec;

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
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let n_chunks = (n_rows + chunk_size - 1) / chunk_size;
    let chunk_ranges: Vec<(usize, usize)> = (0..n_chunks)
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n_rows);
            (start, end)
        })
        .collect();

    let qs = &query_scores_vec;

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
                        let a_val = ad[a_idx];
                        let b_row_start = bp[j] as usize;
                        let b_row_end = bp[j + 1] as usize;
                        for b_idx in b_row_start..b_row_end {
                            let k = bi[b_idx] as usize;
                            unsafe {
                                *acc.get_unchecked_mut(k) += a_val * bd[b_idx];
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

    // Assemble results
    let mut ctrl_means = vec![0.0f64; n_rows * n_sets];
    let mut pval_out = if compute_pvalues {
        Some(vec![0.0f64; n_rows * n_sets])
    } else {
        None
    };

    for (chunk_idx, (means, pvals)) in results.into_iter().enumerate() {
        let c_start = chunk_idx * chunk_size;
        let c_end = (c_start + chunk_size).min(n_rows);
        let chunk_rows = c_end - c_start;
        let dst_start = c_start * n_sets;
        ctrl_means[dst_start..dst_start + chunk_rows * n_sets]
            .copy_from_slice(&means[..chunk_rows * n_sets]);
        if let (Some(ref mut po), Some(pv)) = (&mut pval_out, pvals) {
            po[dst_start..dst_start + chunk_rows * n_sets]
                .copy_from_slice(&pv[..chunk_rows * n_sets]);
        }
    }

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

    }); // end py.allow_threads

    // Convert to numpy arrays (requires GIL)
    let scores_arr = Array1::from_vec(score_out).into_pyarray(py);
    let query_arr = Array1::from_vec(query_scores_vec).into_pyarray(py);
    let sf_arr = Array1::from_vec(scaling_factors).into_pyarray(py);
    let pval_arr = pval_out.map(|v| Array1::from_vec(v).into_pyarray(py));

    Ok((scores_arr, query_arr, sf_arr, pval_arr))
}


/// Python module
#[pymodule]
fn _piaso_score(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_matmul_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(score_complete, m)?)?;
    Ok(())
}
