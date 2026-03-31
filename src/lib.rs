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

    let means_array = Array1::from_vec(means_flat).into_pyarray_bound(py);
    let pval_array = pval_flat.map(|v| Array1::from_vec(v).into_pyarray_bound(py));

    Ok((means_array, pval_array))
}

/// Python module
#[pymodule]
fn _piaso_score(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_matmul_reduce, m)?)?;
    Ok(())
}
