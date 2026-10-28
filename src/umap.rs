//! UMAP's graph and layout (McInnes, Healy & Melville 2018): parallel, and
//! deterministic.
//!
//! **One graph, one start and one seed give one layout.** The number of
//! threads and their scheduling do not enter the result, nor does the order
//! of the nodes in memory, given their names (`identity`, by default their
//! positions). The rules:
//!
//! 1. *A schedule of node blocks.* Each node goes to one of `blocks` (128)
//!    blocks by a hash of its name. An epoch visits every pair of blocks once,
//!    in `blocks` stages: the diagonal pairs (each block with itself), then the
//!    rounds of a round-robin tournament (the circle method), in which each
//!    block meets exactly one other. The pairs of a stage share no node, so
//!    they run on separate threads, and the result is that of one sequential
//!    pass over the edges in a fixed order: stage by stage, pair by pair, and
//!    inside a pair edge by edge in the order of their ends' names. (The
//!    stratified SGD of matrix factorization, Gemulla et al. 2011, applied to
//!    a graph.) The number of blocks decides how the work is cut, not what is
//!    computed; another number changes the bits as another seed does.
//! 2. *Random numbers by hash.* A negative sample is the node whose name is
//!    hash(seed, epoch, the edge's two names, its direction, k) mod n, and its
//!    position is read from the positions the epoch began with: the only
//!    positions another thread could be writing.
//! 3. *No shared sums.* The pull a firing gives the edge's other end is added
//!    to that node's sum, which only the thread holding the node's block
//!    writes, in the order of rule 1, and each node adds its sum when the
//!    epoch ends. (Powers use the platform's `pow`, so the bits are those of
//!    one machine, as with the SVD.)
//!
//! **The schedule is umap-learn's (0.5).** An edge of weight w fires every
//! w_max / w epochs: the k-th time in the first epoch n with n >= k w_max / w,
//! so never in epoch 0. A firing pulls both ends together, then draws
//! `negative_sample_rate` (r) negative samples on average for the edge's own
//! node: at the k-th firing, in epoch n, floor(n r w / w_max) minus the same
//! at the previous firing, and one fewer at the first, as umap-learn's
//! counters give. The step falls linearly from `initial_alpha`, and every
//! gradient component is clipped at 4. The firings and the counts follow from
//! the epoch and the weight, so nothing is stored per edge but its ends and
//! its weight. The graph holds each edge in both directions, as umap-learn's
//! does; the layout keeps it once and fires its two directed copies one after
//! the other. Each copy moves its own node at once and sends the pull on the
//! other end to that node's sum. (umap-learn moves the other end at once.)
//!
//! **The graph** is umap-learn's fuzzy simplicial set: each node's distance to
//! its nearest neighbor (rho) and its bandwidth (sigma, by bisection), the
//! membership exp(-(d - rho) / sigma) of each neighbor, and the fuzzy union
//! A + A^T - A o A^T, in float32 as umap-learn computes it.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

/// Graphs with fewer nodes than this build their fuzzy graph and spectral
/// products on one thread: the pool costs more than it gives (as the Leiden's
/// `SINGLE`).
const SINGLE: usize = 1 << 15;
/// The layout's threshold is lower: a node's work is hundreds of epochs, so
/// threads pay from a few thousand nodes on.
const LAYOUT_SINGLE: usize = 1 << 12;
/// The layout's blocks (rule 1), and the most it takes.
pub const LAYOUT_BLOCKS: usize = 128;
const MAX_BLOCKS: usize = 1024;
/// umap-learn's constants of the bandwidth search.
const SMOOTH_K_TOLERANCE: f64 = 1e-5;
const MIN_K_DIST_SCALE: f64 = 1e-3;
const N_ITER_SIGMA: usize = 64;
const INTERRUPTED: &str = "interrupted";
/// Contexts of the hashes, so that a node's block and a negative sample never
/// share a stream.
const KIND_BLOCK: u64 = 0x626c_6f63_6b31;
const KIND_NEGATIVE: u64 = 0x6e65_6761_7469;
const GOLD: u64 = 0x9E3779B97F4A7C15;

#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A counter-based random number of (seed, kind, epoch, a, b, c).
#[inline]
fn hash6(seed: u64, kind: u64, epoch: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut h = mix(seed ^ 0x9E3779B97F4A7C15);
    h = mix(h ^ kind);
    h = mix(h.wrapping_add(epoch.wrapping_mul(0xD1B54A32D192ED03)));
    h = mix(h ^ a.wrapping_mul(0x9E3779B97F4A7C15));
    h = mix(h.wrapping_add(b.wrapping_mul(0xC2B2AE3D27D4EB4F)));
    mix(h ^ c.wrapping_mul(0x165667B19E3779F9))
}

// ---------------------------------------------------------------------------
// The fuzzy graph
// ---------------------------------------------------------------------------

/// umap-learn's `smooth_knn_dist` for one node: (sigma, rho). `d` is the row
/// of distances, the node itself first at distance 0.
fn smooth_knn_row(d: &[f32], mean_all: f64, local_connectivity: f64, bandwidth: f64) -> (f32, f32) {
    let k = d.len();
    let target = (k as f64).log2() * bandwidth;
    let nz: Vec<f32> = d.iter().copied().filter(|&x| x > 0.0).collect();
    let mut rho: f32 = 0.0;
    if nz.len() as f64 >= local_connectivity {
        let index = local_connectivity.floor() as usize;
        let interpolation = local_connectivity - index as f64;
        if index > 0 {
            rho = nz[index - 1];
            if interpolation > SMOOTH_K_TOLERANCE {
                rho += (interpolation * (nz[index] - nz[index - 1]) as f64) as f32;
            }
        } else {
            rho = (interpolation * nz[0] as f64) as f32;
        }
    } else if !nz.is_empty() {
        rho = nz.iter().copied().fold(f32::MIN, f32::max);
    }
    let (mut lo, mut hi, mut mid) = (0.0f64, f32::MAX as f64, 1.0f64);
    for _ in 0..N_ITER_SIGMA {
        let mut psum = 0.0f64;
        for &x in &d[1..] {
            let dd = x - rho;
            if dd > 0.0 {
                psum += (-(dd as f64 / mid)).exp();
            } else {
                psum += 1.0;
            }
        }
        if (psum - target).abs() < SMOOTH_K_TOLERANCE {
            break;
        }
        if psum > target {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            if hi >= f32::MAX as f64 {
                mid *= 2.0;
            } else {
                mid = (lo + hi) / 2.0;
            }
        }
    }
    let mut sigma = mid as f32;
    if rho > 0.0 {
        let mean_i = d.iter().map(|&x| x as f64).sum::<f64>() / k as f64;
        if (sigma as f64) < MIN_K_DIST_SCALE * mean_i {
            sigma = (MIN_K_DIST_SCALE * mean_i) as f32;
        }
    } else if (sigma as f64) < MIN_K_DIST_SCALE * mean_all {
        sigma = (MIN_K_DIST_SCALE * mean_all) as f32;
    }
    (sigma, rho)
}

/// A CSR matrix with sorted, unique columns per row.
pub struct Csr {
    pub n: usize,
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
}

/// The membership graph (before the union): row i holds its neighbors'
/// memberships, columns sorted and duplicates summed, zeros dropped.
fn membership(idx: &[i64], dist: &[f32], n: usize, k: usize, sigmas: &[f32], rhos: &[f32]) -> Csr {
    let rows: Vec<Vec<(i32, f32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<(i32, f32)> = Vec::with_capacity(k);
            for j in 0..k {
                let c = idx[i * k + j];
                if c < 0 {
                    continue;
                }
                let d = dist[i * k + j];
                let v: f32 = if c as usize == i {
                    0.0
                } else if d - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                    1.0
                } else {
                    libm::expf(-((d - rhos[i]) / sigmas[i]))
                };
                row.push((c as i32, v));
            }
            row.sort_by_key(|x| x.0);
            let mut out: Vec<(i32, f32)> = Vec::with_capacity(row.len());
            for (c, v) in row {
                match out.last_mut() {
                    Some(last) if last.0 == c => last.1 += v,
                    _ => out.push((c, v)),
                }
            }
            out.retain(|x| x.1 != 0.0);
            out
        })
        .collect();
    to_csr(n, rows)
}

fn to_csr(n: usize, rows: Vec<Vec<(i32, f32)>>) -> Csr {
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0i64);
    let mut total = 0i64;
    for r in &rows {
        total += r.len() as i64;
        indptr.push(total);
    }
    let mut indices = Vec::with_capacity(total as usize);
    let mut data = Vec::with_capacity(total as usize);
    for r in rows {
        for (c, v) in r {
            indices.push(c);
            data.push(v);
        }
    }
    Csr { n, indptr, indices, data }
}

/// The transpose of a CSR matrix (columns sorted, since rows are visited in
/// order).
fn transpose(a: &Csr) -> Csr {
    let n = a.n;
    let mut count = vec![0i64; n + 1];
    for &c in &a.indices {
        count[c as usize + 1] += 1;
    }
    for i in 0..n {
        count[i + 1] += count[i];
    }
    let indptr = count.clone();
    let mut next = count;
    let mut indices = vec![0i32; a.indices.len()];
    let mut data = vec![0f32; a.data.len()];
    for i in 0..n {
        for e in a.indptr[i] as usize..a.indptr[i + 1] as usize {
            let c = a.indices[e] as usize;
            let p = next[c] as usize;
            indices[p] = i as i32;
            data[p] = a.data[e];
            next[c] += 1;
        }
    }
    Csr { n, indptr, indices, data }
}

/// The fuzzy union A + A^T - A o A^T, in float32 in umap-learn's order: the
/// sum, then minus the product.
fn fuzzy_union(a: &Csr) -> Csr {
    let t = transpose(a);
    let rows: Vec<Vec<(i32, f32)>> = (0..a.n)
        .into_par_iter()
        .map(|i| {
            let (ra, rt) = (a.indptr[i] as usize..a.indptr[i + 1] as usize, t.indptr[i] as usize..t.indptr[i + 1] as usize);
            let (mut p, mut q) = (ra.start, rt.start);
            let mut out = Vec::with_capacity(ra.len() + rt.len());
            while p < ra.end || q < rt.end {
                let ca = if p < ra.end { a.indices[p] } else { i32::MAX };
                let ct = if q < rt.end { t.indices[q] } else { i32::MAX };
                let (c, x, y) = if ca == ct {
                    let r = (ca, a.data[p], t.data[q]);
                    p += 1;
                    q += 1;
                    r
                } else if ca < ct {
                    let r = (ca, a.data[p], 0.0f32);
                    p += 1;
                    r
                } else {
                    let r = (ct, 0.0f32, t.data[q]);
                    q += 1;
                    r
                };
                let v = (x + y) - x * y;
                if v != 0.0 {
                    out.push((c, v));
                }
            }
            out
        })
        .collect();
    to_csr(a.n, rows)
}

/// The fuzzy graph of a kNN result: (graph, sigmas, rhos).
pub fn fuzzy_graph(idx: &[i64], dist: &[f32], n: usize, k: usize, local_connectivity: f64, bandwidth: f64)
                   -> (Csr, Vec<f32>, Vec<f32>) {
    let mean_all = if dist.is_empty() { 0.0 } else { dist.iter().map(|&x| x as f64).sum::<f64>() / dist.len() as f64 };
    let sr: Vec<(f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| smooth_knn_row(&dist[i * k..(i + 1) * k], mean_all, local_connectivity, bandwidth))
        .collect();
    let sigmas: Vec<f32> = sr.iter().map(|x| x.0).collect();
    let rhos: Vec<f32> = sr.iter().map(|x| x.1).collect();
    let m = membership(idx, dist, n, k, &sigmas, &rhos);
    (fuzzy_union(&m), sigmas, rhos)
}

// ---------------------------------------------------------------------------
// Products for the spectral start
// ---------------------------------------------------------------------------

/// One step of a Chebyshev filter after the product: with ``y`` the product,
/// ``((h * y - c * x) * k1) - k2 * z``, each operation rounded in the order
/// numpy would round it.
pub struct FilterStep<'a> {
    pub z: &'a [f64],
    pub h: f64,
    pub c: f64,
    pub k1: f64,
    pub k2: f64,
}

/// y = s o (G (s o x)) + shift x for every column of x (n x c, row-major),
/// written into `y`: the normalized graph D^-1/2 G D^-1/2 (shifted), one row
/// per task, each row's sum in the order of its columns, so the result does
/// not depend on the number of threads. With `filter`, one step of the
/// Chebyshev filter follows the product.
#[allow(clippy::too_many_arguments)]
fn graph_matmat_into(indptr: &[i64], indices: &[i32], data: &[f64], s: &[f64], x: &[f64], c: usize, shift: f64,
                     filter: Option<&FilterStep>, y: &mut [f64]) {
    y.par_chunks_mut(c).enumerate().for_each(|(i, yi)| {
        yi.iter_mut().for_each(|v| *v = 0.0);
        for e in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[e] as usize;
            let w = data[e] * s[j];
            let xj = &x[j * c..(j + 1) * c];
            for t in 0..c {
                yi[t] += w * xj[t];
            }
        }
        let xi = &x[i * c..(i + 1) * c];
        for t in 0..c {
            yi[t] = s[i] * yi[t] + shift * xi[t];
        }
        if let Some(f) = filter {
            let zi = &f.z[i * c..(i + 1) * c];
            for t in 0..c {
                let a = f.h * yi[t];
                let b = a - f.c * xi[t];
                let d = b * f.k1;
                yi[t] = d - f.k2 * zi[t];
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn graph_matmat(indptr: &[i64], indices: &[i32], data: &[f64], s: &[f64], x: &[f64], c: usize, shift: f64,
                filter: Option<&FilterStep>) -> Vec<f64> {
    let mut y = vec![0f64; (indptr.len() - 1) * c];
    graph_matmat_into(indptr, indices, data, s, x, c, shift, filter, &mut y);
    y
}

/// The spectral start's Chebyshev filter of `degree`, every step in turn on
/// three blocks: from the block `x` and `ax`, its product with
/// A = (I + D^-1/2 G D^-1/2) / 2, y1 = (ax - c x) (sigma1 / e), then for each
/// further degree s' = 1 / (2 / sigma1 - s) and
/// y_next = (A y - c y) (2 s' / e) - (s s') y_prev (Zhou and Saad 2007's scaled
/// recurrence), each operation rounded as numpy would round it.
#[allow(clippy::too_many_arguments)]
fn chebyshev_filter(indptr: &[i64], indices: &[i32], data: &[f64], s: &[f64], x: &[f64], ax: &[f64], cols: usize,
                    c: f64, e: f64, sigma1: f64, degree: usize, cancel: &AtomicBool) -> Result<Vec<f64>, String> {
    let mut y0: Vec<f64> = x.to_vec();
    let t1 = sigma1 / e;
    let mut y1: Vec<f64> = ax.iter().zip(x.iter()).map(|(&p, &q)| (p - c * q) * t1).collect();
    let mut y2 = vec![0f64; y1.len()];
    let mut sigma = sigma1;
    for _ in 2..=degree {
        if cancel.load(Relaxed) {
            return Err(INTERRUPTED.into());
        }
        let s_next = 1.0 / (2.0 / sigma1 - sigma);
        let step = FilterStep { z: &y0, h: 0.5, c, k1: 2.0 * s_next / e, k2: sigma * s_next };
        graph_matmat_into(indptr, indices, data, s, &y1, cols, 1.0, Some(&step), &mut y2);
        // y0 <- y1, y1 <- the new block; the oldest block is the next scratch
        std::mem::swap(&mut y0, &mut y1);
        std::mem::swap(&mut y1, &mut y2);
        sigma = s_next;
    }
    Ok(y1)
}

// ---------------------------------------------------------------------------
// The layout
// ---------------------------------------------------------------------------

struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline]
    fn get(&self) -> *mut T {
        self.0
    }
}

#[inline]
fn clip(x: f64) -> f64 {
    if x > 4.0 {
        4.0
    } else if x < -4.0 {
        -4.0
    } else {
        x
    }
}

pub struct LayoutParams {
    pub n_epochs: usize,
    pub a: f64,
    pub b: f64,
    pub gamma: f64,
    pub initial_alpha: f64,
    pub negative_sample_rate: f64,
    pub blocks: usize,
    pub seed: u64,
}

/// Refuses what the layout cannot use: a malformed or asymmetric graph, an
/// edge listed twice or from a node to itself, or an identity that is not a
/// permutation.
fn check_graph(indptr: &[i64], indices: &[i32], w: &[f32], n: usize, identity: &[u32]) -> Result<(), String> {
    if indptr.len() != n + 1 || indptr[0] != 0 || indptr[n] as usize != indices.len() || w.len() != indices.len() {
        return Err("the graph's indptr, indices and weights do not agree with the number of nodes".into());
    }
    if indptr.windows(2).any(|p| p[1] < p[0]) {
        return Err("the graph's indptr is not non-decreasing".into());
    }
    if indices.iter().any(|&c| c < 0 || c as usize >= n) {
        return Err("the graph holds a column outside 0..n".into());
    }
    if w.iter().any(|&x| !(x >= 0.0) || !x.is_finite()) {
        return Err("the graph's weights must be finite and not negative".into());
    }
    if identity.len() != n {
        return Err("identity must hold one name per node".into());
    }
    let mut seen = vec![false; n];
    for &id in identity {
        if id as usize >= n || seen[id as usize] {
            return Err("identity must be a permutation of 0..n".into());
        }
        seen[id as usize] = true;
    }
    // Rows in order (columns strictly increasing, so no edge twice) are
    // searched as they are; otherwise each row's (column, weight) pairs are
    // sorted into a copy first. Either way an edge's reverse is found by binary
    // search: the linear search before cost the sum of the squared degrees,
    // which hub rows make large.
    let range = |i: usize| indptr[i] as usize..indptr[i + 1] as usize;
    let in_order = (0..n).into_par_iter().all(|i| indices[range(i)].windows(2).all(|p| p[0] < p[1]));
    let sorted_copy: Option<(Vec<i32>, Vec<f32>)> = if in_order {
        None
    } else {
        let rows: Vec<Vec<(i32, f32)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut r: Vec<(i32, f32)> = range(i).map(|e| (indices[e], w[e])).collect();
                r.sort_by_key(|x| x.0);
                r
            })
            .collect();
        let mut ix = Vec::with_capacity(indices.len());
        let mut wt = Vec::with_capacity(w.len());
        for r in rows {
            for (c, x) in r {
                ix.push(c);
                wt.push(x);
            }
        }
        Some((ix, wt))
    };
    let (ix, wt): (&[i32], &[f32]) = match &sorted_copy {
        Some((a, b)) => (a, b),
        None => (indices, w),
    };
    // each neighbor once per row, and no node its own neighbor
    let flaw = (0..n).into_par_iter().map(|i| {
        let row = &ix[range(i)];
        if row.windows(2).any(|p| p[0] == p[1]) {
            Some(format!("the graph holds an edge twice in row {}: sum the duplicates first", i))
        } else if row.binary_search(&(i as i32)).is_ok() {
            Some(format!("the graph holds an edge from node {} to itself, which UMAP's fuzzy graph never has", i))
        } else {
            None
        }
    }).find_first(|x| x.is_some());
    if let Some(Some(msg)) = flaw {
        return Err(msg);
    }
    // symmetric, values included: umap-learn's fuzzy union is
    let bad = (0..n).into_par_iter().find_first(|&i| {
        for e in range(i) {
            let j = ix[e] as usize;
            let rj = range(j);
            match ix[rj.clone()].binary_search(&(i as i32)) {
                Ok(p) if wt[rj.start + p] == wt[e] => {}
                _ => return true,
            }
        }
        false
    });
    if let Some(i) = bad {
        return Err(format!("the graph is not symmetric at node {}: pass UMAP's fuzzy graph (the union of both directions)", i));
    }
    Ok(())
}

/// An undirected edge, stored once: its ends (`u` the one with the smaller
/// name) and its rate w / w_max, the share of the epochs in which it fires.
#[derive(Clone, Copy)]
struct Edge {
    u: u32,
    v: u32,
    rate: f32,
}

/// A node's block: a hash of its name, the same whatever the seed.
#[inline]
fn block_of(id: u32, blocks: usize) -> usize {
    (mix(id as u64 ^ KIND_BLOCK) % blocks as u64) as usize
}

/// The stages of rule 1 for an even number of blocks: the diagonal pairs,
/// then the circle method's rounds (block 0 fixed, the others turning one
/// place a round); each pair named lo * blocks + hi.
fn stages(blocks: usize) -> Vec<Vec<usize>> {
    let p = blocks;
    let mut out = vec![(0..p).map(|x| x * p + x).collect::<Vec<usize>>()];
    let mut others: Vec<usize> = (1..p).collect();
    for _ in 0..p - 1 {
        let ring: Vec<usize> = std::iter::once(0).chain(others.iter().copied()).collect();
        out.push((0..p / 2).map(|i| {
            let (x, y) = (ring[i], ring[p - 1 - i]);
            x.min(y) * p + x.max(y)
        }).collect());
        others.rotate_right(1);
    }
    out
}

/// Each undirected edge once, grouped by its pair of blocks and, inside a
/// pair, in the order of its ends' names; with each pair's range in the list.
/// An edge whose rate rounds to zero never fires and is left out.
fn edges_by_pair(indptr: &[i64], indices: &[i32], w: &[f32], wmax: f32, identity: &[u32], blocks: usize)
                 -> (Vec<Edge>, Vec<usize>) {
    let n = indptr.len() - 1;
    let block: Vec<u32> = identity.par_iter().map(|&id| block_of(id, blocks) as u32).collect();
    let pair = |i: usize, j: usize| {
        let (x, y) = (block[i] as usize, block[j] as usize);
        x.min(y) * blocks + x.max(y)
    };
    let n_pairs = blocks * blocks;
    let mut ptr = vec![0usize; n_pairs + 1];
    for i in 0..n {
        for e in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[e] as usize;
            if identity[i] < identity[j] && w[e] / wmax > 0.0 {
                ptr[pair(i, j) + 1] += 1;
            }
        }
    }
    for k in 0..n_pairs {
        ptr[k + 1] += ptr[k];
    }
    let mut fill = ptr.clone();
    let mut edges = vec![Edge { u: 0, v: 0, rate: 0.0 }; ptr[n_pairs]];
    for i in 0..n {
        for e in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[e] as usize;
            let rate = w[e] / wmax;
            if identity[i] < identity[j] && rate > 0.0 {
                let k = pair(i, j);
                edges[fill[k]] = Edge { u: i as u32, v: j as u32, rate };
                fill[k] += 1;
            }
        }
    }
    let mut parts: Vec<&mut [Edge]> = Vec::with_capacity(n_pairs);
    let mut rest: &mut [Edge] = &mut edges;
    for k in 0..n_pairs {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(ptr[k + 1] - ptr[k]);
        parts.push(head);
        rest = tail;
    }
    // the names are unique, so the order is total
    parts.par_iter_mut().for_each(|s| s.sort_unstable_by_key(|x| (identity[x.u as usize], identity[x.v as usize])));
    (edges, ptr)
}

/// The negative samples of a firing in epoch `nf` of an edge of rate `f` (its
/// `k`-th firing, k = floor(nf f)), as umap-learn's counter gives them: by the
/// k-th firing, in epoch n, the counter has drawn floor(n f r) - 1 in all, so
/// the first firing draws floor(n f r) - 1 and a later one
/// floor(n f r) - floor(p f r), p the epoch of the previous firing.
#[inline]
fn negatives(k: f64, nf: f64, f: f64, r: f64) -> u64 {
    let fr = f * r;
    let here = (nf * fr).floor();
    let before = if k <= 1.0 {
        1.0
    } else {
        // the previous firing's epoch: the first x with floor(x f) >= k - 1
        let target = k - 1.0;
        let mut x = (target / f).ceil();
        while x > 1.0 && ((x - 1.0) * f).floor() >= target {
            x -= 1.0;
        }
        while (x * f).floor() < target {
            x += 1.0;
        }
        (x * fr).floor()
    };
    let m = here - before;
    if m > 0.0 { m as u64 } else { 0 }
}

/// One directed copy (i -> j) of a firing: i moves toward j at once and the
/// pull on j goes to j's sum; then i is pushed away from its `m` negative
/// samples, whose positions are the epoch's start (`snap`).
///
/// # Safety
/// `pos` and `acc` hold n x dim numbers, i and j are distinct nodes below n,
/// and no other thread reads or writes the rows of i and j meanwhile.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn fire(i: usize, j: usize, idi: u64, h0: u64, m: u64, pos: *mut f32, acc: *mut f64, snap: &[f32], inv: &[u32],
               dim: usize, nv: u64, a: f64, b: f64, gamma: f64, alpha: f64) {
    let pi = pos.add(i * dim);
    let pj = pos.add(j * dim) as *const f32;
    // the squared distance: each square in float32, the sum in float64 (umap-learn's rdist)
    let mut d2 = 0f64;
    for t in 0..dim {
        let d = *pi.add(t) - *pj.add(t);
        d2 += (d * d) as f64;
    }
    let coeff = if d2 > 0.0 {
        // d2^(b-1) = d2^b / d2: one power per pull
        let q = d2.powf(b);
        -2.0 * a * b * (q / d2) / (a * q + 1.0)
    } else {
        0.0
    };
    let aj = acc.add(j * dim);
    for t in 0..dim {
        let g = clip(coeff * (*pi.add(t) - *pj.add(t)) as f64);
        *pi.add(t) = (*pi.add(t) as f64 + g * alpha) as f32;
        *aj.add(t) -= g * alpha;
    }
    for s in 0..m {
        let kid = mix(h0.wrapping_add((s + 1).wrapping_mul(GOLD))) % nv;
        if kid == idi {
            continue;
        }
        let k = inv[kid as usize] as usize;
        let o = &snap[k * dim..(k + 1) * dim];
        let mut d2 = 0f64;
        for t in 0..dim {
            let d = *pi.add(t) - o[t];
            d2 += (d * d) as f64;
        }
        if d2 > 0.0 {
            let coeff = 2.0 * gamma * b / ((0.001 + d2) * (a * d2.powf(b) + 1.0));
            for t in 0..dim {
                let g = clip(coeff * (*pi.add(t) - o[t]) as f64);
                *pi.add(t) = (*pi.add(t) as f64 + g * alpha) as f32;
            }
        }
    }
}

/// The layout, in place on `pos` (n x dim, row-major).
pub fn layout(indptr: &[i64], indices: &[i32], w: &[f32], pos: &mut [f32], dim: usize, identity: &[u32], p: &LayoutParams,
              cancel: &AtomicBool) -> Result<(), String> {
    let n = indptr.len() - 1;
    if n == 0 || p.n_epochs == 0 || indices.is_empty() {
        return Ok(());
    }
    let wmax = w.iter().copied().fold(0f32, f32::max);
    if !(wmax > 0.0) {
        return Ok(());
    }
    let (edges, ptr) = edges_by_pair(indptr, indices, w, wmax, identity, p.blocks);
    let stages = stages(p.blocks);
    let mut inv = vec![0u32; n];
    for (v, &id) in identity.iter().enumerate() {
        inv[id as usize] = v as u32;
    }
    let mut snap = vec![0f32; n * dim];
    let mut acc = vec![0f64; n * dim];
    let (a, b, gamma, r) = (p.a, p.b, p.gamma, p.negative_sample_rate);
    let nv = n as u64;
    let mut alpha = p.initial_alpha;
    for epoch in 0..p.n_epochs {
        if cancel.load(Relaxed) {
            return Err(INTERRUPTED.into());
        }
        // epoch 0 fires nothing: an edge's first firing is at w_max / w >= 1
        if epoch > 0 {
            snap.par_chunks_mut(1 << 16).zip(pos.par_chunks(1 << 16)).for_each(|(s, q)| s.copy_from_slice(q));
            let nf = epoch as f64;
            let ep = epoch as u64;
            let pp = Shared(pos.as_mut_ptr());
            let ap = Shared(acc.as_mut_ptr());
            let (snap, inv, edges, ptr) = (&snap, &inv, &edges, &ptr);
            for stage in &stages {
                stage.par_iter().for_each(|&k| {
                    for ed in &edges[ptr[k]..ptr[k + 1]] {
                        let f = ed.rate as f64;
                        let now = (nf * f).floor();
                        if now <= ((nf - 1.0) * f).floor() {
                            continue;
                        }
                        let m = negatives(now, nf, f, r);
                        let (u, v) = (ed.u as usize, ed.v as usize);
                        let (idu, idv) = (identity[u] as u64, identity[v] as u64);
                        // SAFETY: the pairs of a stage share no node, so u, v and their sums belong to this task
                        unsafe {
                            fire(u, v, idu, hash6(p.seed, KIND_NEGATIVE, ep, idu, idv, 0), m, pp.get(), ap.get(), snap,
                                 inv, dim, nv, a, b, gamma, alpha);
                            fire(v, u, idv, hash6(p.seed, KIND_NEGATIVE, ep, idu, idv, 1), m, pp.get(), ap.get(), snap,
                                 inv, dim, nv, a, b, gamma, alpha);
                        }
                    }
                });
            }
            // the epoch's pulls on the other ends
            pos.par_chunks_mut(dim).zip(acc.par_chunks_mut(dim)).for_each(|(pv, av)| {
                for t in 0..dim {
                    pv[t] = (pv[t] as f64 + av[t]) as f32;
                    av[t] = 0.0;
                }
            });
        }
        alpha = p.initial_alpha * (1.0 - (epoch as f64) / (p.n_epochs as f64));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Python
// ---------------------------------------------------------------------------

/// UMAP's fuzzy simplicial set from a kNN result (each node itself first, at
/// distance 0, as `piaso.tl.neighbors` and umap-learn give it).
///
/// Returns ``(indptr, indices, data, sigmas, rhos)``: the symmetric graph in
/// CSR (float32, columns sorted), and each node's bandwidth and distance to
/// its nearest neighbor. Equal to umap-learn's ``fuzzy_simplicial_set`` to
/// float32 rounding.
#[pyfunction]
#[pyo3(signature = (knn_indices, knn_dists, local_connectivity=1.0, bandwidth=1.0, n_threads=0))]
#[allow(clippy::type_complexity)]
pub fn umap_fuzzy_graph<'py>(
    py: Python<'py>,
    knn_indices: Bound<'py, PyAny>,
    knn_dists: PyReadonlyArray2<'py, f32>,
    local_connectivity: f64,
    bandwidth: f64,
    n_threads: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>,
               Bound<'py, PyArray1<f32>>)> {
    let shape = knn_dists.shape();
    let (n, k) = (shape[0], shape[1]);
    let idx: Vec<i64> = if let Ok(a) = knn_indices.extract::<PyReadonlyArray2<'py, i64>>() {
        if a.shape() != shape {
            return Err(PyValueError::new_err("knn_indices and knn_dists must have the same shape"));
        }
        a.as_array().iter().copied().collect()
    } else {
        let a: PyReadonlyArray2<'py, i32> = knn_indices.extract()?;
        if a.shape() != shape {
            return Err(PyValueError::new_err("knn_indices and knn_dists must have the same shape"));
        }
        a.as_array().iter().map(|&x| x as i64).collect()
    };
    let dist: Vec<f32> = knn_dists.as_array().iter().copied().collect();
    if k < 2 {
        return Err(PyValueError::new_err("the kNN needs at least two columns (the node itself and a neighbor)"));
    }
    if idx.iter().any(|&c| c >= n as i64) {
        return Err(PyValueError::new_err("knn_indices holds an index outside 0..n"));
    }
    if dist.iter().any(|&x| !x.is_finite() || x < 0.0) {
        return Err(PyValueError::new_err("knn_dists must be finite and not negative"));
    }
    if !(local_connectivity >= 0.0) || !(bandwidth > 0.0) {
        return Err(PyValueError::new_err("local_connectivity must be >= 0 and bandwidth > 0"));
    }
    let threads = if n < SINGLE { 1 } else { n_threads };
    let (g, sigmas, rhos) = crate::threads::run_released(py, "umap_fuzzy_graph", threads, |_cancel| {
        fuzzy_graph(&idx, &dist, n, k, local_connectivity, bandwidth)
    })?;
    Ok((g.indptr.into_pyarray(py), g.indices.into_pyarray(py), g.data.into_pyarray(py), sigmas.into_pyarray(py),
        rhos.into_pyarray(py)))
}

/// ``y = s * (G @ (s * x)) + shift * x`` for a CSR graph ``G`` and a block
/// ``x`` (n x c, float64): the products of the spectral start. With
/// ``filter=(z, h, c, k1, k2)`` it returns ``((h * y - c * x) * k1) - k2 * z``
/// instead, one step of the start's Chebyshev filter, rounded as numpy would
/// round those operations one by one. One row per task, so the result does not
/// depend on ``n_threads``.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, scale, x, shift=0.0, n_threads=0, filter=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn umap_graph_matmat<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    shift: f64,
    n_threads: usize,
    filter: Option<(PyReadonlyArray2<'py, f64>, f64, f64, f64, f64)>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ip, ix, d, s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, scale.as_slice()?);
    let shape = x.shape();
    let (n, c) = (shape[0], shape[1]);
    if ip.len() != n + 1 || s.len() != n || ix.len() != d.len() || ip[n] as usize != ix.len() {
        return Err(PyValueError::new_err("graph, scale and x do not agree in size"));
    }
    if ix.iter().any(|&j| j < 0 || j as usize >= n) {
        return Err(PyValueError::new_err("the graph holds a column outside 0..n"));
    }
    let xs: Vec<f64> = x.as_array().iter().copied().collect();
    let zs: Option<(Vec<f64>, f64, f64, f64, f64)> = match filter {
        Some((z, h, cc, k1, k2)) => {
            if z.shape() != shape {
                return Err(PyValueError::new_err("filter's z must have x's shape"));
            }
            Some((z.as_array().iter().copied().collect(), h, cc, k1, k2))
        }
        None => None,
    };
    let threads = if n < SINGLE { 1 } else { n_threads };
    let y = crate::threads::run_released(py, "umap_graph_matmat", threads, |_cancel| {
        let step = zs.as_ref().map(|(z, h, cc, k1, k2)| FilterStep { z, h: *h, c: *cc, k1: *k1, k2: *k2 });
        graph_matmat(ip, ix, d, s, &xs, c, shift, step.as_ref())
    })?;
    let arr = numpy::ndarray::Array2::from_shape_vec((n, c), y).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// The spectral start's Chebyshev filter in one call: from the block ``x``
/// (n x c, float64) and ``ax``, its product with
/// ``A = (I + D^-1/2 G D^-1/2) / 2``, the block filtered by the polynomial of
/// ``degree`` that maps [0, 2 c] to [-1, 1] (``e = c``, ``sigma1 = e / (theta -
/// c)`` for the block's largest Ritz value ``theta``), by Zhou and Saad's
/// scaled three-term recurrence. Each step is ``umap_graph_matmat`` with its
/// filter, and the result equals those calls made one after another from
/// Python to the bit, holding three blocks instead of copying each step's in
/// and out. One row per task, so it does not depend on ``n_threads``.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, scale, x, ax, c, e, sigma1, degree, n_threads=0))]
#[allow(clippy::too_many_arguments)]
pub fn umap_chebyshev_filter<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    ax: PyReadonlyArray2<'py, f64>,
    c: f64,
    e: f64,
    sigma1: f64,
    degree: usize,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ip, ix, d, s) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, scale.as_slice()?);
    let shape = x.shape();
    let (n, cols) = (shape[0], shape[1]);
    if ax.shape() != shape {
        return Err(PyValueError::new_err("ax must have x's shape"));
    }
    if ip.len() != n + 1 || s.len() != n || ix.len() != d.len() || ip[n] as usize != ix.len() {
        return Err(PyValueError::new_err("graph, scale and x do not agree in size"));
    }
    if ix.iter().any(|&j| j < 0 || j as usize >= n) {
        return Err(PyValueError::new_err("the graph holds a column outside 0..n"));
    }
    if degree < 1 || cols == 0 || !(e > 0.0) {
        return Err(PyValueError::new_err("degree must be at least 1, x must have a column, and e must be positive"));
    }
    let xs: Vec<f64> = x.as_array().iter().copied().collect();
    let axs: Vec<f64> = ax.as_array().iter().copied().collect();
    let threads = if n < SINGLE { 1 } else { n_threads };
    let y = crate::threads::run_released(py, "umap_chebyshev_filter", threads, |cancel| {
        chebyshev_filter(ip, ix, d, s, &xs, &axs, cols, c, e, sigma1, degree, cancel)
    })?
    .map_err(PyValueError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((n, cols), y).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// UMAP's layout of a symmetric fuzzy graph from a start, parallel and
/// deterministic (see the module's notes).
///
/// ``init`` (n x dim, float32) is the start, as umap-learn scales it before
/// its epochs; the result is a new array. One graph, start and ``seed`` give
/// one layout whatever ``n_threads`` (0: the process's cores; one thread below
/// 4,096 nodes) and whatever the order of the nodes, given ``identity``
/// (their names; default: positions). ``blocks`` (even, 2 to 1,024; default
/// 128) sets how the epochs' work is cut: another number gives another layout,
/// as another seed does.
#[pyfunction]
#[pyo3(signature = (indptr, indices, weights, init, n_epochs, a, b, gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0,
                    seed=0, identity=None, n_threads=0, blocks=LAYOUT_BLOCKS))]
#[allow(clippy::too_many_arguments)]
pub fn umap_layout_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    weights: PyReadonlyArray1<'py, f32>,
    init: PyReadonlyArray2<'py, f32>,
    n_epochs: usize,
    a: f64,
    b: f64,
    gamma: f64,
    initial_alpha: f64,
    negative_sample_rate: f64,
    seed: u64,
    identity: Option<PyReadonlyArray1<'py, i64>>,
    n_threads: usize,
    blocks: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (ip, ix, w) = (indptr.as_slice()?, indices.as_slice()?, weights.as_slice()?);
    let shape = init.shape();
    let (n, dim) = (shape[0], shape[1]);
    if dim == 0 {
        return Err(PyValueError::new_err("init must have at least one column"));
    }
    if n > u32::MAX as usize {
        return Err(PyValueError::new_err("the layout takes at most 4,294,967,295 nodes"));
    }
    if !(a > 0.0) || !(b > 0.0) || !(gamma >= 0.0) || !(initial_alpha > 0.0) || !(negative_sample_rate >= 0.0) {
        return Err(PyValueError::new_err("a, b and initial_alpha must be positive, gamma and negative_sample_rate not negative"));
    }
    if blocks < 2 || blocks % 2 != 0 || blocks > MAX_BLOCKS {
        return Err(PyValueError::new_err("blocks must be an even number from 2 to 1024"));
    }
    let id: Vec<u32> = match identity {
        Some(arr) => {
            let s = arr.as_slice()?;
            if s.iter().any(|&x| x < 0 || x > u32::MAX as i64) {
                return Err(PyValueError::new_err("identity must be a permutation of 0..n"));
            }
            s.iter().map(|&x| x as u32).collect()
        }
        None => (0..n as u32).collect(),
    };
    let mut pos: Vec<f32> = init.as_array().iter().copied().collect();
    if pos.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("init must be finite"));
    }
    let params = LayoutParams { n_epochs, a, b, gamma, initial_alpha, negative_sample_rate, blocks, seed };
    let threads = if n < LAYOUT_SINGLE { 1 } else { n_threads };
    // the graph's check is parallel too, so it runs on the call's own pool:
    // rayon's global pool does not survive a fork
    let result = crate::threads::run_released(py, "umap_layout", threads, |cancel| {
        check_graph(ip, ix, w, n, &id)?;
        layout(ip, ix, w, &mut pos, dim, &id, &params, cancel).map(|_| pos)
    })?;
    let pos = result.map_err(PyValueError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((n, dim), pos).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        // a ring with chords: symmetric, connected
        let mut rows: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n];
        for i in 0..n {
            for &d in &[1usize, 2, 7] {
                let j = (i + d) % n;
                let w = 0.3 + 0.7 * ((i * 31 + d) % 10) as f32 / 10.0;
                rows[i].push((j as i32, w));
                rows[j].push((i as i32, w));
            }
        }
        let mut ip = vec![0i64];
        let (mut ix, mut w) = (Vec::new(), Vec::new());
        for r in rows.iter_mut() {
            r.sort_by_key(|x| x.0);
            let mut merged: Vec<(i32, f32)> = Vec::new();
            for &(c, x) in r.iter() {
                match merged.last_mut() {
                    Some(l) if l.0 == c => l.1 = l.1.max(x),
                    _ => merged.push((c, x)),
                }
            }
            for (c, x) in merged {
                ix.push(c);
                w.push(x);
            }
            ip.push(ix.len() as i64);
        }
        // symmetric by construction only if the weights agree both ways: take the max per pair
        let mut wfix = w.clone();
        for i in 0..n {
            for e in ip[i] as usize..ip[i + 1] as usize {
                let j = ix[e] as usize;
                let p = (ip[j] as usize..ip[j + 1] as usize).find(|&q| ix[q] as usize == i).unwrap();
                let m = w[e].max(w[p]);
                wfix[e] = m;
                wfix[p] = m;
            }
        }
        (ip, ix, wfix)
    }

    fn start(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| [((i as f32) * 0.37).sin() * 10.0, ((i as f32) * 0.11).cos() * 10.0]).collect()
    }

    fn run(n: usize, threads: usize, blocks: usize) -> Vec<f32> {
        static NEVER: AtomicBool = AtomicBool::new(false);
        let (ip, ix, w) = ring(n);
        let id: Vec<u32> = (0..n as u32).collect();
        let mut pos = start(n);
        let p = LayoutParams { n_epochs: 30, a: 1.577, b: 0.895, gamma: 1.0, initial_alpha: 1.0, negative_sample_rate: 5.0,
                               blocks, seed: 7 };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| layout(&ip, &ix, &w, &mut pos, 2, &id, &p, &NEVER)).unwrap();
        pos
    }

    #[test]
    fn threads_do_not_enter_the_result() {
        let a = run(3000, 1, 64);
        let b = run(3000, 4, 64);
        assert_eq!(a, b);
        assert!(a.iter().all(|x| x.is_finite()));
        assert_ne!(a, start(3000));
    }

    #[test]
    fn blocks_change_the_path_not_the_validity() {
        let a = run(2000, 2, 2);
        let b = run(2000, 2, 64);
        assert!(a.iter().chain(b.iter()).all(|x| x.is_finite()));
        assert_ne!(a, b);
    }

    #[test]
    fn every_pair_of_blocks_meets_once_and_a_stage_shares_no_block() {
        for &p in &[2usize, 4, 6, 64] {
            let st = stages(p);
            assert_eq!(st.len(), p);
            let mut seen = vec![0u32; p * p];
            for stage in &st {
                let mut used = vec![false; p];
                for &k in stage {
                    seen[k] += 1;
                    let (lo, hi) = (k / p, k % p);
                    assert!(lo <= hi);
                    assert!(!used[lo] && (lo == hi || !used[hi]));
                    used[lo] = true;
                    used[hi] = true;
                }
            }
            for lo in 0..p {
                for hi in 0..p {
                    assert_eq!(seen[lo * p + hi], if lo <= hi { 1 } else { 0 });
                }
            }
        }
    }

    /// umap-learn's counters (optimize_layout_euclidean) in exact arithmetic,
    /// for rates that float64 holds exactly (w / w_max = num / den, den a power
    /// of two): the epochs of the firings and each firing's count agree.
    /// (umap-learn's own float64 sums drift, e.g. at rate 3/4 its counter
    /// reaches 3.9999 where exact arithmetic gives 4, and it then draws one
    /// sample fewer; the layout follows the exact count.)
    #[test]
    fn the_counts_are_umap_learns() {
        let n_epochs = 500i64;
        let r = 5i64;
        for &(num, den) in &[(1i64, 1i64), (1, 2), (1, 4), (3, 4), (1, 8), (5, 8), (3, 8), (7, 16), (1, 64)] {
            let f = num as f64 / den as f64;
            // in units of 1 / (num r): an interval of den r, a negative-sample interval of den
            let unit = num * r;
            let (mut next, mut next_neg) = (den * r, den);
            let (mut fired, mut drawn, mut last) = (0u64, 0i64, 0i64);
            for n in 0..n_epochs {
                let theirs = if next <= n * unit {
                    let m = (n * unit - next_neg) / den; // int(): toward zero
                    next += den * r;
                    next_neg += m * den;
                    Some(m.max(0) as u64)
                } else {
                    None
                };
                let nf = n as f64;
                let k = (nf * f).floor();
                let ours = if n > 0 && k > ((nf - 1.0) * f).floor() { Some(negatives(k, nf, f, r as f64)) } else { None };
                assert_eq!(theirs, ours, "rate {}/{} epoch {}", num, den, n);
                if let Some(m) = ours {
                    fired += 1;
                    drawn += m as i64;
                    last = n;
                }
            }
            // r per firing on average: floor(n f r) - 1 in all by the last firing
            assert!(fired > 0);
            assert_eq!(drawn + 1, (last as f64 * (f * r as f64)).floor() as i64);
        }
    }

    /// A second construction of the graph, written out plainly, to check the module's against.
    fn reference_fuzzy(idx: &[i64], dist: &[f32], n: usize, k: usize) -> Csr {
        fn to_csr(n: usize, rows: Vec<Vec<(i32, f32)>>) -> Csr {
            let mut indptr = vec![0i64];
            let (mut indices, mut data) = (Vec::new(), Vec::new());
            for r in rows {
                for (c, v) in r {
                    indices.push(c);
                    data.push(v);
                }
                indptr.push(indices.len() as i64);
            }
            Csr { n, indptr, indices, data }
        }
        let mean_all = dist.iter().map(|&x| x as f64).sum::<f64>() / dist.len() as f64;
        let sr: Vec<(f32, f32)> = (0..n).map(|i| smooth_knn_row(&dist[i * k..(i + 1) * k], mean_all, 1.0, 1.0)).collect();
        let rows: Vec<Vec<(i32, f32)>> = (0..n).map(|i| {
            let mut row = Vec::new();
            for j in 0..k {
                let c = idx[i * k + j];
                if c < 0 { continue; }
                let d = dist[i * k + j];
                let (sig, rho) = sr[i];
                let v: f32 = if c as usize == i { 0.0 } else if d - rho <= 0.0 || sig == 0.0 { 1.0 }
                             else { libm::expf(-((d - rho) / sig)) };
                row.push((c as i32, v));
            }
            row.sort_by_key(|x| x.0);
            let mut out: Vec<(i32, f32)> = Vec::new();
            for (c, v) in row {
                match out.last_mut() { Some(l) if l.0 == c => l.1 += v, _ => out.push((c, v)) }
            }
            out.retain(|x| x.1 != 0.0);
            out
        }).collect();
        let a = to_csr(n, rows);
        let t = transpose(&a);
        let rows: Vec<Vec<(i32, f32)>> = (0..n).map(|i| {
            let mut m: Vec<(i32, f32, f32)> = Vec::new();
            for e in a.indptr[i] as usize..a.indptr[i + 1] as usize { m.push((a.indices[e], a.data[e], 0.0)); }
            for e in t.indptr[i] as usize..t.indptr[i + 1] as usize {
                match m.iter_mut().find(|x| x.0 == t.indices[e]) { Some(x) => x.2 = t.data[e],
                                                                  None => m.push((t.indices[e], 0.0, t.data[e])) }
            }
            m.sort_by_key(|x| x.0);
            m.into_iter().map(|(c, x, y)| (c, (x + y) - x * y)).filter(|x| x.1 != 0.0).collect()
        }).collect();
        to_csr(n, rows)
    }

    #[test]
    fn the_fuzzy_graph_matches_a_plain_construction() {
        let (n, k) = (700usize, 9usize);
        let mut s = 12345u64;
        let mut rnd = || { s = mix(s.wrapping_add(GOLD)); s };
        let mut idx = Vec::new();
        let mut dist = Vec::new();
        for i in 0..n {
            idx.push(i as i64);
            dist.push(0.0f32);
            let mut d = 0.0f32;
            for t in 1..k {
                let r = rnd();
                // a missing neighbor now and then, and repeated ones
                let c = if r % 23 == 0 { -1 } else if r % 17 == 0 { idx[i * k + t - 1] } else { (r % n as u64) as i64 };
                idx.push(if c == i as i64 { ((i + 1) % n) as i64 } else { c });
                d += 0.1 + (r % 97) as f32 * 0.01;
                dist.push(d);
            }
        }
        let (g, _, _) = fuzzy_graph(&idx, &dist, n, k, 1.0, 1.0);
        let r = reference_fuzzy(&idx, &dist, n, k);
        assert_eq!(g.indptr, r.indptr);
        assert_eq!(g.indices, r.indices);
        assert_eq!(g.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                   r.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn rows_out_of_order_are_checked_as_rows_in_order() {
        let (n, k) = (300usize, 6usize);
        let mut idx = Vec::new();
        let mut dist = Vec::new();
        for i in 0..n {
            idx.push(i as i64);
            dist.push(0.0f32);
            for t in 1..k {
                idx.push(((i + t * 7) % n) as i64);
                dist.push(t as f32 * 0.3);
            }
        }
        let (g, _, _) = fuzzy_graph(&idx, &dist, n, k, 1.0, 1.0);
        let id: Vec<u32> = (0..n as u32).collect();
        // every row reversed: still symmetric, still valid
        let (mut ix, mut w) = (g.indices.clone(), g.data.clone());
        for i in 0..n {
            let (a, b) = (g.indptr[i] as usize, g.indptr[i + 1] as usize);
            ix[a..b].reverse();
            w[a..b].reverse();
        }
        assert!(check_graph(&g.indptr, &ix, &w, n, &id).is_ok());
        // one weight changed on one side: not symmetric
        let mut w2 = w.clone();
        w2[g.indptr[5] as usize] += 0.5;
        assert!(check_graph(&g.indptr, &ix, &w2, n, &id).unwrap_err().contains("not symmetric"));
        // an edge listed twice
        let mut ix3 = ix.clone();
        let a = g.indptr[3] as usize;
        ix3[a + 1] = ix3[a];
        assert!(check_graph(&g.indptr, &ix3, &w, n, &id).unwrap_err().contains("twice"));
    }

    #[test]
    fn the_union_is_symmetric() {
        let n = 500;
        let k = 6;
        let mut idx = Vec::new();
        let mut dist = Vec::new();
        for i in 0..n {
            idx.push(i as i64);
            dist.push(0.0f32);
            for t in 1..k {
                idx.push(((i + t * 13) % n) as i64);
                dist.push(t as f32 * 0.5 + (i % 7) as f32 * 0.01);
            }
        }
        let (g, _, _) = fuzzy_graph(&idx, &dist, n, k, 1.0, 1.0);
        let id: Vec<u32> = (0..n as u32).collect();
        check_graph(&g.indptr, &g.indices, &g.data, n, &id).unwrap();
    }
}
