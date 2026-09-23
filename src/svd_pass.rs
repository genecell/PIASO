//! One pass of the streaming randomized SVD, in Rust.
//!
//! `_streaming_rsvd` needs two products from the stored matrix: `X @ B`
//! (n_cells x k) and `X^T @ Y` (n_features x k), with TF-IDF applied to the
//! stored counts on the way and a column mask selecting the features. This
//! computes either in one sweep over the stored chunks, in parallel, with
//! bounded memory and a result that does not depend on the thread count.
//!
//! **What a pass costs.** Measured on one thread with the dense operand in
//! cache, an entry costs about 40 ns at k = 40: 25 to decompress and walk
//! the chunk, 15 for the forty-wide multiply-add. The same loop costs three
//! to four times that once the operand -- `B` on a row-major store, `Y` or
//! the output on a column-major one, `min(n_cells, n_features) x k x 8`
//! bytes -- no longer fits in cache, and ten times that when twenty threads
//! contend for memory. Every entry touches one row of that operand at a
//! random address, and that access is the whole cost of the pass.
//!
//! **So the pass is blocked on the operand.** Each chunk, once decompressed,
//! is bucketed by a cache-sized block of the operand's rows with a stable
//! counting sort, two linear passes over data already in memory. The
//! multiply then walks the chunk block by block, so every random access
//! lands in cache. The bucketed chunk is transient; the store is read as it
//! is and never rewritten.
//!
//! **Reading.** Only what is selected is read: a chunk with no kept column
//! (column chunks) or no kept cell (row chunks) is never fetched, and an
//! unkept column or cell never enters the bucket. Blobs are decompressed
//! straight out of SQLite's buffer into vectors of the size their headers
//! record, and each thread reads a run of chunks inside one read
//! transaction, since a transaction per chunk from twenty threads at once
//! serialised on the store's locks.
//!
//! **Two products, one representation.** On a given layout one product is
//! *owned*: each chunk's non-zeros land in output rows no other chunk
//! touches (`X @ B` on row chunks, `X^T @ Y` on column chunks), so chunks
//! run in parallel and each writes its own slice of the output. The other
//! product is *owner-computes*: a batch of chunks is bucketed in parallel,
//! then threads own blocks of the output and walk the batch's chunks in
//! order. No two threads write the same row, so there are no partial
//! buffers, no reduction and no `unsafe`.
//!
//! **Numerics.** The transform mirrors `_normalize_chunk_tfidf` step for
//! step in the precision numpy used. Products accumulate in `f64`. For every
//! output element the terms arrive in one fixed order -- chunk order, then
//! stored order within the chunk -- whatever the thread count, block size or
//! batch size, so a result is reproducible on any machine. On canonical
//! chunks (indices ascending within a row or column, which every writer here
//! produces) the owned product's order is exactly the stored order.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rusqlite::Connection;

use crate::encoder;

/// Bytes of the dense operand a block is allowed to span: rows of `Y` or
/// `B`, or of the output, that one block's entries touch. Sized to sit in a
/// core's second-level cache with room for the entry stream.
const CACHE_BLOCK_BYTES: usize = 256 * 1024;

/// Blocks per thread the owner-computes product aims for, so small matrices
/// still spread over the machine. Affects speed only, never the result.
const BLOCKS_PER_THREAD: usize = 4;

/// Bucketed-chunk memory one owner-computes batch may hold, beyond the one
/// chunk per thread it always holds so the bucketing phase stays parallel.
const BATCH_BYTES: usize = 512 * 1024 * 1024;

/// Bytes per bucketed entry: two `u32` indices and an `f64` weight.
const ENTRY_BYTES: usize = 16;

/// Where a pass spends its time, summed over threads, printed to stderr when
/// `PIASO_SVD_PROFILE` is set. Diagnostic only; off by default and free.
#[derive(Default)]
struct Profile {
    on: bool,
    read_ns: std::sync::atomic::AtomicU64,
    fetch_ns: std::sync::atomic::AtomicU64,
    bucket_ns: std::sync::atomic::AtomicU64,
    multiply_ns: std::sync::atomic::AtomicU64,
}

impl Profile {
    fn new() -> Self {
        Profile { on: std::env::var_os("PIASO_SVD_PROFILE").is_some(), ..Default::default() }
    }
    #[inline]
    fn time<T>(&self, slot: &std::sync::atomic::AtomicU64, f: impl FnOnce() -> T) -> T {
        if !self.on { return f(); }
        let t = std::time::Instant::now();
        let out = f();
        slot.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        out
    }
    fn report(&self, what: &str, nnz: u64, wall: f64, threads: usize) {
        if !self.on { return; }
        let g = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9;
        let (r, f, b, m) = (g(&self.read_ns), g(&self.fetch_ns), g(&self.bucket_ns), g(&self.multiply_ns));
        eprintln!("svd_pass {}: wall {:.1}s on {} threads; thread-seconds read+decompress {:.1} \
                   (of which sqlite fetch {:.1}), bucket {:.1}, multiply {:.1}; per entry per thread {:.0} ns",
                  what, wall, threads, r, f, b, m, wall * threads as f64 / nnz.max(1) as f64 * 1e9);
    }
}

struct ChunkMeta {
    idx: i64,
    /// First row (CSR) or first column (CSC) of the chunk.
    start: i64,
    /// One past the last row (CSR) or column (CSC).
    end: i64,
    nnz: i64,
}

/// Which table the chunks live in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Csr,
    Csc,
}

struct Chunk {
    /// First row (CSR) or first column (CSC).
    start: i64,
    data: Data,
    indices: Vec<i32>,
    indptr: Vec<i32>,
}

enum Data {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// TF-IDF parameters, or `None` for the identity transform.
struct Tfidf<'a> {
    depth: &'a [f64],
    idf: &'a [f32],
    scale: f64,
}

/// A read-only connection, locking or snapshot (see `store::open_ro`).
fn open_ro(path: &str, snapshot: bool) -> PyResult<Connection> {
    crate::store::open_ro(path, snapshot).map_err(PyValueError::new_err)
}

/// What the store holds for a matrix, and which layout a pass should use.
struct LayoutChoice {
    layout: Layout,
    has_csr: bool,
    has_csc: bool,
    n_rows: i64,
    n_cols: i64,
    dtype: String,
}

/// Choose the layout by the size of the dense operand the pass hits at
/// random, which is the whole cost model.
///
/// On row chunks that operand is `n_features x k` (`B`, or the output of
/// `X^T @ Y`); on column chunks it is `n_cells x k` (`Y`, or the output of
/// `X @ B`). Blocking keeps each access in cache either way, but the
/// smaller operand means fewer blocks and less bucketing, so with both
/// layouts on disk the pass takes column-major when there are fewer cells
/// than features (every peak and tile matrix) and row-major otherwise (a
/// gene matrix). Measured before blocking, the wrong choice was 8x on the
/// transpose product.
fn resolve_layout(conn: &Connection, matrix: &str, want: &str) -> PyResult<LayoutChoice> {
    let (n_rows, n_cols, dtype, n_chunks, has_csc): (i64, i64, String, i64, i64) = conn
        .query_row(
            "SELECT n_rows, n_cols, dtype, COALESCE(n_chunks, 0), COALESCE(has_csc, 0) \
             FROM matrix_meta WHERE matrix_name = ?1",
            [matrix],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .map_err(|e| PyValueError::new_err(format!("matrix {}: {}", matrix, e)))?;
    let has_csr = n_chunks > 0;
    let has_csc = has_csc != 0;
    let layout = match want {
        "csr" => Layout::Csr,
        "csc" => Layout::Csc,
        "auto" => match (has_csr, has_csc) {
            (true, true) => if n_rows <= n_cols { Layout::Csc } else { Layout::Csr },
            (true, false) => Layout::Csr,
            (false, true) => Layout::Csc,
            (false, false) => return Err(PyValueError::new_err(
                format!("matrix {} has no chunks in either layout", matrix))),
        },
        other => return Err(PyValueError::new_err(format!("layout must be auto, csr or csc, got {:?}", other))),
    };
    Ok(LayoutChoice { layout, has_csr, has_csc, n_rows, n_cols, dtype })
}

/// Which layout a pass over `matrix_name` would take, and what is on disk.
///
/// Returns ``(layout, has_row_chunks, has_column_chunks, n_rows, n_cols)``,
/// with ``layout`` one of ``"csr"`` / ``"csc"``. Lets the caller say at
/// verbosity 1 which layout is in use, and tell a user with a row-major-only
/// peak matrix that a column-major store would be faster.
#[pyfunction]
pub fn svd_layout(cytome_path: &str, matrix_name: &str) -> PyResult<(String, bool, bool, i64, i64)> {
    let conn = open_ro(cytome_path, false)?;
    let c = resolve_layout(&conn, matrix_name, "auto")?;
    Ok((
        if c.layout == Layout::Csr { "csr".to_string() } else { "csc".to_string() },
        c.has_csr, c.has_csc, c.n_rows, c.n_cols,
    ))
}

fn chunk_list(conn: &Connection, matrix: &str, want: &str)
    -> PyResult<(Vec<ChunkMeta>, String, i64, i64, Layout)> {
    let LayoutChoice { layout, n_rows, n_cols, dtype, .. } = resolve_layout(conn, matrix, want)?;
    let sql = match layout {
        Layout::Csr => "SELECT chunk_idx, row_start, row_end, n_nonzero FROM matrix_chunks \
                        WHERE matrix_name = ?1 ORDER BY chunk_idx",
        Layout::Csc => "SELECT chunk_idx, col_start, col_end, n_nonzero FROM matrix_csc_chunks \
                        WHERE matrix_name = ?1 ORDER BY chunk_idx",
    };
    let mut stmt = conn.prepare(sql).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let chunks = stmt
        .query_map([matrix], |r| Ok(ChunkMeta {
            idx: r.get(0)?, start: r.get(1)?, end: r.get(2)?,
            nnz: r.get::<_, Option<i64>>(3)?.unwrap_or(0),
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if chunks.is_empty() {
        return Err(PyValueError::new_err(format!("matrix {} has no {} chunks", matrix,
            if layout == Layout::Csr { "row" } else { "column" })));
    }
    Ok((chunks, dtype, n_rows, n_cols, layout))
}

/// Decompress a blob straight into a typed buffer when its header states the
/// decompressed size (every zstd frame and lz4 block cytome writes does), so
/// there is no growing buffer and no second copy. Falls back to the general
/// path otherwise. Little-endian only, like the stores.
fn decode_into<T: Copy + Default + FromLe>(blob: &[u8], comp: &str) -> Vec<T> {
    let w = std::mem::size_of::<T>();
    // Same rule as the Python reader: an "lz4" label is trusted; under the
    // legacy "zstd" label the magic bytes decide between zstd, zlib and lz4.
    let sniff = comp == "zstd";
    let zstd_magic = sniff && blob.len() >= 4 && blob[..4] == [0x28, 0xb5, 0x2f, 0xfd];
    let zlib = sniff && blob.len() >= 2 && blob[0] == 0x78 && matches!(blob[1], 0x01 | 0x5e | 0x9c | 0xda);
    let size: Option<usize> = if zstd_magic {
        zstd::zstd_safe::get_frame_content_size(blob).ok().flatten().map(|s| s as usize)
    } else if (comp == "lz4" || (sniff && !zlib)) && blob.len() >= 4 {
        Some(u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize)
    } else { None };
    if let Some(sz) = size {
        if sz % w == 0 {
            let mut out: Vec<T> = vec![T::default(); sz / w];
            // SAFETY: T is a plain little-endian number type; we view its
            // storage as bytes to decompress into it.
            let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, sz) };
            let n = if zstd_magic {
                zstd::bulk::decompress_to_buffer(blob, bytes).expect("zstd decompression failed")
            } else {
                lz4_flex::block::decompress_into(&blob[4..], bytes).expect("lz4 decompression failed")
            };
            assert_eq!(n, sz, "decompressed size disagrees with the header");
            return out;
        }
    }
    let raw = encoder::decompress_blob(blob, comp);
    raw.chunks_exact(w).map(T::from_le).collect()
}

trait FromLe { fn from_le(b: &[u8]) -> Self; }
impl FromLe for i32 { fn from_le(b: &[u8]) -> Self { i32::from_le_bytes(b.try_into().unwrap()) } }
impl FromLe for f32 { fn from_le(b: &[u8]) -> Self { f32::from_le_bytes(b.try_into().unwrap()) } }
impl FromLe for f64 { fn from_le(b: &[u8]) -> Self { f64::from_le_bytes(b.try_into().unwrap()) } }
impl FromLe for u16 { fn from_le(b: &[u8]) -> Self { u16::from_le_bytes(b.try_into().unwrap()) } }
impl FromLe for i16 { fn from_le(b: &[u8]) -> Self { i16::from_le_bytes(b.try_into().unwrap()) } }
impl FromLe for u32 { fn from_le(b: &[u8]) -> Self { u32::from_le_bytes(b.try_into().unwrap()) } }
impl FromLe for i64 { fn from_le(b: &[u8]) -> Self { i64::from_le_bytes(b.try_into().unwrap()) } }
impl FromLe for i8 { fn from_le(b: &[u8]) -> Self { b[0] as i8 } }
impl FromLe for u8 { fn from_le(b: &[u8]) -> Self { b[0] } }

/// Read one chunk. The blobs stay in SQLite's buffer and are decompressed
/// from there straight into typed vectors of the size their headers state:
/// no copy out of SQLite, no growing buffer, no second pass to reinterpret
/// bytes. Stored integers become float32 and float64 stays float64, as in
/// `_normalize_chunk_tfidf`.
fn read_chunk(conn: &Connection, matrix: &str, meta: &ChunkMeta, dtype: &str, layout: Layout,
              prof: &Profile) -> Result<Chunk, String> {
    let sql = match layout {
        Layout::Csr => "SELECT data_blob, indices_blob, indptr_blob, compression FROM matrix_chunks \
                        WHERE matrix_name = ?1 AND chunk_idx = ?2",
        Layout::Csc => "SELECT data_blob, indices_blob, indptr_blob, compression FROM matrix_csc_chunks \
                        WHERE matrix_name = ?1 AND chunk_idx = ?2",
    };
    let mut stmt = conn.prepare_cached(sql).map_err(|e| e.to_string())?;
    let r: Result<(Result<Data, String>, Vec<i32>, Vec<i32>), String> = prof.time(&prof.fetch_ns, || stmt.query_row(
        rusqlite::params![matrix, meta.idx],
        |r| {
            let comp: String = r.get(3)?;
            let d = r.get_ref(0)?.as_blob()?;
            let widen = |v: Vec<f32>| Ok(Data::F32(v));
            let data = match dtype {
                "float32" => widen(decode_into::<f32>(d, &comp)),
                "float64" => Ok(Data::F64(decode_into::<f64>(d, &comp))),
                "int32" => widen(decode_into::<i32>(d, &comp).into_iter().map(|v| v as f32).collect()),
                "int64" => widen(decode_into::<i64>(d, &comp).into_iter().map(|v| v as f32).collect()),
                "int16" => widen(decode_into::<i16>(d, &comp).into_iter().map(|v| v as f32).collect()),
                "uint16" => widen(decode_into::<u16>(d, &comp).into_iter().map(|v| v as f32).collect()),
                "uint32" => widen(decode_into::<u32>(d, &comp).into_iter().map(|v| v as f32).collect()),
                "int8" => widen(decode_into::<i8>(d, &comp).into_iter().map(|v| v as f32).collect()),
                "uint8" => widen(decode_into::<u8>(d, &comp).into_iter().map(|v| v as f32).collect()),
                other => Err(format!("unsupported matrix dtype {:?}", other)),
            };
            let ind = decode_into::<i32>(r.get_ref(1)?.as_blob()?, &comp);
            let ptr = decode_into::<i32>(r.get_ref(2)?.as_blob()?, &comp);
            Ok((data, ind, ptr))
        }).map_err(|e| format!("chunk {}: {}", meta.idx, e)));
    let (data, indices, indptr) = r?;
    Ok(Chunk { start: meta.start, data: data?, indices, indptr })
}

/// The TF-IDF transform of one stored value, in the precision numpy used.
///
/// `_normalize_chunk_tfidf` on float32 data: `np.divide(f32, f64, out=f32)`
/// computes in f64 and stores f32; `*= scale` is a weak Python scalar, so
/// f32; `np.log1p` on f32; `*= idf[indices]` with idf f32. On float64 data
/// every step is f64.
#[inline(always)]
fn tf_f32(v: f32, depth: f64, scale: f64, idf: f32) -> f64 {
    let t1 = (v as f64 / depth) as f32;
    let t2 = t1 * (scale as f32);
    let t3 = t2.ln_1p();
    (t3 * idf) as f64
}
#[inline(always)]
fn tf_f64(v: f64, depth: f64, scale: f64, idf: f32) -> f64 {
    ((v / depth) * scale).ln_1p() * (idf as f64)
}

/// The TF-IDF weight of one stored value, or the value itself.
///
/// `f32_data` says which precision the store held, so a float32 value that
/// was widened to `f64` in the bucket is transformed exactly as numpy would
/// have transformed the float32.
#[inline(always)]
fn weigh(f32_data: bool, v: f64, tfidf: Option<&Tfidf>, r: usize, j: usize) -> f64 {
    match tfidf {
        None => v,
        Some(t) => if f32_data { tf_f32(v as f32, t.depth[r].max(1.0), t.scale, t.idf[j]) }
                   else { tf_f64(v, t.depth[r].max(1.0), t.scale, t.idf[j]) },
    }
}

/// A chunk's entries in global indices, bucketed by block of the operand.
///
/// `own` is the index on the dimension the chunk owns (its columns on a
/// column chunk, its rows on a row chunk); `other` the stored index on the
/// dimension the pass blocks. Entries of block `i` are
/// `offsets[i]..offsets[i + 1]`; within a block they keep the stored order,
/// because the sort is stable. Nothing is looked up here: the maps, the
/// depth and the IDF are read in the multiply, block by block, where they
/// are in cache. Excluded rows and columns ride along and are skipped there.
#[derive(Default)]
struct Bucketed {
    own: Vec<u32>,
    other: Vec<u32>,
    /// The stored value, exactly; float32 widens without rounding.
    val: Vec<f64>,
    offsets: Vec<usize>,
    f32_data: bool,
}

/// Per-thread cursor for the placing pass, kept and reused.
///
/// Bucketed chunks run to tens of megabytes, and allocating them afresh for
/// every chunk meant a page fault per 4 KB, all threads contending for the
/// same kernel lock: five million faults and a sixth of the CPU time on one
/// MC pass. Buffers are cleared, never freed, until the pass ends.
#[derive(Default)]
struct Scratch {
    cursor: Vec<usize>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
    static OWN: std::cell::RefCell<Bucketed> = std::cell::RefCell::new(Bucketed::default());
}

/// Stable counting sort of a decompressed chunk by block of its stored
/// indices, into `out`. Two streaming passes: a histogram over the indices,
/// then one placing pass straight from the chunk.
fn bucket(chunk: &Chunk, block_rows: usize, n_blocks: usize, scratch: &mut Scratch, out: &mut Bucketed,
          kept: Option<&[bool]>) {
    let n_outer = chunk.indptr.len().saturating_sub(1);
    let is_kept = |o: usize| kept.map_or(true, |k| k[chunk.start as usize + o]);
    out.offsets.clear();
    out.offsets.resize(n_blocks + 1, 0);
    let mut n = 0usize;
    if kept.is_none() {
        for &i in &chunk.indices { out.offsets[i as usize / block_rows + 1] += 1; }
        n = chunk.indices.len();
    } else {
        for o in 0..n_outer {
            if !is_kept(o) { continue; }
            let (lo, hi) = (chunk.indptr[o] as usize, chunk.indptr[o + 1] as usize);
            for &i in &chunk.indices[lo..hi] { out.offsets[i as usize / block_rows + 1] += 1; }
            n += hi - lo;
        }
    }
    for i in 0..n_blocks { out.offsets[i + 1] += out.offsets[i]; }
    let cursor = &mut scratch.cursor;
    cursor.clear();
    cursor.extend_from_slice(&out.offsets[..n_blocks]);
    out.own.clear(); out.own.resize(n, 0);
    out.other.clear(); out.other.resize(n, 0);
    out.val.clear(); out.val.resize(n, 0.0);
    out.f32_data = matches!(chunk.data, Data::F32(_));
    for o in 0..n_outer {
        if !is_kept(o) { continue; }
        let (lo, hi) = (chunk.indptr[o] as usize, chunk.indptr[o + 1] as usize);
        let g = (chunk.start as usize + o) as u32;
        for p in lo..hi {
            let i = chunk.indices[p] as usize;
            let d = cursor[i / block_rows];
            cursor[i / block_rows] = d + 1;
            out.own[d] = g;
            out.other[d] = i as u32;
            out.val[d] = match &chunk.data { Data::F32(v) => v[p] as f64, Data::F64(v) => v[p] };
        }
    }
}

/// Rows of the blocked dimension per block: as many as keep one block of
/// the operand within the cache budget, and no more than gives every
/// thread a few blocks. `override_rows` (tests, experiments) wins if set.
/// The block size never changes the result.
pub(crate) fn block_rows(n_b: usize, k: usize, threads: usize, override_rows: usize) -> usize {
    if override_rows > 0 { return override_rows; }
    let cache_rows = (CACHE_BLOCK_BYTES / (k.max(1) * 8)).max(1);
    let spread_rows = n_b.div_ceil(BLOCKS_PER_THREAD * threads.max(1)).max(1);
    cache_rows.min(spread_rows)
}

/// `X_tfidf[:, sel] @ B` or `X_tfidf[:, sel]^T @ Y`, one pass over the stored
/// matrix, in parallel over chunks.
///
/// Parameters
/// ----------
/// mode : ``"xb"`` (``dense`` is ``B``, ``(n_selected, k)``; returns
///     ``(n_out_rows, k)``) or ``"xty"`` (``dense`` is ``Y``, ``(n_out_rows, k)``;
///     returns ``(n_selected, k)``).
/// col_remap : int32 ``(n_cols,)`` -- selected column j maps to ``col_remap[j]``
///     in ``0..n_selected``; ``-1`` excludes it.
/// row_map : optional int32 ``(n_rows,)`` -- kept row r maps to ``row_map[r]``
///     in ``0..n_out_rows``; ``-1`` excludes it. ``None`` keeps every row.
/// cell_depth, idf, scale : TF-IDF parameters; ``cell_depth`` empty means no
///     transform.
/// n_threads : ``0`` for every core.
/// block_rows, batch_bytes : ``0`` to let the engine choose. For tests and
///     experiments only; neither changes the result.
/// layout : ``"auto"``, ``"csr"`` or ``"csc"``.
/// snapshot : open the store ``immutable`` -- no file locks, no change
///     detection, the write-ahead log ignored. Only after the caller has
///     checkpointed the log to empty; see ``EngineOps``.
#[pyfunction]
#[pyo3(signature = (cytome_path, matrix_name, mode, dense, col_remap, n_selected,
                    row_map=None, n_out_rows=None, cell_depth=None, idf=None,
                    scale=1e4, n_threads=0, block_rows=0, batch_bytes=0, layout="auto", snapshot=false))]
#[allow(clippy::too_many_arguments)]
pub fn svd_pass<'py>(
    py: Python<'py>,
    cytome_path: &str,
    matrix_name: &str,
    mode: &str,
    dense: PyReadonlyArray2<'py, f64>,
    col_remap: PyReadonlyArray1<'py, i32>,
    n_selected: usize,
    row_map: Option<PyReadonlyArray1<'py, i32>>,
    n_out_rows: Option<usize>,
    cell_depth: Option<PyReadonlyArray1<'py, f64>>,
    idf: Option<PyReadonlyArray1<'py, f32>>,
    scale: f64,
    n_threads: usize,
    block_rows: usize,
    batch_bytes: usize,
    layout: &str,
    snapshot: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dense = dense.as_array();
    let dense = dense.as_standard_layout();
    let dense_s = dense.as_slice().ok_or_else(|| PyValueError::new_err("dense must be contiguous"))?;
    let k = dense.shape()[1];
    let remap = col_remap.as_slice()?;
    let row_map_owned = row_map.map(|m| m.as_slice().map(|s| s.to_vec())).transpose()?;
    let depth_owned = cell_depth.map(|d| d.as_slice().map(|s| s.to_vec())).transpose()?;
    let idf_owned = idf.map(|d| d.as_slice().map(|s| s.to_vec())).transpose()?;

    let conn = open_ro(cytome_path, snapshot)?;
    let (mut chunks, dtype, n_rows, n_cols, layout) = chunk_list(&conn, matrix_name, layout)?;
    drop(conn);
    if remap.len() != n_cols as usize {
        return Err(PyValueError::new_err(format!(
            "col_remap has {} entries, matrix has {} columns", remap.len(), n_cols)));
    }
    if let Some(m) = &row_map_owned {
        if m.len() != n_rows as usize {
            return Err(PyValueError::new_err(format!(
                "row_map has {} entries, matrix has {} rows", m.len(), n_rows)));
        }
    }
    let n_out = n_out_rows.unwrap_or(n_rows as usize);
    let tfidf = match (&depth_owned, &idf_owned) {
        (Some(d), Some(i)) if !d.is_empty() => {
            if d.len() != n_rows as usize || i.len() != n_cols as usize {
                return Err(PyValueError::new_err("cell_depth / idf length mismatch"));
            }
            Some(Tfidf { depth: d, idf: i, scale })
        }
        _ => None,
    };
    let row_map_ref = row_map_owned.as_deref();

    // Owned: each chunk writes its own output rows (X @ B on row chunks,
    // X^T @ Y on column chunks). Otherwise owner-computes over output blocks.
    let (out_rows, owned) = match (mode, layout) {
        ("xb", Layout::Csr) => (n_out, true),
        ("xb", Layout::Csc) => (n_out, false),
        ("xty", Layout::Csr) => (n_selected, false),
        ("xty", Layout::Csc) => (n_selected, true),
        (other, _) => return Err(PyValueError::new_err(format!("mode must be 'xb' or 'xty', got {:?}", other))),
    };
    let in_rows = if mode == "xb" { n_selected } else { n_out };
    if dense.shape()[0] != in_rows {
        return Err(PyValueError::new_err(format!(
            "dense has {} rows, expected {}", dense.shape()[0], in_rows)));
    }
    // In global indices: a column chunk owns columns and is blocked on rows;
    // a row chunk the reverse. `n_blocked` is the full stored extent of the
    // blocked dimension; kept-ness is applied in the multiply.
    let n_blocked = match layout { Layout::Csc => n_rows, Layout::Csr => n_cols } as usize;
    let threads = if n_threads == 0 { rayon::current_num_threads() } else { n_threads };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let brows = self::block_rows(n_blocked, k, threads, block_rows);
    let n_blocks = n_blocked.div_ceil(brows).max(1);
    let path = cytome_path.to_string();
    let matrix = matrix_name.to_string();
    let prof = Profile::new();
    // Never read or bucket what is not selected. A chunk's outer index (a
    // column on column chunks, a cell on row chunks) is checked in order,
    // never at random: a chunk with no kept outer index is not fetched at
    // all, and an unkept outer index never enters the bucket. The terms that
    // remain keep their order, so the result is unchanged to the bit.
    let outer_kept: Vec<bool> = match layout {
        Layout::Csc => remap.iter().map(|&v| v >= 0).collect(),
        Layout::Csr => (0..n_rows as usize).map(|r| row_map_ref.map_or(true, |m| m[r] >= 0)).collect(),
    };
    let everything = outer_kept.iter().all(|&k| k);
    let kept_ref: Option<&[bool]> = if everything { None } else { Some(&outer_kept) };
    if !everything {
        chunks.retain(|c| (c.start..c.end).any(|o| outer_kept[o as usize]));
    }
    let t_wall = std::time::Instant::now();
    let nnz_total: u64 = chunks.iter().map(|c| c.nnz.max(0) as u64).sum();

    let mut out = vec![0f64; out_rows * k];
    if out.is_empty() || n_selected == 0 || n_out == 0 {
        return Ok(numpy::ndarray::Array2::from_shape_vec((out_rows, k), out).unwrap().into_pyarray(py));
    }
    // Kept count before each stored index, on both dimensions: where a
    // chunk's (or a block's) output rows start.
    let prefix = |dim: usize, kept: &dyn Fn(usize) -> bool| -> Vec<usize> {
        let mut p = Vec::with_capacity(dim + 1);
        p.push(0usize);
        for i in 0..dim { p.push(p[i] + kept(i) as usize); }
        p
    };
    let col_prefix = prefix(n_cols as usize, &|j| remap[j] >= 0);
    let row_prefix = prefix(n_rows as usize, &|r| row_map_ref.map_or(true, |m| m[r] >= 0));
    // Mapped indices of a bucketed entry: `(a, b)` with `a` on the owned
    // dimension and `b` on the blocked one, or `None` if either is excluded;
    // and the (row, column) it came from, for the transform.
    let map = |own: u32, other: u32| -> Option<(usize, usize, usize, usize)> {
        let (r, j) = match layout { Layout::Csc => (other as usize, own as usize),
                                    Layout::Csr => (own as usize, other as usize) };
        let mr = match row_map_ref { Some(m) => { let v = m[r]; if v < 0 { return None; } v as usize }
                                     None => r };
        let mj = remap[j]; if mj < 0 { return None; }
        let mj = mj as usize;
        Some(match layout { Layout::Csc => (mj, mr, r, j), Layout::Csr => (mr, mj, r, j) })
    };

    let result: Result<(), String> = if owned {
        // Each chunk's output rows are contiguous and disjoint, in chunk
        // order, so the output splits into one slice per chunk.
        let a_prefix = match layout { Layout::Csc => &col_prefix, Layout::Csr => &row_prefix };
        let mut slices: Vec<(usize, &mut [f64])> = Vec::with_capacity(chunks.len());
        let mut rest: &mut [f64] = &mut out[..];
        let mut consumed = 0usize;
        for meta in &chunks {
            let (lo, hi) = (a_prefix[meta.start as usize], a_prefix[meta.end as usize]);
            let (_, tail) = std::mem::take(&mut rest).split_at_mut((lo - consumed) * k);
            let (mine, tail) = tail.split_at_mut((hi - lo) * k);
            slices.push((lo, mine));
            rest = tail;
            consumed = hi;
        }
        py.detach(|| pool.install(|| {
            chunks.par_iter().zip(slices.into_par_iter()).map_init(
                || open_ro(&path, snapshot).map_err(|e| e.to_string()),
                |conn, (meta, (a_lo, mine))| -> Result<(), String> {
                    let conn = conn.as_ref().map_err(|e| e.clone())?;
                    let chunk = prof.time(&prof.read_ns, || read_chunk(conn, &matrix, meta, &dtype, layout, &prof))?;
                    SCRATCH.with(|sc| OWN.with(|own| {
                        let (mut sc, mut bk) = (sc.borrow_mut(), own.borrow_mut());
                        prof.time(&prof.bucket_ns, || bucket(&chunk, brows, n_blocks, &mut sc, &mut bk, kept_ref));
                        // Block order, stored order within a block: the maps,
                        // the depth and the operand row `b` stay in cache, and
                        // each output row's sum runs in stored order.
                        prof.time(&prof.multiply_ns, || {
                            for e in 0..bk.val.len() {
                                let Some((a, b, r, j)) = map(bk.own[e], bk.other[e]) else { continue };
                                let w = weigh(bk.f32_data, bk.val[e], tfidf.as_ref(), r, j);
                                let src = &dense_s[b * k..(b + 1) * k];
                                let dst = &mut mine[(a - a_lo) * k..(a - a_lo + 1) * k];
                                for (d, s) in dst.iter_mut().zip(src) { *d += w * *s; }
                            }
                        });
                    }));
                    Ok(())
                },
            ).collect::<Result<Vec<()>, String>>().map(|_| ())
        }))
    } else {
        // Owner-computes. Batches of chunks in order; each batch bucketed in
        // parallel over chunks, then multiplied in parallel over blocks of
        // the output, each block walking the batch's chunks in order. A
        // block's output rows are the kept rows of its stored range, so the
        // output splits into one (ragged) slice per block.
        let b_prefix = match layout { Layout::Csc => &row_prefix, Layout::Csr => &col_prefix };
        let mut blocks: Vec<(usize, &mut [f64])> = Vec::with_capacity(n_blocks);
        let mut rest: &mut [f64] = &mut out[..];
        for bi in 0..n_blocks {
            let (lo, hi) = (b_prefix[bi * brows], b_prefix[((bi + 1) * brows).min(n_blocked)]);
            let (mine, tail) = std::mem::take(&mut rest).split_at_mut((hi - lo) * k);
            blocks.push((lo, mine));
            rest = tail;
        }
        let budget = if batch_bytes > 0 { batch_bytes } else { BATCH_BYTES };
        let mut batches: Vec<&[ChunkMeta]> = Vec::new();
        let mut i = 0;
        while i < chunks.len() {
            let mut j = i;
            let mut bytes = 0usize;
            while j < chunks.len() {
                let more = chunks[j].nnz.max(0) as usize * ENTRY_BYTES;
                let min_par = j - i < threads;
                if j > i && !min_par && bytes + more > budget { break; }
                bytes += more;
                j += 1;
            }
            batches.push(&chunks[i..j]);
            i = j;
        }
        // The batch's bucketed chunks live across batches: cleared and
        // refilled, never reallocated.
        let widest = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let mut bucketed: Vec<Bucketed> = (0..widest).map(|_| Bucketed::default()).collect();
        py.detach(|| pool.install(|| {
            for batch in batches {
                // One read transaction per task, not per chunk: twenty
                // threads each opening a transaction per chunk at the same
                // instant serialise on the store's locks (measured: the
                // fetch alone cost ten times what it costs when threads are
                // out of step). A task takes a contiguous run of chunks.
                let slots = &mut bucketed[..batch.len()];
                let per_task = batch.len().div_ceil(threads).max(1);
                batch.par_chunks(per_task).zip(slots.par_chunks_mut(per_task)).map_init(
                    || open_ro(&path, snapshot).map_err(|e| e.to_string()),
                    |conn, (metas, slots)| -> Result<(), String> {
                        let conn = conn.as_ref().map_err(|e| e.clone())?;
                        conn.execute_batch("BEGIN").map_err(|e| e.to_string())?;
                        let r: Result<(), String> = (|| {
                            for (meta, slot) in metas.iter().zip(slots.iter_mut()) {
                                let chunk = prof.time(&prof.read_ns, || read_chunk(conn, &matrix, meta, &dtype, layout, &prof))?;
                                SCRATCH.with(|sc| prof.time(&prof.bucket_ns,
                                    || bucket(&chunk, brows, n_blocks, &mut sc.borrow_mut(), slot, kept_ref)));
                            }
                            Ok(())
                        })();
                        let _ = conn.execute_batch("COMMIT");
                        r
                    },
                ).collect::<Result<Vec<()>, String>>()?;
                let filled: &[Bucketed] = slots;
                blocks.par_iter_mut().enumerate().for_each(|(bi, (base, block))| {
                    prof.time(&prof.multiply_ns, || {
                        for bk in filled {
                            for e in bk.offsets[bi]..bk.offsets[bi + 1] {
                                let Some((a, b, r, j)) = map(bk.own[e], bk.other[e]) else { continue };
                                let w = weigh(bk.f32_data, bk.val[e], tfidf.as_ref(), r, j);
                                let src = &dense_s[a * k..(a + 1) * k];
                                let dst = &mut block[(b - *base) * k..(b - *base + 1) * k];
                                for (d, s) in dst.iter_mut().zip(src) { *d += w * *s; }
                            }
                        }
                    });
                });
            }
            Ok(())
        }))
    };
    result.map_err(PyValueError::new_err)?;
    prof.report(&format!("{} {}", mode, if owned { "owned" } else { "owner-computes" }),
                nnz_total, t_wall.elapsed().as_secs_f64(), threads);
    Ok(numpy::ndarray::Array2::from_shape_vec((out_rows, k), out).unwrap().into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_rows_fits_the_cache_and_spreads_small_matrices() {
        // 256 KB over forty float64 columns: 819 rows of the operand.
        assert_eq!(block_rows(1_000_000, 40, 20, 0), 819);
        // Wider blocks shrink it.
        assert_eq!(block_rows(1_000_000, 256, 20, 0), 128);
        // A small matrix gets at least four blocks per thread.
        assert_eq!(block_rows(4_854, 40, 20, 0), 61);
        assert_eq!(block_rows(4_854, 40, 1, 0), 819);
        // Never zero, and an override wins.
        assert_eq!(block_rows(1, 40, 20, 0), 1);
        assert_eq!(block_rows(1_000_000, 40, 20, 7), 7);
    }

    #[test]
    fn bucketing_is_stable_and_keeps_global_indices() {
        // One column chunk of two columns over five rows, canonical order.
        // Column 0 has rows 0,2,4; column 1 has rows 1,2.
        let chunk = Chunk {
            start: 0,
            data: Data::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            indices: vec![0, 2, 4, 1, 2],
            indptr: vec![0, 3, 5],
        };
        let (mut sc, mut bk) = (Scratch::default(), Bucketed::default());
        bucket(&chunk, 2, 3, &mut sc, &mut bk, None);
        // Blocks of two rows: {0,1}, {2,3}, {4}.
        assert_eq!(bk.offsets, vec![0, 2, 4, 5]);
        // Within a block the stored (column-major) order is kept.
        assert_eq!(bk.other, vec![0, 1, 2, 2, 4]);
        assert_eq!(bk.own, vec![0, 1, 0, 1, 0]);
        assert_eq!(bk.val, vec![1.0, 4.0, 2.0, 5.0, 3.0]);
        assert!(bk.f32_data);
        // Reused buffers give the same answer.
        bucket(&chunk, 5, 1, &mut sc, &mut bk, None);
        assert_eq!(bk.offsets, vec![0, 5]);
        assert_eq!(bk.other, vec![0, 2, 4, 1, 2]);
    }

    #[test]
    fn size_aware_decoding_matches_the_general_decoder() {
        // Every codec cytome writes: zstd frames with the size recorded, lz4
        // with its four-byte size prefix (under its own label and under the
        // legacy 'zstd' label), zlib under 'zstd', and uncompressed bytes.
        let vals: Vec<i32> = (0..10_000).map(|i| (i * 7919) % 100_003 - 50_000).collect();
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let zstd_blob = zstd::bulk::compress(&raw, 3).unwrap();
        let lz4_blob = lz4_flex::block::compress_prepend_size(&raw);
        let zlib_blob = {
            use std::io::Write;
            let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            e.write_all(&raw).unwrap(); e.finish().unwrap()
        };
        for (blob, comp) in [(&zstd_blob, "zstd"), (&lz4_blob, "lz4"), (&lz4_blob, "zstd"),
                             (&zlib_blob, "zstd"), (&raw, "none")] {
            assert_eq!(decode_into::<i32>(blob, comp), vals, "codec {}", comp);
        }
        let f: Vec<f32> = vals.iter().map(|&v| v as f32 * 0.5).collect();
        let fraw: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        let got = decode_into::<f32>(&zstd::bulk::compress(&fraw, 3).unwrap(), "zstd");
        assert!(got.iter().zip(&f).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn bucketing_skips_unkept_outer_indices_and_keeps_order() {
        let chunk = Chunk {
            start: 0,
            data: Data::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            indices: vec![0, 2, 4, 1, 2],
            indptr: vec![0, 3, 5],
        };
        let (mut sc, mut bk) = (Scratch::default(), Bucketed::default());
        // Column 0 dropped: only column 1's entries, rows 1 and 2, in order.
        bucket(&chunk, 2, 3, &mut sc, &mut bk, Some(&[false, true]));
        assert_eq!(bk.offsets, vec![0, 1, 2, 2]);
        assert_eq!(bk.other, vec![1, 2]);
        assert_eq!(bk.own, vec![1, 1]);
        assert_eq!(bk.val, vec![4.0, 5.0]);
    }
}
