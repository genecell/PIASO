/// Read fragment chunks from cytome SQLite database.
///
/// Decodes delta/length-encoded, LZ4-compressed fragment blobs into
/// (starts, ends, cell_indices) arrays.

use rusqlite::Connection;
use crate::encoder;

/// On-disk cytome format major version this build of piaso-rust supports.
/// Mirrors cytome's `_manifest.format_version` (see cytome
/// `cytome/io/sqlite_engine.py` — the source of truth for the schema).
pub const SUPPORTED_FORMAT_MAJOR: u32 = 1;

/// Fail loudly if the cytome's `_manifest.format_version` major is one this
/// build does not understand — so neither the importer writes fragment chunks
/// into, nor the reader mis-reads, a future on-disk format. A missing or
/// unparseable version is allowed (legacy / freshly-created files).
pub fn assert_compatible_format(conn: &Connection) {
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM _manifest WHERE key = 'format_version'",
            [],
            |r| r.get(0),
        )
        .ok();
    if let Some(raw) = raw {
        // Stored as a JSON string, e.g. "\"1.0.0\"" — strip quotes/whitespace.
        let cleaned = raw.trim().trim_matches('"');
        if let Some(major) = cleaned
            .split('.')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
        {
            if major != SUPPORTED_FORMAT_MAJOR {
                panic!(
                    "Incompatible cytome format_version '{}' (major {}); this build of \
                     piaso-rust supports cytome format major {}. Upgrade piaso-tools, or \
                     re-create the cytome with a compatible writer.",
                    cleaned, major, SUPPORTED_FORMAT_MAJOR
                );
            }
        }
    }
}

/// A block of decoded fragments for one chromosome.
pub struct FragmentBlock {
    pub starts: Vec<i32>,
    pub ends: Vec<i32>,
    pub cell_indices: Vec<i32>,
}

/// Load cell_idx → cluster_int mapping from cytome cells table.
/// Returns array where arr[cell_idx] = cluster_int, or -1 if unmatched.
pub fn build_cell_to_cluster_int(
    conn: &Connection,
    barcode_to_cluster: &std::collections::HashMap<String, String>,
    cluster_to_int: &std::collections::HashMap<String, i32>,
) -> Vec<i32> {
    let mut stmt = conn.prepare("SELECT cell_idx, barcode FROM cells").unwrap();
    let rows: Vec<(i32, String)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?))
    }).unwrap().filter_map(|r| r.ok()).collect();

    let max_idx = rows.iter().map(|(idx, _)| *idx).max().unwrap_or(0) as usize;
    let mut mapping = vec![-1i32; max_idx + 1];

    for (idx, barcode) in &rows {
        if let Some(cluster) = barcode_to_cluster.get(barcode) {
            if let Some(&cint) = cluster_to_int.get(cluster) {
                mapping[*idx as usize] = cint;
            }
        }
    }
    mapping
}

/// Get chromosomes and their fragment counts from cytome, ordered largest-first.
pub fn get_chrom_fragment_counts(conn: &Connection) -> Vec<(String, i64)> {
    let mut stmt = conn.prepare(
        "SELECT chrom, SUM(n_fragments) FROM fragment_chunks GROUP BY chrom ORDER BY SUM(n_fragments) DESC"
    ).unwrap();
    stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }).unwrap().filter_map(|r| r.ok()).collect()
}

/// Iterate fragment blocks for a chromosome, yielding blocks of up to `block_size` fragments.
pub fn iter_chromosome_blocks(
    conn: &Connection,
    chrom: &str,
    block_size: usize,
) -> Vec<FragmentBlock> {
    let mut stmt = conn.prepare(
        "SELECT starts_blob, ends_blob, cell_idx_blob, compression, COALESCE(encoding, 0) \
         FROM fragment_chunks WHERE chrom = ? ORDER BY chunk_idx"
    ).unwrap();

    let mut acc_starts: Vec<i32> = Vec::new();
    let mut acc_ends: Vec<i32> = Vec::new();
    let mut acc_cells: Vec<i32> = Vec::new();
    let mut blocks = Vec::new();

    let rows: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, String, i32)> = stmt.query_map(
        [chrom], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i32>(4)?,
            ))
        }
    ).unwrap().filter_map(|r| r.ok()).collect();

    for (starts_blob, ends_blob, cell_blob, compression, encoding) in rows {
        let starts = encoder::decode_starts(&starts_blob, &compression, encoding);
        let ends = encoder::decode_ends(&ends_blob, &compression, &starts, encoding);
        let cells = encoder::decode_cell_indices(&cell_blob, &compression);

        acc_starts.extend_from_slice(&starts);
        acc_ends.extend_from_slice(&ends);
        acc_cells.extend_from_slice(&cells);

        while acc_starts.len() >= block_size {
            let s: Vec<i32> = acc_starts.drain(..block_size).collect();
            let e: Vec<i32> = acc_ends.drain(..block_size).collect();
            let c: Vec<i32> = acc_cells.drain(..block_size).collect();
            blocks.push(ensure_sorted(FragmentBlock { starts: s, ends: e, cell_indices: c }));
        }
    }

    // Remainder
    if !acc_starts.is_empty() {
        blocks.push(ensure_sorted(FragmentBlock {
            starts: acc_starts,
            ends: acc_ends,
            cell_indices: acc_cells,
        }));
    }

    blocks
}

/// Stream fragment blocks for a chromosome, calling a closure on each block.
/// True cursor-based streaming: reads one SQLite row at a time, frees compressed
/// blobs immediately after decoding. Only one decompressed sub-chunk + accumulator
/// in memory at any time.
pub fn stream_chromosome_blocks<F>(
    conn: &Connection,
    chrom: &str,
    block_size: usize,
    mut callback: F,
) where F: FnMut(&FragmentBlock) {
    stream_chromosome_blocks_filtered(conn, chrom, block_size, 0, i32::MAX, &mut callback)
}

/// Like `stream_chromosome_blocks` but keeps only fragments whose length
/// (`end - start`) lies in `[frag_len_min, frag_len_max]`. Used for
/// nucleosome-free (NFR) peak quantification / calling — the filter is applied
/// per decoded chunk before accumulation, so memory stays bounded and no extra
/// pass over the fragments is needed. `frag_len_min=0, frag_len_max=i32::MAX`
/// is the no-filter fast path.
pub fn stream_chromosome_blocks_filtered<F>(
    conn: &Connection,
    chrom: &str,
    block_size: usize,
    frag_len_min: i32,
    frag_len_max: i32,
    callback: &mut F,
) where F: FnMut(&FragmentBlock) {
    let filtering = frag_len_min > 0 || frag_len_max < i32::MAX;
    let mut stmt = conn.prepare(
        "SELECT starts_blob, ends_blob, cell_idx_blob, compression, COALESCE(encoding, 0) \
         FROM fragment_chunks WHERE chrom = ? ORDER BY chunk_idx"
    ).unwrap();

    let mut acc_starts: Vec<i32> = Vec::new();
    let mut acc_ends: Vec<i32> = Vec::new();
    let mut acc_cells: Vec<i32> = Vec::new();

    // True cursor iteration — no .collect(). Each row's compressed blobs are
    // read, decoded, and freed before the next row is fetched from SQLite.
    let mut rows = stmt.query(rusqlite::params![chrom]).unwrap();
    while let Some(row) = rows.next().unwrap() {
        let starts_blob: Vec<u8> = row.get(0).unwrap();
        let ends_blob: Vec<u8> = row.get(1).unwrap();
        let cell_blob: Vec<u8> = row.get(2).unwrap();
        let compression: String = row.get(3).unwrap();
        let encoding: i32 = row.get(4).unwrap();

        let starts = encoder::decode_starts(&starts_blob, &compression, encoding);
        let ends = encoder::decode_ends(&ends_blob, &compression, &starts, encoding);
        let cells = encoder::decode_cell_indices(&cell_blob, &compression);

        if filtering {
            // Keep only NFR-window fragments; preserves start-sorted order.
            for i in 0..starts.len() {
                let len = ends[i] - starts[i];
                if len >= frag_len_min && len <= frag_len_max {
                    acc_starts.push(starts[i]);
                    acc_ends.push(ends[i]);
                    acc_cells.push(cells[i]);
                }
            }
        } else {
            acc_starts.extend_from_slice(&starts);
            acc_ends.extend_from_slice(&ends);
            acc_cells.extend_from_slice(&cells);
        }

        while acc_starts.len() >= block_size {
            let s: Vec<i32> = acc_starts.drain(..block_size).collect();
            let e: Vec<i32> = acc_ends.drain(..block_size).collect();
            let c: Vec<i32> = acc_cells.drain(..block_size).collect();
            let block = ensure_sorted(FragmentBlock { starts: s, ends: e, cell_indices: c });
            callback(&block);
        }
    }

    if !acc_starts.is_empty() {
        let block = ensure_sorted(FragmentBlock {
            starts: acc_starts,
            ends: acc_ends,
            cell_indices: acc_cells,
        });
        callback(&block);
    }
}

/// Ensure fragments are sorted by start position.
fn ensure_sorted(mut block: FragmentBlock) -> FragmentBlock {
    let n = block.starts.len();
    if n <= 1 {
        return block;
    }
    // Check if already sorted
    let sorted = block.starts.windows(2).all(|w| w[0] <= w[1]);
    if sorted {
        return block;
    }
    // Sort by start position
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by_key(|&i| block.starts[i]);
    let s: Vec<i32> = indices.iter().map(|&i| block.starts[i]).collect();
    let e: Vec<i32> = indices.iter().map(|&i| block.ends[i]).collect();
    let c: Vec<i32> = indices.iter().map(|&i| block.cell_indices[i]).collect();
    block.starts = s;
    block.ends = e;
    block.cell_indices = c;
    block
}
