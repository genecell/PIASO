/// Delta/length encoding and LZ4 compression/decompression for fragment blobs.
///
/// Encoding=1 format (matching Python cytome):
///   starts_blob = lz4(delta-encoded int32): [s0, s1-s0, s2-s1, ...]
///   ends_blob   = lz4(length-encoded int32): [e0-s0, e1-s1, ...]
///   cell_idx_blob = lz4(raw int32): [c0, c1, c2, ...]

/// LZ4 decompress with 4-byte LE size header (lz4_flex::compress_prepend_size format).
pub fn lz4_decompress(data: &[u8]) -> Vec<u8> {
    lz4_flex::block::decompress_size_prepended(data)
        .expect("LZ4 decompression failed")
}

/// Decompress blob based on compression method string.
pub fn decompress_blob(data: &[u8], compression: &str) -> Vec<u8> {
    match compression {
        "lz4" => lz4_decompress(data),
        "none" | "" => data.to_vec(),
        _ => panic!("Unsupported compression: {} (piaso-atac supports lz4 only)", compression),
    }
}

/// Interpret raw bytes as i32 slice (little-endian, zero-copy on LE platforms).
pub fn bytes_to_i32(data: &[u8]) -> Vec<i32> {
    assert!(data.len() % 4 == 0, "Data length not multiple of 4");
    let n = data.len() / 4;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
        result.push(i32::from_le_bytes(bytes));
    }
    result
}

/// Decode delta-encoded starts: cumsum of deltas.
pub fn decode_starts_delta(blob: &[u8], compression: &str) -> Vec<i32> {
    let raw = decompress_blob(blob, compression);
    let deltas = bytes_to_i32(&raw);
    let mut starts = Vec::with_capacity(deltas.len());
    let mut acc: i64 = 0;
    for &d in &deltas {
        acc += d as i64;
        starts.push(acc as i32);
    }
    starts
}

/// Decode length-encoded ends: starts + lengths.
pub fn decode_ends_length(blob: &[u8], compression: &str, starts: &[i32]) -> Vec<i32> {
    let raw = decompress_blob(blob, compression);
    let lengths = bytes_to_i32(&raw);
    starts.iter().zip(lengths.iter()).map(|(&s, &l)| s + l).collect()
}

/// Decode raw cell indices (no delta/length encoding).
pub fn decode_cell_indices(blob: &[u8], compression: &str) -> Vec<i32> {
    let raw = decompress_blob(blob, compression);
    bytes_to_i32(&raw)
}

/// Decode starts based on encoding mode.
pub fn decode_starts(blob: &[u8], compression: &str, encoding: i32) -> Vec<i32> {
    if encoding == 1 {
        decode_starts_delta(blob, compression)
    } else {
        let raw = decompress_blob(blob, compression);
        bytes_to_i32(&raw)
    }
}

/// Decode ends based on encoding mode.
pub fn decode_ends(blob: &[u8], compression: &str, starts: &[i32], encoding: i32) -> Vec<i32> {
    if encoding == 1 {
        decode_ends_length(blob, compression, starts)
    } else {
        let raw = decompress_blob(blob, compression);
        bytes_to_i32(&raw)
    }
}

// --- LZ4 compression for writing CSR to cytome ---

/// LZ4 compress raw bytes with 4-byte LE size header (Python-compatible).
pub fn lz4_compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::block::compress_prepend_size(data)
}

/// LZ4 compress f32 slice.
pub fn lz4_compress_f32(data: &[f32]) -> Vec<u8> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    lz4_compress(bytes)
}

/// LZ4 compress i32 slice.
pub fn lz4_compress_i32(data: &[i32]) -> Vec<u8> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    lz4_compress(bytes)
}

/// LZ4 compress i64 slice.
pub fn lz4_compress_i64(data: &[i64]) -> Vec<u8> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
    };
    lz4_compress(bytes)
}

// ── Fragment-blob ENCODE side (writer). Mirrors the decode_* functions above;
//    used by the cytome-import-fragments binary. Co-located here so the
//    encoding=1 format contract lives in ONE module (encode + decode together).
//    encoding=1:
//      starts_blob   = lz4(delta-encoded int32): [s0, s1-s0, s2-s1, ...]
//      ends_blob     = lz4(length-encoded int32): [e0-s0, e1-s1, ...]
//      cell_idx_blob = lz4(raw int32)

/// Convert an i32 slice to little-endian bytes.
fn to_le_bytes_i32(vals: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Delta-encode sorted start positions -> raw bytes (inverse of decode_starts_delta).
pub fn encode_starts_delta(starts: &[i32]) -> Vec<u8> {
    let n = starts.len();
    let mut deltas = Vec::with_capacity(n);
    if n > 0 {
        deltas.push(starts[0]);
        for i in 1..n {
            deltas.push(starts[i] - starts[i - 1]);
        }
    }
    to_le_bytes_i32(&deltas)
}

/// Length-encode ends (end - start) -> raw bytes (inverse of decode_ends_length).
pub fn encode_ends_length(starts: &[i32], ends: &[i32]) -> Vec<u8> {
    let lengths: Vec<i32> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&s, &e)| e - s)
        .collect();
    to_le_bytes_i32(&lengths)
}

/// Raw int32 bytes for cell indices (no delta encoding).
pub fn encode_cell_indices(cells: &[i32]) -> Vec<u8> {
    to_le_bytes_i32(cells)
}
