//! Opening a cytome for reading, with or without file locks.
//!
//! Shared by every Rust reader of a store (the SVD engine, the peak caller).
//! A `snapshot` connection opens the file as `immutable`: SQLite takes no
//! locks and does no change detection, so a network file system never
//! purges its cache of the store on our account. An immutable reader cannot
//! see the write-ahead log, so the caller must have checkpointed it to
//! empty first -- that is the Python side's job (`snapshot_store`) and its
//! guard; the readers here trust the flag they are given.

use rusqlite::{Connection, OpenFlags};

/// Percent-encode a path for a SQLite `file:` URI. Keeps the RFC 3986
/// unreserved set, so `?`, `#`, `%`, spaces and non-ASCII names survive.
/// `/` is kept, since it is the path separator the URI expects.
pub fn uri_encode(path: &str) -> String {
    let mut out = String::with_capacity(path.len() + 8);
    for &b in path.as_bytes() {
        let keep = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'/');
        if keep { out.push(b as char) } else { out.push_str(&format!("%{:02X}", b)) }
    }
    out
}

/// A read-only connection; see the module note for what `snapshot` means.
pub fn open_ro(path: &str, snapshot: bool) -> Result<Connection, String> {
    let conn = if snapshot {
        Connection::open_with_flags(
            format!("file:{}?immutable=1", uri_encode(path)),
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_URI,
        )
    } else {
        Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_URI,
        )
    }
    .map_err(|e| format!("open {}: {}", path, e))?;
    conn.busy_timeout(std::time::Duration::from_secs(120)).map_err(|e| e.to_string())?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_encoding_keeps_slashes_and_escapes_the_rest() {
        assert_eq!(uri_encode("/data/a b/x%y?z#w.cytome"), "/data/a%20b/x%25y%3Fz%23w.cytome");
        assert_eq!(uri_encode("/plain/path_1.cytome"), "/plain/path_1.cytome");
        assert_eq!(uri_encode("/dé/ü.cytome"), "/d%C3%A9/%C3%BC.cytome");
    }

    #[test]
    fn a_snapshot_connection_reads_what_is_checkpointed_and_takes_no_lock() {
        let dir = std::env::temp_dir().join(format!("piaso_store_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.cytome");
        let p = path.to_str().unwrap();
        {
            let w = Connection::open(p).unwrap();
            w.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE t(x); INSERT INTO t VALUES (1),(2);
                             PRAGMA wal_checkpoint(TRUNCATE);").unwrap();
        }
        for snap in [false, true] {
            let c = open_ro(p, snap).unwrap();
            let n: i64 = c.query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0)).unwrap();
            assert_eq!(n, 2);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
