"""Which storage layout the SVD engine takes, and what that costs.

On a given layout one product is owned (each chunk writes its own output
rows) and the other is scattered into a buffer every chunk touches: row-major
scatters the transpose product into ``features x block``, column-major
scatters the forward product into ``cells x block``. So the layout to use is
the one whose scatter buffer is smaller -- column-major when there are fewer
cells than features, row-major otherwise. Measured on one matrix stored both
ways, the wrong choice is 8x on the transpose product.

The old rule was "row-major whenever row chunks exist", which picked the slow
layout for every peak matrix that also carried a feature index.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from conftest import make_column_major

pytest.importorskip("piaso._piaso")


def _store(path, X, modality="ATAC"):
    import cytome

    n_cells, n_feat = X.shape
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells), "barcode": [f"B{i}" for i in range(n_cells)]}))
    if modality == "ATAC":
        ds.set_entity("peaks", pd.DataFrame({
            "peak_idx": np.arange(n_feat),
            "peak_id": [f"chr1:{i*1000}-{i*1000+500}" for i in range(n_feat)],
            "chr": ["chr1"] * n_feat, "start": [i * 1000 for i in range(n_feat)],
            "end_": [i * 1000 + 500 for i in range(n_feat)]}))
    else:
        ds.set_entity("genes", pd.DataFrame({
            "gene_idx": np.arange(n_feat), "gene_id": [f"G{i}" for i in range(n_feat)],
            "symbol": [f"Gene{i}" for i in range(n_feat)]}))
    ds.add_matrix(f"{modality}_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


def _add_column_chunks(ds, X, matrix_name, chunk_cols=None):
    """Add CSC chunks WITHOUT removing the row chunks: a store with both."""
    import numpy as _np
    import scipy.sparse as _sp

    csc = _sp.csc_matrix(X); n_cols = X.shape[1]
    width = int(chunk_cols or max(1, n_cols // 3 or 1))
    declared = ds._conn.execute("SELECT dtype FROM matrix_meta WHERE matrix_name=?",
                                (matrix_name,)).fetchone()[0]
    dtype = _np.dtype(declared)
    ds._conn.execute("DELETE FROM matrix_csc_chunks WHERE matrix_name=?", (matrix_name,))
    n = 0
    for k, start in enumerate(range(0, n_cols, width)):
        end = min(start + width, n_cols); sub = csc[:, start:end].tocsc()
        ds._conn.execute(
            "INSERT INTO matrix_csc_chunks (matrix_name, chunk_idx, col_start, col_end,"
            " n_nonzero, data_blob, indices_blob, indptr_blob, dtype, compression) "
            "VALUES (?,?,?,?,?,?,?,?,?,'none')",
            (matrix_name, k, start, end, int(sub.nnz), sub.data.astype(dtype).tobytes(),
             sub.indices.astype(_np.int32).tobytes(), sub.indptr.astype(_np.int32).tobytes(),
             str(dtype)))
        n = k + 1
    ds._conn.execute("UPDATE matrix_meta SET has_csc=1, csc_chunk_size=?, csc_n_chunks=? "
                     "WHERE matrix_name=?", (width, n, matrix_name))
    ds._conn.commit()


def _layout(ds, matrix):
    from piaso._piaso import svd_layout
    from piaso.tools._rsvd_core import snapshot_store
    # svd_layout reads a snapshot by default (it runs in the process that holds
    # the store open): checkpoint first, as runSVD does.
    ok, why = snapshot_store(ds)
    assert ok, why
    return svd_layout(str(ds.path), matrix)


def test_with_both_layouts_fewer_cells_than_features_takes_column_major(tmp_path):
    X = (np.random.default_rng(0).random((30, 90)) < 0.4).astype(np.float32)
    ds = _store(tmp_path / "wide.cytome", X)
    _add_column_chunks(ds, X, "ATAC_counts")
    lay, has_csr, has_csc, n, m = _layout(ds, "ATAC_counts")
    assert (has_csr, has_csc) == (True, True)
    assert (n, m) == (30, 90)
    assert lay == "csc"
    ds.close()


def test_with_both_layouts_fewer_features_than_cells_takes_row_major(tmp_path):
    """The rule is the buffer size, not 'column-major always'.

    A gene matrix has far more cells than features; its row-major transpose
    scatters into a small features x block buffer, and column-major would
    scatter into a cells x block one that is larger. The first version of
    this rule would have made the engine 8x slower on exactly those.
    """
    X = (np.random.default_rng(1).random((120, 25)) < 0.4).astype(np.float32)
    ds = _store(tmp_path / "tall.cytome", X, modality="RNA")
    _add_column_chunks(ds, X, "RNA_counts")
    lay, has_csr, has_csc, n, m = _layout(ds, "RNA_counts")
    assert (has_csr, has_csc) == (True, True)
    assert lay == "csr"
    ds.close()


def test_with_one_layout_that_layout_is_used(tmp_path):
    X = (np.random.default_rng(2).random((30, 90)) < 0.4).astype(np.float32)
    ds = _store(tmp_path / "rowonly.cytome", X)
    assert _layout(ds, "ATAC_counts")[:3] == ("csr", True, False)
    ds.close()
    ds = _store(tmp_path / "colonly.cytome", X)
    make_column_major(ds, X, "ATAC_counts")
    ds.close()
    import cytome
    ds = cytome.open(str(tmp_path / "colonly.cytome"))
    assert _layout(ds, "ATAC_counts")[:3] == ("csc", False, True)
    ds.close()


def _ops(ds, matrix, n, m, threads, layout="auto"):
    from piaso.tools._rsvd_core import EngineOps
    return EngineOps(str(ds.path), matrix, n, m, n_threads=threads)


@pytest.mark.parametrize("orientation", ["wide", "tall"])
def test_both_products_agree_across_layouts_and_thread_counts(tmp_path, orientation):
    """The kernel rewrite must not move the numbers.

    Across layouts the products agree to float32 rounding; across thread
    counts on one layout they are identical, because the fold order of the
    scattered product depends only on the store's chunk count.
    """
    rng = np.random.default_rng(3)
    shape = (40, 300) if orientation == "wide" else (300, 40)
    X = ((rng.random(shape) < 0.3).astype(np.float32) * rng.integers(1, 5, shape)).astype(np.float32)
    mod = "ATAC" if orientation == "wide" else "RNA"
    n, m = shape
    ds = _store(tmp_path / "both.cytome", X, modality=mod)
    _add_column_chunks(ds, X, f"{mod}_counts", chunk_cols=37)
    B = rng.standard_normal((m, 7)); Y = rng.standard_normal((n, 7))
    ref_xb, ref_xty = X @ B, X.T @ Y
    out = {}
    for lay in ("csr", "csc"):
        for threads in (1, 4):
            from piaso import _piaso
            remap = np.arange(m, dtype=np.int32)
            xb = _piaso.svd_pass(str(ds.path), f"{mod}_counts", "xb",
                                 np.ascontiguousarray(B), remap, m, n_threads=threads, layout=lay)
            xty = _piaso.svd_pass(str(ds.path), f"{mod}_counts", "xty",
                                  np.ascontiguousarray(Y), remap, m, n_threads=threads, layout=lay)
            out[(lay, threads)] = (np.asarray(xb), np.asarray(xty))
            np.testing.assert_allclose(xb, ref_xb, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(xty, ref_xty, rtol=1e-5, atol=1e-5)
    for lay in ("csr", "csc"):
        np.testing.assert_array_equal(out[(lay, 1)][0], out[(lay, 4)][0])
        np.testing.assert_array_equal(out[(lay, 1)][1], out[(lay, 4)][1])
    ds.close()


def test_tfidf_products_still_match_the_python_path(tmp_path):
    """The TF-IDF transform must still match the Python path.

    Against numpy the engine has never been bit-identical: numpy's SIMD
    float32 ``log1p`` and libm's differ by a few ulp on a fraction of inputs
    (the existing engine tests document this). So the standard here is the
    same as theirs -- agreement to float32 rounding, with most entries
    bit-identical.
    """
    rng = np.random.default_rng(4)
    n, m = 50, 200
    X = ((rng.random((n, m)) < 0.4).astype(np.float32) * rng.integers(1, 4, (n, m))).astype(np.float32)
    ds = _store(tmp_path / "tf.cytome", X)
    depth = np.asarray(X.sum(axis=1), dtype=np.float64)
    idf = np.log1p(n / np.maximum(np.asarray((X > 0).sum(axis=0), dtype=np.float64), 1.0)).astype(np.float32)
    from piaso.tools._runTFIDF import _normalize_chunk_tfidf
    T = _normalize_chunk_tfidf(sp.csr_matrix(X), depth, idf, 1e4)
    B = rng.standard_normal((m, 5))
    ref = np.asarray(T.astype(np.float64) @ B)
    from piaso import _piaso
    got = np.asarray(_piaso.svd_pass(str(ds.path), "ATAC_counts", "xb", np.ascontiguousarray(B),
                                     np.arange(m, dtype=np.int32), m, cell_depth=depth, idf=idf,
                                     scale=1e4, n_threads=1, layout="csr"))
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    assert int(np.sum(got == ref)) >= 0.5 * got.size
    ds.close()


def test_runSVD_names_the_layout_and_advises_on_a_row_major_peak_matrix(tmp_path, capsys):
    import piaso
    X = (np.random.default_rng(5).random((30, 90)) < 0.4).astype(np.float32)
    ds = _store(tmp_path / "advise.cytome", X)
    ds.peaks["highly_variable"] = np.ones(90, dtype=bool); ds.flush()
    piaso.tl.runSVD(ds, modality="ATAC", measurement="counts", streaming=True,
                    auto_tfidf=True, n_components=3, n_iter=2, random_state=0,
                    verbosity=1, key_added="X_l")
    out = capsys.readouterr().out
    assert "row-major store" in out
    assert "column-major store" in out and "layout='csc'" in out
    ds.close()


def _sequential_reference(X, dense, layout, mode):
    """The products summed in the one order the engine promises.

    Chunk order, then stored order within a chunk. Chunks are contiguous
    ranges of columns (column-major) or rows (row-major), so on a canonical
    matrix that is simply the column-major or row-major walk of the whole
    matrix, each output element accumulated one term at a time in float64.
    """
    n, m = X.shape
    k = dense.shape[1]
    if layout == "csc":
        M = sp.csc_matrix(X); M.sort_indices()
        out = np.zeros((m if mode == "xty" else n, k))
        for j in range(m):
            for p in range(M.indptr[j], M.indptr[j + 1]):
                r = M.indices[p]; w = float(M.data[p])
                if mode == "xty": out[j] += w * dense[r]
                else: out[r] += w * dense[j]
    else:
        M = sp.csr_matrix(X); M.sort_indices()
        out = np.zeros((m if mode == "xty" else n, k))
        for r in range(n):
            for p in range(M.indptr[r], M.indptr[r + 1]):
                j = M.indices[p]; w = float(M.data[p])
                if mode == "xty": out[j] += w * dense[r]
                else: out[r] += w * dense[j]
    return out


@pytest.mark.parametrize("layout", ["csr", "csc"])
@pytest.mark.parametrize("mode", ["xb", "xty"])
def test_every_product_is_the_sequential_sum_in_stored_order_to_the_bit(tmp_path, layout, mode):
    """Bit-identity, not closeness, whatever the threads, blocks or batches.

    The engine blocks each chunk on the dense operand and, for the product a
    chunk does not own, has threads own blocks of the output. Neither may
    leave a trace in the numbers: every output element is one float64 sum in
    chunk order then stored order, and a Python loop in that order must
    reproduce it exactly. Thread count, block size and batch budget are
    speed knobs only.
    """
    from piaso import _piaso

    rng = np.random.default_rng(21)
    n, m, k = 45, 310, 6
    X = ((rng.random((n, m)) < 0.25).astype(np.float32) * rng.integers(1, 9, (n, m))).astype(np.float32)
    ds = _store(tmp_path / f"seq_{layout}_{mode}.cytome", X)
    _add_column_chunks(ds, X, "ATAC_counts", chunk_cols=41)
    dense = rng.standard_normal((m if mode == "xb" else n, k))
    ref = _sequential_reference(X, dense, layout, mode)
    remap = np.arange(m, dtype=np.int32)
    for threads in (1, 3):
        for block_rows in (0, 1, 7):
            for batch_bytes in (0, 1):
                got = _piaso.svd_pass(str(ds.path), "ATAC_counts", mode, np.ascontiguousarray(dense),
                                      remap, m, n_threads=threads, block_rows=block_rows,
                                      batch_bytes=batch_bytes, layout=layout)
                np.testing.assert_array_equal(np.asarray(got), ref)
    ds.close()


def test_row_and_column_selection_apply_before_blocking(tmp_path):
    """Excluded rows and columns vanish from both products, in every layout.

    A kept-row map with holes and a column mask with holes, against the
    sequential reference on the sub-matrix.
    """
    from piaso import _piaso

    rng = np.random.default_rng(22)
    n, m, k = 30, 120, 4
    X = ((rng.random((n, m)) < 0.3).astype(np.float32) * rng.integers(1, 5, (n, m))).astype(np.float32)
    ds = _store(tmp_path / "sel.cytome", X)
    _add_column_chunks(ds, X, "ATAC_counts", chunk_cols=23)
    col_mask = rng.random(m) < 0.6
    row_keep = np.flatnonzero(rng.random(n) < 0.7)
    remap = np.full(m, -1, dtype=np.int32); remap[col_mask] = np.arange(col_mask.sum())
    row_map = np.full(n, -1, dtype=np.int32); row_map[row_keep] = np.arange(row_keep.size)
    Xs = X[row_keep][:, col_mask]
    for layout in ("csr", "csc"):
        for mode in ("xb", "xty"):
            dense = rng.standard_normal((Xs.shape[1] if mode == "xb" else Xs.shape[0], k))
            ref = _sequential_reference(Xs, dense, layout, mode)
            got = _piaso.svd_pass(str(ds.path), "ATAC_counts", mode, np.ascontiguousarray(dense),
                                  remap, int(col_mask.sum()), row_map=row_map,
                                  n_out_rows=int(row_keep.size), n_threads=2, layout=layout)
            np.testing.assert_array_equal(np.asarray(got), ref)
    ds.close()


def test_integer_stores_of_every_width_are_read(tmp_path):
    """Counts stored as uint16 (what the RNA importer writes), uint8, int16 or
    int32 must all be read; the engine used to raise on unsigned types."""
    from piaso import _piaso

    rng = np.random.default_rng(31)
    n, m, k = 20, 50, 3
    X = ((rng.random((n, m)) < 0.4) * rng.integers(1, 300, (n, m))).astype(np.int64)
    B = rng.standard_normal((m, k))
    remap = np.arange(m, dtype=np.int32)
    for dtype in (np.uint16, np.uint8, np.int16, np.int32, np.uint32):
        ds = _store(tmp_path / f"i_{np.dtype(dtype).name}.cytome", X.astype(dtype))
        declared = ds._conn.execute("SELECT dtype FROM matrix_meta WHERE matrix_name='ATAC_counts'").fetchone()[0]
        got = _piaso.svd_pass(str(ds.path), "ATAC_counts", "xb", np.ascontiguousarray(B), remap, m, n_threads=2)
        np.testing.assert_allclose(np.asarray(got), X.astype(dtype).astype(np.float64) @ B, rtol=1e-12, atol=1e-9)
        ds.close()
        assert declared in ("uint16", "uint8", "int16", "int32", "uint32", "float32", "int64", "float64"), declared


def test_legacy_zlib_blobs_labelled_zstd_are_read(tmp_path):
    """Older stores carry zlib streams under the label ``'zstd'``; the Python
    reader trusts the magic bytes, and so must the engine."""
    import zlib
    from cytome.io.compression import decompress_blob
    from piaso import _piaso

    rng = np.random.default_rng(41)
    n, m, k = 30, 80, 4
    X = ((rng.random((n, m)) < 0.3) * rng.integers(1, 9, (n, m))).astype(np.float32)
    ds = _store(tmp_path / "zlib.cytome", X)
    rows = ds._conn.execute("SELECT chunk_idx, data_blob, indices_blob, indptr_blob, compression "
                            "FROM matrix_chunks WHERE matrix_name='ATAC_counts'").fetchall()
    for idx, d, i, p, comp in rows:
        ds._conn.execute("UPDATE matrix_chunks SET data_blob=?, indices_blob=?, indptr_blob=?, "
                         "compression='zstd' WHERE matrix_name='ATAC_counts' AND chunk_idx=?",
                         (zlib.compress(decompress_blob(d, comp)), zlib.compress(decompress_blob(i, comp)),
                          zlib.compress(decompress_blob(p, comp)), idx))
    ds._conn.commit()
    B = rng.standard_normal((m, k))
    got = _piaso.svd_pass(str(ds.path), "ATAC_counts", "xb", np.ascontiguousarray(B),
                          np.arange(m, dtype=np.int32), m, n_threads=2)
    np.testing.assert_allclose(np.asarray(got), X.astype(np.float64) @ B, rtol=1e-12, atol=1e-9)
    ds.close()


def _subspace_agreement(a, b):
    qa, _ = np.linalg.qr(a); qb, _ = np.linalg.qr(b)
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def test_runSVD_on_stored_values_takes_the_engine_and_the_fallback_accepts_lanczos(tmp_path, capsys):
    """An RNA (or any) cytome SVD with no per-chunk transform runs on the
    engine with the selected features as its column mask; and with the
    engine off, ``method='auto'`` still works, cached or not. The three
    embeddings span the same subspace.
    """
    import piaso

    rng = np.random.default_rng(42)
    n, m = 120, 300
    X = ((rng.random((n, m)) < 0.2) * rng.integers(1, 20, (n, m))).astype(np.float32)
    X[:, :40] *= 5  # a few strong genes so the leading subspace is well defined
    ds = _store(tmp_path / "rna.cytome", X, modality="RNA")
    hv = np.zeros(m, dtype=bool); hv[rng.choice(m, 150, replace=False)] = True; hv[:40] = True
    ds.genes["highly_variable"] = hv; ds.flush()
    common = dict(modality="RNA", measurement="counts", streaming=True, auto_tfidf=False,
                  n_components=4, method="auto", random_state=0, verbosity=1)
    piaso.tl.runSVD(ds, key_added="X_eng", **common)
    out = capsys.readouterr().out
    assert "SVD engine: Rust pass over RNA_counts" in out and f"{int(hv.sum())} selected columns" in out
    assert "values as stored" in out
    piaso.tl.runSVD(ds, key_added="X_fac", use_engine=False, **common)
    out = capsys.readouterr().out
    assert "SVD engine" not in out and "block Lanczos" in out
    piaso.tl.runSVD(ds, key_added="X_cached", use_engine=False, cache_chunks=True, **common)
    from piaso.tools._embedding_names import storage_name
    e = {k: np.asarray(ds.embeddings[storage_name(k, "RNA")]) for k in ("X_eng", "X_fac", "X_cached")}
    assert _subspace_agreement(e["X_eng"], e["X_fac"]) > 0.999
    np.testing.assert_allclose(e["X_fac"], e["X_cached"], rtol=1e-10, atol=1e-10)
    ds.close()


def _rechunk_rows(ds, X, matrix_name, chunk_rows):
    """Rewrite the row chunks at ``chunk_rows`` rows each, uncompressed."""
    import scipy.sparse as _sp
    csr = _sp.csr_matrix(X)
    declared = ds._conn.execute("SELECT dtype FROM matrix_meta WHERE matrix_name=?",
                                (matrix_name,)).fetchone()[0]
    dtype = np.dtype(declared)
    ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name=?", (matrix_name,))
    n = 0
    for k, start in enumerate(range(0, X.shape[0], chunk_rows)):
        end = min(start + chunk_rows, X.shape[0]); sub = csr[start:end]
        ds._conn.execute(
            "INSERT INTO matrix_chunks (matrix_name, chunk_idx, row_start, row_end, n_nonzero,"
            " data_blob, indices_blob, indptr_blob, dtype, compression) VALUES (?,?,?,?,?,?,?,?,?,'none')",
            (matrix_name, k, start, end, int(sub.nnz), sub.data.astype(dtype).tobytes(),
             sub.indices.astype(np.int32).tobytes(), sub.indptr.astype(np.int32).tobytes(), str(dtype)))
        n = k + 1
    ds._conn.execute("UPDATE matrix_meta SET n_chunks=?, chunk_size=? WHERE matrix_name=?",
                     (n, chunk_rows, matrix_name))
    ds._conn.commit()


def test_chunks_holding_nothing_selected_are_skipped_without_changing_the_result(tmp_path):
    """A cell subset that is one contiguous run (one library) and a feature
    selection confined to one column chunk leave whole chunks unread. The
    result must still be the sequential sum over the sub-matrix, to the bit,
    including the edge case where nothing at all is selected in a layout's
    outer dimension except one index."""
    from piaso import _piaso

    rng = np.random.default_rng(51)
    n, m, k = 64, 200, 5
    X = ((rng.random((n, m)) < 0.25) * rng.integers(1, 7, (n, m))).astype(np.float32)
    ds = _store(tmp_path / "skip.cytome", X)
    _rechunk_rows(ds, X, "ATAC_counts", 10)       # 7 row chunks
    _add_column_chunks(ds, X, "ATAC_counts", chunk_cols=25)   # 8 column chunks
    assert ds._conn.execute("SELECT count(*) FROM matrix_chunks").fetchone()[0] == 7
    cases = {
        "one library of cells": (np.arange(m), np.arange(20, 37)),
        "features inside one column chunk": (np.arange(50, 70), np.arange(n)),
        "a single cell and a single feature": (np.array([123]), np.array([40])),
    }
    for name, (cols, rows) in cases.items():
        remap = np.full(m, -1, dtype=np.int32); remap[cols] = np.arange(cols.size)
        row_map = np.full(n, -1, dtype=np.int32); row_map[rows] = np.arange(rows.size)
        Xs = X[rows][:, cols]
        for layout in ("csr", "csc"):
            for mode in ("xb", "xty"):
                dense = rng.standard_normal((Xs.shape[1] if mode == "xb" else Xs.shape[0], k))
                ref = _sequential_reference(Xs, dense, layout, mode)
                got = _piaso.svd_pass(str(ds.path), "ATAC_counts", mode, np.ascontiguousarray(dense),
                                      remap, int(cols.size), row_map=row_map,
                                      n_out_rows=int(rows.size), n_threads=3, layout=layout)
                np.testing.assert_array_equal(np.asarray(got), ref, err_msg=f"{name}, {layout}, {mode}")
    ds.close()


def test_runSVD_defaults_tfidf_for_peak_counts_only_and_converges_by_default(tmp_path, capsys):
    """Left unset, auto_tfidf is on for ATAC/tiles raw counts and off for RNA,
    an explicit False is honoured, and the solver is the tolerance-driven one."""
    import piaso

    rng = np.random.default_rng(61)
    X = ((rng.random((80, 240)) < 0.2) * rng.integers(1, 5, (80, 240))).astype(np.float32)
    atac = _store(tmp_path / "atac.cytome", X)
    atac.peaks["highly_variable"] = np.ones(240, dtype=bool); atac.flush()
    common = dict(measurement="counts", n_components=3, random_state=0, verbosity=1)

    piaso.tl.runSVD(atac, modality="ATAC", key_added="X_a", **common)
    out = capsys.readouterr().out
    assert "TF-IDF on the fly" in out and "chosen by method='auto'" in out

    piaso.tl.runSVD(atac, modality="ATAC", key_added="X_raw", auto_tfidf=False, **common)
    assert "values as stored" in capsys.readouterr().out

    piaso.tl.runSVD(atac, modality="ATAC", key_added="X_p", method="power", n_iter=2, **common)
    out = capsys.readouterr().out
    assert "power iteration" in out and "chosen by method='auto'" not in out
    atac.close()

    rna = _store(tmp_path / "rna.cytome", X, modality="RNA")
    rna.genes["highly_variable"] = np.ones(240, dtype=bool); rna.flush()
    piaso.tl.runSVD(rna, modality="RNA", key_added="X_r", **common)
    out = capsys.readouterr().out
    assert "values as stored" in out and "auto_tfidf" not in out
    rna.close()


def test_runSVD_reads_peak_and_tile_counts_when_no_layer_is_named(tmp_path):
    """Left unnamed, the layer follows the modality: runSVD(ds, modality='ATAC')
    on a store holding only its counts reads the counts, the same embedding as
    measurement='counts' (it looked for an INFOG layer and failed); tiles the
    same; RNA still reads INFOG."""
    import cytome
    import piaso

    rng = np.random.default_rng(67)
    X = ((rng.random((80, 240)) < 0.2) * rng.integers(1, 5, (80, 240))).astype(np.float32)
    common = dict(n_components=3, random_state=0)

    atac = _store(tmp_path / "atac.cytome", X)
    atac.peaks["highly_variable"] = np.ones(240, dtype=bool); atac.flush()
    tiles = cytome.create(str(tmp_path / "tiles.cytome"))
    tiles.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(80), "barcode": [f"B{i}" for i in range(80)]}))
    tiles.set_entity("tiles", pd.DataFrame({
        "tile_idx": np.arange(240), "tile_id": [f"chr1:{i*500}-{i*500+500}" for i in range(240)],
        "chr": ["chr1"] * 240, "start": np.arange(240) * 500, "end_": np.arange(240) * 500 + 500,
        "highly_variable": np.ones(240, dtype=bool)}))
    tiles.add_matrix("tiles_counts", sp.csr_matrix(X)); tiles.flush()
    for ds, modality in ((atac, "ATAC"), (tiles, "tiles")):
        piaso.tl.runSVD(ds, modality=modality, key_added="X_named", measurement="counts", **common)
        piaso.tl.runSVD(ds, modality=modality, key_added="X_unnamed", **common)
        np.testing.assert_array_equal(np.asarray(ds.embeddings[f"{modality}_unnamed"]),
                                      np.asarray(ds.embeddings[f"{modality}_named"]), err_msg=modality)
        ds.close()

    rna = _store(tmp_path / "rna.cytome", X, modality="RNA")
    rna.genes["highly_variable"] = np.ones(240, dtype=bool); rna.flush()
    with pytest.raises(KeyError, match="RNA_infog"):
        piaso.tl.runSVD(rna, modality="RNA", **common)
    rna.close()


def test_default_tfidf_steps_aside_for_a_layer_that_is_not_raw_counts(tmp_path, capsys):
    """Switched on by the default, TF-IDF on a layer holding negative values
    warns and factorises the values as stored; asked for explicitly, it still
    raises. And with use_highly_variable=False it uses every feature without
    needing a selection column."""
    import piaso

    rng = np.random.default_rng(71)
    Z = rng.standard_normal((50, 90)).astype(np.float32)
    ds = _store(tmp_path / "z.cytome", Z)
    with pytest.warns(UserWarning, match="does not look like raw counts"):
        piaso.tl.runSVD(ds, modality="ATAC", measurement="counts", use_highly_variable=False,
                        n_components=3, random_state=0, verbosity=1, key_added="X_z")
    assert "values as stored" in capsys.readouterr().out
    with pytest.raises(ValueError, match="raw counts"):
        piaso.tl.runSVD(ds, modality="ATAC", measurement="counts", use_highly_variable=False,
                        auto_tfidf=True, n_components=3, random_state=0, key_added="X_z2")
    ds.close()

    X = ((rng.random((50, 90)) < 0.3) * rng.integers(1, 5, (50, 90))).astype(np.float32)
    ds = _store(tmp_path / "all.cytome", X)          # no selection column at all
    piaso.tl.runSVD(ds, modality="ATAC", measurement="counts", use_highly_variable=False,
                    n_components=3, random_state=0, verbosity=1, key_added="X_all")
    out = capsys.readouterr().out
    assert "TF-IDF on the fly" in out and "90 selected columns" in out
    ds.close()


def _matrix_rows(ds, matrix="ATAC_counts"):
    return ds._conn.execute("SELECT n_rows FROM matrix_meta WHERE matrix_name=?", (matrix,)).fetchone()[0]


def test_snapshot_reader_sees_only_what_is_checkpointed_and_the_guard_knows(tmp_path):
    """The trap, reproduced through the engine: an immutable reader cannot see
    pages still in the write-ahead log. snapshot_store empties the log and
    reports honestly; with frames left in it, it reports False."""
    from piaso import _piaso
    from piaso.tools._rsvd_core import snapshot_store

    rng = np.random.default_rng(81)
    n, m, k = 40, 120, 3
    X = ((rng.random((n, m)) < 0.3) * rng.integers(1, 5, (n, m))).astype(np.float32)
    ds = _store(tmp_path / "snap.cytome", X)
    ds.checkpoint("TRUNCATE")
    # Now put a second, different matrix into the log only: autocheckpoint off,
    # so the frames stay in -wal.
    ds._conn.execute("PRAGMA wal_autocheckpoint=0")
    Y = (X * 2).astype(np.float32)
    ds.add_matrix("ATAC_double", sp.csr_matrix(Y)); ds.flush()
    assert os.path.getsize(str(ds.path) + "-wal") > 0
    remap = np.arange(m, dtype=np.int32); B = rng.standard_normal((m, k))
    # Locking reader sees the new matrix; the immutable reader must not be
    # used yet -- it cannot even find the matrix, which is the visible form
    # of "cannot see the log".
    got = _piaso.svd_pass(str(ds.path), "ATAC_double", "xb", np.ascontiguousarray(B), remap, m, n_threads=1)
    np.testing.assert_allclose(np.asarray(got), Y.astype(np.float64) @ B, rtol=1e-12, atol=1e-9)
    with pytest.raises(ValueError):
        _piaso.svd_pass(str(ds.path), "ATAC_double", "xb", np.ascontiguousarray(B), remap, m,
                        n_threads=1, snapshot=True)
    # The guard: checkpoint, verify the files, then the snapshot reader agrees.
    ok, reason = snapshot_store(ds)
    assert ok, reason
    assert os.path.getsize(str(ds.path) + "-wal") == 0
    snap = _piaso.svd_pass(str(ds.path), "ATAC_double", "xb", np.ascontiguousarray(B), remap, m,
                           n_threads=1, snapshot=True)
    np.testing.assert_array_equal(np.asarray(snap), np.asarray(got))
    ds.close()


@pytest.mark.parametrize("layout", ["csr", "csc"])
@pytest.mark.parametrize("mode", ["xb", "xty"])
def test_snapshot_reader_is_bit_identical_to_the_locking_reader(tmp_path, layout, mode):
    from piaso import _piaso
    from piaso.tools._rsvd_core import snapshot_store

    rng = np.random.default_rng(82)
    n, m, k = 45, 310, 6
    X = ((rng.random((n, m)) < 0.25) * rng.integers(1, 9, (n, m))).astype(np.float32)
    ds = _store(tmp_path / f"bit_{layout}_{mode}.cytome", X)
    _add_column_chunks(ds, X, "ATAC_counts", chunk_cols=41)
    ok, reason = snapshot_store(ds); assert ok, reason
    dense = rng.standard_normal((m if mode == "xb" else n, k)); remap = np.arange(m, dtype=np.int32)
    a = _piaso.svd_pass(str(ds.path), "ATAC_counts", mode, np.ascontiguousarray(dense), remap, m,
                        n_threads=3, layout=layout)
    b = _piaso.svd_pass(str(ds.path), "ATAC_counts", mode, np.ascontiguousarray(dense), remap, m,
                        n_threads=3, layout=layout, snapshot=True)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(b), _sequential_reference(X, dense, layout, mode))
    ds.close()


def test_snapshot_reader_opens_an_awkward_path(tmp_path):
    """A space, a percent sign and a question mark in the directory name must
    reach the same file: the URI is percent-encoded."""
    from piaso import _piaso
    from piaso.tools._rsvd_core import snapshot_store

    d = tmp_path / "odd dir 100% sure?"
    d.mkdir()
    rng = np.random.default_rng(83)
    X = ((rng.random((20, 50)) < 0.4) * rng.integers(1, 4, (20, 50))).astype(np.float32)
    ds = _store(d / "a.cytome", X)
    ok, reason = snapshot_store(ds); assert ok, reason
    B = rng.standard_normal((50, 2)); remap = np.arange(50, dtype=np.int32)
    got = _piaso.svd_pass(str(ds.path), "ATAC_counts", "xb", np.ascontiguousarray(B), remap, 50,
                          n_threads=1, snapshot=True)
    np.testing.assert_allclose(np.asarray(got), X.astype(np.float64) @ B, rtol=1e-12, atol=1e-9)
    ds.close()


def test_runSVD_does_not_need_an_empty_log(tmp_path, capsys):
    """A second connection holding a read transaction pins frames in the
    log. The engine used to need the log empty (a lock-free snapshot) and
    fell back to a locking read otherwise; it now reads through the
    dataset's own connection, so the pinned log changes nothing: no
    warning, the same result."""
    import sqlite3
    import warnings
    import piaso

    rng = np.random.default_rng(84)
    X = ((rng.random((60, 150)) < 0.3) * rng.integers(1, 5, (60, 150))).astype(np.float32)
    ds = _store(tmp_path / "fb.cytome", X)
    ds.peaks["highly_variable"] = np.ones(150, dtype=bool); ds.flush()
    common = dict(modality="ATAC", measurement="counts", n_components=3, random_state=0, verbosity=1)
    piaso.tl.runSVD(ds, key_added="X_free", **common)
    ds._conn.execute("PRAGMA wal_autocheckpoint=0")
    holder = sqlite3.connect(str(ds.path)); holder.execute("BEGIN"); holder.execute("SELECT count(*) FROM peaks").fetchone()
    ds.peaks["dummy_note"] = np.arange(150); ds.flush()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            piaso.tl.runSVD(ds, key_added="X_pinned", **common)
        out = capsys.readouterr().out
        assert "file locks" not in out, out
    finally:
        holder.rollback(); holder.close()
    from piaso.tools._embedding_names import storage_name
    a = np.asarray(ds.embeddings[storage_name("X_free", "ATAC")]); b = np.asarray(ds.embeddings[storage_name("X_pinned", "ATAC")])
    assert np.array_equal(a, b)
    ds.close()
