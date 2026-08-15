"""Tests for n_cells / matrix row count mismatch handling (Bug 4).

Tests both:
  D — cytome repair() fixes mismatched matrix metadata
  E — PIASO streaming functions handle mismatches defensively
"""
import os
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import E18_CYTOME

pytestmark = pytest.mark.requires_e18

# tmp_dir fixture provided by conftest.py


# ──────────────────────────────────────────────────────────────────────
# Part D: validate + repair
# ──────────────────────────────────────────────────────────────────────

def test_validate_clean_cytome():
    """A normal cytome passes validation."""
    import cytome
    from cytome.utils.validation import validate

    ds = cytome.open(E18_CYTOME)
    report = validate(ds)
    ds.close()
    # May have minor issues but matrix_rows should pass
    matrix_fails = [f for f in report.checks_failed if f.startswith("matrix_rows:")]
    assert len(matrix_fails) == 0, f"Unexpected matrix row mismatches: {matrix_fails}"


def test_subset_produces_consistent_cytome(tmp_dir):
    """cytome.subset() creates a valid subset with matching row counts."""
    import cytome
    from cytome.utils.validation import validate
    from cytome.io.subset import subset

    ds = cytome.open(E18_CYTOME)
    n = ds.n_cells
    # Subset to first half
    keep = np.arange(n // 2)
    out_path = os.path.join(tmp_dir, "subset.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)

    report = validate(out)
    assert report.passed, f"Subset validation failed: {report.checks_failed}"
    assert out.n_cells == len(keep)

    # Verify matrix row counts match
    for name, n_rows in out._conn.execute(
        "SELECT matrix_name, n_rows FROM matrix_meta"
    ).fetchall():
        assert int(n_rows) == out.n_cells, \
            f"Matrix {name}: n_rows={n_rows} != n_cells={out.n_cells}"

    out.close()
    ds.close()


def test_repair_fixes_matrix_mismatch(tmp_dir):
    """repair() fixes matrix_meta.n_rows when it exceeds cells count."""
    import cytome
    from cytome.utils.validation import validate, repair
    from cytome.io.subset import subset

    ds = cytome.open(E18_CYTOME)
    n = ds.n_cells

    # Create a clean subset
    keep = np.arange(n // 2)
    out_path = os.path.join(tmp_dir, "mismatch.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    true_n = out.n_cells

    # Artificially break it: inflate matrix_meta.n_rows by 1
    for name, in out._conn.execute("SELECT matrix_name FROM matrix_meta").fetchall():
        out._conn.execute(
            "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = ?",
            (true_n + 1, name),
        )
    out._conn.commit()

    # Validate should fail
    report = validate(out)
    matrix_fails = [f for f in report.checks_failed if f.startswith("matrix_rows:")]
    assert len(matrix_fails) > 0, "Expected matrix_rows failure"

    # Repair should fix it
    repair(out)

    report2 = validate(out)
    matrix_fails2 = [f for f in report2.checks_failed if f.startswith("matrix_rows:")]
    assert len(matrix_fails2) == 0, f"Repair didn't fix: {report2.checks_failed}"

    # Verify n_rows matches n_cells after repair
    for name, n_rows in out._conn.execute(
        "SELECT matrix_name, n_rows FROM matrix_meta"
    ).fetchall():
        assert int(n_rows) == true_n, \
            f"After repair, {name}: n_rows={n_rows} != n_cells={true_n}"

    out.close()


def test_repair_truncates_excess_data(tmp_dir):
    """repair() properly truncates CSR data when matrix has excess rows."""
    import cytome
    from cytome.utils.validation import repair
    from cytome.io.subset import subset
    from cytome.core.measurement import MeasurementLayer

    ds = cytome.open(E18_CYTOME)
    n = ds.n_cells
    keep = np.arange(n // 2)
    out_path = os.path.join(tmp_dir, "truncate.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    # Get a matrix name
    mat_name = out._conn.execute(
        "SELECT matrix_name FROM matrix_meta LIMIT 1"
    ).fetchone()[0]

    # Read original data
    ml = MeasurementLayer(out._conn, mat_name)
    original = ml.to_memory()
    true_n = original.shape[0]

    # Inflate n_rows and add a fake extra row to the last chunk
    out._conn.execute(
        "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = ?",
        (true_n + 5, mat_name),
    )
    out._conn.commit()

    # Repair
    repair(out)

    # Read back — should be truncated to true_n
    ml2 = MeasurementLayer(out._conn, mat_name)
    repaired = ml2.to_memory()
    assert repaired.shape[0] == true_n, \
        f"Expected {true_n} rows after repair, got {repaired.shape[0]}"

    # Data should be identical
    diff = (original - repaired).nnz
    assert diff == 0, f"Repaired matrix differs: {diff} nonzero differences"

    out.close()


# ──────────────────────────────────────────────────────────────────────
# Part E: defensive allocation in streaming functions
# ──────────────────────────────────────────────────────────────────────

def test_safe_n_cells_warns_on_mismatch(tmp_dir):
    """_safe_n_cells warns and returns max when mismatch detected."""
    import cytome
    from cytome.io.subset import subset
    from piaso.tools._normalization import _safe_n_cells

    ds = cytome.open(E18_CYTOME)
    keep = np.arange(ds.n_cells // 2)
    out_path = os.path.join(tmp_dir, "warn.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    true_n = out.n_cells

    # Get a matrix name and its modality/layer
    mat_name = out._conn.execute(
        "SELECT matrix_name FROM matrix_meta LIMIT 1"
    ).fetchone()[0]
    parts = mat_name.split("_", 1)
    modality, layer = parts[0], parts[1]

    # No mismatch → no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n_alloc, n_true = _safe_n_cells(out, modality, layer)
        assert n_alloc == true_n
        assert n_true == true_n
        assert len(w) == 0

    # Create mismatch
    out._conn.execute(
        "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = ?",
        (true_n + 10, mat_name),
    )
    out._conn.commit()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n_alloc, n_true = _safe_n_cells(out, modality, layer)
        assert n_alloc == true_n + 10  # max of the two
        assert n_true == true_n
        assert len(w) == 1
        assert "consistency" in str(w[0].message).lower()

    out.close()


def test_streaming_svd_on_subset(tmp_dir):
    """Streaming SVD works correctly on a subset cytome."""
    import cytome
    import piaso
    from cytome.io.subset import subset

    ds = cytome.open(E18_CYTOME)
    keep = np.arange(ds.n_cells // 2)
    out_path = os.path.join(tmp_dir, "svd_subset.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    # Check if RNA_counts matrix exists
    has_rna = out._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'RNA_counts' LIMIT 1"
    ).fetchone()
    if not has_rna:
        out.close()
        pytest.skip("Subset doesn't have RNA_counts matrix")

    result = piaso.tl.runSVD(
        out, use_highly_variable=False, n_components=10,
        n_iter=2, random_state=10, streaming=True, batch_size=512,
        measurement='counts', modality='RNA', verbosity=1,
        return_svd=True,   # cytome path returns None by default now
    )
    embeddings = result[0]
    assert embeddings.shape == (out.n_cells, 10), \
        f"Expected ({out.n_cells}, 10), got {embeddings.shape}"

    out.close()


def test_streaming_svd_with_mismatch(tmp_dir):
    """Streaming SVD handles n_rows > n_cells without crashing."""
    import cytome
    import piaso
    from cytome.io.subset import subset

    ds = cytome.open(E18_CYTOME)
    keep = np.arange(ds.n_cells // 2)
    out_path = os.path.join(tmp_dir, "svd_mismatch.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    has_rna = out._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'RNA_counts' LIMIT 1"
    ).fetchone()
    if not has_rna:
        out.close()
        pytest.skip("Subset doesn't have RNA_counts matrix")

    true_n = out.n_cells

    # Inflate matrix_meta.n_rows to simulate mismatch
    out._conn.execute(
        "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = 'RNA_counts'",
        (true_n + 5,),
    )
    out._conn.commit()

    # Should warn but NOT crash
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = piaso.tl.runSVD(
            out, use_highly_variable=False, n_components=10,
            n_iter=2, random_state=10, streaming=True, batch_size=512,
            measurement='counts', modality='RNA', verbosity=0,
            return_svd=True,   # cytome path returns None by default now
        )
        consistency_warnings = [x for x in w if "consistency" in str(x.message).lower()]
        assert len(consistency_warnings) > 0, "Expected consistency warning"

    embeddings = result[0]
    # Output should be truncated to true cell count
    assert embeddings.shape == (true_n, 10), \
        f"Expected ({true_n}, 10), got {embeddings.shape}"

    out.close()


def test_tfidf_stats_with_mismatch(tmp_dir):
    """compute_tfidf_stats handles n_rows > n_cells without crashing."""
    import cytome
    from cytome.io.subset import subset

    ds = cytome.open(E18_CYTOME)

    # Check if ATAC_counts exists on the original
    has_atac = ds._conn.execute(
        "SELECT 1 FROM matrix_meta WHERE matrix_name = 'ATAC_counts' LIMIT 1"
    ).fetchone()
    if not has_atac:
        ds.close()
        pytest.skip("E18 cytome doesn't have ATAC_counts matrix")

    keep = np.arange(ds.n_cells // 2)
    out_path = os.path.join(tmp_dir, "tfidf_mismatch.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    true_n = out.n_cells

    # Inflate n_rows
    out._conn.execute(
        "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = 'ATAC_counts'",
        (true_n + 3,),
    )
    out._conn.commit()

    from piaso.tools._runTFIDF import compute_tfidf_stats

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stats = compute_tfidf_stats(out, measurement='counts', modality='ATAC')
        consistency_warnings = [x for x in w if "consistency" in str(x.message).lower()]
        assert len(consistency_warnings) > 0, "Expected consistency warning"

    # cell_depth should be allocated to the larger size but only true_n entries used
    assert stats["cell_depth"].shape[0] >= true_n

    out.close()


# ──────────────────────────────────────────────────────────────────────
# Part F: iter_chunks root capping + column repair
# ──────────────────────────────────────────────────────────────────────

def test_iter_chunks_caps_at_entity_count(tmp_dir):
    """iter_chunks() caps yielded indices at entity count, never exceeds n_cells."""
    import cytome
    from cytome.io.subset import subset

    ds = cytome.open(E18_CYTOME)
    keep = np.arange(ds.n_cells // 2)
    out_path = os.path.join(tmp_dir, "cap.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    true_n = out.n_cells

    # Inflate n_rows beyond true cell count
    out._conn.execute(
        "UPDATE matrix_meta SET n_rows = ? WHERE matrix_name = 'RNA_counts'",
        (true_n + 20,),
    )
    out._conn.commit()

    # Iterate — no index should exceed true_n - 1
    max_idx = -1
    for chunk, indices in out.iter_chunks(modality='RNA', layer='counts', batch_size=512):
        if len(indices) > 0:
            max_idx = max(max_idx, int(indices.max()))

    assert max_idx < true_n, \
        f"iter_chunks yielded index {max_idx} >= n_cells {true_n}"
    assert max_idx == true_n - 1, \
        f"Expected max index {true_n - 1}, got {max_idx}"

    out.close()


def test_repair_fixes_column_mismatch(tmp_dir):
    """repair() fixes matrix_meta.n_cols when it doesn't match entity count."""
    import cytome
    from cytome.utils.validation import validate, repair
    from cytome.io.subset import subset

    ds = cytome.open(E18_CYTOME)
    keep = np.arange(ds.n_cells // 2)
    out_path = os.path.join(tmp_dir, "col_mismatch.cytome")
    out = subset(ds, keep, out_path, include_fragments=False)
    ds.close()

    # Get a matrix and its col_entity
    meta = out._conn.execute(
        "SELECT matrix_name, n_cols, col_entity FROM matrix_meta LIMIT 1"
    ).fetchone()
    mat_name, orig_n_cols, col_entity = meta

    if col_entity not in ("genes", "peaks"):
        out.close()
        pytest.skip(f"col_entity '{col_entity}' not testable")

    # Inflate n_cols
    out._conn.execute(
        "UPDATE matrix_meta SET n_cols = ? WHERE matrix_name = ?",
        (int(orig_n_cols) + 10, mat_name),
    )
    out._conn.commit()

    # Validate should fail
    report = validate(out)
    col_fails = [f for f in report.checks_failed if f.startswith("matrix_cols:")]
    assert len(col_fails) > 0, "Expected column mismatch failure"

    # Repair should fix it
    repair(out)

    report2 = validate(out)
    col_fails2 = [f for f in report2.checks_failed if f.startswith("matrix_cols:")]
    assert len(col_fails2) == 0, f"Repair didn't fix cols: {report2.checks_failed}"

    out.close()
