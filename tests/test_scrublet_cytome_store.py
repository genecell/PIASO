"""Tests for the scrublet → cytome write path.

Pins the fix for the user-reported
``OperationalError: cannot start a transaction within a transaction``
that surfaced on multi-library cytomes. Root cause was an explicit
``conn.execute('BEGIN')`` after an ``ALTER TABLE``; Python's sqlite3
driver had already opened an implicit transaction for the schema
change.

The fix:
- defensively commit any pending implicit transaction before ALTER
- commit after ALTER to finalize the schema change
- bulk write via ``executemany`` inside ``with conn:`` instead of an
  explicit BEGIN
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_multilib_cytome(path, libs=("A", "B", "C"), per_lib=60, seed=0):
    """Build a tiny multi-library cytome with enough cells per library
    that scrublet's >50 cells/library guard passes."""
    import cytome
    rng = np.random.default_rng(seed)
    n_cells = per_lib * len(libs)
    n_genes = 80
    sample = np.repeat(np.array(libs, dtype=object), per_lib)
    rng.shuffle(sample)

    X = rng.negative_binomial(2, 0.4, size=(n_cells, n_genes)).astype(np.float32)
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"BC-{i}" for i in range(n_cells)],
        "Sample": sample.tolist(),
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "gene_id": [f"g{i}" for i in range(n_genes)],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


def test_store_cytome_results_after_alter_table(tmp_path):
    """The original failure mode: fresh cytome, no scrublet columns,
    ALTER TABLE runs THEN the store path used to BEGIN explicitly.
    With the fix the writes must complete cleanly."""
    import cytome
    from piaso.preprocessing._scrublet import _store_cytome_results

    work = tmp_path / "freshcols.cytome"
    ds = _build_multilib_cytome(str(work), libs=("A", "B"), per_lib=20)
    n_cells = ds.n_cells

    # Sanity: columns do not exist yet
    cols = [r[1] for r in ds._conn.execute("PRAGMA table_info(cells)").fetchall()]
    assert "scrublet_score" not in cols
    assert "is_doublet" not in cols

    scores = np.linspace(0.0, 0.5, n_cells, dtype=np.float64)
    predicted = (scores > 0.3).astype(bool)

    # Pre-fix this raised OperationalError after the ALTER TABLE.
    _store_cytome_results(ds, scores, predicted)

    # Re-read and confirm round-trip
    df = ds.cells.to_pandas()
    assert "scrublet_score" in df.columns
    assert "is_doublet" in df.columns
    np.testing.assert_allclose(
        df["scrublet_score"].values.astype(np.float64), scores, rtol=1e-7
    )
    np.testing.assert_array_equal(
        df["is_doublet"].values.astype(bool), predicted
    )
    ds.close()


def test_store_cytome_results_when_columns_already_exist(tmp_path):
    """Re-running scrublet after a prior call (columns already present):
    no ALTER fires, the store path must still work."""
    import cytome
    from piaso.preprocessing._scrublet import _store_cytome_results

    work = tmp_path / "rerun.cytome"
    ds = _build_multilib_cytome(str(work), libs=("A",), per_lib=30)
    n_cells = ds.n_cells

    # First call adds the columns.
    s1 = np.full(n_cells, 0.1, dtype=np.float64)
    p1 = np.zeros(n_cells, dtype=bool)
    _store_cytome_results(ds, s1, p1)

    # Second call: columns already exist, ALTER skipped. Values updated.
    s2 = np.full(n_cells, 0.7, dtype=np.float64)
    p2 = np.ones(n_cells, dtype=bool)
    _store_cytome_results(ds, s2, p2)

    df = ds.cells.to_pandas()
    np.testing.assert_allclose(df["scrublet_score"].values.astype(np.float64), s2)
    np.testing.assert_array_equal(df["is_doublet"].values.astype(bool), p2)

    # Ensure no duplicate columns were created
    cols = [r[1] for r in ds._conn.execute("PRAGMA table_info(cells)").fetchall()]
    assert cols.count("scrublet_score") == 1
    assert cols.count("is_doublet") == 1
    ds.close()


def test_store_cytome_results_with_open_transaction(tmp_path):
    """Simulate the multi-library streaming case: an earlier pass left
    an implicit transaction open. The defensive ``commit`` in the fix
    must clear it so ALTER TABLE doesn't trip."""
    import cytome
    from piaso.preprocessing._scrublet import _store_cytome_results

    work = tmp_path / "openTxn.cytome"
    ds = _build_multilib_cytome(str(work), libs=("A", "B"), per_lib=10)
    n_cells = ds.n_cells

    # Force the connection into an open transaction (mimic
    # an in-flight streaming UPDATE / temp-table write).
    ds._conn.execute("CREATE TEMP TABLE _tmp_probe (x INTEGER)")
    ds._conn.execute("INSERT INTO _tmp_probe VALUES (1)")
    assert ds._conn.in_transaction, (
        "test fixture failed: connection should be in a transaction now"
    )

    scores = np.zeros(n_cells, dtype=np.float64)
    predicted = np.zeros(n_cells, dtype=bool)

    # Must not raise.
    _store_cytome_results(ds, scores, predicted)

    df = ds.cells.to_pandas()
    assert "scrublet_score" in df.columns
    ds.close()


def test_scrublet_multilibrary_end_to_end_no_BEGIN_error(tmp_path):
    """The full piaso.pp.scrublet call on a multi-library cytome. This
    is the exact code path the user reported failing on."""
    import piaso

    work = tmp_path / "multilib.cytome"
    ds = _build_multilib_cytome(str(work), libs=("A", "B", "C"), per_lib=70)

    # The user's reported invocation:
    #   piaso.pp.scrublet(ds, library_key='Sample', random_state=10)
    # Pre-fix this raised OperationalError in _store_cytome_results.
    piaso.pp.scrublet(ds, library_key="Sample", random_state=10, verbose=False)

    df = ds.cells.to_pandas()
    assert "scrublet_score" in df.columns
    assert "is_doublet" in df.columns
    # Library labels preserved
    assert sorted(df["Sample"].unique()) == ["A", "B", "C"]
    ds.close()
