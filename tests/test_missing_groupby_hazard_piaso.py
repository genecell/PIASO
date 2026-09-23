"""Every place PIASO interpolates a user-supplied column name into SQL.

SQLite has a compatibility misfeature: a double-quoted token that does not
resolve to a column is treated as a *string literal* rather than raising. So
`SELECT "Leiden" FROM cells` on a cytome with no `Leiden` column returns the
string 'Leiden' for every row. The caller sees one group, named after the typo,
and every number downstream is internally consistent with it -- a confident
wrong answer, which is worse than a crash.

These tests walk each user-facing entry point that reaches such a query and
check it looks the column up first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _rna_cytome(path):
    import anndata as ad
    import cytome
    rng = np.random.default_rng(0)
    n, g = 60, 30
    a = ad.AnnData(
        X=sp.csr_matrix(rng.poisson(1.0, (n, g)).astype(np.float32)),
        obs=pd.DataFrame({"Leiden": np.array(["0", "1"])[rng.integers(0, 2, n)]},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]),
    )
    return cytome.from_anndata(a, modality="RNA", output=str(path))


def test_sqlite_really_treats_an_unknown_quoted_name_as_a_string(tmp_path):
    """The premise, pinned on its own so the guards can be revisited if it
    ever stops being true."""
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE cells (cell_idx INTEGER)")
    con.executemany("INSERT INTO cells VALUES (?)", [(0,), (1,)])
    rows = con.execute('SELECT "Leiden" FROM cells').fetchall()
    assert rows == [("Leiden",), ("Leiden",)], (
        "SQLite no longer falls back to a string literal — the guards that "
        "exist because of it can be reconsidered")
    con.close()


def test_require_cells_column_names_what_is_there(tmp_path):
    from piaso.utils._cytome_compat import require_cells_column
    ds = _rna_cytome(tmp_path / "a.cytome")
    require_cells_column(ds, "Leiden")            # present: no raise
    with pytest.raises(KeyError) as exc:
        require_cells_column(ds, "Cluster")
    assert "Cluster" in str(exc.value) and "Leiden" in str(exc.value)
    ds.close()


def test_a_case_variant_is_accepted_because_sqlite_accepts_it(tmp_path):
    """SQLite column names are case-insensitive, so `leiden` really does name
    `Leiden`. The guard is for a name that resolves to nothing -- turning away
    one that resolves fine would break working code in the name of safety."""
    from piaso.utils._cytome_compat import require_cells_column
    ds = _rna_cytome(tmp_path / "b.cytome")
    require_cells_column(ds, "leiden")           # no raise
    rows = ds._conn.execute('SELECT "leiden" FROM cells LIMIT 2').fetchall()
    assert {r[0] for r in rows} <= {"0", "1"}
    ds.close()


def test_runSCALAR_checks_before_it_queries(tmp_path):
    from piaso.tools._ligandReceptor import _scalar_labels
    ds = _rna_cytome(tmp_path / "c.cytome")
    with pytest.raises(KeyError, match="not a column"):
        _scalar_labels(ds, "Leiden_typo")
    labels = _scalar_labels(ds, "Leiden")
    assert set(labels) == {"0", "1"}, "the real column must still work"
    ds.close()


def test_runSCALAR_on_anndata_lists_the_columns_too(tmp_path):
    import anndata as ad
    from piaso.tools._ligandReceptor import _scalar_labels
    a = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"Leiden": list("aabb")}, index=list("wxyz")))
    with pytest.raises(KeyError) as exc:
        _scalar_labels(a, "leiden")
    assert "Did you mean 'Leiden'" in str(exc.value)
