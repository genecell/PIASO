"""filter_cells keeps the same cells on a cytome as on AnnData when a column has missing values.

Regression tests for the code audit of 2026-10-27 (review I1, finding 8): negated predicates
(``!=``, ``not``, ``~isin``, ``not in``) were pushed down to SQL, where any comparison with NULL is
unknown and the row is dropped, while pandas keeps it. ``cell_type != 'Doublet'`` kept 6 of 10 cells
on a cytome and 9 of 10 on AnnData, and ``inplace=True`` then deleted the unlabelled cells for good;
``sample_id != 'X'`` kept no cell of a pipeline store (its sample column is empty). The store now
follows the pandas semantics. Also: dict masks read ``None`` as "missing" on both inputs, negative
integer indices are refused, and the QC thresholds take ``layer=``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import piaso

cytome = pytest.importorskip("cytome")

N = 12
CT = np.array(["Doublet", "A", None, "B", None, "A", "B", None, "Doublet", "A", "B", "A"], dtype=object)
NF = np.array([1., 2., np.nan, 4., 5., np.nan, 7., 8., 9., 10., 11., np.nan])
FLAG = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])


def _pair(tmp_path, with_null_flag=False):
    X = sp.csr_matrix(np.arange(N * 4, dtype=np.float32).reshape(N, 4) % 5)
    obs = pd.DataFrame({"cell_type": CT, "sample_id": np.array([None] * N, dtype=object), "nf": NF,
                        "flag": FLAG}, index=[f"c{i}" for i in range(N)])
    a = AnnData(X, obs=obs)
    a.layers["raw"] = X * 10
    ds = cytome.create(str(tmp_path / "f.cytome"))
    cells = obs.reset_index(drop=True)
    cells.insert(0, "barcode", obs.index)
    cells.insert(0, "cell_idx", np.arange(N))
    ds.set_entity("cells", cells)
    ds.set_entity("genes", pd.DataFrame({"gene_idx": np.arange(4), "gene_id": list("abcd")}))
    ds.add_matrix("RNA_counts", X)
    ds.add_matrix("RNA_raw", X * 10)
    ds.flush()
    return a, ds


QUERIES = [
    "cell_type != 'Doublet'",
    "~cell_type.isin(['Doublet'])",
    "cell_type not in ['Doublet']",
    "cell_type == 'A'",
    "cell_type in ['A', 'B']",
    "not (nf > 4)",
    "~(nf > 4)",
    "nf != 5",
    "nf.between(2, 8)",
    "not nf.between(2, 8)",
    "sample_id != 'X'",
    "sample_id == 'X'",
    "cell_type == 'A' or nf < 3",
    "not (cell_type == 'A' or nf < 3)",
    "(cell_type != 'B') & (nf >= 2)",
    "flag == 1",
    "not (flag == 1)",
]


@pytest.mark.parametrize("query", QUERIES)
def test_queries_keep_the_same_cells(tmp_path, query):
    a, ds = _pair(tmp_path)
    try:
        want = piaso.pp.filter_cells(a, mask=query, inplace=False, verbose=0)
        got = piaso.pp.filter_cells(ds, mask=query, inplace=False, verbose=0)
        # and the reference is pandas itself
        np.testing.assert_array_equal(want, a.obs.eval(query).to_numpy().astype(bool))
        np.testing.assert_array_equal(got, want)
    finally:
        ds.close()


@pytest.mark.parametrize("spec", [
    {"cell_type": "A"},
    {"cell_type": ["A", "B"]},
    {"cell_type": None},
    {"cell_type": ["A", None]},
    {"nf": (3, None)},
    {"nf": (None, 8)},
    {"sample_id": None},
    {"cell_type": "A", "nf": (1, 10)},
])
def test_dict_masks_keep_the_same_cells(tmp_path, spec):
    a, ds = _pair(tmp_path)
    try:
        want = piaso.pp.filter_cells(a, mask=spec, inplace=False, verbose=0)
        got = piaso.pp.filter_cells(ds, mask=spec, inplace=False, verbose=0)
        np.testing.assert_array_equal(got, want)
    finally:
        ds.close()


def test_dict_none_selects_the_missing_cells(tmp_path):
    a, ds = _pair(tmp_path)
    try:
        m = piaso.pp.filter_cells(ds, mask={"cell_type": None}, inplace=False, verbose=0)
        np.testing.assert_array_equal(m, pd.isna(CT))
    finally:
        ds.close()


def test_inplace_filter_keeps_the_unlabelled_cells_of_a_store(tmp_path):
    a, ds = _pair(tmp_path)
    try:
        n_after = piaso.pp.filter_cells(ds, mask="cell_type != 'Doublet'", inplace=True, verbose=0)
        assert n_after == N - 2
        piaso.pp.filter_cells(a, mask="cell_type != 'Doublet'", inplace=True, verbose=0)
        assert a.n_obs == n_after
    finally:
        ds.close()


def test_negative_indices_are_refused(tmp_path):
    a, ds = _pair(tmp_path)
    try:
        for data in (a, ds):
            with pytest.raises(ValueError, match="indices"):
                piaso.pp.filter_cells(data, mask=np.array([0, -1]), inplace=False, verbose=0)
    finally:
        ds.close()


def test_thresholds_take_a_layer_on_both_inputs(tmp_path):
    a, ds = _pair(tmp_path)
    try:
        for layer, cut in ((None, 8), ("raw", 80)):
            wa = piaso.pp.filter_cells(a, min_counts=cut, layer=layer, inplace=False, verbose=0)
            wc = piaso.pp.filter_cells(ds, min_counts=cut, layer=layer, inplace=False, verbose=0)
            np.testing.assert_array_equal(wa, wc)
            assert 0 < wa.sum() < N
        with pytest.raises(KeyError, match="nope"):
            piaso.pp.filter_cells(a, min_counts=1, layer="nope", inplace=False, verbose=0)
    finally:
        ds.close()


# ------------------------------------------------------------------ pandas' nullable dtypes

NULLABLE_MASKS = ["~is_doublet", "is_doublet", "not is_doublet", "is_doublet == False", "is_doublet != True",
                  "~is_doublet & (n > 1)", {"is_doublet": False}, {"is_doublet": [False, None]}]


def _nullable_pair(tmp_path):
    calls = pd.array([True, False, pd.NA, False, pd.NA, True], dtype="boolean")
    n = len(calls)
    X = sp.csr_matrix(np.ones((n, 2), dtype=np.float32))
    a = AnnData(X, obs=pd.DataFrame({"is_doublet": calls, "n": pd.array([1, 2, 3, None, 5, 6], dtype="Int64")},
                                    index=[f"c{i}" for i in range(n)]))
    ds = cytome.create(str(tmp_path / "nb.cytome"))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": a.obs_names}))
    ds.set_entity("genes", pd.DataFrame({"gene_idx": [0, 1], "gene_id": ["a", "b"]}))
    ds.add_matrix("RNA_counts", X)
    ds.flush()
    ds._conn.execute("ALTER TABLE cells ADD COLUMN is_doublet INTEGER")
    ds._conn.execute("ALTER TABLE cells ADD COLUMN n INTEGER")
    for i, (v, k) in enumerate(zip([1, 0, None, 0, None, 1], [1, 2, 3, None, 5, 6])):
        ds._conn.execute("UPDATE cells SET is_doublet=?, n=? WHERE cell_idx=?", (v, k, i))
    ds._conn.commit()
    return a, ds


@pytest.mark.parametrize("spec", NULLABLE_MASKS, ids=str)
def test_a_nullable_column_keeps_the_same_cells(tmp_path, spec):
    """Scrublet's is_doublet is a nullable boolean once some cells are not scored. pandas' eval
    returns missing values there that no mask can hold, so every one of these failed on AnnData
    (TypeError) while the store answered; both now give the store's answer."""
    a, ds = _nullable_pair(tmp_path)
    try:
        want = piaso.pp.filter_cells(ds, mask=spec, inplace=False, verbose=0)
        got = piaso.pp.filter_cells(a, mask=spec, inplace=False, verbose=0)
        np.testing.assert_array_equal(got, want)
    finally:
        ds.close()


def test_negating_the_doublet_call_keeps_the_unscored_cells(tmp_path):
    a, ds = _nullable_pair(tmp_path)
    try:
        for data in (a, ds):
            keep = piaso.pp.filter_cells(data, mask="~is_doublet", inplace=False, verbose=0)
            assert np.flatnonzero(keep).tolist() == [1, 2, 3, 4]
    finally:
        ds.close()


def test_a_query_that_is_not_a_condition_is_refused(tmp_path):
    a, ds = _pair(tmp_path)
    try:
        # pandas evaluates this to numbers; they were read as cell indices
        with pytest.raises(TypeError, match="True / False"):
            piaso.pp.filter_cells(a, mask="flag + 1", inplace=False, verbose=0)
        with pytest.raises(TypeError, match="True / False"):
            piaso.pp.filter_cells(ds, mask="flag + 1", inplace=False, verbose=0)
    finally:
        ds.close()
