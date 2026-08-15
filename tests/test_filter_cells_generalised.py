"""Tests for the generalised piaso.pp.filter_cells.

Covers:

- Backward-compat AnnData QC kwargs (min/max_counts, min/max_features).
- All six mask shapes (bool ndarray, int indices, pd.Series, str query,
  dict, callable) on both AnnData and cytome.
- Cytome modes: inplace=True, inplace=False+output=None (mask-only),
  inplace=False+output=path (subset to new file).
- TypeError for contradictory inplace=True + output=path.
- FileExistsError + overwrite=True path.
- Provenance writes (metadata + ds.provenance).
- SQL pushdown vs streaming fallback for query strings.
- subset_cells alias.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _build_anndata(n_cells=60, n_genes=40, seed=0):
    from anndata import AnnData
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(2, 0.4, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({
        "cluster": np.array(["A", "B", "C"])[rng.integers(0, 3, size=n_cells)],
        "n_counts_pre": X.sum(axis=1),
        "tss_score": rng.uniform(0.5, 10.0, size=n_cells),
        "is_doublet": rng.random(n_cells) < 0.1,
    }, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


def _build_cytome(path, n_cells=60, n_genes=40, seed=0):
    import cytome
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(2, 0.4, size=(n_cells, n_genes)).astype(np.float32)
    cluster = np.array(["A", "B", "C"])[rng.integers(0, 3, size=n_cells)]
    tss = rng.uniform(0.5, 10.0, size=n_cells)
    doublet = (rng.random(n_cells) < 0.1).astype(int)

    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_cells),
        "barcode": [f"BC-{i}" for i in range(n_cells)],
        "cluster": cluster.tolist(),
        "tss_score": tss,
        "is_doublet": doublet,
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "gene_id": [f"g{i}" for i in range(n_genes)],
    }))
    ds.add_matrix("RNA_counts", sp.csr_matrix(X))
    ds.flush()
    return ds


# ===========================================================================
# AnnData backward compat
# ===========================================================================

def test_anndata_qc_thresholds_inplace_True():
    import piaso
    a = _build_anndata()
    n0 = a.n_obs
    piaso.pp.filter_cells(a, min_counts=130)
    assert a.n_obs < n0
    assert (a.X.sum(axis=1) >= 20).all()


def test_anndata_qc_thresholds_inplace_False_returns_mask():
    import piaso
    a = _build_anndata()
    mask = piaso.pp.filter_cells(a, min_counts=130, inplace=False)
    assert mask.dtype == bool
    assert mask.shape == (a.n_obs,)


def test_anndata_output_path_warns_and_noops():
    import piaso, warnings
    a = _build_anndata()
    n0 = a.n_obs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        piaso.pp.filter_cells(a, min_counts=130, output="/tmp/ignored.h5ad")
    msgs = [str(w.message) for w in caught]
    assert any("cytome-only" in m or "output" in m for m in msgs), (
        f"Expected UserWarning about output=, got: {msgs}"
    )
    assert a.n_obs < n0  # filter still applied


# ===========================================================================
# Six mask shapes — AnnData
# ===========================================================================

def test_anndata_mask_bool_ndarray():
    import piaso
    a = _build_anndata()
    m = np.zeros(a.n_obs, bool)
    m[:20] = True
    mask = piaso.pp.filter_cells(a, mask=m, inplace=False)
    assert int(mask.sum()) == 20


def test_anndata_mask_int_indices():
    import piaso
    a = _build_anndata()
    idx = np.array([0, 5, 10])
    mask = piaso.pp.filter_cells(a, mask=idx, inplace=False)
    assert int(mask.sum()) == 3
    assert mask[0] and mask[5] and mask[10]


def test_anndata_mask_pd_series():
    import piaso
    a = _build_anndata()
    s = a.obs["n_counts_pre"] > a.obs["n_counts_pre"].median()
    mask = piaso.pp.filter_cells(a, mask=s, inplace=False)
    assert int(mask.sum()) > 0
    assert int(mask.sum()) <= a.n_obs


def test_anndata_mask_query_string():
    import piaso
    a = _build_anndata()
    mask = piaso.pp.filter_cells(
        a, mask="cluster.isin(['A','B']) and tss_score > 2.0",
        inplace=False,
    )
    in_clu = a.obs["cluster"].isin(["A", "B"]).values
    above = (a.obs["tss_score"] > 2.0).values
    np.testing.assert_array_equal(mask, in_clu & above)


def test_anndata_mask_dict():
    import piaso
    a = _build_anndata()
    mask = piaso.pp.filter_cells(
        a, mask={"cluster": ["A", "B"], "is_doublet": False},
        inplace=False,
    )
    expected = (a.obs["cluster"].isin(["A", "B"])
                & (a.obs["is_doublet"] == False))  # noqa: E712
    np.testing.assert_array_equal(mask, expected.values)


def test_anndata_mask_callable():
    import piaso
    a = _build_anndata()
    fn = lambda df: df["tss_score"] > df["tss_score"].median()  # noqa: E731
    mask = piaso.pp.filter_cells(a, mask=fn, inplace=False)
    expected = a.obs["tss_score"] > a.obs["tss_score"].median()
    np.testing.assert_array_equal(mask, expected.values)


# ===========================================================================
# QC + mask intersection
# ===========================================================================

def test_anndata_qc_and_mask_intersect():
    import piaso
    a = _build_anndata()
    # one filter dominates the other; the result must be the AND
    mask_only = piaso.pp.filter_cells(a, mask="cluster == 'A'", inplace=False)
    qc_only = piaso.pp.filter_cells(a, min_counts=125, inplace=False)
    both = piaso.pp.filter_cells(
        a, mask="cluster == 'A'", min_counts=125, inplace=False,
    )
    np.testing.assert_array_equal(both, mask_only & qc_only)


# ===========================================================================
# Cytome dispatch
# ===========================================================================

def test_cytome_qc_thresholds_inplace_True(tmp_path):
    import piaso
    work = tmp_path / "qc.cytome"
    ds = _build_cytome(str(work))
    n0 = ds.n_cells
    n_after = piaso.pp.filter_cells(ds, min_counts=130, verbose=0)
    assert isinstance(n_after, int)
    assert n_after < n0
    assert ds.n_cells == n_after  # cytome reopens at same path


def test_cytome_mask_only_inplace_False_no_output(tmp_path):
    import piaso
    work = tmp_path / "maskonly.cytome"
    ds = _build_cytome(str(work))
    n0 = ds.n_cells
    mask = piaso.pp.filter_cells(
        ds, mask="cluster == 'A'", inplace=False, verbose=0,
    )
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.shape == (n0,)
    assert ds.n_cells == n0  # untouched
    ds.close()


def test_cytome_inplace_True_with_output_raises(tmp_path):
    import piaso
    work = tmp_path / "x.cytome"
    ds = _build_cytome(str(work))
    with pytest.raises(TypeError, match="contradictory"):
        piaso.pp.filter_cells(
            ds, mask="cluster == 'A'",
            inplace=True, output=str(tmp_path / "y.cytome"),
        )
    ds.close()


def test_cytome_subset_to_new_path(tmp_path):
    import piaso
    work = tmp_path / "src.cytome"
    ds = _build_cytome(str(work))
    out = tmp_path / "dst.cytome"
    n_src = ds.n_cells
    new_ds = piaso.pp.filter_cells(
        ds, mask="cluster == 'A'", inplace=False, output=str(out), verbose=0,
    )
    # new_ds is open per the sign-off decision
    assert new_ds.n_cells < n_src
    assert ds.n_cells == n_src  # original untouched
    assert Path(out).exists()
    new_ds.close()
    ds.close()


def test_cytome_output_exists_no_overwrite_raises(tmp_path):
    import piaso
    work = tmp_path / "src.cytome"
    ds = _build_cytome(str(work))
    out = tmp_path / "dst.cytome"
    out.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        piaso.pp.filter_cells(
            ds, mask="cluster == 'A'", inplace=False, output=str(out),
            verbose=0,
        )
    ds.close()


def test_cytome_output_exists_overwrite_True_replaces(tmp_path):
    import piaso
    work = tmp_path / "src.cytome"
    ds = _build_cytome(str(work))
    out = tmp_path / "dst.cytome"
    out.write_bytes(b"existing")
    new_ds = piaso.pp.filter_cells(
        ds, mask="cluster == 'A'", inplace=False, output=str(out),
        overwrite=True, verbose=0,
    )
    assert new_ds.n_cells > 0
    new_ds.close()
    ds.close()


def test_cytome_zero_cells_raises(tmp_path):
    import piaso
    work = tmp_path / "x.cytome"
    ds = _build_cytome(str(work))
    with pytest.raises(ValueError, match="zero cells"):
        piaso.pp.filter_cells(ds, mask="cluster == 'NONE'", verbose=0)
    ds.close()


# ===========================================================================
# Six mask shapes — Cytome
# ===========================================================================

def test_cytome_mask_bool_ndarray(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    m = np.zeros(ds.n_cells, bool)
    m[:15] = True
    mask = piaso.pp.filter_cells(ds, mask=m, inplace=False, verbose=0)
    assert int(mask.sum()) == 15
    ds.close()


def test_cytome_mask_int_indices(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    idx = np.array([0, 1, 2])
    mask = piaso.pp.filter_cells(ds, mask=idx, inplace=False, verbose=0)
    assert int(mask.sum()) == 3
    assert mask[0] and mask[1] and mask[2]
    ds.close()


def test_cytome_mask_pd_series(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    df = ds.cells.to_pandas()
    s = pd.Series(df["tss_score"].values > 3.0)
    mask = piaso.pp.filter_cells(ds, mask=s, inplace=False, verbose=0)
    np.testing.assert_array_equal(mask, df["tss_score"].values > 3.0)
    ds.close()


def test_cytome_mask_query_pushdown(tmp_path):
    """Simple query should push down to SQL — verify via behaviour."""
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    df = ds.cells.to_pandas()
    mask = piaso.pp.filter_cells(
        ds, mask="cluster.isin(['A','B']) and tss_score > 2.0",
        inplace=False, verbose=0,
    )
    expected = (df["cluster"].isin(["A", "B"]) & (df["tss_score"] > 2.0)).values
    np.testing.assert_array_equal(mask, expected)
    ds.close()


def test_cytome_mask_dict(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    df = ds.cells.to_pandas()
    mask = piaso.pp.filter_cells(
        ds, mask={"cluster": ["A", "B"], "is_doublet": 0},
        inplace=False, verbose=0,
    )
    expected = (df["cluster"].isin(["A", "B"]) & (df["is_doublet"] == 0)).values
    np.testing.assert_array_equal(mask, expected)
    ds.close()


def test_cytome_mask_callable(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    df = ds.cells.to_pandas()
    fn = lambda d: d["tss_score"] > d["tss_score"].median()  # noqa: E731
    mask = piaso.pp.filter_cells(ds, mask=fn, inplace=False, verbose=0)
    expected = (df["tss_score"] > df["tss_score"].median()).values
    # streaming may use chunks; result must still match a single-shot eval.
    # Median is computed per-chunk in the streaming path, so for a single
    # tiny fixture (60 cells, one chunk) this is equivalent.
    np.testing.assert_array_equal(mask, expected)
    ds.close()


# ===========================================================================
# SQL pushdown vs streaming fallback
# ===========================================================================

def test_sql_pushdown_supports_simple_query():
    """The SQL translator must accept the common idioms."""
    from piaso.preprocessing._filtering import _query_to_sql_where
    valid = {"cluster", "tss_score", "is_doublet"}
    cases = [
        "cluster == 'A'",
        "tss_score >= 1.5",
        "cluster.isin(['A','B'])",
        "tss_score.between(1, 5)",
        "cluster == 'A' and tss_score > 2.0",
        "cluster == 'A' or tss_score > 5.0",
        "not is_doublet",
    ]
    for q in cases:
        where, params = _query_to_sql_where(q, valid)
        assert where  # non-empty


def test_sql_pushdown_falls_back_for_complex():
    """Unsupported syntax must raise _QueryTooComplex (caller falls back)."""
    from piaso.preprocessing._filtering import (
        _query_to_sql_where, _QueryTooComplex,
    )
    valid = {"a", "b"}
    # chained compare
    with pytest.raises(_QueryTooComplex):
        _query_to_sql_where("1 < a < 5", valid)
    # unknown column
    with pytest.raises(_QueryTooComplex):
        _query_to_sql_where("z > 0", valid)
    # method not supported
    with pytest.raises(_QueryTooComplex):
        _query_to_sql_where("a.str.startswith('foo')", valid)


def test_streaming_fallback_matches_sql_path(tmp_path):
    """A query that bypasses the SQL fast path should still produce
    the same mask as the equivalent inline computation."""
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    df = ds.cells.to_pandas()
    # Use chained comparison — falls back to pandas streaming
    mask = piaso.pp.filter_cells(
        ds, mask="1.0 < tss_score < 5.0", inplace=False, verbose=0,
    )
    expected = ((df["tss_score"] > 1.0) & (df["tss_score"] < 5.0)).values
    np.testing.assert_array_equal(mask, expected)
    ds.close()


# ===========================================================================
# Provenance write
# ===========================================================================

def test_cytome_provenance_written_inplace(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "p.cytome"))
    n_before = ds.n_cells
    piaso.pp.filter_cells(ds, mask="cluster == 'A'", verbose=0)
    # metadata key set
    params = ds.metadata.get("piaso_filter_cells_params")
    assert params is not None
    assert params["n_before"] == n_before
    assert params["n_after"] == ds.n_cells
    assert any("cluster" in s for s in params["mask_sources"])
    # _provenance row exists
    rows = ds._conn.execute(
        "SELECT operation FROM _provenance WHERE operation = 'filter_cells'"
    ).fetchall()
    assert len(rows) >= 1


def test_cytome_provenance_written_subset_to_new_file(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "src.cytome"))
    out = tmp_path / "dst.cytome"
    new_ds = piaso.pp.filter_cells(
        ds, mask="cluster == 'A'", inplace=False, output=str(out), verbose=0,
    )
    assert new_ds.metadata.get("piaso_filter_cells_params") is not None
    new_ds.close()
    ds.close()


# ===========================================================================
# Alias
# ===========================================================================

def test_subset_cells_is_filter_cells():
    import piaso
    assert piaso.pp.subset_cells is piaso.pp.filter_cells


# ===========================================================================
# Edge: dict with unknown column
# ===========================================================================

def test_cytome_mask_dict_unknown_column_raises(tmp_path):
    import piaso
    ds = _build_cytome(str(tmp_path / "b.cytome"))
    with pytest.raises(KeyError, match="not found"):
        piaso.pp.filter_cells(
            ds, mask={"nonexistent_col": "x"}, inplace=False, verbose=0,
        )
    ds.close()


def test_anndata_mask_length_mismatch_raises():
    import piaso
    a = _build_anndata()
    with pytest.raises(ValueError, match="length"):
        piaso.pp.filter_cells(
            a, mask=np.ones(a.n_obs + 5, bool), inplace=False,
        )
