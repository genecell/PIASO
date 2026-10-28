"""runSCALAR after the code audit of 2026-10-27 (part 3 minors; review I1's SCALAR minors).

- call_breadth divides by the sender-receiver pairs where the interaction was tested (it divided by all of them,
  so a floor that made a pair untestable in most types made it look specific);
- a misspelt layer raises (it fell back to .X without a word);
- layer=None reads the same matrix on a store as on its AnnData (the store read the counts, the AnnData .X);
- a ligand-receptor pair listed twice is tested once;
- verbose=False prints nothing.
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso

pytest.importorskip("cosg")


def _adata(n=300, g=200, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(["A", "B", "C"], n // 3)
    X = rng.poisson(0.3, (n, g)).astype(np.float32)
    for k, grp in enumerate("ABC"):
        X[groups == grp, 10 * k:10 * k + 6] += rng.poisson(8, ((groups == grp).sum(), 6))
    X[groups != "A", 0] = 0                      # G0 is detected in A only
    a = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"g": groups}, index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    a.layers["counts"] = a.X.copy()
    lognorm = a.X.multiply(1e4 / np.maximum(a.X.sum(1), 1)).tocsr()
    lognorm.data = np.log1p(lognorm.data)
    a.layers["lognorm"] = lognorm
    return a


def _spec(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return piaso.tl.specificity_matrix(a, groupby="g", cosg_layer="counts", verbose=0,
                                           cosg_remove_lowly_expressed=False)


LR = pd.DataFrame({"ligand": ["G0", "G0", "G10", "G2", "G21"], "receptor": ["G11", "G22", "G1", "G12", "G3"]})


def _scalar(data, spec, **kw):
    kw.setdefault("verbose", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return piaso.tl.runSCALAR(data, specificity_matrix=spec, lr_pairs=kw.pop("lr", LR), groupby="g",
                                  n_nearest_neighbors=20, **kw)


def test_call_breadth_counts_only_the_tested_groups():
    a = _adata()
    res = _scalar(a, _spec(a), layer="counts")
    assert not res.testable.all(), "the floor makes G0's pairs untestable outside A"
    for (lig, rec), grp in res.groupby(["ligand", "receptor"]):
        n_tested = int(grp.testable.sum())
        n_called = int(((grp.p_value_fdr < 0.05) & grp.testable).sum())
        want = n_called / n_tested if n_tested else np.nan
        np.testing.assert_allclose(grp.call_breadth.unique(), [want], equal_nan=True)
    # G0 -> G11 is tested only with sender A (3 of 9 groups) and called in A -> B: 1/3, not 1/9
    g = res[(res.ligand == "G0") & (res.receptor == "G11")]
    assert g.testable.sum() == 3 and ((g.p_value_fdr < 0.05) & g.testable).sum() == 1
    assert g.call_breadth.unique().tolist() == [pytest.approx(1 / 3)]


def test_a_misspelt_layer_raises():
    a = _adata()
    with pytest.raises(KeyError, match="layer='lognrom'"):
        _scalar(a, _spec(a), layer="lognrom")


def test_duplicate_pairs_are_tested_once():
    a = _adata()
    lr = pd.concat([LR, LR.iloc[[2]]], ignore_index=True)
    with pytest.warns(UserWarning, match="1 duplicated ligand-receptor pair"):
        res = piaso.tl.runSCALAR(a, specificity_matrix=_spec(a), lr_pairs=lr, groupby="g", layer="counts",
                                 n_nearest_neighbors=20, verbose=False)
    assert not res.duplicated(["ligand", "receptor", "sender", "receiver"]).any()
    ref = _scalar(a, _spec(a), layer="counts")
    pd.testing.assert_frame_equal(res.reset_index(drop=True), ref.reset_index(drop=True))


def test_verbose_false_prints_nothing(capsys):
    a = _adata()
    _scalar(a, _spec(a), layer="counts", verbose=False)
    out = capsys.readouterr()
    assert out.out == "" and "Processing" not in out.err


def test_layer_none_reads_the_same_matrix_on_a_store(tmp_path):
    cytome = pytest.importorskip("cytome")
    a = _adata()
    a.X = a.layers["lognorm"].copy()                  # .X normalised, raw counts in a layer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cytome.from_anndata(a, output=str(tmp_path / "s.cytome"), counts_layer="counts").close()
    spec = _spec(a)
    mem = _scalar(a, spec, layer=None)
    ds = cytome.open(str(tmp_path / "s.cytome"))
    try:
        store = _scalar(ds, spec, layer=None)
    finally:
        ds.close()
    cols = ["ligand", "receptor", "sender", "receiver", "p_value", "null_matchability"]
    key = ["ligand", "receptor", "sender", "receiver"]
    pd.testing.assert_frame_equal(mem[cols].sort_values(key).reset_index(drop=True),
                                  store[cols].sort_values(key).reset_index(drop=True), rtol=1e-6)
