"""``pl.dotplot`` must order groups the way the data declares them.

The dotplot used to build its row order with ``sorted(..., key=str)``, which
turns Leiden's 0,1,2,…,10,11 into the lexical 0,1,10,11,…,2. ``tl.leiden``
stores a numeric category order precisely to avoid that, and ``pl.embedding``
honours it, so the two plots of the same column disagreed.
"""

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
import scipy.sparse as sp

from piaso.plotting._plotDotplot import _get_expression_data_anndata, _ordered_groups


def _toy(n_groups=12, n_cells=240, n_genes=6):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, (n_cells, n_genes)).astype("float32"))
    a = anndata.AnnData(X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    labels = np.array([str(i % n_groups) for i in range(n_cells)])
    numeric = [str(i) for i in range(n_groups)]
    a.obs["leiden"] = pd.Categorical(labels, categories=numeric)
    return a, numeric


def test_numeric_category_order_survives_the_dotplot():
    a, numeric = _toy()
    # The bug: lexical order would put "10" between "1" and "2".
    assert sorted(numeric, key=str) != numeric, "toy must expose the lexical trap"

    frac, mean = _get_expression_data_anndata(a, ["g0", "g1"], "leiden")
    assert list(frac.index) == numeric
    assert list(mean.index) == numeric


def test_falls_back_to_string_sort_without_a_declared_order():
    a, _ = _toy()
    a.obs["plain"] = a.obs["leiden"].astype(str)      # not categorical
    frac, _ = _get_expression_data_anndata(a, ["g0"], "plain")
    assert list(frac.index) == sorted(frac.index, key=str)


def test_levels_absent_from_the_declared_order_are_appended_not_dropped():
    a, numeric = _toy()
    # Declare fewer categories than are present: pandas would make the rest NaN,
    # so append directly on the raw array instead.
    a.obs["mixed"] = np.where(np.arange(a.n_obs) < 10, "extra", a.obs["leiden"].astype(str))
    a.obs["mixed"] = pd.Categorical(a.obs["mixed"], categories=numeric + ["extra"])
    frac, _ = _get_expression_data_anndata(a, ["g0"], "mixed")
    assert list(frac.index)[-1] == "extra"
    assert set(frac.index) == set(a.obs["mixed"].dropna().unique())


def test_ordered_groups_keeps_unlisted_levels():
    a, numeric = _toy(n_groups=4, n_cells=40)
    present = np.array(["0", "1", "zz", "2", "3"], dtype=object)
    out = _ordered_groups(a, "leiden", present)
    assert out[:4] == ["0", "1", "2", "3"]
    assert out[-1] == "zz"
