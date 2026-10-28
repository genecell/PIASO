"""dotplot: expression_cutoff counts, and the size key draws the dots' sizes.

Regressions for the audit of 2026-10-27 (review I2, findings 9 and 10):

* ``expression_cutoff`` was accepted and ignored; "expressing" was hard-coded
  to ``> 0``, so with a cutoff of 1.0 a group where 20 % of cells exceed it
  was drawn at 100 %;
* the size key's dots were capped at 130 while data dots reach
  ``size_scale`` (200): the 80 % and 100 % key dots were identical and both
  smaller than a 100 % data dot.
"""
import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso
from piaso.plotting._plotDotplot import _aggregate_by_group, _group_indicator


def _adata():
    # group a: every cell expresses, 20 % above 1.0; group b: half express
    a_vals = np.array([0.5] * 8 + [2.0] * 2)
    b_vals = np.array([0.0] * 5 + [0.5] * 5)
    X = np.concatenate([a_vals, b_vals])[:, None].astype(np.float32)
    obs = pd.DataFrame({"grp": pd.Categorical(["a"] * 10 + ["b"] * 10)},
                       index=[f"c{i}" for i in range(20)])
    return ad.AnnData(sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=["g"]))


@pytest.mark.parametrize("sparse", [True, False])
def test_aggregate_counts_above_the_cutoff(sparse):
    a = _adata()
    groups = ["a", "b"]
    ind = _group_indicator(a.obs["grp"].values, groups)
    sizes = np.asarray(ind.sum(axis=1)).ravel()
    X = a.X if sparse else a.X.toarray()
    frac, mean = _aggregate_by_group(X, ind, sizes, False, cutoff=1.0)
    np.testing.assert_allclose(frac[:, 0], [0.2, 0.0])
    frac0, _ = _aggregate_by_group(X, ind, sizes, False, cutoff=0.0)
    np.testing.assert_allclose(frac0[:, 0], [1.0, 0.5])
    # mean over expressing cells only: the cells above the cutoff
    _, mean_e = _aggregate_by_group(X, ind, sizes, True, cutoff=1.0)
    np.testing.assert_allclose(mean_e[:, 0], [2.0, 0.0])


@pytest.mark.parametrize("sparse", [True, False])
def test_negative_cutoff_counts_zeros_as_expressing(sparse):
    a = _adata()
    groups = ["a", "b"]
    ind = _group_indicator(a.obs["grp"].values, groups)
    sizes = np.asarray(ind.sum(axis=1)).ravel()
    X = a.X if sparse else a.X.toarray()
    frac, _ = _aggregate_by_group(X, ind, sizes, False, cutoff=-0.1)
    np.testing.assert_allclose(frac[:, 0], [1.0, 1.0])


def test_dotplot_draws_the_fraction_above_the_cutoff():
    a = _adata()
    plt.close("all")
    fig, ax = piaso.pl.dotplot(a, ["g"], groupby="grp", expression_cutoff=1.0,
                               dot_max=1.0, show=False, return_fig=True)
    sizes = ax.collections[0].get_sizes()
    # rows: a then b; fraction 0.2 -> 0.2 * size_scale, fraction 0 -> no dot
    np.testing.assert_allclose(sizes, [0.2 * 200, 0.0])
    plt.close(fig)


def test_size_key_matches_the_data_dots():
    a = _adata()
    plt.close("all")
    fig, ax = piaso.pl.dotplot(a, ["g"], groupby="grp", dot_max=1.0, show=False,
                               return_fig=True)
    leg = [lg for x in fig.axes for lg in [x.get_legend()] if lg is not None][0]
    key = {t.get_text(): float(h.get_sizes()[0])
           for h, t in zip(leg.legend_handles, leg.get_texts())}
    assert key["100%"] == pytest.approx(200.0)          # size_scale, not capped
    assert key["80%"] == pytest.approx(160.0)
    assert len(set(key.values())) == len(key)            # no two key dots alike
    plt.close(fig)
