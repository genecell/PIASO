"""expressed_pct must reach COSG on the AnnData path of runGDR/runCOSGParallel.

The parameter was added to runGDR's signature for the runGDRParallel shim but was
only wired into the cytome dispatch; the AnnData COSG calls hardcoded 0.1.
"""
import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso
from piaso.tools._runGDR import (
    _runCOSGParallel_single_batch,
    runCOSGParallel,
    runGDR,
)


def _half_penetrant_markers(seed=0, n_cells=600, n_genes=60, n_groups=3):
    """Markers that are perfectly specific but expressed in only half of their group.

    They survive expressed_pct=0.05 and are filtered out at expressed_pct=0.8, so any
    call that hardcodes 0.1 produces the same answer for both settings.
    """
    rng = np.random.default_rng(seed)
    grp = np.repeat(np.arange(n_groups), n_cells // n_groups)
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)
    for k in range(n_groups):
        idx = np.where(grp == k)[0]
        cols = np.arange(k * 5, (k + 1) * 5)
        X[np.ix_(idx, cols)] = 0.0
        X[np.ix_(idx[: len(idx) // 2], cols)] = 50.0
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.var_names = [f"g{j}" for j in range(n_genes)]
    a.obs["ct"] = pd.Categorical([f"c{k}" for k in grp])
    a.obs["batch"] = pd.Categorical(np.array(["b0", "b1"])[np.arange(n_cells) % 2])
    return a


@pytest.mark.parametrize("max_workers", [1, 2])
def test_runGDR_anndata_honours_expressed_pct(max_workers):
    out = {}
    for ep in (0.05, 0.8):
        a = _half_penetrant_markers()
        runGDR(
            a, batch_key="batch", groupby="ct", n_gene=3, layer=None,
            key_added="X_gdr", max_workers=max_workers, verbosity=0,
            expressed_pct=ep,
        )
        out[ep] = np.asarray(a.obsm["X_gdr"]).copy()
    assert out[0.05].shape[0] == out[0.8].shape[0]
    same = (
        out[0.05].shape == out[0.8].shape
        and np.allclose(out[0.05], out[0.8])
    )
    assert not same, "expressed_pct did not reach COSG (result identical for 0.05 vs 0.8)"


def test_runCOSGParallel_honours_expressed_pct():
    genes = {}
    for ep in (0.05, 0.8):
        a = _half_penetrant_markers()
        markers, _ = runCOSGParallel(
            a, batch_key="batch", groupby="ct", n_gene=3, layer=None,
            return_gene_names=True, max_workers=2, verbosity=0, expressed_pct=ep,
        )
        genes[ep] = set(np.asarray(markers).ravel().tolist())
    assert genes[0.05] != genes[0.8]


def test_expressed_pct_is_declared_all_the_way_down():
    for fn in (runGDR, runCOSGParallel, _runCOSGParallel_single_batch):
        assert "expressed_pct" in inspect.signature(fn).parameters, fn.__name__
