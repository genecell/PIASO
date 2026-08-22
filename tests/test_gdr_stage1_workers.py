"""Stage 1 can run batches concurrently, but only under a thread-safe numba.

An earlier attempt aborted the whole process with "Numba workqueue threading
layer is terminating: Concurrent access has been detected". The cause was
recorded as Leiden entering numba; Leiden is igraph, the numba is pynndescent
inside neighbors, and the threading layer was ours to pick. So the guard is on
the layer, and requesting workers under an unsafe one must degrade to serial
with a warning rather than kill the caller's process.
"""
import os

import numpy as np
import pytest

from piaso.tools._runGDR import _numba_threading_is_safe


def test_reports_the_resolved_layer_not_the_request(monkeypatch):
    """The probe must read what numba ended up with, not what we asked for.

    Reading NUMBA_THREADING_LAYER alone core-dumped a run: pynndescent rewrites
    the setting at import, turning a request for 'omp' into 'tbb' when tbb is
    installed and into 'workqueue' when it is not.
    """
    import numba
    monkeypatch.setattr(numba.config, "THREADING_LAYER", "workqueue", raising=False)
    assert _numba_threading_is_safe() == (False, "workqueue")

    monkeypatch.setattr(numba.config, "THREADING_LAYER", "tbb", raising=False)
    assert _numba_threading_is_safe() == (True, "tbb")


def test_unknown_layer_is_unsafe(monkeypatch):
    import numba
    monkeypatch.setattr(numba.config, "THREADING_LAYER", "something_else", raising=False)
    assert _numba_threading_is_safe() == (False, "something_else")


def test_blank_layer_is_unsafe(monkeypatch):
    """Never read an empty/missing value as permission to use threads."""
    import numba
    monkeypatch.setattr(numba.config, "THREADING_LAYER", "", raising=False)
    assert _numba_threading_is_safe() == (False, "unknown")


def test_workers_under_unsafe_layer_warn_and_fall_back(monkeypatch, tmp_path):
    """The whole point: never abort the caller's process."""
    anndata = pytest.importorskip("anndata")
    cytome = pytest.importorskip("cytome")
    import scipy.sparse as sp
    import piaso

    import numba
    monkeypatch.setattr(numba.config, "THREADING_LAYER", "workqueue", raising=False)
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(120, 40)).astype(np.float32))
    a = anndata.AnnData(X=X)
    a.obs["batch"] = ["a"] * 60 + ["b"] * 60
    d = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    piaso.tl.infog(d, n_top_genes=20, save_layer=True)

    with pytest.warns(RuntimeWarning, match="not thread-safe"):
        piaso.tl.runGDR(d, batch_key="batch", groupby=None, n_gene=5,
                        layer="infog", key_added="X_gdr", verbosity=0,
                        stage1_workers=4)
    assert "X_gdr" in list(d.embeddings.keys())
    d.close()


def test_tbb_is_optional_not_required():
    """tbb must NOT be a hard dependency.

    Measured 2026-08-22 by blocking numba's tbbpool import: with
    NUMBA_THREADING_LAYER unset, numba's 'default' resolves to omp -- which is
    thread-safe and ships in numba's own wheel -- so the per-batch pool runs
    with nothing installed. The serial fallback only happens when omp is asked
    for EXPLICITLY, because pynndescent then rewrites it to workqueue.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    deps = text.split("[project.optional-dependencies]")[0]
    assert '"tbb' not in deps, "tbb must not be a required dependency"

    extras = text.split("[project.optional-dependencies]")[1]
    dep = re.search(r'^\s*"tbb[^"]*",\s*$', extras, re.M)
    assert dep is not None, "tbb should still be offered as an extra"

    from packaging.requirements import Requirement

    req = Requirement(dep.group(0).strip().strip(",").strip('"'))
    assert req.name == "tbb"
    # No arm64/macOS wheels, so even the extra must be marker-gated.
    assert req.marker is not None
    assert req.marker.evaluate({"platform_machine": "x86_64", "sys_platform": "linux"})
    assert not req.marker.evaluate({"platform_machine": "arm64", "sys_platform": "darwin"})


def test_unsafe_layer_warning_leads_with_the_free_fix(monkeypatch):
    """The message must name the actual trigger, and the fix that costs nothing.

    The previous version told users to set NUMBA_THREADING_LAYER=omp, which is
    the one setting that CAUSES the serial fallback.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "piaso" / "tools" / "_runGDR.py").read_text()
    start = src.index("stage1_workers={_n_workers} requested")
    msg = src[max(0, start - 1200):start + 900]
    assert "unset NUMBA_THREADING_LAYER" in msg
    assert "piaso[tbb]" in msg
    assert "numba issue #3341" in msg
    # and the old, wrong advice is gone
    assert "Set NUMBA_THREADING_LAYER=omp" not in src
