"""INFOG must refuse input that is not raw counts, with a documented escape.

INFOG models count dispersion. Handed normalized, log-transformed or scaled
values it returns numbers that look fine and mean nothing -- a silent failure
that cost one of our own benchmarks 0.10 ARI. So it errors, and the error names
the fixes.
"""
import inspect

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import piaso
from piaso.tools._normalization import _check_integer_counts, _sample_stored_values


def _counts(n=200, g=50, seed=0):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32))


def _adata(X):
    a = ad.AnnData(X=X.copy())
    a.var_names = [f"g{j}" for j in range(X.shape[1])]
    a.layers["counts"] = _counts(X.shape[0], X.shape[1])
    return a


# ---------------------------------------------------------------- the sampler

def test_sampler_reads_nonzeros_without_densifying(monkeypatch):
    X = _counts(500, 400)
    monkeypatch.setattr(
        type(X), "toarray",
        lambda self: pytest.fail("guard densified the matrix"),
    )
    vals = _sample_stored_values(X, n=100)
    assert vals.size == 100
    assert np.all(vals > 0)  # sparse sampling sees stored values only


def test_sampler_handles_dense_and_empty():
    assert _sample_stored_values(np.arange(12).reshape(3, 4), n=5).size == 5
    assert _sample_stored_values(sp.csr_matrix((4, 4))).size == 0


def test_sampler_is_deterministic_and_leaves_the_global_rng_alone():
    X = _counts(500, 400)
    np.random.seed(1234)
    before = np.random.rand()
    np.random.seed(1234)
    a = _sample_stored_values(X, n=50)
    b = _sample_stored_values(X, n=50)
    assert np.array_equal(a, b)
    assert np.random.rand() == before, "guard perturbed the global RNG stream"


# ------------------------------------------------------------------ the guard

def test_guard_accepts_integer_valued_floats():
    _check_integer_counts(_counts(), layer=None, allow_non_integer=False)


def test_guard_rejects_normalized_values():
    with pytest.raises(ValueError, match="not raw UMI counts"):
        _check_integer_counts(_counts().multiply(1 / 3.0).tocsr(),
                              layer=None, allow_non_integer=False)


def test_guard_message_names_every_way_out():
    with pytest.raises(ValueError) as exc:
        _check_integer_counts(_counts().multiply(1 / 3.0).tocsr(),
                              layer=None, allow_non_integer=False)
    msg = str(exc.value)
    for hint in ("layer='counts'", "adata.raw.to_adata()", "infog_layer='counts'",
                 "allow_non_integer=True", "Smart-seq2"):
        assert hint in msg, hint


@pytest.mark.parametrize("kwargs", [
    {"allow_non_integer": True},
    {"layer": "counts"},
])
def test_guard_is_silent_when_the_user_has_answered(kwargs):
    """Naming a layer is a deliberate act; so is the escape flag."""
    _check_integer_counts(_counts().multiply(1 / 3.0).tocsr(),
                          **{"layer": None, "allow_non_integer": False, **kwargs})


# ------------------------------------------------------------- wired into API

def test_infog_in_memory_path_is_guarded():
    a = _adata(_counts().multiply(1 / 3.0).tocsr())
    with pytest.raises(ValueError, match="not raw UMI counts"):
        piaso.tl.infog(a, n_top_genes=10, verbosity=0)
    piaso.tl.infog(a, n_top_genes=10, verbosity=0, allow_non_integer=True)
    piaso.tl.infog(a, n_top_genes=10, verbosity=0, layer="counts")


def test_infog_streaming_path_is_guarded():
    a = _adata(_counts().multiply(1 / 3.0).tocsr())
    with pytest.raises(ValueError, match="not raw UMI counts"):
        piaso.tl.infog(a, streaming=True, n_top_genes=10, verbosity=0)
    piaso.tl.infog(a, streaming=True, n_top_genes=10, verbosity=0,
                   allow_non_integer=True)


def test_infog_svd_forwards_the_escape():
    a = _adata(_counts().multiply(1 / 3.0).tocsr())
    with pytest.raises(ValueError, match="not raw UMI counts"):
        piaso.tl.infog_svd(a, layer="infog", n_components=5, n_top_genes=10)
    piaso.tl.infog_svd(a, layer="infog", n_components=5, n_top_genes=10,
                       allow_non_integer=True)


def test_escape_is_declared_all_the_way_down():
    from piaso.tools._runGDR import (
        _runCOSGParallel_single_batch,
        _runGDRParallel_cytome,
        runCOSGParallel,
    )
    from piaso.tools._runSVD import _runSVDLazy_original, _runSVDLazy_streaming

    for fn in (piaso.tl.infog, piaso.tl.infog_svd, piaso.tl.runGDR,
               _runSVDLazy_original, _runSVDLazy_streaming, runCOSGParallel,
               _runCOSGParallel_single_batch, _runGDRParallel_cytome):
        assert "allow_non_integer" in inspect.signature(fn).parameters, fn.__name__
