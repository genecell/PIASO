"""GDR stage 1 caches a batch's chunks, but only when it fits.

Randomised SVD makes 3 + 2*n_iter passes and each one re-read the batch from
disk. Caching turned the per-batch SVD from 13.6 s to 3.9 s on a 5-batch
cytome with no change in peak RSS. Unconditional caching would break the
memory bound the format exists to provide, so it is gated on an estimate.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from piaso.tools._runGDR import _estimate_batch_cache_bytes

anndata = pytest.importorskip("anndata")
cytome = pytest.importorskip("cytome")

N_CELLS, N_GENES = 400, 30


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.5, size=(N_CELLS, N_GENES)).astype(np.float32))
    a = anndata.AnnData(X=X)
    d = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    yield d, X
    d.close()


def test_estimate_scales_with_batch_size(ds):
    d, X = ds
    small = _estimate_batch_cache_bytes(d, "RNA", "counts", 10)
    big = _estimate_batch_cache_bytes(d, "RNA", "counts", 100)
    assert small is not None and big is not None
    assert big > small
    assert big == pytest.approx(small * 10, rel=0.01)


def test_estimate_overshoots_never_undershoots(ds):
    """The guard must err high: caching more than predicted breaks the bound."""
    d, X = ds
    est = _estimate_batch_cache_bytes(d, "RNA", "counts", N_CELLS)
    actual = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    assert est >= actual, f"estimate {est} under-counts actual {actual}"


def test_estimate_returns_none_for_unknown_layer(ds):
    d, _ = ds
    assert _estimate_batch_cache_bytes(d, "RNA", "no_such_layer", 100) is None


def test_estimate_returns_none_for_empty_matrix(ds):
    """A zero-row matrix must not divide by zero; the caller then streams.

    (n_nonzero itself is NOT NULL in the current schema, so the None-nnz branch
    is unreachable here — it stays as a guard for older files.)"""
    d, _ = ds
    d._conn.execute(
        "INSERT INTO matrix_meta (matrix_name, n_rows, n_cols, n_nonzero, dtype,"
        " row_entity, col_entity, chunk_size, n_chunks) "
        "VALUES ('RNA_empty', 0, 30, 0, 'float32', 'cells', 'genes', 100, 0)")
    assert _estimate_batch_cache_bytes(d, "RNA", "empty", 100) is None


@pytest.mark.parametrize("n_batch,budget,should_cache", [
    (10, 10 ** 9, True),          # tiny batch, big budget
    (400, 1, False),              # any batch, no budget
])
def test_budget_gate(ds, n_batch, budget, should_cache):
    """The exact gate the stage-1 loop applies."""
    d, _ = ds
    est = _estimate_batch_cache_bytes(d, "RNA", "counts", n_batch)
    assert (est is not None and est <= budget) is should_cache


def test_batch_too_big_for_the_default_budget_streams(ds):
    """Past the budget the gate must flip to streaming.

    The cell count is derived from this fixture's own density (~7.5 nnz/cell,
    far sparser than real data) rather than hard-coded, so the test asserts the
    gate's behaviour and not the toy's dimensions.
    """
    d, _ = ds
    budget = 512 * 1024 ** 2
    per_cell = _estimate_batch_cache_bytes(d, "RNA", "counts", 1_000_000) / 1_000_000
    n_over = int(budget / per_cell) + 1_000
    est = _estimate_batch_cache_bytes(d, "RNA", "counts", n_over)
    assert est > budget
    assert not (est is not None and est <= budget)


def test_gate_is_off_when_estimate_unavailable(ds):
    """No estimate must mean stream, never mean cache."""
    d, _ = ds
    est = _estimate_batch_cache_bytes(d, "RNA", "nope", 100)
    assert not (est is not None and est <= 10 ** 12)
