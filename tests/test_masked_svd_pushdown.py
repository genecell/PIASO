"""cell_mask is pushed into iter_chunks, so it must not change the SVD result.

Before this, runSVD masked rows *after* the storage layer had handed them
over, which meant a per-batch GDR decompressed the whole matrix once per
batch per power iteration. Pushing the mask down changes which chunks are
read; it must not change what comes out.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import piaso
from piaso.tools._runSVD import _mask_to_indices

anndata = pytest.importorskip("anndata")
cytome = pytest.importorskip("cytome")

N_CELLS, N_GENES = 300, 40


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.5, size=(N_CELLS, N_GENES)).astype(np.float32))
    a = anndata.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(N_CELLS)]
    a.var_names = [f"g{j}" for j in range(N_GENES)]
    d = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    yield d, X
    d.close()


def _svd(d, mask):
    from piaso.tools._runSVD import _runSVD_streaming
    return _runSVD_streaming(
        d, use_highly_variable=False, n_components=5, n_iter=2,
        random_state=0, layer="counts", cell_mask=mask,
        return_svd=False, verbosity=0,
    )


def test_masked_svd_shape_and_order(ds):
    d, _ = ds
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[50:130] = True
    out = _svd(d, keep)
    emb = out[0] if isinstance(out, tuple) else out
    assert emb.shape == (80, 5)


def test_masked_svd_matches_unmasked_rows(ds, tmp_path):
    """Rows selected by a mask must equal the same rows of a full-matrix SVD.

    Not bit-identical — randomised SVD sees a different matrix — so compare
    the subspace: |cosine| between matched components.
    """
    d, X = ds
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[100:200] = True
    masked = _svd(d, keep)
    masked = masked[0] if isinstance(masked, tuple) else masked
    tmp_sub = tmp_path / "sub.cytome"

    from piaso.tools._runSVD import _runSVD_streaming
    sub = cytome.from_anndata(anndata.AnnData(X=sp.csr_matrix(X[keep])),
                              output=str(tmp_sub))
    # all-True mask: the reference cytome holds only the kept rows, and the
    # mask keeps runSVD on the return-the-embedding path rather than the
    # write-to-cytome one.
    ref = _runSVD_streaming(
        sub, use_highly_variable=False, n_components=5, n_iter=2,
        random_state=0, layer="counts",
        cell_mask=np.ones(int(keep.sum()), dtype=bool), verbosity=0,
    )
    sub.close()
    ref = ref[0] if isinstance(ref, tuple) else ref
    for k in range(5):
        c = abs(np.corrcoef(masked[:, k], ref[:, k])[0, 1])
        assert c > 0.99, f"component {k} diverged: |r|={c:.4f}"


def test_scattered_mask_still_correct(ds):
    d, _ = ds
    rng = np.random.default_rng(3)
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[rng.choice(N_CELLS, 90, replace=False)] = True
    out = _svd(d, keep)
    emb = out[0] if isinstance(out, tuple) else out
    assert emb.shape == (90, 5)
    assert np.isfinite(emb).all()


@pytest.mark.parametrize("mask,expected", [
    (np.array([True, False, True, False]), [0, 2]),
    (np.array([0, 2]), [0, 2]),
    (np.array([True, False, True]), [0, 2]),          # short: padded with False
    (np.array([True] * 6), [0, 1, 2, 3]),             # long: truncated
    (np.array([0, 2, 9]), [0, 2]),                    # out-of-range dropped
])
def test_mask_to_indices(mask, expected):
    np.testing.assert_array_equal(_mask_to_indices(mask, 4), expected)
