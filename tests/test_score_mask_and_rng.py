"""score() must skip chunks it does not need, and not use the global RNG.

Two independent defects, both found by measuring rather than reading:

1. Neither of score()'s streaming passes passed cell_mask to iter_chunks, so
   scoring one batch read the whole matrix and discarded it -- 24.31 s against
   0.37 s on one ADVIS batch, 196 chunks where 6 hold the rows, and stage 3
   does that 70 times on a 35-batch run.

2. Control genes were drawn after np.random.seed(), which is process-global.
   Two threads scoring different batches interleaved their draws, and four
   concurrent stage-3 workers moved the embedding by 0.41 while two happened to
   agree.
"""
import numpy as np
import pandas as pd
import pytest
from concurrent.futures import ThreadPoolExecutor

import piaso.tools._normalization as N

anndata = pytest.importorskip("anndata")
cytome = pytest.importorskip("cytome")

N_CELLS, N_GENES = 600, 80


@pytest.fixture
def ds(tmp_path):
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.5, size=(N_CELLS, N_GENES)).astype(np.float32))
    a = anndata.AnnData(X=X)
    a.var_names = [f"g{j}" for j in range(N_GENES)]
    d = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    import piaso
    piaso.tl.infog(d, n_top_genes=40, save_layer=True)
    yield d
    d.close()


def _sets(ds, n=6):
    names = np.asarray(ds.genes["gene_id"])
    rng = np.random.default_rng(1)
    return pd.DataFrame({f"s{i}": rng.choice(names, 8, replace=False) for i in range(n)})


def test_masked_score_reads_only_needed_chunks(ds):
    """The mask must reach the storage layer, not be applied after it."""
    import cytome.io.chunked_io as cio
    gs = _sets(ds)
    mask = np.zeros(N_CELLS, dtype=bool)
    mask[: N_CELLS // 6] = True

    n = 0
    real = cio.decompress_blob

    def counting(b, c):
        nonlocal n
        n += 1
        return real(b, c)

    cio.decompress_blob = counting
    try:
        N.score(ds, gene_list=gs, layer="infog", modality="RNA",
                cell_mask=mask, max_workers=1, batch_size=64, verbosity=0)
    finally:
        cio.decompress_blob = real

    spans = ds._conn.execute(
        "SELECT row_start, row_end FROM matrix_chunks WHERE matrix_name='RNA_infog'"
    ).fetchall()
    keep = np.flatnonzero(mask)
    needed = sum(1 for rs, re_ in spans
                 if np.searchsorted(keep, rs, "left") < np.searchsorted(keep, re_, "left"))
    # 3 blobs per chunk, two passes over the data
    assert n // 3 <= needed * 2 + 2, (
        f"decompressed {n//3} chunks for a mask needing {needed} (x2 passes)")
    assert needed < len(spans), "fixture must have skippable chunks to be meaningful"


def test_masked_score_rows_match_full_then_slice_shape(ds):
    gs = _sets(ds)
    mask = np.zeros(N_CELLS, dtype=bool)
    mask[100:250] = True
    sm, _, _ = N.score(ds, gene_list=gs, layer="infog", modality="RNA",
                       cell_mask=mask, max_workers=1, batch_size=64, verbosity=0)
    assert sm.shape == (int(mask.sum()), gs.shape[1])
    assert np.isfinite(sm).all()


def test_concurrent_scoring_is_reproducible(ds):
    """The regression: the same batches scored serially and concurrently."""
    gs = _sets(ds)
    masks = []
    for i in range(4):
        m = np.zeros(N_CELLS, dtype=bool)
        m[i * 150:(i + 1) * 150] = True
        masks.append(m)

    path = str(ds.path)

    def one(m):
        # Its OWN Dataset per call: sqlite3 connections cannot cross threads,
        # which is exactly why stage 3 opens one per worker. (score() itself
        # does not accept a path the way runGDR does.)
        d = N._open_cytome(path)
        try:
            return N.score(d, gene_list=gs, layer="infog", modality="RNA",
                           cell_mask=m, max_workers=1, batch_size=64,
                           verbosity=0)[0]
        finally:
            d.close()

    serial = [one(m) for m in masks]
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = list(pool.map(one, masks))
    for i, (a, b) in enumerate(zip(serial, threaded)):
        assert np.array_equal(a, b), f"batch {i} differed under concurrency"


def test_score_does_not_disturb_the_global_rng(ds):
    """Drawing control genes must not move the caller's np.random stream."""
    gs = _sets(ds)
    np.random.seed(12345)
    before = np.random.randint(0, 1_000_000, size=5)
    np.random.seed(12345)
    N.score(ds, gene_list=gs, layer="infog", modality="RNA",
            max_workers=1, batch_size=64, verbosity=0)
    after = np.random.randint(0, 1_000_000, size=5)
    np.testing.assert_array_equal(before, after)


def test_query_scores_match_the_per_set_loop(ds):
    """The vectorised query scores must equal the loop they replaced.

    score() computed each gene set's per-cell query score by column-subsetting
    the chunk once per set, which is O(nnz) per set and grew linearly in
    n_sets: 1.95 / 6.45 / 24.41 s at 20 / 80 / 320 sets against 0.55 / 0.62 /
    0.78 s for the same thing as one sparse matmul.
    """
    import scipy.sparse as sp
    from piaso.tools._normalization import _gene_set_weight_matrix

    rng = np.random.default_rng(2)
    n_genes = N_GENES
    idxs = [np.sort(rng.choice(n_genes, 7, replace=False)) for _ in range(9)]
    ws = [rng.random(7) for _ in idxs]
    W = _gene_set_weight_matrix(idxs, ws, n_genes)
    assert W.shape == (n_genes, len(idxs))

    X = sp.csr_matrix(rng.random((40, n_genes)))
    loop = np.column_stack([
        np.ravel(X[:, gi].multiply(w).sum(axis=1)) for gi, w in zip(idxs, ws)
    ])
    np.testing.assert_allclose((X @ W).toarray(), loop, rtol=0, atol=0)


def test_weight_matrix_sums_repeated_genes(ds):
    """A gene listed twice in one set contributed twice under the old loop."""
    from piaso.tools._normalization import _gene_set_weight_matrix
    W = _gene_set_weight_matrix([np.array([3, 3, 5])], [np.array([1.0, 2.0, 4.0])], 10)
    assert W[3, 0] == pytest.approx(3.0)   # 1.0 + 2.0
    assert W[5, 0] == pytest.approx(4.0)


def test_empty_gene_set_list_is_handled(ds):
    from piaso.tools._normalization import _gene_set_weight_matrix
    W = _gene_set_weight_matrix([], [], 10)
    assert W.shape == (10, 0)
