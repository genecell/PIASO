"""The SVD engine on chunks handed over by the dataset, against the engine
reading the store itself: identical to the bit, in both layouts, every
batching, any thread count, with masks; and the chunk cache."""
import threading

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cytome
from piaso.tools._rsvd_core import EngineOps, rsvd_core

try:
    from piaso import _piaso
    HAVE = hasattr(_piaso, "svd_pass_source")
except Exception:  # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="Rust extension without svd_pass_source")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    rng = np.random.default_rng(11)
    n, g = 1500, 3000
    X = sp.random(n, g, density=0.04, format="csr", random_state=2, dtype=np.float32)
    X.data = rng.poisson(2, X.nnz).astype(np.float32) + 1
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = [f"chr1:{1000 * j}-{1000 * j + 500}" for j in range(g)]
    p = tmp_path_factory.mktemp("src") / "m.cytome"
    ds = cytome.from_anndata(A, modality="ATAC", output=str(p))
    ds.ATAC.counts.build_feature_index()           # both layouts on disk
    ds.close()
    depth = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    idf = np.log1p(n / np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)).astype(np.float32)
    return str(p), X, depth, idf


def _args(X, depth, idf, cols, rows):
    n, g = X.shape
    remap = np.full(g, -1, dtype=np.int32)
    remap[cols] = np.arange(len(cols), dtype=np.int32)
    if rows is None:
        row_map, n_out = None, n
    else:
        row_map = np.full(n, -1, dtype=np.int32)
        row_map[rows] = np.arange(len(rows), dtype=np.int32)
        n_out = len(rows)
    return remap, len(cols), row_map, n_out


@pytest.mark.parametrize("layout", ["csr", "csc"])
@pytest.mark.parametrize("mode", ["xb", "xty"])
@pytest.mark.parametrize("masked", [False, True])
def test_handed_over_chunks_give_the_same_bits(store, layout, mode, masked):
    path, X, depth, idf = store
    rng = np.random.default_rng(5)
    cols = np.sort(rng.choice(X.shape[1], 2000, replace=False)) if masked else np.arange(X.shape[1])
    rows = np.sort(rng.choice(X.shape[0], 900, replace=False)) if masked else None
    remap, n_sel, row_map, n_out = _args(X, depth, idf, cols, rows)
    dense = rng.standard_normal((n_sel if mode == "xb" else n_out, 7))
    ref = _piaso.svd_pass(path, "ATAC_counts", mode, dense, remap, n_sel, row_map, n_out,
                          depth, idf, 1e4, n_threads=3, layout=layout)
    ds = cytome.open(path)
    for threads, batch_bytes in ((1, 0), (4, 0), (4, 16 * 5000), (7, 16 * 777)):
        with ds.chunk_source("ATAC_counts", n_readers=3) as src:
            got = _piaso.svd_pass_source(src, mode, dense, remap, n_sel, row_map, n_out,
                                         depth, idf, 1e4, n_threads=threads,
                                         batch_bytes=batch_bytes, layout=layout)
        assert np.array_equal(got, ref), (threads, batch_bytes)
    ds.close()


def test_the_whole_svd_is_identical(store):
    path, X, depth, idf = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    a = EngineOps(path, "ATAC_counts", *X.shape, tfidf=tf, n_threads=4)
    ds = cytome.open(path)
    with ds.chunk_source("ATAC_counts") as src:
        b = EngineOps(path, "ATAC_counts", *X.shape, tfidf=tf, n_threads=4, source=src)
        assert b.layout == a.layout == "csc"
        ea, sa, va, _ = rsvd_core(a, n_components=5, n_iter=6, random_state=3)
        eb, sb, vb, _ = rsvd_core(b, n_components=5, n_iter=6, random_state=3)
    ds.close()
    assert np.array_equal(ea, eb) and np.array_equal(sa, sb) and np.array_equal(va, vb)


def test_cache_reads_the_store_once(store):
    path, X, depth, idf = store
    tf = {"cell_depth": depth, "idf": idf, "scale_factor": 1e4}
    ds = cytome.open(path)
    with ds.chunk_source("ATAC_counts", cache=True, cache_budget_bytes=1 << 30) as src:
        ops = EngineOps(path, "ATAC_counts", *X.shape, tfidf=tf, source=src)
        B = np.random.default_rng(0).standard_normal((X.shape[1], 5))
        first = ops.xb(B)
        once = src.bytes_read
        assert once > 0 and src.cached_bytes == once
        again = ops.xb(B)
        ops.xty(first)
        assert src.bytes_read == once
        assert np.array_equal(first, again)
    ds.close()


def test_runSVD_with_cache_chunks_matches_without(store):
    import piaso
    path, X, depth, idf = store
    ds = cytome.open(path)
    ds.features("ATAC")["selected"] = np.ones(X.shape[1], dtype=bool)
    kw = dict(modality="ATAC", measurement="counts", auto_tfidf=True, n_components=4,
              n_iter=4, random_state=10, return_svd=True, verbosity=0,
              selected_feature_col_name="selected")
    e1, s1, _ = piaso.tl.runSVD(ds, key_added="X_a", cache_chunks=False, **kw)
    e2, s2, _ = piaso.tl.runSVD(ds, key_added="X_b", cache_chunks=True, **kw)
    ds.close()
    assert np.array_equal(e1, e2) and np.array_equal(s1, s2)


def test_runSVD_in_a_worker_thread(store):
    """The source reads on its own threads and is called only from the
    thread that runs runSVD; a dataset opened in a worker thread works."""
    import piaso
    path, X, depth, idf = store
    out = {}

    def work():
        try:
            ds = cytome.open(path)
            e, s, _ = piaso.tl.runSVD(ds, modality="ATAC", measurement="counts", auto_tfidf=True,
                                      n_components=3, n_iter=3, random_state=1, return_svd=True,
                                      use_highly_variable=False, verbosity=0)
            ds.close()
            out["e"] = e
        except Exception as exc:  # pragma: no cover
            out["err"] = exc
    t = threading.Thread(target=work)
    t.start(); t.join()
    assert "err" not in out, out.get("err")
    assert out["e"].shape == (X.shape[0], 3)


def test_a_source_that_returns_the_wrong_number_of_chunks_is_an_error(store):
    path, X, depth, idf = store
    ds = cytome.open(path)

    class Short:
        def __init__(self, s): self.s, self.matrix_name = s, s.matrix_name
        def meta(self): return self.s.meta()
        def chunks(self, layout): return self.s.chunks(layout)
        def fetch(self, layout, idx, nxt=None): return self.s.fetch(layout, idx)[:-1]
    with ds.chunk_source("ATAC_counts") as src:
        remap = np.arange(X.shape[1], dtype=np.int32)
        with pytest.raises(ValueError, match="returned"):
            _piaso.svd_pass_source(Short(src), "xb", np.ones((X.shape[1], 2)), remap, X.shape[1])
    ds.close()
