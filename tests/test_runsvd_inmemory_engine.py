"""runSVD on an in-memory AnnData through the Rust pass engine.

The AnnData's matrix is handed to the engine as a chunk source
(piaso.tools._svd_memsource.MatrixChunkSource), so an AnnData and the same
matrix in a store run one solver over the same products: identical to the bit,
whatever the container's layout (CSR, CSC, dense, a non-canonical matrix), with
TF-IDF on the fly, a materialised layer, a feature selection and a cell subset.
``method``, ``tol``, ``n_iter`` and ``drop_first_component`` mean what they mean
on a store; ``use_engine=False`` computes the same products in Python.
"""
import re

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import piaso
from piaso.tools._svd_memsource import MatrixChunkSource, engine_ready

try:
    from piaso import _piaso
    HAVE = hasattr(_piaso, "svd_pass_source")
except Exception:  # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="Rust extension without svd_pass_source")

N, G = 900, 1600
KW = dict(n_components=6, random_state=3, n_threads=3)


def counts(seed=11, dtype=np.float32):
    """Poisson counts with three groups of cells, each with its own accessible
    features, so the leading components are separated by real gaps."""
    rng = np.random.default_rng(seed)
    X = sp.random(N, G, density=0.04, format="csr", random_state=seed, dtype=np.float64)
    X.data = (rng.poisson(1.5, X.nnz) + 1).astype(np.float64)
    for g in range(3):
        rows = np.flatnonzero(np.arange(N) % 3 == g)
        B = sp.random(len(rows), 200, density=0.3, random_state=seed + g, format="coo")
        X = X + sp.coo_matrix((np.ones(B.nnz), (rows[B.row], B.col + 200 * g)), shape=(N, G))
    return sp.csr_matrix(X).astype(dtype)


def atac(X=None, seed=11):
    X = counts(seed) if X is None else X
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(N)]
    A.var_names = [f"chr1:{1000 * j}-{1000 * j + 500}" for j in range(G)]
    sel = np.zeros(G, dtype=bool)
    sel[np.random.default_rng(seed).choice(G, 1200, replace=False)] = True
    A.var["highly_variable"] = sel
    return A


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """The same counts in a store with both layouts on disk (the engine takes
    column chunks there, row chunks for a CSR AnnData)."""
    import cytome
    A = atac()
    p = str(tmp_path_factory.mktemp("mem") / "atac.cytome")
    ds = cytome.from_anndata(A, modality="ATAC", output=p)
    ds.ATAC.counts.build_feature_index()
    ds.features("ATAC")["highly_variable"] = np.asarray(A.var["highly_variable"])
    ds.flush()
    ds.close()
    return p, A


def test_an_anndata_and_its_store_give_the_same_bits(store):
    import cytome
    p, A = store
    ds = cytome.open(p)
    ref = piaso.tl.runSVD(ds, modality="ATAC", measurement="counts", return_svd=True,
                          key_added="X_ref", **KW)
    ds.close()
    got = piaso.tl.runSVD(A.copy(), modality="ATAC", auto_tfidf=True, streaming=True, **KW)
    for a, b in zip(got, ref):
        assert np.array_equal(a, b)


def test_tfidf_statistics_equal_the_stores(store):
    import cytome
    from piaso.tools._runTFIDF import compute_tfidf_stats
    from piaso.tools._svd_memsource import tfidf_stats_matrix
    p, A = store
    ds = cytome.open(p)
    want = compute_tfidf_stats(ds, measurement="counts", modality="ATAC", write_to_metadata=False)
    ds.close()
    got = tfidf_stats_matrix(A.X)
    assert np.array_equal(got["cell_depth"], want["cell_depth"])
    assert np.array_equal(got["idf"], want["idf"]) and got["idf"].dtype == np.float32
    assert got["scale_factor"] == want["scale_factor"]


@pytest.mark.parametrize("form", ["csc", "dense", "coo", "noncanonical"])
def test_the_container_does_not_change_the_bits(form):
    A = atac()
    ref = piaso.tl.runSVD(A.copy(), modality="ATAC", auto_tfidf=True, streaming=True, **KW)
    B = A.copy()
    if form == "csc":
        B.X = sp.csc_matrix(A.X)
    elif form == "dense":
        B.X = A.X.toarray()
    elif form == "coo":
        try:
            B.X = sp.coo_matrix(A.X)
        except ValueError as e:                     # anndata 0.13 accepts only CSR and CSC sparse matrices
            pytest.skip(f"this anndata refuses a COO matrix as X: {e}")
    else:
        # every entry stored twice as halves, each row's indices reversed:
        # duplicates and unsorted indices
        X = A.X
        per_row = np.diff(X.indptr)
        indptr = np.r_[0, np.cumsum(2 * per_row)]
        indices = np.empty(2 * X.nnz, dtype=np.int32)
        data = np.empty(2 * X.nnz, dtype=X.dtype)
        for r in range(N):
            a, b, o = X.indptr[r], X.indptr[r + 1], indptr[r]
            c, d = X.indices[a:b][::-1], X.data[a:b][::-1] / 2
            indices[o:o + 2 * (b - a)], data[o:o + 2 * (b - a)] = np.r_[c, c], np.r_[d, d]
        Y = sp.csr_matrix((data, indices, indptr), shape=X.shape)
        assert not Y.has_canonical_format
        before = (Y.data.copy(), Y.indices.copy())
        B.X = Y
    got = piaso.tl.runSVD(B, modality="ATAC", auto_tfidf=True, streaming=True, **KW)
    for a, b in zip(got, ref):
        assert np.array_equal(a, b), form
    if form == "noncanonical":    # the caller's matrix is left as it was
        assert np.array_equal(B.X.data, before[0]) and np.array_equal(B.X.indices, before[1])


def test_a_layer_as_stored_and_a_cell_subset():
    A = atac()
    piaso.tl.run_TFIDF(A, output_layer="tfidf")
    e_x, _, _ = piaso.tl.runSVD(A, streaming=True, **KW)                  # raw counts in X
    e_l, _, _ = piaso.tl.runSVD(A, layer="tfidf", streaming=True, **KW)
    assert not np.allclose(e_x, e_l)
    mask = np.zeros(N, dtype=bool)
    mask[np.random.default_rng(1).choice(N, 400, replace=False)] = True
    e_m, s_m, v_m = piaso.tl.runSVD(A, layer="tfidf", cell_mask=mask, **KW)
    sub = A[mask].copy()
    e_s, s_s, v_s = piaso.tl.runSVD(sub, layer="tfidf", streaming=True, **KW)
    assert np.array_equal(e_m, e_s) and np.array_equal(s_m, s_s) and np.array_equal(v_m, v_s)
    with pytest.raises(ValueError, match="not found in adata.layers"):
        piaso.tl.runSVD(A, layer="nope", **KW)


def test_the_selected_features_and_explicit_indices_agree():
    A = atac()
    sel = np.flatnonzero(np.asarray(A.var["highly_variable"]))
    e1 = piaso.tl.runSVD(A.copy(), modality="ATAC", auto_tfidf=True, streaming=True, **KW)[0]
    e2 = piaso.tl.runSVD(A.copy(), modality="ATAC", auto_tfidf=True, streaming=True,
                         use_highly_variable=False, hvg_indices_override=sel, **KW)[0]
    from piaso.tools._svd_memsource import tfidf_stats_matrix
    tf = dict(tfidf_stats_matrix(A.X), col_mask=np.asarray(A.var["highly_variable"]))
    e3 = piaso.tl.runSVD(A.copy(), tfidf_params=tf, streaming=True, **KW)[0]
    assert np.array_equal(e1, e2) and np.array_equal(e1, e3)


def test_power_runs_exactly_its_rounds_and_tol_stops_early(capsys):
    A = atac()
    for n_iter in (2, 5):
        piaso.tl.runSVD(A, auto_tfidf=True, method="power", n_iter=n_iter, verbosity=1, **KW)
        out = capsys.readouterr().out
        assert f"{n_iter} rounds, {2 + 2 * n_iter} passes" in out and "SVD engine: Rust pass" in out
    passes = {}
    for tol in (1e-1, 1e-9):
        piaso.tl.runSVD(A, auto_tfidf=True, method="krylov", n_iter=25, tol=tol, verbosity=1, **KW)
        passes[tol] = int(re.search(r"(\d+) passes \(", capsys.readouterr().out).group(1))
    assert passes[1e-1] < passes[1e-9]
    e_p = piaso.tl.runSVD(A, auto_tfidf=True, method="power", n_iter=2, streaming=True, **KW)[0]
    e_k = piaso.tl.runSVD(A, auto_tfidf=True, method="krylov", n_iter=8, streaming=True, **KW)[0]
    assert not np.allclose(e_p, e_k)
    with pytest.raises(ValueError, match="method must be"):
        piaso.tl.runSVD(A, method="randomized", **KW)


def test_drop_first_component_keeps_components_two_onwards():
    # the components themselves, so Euclidean (TF-IDF here would otherwise be scaled to unit length, over five
    # components in one call and six in the other)
    A, B = atac(), atac()
    e1, s1, v1 = piaso.tl.runSVD(A, auto_tfidf=True, streaming=True, drop_first_component=True, unit_length=False,
                                 **dict(KW, n_components=5))
    e2, s2, v2 = piaso.tl.runSVD(B, auto_tfidf=True, streaming=True, unit_length=False, **dict(KW, n_components=6))
    assert np.array_equal(e1, e2[:, 1:]) and np.array_equal(s1, s2[1:]) and np.array_equal(v1, v2[1:])
    assert A.uns["X_svd_params"]["components"] == [2, 6]
    assert np.array_equal(A.obsm["X_svd"], e1)


def test_float64_and_integer_input():
    A = atac()
    e32, s32, _ = piaso.tl.runSVD(A, auto_tfidf=True, streaming=True, **KW)
    B = atac(counts(dtype=np.float64))
    e64, s64, _ = piaso.tl.runSVD(B, auto_tfidf=True, streaming=True, **KW)
    C = atac(counts(dtype=np.int32))
    ei = piaso.tl.runSVD(C, auto_tfidf=True, streaming=True, **KW)[0]
    # integers are widened to float32 as on a store; float64 computes in float64
    assert np.array_equal(ei, e32)
    assert np.allclose(s64, s32, rtol=1e-5)
    lead = slice(0, 2)
    assert np.allclose(np.abs(e64[:, lead]), np.abs(e32[:, lead]), rtol=1e-3, atol=1e-5 * np.abs(e32).max())


def test_empty_cells_and_features():
    X = counts().tolil()
    X[:50] = 0
    X[:, :100] = 0
    A = atac(X.tocsr())
    e = piaso.tl.runSVD(A, auto_tfidf=True, streaming=True, use_highly_variable=False, **KW)[0]
    assert np.isfinite(e).all() and np.abs(e[:50]).max() <= 1e-10 * np.abs(e).max()


def test_python_products_equal_the_engine_to_rounding():
    A = atac()
    e1, s1, _ = piaso.tl.runSVD(A, auto_tfidf=True, streaming=True, **KW)
    e2, s2, _ = piaso.tl.runSVD(A, auto_tfidf=True, streaming=True, use_engine=False, **KW)
    assert np.allclose(s1, s2, rtol=1e-8)
    assert np.allclose(e1, e2, rtol=1e-5, atol=1e-6 * np.abs(e1).max())


def test_scale_data_and_cache_chunks():
    A = atac()
    piaso.tl.run_TFIDF(A, output_layer="tfidf")
    e1 = piaso.tl.runSVD(A, layer="tfidf", streaming=True, **KW)[0]
    e2 = piaso.tl.runSVD(A, layer="tfidf", streaming=True, cache_chunks=True, **KW)[0]
    assert np.array_equal(e1, e2)
    e3 = piaso.tl.runSVD(A, layer="tfidf", streaming=True, scale_data=True, **KW)[0]
    assert e3.shape == e1.shape and not np.allclose(e1, e3)
    with pytest.raises(ValueError, match="scale_data"):
        piaso.tl.runSVD(A, auto_tfidf=True, scale_data=True, **KW)


def test_auto_tfidf_refuses_normalised_values():
    A = atac()
    piaso.tl.run_TFIDF(A, output_layer="tfidf")
    with pytest.raises(ValueError, match="auto_tfidf needs raw counts.*fractional"):
        piaso.tl.runSVD(A, layer="tfidf", auto_tfidf=True, **KW)


def test_not_an_anndata_is_refused():
    with pytest.raises(TypeError, match="AnnData"):
        piaso.tl.runSVD(counts(), **KW)


@pytest.mark.parametrize("mode", ["xb", "xty"])
def test_the_source_against_the_engine_reading_a_store(store, mode):
    """The protocol on its own: one pass from the in-memory source equals the
    engine reading the store itself, in either of the store's layouts."""
    p, A = store
    X = engine_ready(A.X)
    rng = np.random.default_rng(5)
    cols = np.sort(rng.choice(G, 1000, replace=False))
    rows = np.sort(rng.choice(N, 500, replace=False))
    remap = np.full(G, -1, dtype=np.int32)
    remap[cols] = np.arange(len(cols), dtype=np.int32)
    row_map = np.full(N, -1, dtype=np.int32)
    row_map[rows] = np.arange(len(rows), dtype=np.int32)
    depth = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()
    idf = np.log1p(N / np.maximum(np.asarray(X.sum(axis=0, dtype=np.float64)).ravel(), 1)).astype(np.float32)
    dense = rng.standard_normal((len(cols) if mode == "xb" else len(rows), 4))
    for layout in ("csr", "csc"):
        ref = _piaso.svd_pass(p, "ATAC_counts", mode, dense, remap, len(cols), row_map, len(rows),
                              depth, idf, 1e4, n_threads=2, layout=layout)
        for threads, batch in ((1, 0), (3, 16 * 4000)):
            src = MatrixChunkSource(X, n_threads=threads)
            got = _piaso.svd_pass_source(src, mode, dense, remap, len(cols), row_map, len(rows),
                                         depth, idf, 1e4, n_threads=threads, batch_bytes=batch)
            assert np.array_equal(got, ref), (layout, threads, batch)


def test_the_source_caches_within_its_budget():
    X = engine_ready(counts())
    src = MatrixChunkSource(X, n_threads=2, cache=True, cache_budget_bytes=1 << 30)
    idx, _, _, nnz = src.chunks("csr")
    first = src.fetch("csr", idx)
    once = src.bytes_read
    assert once == sum(len(b[0]) + len(b[1]) + len(b[2]) for b in first) == src.cached_bytes
    again = src.fetch("csr", idx)
    assert src.bytes_read == once and all(a is b for a, b in zip(first, again))
    assert int(nnz.sum()) == X.nnz
    small = MatrixChunkSource(X, n_threads=2, cache=True, cache_budget_bytes=1000)
    with pytest.warns(RuntimeWarning, match="budget"):
        small.fetch("csr", idx)
    assert small.cache_declined is not None and small.cached_bytes == 0
    with pytest.raises(ValueError, match="csr chunks"):
        src.chunks("csc")


def test_infog_params_are_applied_on_an_anndata():
    """INFOG with explicit parameters is computed in Python, on full rows, then
    the selected features: the same as factorising the normalised matrix."""
    from piaso.tools._normalization import _normalize_chunk_infog
    A = atac()
    X = A.X
    depth = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()
    gene = np.asarray(X.sum(axis=0, dtype=np.float64)).ravel()
    params = {"cell_depth": depth, "inv_gene_depth": np.where(gene > 0, 1.0 / np.maximum(gene, 1e-300), 0.0),
              "scale": float(np.median(depth)), "counts_sum": float(depth.sum()),
              "threshold": float(np.sqrt(N))}
    A.layers["infog_manual"] = sp.csr_matrix(_normalize_chunk_infog(
        X, depth, params["inv_gene_depth"], params["scale"], params["counts_sum"], params["threshold"]))
    e1, s1, _ = piaso.tl.runSVD(A, infog_params=params, streaming=True, **KW)
    e2, s2, _ = piaso.tl.runSVD(A, layer="infog_manual", streaming=True, **KW)
    assert np.allclose(s1, s2, rtol=1e-8)
    assert np.allclose(np.abs(e1[:, :2]), np.abs(e2[:, :2]), rtol=1e-5, atol=1e-8 * np.abs(e2).max())
    e3 = piaso.tl.runSVD(A, streaming=True, **KW)[0]             # raw counts: not the same
    assert not np.allclose(np.abs(e1[:, :2]), np.abs(e3[:, :2]), rtol=1e-3)
