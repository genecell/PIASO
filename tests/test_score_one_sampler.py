"""score()'s control sets are one definition on every path (2026-10-22).

- One sampler (`_draw_controls`): a fresh RandomState(seed) per gene set, gene by gene (PIASO 1.0's stream), used
  by the AnnData path with and without the compiled extension, by the single-set path and by the cytome path.
- Gene statistics summed in float64 over the same blocks on both paths (`_precompute_stats`,
  `_precompute_stats_streaming`), so the nearest genes, hence the control genes, do not depend on the path.
- `exclude_set_genes` (the default): each gene's controls drawn from its nearest genes outside its own set.
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import piaso
from piaso.tools._normalization import (_PASS1_BLOCK_ROWS, _draw_controls, _knn_from_mean_var,
                                        _precompute_stats, _precompute_stats_streaming)


def toy(n_cells=300, n_genes=400, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.5, 1.0, size=(1, n_genes)) * rng.gamma(2.0, 0.5, size=(n_cells, 1))
    X = sp.csr_matrix(rng.poisson(lam).astype(dtype))
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    sets = {"A": list(a.var_names[:12]), "B": list(a.var_names[40:70]), "C": list(a.var_names[100:103])}
    return a, sets


def test_draws_are_pinned_and_are_numpy_legacy_stream():
    knn = np.arange(4 * 30).reshape(4, 30)
    d = _draw_controls(knn, 6, 1927)
    assert d.tolist() == [[2, 47, 61, 98], [7, 50, 69, 104], [23, 44, 70, 118], [24, 35, 76, 108],
                          [0, 55, 61, 116], [11, 49, 86, 93]]
    # the definition: RandomState(seed).randint over each gene's k candidates, gene by gene (PIASO 1.0)
    idx = np.random.RandomState(1927).randint(0, 30, size=(4, 6))
    assert np.array_equal(d, knn[np.arange(4)[:, None], idx].T)


def test_anndata_with_and_without_the_extension_and_a_cytome_give_the_same_scores(tmp_path):
    import warnings
    import cytome
    pytest.importorskip("piaso._piaso")         # without the extension both AnnData runs would be the Python path
    a, sets = toy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # the extension must really be used
        piaso.tl.score(a, gene_list=sets, layer=None, random_seed=1927)
    s_rust, n1, p_rust = piaso.tl.score(a, gene_list=sets, layer=None, random_seed=1927, compute_pvalues=True)
    s_py, n2, p_py = piaso.tl.score(a, gene_list=sets, layer=None, random_seed=1927, compute_pvalues=True,
                                    use_rust=False)
    path = str(tmp_path / "toy.cytome")
    ds = cytome.from_anndata(a, modality="RNA", output=path)
    ds.close()
    ds = cytome.open(path, mode="r")
    try:
        s_cy, n3, p_cy = piaso.tl.score(ds, gene_list=sets, modality="RNA", layer=None, random_seed=1927,
                                        compute_pvalues=True)
    finally:
        ds.close()
    assert list(n1) == list(n2) == list(n3)
    assert np.abs(s_rust - s_py).max() < 1e-10
    assert np.abs(s_rust - s_cy).max() < 1e-10
    assert np.allclose(p_rust, p_py) and np.allclose(p_rust, p_cy)


def test_single_set_mode_draws_the_same_controls_as_multi_set_mode():
    a, sets = toy()
    multi, names, _ = piaso.tl.score(a, gene_list=sets, layer=None, random_seed=1927)
    piaso.tl.score(a, gene_list=sets["B"], layer=None, random_seed=1927, key_added="single_B")
    assert np.allclose(a.obs["single_B"].to_numpy(), multi[:, list(names).index("B")], atol=1e-10)


def test_statistics_are_float64_and_the_same_on_both_paths():
    rng = np.random.default_rng(1)
    X = sp.random(9000, 300, density=0.05, random_state=rng, format="csr",
                  data_rvs=lambda n: rng.gamma(1.0, 1.0, n)).astype(np.float32)   # a float32 matrix, > one block
    k_mem = _precompute_stats(X, 30, 40)
    # the float64 reference
    D = X.toarray().astype(np.float64)
    m = D.mean(0)
    ref = _knn_from_mean_var(np.column_stack([m, (D * D).mean(0) - m * m]), 30, 40)
    assert np.array_equal(k_mem, ref)

    def blocks():                                   # the cytome path's first pass reads blocks of _PASS1_BLOCK_ROWS
        for a0 in range(0, X.shape[0], _PASS1_BLOCK_ROWS):
            yield X[a0:a0 + _PASS1_BLOCK_ROWS], np.arange(a0, min(a0 + _PASS1_BLOCK_ROWS, X.shape[0]))
    k_str = _precompute_stats_streaming(blocks, X.shape[0], X.shape[1], 30, 40)
    assert np.array_equal(k_mem, k_str)


def test_excluding_the_set_genes_changes_only_the_draws_that_were_members():
    rng = np.random.default_rng(2)
    n_genes, k = 200, 10
    knn_ext = np.array([rng.permutation(np.delete(np.arange(n_genes), g))[:2 * k] for g in range(n_genes)])
    members = np.arange(0, 40, 2)                   # a set of 20 genes
    rows = knn_ext[members]
    excl = _draw_controls(rows, 50, 7, members=members, k=k)
    plain = _draw_controls(rows[:, :k], 50, 7)
    assert not np.isin(excl, members).any()
    was_member = np.isin(plain, members)
    assert was_member.any() and (~was_member).any()
    assert np.array_equal(excl[~was_member], plain[~was_member])     # every other draw is the same gene
    clean = ~np.isin(rows[:, :k], members).any(axis=1)              # genes with no member among their k nearest
    assert clean.any() and np.array_equal(excl[:, clean], plain[:, clean])


def test_excluded_draws_are_uniform_over_the_k_nearest_non_members():
    k = 30
    row = np.arange(100, 160)[None, :]                       # one gene's 60 nearest, in order
    members = np.array([100, 104, 117, 129, 131])            # 4 among its 30 nearest, 1 beyond
    d = _draw_controls(row, 60000, 11, members=members, k=k)[:, 0]
    expect = [g for g in row[0] if g not in set(members)][:k]
    vals, counts = np.unique(d, return_counts=True)
    assert vals.tolist() == sorted(expect)
    assert np.abs(counts / len(d) - 1 / k).max() < 0.006     # about 3.5 standard errors at 60,000 draws


def test_a_row_short_of_non_members_draws_again_from_those_it_has():
    row = np.array([[1, 2, 3, 4, 5, 6]])
    members = np.array([1, 2, 3, 6])
    d = _draw_controls(row, 4000, 3, members=members, k=3)[:, 0]
    assert set(np.unique(d)) == {4, 5}


def test_the_exclusion_is_the_default_everywhere_it_is_offered():
    import inspect
    for f in (piaso.tl.score, piaso.tl.calculateScoreParallel, piaso.tl.predictCellTypeByMarker):
        assert inspect.signature(f).parameters["exclude_set_genes"].default is True
    a, sets = toy()
    s_default, _, _ = piaso.tl.score(a, gene_list=sets, layer=None)
    s_on, _, _ = piaso.tl.score(a, gene_list=sets, layer=None, exclude_set_genes=True)
    assert np.array_equal(s_default, s_on)


def test_score_excluding_the_set_genes_runs_on_every_path(tmp_path):
    import cytome
    a, sets = toy()
    s1, _, _ = piaso.tl.score(a, gene_list=sets, layer=None, exclude_set_genes=True)
    s2, _, _ = piaso.tl.score(a, gene_list=sets, layer=None, exclude_set_genes=True, use_rust=False)
    path = str(tmp_path / "toy.cytome")
    ds = cytome.from_anndata(a, modality="RNA", output=path)
    ds.close()
    ds = cytome.open(path, mode="r")
    try:
        s3, _, _ = piaso.tl.score(ds, gene_list=sets, modality="RNA", layer=None, exclude_set_genes=True)
    finally:
        ds.close()
    assert np.isfinite(s1).all()
    assert np.abs(s1 - s2).max() < 1e-10 and np.abs(s1 - s3).max() < 1e-10


def test_a_wider_neighbour_table_changes_nothing_with_the_exclusion_off():
    knn = np.arange(4 * 60).reshape(4, 60)
    assert np.array_equal(_draw_controls(knn, 6, 1927, k=30), _draw_controls(knn[:, :30], 6, 1927))
    narrow = _draw_controls(knn[:, :20], 6, 1927, k=30)            # k capped at the table's width
    assert narrow.max() < 4 * 60 and np.isin(narrow, knn[:, :20]).all()


def test_a_frozen_reference_table_gives_the_scores_of_score_own_table():
    """projectGDR scores the reference and the query with one frozen table (2 x 30 neighbours). With the
    exclusion on (the default) that is score()'s own table. With it off, the first 30 columns are used:
    the 30 nearest genes again, though where several genes are equally far at the 30th place (common in
    count data) a 60-neighbour query may keep a different one of them than a 30-neighbour query."""
    from piaso.tools._projectGDR import _reference_knn
    a, sets = toy()
    frozen = _reference_knn(a.X, 30, 40)
    assert frozen.shape[1] == 60
    own, _, _ = piaso.tl.score(a, gene_list=sets, layer=None)
    via, _, _ = piaso.tl.score(a, gene_list=sets, layer=None, precomputed_knn=frozen)
    assert np.abs(own - via).max() < 1e-10
    off, _, _ = piaso.tl.score(a, gene_list=sets, layer=None, exclude_set_genes=False, precomputed_knn=frozen)
    first, _, _ = piaso.tl.score(a, gene_list=sets, layer=None, exclude_set_genes=False,
                                 precomputed_knn=np.ascontiguousarray(frozen[:, :30]))
    assert np.abs(off - first).max() < 1e-10


def test_a_panel_smaller_than_the_neighbours_asked_for_still_scores():
    """40 genes: fewer than the 2 x 30 + 1 the exclusion's search asks for, so the search is capped at 39,
    but every gene still has its 30 candidates: no warning."""
    import warnings
    a, _ = toy(n_genes=40)
    sets = {"A": list(a.var_names[:5]), "B": list(a.var_names[10:20])}
    for flag in (True, False):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s, names, _ = piaso.tl.score(a, gene_list=sets, layer=None, exclude_set_genes=flag)
        assert np.isfinite(s).all() and list(names) == ["A", "B"]
        assert not [x for x in w if "candidate" in str(x.message)]


def test_genes_short_of_candidates_are_reported_once_per_call():
    """20 genes: each gene has at most 19 candidates, fewer still once its set's own genes are left out."""
    import warnings
    a, _ = toy(n_genes=20)
    sets = {f"S{i}": list(a.var_names[i:i + 4]) for i in range(0, 20, 4)}         # 5 sets of 4
    for flag, fewest in ((True, 16), (False, 19)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s, _, _ = piaso.tl.score(a, gene_list=sets, layer=None, exclude_set_genes=flag)
        msgs = [str(x.message) for x in w if "candidate control genes" in str(x.message)]
        assert np.isfinite(s).all()
        assert len(msgs) == 1, msgs
        assert "20 gene(s) in 5 gene set(s) have fewer than 30" in msgs[0] and f"the fewest: {fewest}" in msgs[0]
    a400, sets400 = toy()                            # 400 genes: every gene keeps 30 candidates
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        piaso.tl.score(a400, gene_list=sets400, layer=None)
    assert not [x for x in w if "candidate" in str(x.message)]


def test_the_neighbour_table_is_int32():
    """The exclusion doubles the table's width; int32 keeps it at the size the int64 table had before."""
    rng = np.random.default_rng(4)
    mv = rng.random((500, 2))
    full = _knn_from_mean_var(mv, 60, 40)
    sub = _knn_from_mean_var(mv, 60, 40, query_rows=np.array([3, 7, 9]))
    assert full.dtype == np.int32 and sub.dtype == np.int32
    assert np.array_equal(sub[[3, 7, 9]], full[[3, 7, 9]])
