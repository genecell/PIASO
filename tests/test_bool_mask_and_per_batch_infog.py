"""Boolean entity columns, and per-batch INFOG on the cytome GDR path.

Two defects fixed together because the second exposed the first:

1. cytome stored boolean columns as TEXT ('1'/'0'), because np.bool_ is not
   np.integer and the type map fell through. Read back that is a <U1 array and
   a non-empty string is truthy, so `col.astype(bool)` returned ALL TRUE --
   `highly_variable` silently became "every gene", and runSVD computed on
   32,285 genes instead of 3,000 on any file where INFOG created the column.

2. The cytome GDR recomputed neither INFOG nor HVGs per batch; it reused the
   whole-dataset layer and the global column, while the AnnData path had always
   called infog_svd(adata_i, ...) per batch. A different method, not a faster
   one.
"""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from piaso.utils._bool_mask import as_bool_mask

anndata = pytest.importorskip("anndata")
cytome = pytest.importorskip("cytome")


@pytest.mark.parametrize("stored,expected", [
    ([True, False, True], [True, False, True]),
    ([1, 0, 1], [True, False, True]),
    (["1", "0", "1"], [True, False, True]),
    (["True", "False", "True"], [True, False, True]),
    (["true", "false", "true"], [True, False, True]),
    ([1.0, 0.0, 1.0], [True, False, True]),
    ([None, "", "1"], [False, False, True]),
])
def test_every_spelling_coerces(stored, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = as_bool_mask(np.asarray(stored, dtype=object), name="t")
    assert got.tolist() == expected


def test_text_column_would_have_selected_everything():
    """The exact failure: '0' is truthy, so astype(bool) selects all."""
    col = np.asarray(["1", "0", "0", "0"])
    assert col.astype(bool).sum() == 4          # the bug
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert as_bool_mask(col, name="t").sum() == 1   # the fix


def test_text_column_warns():
    with pytest.warns(UserWarning, match="stored as text"):
        as_bool_mask(np.asarray(["1", "0"]), name="genes.highly_variable")


def test_all_true_mask_warns():
    with pytest.warns(UserWarning, match="every one of"):
        as_bool_mask(np.array([True, True, True]), name="genes.highly_variable")


def test_unrecognised_values_raise():
    with pytest.raises(ValueError, match="cannot read as a boolean mask"):
        as_bool_mask(np.asarray(["yes-ish", "0"]), name="t")


def test_bool_column_round_trips_as_integer(tmp_path):
    """cytome must store booleans as INTEGER, so new files never hit this."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(30, 8)).astype(np.float32))
    d = cytome.from_anndata(anndata.AnnData(X=X), output=str(tmp_path / "t.cytome"))
    mask = np.array([True, False] * 4)
    d.features("RNA")["hv"] = mask
    d.flush()
    declared = [r[2] for r in d._conn.execute("PRAGMA table_info(genes)") if r[1] == "hv"][0]
    back = np.asarray([r[0] for r in d._conn.execute("SELECT hv FROM genes ORDER BY rowid")])
    assert declared == "INTEGER"
    assert back.dtype != np.dtype("<U1")
    assert np.array_equal(back.astype(bool), mask)
    d.close()


def test_masked_infog_equals_infog_on_the_subset(tmp_path):
    """Per-batch INFOG must see exactly what a cytome of those cells would."""
    from piaso.tools._normalization import _infog_streaming
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(400, 120)).astype(np.float32))
    full = cytome.from_anndata(anndata.AnnData(X=X), output=str(tmp_path / "f.cytome"))
    mask = np.zeros(400, dtype=bool)
    mask[100:260] = True
    sub = cytome.from_anndata(anndata.AnnData(X=sp.csr_matrix(X[mask])),
                              output=str(tmp_path / "s.cytome"))

    a = _infog_streaming(full, n_top_genes=30, cell_mask=mask, write=False, verbosity=0)
    b = _infog_streaming(sub, n_top_genes=30, write=False, verbosity=0)

    assert np.array_equal(a["hvg_indices"], b["hvg_indices"])
    assert a["scale"] == b["scale"]
    np.testing.assert_allclose(a["gene_var"], b["gene_var"], rtol=0, atol=0)
    # cell_depth stays FULL length so the normaliser's global lookups work
    assert len(a["infog_params"]["cell_depth"]) == 400
    assert np.all(a["infog_params"]["cell_depth"][~mask] == 0)
    full.close(); sub.close()


def test_write_false_leaves_the_cytome_untouched(tmp_path):
    """35 batches writing highly_variable into one file would clobber it."""
    from piaso.tools._normalization import _infog_streaming
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.poisson(1.0, size=(200, 60)).astype(np.float32))
    d = cytome.from_anndata(anndata.AnnData(X=X), output=str(tmp_path / "t.cytome"))
    before = {r[1] for r in d._conn.execute("PRAGMA table_info(genes)")}
    _infog_streaming(d, n_top_genes=20, write=False, verbosity=0)
    after = {r[1] for r in d._conn.execute("PRAGMA table_info(genes)")}
    assert before == after
    assert "RNA_infog_params" not in dict(d.metadata) if hasattr(d, "metadata") else True
    d.close()


def test_runsvd_survives_a_categorical_hvg_column():
    """An h5ad highly_variable can come back as `category` after a round-trip.

    Indexing with it raises "Unknown indexer"; .astype(bool) on it is all-True.
    Both AnnData runSVD paths must coerce instead.
    """
    import pandas as pd
    import piaso
    rng = np.random.default_rng(0)
    for col in (pd.Categorical(["True", "False"] * 15),
                np.array([True, False] * 15),
                np.array(["1", "0"] * 15)):
        a = anndata.AnnData(X=sp.csr_matrix(rng.poisson(1.0, (60, 30)).astype(np.float32)))
        a.var["highly_variable"] = col
        a.layers["infog"] = a.X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piaso.tl.runSVD(a, layer="infog", n_components=5, key_added="X_svd",
                            use_highly_variable=True)
        assert a.obsm["X_svd"].shape == (60, 5)


def test_gdr_records_enough_to_reproduce_itself(tmp_path):
    """A stored embedding must say how it was made.

    The old parameter set recorded neither groupby nor batch_key nor
    resolution, so a stored X_gdr could not be re-derived, nor even told apart
    from a supervised run -- which made comparing against one uninterpretable.
    """
    import piaso
    rng = np.random.default_rng(0)
    n_cells, n_genes = 900, 300
    grp = np.repeat(np.arange(3), n_cells // 3)
    X = rng.poisson(0.3, size=(n_cells, n_genes)).astype(np.float32)
    for g in range(3):
        X[grp == g, g * 60:(g + 1) * 60] += rng.poisson(6.0, size=(int((grp == g).sum()), 60))
    a = anndata.AnnData(X=sp.csr_matrix(X))
    a.var_names = [f"g{j}" for j in range(n_genes)]
    a.obs["batch"] = np.array(["b0", "b1"])[np.arange(n_cells) % 2]
    d = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    piaso.tl.infog(d, n_top_genes=100, save_layer=True)
    piaso.tl.runGDR(d, batch_key="batch", groupby=None, n_gene=5, layer="infog",
                    key_added="X_g", max_workers=2, verbosity=0)

    params = dict(d.metadata["X_g_params"])
    for required in ("batch_key", "groupby", "resolution", "n_svd_dims",
                     "n_highly_variable_genes", "random_seed", "piaso_version",
                     "per_batch_infog"):
        assert required in params, f"{required} not recorded"
    assert params["batch_key"] == "batch"
    assert params["groupby"] is None          # de novo, not supervised
    d.close()


@pytest.mark.parametrize("workers", [1, 2])
def test_per_batch_infog_runs_at_every_worker_count(tmp_path, workers, monkeypatch):
    """Stage 1 must not have two bodies that can drift apart.

    It did: the threaded path got per-batch INFOG and the serial path kept the
    whole-dataset layer, so with the default stage1_workers=1 every caller
    silently got the old method. It also meant every worker sweep was comparing
    two different algorithms, which is what produced the "two workers is
    anomalously slow" reading -- w=1 was simply doing less work.
    """
    import piaso
    import piaso.tools._normalization as norm

    rng = np.random.default_rng(0)
    n_cells, n_genes = 600, 200
    grp = np.repeat(np.arange(3), n_cells // 3)
    X = rng.poisson(0.3, size=(n_cells, n_genes)).astype(np.float32)
    for g in range(3):
        X[grp == g, g * 40:(g + 1) * 40] += rng.poisson(6.0, size=(int((grp == g).sum()), 40))
    a = anndata.AnnData(X=sp.csr_matrix(X))
    a.var_names = [f"g{j}" for j in range(n_genes)]
    a.obs["batch"] = np.array(["b0", "b1", "b2"])[np.arange(n_cells) % 3]
    d = cytome.from_anndata(a, output=str(tmp_path / "t.cytome"))
    piaso.tl.infog(d, n_top_genes=60, save_layer=True)

    seen = {"n": 0}
    orig = norm._infog_streaming

    def counted(*args, **kwargs):
        if kwargs.get("cell_mask") is not None:
            seen["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(norm, "_infog_streaming", counted)
    piaso.tl.runGDR(d, batch_key="batch", groupby=None, n_gene=5, layer="infog",
                    key_added="X_g", max_workers=2, verbosity=0,
                    stage1_workers=workers)
    assert seen["n"] == 3, (
        f"per-batch INFOG ran {seen['n']} times for 3 batches at "
        f"stage1_workers={workers}")
    d.close()


def test_rungdrparallel_forwards_rather_than_duplicating():
    """The deprecated entry point must not be a second copy of runGDR.

    It was a ~400-line copy, and the two drifted: fixes landed in runGDR while
    runGDRParallel kept an O(n_batches**2) scoring loop. Same defect class as
    stage 1's duplicated body and the two INFOG implementations.
    """
    import pathlib as _p
    import piaso.tools._runGDR as m

    # Read the FILE, not inspect.getsource: functools.wraps sets __wrapped__
    # and getsource follows it, so it would return runGDR's 2,400 lines and
    # this test would measure the wrong function.
    lines = _p.Path(m.__file__).read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("def runGDRParallel("))
    body = lines[start:]
    end = next((i for i, l in enumerate(body[1:], 1) if l.startswith("def ")), len(body))
    src = "\n".join(body[:end])
    assert end < 40, f"runGDRParallel has grown a body again ({end} lines)"
    assert "return runGDR(" in src, "runGDRParallel must forward to runGDR"

    # and the forwarding must stay introspectable
    import inspect
    import piaso
    assert "modality" in inspect.signature(piaso.tl.runGDRParallel).parameters


def test_no_duplicated_algorithm_bodies_in_rungdr():
    """Guard the file against regrowing a second copy of a stage.

    Counts how many places run the per-batch chain. There must be exactly one
    of each: two copies is how per-batch INFOG ended up active only when
    stage1_workers > 1, which silently gave the default path the old method.
    """
    import pathlib
    src = pathlib.Path(piaso_runGDR_path()).read_text()
    # the stage-1 chain: runSVD -> neighbors -> leiden, per batch
    assert src.count("_piaso_mod.tl.leiden(") <= 2, (
        "more than two leiden call sites in _runGDR.py — a stage body may have "
        "been duplicated again")
    # per-batch INFOG must exist exactly once
    assert src.count("_info = _infog_streaming(") == 1


def piaso_runGDR_path():
    import piaso.tools._runGDR as m
    return m.__file__


def test_stage_policies_differ_and_match_the_measurements(tmp_path):
    """Stage 1 spends the budget outward, stage 3 inward. They are opposites.

    Stage 1's per-batch work is serial inside, so outer concurrency is the only
    axis that pays (ADVIS: 244.2 s at one worker, 76.8 s at eight). Stage 3's
    score() is already parallel inside via the Rust rayon pool, so outer
    workers fragment it (252.9 s at one, 313.9 s at eight). A single shared
    policy would pick eight for both and get stage 3 wrong.
    """
    import piaso
    import piaso.tools._runGDR as G

    src = __import__("pathlib").Path(G.__file__).read_text()
    # stage 1 auto: bounded by the budget, not by _determine_parallelism
    assert "_n_workers = max(1, min(len(batches), max_workers or 1))" in src
    # stage 3 auto: two outer, whole budget inside
    assert "_n_score_workers = max(1, min(2, len(batches)))" in src
    assert "_score_threads = max(1, max_workers or 1)" in src
    # and stage 3 must NOT use the outer-maximising helper
    i = src.index("_n_score_workers = max(1, min(2, len(batches)))")
    window = src[i - 1500:i]
    assert "_determine_parallelism(len(batches)" not in window


def test_stage2_keeps_its_measured_cap():
    """COSG has no inner-thread knob, so a budget cannot be spent inside it."""
    import piaso.tools._runGDR as G
    assert G._COSG_THREAD_CAP == 2
