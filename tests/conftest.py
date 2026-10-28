"""Shared pytest fixtures, data roots and markers of the PIASO test suite.

Data
----
Tests that need data find it under two roots, each a search path (``os.pathsep``-separated,
searched in order, like PATH):

    PIASO_TEST_DATA      input datasets: MouseE18Brain/, MTG_10K/, PIASO-data/, the MTG
                         h5ads (mtg_full.h5ad, mtg_sub_10000.h5ad) ... (default: <repo>/data)
    PIASO_TEST_RESULTS   pipeline outputs: snakemake/e18/, quickstart/ (default: <repo>/results)
    PIASO_TEST_E18_CYTOME  the E18 store itself, if it lives elsewhere

A test that misses its data is skipped; with ``PIASO_REQUIRE_DATA=1`` (a machine that
should have it) it fails instead, so a stale path cannot hide a whole tier again: until
2026-10-27 the E18 path named ``pipeline.cytome``, which the workflow had renamed
``e18_pipeline.cytome`` on 2026-05-22, and 148 tests skipped on a machine with the data.

Shared stores are never opened in place. Opening a store read-write can change it (cytome
adds the tables of its version) and some tests write, so the first test that needs one gets
a copy in this session's temporary folder (SQLite's backup from a read-only connection) and
the session uses that copy. ``E18_CYTOME``, ``QUICKSTART_CYTOME`` and ``MTG_RNA_CYTOME`` are
those copies' paths; a test asks for one with ``@pytest.mark.requires_e18`` or
``@pytest.mark.data(QUICKSTART_CYTOME)``, and the copy is made before it runs.

Markers
-------
``data(*paths)``  needs these files (skip, or fail with PIASO_REQUIRE_DATA=1)
``requires_e18``  needs the E18 store (``data(E18_CYTOME)``)
``snakemake``     runs Snakemake (tests/_workflow_harness.py; skipped without it)
``network``       may download
``slow``          a call of 20 s or more: the tests listed in tests/slow_tests.txt (measured)
pyproject.toml runs pytest with ``--strict-markers``: an unregistered marker is an error.

How to use
----------
- Paths: ``from conftest import E18_CYTOME, data_path, results_path``
- Data a test reads in place: ``@pytest.mark.data(data_path("MouseE18Brain", "E18_final.h5ad"))``
- Inside a fixture: ``require(path)`` skips or fails like the marker.
- Temp dirs: the ``tmp_dir`` fixture. A writable E18 store: ``tmp_cytome``; a 500-cell
  subset of it, open: ``subset_cytome``.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _roots(env, default):
    value = os.environ.get(env, "").strip()
    if not value:
        return [default]
    return [Path(v).expanduser() for v in value.split(os.pathsep) if v.strip()]


DATA_ROOTS = _roots("PIASO_TEST_DATA", REPO_ROOT / "data")
RESULTS_ROOTS = _roots("PIASO_TEST_RESULTS", REPO_ROOT / "results")
DATA_ROOT, RESULTS_ROOT = DATA_ROOTS[0], RESULTS_ROOTS[0]
REQUIRE_DATA = os.environ.get("PIASO_REQUIRE_DATA", "").strip() not in ("", "0")


def _find(roots, parts) -> str:
    for root in roots:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return str(candidate)
    return str(roots[0].joinpath(*parts))


def data_path(*parts) -> str:
    """A path under the input-data roots (PIASO_TEST_DATA): the first root that has it."""
    return _find(DATA_ROOTS, parts)


def results_path(*parts) -> str:
    """A path under the pipeline-results roots (PIASO_TEST_RESULTS)."""
    return _find(RESULTS_ROOTS, parts)


def missing(path, what="test data"):
    """Skip a test whose data is absent; with PIASO_REQUIRE_DATA=1, fail it."""
    msg = (f"{what} not found at {path} (PIASO_TEST_DATA="
           f"{os.pathsep.join(map(str, DATA_ROOTS))}, PIASO_TEST_RESULTS="
           f"{os.pathsep.join(map(str, RESULTS_ROOTS))})")
    if REQUIRE_DATA:
        pytest.fail(msg + "; PIASO_REQUIRE_DATA=1 asks for every data test to run", pytrace=False)
    pytest.skip(msg)


def pipeline_store(dataset) -> str:
    """The workflow's store of ``dataset`` under the results roots: ``<dataset>_pipeline.cytome``
    since the workflow named stores after their dataset (2026-05-22), else the older
    ``pipeline.cytome``."""
    for name in (f"{dataset}_pipeline.cytome", "pipeline.cytome"):
        path = results_path("snakemake", dataset, name)
        if os.path.exists(path):
            return path
    return results_path("snakemake", dataset, f"{dataset}_pipeline.cytome")


def _e18_source() -> str:
    env = os.environ.get("PIASO_TEST_E18_CYTOME", "").strip()
    return env or pipeline_store("e18")


# ---------------------------------------------------------------------
# Shared stores: one copy per session, made when a test first needs it
# ---------------------------------------------------------------------

_SESSION_DIR = None
_STORES = {}          # the session copy's path -> the shared store it copies
_MADE = set()


def _session_dir() -> str:
    global _SESSION_DIR
    if _SESSION_DIR is None:
        _SESSION_DIR = tempfile.mkdtemp(prefix="piaso_test_stores_")
    return _SESSION_DIR


def copy_store(src, dst):
    """A consistent copy of the store ``src`` at ``dst``, whatever ``src``'s write-ahead log
    holds: SQLite's online backup from a read-only connection into ``dst.partial``, then a
    rename, with no -wal, -shm or -journal of an earlier file left beside ``dst``."""
    src, dst = str(src), str(dst)
    partial = dst + ".partial"
    for f in (partial, dst):
        for sfx in ("", "-wal", "-shm", "-journal"):
            if os.path.exists(f + sfx):
                os.remove(f + sfx)
    s = sqlite3.connect(f"file:{os.path.abspath(src)}?mode=ro", uri=True)
    d = sqlite3.connect(partial)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    os.replace(partial, dst)
    return dst


def shared_store(src, name=None) -> str:
    """The path of this session's copy of the shared store ``src`` (made when a test that
    asks for it runs: ``@pytest.mark.data(<this path>)``, ``requires_e18``, or ``ensure``)."""
    dst = os.path.join(_session_dir(), name or os.path.basename(str(src)))
    _STORES[dst] = str(src)
    return dst


def ensure(path, what="test data"):
    """Make ``path`` exist for the running test, or skip (fail) it: a shared store's session
    copy is made from its source; any other path must exist already."""
    path = str(path)
    src = _STORES.get(path)
    if src is not None:
        if not os.path.exists(src):
            missing(src, what)
        if path not in _MADE:
            copy_store(src, path)
            _MADE.add(path)
        return path
    if not os.path.exists(path):
        missing(path, what)
    return path


def require(*paths, what="test data"):
    """``ensure`` for each path."""
    for p in paths:
        ensure(p, what)


E18_SOURCE = _e18_source()       # the shared store itself: for standalone scripts, never tests
E18_CYTOME = shared_store(E18_SOURCE, "e18_pipeline.cytome")
QUICKSTART_CYTOME = shared_store(results_path("quickstart", "pipeline.cytome"),
                                 "quickstart_pipeline.cytome")
MTG_RNA_CYTOME = shared_store(data_path("MTG_10K", "mtg_10k_rna.cytome"))

# Files read in place (never written by a test)
E18_H5AD = data_path("MouseE18Brain", "E18_final.h5ad")
PBMC_H5AD = data_path("HumanPBMC_SAN2", "HumanPBMCs_Multiome_RNA_QC.h5ad")
MTG_H5AD = data_path("mtg_full.h5ad")
MTG_SUB_H5AD = data_path("mtg_sub_10000.h5ad")
PIASO_DATA_SOURCE = data_path("PIASO-data")


# ---------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------

_MARKERS = {
    "data": "data(*paths): needs these files from the data roots (PIASO_TEST_DATA, "
            "PIASO_TEST_RESULTS); skipped without them, failed with PIASO_REQUIRE_DATA=1",
    "requires_e18": "needs the E18 store built by the workflow (data(E18_CYTOME))",
    "snakemake": "runs Snakemake (PIASO_SNAKEMAKE or PATH); skipped without it",
    "network": "may download (PIASO-data, Zenodo); skipped when the download fails",
    "slow": "a call of 20 s or more (tests/slow_tests.txt); deselect with -m 'not slow'",
}


def pytest_configure(config):
    for name, text in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")
    # Testing an installed wheel (CI): piaso must come from the installation. It is imported
    # here, before any test module, because some modules put the repository on sys.path, and
    # a package imported from the checkout has no compiled extension and is not what ships.
    if os.environ.get("PIASO_TEST_INSTALLED", "").strip() == "1":
        import piaso
        where = Path(piaso.__file__).resolve()
        if REPO_ROOT in where.parents:
            raise pytest.UsageError(
                f"PIASO_TEST_INSTALLED=1, but piaso is imported from the checkout ({where}). Run "
                f"the pytest command (not python -m pytest, which puts the current folder first).")
        print(f"testing the installed piaso {piaso.__version__} at {where.parent}")


def _slow_tests():
    """file::test names from tests/slow_tests.txt (absent from the public tree: it names
    tests of unpublished modules, which publish.sh removes)."""
    path = Path(__file__).with_name("slow_tests.txt")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return frozenset()
    return frozenset(s.strip() for s in lines if s.strip() and not s.lstrip().startswith("#"))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """requires_e18 is data(E18_CYTOME); the score() fixture may download (network); the
    tests listed in slow_tests.txt are slow."""
    slow = _slow_tests()
    for item in items:
        if item.get_closest_marker("requires_e18") is not None:
            item.add_marker(pytest.mark.data(E18_CYTOME))
        if "scored_adata" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.network)
        if slow:
            path = getattr(item, "path", None) or Path(str(item.fspath))
            name = f"{Path(path).name}::{getattr(item, 'originalname', item.name)}"
            if name in slow:
                item.add_marker(pytest.mark.slow)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    for mark in item.iter_markers("data"):
        for path in mark.args:
            ensure(path, mark.kwargs.get("what", "test data"))


def pytest_unconfigure(config):
    if _SESSION_DIR is not None:
        shutil.rmtree(_SESSION_DIR, ignore_errors=True)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Yield a fresh temporary directory; clean it up on teardown.

    Equivalent to pytest's built-in ``tmp_path`` but returns a string and
    matches the pattern several test files were already using locally.
    """
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_cytome(tmp_dir):
    """A fresh copy of the E18 cytome at a temp path. Mutate freely.

    Useful for tests that need to mutate a cytome (e.g. delete embeddings,
    filter cells) without polluting the session's copy. Returns the path.
    """
    src = ensure(E18_CYTOME, "the E18 store")
    return copy_store(src, os.path.join(tmp_dir, "work.cytome"))


@pytest.fixture
def subset_cytome(tmp_dir):
    """Open a small subset of the E18 cytome and yield the Dataset.

    Defaults to the first 500 cells. Closes the dataset on teardown.
    Replaces the ad-hoc helper that test_bug5_cytome_functions.py and
    test_subset_mismatch.py were both implementing.
    """
    src = ensure(E18_CYTOME, "the E18 store")
    import cytome
    from cytome.io.subset import subset
    ds = cytome.open(src)
    try:
        keep = np.arange(min(500, ds.n_cells))
        out_path = os.path.join(tmp_dir, "subset.cytome")
        out = subset(ds, keep, out_path, include_fragments=False)
    finally:
        ds.close()
    yield out
    out.close()


@pytest.fixture(autouse=True)
def _isolate_matplotlib_rcparams():
    """Undo any global matplotlib state a test leaves behind.

    `piaso.settings.set_figure_params` writes into `matplotlib.rcParams`, which
    is process-global. A test that calls it — the journal-style tests do,
    deliberately, since that is what they are testing — changes the default
    figure size for every test that runs afterwards.

    That produced a failure which only appeared in full-suite order and passed
    in isolation: `test_scatter_hexbin_colorbar_does_not_squeeze` asserts the
    colorbar is height-matched to the main axes, and with a different default
    figsize the equal-aspect adjustment shrank the axes to 0.40 of the figure
    while the divider colorbar kept its 0.79. The assertion was right; the
    figure it measured had been set up by an unrelated test.

    Snapshotting and restoring is cheap and fixes the whole class, rather than
    this one instance of it.
    """
    import matplotlib as mpl
    saved = mpl.rcParams.copy()
    try:
        yield
    finally:
        # `rcParams.update` on a full copy restores every key, including ones
        # the test added; `rcdefaults()` would instead discard the suite's own
        # configuration.
        mpl.rcParams.update(saved)


@pytest.fixture(scope="session")
def scored_adata():
    """A small real dataset carrying `infog` and an embedding, for score()."""
    piaso = pytest.importorskip("piaso")
    try:
        adata = piaso.data.load_dataset("adult_cortex_multiome_rna")
    except Exception as exc:                     # not cached on this machine
        pytest.skip(f"adult_cortex_multiome_rna unavailable: {exc}")
    adata = adata[:, :4000].copy()
    piaso.tl.infog(adata, layer="raw", n_top_genes=1500, verbosity=0)
    piaso.tl.runSVD(adata, layer="infog", n_components=20, key_added="X_svd")
    return adata, list(adata.var_names[:8])


def make_column_major(ds, X, matrix_name, chunk_cols=None):
    """Rewrite ``matrix_name`` as CSC-only, the importer's default layout.

    Shared by every test that needs to prove a reader handles both layouts.
    A column-major matrix has `n_chunks = 0` and `has_csc = 1`, so anything
    that iterates row chunks raises instead of silently returning nothing.
    """
    import numpy as _np
    import scipy.sparse as _sp

    csc = _sp.csc_matrix(X)
    n_cols = X.shape[1]
    # Write the blobs in the dtype `matrix_meta` declares. Readers size the
    # data buffer from that column, so a float32 blob under a float64
    # declaration is read as half as many values and indexes off the end --
    # which is a fixture bug that looks exactly like an engine bug.
    declared = ds._conn.execute(
        "SELECT dtype FROM matrix_meta WHERE matrix_name=?", (matrix_name,)
    ).fetchone()
    dtype = _np.dtype(declared[0]) if declared else _np.dtype("float32")
    width = int(chunk_cols or max(1, n_cols // 3 or 1))
    ds._conn.execute("DELETE FROM matrix_chunks WHERE matrix_name=?", (matrix_name,))
    ds._conn.execute("DELETE FROM matrix_csc_chunks WHERE matrix_name=?", (matrix_name,))
    n_chunks = 0
    for k, start in enumerate(range(0, n_cols, width)):
        end = min(start + width, n_cols)
        sub = csc[:, start:end].tocsc()
        ds._conn.execute(
            "INSERT INTO matrix_csc_chunks (matrix_name, chunk_idx, col_start, col_end,"
            " n_nonzero, data_blob, indices_blob, indptr_blob, dtype, compression) "
            "VALUES (?,?,?,?,?,?,?,?,?,'none')",
            (matrix_name, k, start, end, int(sub.nnz),
             sub.data.astype(dtype).tobytes(),
             sub.indices.astype(_np.int32).tobytes(),
             sub.indptr.astype(_np.int32).tobytes(), str(dtype)))
        n_chunks = k + 1
    ds._conn.execute(
        "UPDATE matrix_meta SET n_chunks=0, has_csc=1, csc_chunk_size=?, csc_n_chunks=? "
        "WHERE matrix_name=?", (width, n_chunks, matrix_name))
    ds._conn.commit()
