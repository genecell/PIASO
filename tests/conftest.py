"""Shared pytest fixtures and constants for the PIASO test suite.

Centralises the E18 cytome path (15 test files used to redefine it) and
the small fixtures (``tmp_dir``, ``subset_cytome``) that two or more test
files were copy-pasting.

How to use
----------
- For path: ``from conftest import E18_CYTOME``
- For skip-if-missing: decorate the test (or set module-level
  ``pytestmark``) with ``@pytest.mark.requires_e18`` — pytest will
  skip the test automatically when the file is missing.
- For temp dirs: take ``tmp_dir`` as a fixture argument.
- For a one-shot subset of E18: take ``subset_cytome`` as a fixture
  argument (returns an open Dataset that is closed on teardown).
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------

# Repo-relative path to the E18 cytome built by the snakemake pipeline.
# Computed from this file so the tests work whichever clone of the repo
# pytest is run from.
E18_CYTOME = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "snakemake",
        "e18",
        "pipeline.cytome",
    )
)


# ---------------------------------------------------------------------
# Marker registration: @pytest.mark.requires_e18
# ---------------------------------------------------------------------

def pytest_configure(config):
    """Register the requires_e18 marker so it doesn't trigger warnings."""
    config.addinivalue_line(
        "markers",
        "requires_e18: skip if the E18 cytome built by the snakemake "
        "pipeline is not present at <repo>/results/snakemake/e18/pipeline.cytome",
    )


def pytest_collection_modifyitems(config, items):
    """Apply the skip-if-missing rule to every requires_e18 item."""
    if os.path.exists(E18_CYTOME):
        return
    skip_no_e18 = pytest.mark.skip(reason=f"E18 cytome not found at {E18_CYTOME}")
    for item in items:
        if "requires_e18" in item.keywords:
            item.add_marker(skip_no_e18)


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
    filter cells) without polluting the canonical pipeline output.
    Returns the path string.
    """
    if not os.path.exists(E18_CYTOME):
        pytest.skip(f"E18 cytome not found at {E18_CYTOME}")
    work = os.path.join(tmp_dir, "work.cytome")
    shutil.copy2(E18_CYTOME, work)
    return work


@pytest.fixture
def subset_cytome(tmp_dir):
    """Open a small subset of the E18 cytome and yield the Dataset.

    Defaults to the first 500 cells. Closes the dataset on teardown.
    Replaces the ad-hoc helper that test_bug5_cytome_functions.py and
    test_subset_mismatch.py were both implementing.
    """
    if not os.path.exists(E18_CYTOME):
        pytest.skip(f"E18 cytome not found at {E18_CYTOME}")
    import cytome
    from cytome.io.subset import subset
    ds = cytome.open(E18_CYTOME)
    try:
        keep = np.arange(min(500, ds.n_cells))
        out_path = os.path.join(tmp_dir, "subset.cytome")
        out = subset(ds, keep, out_path, include_fragments=False)
    finally:
        ds.close()
    yield out
    out.close()
