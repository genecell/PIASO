"""The h5ad -> cytome converter must never ship normalized values as counts.

Every published .h5ad keeps its raw UMIs somewhere different -- `X` in two of
them, `layers['raw']` in one, `layers['UMIs']` in another -- and one file has a
``.raw`` attribute holding log1p values, so the attribute name is not evidence.
A wrong guess produces a cytome that looks fine and normalizes twice.

So the converter declares the source per dataset and verifies it. These tests
pin the verification, not the conversion: they build tiny h5ad files with counts
in each of the places the real data uses.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "h5ad_to_cytome",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "h5ad_to_cytome.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _write(tmp_path, X, layers=None, name="t.h5ad"):
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(X.shape[1])]
    for k, v in (layers or {}).items():
        a.layers[k] = v
    p = tmp_path / name
    a.write_h5ad(p)
    return p


def _counts(n=40, m=12, seed=0):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix(rng.poisson(2, size=(n, m)).astype("float32"))


def test_counts_in_X_are_accepted(tmp_path):
    p = _write(tmp_path, _counts())
    info = H.verify_counts(p, "X")
    assert info["min"] >= 0 and info["source"] == "X"


def test_counts_in_a_layer_are_accepted(tmp_path):
    c = _counts()
    p = _write(tmp_path, c.multiply(0.001).tocsr(), layers={"UMIs": c})
    info = H.verify_counts(p, ("layer", "UMIs"))
    assert info["source"] == "layers['UMIs']"


def test_normalized_X_is_refused(tmp_path):
    """The failure this whole design exists to prevent."""
    c = _counts()
    norm = sp.csr_matrix(np.log1p(c.toarray() / 3.0).astype("float32"))
    p = _write(tmp_path, norm)
    with pytest.raises(SystemExit, match="NOT raw counts"):
        H.verify_counts(p, "X")


def test_negative_values_are_refused(tmp_path):
    """A scaled matrix is integer-ish nowhere and negative somewhere."""
    rng = np.random.default_rng(1)
    scaled = sp.csr_matrix(rng.normal(size=(30, 8)).astype("float32"))
    p = _write(tmp_path, scaled)
    with pytest.raises(SystemExit, match="NOT raw counts"):
        H.verify_counts(p, "X")


def test_a_missing_layer_names_what_is_present(tmp_path):
    p = _write(tmp_path, _counts(), layers={"UMIs": _counts()})
    with pytest.raises(SystemExit, match="layers present"):
        H.verify_counts(p, ("layer", "nope"))


def test_every_declared_dataset_states_its_counts_source():
    for name, spec in H.DATASETS.items():
        src = spec["counts_from"]
        assert src == "X" or (isinstance(src, tuple) and src[0] == "layer"), name
        assert spec["h5ad"] and spec["title"] and spec["species"], name


def test_derived_layers_are_skipped():
    """infog/log1p recompute in seconds and would double the upload."""
    assert {"infog", "log1p"} <= set(H.SKIP_LAYERS)
