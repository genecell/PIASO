"""Cell Ranger input: readers, the cytome importer, and the boundary between them.

Before these existed, ``piaso.data.load_dataset`` handled ``10x_h5`` by calling
``scanpy.read_10x_h5``. scanpy is an optional extra, and **six of the nine
datasets in PIASO's own registry are 10x_h5** -- so the first line of an RNA
tutorial required a package the rest of the workflow does not.

The bar for replacing a widely-used reader is that it produce the same thing,
so the equivalence test below compares against scanpy directly when it is
available, and is skipped (not silently passed) when it is not.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse as sp
from scipy.io import mmwrite

import piaso


def _write_10x_h5(path, X, gene_ids, symbols, barcodes, feature_types=None,
                  genome="testgenome", intervals=None):
    """Write a Cell Ranger v3 style HDF5: CSC over (features, cells)."""
    import h5py

    csc = sp.csc_matrix(X.T)          # (n_features, n_cells)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=csc.data.astype(np.int32))
        g.create_dataset("indices", data=csc.indices.astype(np.int64))
        g.create_dataset("indptr", data=csc.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(csc.shape, dtype=np.int64))
        g.create_dataset("barcodes", data=np.array([b.encode() for b in barcodes]))
        fg = g.create_group("features")
        fg.create_dataset("id", data=np.array([x.encode() for x in gene_ids]))
        fg.create_dataset("name", data=np.array([x.encode() for x in symbols]))
        ft = feature_types or ["Gene Expression"] * len(gene_ids)
        fg.create_dataset("feature_type", data=np.array([x.encode() for x in ft]))
        fg.create_dataset("genome", data=np.array([genome.encode()] * len(gene_ids)))
        if intervals is not None:
            fg.create_dataset("interval", data=np.array([x.encode() for x in intervals]))


@pytest.fixture
def rna_h5(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.poisson(3, (12, 8)).astype(np.int32)
    p = tmp_path / "rna.h5"
    _write_10x_h5(p, X, [f"ENSG{i:05d}" for i in range(8)],
                  ["A", "B", "C", "A", "D", "E", "B", "F"],   # deliberate dups
                  [f"CELL{i}" for i in range(12)])
    return p, X


@pytest.fixture
def multiome_h5(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.poisson(2, (6, 7)).astype(np.int32)
    p = tmp_path / "multi.h5"
    _write_10x_h5(
        p, X,
        [f"F{i}" for i in range(7)],
        ["G1", "G2", "G3", "chr1:100-200", "chr1:300-400", "chr2:1-2", "chr2:5-9"],
        [f"C{i}" for i in range(6)],
        feature_types=["Gene Expression"] * 3 + ["Peaks"] * 4,
        intervals=["NA", "NA", "NA", "chr1:100-200", "chr1:300-400", "chr2:1-2", "chr2:5-9"],
    )
    return p, X


def test_reads_counts_exactly(rna_h5):
    p, X = rna_h5
    a = piaso.pp.read_10x(p)
    assert a.shape == X.shape
    np.testing.assert_array_equal(a.X.toarray(), X)
    assert sp.issparse(a.X), "densifying a count matrix defeats the point"


def test_orientation_is_cells_by_genes(rna_h5):
    """Cell Ranger stores features x cells; getting this backwards is silent.

    The arrays are reinterpreted as CSR over (cells, features) rather than
    transposed, so a mistake here would not raise -- it would produce a matrix
    of the wrong shape only when n_cells == n_features.
    """
    p, X = rna_h5
    a = piaso.pp.read_10x(p)
    assert a.n_obs == 12 and a.n_vars == 8
    np.testing.assert_array_equal(a.X.toarray()[0], X[0])


def test_duplicate_symbols_are_made_unique(rna_h5):
    p, _ = rna_h5
    a = piaso.pp.read_10x(p)
    assert a.var_names.is_unique
    assert list(a.var_names)[:4] == ["A", "B", "C", "A-1"]
    kept = piaso.pp.read_10x(p, make_unique=False)
    assert not kept.var_names.is_unique


def test_var_names_gene_ids(rna_h5):
    p, _ = rna_h5
    a = piaso.pp.read_10x(p, var_names="gene_ids")
    assert list(a.var_names)[0] == "ENSG00000"
    with pytest.raises(KeyError, match="var_names"):
        piaso.pp.read_10x(p, var_names="not_a_column")


def test_multiome_modality_split_is_exact(multiome_h5):
    p, _ = multiome_h5
    rna = piaso.pp.read_10x(p, modality="rna")
    atac = piaso.pp.read_10x(p, modality="atac")
    both = piaso.pp.read_10x(p, modality="all")
    assert rna.n_vars == 3 and atac.n_vars == 4
    assert rna.n_vars + atac.n_vars == both.n_vars
    assert set(both.var["feature_types"]) == {"Gene Expression", "Peaks"}
    assert "interval" in atac.var.columns


def test_absent_modality_raises_naming_what_is_there(rna_h5):
    """The error must say what the file *does* contain.

    An empty AnnData would be the unhelpful alternative: downstream code fails
    much later with a shape error and no clue that modality= was the problem.
    """
    p, _ = rna_h5
    with pytest.raises(ValueError, match="Gene Expression"):
        piaso.pp.read_10x(p, modality="atac")


def test_dispatch_by_path_type(rna_h5, tmp_path):
    p, X = rna_h5
    # directory -> mtx
    d = tmp_path / "mtx"
    d.mkdir()
    mmwrite(str(d / "matrix.mtx"), sp.coo_matrix(X.T))
    with open(d / "barcodes.tsv", "w") as f:
        f.write("\n".join(f"CELL{i}" for i in range(12)) + "\n")
    with open(d / "features.tsv", "w") as f:
        f.write("\n".join(f"ENSG{i:05d}\tSYM{i}\tGene Expression" for i in range(8)) + "\n")
    a = piaso.pp.read_10x(d)
    np.testing.assert_array_equal(a.X.toarray(), X)
    assert list(a.var_names) == [f"SYM{i}" for i in range(8)]

    # a file that is neither .h5 nor a directory
    stray = tmp_path / "notes.txt"
    stray.write_text("")
    with pytest.raises(ValueError, match="MTX directory"):
        piaso.pp.read_10x(stray)
    with pytest.raises(FileNotFoundError):
        piaso.pp.read_10x(tmp_path / "missing.h5")


def test_gzipped_mtx(rna_h5, tmp_path):
    p, X = rna_h5
    d = tmp_path / "mtxgz"
    d.mkdir()
    mmwrite(str(d / "matrix.mtx"), sp.coo_matrix(X.T))
    with open(d / "matrix.mtx", "rb") as src, gzip.open(d / "matrix.mtx.gz", "wb") as dst:
        dst.write(src.read())
    (d / "matrix.mtx").unlink()
    for name, rows in [("barcodes.tsv.gz", [f"CELL{i}" for i in range(12)]),
                       ("features.tsv.gz", [f"E{i}\tS{i}\tGene Expression" for i in range(8)])]:
        with gzip.open(d / name, "wt") as f:
            f.write("\n".join(rows) + "\n")
    a = piaso.pp.read_10x(d)
    np.testing.assert_array_equal(a.X.toarray(), X)


def test_matches_scanpy(rna_h5, multiome_h5):
    """The equivalence that justifies replacing scanpy on this path."""
    sc = pytest.importorskip("scanpy", reason="scanpy absent; nothing to compare")
    for p, _ in (rna_h5, multiome_h5):
        ours = piaso.pp.read_10x(p, modality="rna")
        theirs = sc.read_10x_h5(str(p))
        theirs.var_names_make_unique()
        assert ours.shape == theirs.shape
        assert list(ours.obs_names) == list(theirs.obs_names)
        assert list(ours.var_names) == list(theirs.var_names)
        A = ours.X.toarray()
        B = theirs.X.toarray() if sp.issparse(theirs.X) else np.asarray(theirs.X)
        np.testing.assert_array_equal(A, B)


def test_load_dataset_does_not_use_scanpy_for_10x():
    """The reason this module exists, pinned.

    If load_dataset reverts to scanpy, six registry datasets silently require
    an optional extra again -- and nothing else in the workflow would fail, so
    no other test would notice.
    """
    import ast
    import inspect

    from piaso.data import _datasets

    src = inspect.getsource(_datasets)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.Import):
            mod = node.names[0].name
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
        if mod and mod.split(".")[0] == "scanpy":
            pytest.fail(
                "piaso.data._datasets imports scanpy again; the 10x_h5 path "
                "must use piaso.pp.read_10x_h5 so load_dataset works on a "
                "default install."
            )


# ---------------------------------------------------------------------------
# The two entry points share input detection, so neither reports a misleading
# error about the other's input shape. Before that, read_10x pointed at a Cell
# Ranger outs/ folder said "matrix.mtx not found" with the .h5 beside it, and
# importCellRanger pointed at a .h5 reported a missing folder *inside the file*.
# ---------------------------------------------------------------------------

def test_read_10x_accepts_a_cellranger_outs_folder(rna_h5, tmp_path):
    p, X = rna_h5
    outs = tmp_path / "outs"
    outs.mkdir()
    (outs / "filtered_feature_bc_matrix.h5").write_bytes(p.read_bytes())
    (outs / "metrics_summary.csv").write_text("x\n")
    a = piaso.pp.read_10x(outs)
    np.testing.assert_array_equal(a.X.toarray(), X)


def test_importCellRanger_on_a_matrix_file_names_the_alternative(rna_h5, tmp_path):
    """A wrong-shape input should say what to use instead, not what is missing."""
    p, _ = rna_h5
    with pytest.raises(ValueError) as exc:
        piaso.pp.importCellRanger(p, tmp_path / "out.cytome", modality="rna")
    msg = str(exc.value)
    assert "not a Cell Ranger output folder" in msg
    assert "cytome.from_10x_h5" in msg and "piaso.pp.read_10x" in msg


def test_detector_rejects_an_unrelated_directory(tmp_path):
    d = tmp_path / "random"
    d.mkdir()
    (d / "notes.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="not a Cell Ranger input"):
        piaso.pp.read_10x(d)


def test_the_two_entry_points_are_separate_functions():
    """They are not one function with a return type that depends on a keyword.

    read_10x returns an AnnData and has no side effects; importCellRanger
    writes a file and returns a dataset handle. Merging them was considered and
    rejected -- see docs/discussion. This pins the decision so it is not undone
    by accident.
    """
    import inspect

    read_params = inspect.signature(piaso.pp.read_10x).parameters
    assert "output" not in read_params, (
        "read_10x grew an output= parameter; if it can write a file its return "
        "type now depends on a keyword. Use importCellRanger for that."
    )
    imp_params = inspect.signature(piaso.pp.importCellRanger).parameters
    assert "output" in imp_params, "importCellRanger must take an explicit output path"
