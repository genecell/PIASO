"""Reading data in, and writing it to a cytome.

Named for what it does rather than for the format it currently supports: today
that is Cell Ranger output, and other platforms belong here as they arrive.
Not ``_read_data`` -- half of this module does not read into memory, it writes
a file, and a name covering one verb would misdescribe the other. ``readwrite``
is what scanpy calls the equivalent module, so it should be familiar.

Two entry points, one input format so far. Which you want depends on where the
data should end up, and that is the only axis:

    piaso.pp.read_10x(path)                  -> AnnData, in memory
    piaso.pp.importCellRanger(path, output)  -> writes a .cytome, returns it

They live together because users arrive with the same thing -- Cell Ranger
output -- and have to pick. They are **not** one function, deliberately. One
returns an in-memory object and has no side effects; the other writes a file,
can merge several samples into it, and reaches for the Rust fragment importer
when ATAC fragments are present. Collapsing them would make the return type
depend on a keyword, which is the shape of bug this project has spent a release
removing.

Input detection is shared, so both accept what the other does and neither
reports a misleading error (e.g. ``read_10x`` on a Cell Ranger ``outs/``
folder finds ``filtered_feature_bc_matrix.h5`` rather than complaining that
``matrix.mtx`` is missing).
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse as sp


#: What a path points at. Cell Ranger writes an ``outs/`` folder containing
#: both a ``.h5`` and an MTX directory, so "a directory" is ambiguous until you
#: look inside.
def _detect_10x_input(path):
    """Classify ``path`` as ``'h5'``, ``'mtx'``, ``'outs'`` or raise.

    Returns ``(kind, resolved_path)``. For ``'outs'`` the resolved path is the
    folder itself -- the caller decides whether to read the ``.h5`` inside or
    hand the whole folder to the cytome importer.
    """
    p = Path(path)
    if p.is_file():
        if p.suffix in (".h5", ".hdf5"):
            return "h5", p
        raise ValueError(
            f"{p.name}: expected a Cell Ranger .h5 file, an MTX directory, or a "
            f"Cell Ranger output folder. For .h5ad use anndata.read_h5ad; for "
            f".cytome use cytome.open."
        )
    if not p.is_dir():
        raise FileNotFoundError(f"No such file or directory: {p}")

    # An MTX directory has the matrix at its top level.
    for cand in ("matrix.mtx", "matrix.mtx.gz"):
        if (p / cand).is_file():
            return "mtx", p
    # A Cell Ranger outs/ folder has the .h5 and/or the MTX subdirectory.
    for cand in ("filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5",
                 "filtered_feature_bc_matrix", "raw_feature_bc_matrix"):
        if (p / cand).exists():
            return "outs", p
    raise FileNotFoundError(
        f"{p}: not a Cell Ranger input. Expected matrix.mtx[.gz] (MTX "
        f"directory), or filtered_feature_bc_matrix.h5 / "
        f"filtered_feature_bc_matrix/ (Cell Ranger outs folder). "
        f"Found: {sorted(os.listdir(p))[:8]}"
    )


def _h5_inside_outs(folder):
    """The matrix file inside a Cell Ranger ``outs/`` folder, filtered first."""
    for cand in ("filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5"):
        if (Path(folder) / cand).is_file():
            return Path(folder) / cand
    for cand in ("filtered_feature_bc_matrix", "raw_feature_bc_matrix"):
        if (Path(folder) / cand).is_dir():
            return Path(folder) / cand
    return None


#: Cell Ranger ``feature_type`` strings, keyed by the short name callers use.
_MODALITY_FEATURE_TYPES = {
    "rna": ("Gene Expression",),
    "atac": ("Peaks",),
    "adt": ("Antibody Capture",),
    "hto": ("Multiplexing Capture", "Antibody Capture"),
    "crispr": ("CRISPR Guide Capture",),
}


def _resolve_feature_types(modality):
    """Map ``modality=`` to the Cell Ranger ``feature_type`` values to keep."""
    if modality in (None, "all"):
        return None
    if isinstance(modality, str):
        key = modality.lower()
        if key in _MODALITY_FEATURE_TYPES:
            return _MODALITY_FEATURE_TYPES[key]
        # Allow the Cell Ranger string itself, e.g. "Gene Expression".
        return (modality,)
    return tuple(modality)


def _make_unique(names):
    """Append ``-1``, ``-2``, ... to duplicates, in first-seen order.

    Cell Ranger gene *symbols* are not unique (gene *ids* are). Leaving
    duplicates in ``var_names`` makes every downstream ``adata[:, gene]``
    ambiguous, so this mirrors what users expect from the ecosystem.
    """
    names = np.asarray(names, dtype=object)
    seen: dict = {}
    out = np.empty(len(names), dtype=object)
    for i, n in enumerate(names):
        if n not in seen:
            seen[n] = 0
            out[i] = n
        else:
            seen[n] += 1
            out[i] = f"{n}-{seen[n]}"
    return out.astype(str)


def _build_anndata(X, obs_names, var, var_names_key, make_unique, dtype):
    from anndata import AnnData

    var = var.copy()
    if var_names_key not in var.columns:
        raise KeyError(
            f"var_names={var_names_key!r} is not available; the file provides "
            f"{sorted(var.columns)}. Use 'gene_symbols' or 'gene_ids'."
        )
    names = var[var_names_key].to_numpy()
    if make_unique:
        names = _make_unique(names)
    var.index = pd.Index(np.asarray(names, dtype=str), name=None)

    adata = AnnData(X=X.astype(dtype), obs=pd.DataFrame(index=pd.Index(obs_names, dtype=str)), var=var)
    return adata


def read_10x_h5(
    path,
    modality: str = "rna",
    var_names: str = "gene_symbols",
    make_unique: bool = True,
    dtype: str = "float32",
    genome: Optional[str] = None,
):
    """Read a Cell Ranger HDF5 matrix into an :class:`~anndata.AnnData`.

    Parameters
    ----------
    path
        Path to ``*_feature_bc_matrix.h5`` (filtered or raw).
    modality
        Which features to keep: ``'rna'`` (default, ``Gene Expression``),
        ``'atac'`` (``Peaks``), ``'adt'``, ``'crispr'``, a literal Cell Ranger
        ``feature_type`` string, or ``'all'`` to keep everything. Multiome files
        contain more than one; with ``'all'`` the type is kept in
        ``.var['feature_types']``.
    var_names
        ``'gene_symbols'`` (default) or ``'gene_ids'``.
    make_unique
        Disambiguate repeated symbols with ``-1``, ``-2``, ... Gene symbols are
        not unique in Cell Ranger references.
    dtype
        dtype of ``.X``. The file stores integer counts; the default float32
        keeps them exact well past any realistic UMI count while matching what
        downstream numerical code expects.
    genome
        Keep only features from this genome. Only meaningful for barnyard
        references; raises if the file has no such genome.

    Returns
    -------
    AnnData
        ``n_cells x n_features``, raw counts in ``.X``, with ``gene_ids``,
        ``feature_types``, ``genome`` and (for peaks) ``interval`` in ``.var``.
    """
    import h5py

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    with h5py.File(path, "r") as f:
        if "matrix" in f:
            grp = f["matrix"]
        else:
            # Cell Ranger v2 keyed the group by genome name.
            keys = [k for k in f.keys()]
            if len(keys) != 1:
                raise ValueError(
                    f"{path.name}: expected a 'matrix' group or exactly one "
                    f"genome group, found {keys}."
                )
            grp = f[keys[0]]

        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        shape = tuple(grp["shape"][:])          # (n_features, n_cells)
        barcodes = grp["barcodes"][:]

        n_features, n_cells = int(shape[0]), int(shape[1])

        # Cell Ranger stores CSC over (features, cells). The same three arrays
        # are a CSR over (cells, features) -- so the transpose is free, no
        # copy and no sort.
        X = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_features))

        fgrp = grp["features"] if "features" in grp else grp
        def _col(name, fallback=None):
            if name in fgrp:
                return np.asarray([x.decode() if isinstance(x, bytes) else x
                                   for x in fgrp[name][:]])
            return fallback

        var = pd.DataFrame({
            "gene_ids": _col("id", _col("genes")),
            "gene_symbols": _col("name", _col("gene_names")),
        })
        for optional in ("feature_type", "genome", "interval"):
            col = _col(optional)
            if col is not None:
                var["feature_types" if optional == "feature_type" else optional] = col

    obs_names = np.asarray([b.decode() if isinstance(b, bytes) else b for b in barcodes])

    if genome is not None:
        if "genome" not in var.columns:
            raise ValueError(f"{path.name} records no genome; drop genome=.")
        keep = var["genome"].to_numpy() == genome
        if not keep.any():
            raise ValueError(
                f"genome={genome!r} matches no features; file has "
                f"{sorted(set(var['genome']))}."
            )
        X, var = X[:, keep], var.loc[keep].reset_index(drop=True)

    wanted = _resolve_feature_types(modality)
    if wanted is not None:
        if "feature_types" not in var.columns:
            # Single-modality file from an older Cell Ranger: nothing to select.
            pass
        else:
            keep = np.isin(var["feature_types"].to_numpy(), wanted)
            if not keep.any():
                raise ValueError(
                    f"modality={modality!r} selected no features; {path.name} "
                    f"contains {sorted(set(var['feature_types']))}. Pass "
                    f"modality='all' to keep everything."
                )
            X, var = X[:, keep], var.loc[keep].reset_index(drop=True)

    return _build_anndata(X, obs_names, var, var_names, make_unique, dtype)


def read_10x_mtx(
    path,
    modality: str = "rna",
    var_names: str = "gene_symbols",
    make_unique: bool = True,
    dtype: str = "float32",
    prefix: str = "",
):
    """Read a Cell Ranger MTX directory into an :class:`~anndata.AnnData`.

    Expects ``matrix.mtx[.gz]`` plus ``features.tsv[.gz]`` (or the v2
    ``genes.tsv[.gz]``) and ``barcodes.tsv[.gz]``. Arguments match
    :func:`read_10x_h5`; ``prefix`` handles files named e.g.
    ``sample_matrix.mtx.gz``.
    """
    from scipy.io import mmread

    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    def _find(*candidates):
        for c in candidates:
            for suffix in ("", ".gz"):
                p = path / f"{prefix}{c}{suffix}"
                if p.is_file():
                    return p
        raise FileNotFoundError(
            f"{path}: none of {candidates} found (with or without .gz). "
            f"Directory contains: {sorted(os.listdir(path))[:10]}"
        )

    mtx_path = _find("matrix.mtx")
    bc_path = _find("barcodes.tsv")
    feat_path = _find("features.tsv", "genes.tsv")

    def _read_tsv(p):
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt") as fh:
            return [line.rstrip("\n").split("\t") for line in fh]

    opener = gzip.open if mtx_path.suffix == ".gz" else open
    with opener(mtx_path, "rb") as fh:
        X = mmread(fh)                       # (n_features, n_cells)
    X = sp.csr_matrix(X.T)                   # -> (n_cells, n_features)

    obs_names = np.asarray([r[0] for r in _read_tsv(bc_path)])
    feats = _read_tsv(feat_path)
    ncol = max(len(r) for r in feats)
    var = pd.DataFrame({
        "gene_ids": [r[0] for r in feats],
        "gene_symbols": [r[1] if len(r) > 1 else r[0] for r in feats],
    })
    if ncol > 2:
        var["feature_types"] = [r[2] if len(r) > 2 else "Gene Expression" for r in feats]

    wanted = _resolve_feature_types(modality)
    if wanted is not None and "feature_types" in var.columns:
        keep = np.isin(var["feature_types"].to_numpy(), wanted)
        if not keep.any():
            raise ValueError(
                f"modality={modality!r} selected no features; {path} contains "
                f"{sorted(set(var['feature_types']))}. Pass modality='all'."
            )
        X, var = X[:, keep], var.loc[keep].reset_index(drop=True)

    return _build_anndata(X, obs_names, var, var_names, make_unique, dtype)


def read_10x(path, **kwargs):
    """Read Cell Ranger output, dispatching on what ``path`` points at.

    A ``.h5`` file goes to :func:`read_10x_h5`; a directory goes to
    :func:`read_10x_mtx`. Keyword arguments are forwarded.

        adata = piaso.pp.read_10x("filtered_feature_bc_matrix.h5")
        adata = piaso.pp.read_10x("filtered_feature_bc_matrix/")

    To read a Cell Ranger matrix into a cytome instead of an AnnData, use
    ``cytome.from_10x_h5(path, output)`` -- writing a file is a different
    operation and keeps its own function.
    """
    kind, resolved = _detect_10x_input(path)
    if kind == "h5":
        return read_10x_h5(resolved, **kwargs)
    if kind == "mtx":
        return read_10x_mtx(resolved, **kwargs)
    # A Cell Ranger outs/ folder: read the matrix inside it. Reporting
    # "matrix.mtx not found" here, with the .h5 sitting beside it, is the kind
    # of error that sends people to the issue tracker.
    inner = _h5_inside_outs(resolved)
    if inner is None:                       # pragma: no cover - detector guarantees one
        raise FileNotFoundError(f"{resolved}: no matrix found inside the outs folder")
    return (read_10x_h5 if inner.is_file() else read_10x_mtx)(inner, **kwargs)


def importCellRanger(
    path,
    output: str | Path,
    sample_name=None,
    modality: str = "both",
    genome: Optional[str] = None,
    keep_chroms: str = "standard",
    min_fragments: int = 0,
    threads: int = 8,
    tss_bed: Optional[str] = None,
    compression: str = "lz4",
    build_index: bool = True,
    rust_binary: Optional[str] = None,
    verbose: bool = True,
    force: bool = False,
):
    """Build a Cytome dataset from Cell Ranger output(s), with Rust fragment import.

    ``cytome.from_cellranger`` writes the count matrices (RNA genes and/or ATAC peaks per
    ``modality``); when ATAC is requested and ``atac_fragments.tsv.gz`` is present,
    :func:`piaso.pp.importFragments` imports the fragments with the Rust binary (inline tile
    quantification + optional TSS enrichment). **For multiple folders the count matrices are
    merged first and then ALL fragment files are imported in a single Rust k-way merge** (the
    merged barcodes are suffixed ``{barcode}-{i}`` per library so colliding Cell Ranger barcodes
    map to the right merged cells) — replacing the older per-library-then-merge fragment path.

    Parameters
    ----------
    path : str | Path | list of (str | Path)
        A single Cell Ranger output folder, or a list of folders (merged into one dataset).
    output : str | Path
        Output ``.cytome`` path.
    sample_name : str | list of str, optional
        Sample id(s) written to ``cells.sample_id`` (one per folder for a list).
    modality : {"both", "rna", "atac"}, default "both"
        ``"rna"`` → RNA genes + counts only (**no ATAC**, no fragments).
        ``"atac"`` → ATAC peaks + Rust fragments only (no RNA).
        ``"both"`` → RNA + ATAC peaks + Rust fragments.
    genome : str, **required for ATAC** (no default)
        Reference for the Rust importer's tile quantification — ``'hg38'``/``'hg19'``/``'mm10'``/
        ``'mm39'`` or a ``.fai``/``.chrom.sizes`` path. Required whenever fragments are imported
        (``modality="both"``/``"atac"``); only ``modality="rna"`` may omit it. There is **no
        default** — naming the wrong genome (or relying on a silent default) builds the tile grid
        for the wrong assembly and corrupts the tiles, so the genome must be stated explicitly.
    keep_chroms : {"standard", "all"}, default "standard"
        Forwarded to ``cytome.from_cellranger`` (drops non-standard scaffolds from ATAC peaks).
    min_fragments : int, default 0
        Per-barcode fragment floor for the Rust importer. ``0`` keeps every cell already present
        from the Cell Ranger filtered matrix (recommended; the matrix is already QC'd).
    threads : int, default 8
        Threads for the Rust importer.
    tss_bed : str, optional
        TSS BED for per-cell TSS enrichment during fragment import.
    compression : str, default "lz4"
        Fragment chunk compression (``'lz4'``/``'zlib'``/``'zstd'``).
    build_index : bool, default True
        Build the peak / fragment spatial index.
    rust_binary : str, optional
        Explicit path to ``cytome-import-fragments`` (auto-discovered if None).
    verbose : bool, default True

    Returns
    -------
    cytome.Dataset

    Examples
    --------
    >>> import piaso
    >>> ds = piaso.pp.importCellRanger(
    ...     ["run/E15Satb2_ctrl", "run/E15Satb2_het", "run/E15Satb2_cko"],
    ...     output="E15Satb2.cytome",
    ...     sample_name=["ctrl", "het", "cko"],
    ...     genome="mm10",
    ... )
    >>> # RNA-only
    >>> ds = piaso.pp.importCellRanger("run/outs", "rna.cytome", modality="rna")
    """
    import cytome
    from ._importFragments import importFragments

    modality = str(modality).lower()
    if modality not in ("both", "rna", "atac"):
        raise ValueError(f"modality must be 'both', 'rna' or 'atac', got {modality!r}.")
    want_frag = modality in ("both", "atac")

    # ``genome`` has no default: importing ATAC fragments without naming the genome would
    # silently build the tile grid for the wrong assembly (e.g. mm10 data quantified on an hg38
    # grid → corrupt tiles + a TSS panic). Require it explicitly whenever fragments are imported.
    if want_frag and not genome:
        raise ValueError(
            "genome is required when importing ATAC fragments (modality='both'/'atac'). "
            "Pass genome='mm10'/'mm39' for mouse, 'hg38'/'hg19' for human, or a "
            ".fai/.chrom.sizes path. (Only modality='rna' may omit it.)")

    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]

    # A single .h5 or MTX directory is a reasonable thing to point this at, and
    # it used to fail with "No 'filtered_feature_bc_matrix.h5' in <the .h5
    # file>". cytome.from_10x_h5 is the right tool for that shape; say so
    # rather than reporting a folder that was never expected to exist.
    for _p in paths:
        _kind, _ = _detect_10x_input(_p)
        if _kind in ("h5", "mtx"):
            raise ValueError(
                f"{_p.name} is a {'matrix file' if _kind == 'h5' else 'MTX directory'}, "
                f"not a Cell Ranger output folder. importCellRanger builds a cytome "
                f"from an outs/ folder (so it can also find ATAC fragments).\n"
                f"  For a single matrix -> cytome:  cytome.from_10x_h5({_p.name!r}, output)\n"
                f"  For a single matrix -> AnnData: piaso.pp.read_10x({_p.name!r})"
            )
    if sample_name is None:
        names: List[Optional[str]] = [None] * len(paths)
    elif isinstance(sample_name, (str, Path)):
        if len(paths) > 1:
            raise ValueError(
                "Multiple folders given but a single sample_name; pass a list of length "
                f"{len(paths)} (one per folder).")
        names = [str(sample_name)]
    else:
        names = [None if s is None else str(s) for s in sample_name]
        if len(names) != len(paths):
            raise ValueError(
                f"sample_name list length ({len(names)}) != number of folders ({len(paths)}).")

    def _frag(folder):
        f = Path(folder) / "atac_fragments.tsv.gz"
        return str(f) if f.exists() else None

    def _build_one(folder, out_path, label):
        ds = cytome.from_cellranger(
            folder, str(out_path), sample_name=label, modalities=modality,
            import_fragments=False, keep_chroms=keep_chroms, build_index=build_index,
            force=force)
        ds.close()
        if want_frag:
            ff = _frag(folder)
            if ff is not None:
                importFragments(cytome=str(out_path), fragments=ff, genome=genome,
                                min_fragments=min_fragments, threads=threads, tss_bed=tss_bed,
                                compression=compression, rust_binary=rust_binary,
                                verbose=verbose)
            elif verbose:
                print(f"[importCellRanger] no atac_fragments.tsv.gz in {folder}; "
                      f"counts/peaks only.")

    if len(paths) == 1:
        _build_one(paths[0], output, names[0])
        return cytome.open(str(output))

    # Multiple folders → build the MERGED count matrices first (no fragments), then import ALL
    # fragment files in ONE Rust k-way merge. Cell Ranger barcodes collide across libraries and
    # cytome.merge keeps raw barcodes (rows ordered by library + a sample_id column), so we suffix
    # each library's barcodes with its file index (``{barcode}-{i}``) and import with the matching
    # ``barcode_suffixes`` — the importer maps fragment ``{barcode}`` in file i → ``{barcode}-{i}``.
    labels = [nm if nm is not None else f"sample{i + 1}" for i, nm in enumerate(names)]
    out = Path(output)
    cytome.from_cellranger([str(p) for p in paths], str(out), sample_name=labels,
                           modalities=modality, import_fragments=False,
                           keep_chroms=keep_chroms, build_index=build_index, force=force)
    if not want_frag:
        return cytome.open(str(out))

    have = [(i, _frag(p)) for i, p in enumerate(paths)]
    have = [(i, f) for i, f in have if f is not None]
    if not have:
        if verbose:
            print("[importCellRanger] no atac_fragments.tsv.gz in any folder; counts/peaks only.")
        return cytome.open(str(out))
    if len(have) < len(paths) and verbose:
        miss = [str(paths[i]) for i, p in enumerate(paths) if _frag(p) is None]
        print(f"[importCellRanger] no atac_fragments.tsv.gz in {miss}; those libraries get peaks only.")

    # suffix merged barcodes per library so the single k-way import can disambiguate collisions
    ds = cytome.open(str(out))
    cells = ds.cells.to_pandas()
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    suf = cells["sample_id"].map(lab_to_idx).astype("Int64").astype(str)
    cells["barcode"] = cells["barcode"].astype(str) + "-" + suf
    ds.set_entity("cells", cells)
    ds.close()

    importFragments(cytome=str(out), fragments=[f for _, f in have],
                    barcode_suffixes=[str(i) for i, _ in have], genome=genome,
                    min_fragments=min_fragments, threads=threads, tss_bed=tss_bed,
                    compression=compression, rust_binary=rust_binary, verbose=verbose)
    return cytome.open(str(out))
