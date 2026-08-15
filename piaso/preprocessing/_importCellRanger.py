"""``piaso.pp.importCellRanger`` — Cell Ranger → Cytome with the fast Rust fragment importer.

A one-call loader that builds the RNA/ATAC count matrices with ``cytome.from_cellranger`` and
imports ATAC fragments with the **Rust** ``cytome-import-fragments`` binary (via
:func:`piaso.pp.importFragments`) — replacing the slow pure-Python fragment path that
``cytome.from_cellranger(import_fragments=True)`` used. The ``modality`` switch selects RNA,
ATAC, or both.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


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
