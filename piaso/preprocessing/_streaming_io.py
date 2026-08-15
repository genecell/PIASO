"""
Streaming infrastructure for ATAC-seq quantification.

Components:
    - Fragment reader (auto-selects fastest available decompression)
    - Barcode index builder
    - ChunkBucketWriter (routes hits to per-chunk temp files)
    - Chunk CSR processor
    - CSR assembly (in-RAM, direct-to-h5ad, or direct-to-cytome)

Memory design:
    All components use compact representations (array.array, numpy) to avoid
    Python object overhead.  A Python tuple of 3 ints costs ~140 bytes;
    3 x array.array('i') entries cost 12 bytes — 12x less.

    When *output_file* is set, chunk CSRs are written directly to HDF5
    arrays one at a time.  Peak RAM = one chunk (~30 MB) instead of the
    full assembled matrix (~8 GB at 500k cells).

    When *output_cytome* is set, chunk CSRs are written via Cytome's
    ChunkedLayerWriter with zstd compression.
"""

import array
import gzip
import math
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome_dataset
from ..utils._cytome_compat import open_cytome_sync as _open_cytome


# ===================================================================
#  Fragment reader
# ===================================================================

def _open_fragments(
    fragment_file: str,
    chromosomes: List[str] = None,
    tabix_path: str = None,
    bgzip_path: str = None,
    pigz_path: str = None,
) -> Iterator[Tuple[str, int, int, str]]:
    """
    Iterate over a 10x fragments.tsv.gz file.

    Yields
    ------
    (chrom, start, end, barcode) for each fragment.

    The 5th column (*dup_count*) is intentionally **not** yielded — it is
    a PCR artifact and must not be used in quantification.  Cell Ranger
    already deduplicated: each line is one unique fragment (one original
    DNA molecule).

    Reader priority (auto-selected, fastest available first):

    === ===================== ====================================
     #  Reader                Why
    === ===================== ====================================
     1  pysam TabixFile       C decompression + per-chrom seeking
     2  ``tabix`` CLI         Per-chrom seeking without pysam
     3  ``bgzip -dc``         Block-aware decompression (optimal
                              for bgzipped files)
     4  ``pigz -dc``          Multi-threaded gzip decompression
     5  ``gzip -dc``          Standard single-threaded
     6  Python ``gzip``       Always available, slowest
    === ===================== ====================================
    """
    tbi_path = fragment_file + ".tbi"
    has_tbi = os.path.exists(tbi_path)

    # 1. pysam TabixFile
    try:
        import pysam

        if has_tbi:
            yield from _read_fragments_pysam(fragment_file, chromosomes)
            return
    except ImportError:
        pass

    # 2. tabix CLI
    tabix_bin = tabix_path or shutil.which("tabix")
    if has_tbi and tabix_bin:
        yield from _read_fragments_tabix_cli(
            fragment_file, chromosomes, tabix_bin=tabix_bin,
        )
        return

    # 3. bgzip -dc
    bgzip_bin = bgzip_path or shutil.which("bgzip")
    if bgzip_bin:
        yield from _read_fragments_subprocess(fragment_file, [bgzip_bin, "-dc"])
        return

    # 4. pigz -dc
    pigz_bin = pigz_path or shutil.which("pigz")
    if pigz_bin:
        yield from _read_fragments_subprocess(fragment_file, [pigz_bin, "-dc"])
        return

    # 5. gzip -dc
    if shutil.which("gzip"):
        yield from _read_fragments_subprocess(fragment_file, ["gzip", "-dc"])
        return

    # 6. Python gzip
    yield from _read_fragments_python(fragment_file)


# -------------------------------------------------------------------
#  Reader implementations
# -------------------------------------------------------------------

def _read_fragments_pysam(fragment_file, chromosomes=None):
    """Read via pysam TabixFile.  C-level decompression, per-chrom seeking."""
    import pysam

    tbx = pysam.TabixFile(fragment_file)
    try:
        contigs = chromosomes if chromosomes else tbx.contigs
        for chrom in contigs:
            if chrom not in tbx.contigs:
                continue
            for row in tbx.fetch(chrom):
                parts = row.split("\t")
                yield parts[0], int(parts[1]), int(parts[2]), parts[3]
    finally:
        tbx.close()


def _read_fragments_tabix_cli(fragment_file, chromosomes=None, tabix_bin="tabix"):
    """Read via ``tabix`` CLI.  Per-chromosome seeking without pysam."""
    if chromosomes is None:
        proc = subprocess.Popen(
            [tabix_bin, "-l", fragment_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        chromosomes = [line.strip() for line in proc.stdout if line.strip()]
        proc.wait()

    for chrom in chromosomes:
        proc = subprocess.Popen(
            [tabix_bin, fragment_file, chrom],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1 << 20,
        )
        try:
            for line in proc.stdout:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    yield parts[0], int(parts[1]), int(parts[2]), parts[3]
        finally:
            proc.stdout.close()
            proc.wait()


def _read_fragments_subprocess(fragment_file, cmd):
    """Read via a subprocess that decompresses to stdout."""
    proc = subprocess.Popen(
        cmd + [fragment_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1 << 20,
    )
    try:
        for line in proc.stdout:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                yield parts[0], int(parts[1]), int(parts[2]), parts[3]
    finally:
        proc.stdout.close()
        proc.wait()


def _read_fragments_python(fragment_file):
    """Read via Python gzip module.  Always available, slowest."""
    with gzip.open(fragment_file, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                yield parts[0], int(parts[1]), int(parts[2]), parts[3]


# ===================================================================
#  Barcode index
# ===================================================================

def _build_barcode_index(
    fragment_file: str,
    barcodes: list = None,
    barcode_file: str = None,
    allow_scan: bool = False,
    tabix_path: str = None,
    bgzip_path: str = None,
    pigz_path: str = None,
) -> Dict[str, int]:
    """
    Build barcode -> cell_idx mapping.

    Parameters
    ----------
    fragment_file : str
        Path to fragments.tsv.gz.
    barcodes : list, optional
        Explicit barcode list.  Order determines cell_idx.
    barcode_file : str, optional
        Path to file with one barcode per line.
    allow_scan : bool
        If *True* and no barcodes provided, scan the entire fragment file
        to discover barcodes (slow — reads the whole file).

    Returns
    -------
    dict
        barcode string -> integer cell index.
    """
    if barcodes is not None:
        return {bc: i for i, bc in enumerate(barcodes)}

    if barcode_file is not None:
        with open(barcode_file, "r") as fh:
            bcs = [line.strip() for line in fh if line.strip()]
        return {bc: i for i, bc in enumerate(bcs)}

    if not allow_scan:
        raise ValueError(
            "Streaming quantification requires a barcode list. "
            "Pass barcodes= (list) or barcode_file= (path to file with "
            "one barcode per line, e.g. Cell Ranger's barcodes.tsv). "
            "This avoids reading the fragment file twice."
        )

    unique: Set[str] = set()
    for _, _, _, barcode in _open_fragments(
        fragment_file,
        tabix_path=tabix_path,
        bgzip_path=bgzip_path,
        pigz_path=pigz_path,
    ):
        unique.add(barcode)
    return {bc: i for i, bc in enumerate(sorted(unique))}


# ===================================================================
#  Chunk parameters
# ===================================================================

MAX_CHUNK_FILES = 500
DEFAULT_CHUNK_SIZE = 2000


def _compute_chunk_params(n_cells: int) -> Tuple[int, int]:
    """Return ``(chunk_size, n_chunks)`` keeping *n_chunks* <= MAX_CHUNK_FILES."""
    chunk_size = max(DEFAULT_CHUNK_SIZE, math.ceil(n_cells / MAX_CHUNK_FILES))
    n_chunks = math.ceil(n_cells / chunk_size)
    return chunk_size, n_chunks


# ===================================================================
#  ChunkBucketWriter
# ===================================================================

class ChunkBucketWriter:
    """
    Buffered writer that routes (cell_idx, col_idx) hits to per-cell-chunk
    temp files.

    Each chunk covers *chunk_size* cells.  A hit for *cell_idx* goes to
    ``chunk_files[cell_idx // chunk_size]``.  Within each file hits are
    stored as ``(local_row, col)`` int32 pairs.

    File handles are opened lazily — chunks with no hits produce no file.

    Memory
    ------
    ``n_chunks x BUFFER_FLUSH x 8`` bytes worst case.
    250 chunks x 50 000 buffer ~ 100 MB.
    """

    BUFFER_FLUSH = 50_000  # pairs per chunk before flushing

    def __init__(self, n_chunks: int, chunk_size: int, tmpdir: str):
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.tmpdir = tmpdir
        self.buffers: List[array.array] = [
            array.array("i") for _ in range(n_chunks)
        ]
        self.handles: List[Optional[object]] = [None] * n_chunks

    def add(self, cell_idx: int, col_idx: int):
        """Route one hit to the appropriate chunk buffer."""
        chunk_id = cell_idx // self.chunk_size
        local_row = cell_idx % self.chunk_size
        buf = self.buffers[chunk_id]
        buf.append(local_row)
        buf.append(col_idx)
        if len(buf) >= self.BUFFER_FLUSH * 2:
            self._flush(chunk_id)

    def flush_arrays(self, cell_indices: np.ndarray, col_indices: np.ndarray):
        """Route numpy arrays of hits to chunk bucket files (vectorized).

        Partitions by chunk_id using argsort + searchsorted, then writes
        each partition directly to its chunk file.  10-50x faster than
        calling add() per hit.
        """
        if len(cell_indices) == 0:
            return

        chunk_ids = cell_indices // self.chunk_size
        local_rows = (cell_indices % self.chunk_size).astype(np.int32)
        col_arr = col_indices.astype(np.int32)

        order = np.argsort(chunk_ids, kind='stable')
        sorted_chunk_ids = chunk_ids[order]
        sorted_rows = local_rows[order]
        sorted_cols = col_arr[order]

        boundaries = np.searchsorted(sorted_chunk_ids,
                                     np.arange(self.n_chunks + 1))

        for cid in range(self.n_chunks):
            start, end = int(boundaries[cid]), int(boundaries[cid + 1])
            if start == end:
                continue
            pairs = np.empty(2 * (end - start), dtype=np.int32)
            pairs[0::2] = sorted_rows[start:end]
            pairs[1::2] = sorted_cols[start:end]
            if self.handles[cid] is None:
                path = os.path.join(self.tmpdir, f"chunk_{cid}.bin")
                self.handles[cid] = open(path, "ab")
            pairs.tofile(self.handles[cid])

    def close(self):
        """Flush all remaining buffers and close file handles."""
        for i in range(self.n_chunks):
            self._flush(i)
            if self.handles[i] is not None:
                self.handles[i].close()
                self.handles[i] = None

    def _flush(self, chunk_id: int):
        buf = self.buffers[chunk_id]
        if not buf:
            return
        if self.handles[chunk_id] is None:
            path = os.path.join(self.tmpdir, f"chunk_{chunk_id}.bin")
            self.handles[chunk_id] = open(path, "ab")
        np.array(buf, dtype=np.int32).tofile(self.handles[chunk_id])
        self.buffers[chunk_id] = array.array("i")


# ===================================================================
#  Chunk -> CSR
# ===================================================================

def _process_chunk(
    chunk_path: str,
    chunk_rows: int,
    n_cols: int,
    binary: bool,
) -> csr_matrix:
    """
    Read one chunk file and convert to CSR matrix.

    ``csr_matrix((data, (rows, cols)), shape=...)`` automatically sums
    duplicate ``(row, col)`` entries — multiple distinct fragments from the
    same cell overlapping the same feature are summed to give the count of
    distinct overlapping fragments.

    For *binary* mode the result is clipped to 0/1 afterwards.
    """
    if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
        return csr_matrix((chunk_rows, n_cols), dtype=np.float32)

    pairs = np.fromfile(chunk_path, dtype=np.int32).reshape(-1, 2)
    rows = pairs[:, 0]
    cols = pairs[:, 1]
    vals = np.ones(len(rows), dtype=np.float32)

    chunk_csr = csr_matrix(
        (vals, (rows, cols)), shape=(chunk_rows, n_cols), dtype=np.float32,
    )

    if binary:
        chunk_csr.data = np.minimum(chunk_csr.data, 1.0)

    return chunk_csr


# ===================================================================
#  CSR assembly — in RAM
# ===================================================================

def _assemble_csr(
    chunks: List[csr_matrix],
    n_rows: int,
    n_cols: int,
) -> csr_matrix:
    """
    Concatenate CSR chunks into a single matrix by splicing arrays.

    More efficient than ``scipy.sparse.vstack`` (no intermediate COO).
    """
    if not chunks:
        return csr_matrix((n_rows, n_cols), dtype=np.float32)

    total_nnz = sum(c.nnz for c in chunks)
    if total_nnz == 0:
        return csr_matrix((n_rows, n_cols), dtype=np.float32)

    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0

    nnz_off = 0
    row_off = 0

    for chunk in chunks:
        cr = chunk.shape[0]
        cn = chunk.nnz

        if cn > 0:
            data[nnz_off : nnz_off + cn] = chunk.data
            indices[nnz_off : nnz_off + cn] = chunk.indices

        indptr[row_off + 1 : row_off + cr + 1] = chunk.indptr[1:] + nnz_off

        nnz_off += cn
        row_off += cr

    return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


# ===================================================================
#  CSR assembly — direct to h5ad (constant memory)
# ===================================================================

def _write_chunks_to_h5ad(
    output_file: str,
    chunks_dir: str,
    n_chunks: int,
    chunk_size: int,
    n_cells: int,
    n_cols: int,
    binary: bool,
    obs_names: List[str],
    var_names: List[str],
) -> "anndata.AnnData":
    """
    Write chunk files directly to h5ad.  Peak RAM = one chunk (~30 MB).

    Two-pass approach:
        Pass 1 — process each chunk, record nnz, discard CSR immediately.
        Pass 2 — process again, write data/indices/indptr to HDF5 arrays.
    """
    import h5py
    import anndata

    # -- Pass 1: count nnz per chunk --
    chunk_nnzs = []
    for chunk_id in range(n_chunks):
        chunk_rows = min(chunk_size, n_cells - chunk_id * chunk_size)
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}.bin")
        chunk_csr = _process_chunk(chunk_path, chunk_rows, n_cols, binary)
        chunk_nnzs.append(chunk_csr.nnz)
        del chunk_csr

    total_nnz = sum(chunk_nnzs)

    # -- Write HDF5 structure --
    with h5py.File(output_file, "w") as f:

        if total_nnz > 0:
            data_ds = f.create_dataset(
                "X/data", shape=(total_nnz,), dtype=np.float32,
            )
            idx_ds = f.create_dataset(
                "X/indices", shape=(total_nnz,), dtype=np.int32,
            )
        else:
            data_ds = f.create_dataset(
                "X/data", shape=(0,), dtype=np.float32,
            )
            idx_ds = f.create_dataset(
                "X/indices", shape=(0,), dtype=np.int32,
            )

        indptr = np.zeros(n_cells + 1, dtype=np.int64)

        # -- Pass 2: process + write sequentially --
        nnz_off = 0
        row_off = 0

        for chunk_id in range(n_chunks):
            chunk_rows = min(chunk_size, n_cells - chunk_id * chunk_size)
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}.bin")
            chunk_csr = _process_chunk(chunk_path, chunk_rows, n_cols, binary)

            cn = chunk_csr.nnz
            if cn > 0:
                data_ds[nnz_off : nnz_off + cn] = chunk_csr.data
                idx_ds[nnz_off : nnz_off + cn] = chunk_csr.indices

            indptr[row_off + 1 : row_off + chunk_rows + 1] = (
                chunk_csr.indptr[1:] + nnz_off
            )

            nnz_off += cn
            row_off += chunk_rows
            del chunk_csr

        f.create_dataset("X/indptr", data=indptr)
        del indptr

        # CSR encoding attributes (anndata format)
        f["X"].attrs["encoding-type"] = "csr_matrix"
        f["X"].attrs["encoding-version"] = "0.1.0"
        f["X"].attrs["shape"] = np.array([n_cells, n_cols], dtype=np.int64)

        # -- obs (cell barcodes) --
        obs_grp = f.create_group("obs")
        dt = h5py.string_dtype()
        obs_grp.create_dataset("_index", data=np.array(obs_names, dtype=object),
                               dtype=dt)
        obs_grp.attrs["_index"] = "_index"
        obs_grp.attrs["encoding-type"] = "dataframe"
        obs_grp.attrs["encoding-version"] = "0.2.0"
        obs_grp.attrs["column-order"] = []

        # -- var (feature names) --
        var_grp = f.create_group("var")
        var_grp.create_dataset("_index", data=np.array(var_names, dtype=object),
                               dtype=dt)
        var_grp.attrs["_index"] = "_index"
        var_grp.attrs["encoding-type"] = "dataframe"
        var_grp.attrs["encoding-version"] = "0.2.0"
        var_grp.attrs["column-order"] = []

    # Return backed AnnData (X lives on disk, ~4 MB in RAM)
    return anndata.read_h5ad(output_file, backed="r")


# ===================================================================
#  CSR assembly — direct to Cytome (constant memory)
# ===================================================================

def _parse_peak_metadata(var_names: List[str]) -> pd.DataFrame:
    """Parse peak names like 'chr1:100-200' into a DataFrame with chrom/start/end."""
    return _parse_feature_metadata(var_names, "peaks")


def _parse_feature_metadata(var_names: List[str], col_entity: str = "peaks") -> pd.DataFrame:
    """Parse feature names like 'chr1:100-200' into a DataFrame with chrom/start/end.

    Uses the correct ID column name based on col_entity (peak_id, tile_id, etc.).
    """
    id_col_map = {"peaks": "peak_id", "tiles": "tile_id"}
    id_col = id_col_map.get(col_entity, f"{col_entity.rstrip('s')}_id")

    chroms, starts, ends = [], [], []
    for name in var_names:
        try:
            chrom, coords = name.split(":", 1)
            s, e = coords.split("-")
            chroms.append(chrom)
            starts.append(int(s))
            ends.append(int(e))
        except (ValueError, IndexError):
            chroms.append("")
            starts.append(0)
            ends.append(0)
    return pd.DataFrame({
        id_col: var_names,
        "chr": chroms,
        "start": starts,
        "end_": ends,
    })


def _write_chunks_to_cytome(
    ds,
    chunks_dir: str,
    n_chunks: int,
    chunk_size: int,
    n_cells: int,
    n_cols: int,
    binary: bool,
    obs_names: List[str],
    var_names: List[str] = None,
    feature_df: "pd.DataFrame" = None,
    measurement: str = "counts",
    col_entity: str = "peaks",
    modality: str = "ATAC",
):
    """
    Write chunk files directly to Cytome.  Peak RAM = one chunk (~30 MB).

    Uses Cytome's ChunkedLayerWriter for zstd-compressed chunked storage.

    Parameters
    ----------
    ds : CytomeDataset
        Open dataset in read-write mode.
    chunks_dir : str
        Directory containing ``chunk_*.bin`` files.
    n_chunks : int
        Number of chunks.
    chunk_size : int
        Cells per chunk.
    n_cells : int
        Total cell count.
    n_cols : int
        Feature count (peaks or tiles).
    binary : bool
        Clip non-zero values to 1.
    obs_names : list of str
        Cell barcodes in row order.
    var_names : list of str, optional
        Feature names (peak coords or tile coords).  Parsed into a DataFrame
        via ``_parse_feature_metadata``.  Ignored when *feature_df* is given.
    feature_df : pd.DataFrame, optional
        Pre-built feature metadata DataFrame.  When provided, used directly
        instead of parsing *var_names*.  Avoids constructing + re-parsing
        5M+ tile name strings (~400 MB savings).
    measurement : str
        Name for the measurement layer (e.g., "counts", "tiles").
    col_entity : str
        Column entity type ("peaks" or "tiles").
    modality : str
        Modality prefix for the layer name (e.g., "ATAC", "tiles").
    """
    layer_name = f"{modality}_{measurement}"

    writer = ds.create_layer_writer(
        layer_name=layer_name,
        n_rows=n_cells,
        n_cols=n_cols,
        dtype=np.float32,
        compression="zstd",
        col_entity=col_entity,
        overwrite=True,
    )

    for chunk_id in range(n_chunks):
        chunk_rows = min(chunk_size, n_cells - chunk_id * chunk_size)
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}.bin")
        chunk_csr = _process_chunk(chunk_path, chunk_rows, n_cols, binary)
        writer.write_chunk(chunk_csr, row_offset=chunk_id * chunk_size)
        del chunk_csr

    # Write entity metadata BEFORE finalize() so that if the process is
    # interrupted between finalize (which commits matrix_meta) and flush,
    # the entity table is already consistent.  If crash after flush but
    # before finalize: matrix_meta absent → clean "not found" error.
    # If crash after finalize: both committed → consistent.

    # Write cell metadata if not already present
    if ds.n_cells == 0:
        ds.set_entity("cells", pd.DataFrame({"barcode": obs_names}))

    # Write peak/tile/feature metadata
    if feature_df is not None:
        ds.set_entity(col_entity, feature_df)
    else:
        if var_names is None:
            raise ValueError("Either var_names or feature_df must be provided")
        feat_df = _parse_feature_metadata(var_names, col_entity)
        ds.set_entity(col_entity, feat_df)

    ds.flush()

    # Finalize writes matrix_meta and commits — the last step, so a crash
    # before this leaves no matrix_meta (clean error) rather than a
    # matrix_meta/entity mismatch (cryptic error).
    writer.finalize()
