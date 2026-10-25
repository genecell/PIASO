from ._runSVD import infog_svd, runSVD
from ._runGDR import runGDR
from ._neighbors import neighbors as _piaso_neighbors
from ._leiden import leiden as _piaso_leiden
from .external import runHarmony as _piaso_harmony


###### Some basics
from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Iterable, Union, Optional

### Refer to: https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/_compat.py
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass
######



from ..utils._cytome_compat import _is_cytome_dataset_obj as _is_cytome
from ..utils._cytome_compat import open_cytome_sync as _open_cytome


### Run leiden at another scale, i.e., locally
from typing import Sequence


def _leiden_local_group(ds, gmask, g, sub, *, modality, measurement, select,
                        max_nnz_percentile, n_components, svd_method, n_iter, tol,
                        stop_on="residuals",
                        n_neighbors, batch_size, random_state, resolution, n_threads=0,
                        backend='auto', unit_length=None):
    """Cluster one coarse group in its own space.

    ``sub`` is the group's own store (from ``ds.split``), holding only its
    cells and the matrix being clustered; it is embedded and clustered here
    and closed at the end. Returns ``(labels, n_selected)``: the sub-labels
    ``<g>-<k>`` of the group's cells in the parent's order, or ``None`` for
    the labels when the group is too degenerate to embed and should stay one
    sub-cluster.
    """
    import os
    from ._runTFIDF import compute_tfidf_stats
    try:
        from ..preprocessing._selectPeaks import selectPeaks
    except ImportError:
        selectPeaks = None

    peak_like = modality in ("ATAC", "tiles")
    matrix = f"{modality}_{measurement}"
    n_g = int(np.asarray(gmask).sum())
    try:
        svd_kw = dict(modality=modality, measurement=measurement, streaming=True,
                      key_added="X_svd", method=svd_method, n_iter=n_iter, tol=tol,
                      stop_on=stop_on,
                      random_state=random_state, batch_size=batch_size, verbosity=0,
                      n_threads=n_threads, unit_length=unit_length)
        if select and peak_like and selectPeaks is not None:
            # 1) per-group peak selection (20% cumulative-nnz, as the pipeline)
            selectPeaks(sub,
                        max_cumulative_nnz_percentile=max_nnz_percentile,
                        measurement=measurement, modality=modality,
                        key_added="highly_variable", verbose=False)
            selected = np.asarray(_read_selected_mask(sub, modality), dtype=bool)
            n_sel = int(selected.sum())
            nc = int(min(n_components, n_g - 1, max(n_sel - 1, 1)))
            if nc < 2 or n_sel < 2:
                return None, n_sel
            # 2) TF-IDF statistics + 3) streaming SVD on the selected peaks
            tfidf_stats = compute_tfidf_stats(sub, measurement=measurement, modality=modality)
            tfidf_stats["col_mask"] = selected
            runSVD(sub, tfidf_params=tfidf_stats, n_components=nc, **svd_kw)
        else:
            use_hv = False
            if select:
                try:
                    hv = np.asarray(_read_selected_mask(sub, modality), dtype=bool)
                    use_hv = bool(hv.any())
                except Exception:
                    use_hv = False
            n_feat = int(sub._conn.execute(
                "SELECT n_cols FROM matrix_meta WHERE matrix_name=?", (matrix,)).fetchone()[0])
            n_sel = int(hv.sum()) if use_hv else n_feat
            nc = int(min(n_components, n_g - 1, max(n_sel - 1, 1)))
            if nc < 2:
                return None, n_sel
            runSVD(sub, use_highly_variable=use_hv, n_components=nc, **svd_kw)

        # 4) neighbours + 5) Leiden in the group's own space. modality is
        # passed on: the embedding was stored under it.
        _piaso_neighbors(sub, use_rep="X_svd", modality=modality,
                         n_neighbors=min(n_neighbors, n_g - 1),
                         random_state=random_state, key_added="neighbors")
        # Private, case-collision-proof key: the subset inherits the
        # parent's cell columns, and SQLite columns are case-insensitive,
        # so writing 'leiden_local' beside an existing 'Leiden_local'
        # would land in it.
        _sub_key = "_piaso_ll_subleiden"
        _piaso_leiden(sub, resolution=resolution, key_added=_sub_key,
                      random_state=random_state, n_iterations=10,
                      adjacency_key="connectivities", backend=backend, n_threads=n_threads)
        sub_labels = np.asarray(sub.cells[_sub_key]).astype("U")
        sub_bc = np.asarray(sub.cells["barcode"], dtype=object)
        lab_map = {b: f"{g}-{lab}" for b, lab in zip(sub_bc, sub_labels)}
        parent_bc = np.asarray(ds.cells["barcode"], dtype=object)[np.asarray(gmask)]
        return np.array([lab_map[b] for b in parent_bc], dtype=object), n_sel
    finally:
        try:
            sub.close()
        except Exception:
            pass


def _leiden_local_cytome_streaming_svd(
    ds,
    groupby: str,
    groups,
    resolution: float,
    key_added: str,
    modality: str = "ATAC",
    measurement: str = "counts",
    select: bool = True,
    max_nnz_percentile: float = 20.0,
    n_components: int = 30,
    svd_method: str = "auto",
    n_iter=None,
    tol=None,
    stop_on: str = "residuals",
    n_neighbors: int = 15,
    batch_size: int = 1024,
    random_state: int = 10,
    min_cells: int = 20,
    verbose: bool = True,
    n_threads: int = 0,
    backend: str = 'auto',
    tmp_dir=None,
    unit_length: Optional[bool] = None,
):
    """Streaming, RAM-bounded `leiden_local` on a cytome, one group at a time.

    Each coarse group is **subset to its own cytome** carrying only the
    matrix being clustered and its feature table (peak RAM about one chunk,
    never the whole matrix), and re-embedded in its own space:

    - ATAC / tiles with ``select``: ``selectPeaks`` by cumulative non-zeros
      -> TF-IDF -> randomized SVD -> neighbours -> Leiden (``dr_method='X_svd'``).
    - RNA / GA with ``select``: the store's ``highly_variable`` column if it
      has one, every feature otherwise; values as stored (``cytome_layer``).
    - any modality without ``select``: every feature with at least one
      count, values as stored (``dr_method='X_svd_full'``); ``runSVD``'s own
      defaults apply, so ATAC and tiles counts still get TF-IDF.

    Sub-labels are ``<group>-<k>`` in ``ds.cells[key_added]``. A group with
    fewer than ``min_cells`` cells, or one whose chain fails, is kept as
    one sub-cluster ``<group>-0`` (the failure with a ``RuntimeWarning``).

    All groups' stores are written in one scan of the matrix (``ds.split``)
    to a new folder under ``tmp_dir``; the groups are then clustered one at a
    time, each store removed when its group is done. When the folder's disk
    has room for only some of the stores, the groups are split in batches
    that fit, in their order, one scan each (``_split_batches``).
    """
    import tempfile
    import warnings
    try:
        from ..preprocessing._selectPeaks import selectPeaks
    except ImportError:
        # Per-group peak selection is part of the peak chain, which not every
        # build carries. Without it each group is embedded on the selection
        # the store already holds (`highly_variable`), or on every feature.
        selectPeaks = None

    peak_like = modality in ("ATAC", "tiles")
    if select and peak_like and selectPeaks is None and verbose:
        print("  [leiden_local] per-group peak selection is not available in this build; "
              "each group uses its stored 'highly_variable' selection, or every feature")
    coarse = np.asarray(ds.cells[groupby]).astype("U")
    if groups is None:
        groups = np.unique(coarse)
    # object dtype -- a fixed-width "<U1" (from a single-char coarse label)
    # would silently TRUNCATE the prefixed sub-labels ("A-3" -> "A").
    result = coarse.astype(object)
    params = dict(modality=modality, measurement=measurement, select=bool(select),
                  max_nnz_percentile=max_nnz_percentile, n_components=n_components,
                  svd_method=svd_method, n_iter=n_iter, tol=tol, stop_on=stop_on,
                  n_neighbors=n_neighbors,
                  batch_size=batch_size, random_state=random_state, resolution=resolution,
                  n_threads=n_threads, backend=backend, unit_length=unit_length)

    todo = []
    for g in groups:
        n_g = int((coarse == str(g)).sum())
        if n_g == 0:
            continue
        if n_g < max(min_cells, 3):
            result[coarse == str(g)] = [f"{g}-0"] * n_g
            if verbose:
                print(f"  [leiden_local] group {g}: {n_g} cells (<{min_cells}) -> single sub-cluster")
            continue
        todo.append((str(g), n_g))

    def on_done(g, labels, n_sel, error):
        gmask = coarse == g
        n_g = int(gmask.sum())
        if error is not None:
            warnings.warn(
                f"leiden_local: group {g} failed ({error}); "
                f"keeping it as a single sub-cluster.", RuntimeWarning, stacklevel=3)
            result[gmask] = [f"{g}-0"] * n_g
        elif labels is None:
            result[gmask] = [f"{g}-0"] * n_g
            if verbose:
                print(f"  [leiden_local] group {g}: n_sel={n_sel} -> single sub-cluster")
        else:
            result[gmask] = labels
            if verbose:
                print(f"  [leiden_local] group {g}: {n_g} cells, {n_sel} features -> "
                      f"{len(set(labels))} sub-clusters")

    import os
    import shutil
    if tmp_dir is not None:
        os.makedirs(tmp_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="leiden_local_", dir=tmp_dir)
    try:
        # Every group's store in one scan of the matrix (``split``), in
        # batches when the disk is short of them all; older cytomes without
        # ``split`` make them one at a time.
        outputs = {g: os.path.join(tmpdir, f"g{gi}.cytome") for gi, (g, _n) in enumerate(todo)}
        matrix = f"{modality}_{measurement}"
        if hasattr(ds, "split") and todo:
            batches = _split_batches(ds, todo, matrix, tmpdir, verbose)
        else:
            batches = [[g for g, _n in todo]]
        for batch in batches:
            stores = (ds.split(coarse, {g: outputs[g] for g in batch}, [matrix],
                               include_fragments=False, include_embeddings=False)
                      if hasattr(ds, "split") and batch else None)
            for g in batch:
                try:
                    sub = stores.pop(g) if stores is not None else ds.subset(
                        coarse == g, output=outputs[g], matrices=[matrix],
                        include_fragments=False, include_embeddings=False)
                    labels, n_sel = _leiden_local_group(ds, coarse == g, g, sub, **params)
                    on_done(g, labels, n_sel, None)
                except Exception as exc:  # one bad group must not kill the run
                    on_done(g, None, 0, f"{type(exc).__name__}: {exc}")
                finally:
                    for suf in ("", "-wal", "-shm"):
                        try:
                            os.remove(outputs[g] + suf)
                        except OSError:
                            pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    ds.cells[key_added] = result
    ds.flush()
    return None

def _split_batches(ds, todo, matrix, directory, verbose):
    """The groups of ``todo`` in batches whose stores fit in ``directory``'s free space.

    One batch when they all fit, or when the store's cytome cannot estimate a
    split's size (before 0.3.7). Otherwise batches in the groups' order, each
    within the free space less the split's own margin; a group too large for
    the disk on its own is a batch of one, which ``split`` then refuses with
    both numbers. The free space is measured once: each batch's stores are
    removed before the next is written.
    """
    import shutil
    groups = [g for g, _n in todo]
    try:
        from cytome.io.subset import _split_bytes_to_write as estimate
    except ImportError:
        return [groups]
    n = max(int(ds.n_cells), 1)
    need = [estimate(ds, [n_g / n], [matrix], False) for _g, n_g in todo]
    free = shutil.disk_usage(directory).free - (320 << 20)   # split keeps 256 MB in hand
    if sum(need) <= free:
        return [groups]
    batches, cur, used = [], [], 0
    for g, b in zip(groups, need):
        if cur and used + b > free:
            batches.append(cur)
            cur, used = [], 0
        cur.append(g)
        used += b
    batches.append(cur)
    if verbose:
        print(f"  [leiden_local] the {len(groups)} group stores need about {sum(need) / 1e9:.1f} GB "
              f"and {directory} has {max(free, 0) / 1e9:.1f} GB free: writing them in "
              f"{len(batches)} batches, one scan of {matrix} each")
    return batches


def _read_selected_mask(ds, modality_or_table):
    """The boolean 'highly_variable' column of a modality's feature table.

    Takes a modality ('ATAC', 'RNA', 'GA', 'tiles') or the raw table name.
    Resolved through ``ds.features(modality)`` rather than
    ``getattr(ds, table)``: for tiles and gene activity the attribute returns
    a ``Modality``, which has no ``__getitem__`` over columns, so the old form
    raised or read the wrong thing for every modality but ATAC and RNA.
    """
    from ..utils._bool_mask import as_bool_mask

    name = str(modality_or_table)
    _TABLE_TO_MODALITY = {"peaks": "ATAC", "genes": "RNA",
                          "GA_genes": "GA", "tiles": "tiles"}
    modality = _TABLE_TO_MODALITY.get(name, name)
    def _coerce(values, label):
        # No `dtype=object`: `as_bool_mask` reads the array's dtype to decide
        # how the column was stored, so boxing a perfectly good INTEGER column
        # as object sends it down the text branch and warns that it is text.
        # Let numpy infer, and only fall back to object for genuinely mixed
        # values.
        try:
            arr = np.asarray(values)
        except Exception:
            arr = np.asarray(list(values), dtype=object)
        return as_bool_mask(arr, name=label)

    try:
        ent = ds.features(modality)
    except Exception:
        # A table name no registry knows; fall back to reading it directly.
        rows = ds._conn.execute(
            f"SELECT highly_variable FROM {name} ORDER BY ROWID").fetchall()
        return _coerce([r[0] for r in rows], f"{name}.highly_variable")
    return _coerce(ent["highly_variable"], f"{modality}.highly_variable")


_DR_METHODS = ('X_svd', 'X_svd_full', 'X_pca', 'X_gdr', 'X_pca_harmony', 'X_svd_full_harmony')


def _embed_group(adata_i, dr_method, n_components, batch_key, gdr_resolution,
                 svd_method, n_iter, svd_tol, random_state, verbosity, said, n_threads=0,
                 unit_length=None, svd_stop_on="residuals"):
    """Embed one group's AnnData by ``dr_method``; return the ``obsm`` key."""
    if dr_method == 'X_svd':
        hv = 'highly_variable' in adata_i.var.columns and bool(np.asarray(adata_i.var['highly_variable']).any())
        if not hv and verbosity > 0 and not said.get('all_features'):
            print("  [leiden_local] no 'highly_variable' column in var: using every feature "
                  "(dr_method='X_svd'); run selectFeatures/infog first to select.")
            said['all_features'] = True
        runSVD(adata_i, use_highly_variable=hv, n_components=n_components, random_state=random_state,
               scale_data=False, method=svd_method, n_iter=n_iter, tol=svd_tol, stop_on=svd_stop_on,
               key_added='X_svd',
               n_threads=n_threads, unit_length=unit_length)
        return 'X_svd'
    if dr_method in ('X_svd_full', 'X_svd_full_harmony'):
        runSVD(adata_i, use_highly_variable=False, n_components=n_components, random_state=random_state,
               scale_data=False, method=svd_method, n_iter=n_iter, tol=svd_tol, stop_on=svd_stop_on,
               key_added='X_svd_full',
               n_threads=n_threads, unit_length=unit_length)
        if dr_method == 'X_svd_full':
            return 'X_svd_full'
        _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_svd_full', key_added='X_svd_full_harmony')
        return 'X_svd_full_harmony'
    # the INFOG-first family
    infog_svd(adata_i, n_components=n_components)
    if dr_method == 'X_pca':
        return 'X_pca'
    if dr_method == 'X_gdr':
        knn = _piaso_neighbors(adata_i, use_rep='X_pca', n_neighbors=15, random_state=random_state)
        _piaso_leiden(adata_i, resolution=gdr_resolution, key_added='leiden_tmp', knn_result=knn,
                      random_state=random_state)
        runGDR(adata_i, batch_key=None, groupby='leiden_tmp')
        return 'X_gdr'
    _piaso_harmony(adata_i, batch_key=batch_key, use_rep='X_pca', key_added='X_pca_harmony')
    return 'X_pca_harmony'


def leiden_local(
    adata,
    clustering_type: str = 'each',
    groupby: str = 'Leiden',
    groups: Optional[Sequence[str]] = None,
    resolution: float = 0.25,
    batch_key: Optional[Sequence[str]] = None,
    key_added: str = 'Leiden_local',
    dr_method: str = 'X_svd',
    gdr_resolution: float = 1.0,
    copy: bool = False,
    modality: Optional[str] = None,
    cytome_layer: str = "counts",
    max_nnz_percentile: float = 20.0,
    n_components: int = 30,
    svd_method: str = "auto",
    n_iter: Optional[int] = None,
    svd_tol: Optional[float] = None,
    svd_stop_on: str = "residuals",
    svd_unit_length: Optional[bool] = None,
    n_neighbors: int = 15,
    batch_size: int = 1024,
    random_state: int = 10,
    min_cells: int = 20,
    verbosity: int = 1,
    n_threads: int = 0,
    backend: str = 'auto',
    tmp_dir: Optional[str] = None,
):
    """Leiden clustering within each group of a coarse clustering.

    Every group is re-embedded in its own low-dimensional space and
    clustered there, so structure the whole-dataset embedding merged is
    recovered. Works on an AnnData in memory and on a cytome (open
    ``Dataset`` or path) streaming, per group, in bounded memory.

    Parameters
    ----------
    adata : AnnData, cytome.Dataset or str
        The data; a cytome may be given as a path.
    clustering_type : {'each', 'all'}, default 'each'
        ``'each'`` clusters every selected group on its own; ``'all'``
        pools the selected groups and clusters them together (sub-labels
        prefixed ``M-``). A cytome supports ``'each'`` only.
    groupby : str, default 'Leiden'
        The coarse labels: an ``obs`` column, or a ``cells`` column on a
        cytome.
    groups : sequence of str, optional
        Which groups to cluster; all of them when ``None``.
    resolution : float, default 0.25
        Leiden resolution inside each group; higher gives more clusters.
    batch_key : str, optional
        Batch column for the Harmony variants of ``dr_method``.
    key_added : str, default 'Leiden_local'
        Where the labels go: ``obs[key_added]`` or ``cells[key_added]``,
        as ``<group>-<k>``.
    dr_method : str, default 'X_svd'
        How each group is embedded:

        - ``'X_svd'``: ``runSVD`` on the group's **selected** features. On
          an ATAC or tiles cytome the selection is made per group by
          cumulative non-zeros (``max_nnz_percentile``) with TF-IDF; on an
          RNA or GA cytome and on an AnnData it is the ``highly_variable``
          column, or every feature if there is none (said once). An ATAC
          AnnData should be TF-IDF-normalised first (``run_TFIDF``).
        - ``'X_svd_full'``: ``runSVD`` on **every** feature with at least
          one count, values as stored.
        - ``'X_pca'``: INFOG normalisation and selection, then SVD.
        - ``'X_gdr'``, ``'X_pca_harmony'``, ``'X_svd_full_harmony'``: as
          their names say; Harmony needs ``batch_key``.

        On a cytome, ``'X_svd'`` and ``'X_svd_full'`` stream per group;
        the other methods convert the store to an AnnData first.
    gdr_resolution : float, default 1.0
        Leiden resolution of the temporary clustering GDR is built on
        (``dr_method='X_gdr'``).
    copy : bool, default False
        AnnData only: return a modified copy instead of editing in place.
    modality : str, optional
        Cytome only: which modality to cluster (``'ATAC'``, ``'tiles'``,
        ``'RNA'``, ``'GA'``). Defaults to ``'ATAC'`` on the streaming path
        and ``'RNA'`` for the AnnData conversion.
    cytome_layer : str, default 'counts'
        Cytome only: the layer to read. For RNA that is usually a
        materialised ``'infog'``.
    max_nnz_percentile : float, default 20.0
        ATAC/tiles streaming only: the cumulative-non-zero percentile that
        selects a group's peaks.
    n_components : int, default 30
        SVD dimensions per group, capped at one less than the group's cell
        count and than its selected feature count (and, for the AnnData
        path, at the square root of the cell count, as before).
    svd_method : {'auto', 'power', 'krylov'}, default 'auto'
        The per-group solver; see ``runSVD``.
    n_iter : int, optional
        Rounds for ``'power'``/``'krylov'``, a ceiling for ``'auto'``; see
        ``runSVD``.
    svd_tol : float, optional
        Convergence tolerance passed to ``runSVD``.
    svd_stop_on : {'residuals', 'singular_values'}, default 'residuals'
        What ``svd_tol`` bounds in each group's SVD; see ``runSVD``'s
        ``stop_on``. ``'singular_values'`` gives the groups of 1.2.6 and
        earlier.
    svd_unit_length : bool, optional
        Passed to each group's ``runSVD`` as ``unit_length``: with True every
        cell's vector in the group's own embedding is scaled to length 1
        before the group's neighbors and Leiden (``dr_method`` 'X_svd' and
        'X_svd_full'). Left unset, each group's ``runSVD`` decides by what it
        factorises, as ``runSVD`` does: on for an ATAC or tiles store's groups,
        off for RNA. On an AnnData the groups' SVDs see no modality, so pass
        True there for ATAC.
    n_neighbors : int, default 15
        Neighbours for the per-group graph (capped below the group size).
    batch_size : int, default 1024
        Rows per chunk for the Python reader, where it is used.
    random_state : int, default 10
        Seed for the SVD, the graph and Leiden.
    min_cells : int, default 20
        Cytome streaming only: a group smaller than this is kept as one
        sub-cluster ``<group>-0`` rather than re-embedded.
    verbosity : int, default 1
        ``1`` reports each group's outcome.
    n_threads : int, default 0
        Threads for each group's SVD and Leiden; 0 uses all the cores this
        process may use. The labels do not depend on it.
    backend : {'auto', 'rust', 'igraph'}, default 'auto'
        The Leiden of every group, as in :func:`piaso.tl.leiden`.
    tmp_dir : str, optional
        Cytome streaming only: where the groups' temporary stores are
        written, in a new folder removed at the end. Together they hold
        about one copy of the matrix clustered; each is removed when its
        group is done. ``None`` uses Python's temporary folder
        (``$TMPDIR``, else ``/tmp``), which on a cluster node or a small
        system disk may be far smaller than the store, so for a large
        store name a folder on a disk with room (the store's own, say).
        When the disk has room for only some of the stores, the groups are
        written in batches that fit, one scan of the matrix each.

    Returns
    -------
    AnnData or None
        The copy when ``copy=True``; otherwise ``None`` after writing the
        labels in place.

    Notes
    -----
    On the cytome streaming path a group whose chain fails is kept as one
    sub-cluster ``<group>-0`` with a ``RuntimeWarning`` naming the error,
    so one bad group does not lose the run; a result full of ``-0`` labels
    is the sign to read the warnings.
    """
    # A store that cannot be written fails here, not after the work.
    from ..utils._cytome_compat import require_writable as _require_writable
    _require_writable(adata, "piaso.tl.leiden_local")
    if clustering_type not in {'each', 'all'}:
        raise ValueError(f"Invalid value for clustering_type: '{clustering_type}'. "
                         "Allowed values are 'each' or 'all'.")
    if dr_method not in _DR_METHODS:
        raise ValueError(f"Invalid value for dr_method: '{dr_method}'. Allowed values are: "
                         + ", ".join(repr(m) for m in _DR_METHODS) + ".")
    verbose = verbosity > 0

    # --- Cytome, streaming per group ---
    if dr_method in ('X_svd', 'X_svd_full') and (isinstance(adata, str) or _is_cytome(adata)):
        if clustering_type != 'each':
            raise ValueError("On a cytome, leiden_local supports clustering_type='each' only.")
        _ds = _open_cytome(adata) if isinstance(adata, str) else adata
        return _leiden_local_cytome_streaming_svd(
            _ds, groupby=groupby, groups=groups, resolution=resolution, key_added=key_added,
            modality=modality if modality is not None else "ATAC", measurement=cytome_layer,
            select=(dr_method == 'X_svd'), max_nnz_percentile=max_nnz_percentile,
            n_components=n_components, svd_method=svd_method, n_iter=n_iter, tol=svd_tol,
            stop_on=svd_stop_on,
            n_neighbors=n_neighbors, batch_size=batch_size, random_state=random_state,
            min_cells=min_cells, verbose=verbose, n_threads=n_threads, backend=backend,
            tmp_dir=tmp_dir, unit_length=svd_unit_length,
        )

    # --- Cytome, other methods: convert, run, write back ---
    _cytome_ds = None
    if isinstance(adata, str) or _is_cytome(adata):
        _cytome_ds = _open_cytome(adata) if isinstance(adata, str) else adata
        adata = _cytome_ds.to_anndata(modality=modality if modality is not None else "RNA",
                                      layer=cytome_layer)

    adata = adata.copy() if copy else adata
    existing_groups = adata.obs[groupby].astype('U').copy()
    if groups is None:
        groups = np.unique(adata.obs[groupby])
    said: dict = {}

    def _cluster(group_idx, prefix):
        adata_i = adata[group_idx].copy()
        adata_i = adata_i[:, np.asarray(adata_i.X.getnnz(axis=0)).ravel() >= 1].copy()
        sqrt_ncells = int(np.sqrt(adata_i.shape[0]))
        nc = n_components if sqrt_ncells > n_components else sqrt_ncells
        use_rep = _embed_group(adata_i, dr_method, nc, batch_key, gdr_resolution,
                               svd_method, n_iter, svd_tol, random_state, verbosity, said,
                               n_threads=n_threads, unit_length=svd_unit_length,
                               svd_stop_on=svd_stop_on)
        knn = _piaso_neighbors(adata_i, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_state)
        _piaso_leiden(adata_i, resolution=resolution, key_added='leiden_local', knn_result=knn,
                      random_state=random_state, backend=backend, n_threads=n_threads)
        labels = adata_i.obs['leiden_local'].astype(str)
        width = max(len(x) for x in labels)          # equal-width numbers
        existing_groups[group_idx] = [f"{prefix}-{x.zfill(width)}" for x in labels]
        if verbose:
            print(f"  [leiden_local] {prefix}: {adata_i.n_obs} cells -> {labels.nunique()} sub-clusters")

    if clustering_type == 'each':
        for group in groups:
            _cluster(adata.obs[groupby].isin([group]), str(group))
    else:
        _cluster(adata.obs[groupby].isin(groups), 'M' if len(groups) > 1 else str(groups[0]))

    adata.obs[key_added] = existing_groups
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    if _cytome_ds is not None:
        _cytome_ds.cells[key_added] = adata.obs[key_added].values.astype(str)
        _cytome_ds.flush()
        return None
    return adata if copy else None

    

