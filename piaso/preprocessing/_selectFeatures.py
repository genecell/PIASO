"""Mark a set of named features on a cytome, by identifier.

The recurring task this replaces: you have a list of feature identifiers from
somewhere else -- COSG marker peaks, a published peak set, a gene panel -- and
you want the cytome's feature table to carry a boolean column saying which of
its features are in that list, so ``runSVD`` and everything downstream can
read it.

Written by hand that is::

    ids = marker_df[marker_df['rank'] < 2000]['feature'].values
    ds.features(modality='ATAC')['highly_variable'] = np.isin(
        ds.features(modality='ATAC').to_pandas()['peak_id'], ids)
    ds.flush()

which works, and has three ways to go quietly wrong:

* it picks one identifier column by name, so a list in the other vocabulary
  (symbols against Ensembl ids, say) matches nothing and writes all-False;
* it reports nothing, so identifiers absent from this cytome -- a peak set
  called on different data, a gene renamed between annotations -- vanish
  without a word, and you cannot tell 45,398 matches from 4,398;
* a mask that is not a numpy boolean array lands as a TEXT column, and a TEXT
  boolean read back with ``.astype(bool)`` selects **every** feature, because
  a non-empty string is truthy.

This module does the same thing and closes all three.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..utils._cytome_compat import is_cytome_input as _is_cytome_input
from ..utils._cytome_compat import open_cytome_sync as _open_cytome

__all__ = ["selectFeatures"]


def selectFeatures(
    data,
    features: Iterable,
    key_added: str = "highly_variable",
    modality: str = "ATAC",
    match: str = "any",
    feature_col: Optional[str] = None,
    overwrite: bool = True,
    min_matched: Optional[float] = None,
    return_mask: bool = False,
    verbose: bool = True,
):
    """Flag the features named in ``features`` on the cytome's feature table.

    Parameters
    ----------
    data : cytome.Dataset or str
        An open cytome, or a path to one.
    features : iterable of str
        The identifiers to mark. Peak ids (``'chr1:100-200'``), gene symbols,
        Ensembl ids -- whatever the cytome stores, in any of its identifier
        columns unless ``feature_col`` pins one.
    key_added : str, default 'highly_variable'
        The boolean column to write. This is the name ``runSVD`` and
        ``leiden_local`` read by default.
    modality : str, default 'ATAC'
        Which feature table: ``'ATAC'`` -> peaks, ``'RNA'`` -> genes,
        ``'GA'`` -> GA_genes, ``'tiles'`` -> tiles. Resolved through the
        modality registry, never by guessing the table name.
    match : {'any', 'id'}, default 'any'
        ``'any'`` resolves an identifier against every candidate id/name
        column the modality registers, so a list in either vocabulary works.
        ``'id'`` matches only the registry's canonical id column, which is
        what you want when two columns could both match and you know which
        one you mean.
    feature_col : str, optional
        Match against exactly this column, overriding ``match``.
    overwrite : bool, default True
        Refuse when ``key_added`` already exists and this is False. A
        name that differs from an existing column only in case is refused
        either way unless ``overwrite`` is set, because SQLite column names
        are case-insensitive and the two would silently become one.
    min_matched : float, optional
        Raise unless at least this fraction of ``features`` matched
        something. ``0.9`` turns "most of my list is missing" from a line of
        output into an error. The default reports and continues.
    return_mask : bool, default False
        Also return the boolean mask. Off by default: this writes the column
        in place like every other writer in the package, and returning an
        860,000-entry array prints it into a notebook.
    verbose : bool, default True
        Report how many features were flagged and how many identifiers went
        unmatched.

    Returns
    -------
    None, or the boolean mask when ``return_mask=True``.

    Examples
    --------
    >>> ids = marker_df[marker_df['rank'] < 2000]['feature'].values
    >>> piaso.pp.selectFeatures(ds, ids, modality='ATAC')
      selectFeatures: flagged 45,398 of 834,686 peaks as 'highly_variable'
    >>> piaso.tl.runSVD(ds, modality='ATAC', use_highly_variable=True)
    """
    if match not in ("any", "id"):
        raise ValueError(f"match={match!r} is not 'any' or 'id'")

    opened = isinstance(data, str)
    ds = _open_cytome(data) if opened else data
    try:
        if not _is_cytome_input(ds):
            raise TypeError(
                "selectFeatures needs a cytome dataset or a path to one; got "
                f"{type(data).__name__}. For an AnnData, assign the mask to "
                "adata.var directly.")

        from cytome import modality_var_entity
        feat_tbl, id_col = modality_var_entity(modality)
        n_features = ds._conn.execute(
            f"SELECT COUNT(*) FROM {feat_tbl}").fetchone()[0]

        wanted = [str(f) for f in features
                  if f is not None and str(f) != "" and str(f) != "nan"]
        if not wanted:
            raise ValueError(
                "selectFeatures: `features` is empty after dropping blanks, "
                "so every feature would be deselected. Pass the identifiers "
                "you meant, or write the column yourself if that is the "
                "intent.")

        mask, matched = _mask_from_ids(
            ds, modality, feat_tbl, id_col, wanted, match, feature_col,
            n_features)

        unmatched = len(set(wanted)) - matched
        if min_matched is not None:
            frac = matched / max(len(set(wanted)), 1)
            if frac < float(min_matched):
                raise ValueError(
                    f"selectFeatures: only {matched:,} of "
                    f"{len(set(wanted)):,} identifiers ({frac:.1%}) matched a "
                    f"feature in '{feat_tbl}', below min_matched="
                    f"{min_matched}. The list and this cytome probably come "
                    f"from different feature sets or annotations.")

        _write_feature_column(ds, modality, feat_tbl, key_added, mask,
                              overwrite=overwrite)

        if verbose:
            entity = feat_tbl
            print(f"  selectFeatures: flagged {int(mask.sum()):,} of "
                  f"{n_features:,} {entity} as {key_added!r}")
            if unmatched:
                print(f"  {unmatched:,} of {len(set(wanted)):,} identifiers "
                      f"matched no feature and were ignored")
        return mask if return_mask else None
    finally:
        if opened:
            ds.close()


def _mask_from_ids(ds, modality, feat_tbl, id_col, wanted, match, feature_col,
                   n_features):
    """(boolean mask, number of distinct identifiers that matched something)."""
    want = set(wanted)

    if feature_col is not None or match == "id":
        col = feature_col or id_col
        cols = [r[1] for r in ds._conn.execute(
            f"PRAGMA table_info({feat_tbl})").fetchall()]
        if col not in cols:
            raise KeyError(
                f"selectFeatures: column {col!r} not found on '{feat_tbl}'. "
                f"Available: {cols}")
        idx_col = cols[0]
        mask = np.zeros(n_features, dtype=bool)
        hit = set()
        for ridx, val in ds._conn.execute(
                f'SELECT {idx_col}, "{col}" FROM {feat_tbl} ORDER BY {idx_col}'):
            if val is not None and str(val) in want:
                mask[int(ridx)] = True
                hit.add(str(val))
        return mask, len(hit)

    # 'any': the alias map scoring already uses, so a list in either
    # vocabulary resolves and a duplicated symbol flags all of its rows.
    from ..tools._normalization import (_gene_name_alias_map,
                                        _resolve_gene_names)
    var_names = _resolve_gene_names(ds, modality)
    name_to_idx = _gene_name_alias_map(ds, modality, var_names)
    mask = np.zeros(n_features, dtype=bool)
    matched = 0
    for name in want:
        rows = name_to_idx.get(name)
        if not rows:
            continue
        matched += 1
        for r in rows:
            if 0 <= int(r) < n_features:
                mask[int(r)] = True
    return mask, matched


def _write_feature_column(ds, modality, feat_tbl, name, values, *,
                          overwrite=True):
    """Write a boolean column onto a modality's feature table.

    The feature-axis counterpart of ``write_cells_column``. Two things it
    exists to get right:

    * ``ds.features(modality)`` rather than ``getattr(ds, feat_tbl)`` -- the
      attribute resolves to a ``Modality`` for tiles and gene activity, which
      has no column assignment, so the write goes nowhere;
    * a real numpy bool array, so the column is created INTEGER. A TEXT
      boolean read back with ``.astype(bool)`` selects every feature.
    """
    import warnings

    values = np.asarray(values, dtype=bool)
    existing = [r[1] for r in ds._conn.execute(
        f"PRAGMA table_info({feat_tbl})").fetchall()]
    if len(values) != ds._conn.execute(
            f"SELECT COUNT(*) FROM {feat_tbl}").fetchone()[0]:
        raise ValueError(
            f"selectFeatures: mask has {len(values)} entries but "
            f"'{feat_tbl}' has "
            f"{ds._conn.execute(f'SELECT COUNT(*) FROM {feat_tbl}').fetchone()[0]}"
            f" rows.")
    clash = next((c for c in existing if c.lower() == name.lower()), None)
    if clash is not None and not overwrite:
        raise ValueError(
            f"selectFeatures: '{feat_tbl}' already has a column {clash!r} and "
            f"overwrite=False. Pass overwrite=True to replace it, or choose "
            f"another key_added.")
    if clash is not None and clash != name:
        warnings.warn(
            f"selectFeatures: {feat_tbl}[{name!r}] resolves to the existing "
            f"column {clash!r} (SQLite column names are case-insensitive); "
            f"writing {clash!r}.", UserWarning, stacklevel=3)
        name = clash

    ds.features(modality)[name] = values
    # Buffered until flushed: a read before this point goes straight to SQLite
    # and would miss the column entirely.
    ds.flush()
    return name
