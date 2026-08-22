"""Ligand-receptor interaction databases for :func:`piaso.tl.runSCALAR`.

SCALAR needs a table of interacting gene pairs; it does not ship one, because
which database to use is a scientific choice. These are CellChatDB v1
reformatted to a flat ligand/receptor table with the pathway and interaction
class carried alongside — the ``annotation`` column
(``Secreted Signaling`` / ``ECM-Receptor`` / ``Cell-Cell Contact`` /
``Non-protein Signaling``) is what the LR plotting functions colour and group
by, so keeping it is the point of the reformatting.

Opt-in, like the genome and motif fetchers: nothing downloads until
:func:`fetch_lr_database` is called, and the file is cached afterwards.

Reference
---------
Jin et al. Inference and analysis of cell-cell communication using CellChat.
Nat Commun 12, 1088 (2021).
"""
from __future__ import annotations

import os
from typing import Optional

LR_URLS = {
    "human": ("https://zenodo.org/records/19981287/files/"
              "human_lr_database_CellChatDB_formatted_v2.csv?download=1"),
    "mouse": ("https://zenodo.org/records/19981287/files/"
              "mouse_lr_database_CellChatDB_formatted_v2.csv?download=1"),
}

#: The four interaction classes CellChatDB assigns. Useful for restricting an
#: analysis to one mechanism — secreted signalling and direct contact answer
#: different questions and should not be pooled without saying so.
ANNOTATION_CLASSES = ("Secreted Signaling", "ECM-Receptor",
                      "Cell-Cell Contact", "Non-protein Signaling")


def _cache_dir(dest_dir: Optional[str] = None) -> str:
    d = dest_dir or os.path.join(os.path.expanduser("~"), ".piaso", "data")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_lr_path(species: str, lr_csv: Optional[str] = None,
                    data_dir: Optional[str] = None) -> Optional[str]:
    """Local path for a cached LR database, or None if it is not there yet."""
    if lr_csv:
        return lr_csv if os.path.exists(lr_csv) else None
    name = f"{species}_lr_database_CellChatDB.csv"
    cands = []
    if data_dir:
        cands.append(os.path.join(data_dir, name))
    cands.append(os.path.join(os.path.expanduser("~"), ".piaso", "data", name))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def fetch_lr_database(species: str = "human", dest_dir: Optional[str] = None,
                      force: bool = False) -> str:
    """Download the CellChatDB ligand-receptor table for ``species``.

    Parameters
    ----------
    species
        ``'human'`` or ``'mouse'``.
    dest_dir
        Where to cache it. Defaults to ``~/.piaso/data``.
    force
        Re-download even if it is already cached.

    Returns
    -------
    str
        Local path to the CSV.
    """
    species = species.lower()
    if species not in LR_URLS:
        raise ValueError(
            f"species must be one of {sorted(LR_URLS)}; got {species!r}. "
            "For another organism, pass your own table to runSCALAR — any "
            "CSV with a ligand column and a receptor column works.")

    out = os.path.join(_cache_dir(dest_dir),
                       f"{species}_lr_database_CellChatDB.csv")
    if os.path.exists(out) and not force:
        return out

    import urllib.request
    tmp = out + ".part"
    urllib.request.urlretrieve(LR_URLS[species], tmp)
    os.replace(tmp, out)
    return out


def load_lr_database(species: str = "human",
                     annotation: Optional[str] = None,
                     lr_csv: Optional[str] = None,
                     data_dir: Optional[str] = None):
    """Load the CellChatDB table, fetching it first if needed.

    Parameters
    ----------
    species
        ``'human'`` or ``'mouse'``. Ignored when ``lr_csv`` is given.
    annotation
        Keep only one interaction class — one of
        :data:`ANNOTATION_CLASSES`. Secreted signalling and direct contact are
        different mechanisms; pooling them is a choice worth making explicitly.
    lr_csv
        Use this file instead of the cached one.
    data_dir
        Look here before the default cache.

    Returns
    -------
    pandas.DataFrame
        Columns include ``ligand``, ``receptor``, ``pathway_name`` and
        ``annotation``, plus CellChatDB's own metadata.
    """
    import pandas as pd

    path = resolve_lr_path(species, lr_csv=lr_csv, data_dir=data_dir)
    if path is None:
        path = fetch_lr_database(species, dest_dir=data_dir)
    df = pd.read_csv(path)

    if annotation is not None:
        if annotation not in set(df["annotation"].dropna()):
            raise ValueError(
                f"annotation {annotation!r} not in this table. Available: "
                f"{sorted(set(df['annotation'].dropna()))}")
        df = df[df["annotation"] == annotation].copy()
    return df
