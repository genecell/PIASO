"""ChEMBL drug-target gene sets for :func:`piaso.tl.score`.

A drug set is a gene set: the genes a compound is known to act on. Scoring
those sets per cell asks which cell types express the targets of a given drug,
which is the question behind most "is this cell type druggable" analyses.

ChEMBL publishes drug-target activities, but the raw table is not usable as
gene sets directly — it contains every measured activity, including inactive
ones, assays of the wrong type, and compounds that never left preclinical.
Turning it into gene sets is a filtering problem, and the filtering is the
scientific content: a 1 µM cut-off means something different for a kinase than
for an ion channel.

The filter here follows Kanemaru et al. (Nature 2023), whose ``drug2cell``
package introduced this analysis. ``drug2cell`` is not a dependency — the part
of it used to build the dictionary is a sequence of pandas filters, reproduced
here so the gene sets can be built with pandas alone, and so the thresholds are
visible rather than buried in a call.

References
----------
Kanemaru et al. Spatially resolved multiomics of human cardiac niches.
Nature 619, 801-810 (2023).
ChEMBL 30, https://www.ebi.ac.uk/chembl/
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

#: The pre-merged ChEMBL 30 table with human gene symbols attached, published
#: alongside the drug2cell paper. ~2.7 GB.
CHEMBL_URL = ("ftp://ftp.sanger.ac.uk/pub/users/kp9/"
              "chembl_30_merged_genesymbols_humans.pkl")

#: pChEMBL is -log10(activity), so a *higher* number is a *tighter* binder.
#: The cut-off has to depend on the target class, because what counts as potent
#: differs by orders of magnitude between them: 30 nM is unremarkable for a
#: kinase inhibitor and exceptional for an ion-channel blocker. Values as
#: revised 21 Dec 2024.
PCHEMBL_THRESHOLDS: Dict[str, float] = {
    "none": 6.0,          # 1 uM   -- unclassified targets
    "NHR": 7.0,           # 100 nM -- nuclear hormone receptors
    "GPCR": 7.0,          # 100 nM
    "Ion Channel": 5.0,   # 10 uM
    "Kinase": 7.53,       # 30 nM
}

#: `activity_comment` strings that mean the compound did *not* act.
INACTIVE_COMMENTS = (
    "inactive",
    "Inactive",
    "Not Active",
    "Not Active (inhibition < 50% @ 10 uM and thus dose-reponse curve "
    "not measured)",
)
ACTIVE_COMMENTS = ("active", "Active")

#: Column names in the published table. Kept in one place because they are
#: `table|column` strings that are easy to mistype and impossible to guess.
COL_MAX_PHASE = "molecule_dictionary|max_phase"
COL_CHEMBL_ID = "molecule_dictionary|chembl_id"
COL_PREF_NAME = "molecule_dictionary|pref_name"
COL_MECHANISM = "drug_mechanism|molregno"
COL_ASSAY_TYPE = "assays|assay_type"
COL_ACTIVITY_COMMENT = "activities|activity_comment"
COL_PCHEMBL = "activities|pchembl_value"
COL_SYNONYMS = "component_synonyms|component_synonym"


def _cache_dir(dest_dir: Optional[str] = None) -> str:
    d = dest_dir or os.path.join(os.path.expanduser("~"), ".piaso", "data")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_chembl_path(chembl_pkl: Optional[str] = None,
                        data_dir: Optional[str] = None) -> Optional[str]:
    """Local path to a cached ChEMBL table, or None if it is not there."""
    if chembl_pkl:
        return chembl_pkl if os.path.exists(chembl_pkl) else None
    name = "chembl_30_merged_genesymbols_humans.pkl"
    cands = []
    if data_dir:
        cands.append(os.path.join(data_dir, name))
    cands.append(os.path.join(os.path.expanduser("~"), ".piaso", "data", name))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def fetch_chembl(dest_dir: Optional[str] = None, force: bool = False) -> str:
    """Download the merged ChEMBL 30 activity table (OPT-IN, ~2.7 GB).

    This is much larger than the other references PIASO fetches, and reading it
    needs roughly 10 GB of RAM. It is not downloaded implicitly by anything;
    call this when you want it.

    Returns
    -------
    str
        Local path to the ``.pkl``.
    """
    out = os.path.join(_cache_dir(dest_dir),
                       "chembl_30_merged_genesymbols_humans.pkl")
    if os.path.exists(out) and not force:
        return out

    import urllib.request
    tmp = out + ".part"
    urllib.request.urlretrieve(CHEMBL_URL, tmp)
    os.replace(tmp, out)
    return out


def filter_chembl_activities(
    dataframe,
    drug_max_phase: Optional[Union[int, Sequence[int]]] = 4,
    assay_type: Optional[Union[str, Sequence[str]]] = "F",
    add_drug_mechanism: bool = True,
    remove_inactive: bool = True,
    include_active: bool = True,
    pchembl_target_column: Optional[str] = "target_class",
    pchembl_threshold: Optional[Union[float, Dict[str, float]]] = None,
    default_pchembl: float = 6.0,
    verbose: bool = True,
):
    """Filter raw ChEMBL activities down to confident drug-target pairs.

    The filters are applied in the order of the arguments, and two of them
    (``add_drug_mechanism``, ``include_active``) do not remove rows — they mark
    rows as *protected* so later filters cannot drop them. A curated mechanism
    of action is better evidence than any activity measurement, so a row
    carrying one survives an assay-type or potency cut it would otherwise fail.

    Parameters
    ----------
    dataframe
        The merged ChEMBL table from :func:`fetch_chembl`.
    drug_max_phase
        Clinical stage to keep. ``4`` is approved drugs; pass a list for
        several, or None for no filter.
    assay_type
        ChEMBL assay type. ``'F'`` (functional) measures a biological effect;
        ``'B'`` (binding) measures affinity.
    add_drug_mechanism
        Protect rows that have a curated drug mechanism.
    remove_inactive
        Drop rows whose activity comment says the compound was inactive.
    include_active
        Protect rows whose activity comment says it was active.
    pchembl_target_column
        Column holding the target class, used to look up per-class thresholds.
    pchembl_threshold
        A single pChEMBL minimum, or a dict of ``{target_class: minimum}``.
        Defaults to :data:`PCHEMBL_THRESHOLDS`.
    default_pchembl
        Threshold applied to target classes absent from the dict, so tables
        whose class vocabulary differs from :data:`PCHEMBL_THRESHOLDS` still
        filter instead of raising; `verbose` reports which classes took the
        fallback.
    verbose
        Report how many rows each step removed.

    Returns
    -------
    pandas.DataFrame
        The surviving rows, with the helper columns ``keep`` and
        ``pchembl_active`` retained so the decisions stay inspectable.
    """
    import numpy as np
    import pandas as pd

    df = dataframe.copy()
    n0 = len(df)

    def _report(step, n_before):
        if verbose:
            print(f"  {step}: {n_before:,} -> {len(df):,} "
                  f"({n_before - len(df):,} removed)")

    df["keep"] = False

    if drug_max_phase is not None:
        n = len(df)
        wanted = ([drug_max_phase] if isinstance(drug_max_phase, (int, float))
                  else list(drug_max_phase))
        df = df[df[COL_MAX_PHASE].isin(wanted)]
        _report(f"max_phase in {wanted}", n)

    if add_drug_mechanism:
        protected = df[COL_MECHANISM].notnull()
        df.loc[protected, "keep"] = True
        if verbose:
            print(f"  protected by drug mechanism: {int(protected.sum()):,}")

    if assay_type is not None:
        n = len(df)
        wanted = [assay_type] if isinstance(assay_type, str) else list(assay_type)
        df = df[df[COL_ASSAY_TYPE].isin(wanted) | df["keep"]]
        _report(f"assay_type in {wanted}", n)

    if remove_inactive:
        n = len(df)
        df = df[~df[COL_ACTIVITY_COMMENT].isin(INACTIVE_COMMENTS) | df["keep"]]
        _report("inactive activity comments", n)

    if include_active:
        active = df[COL_ACTIVITY_COMMENT].isin(ACTIVE_COMMENTS)
        df.loc[active, "keep"] = True
        if verbose:
            print(f"  protected by active comment: {int(active.sum()):,}")

    if pchembl_threshold is None:
        pchembl_threshold = PCHEMBL_THRESHOLDS
    if pchembl_threshold is not None:
        n = len(df)
        if isinstance(pchembl_threshold, dict):
            if pchembl_target_column not in df.columns:
                raise KeyError(
                    f"pchembl_target_column={pchembl_target_column!r} is not a "
                    f"column. Pass a single float threshold instead, or name "
                    f"the column holding the target class.")
            cutoff = (df[pchembl_target_column].map(pchembl_threshold)
                      .astype(float))
            missing = sorted(set(
                df.loc[cutoff.isna(), pchembl_target_column].dropna().unique()))
            if missing and verbose:
                print(f"  target classes without a threshold, using "
                      f"{default_pchembl}: {missing}")
            cutoff = cutoff.fillna(default_pchembl)
        else:
            cutoff = pd.Series(float(pchembl_threshold), index=df.index)
        df["pchembl_active"] = (
            pd.to_numeric(df[COL_PCHEMBL], errors="coerce") >= cutoff)
        df = df[df["pchembl_active"] | df["keep"]]
        _report("pChEMBL threshold", n)

    if verbose:
        print(f"  kept {len(df):,} of {n0:,} activities "
              f"({100 * len(df) / max(n0, 1):.2f}%)")
    return df


def chembl_targets_to_dict(dataframe, sep: str = "|") -> Dict[str, List[str]]:
    """Collapse filtered activities into ``{'CHEMBL_ID|NAME': [genes]}``.

    The gene column holds pipe-separated synonyms for one protein, so it is
    split and de-duplicated per compound.
    """
    out: Dict[str, List[str]] = {}
    for (chembl_id, name), sub in dataframe.groupby(
            [COL_CHEMBL_ID, COL_PREF_NAME], dropna=False):
        genes = set()
        for entry in sub[COL_SYNONYMS].dropna():
            genes.update(g for g in str(entry).split(sep) if g)
        if genes:
            out[f"{chembl_id}|{name}"] = sorted(genes)
    return out


def load_chembl_targets(
    chembl_pkl: Optional[str] = None,
    data_dir: Optional[str] = None,
    min_genes: int = 1,
    return_table: bool = False,
    **filter_kwargs,
):
    """Build drug-target gene sets from ChEMBL, ready for :func:`piaso.tl.score`.

    Fetches the table on first use (~2.7 GB, see :func:`fetch_chembl`), applies
    :func:`filter_chembl_activities`, and returns
    ``{'CHEMBL_ID|DRUG NAME': [target gene symbols]}``.

    Parameters
    ----------
    chembl_pkl
        Use this file instead of the cached one.
    data_dir
        Look here before the default cache.
    min_genes
        Drop drugs with fewer than this many targets. A one-gene set is a
        legitimate result for a selective drug, so the default keeps them.
    return_table
        Also return the filtered DataFrame.
    **filter_kwargs
        Passed to :func:`filter_chembl_activities` — `drug_max_phase`,
        `assay_type`, `pchembl_threshold` and so on.

    Returns
    -------
    dict, or (dict, DataFrame) when ``return_table=True``
    """
    import pandas as pd

    path = resolve_chembl_path(chembl_pkl=chembl_pkl, data_dir=data_dir)
    if path is None:
        path = fetch_chembl(dest_dir=data_dir)

    raw = pd.read_pickle(path)
    filtered = filter_chembl_activities(raw, **filter_kwargs)
    targets = chembl_targets_to_dict(filtered)
    if min_genes > 1:
        targets = {k: v for k, v in targets.items() if len(v) >= min_genes}
    return (targets, filtered) if return_table else targets
