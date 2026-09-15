"""`key_added` must not be accepted and then quietly dropped.

`score(adata, gene_list={"ASPIRIN": [...]}, key_added="aspirin_targets")` takes
the multi-set branch, which names each score by its own key and so had nothing
to do with `key_added`. Nothing was written and nothing was said. The next line
of the tutorial coloured an embedding by "aspirin_targets" and raised
`KeyError: Could not find key aspirin_targets` -- a message pointing at the
plot, one call after the place that dropped it.

One set in a dict is one set, and the documented contract for a single set is
`adata.obs[key_added]`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _key_added_warnings(record):
    return [str(w.message) for w in record if "key_added" in str(w.message)]


def test_one_set_in_a_dict_writes_the_column(scored_adata):
    adata, genes = scored_adata
    import piaso

    before = set(adata.obs.columns)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        piaso.tl.score(adata, gene_list={"SET_A": genes}, layer="infog",
                       key_added="my_key", verbosity=0)

    assert "my_key" in adata.obs.columns, (
        f"key_added was accepted and dropped; new columns: "
        f"{sorted(set(adata.obs.columns) - before)}")
    assert np.isfinite(adata.obs["my_key"].to_numpy()).any()
    assert not _key_added_warnings(rec), (
        "one set is not 'several sets' -- it should not warn")


def test_several_sets_say_so_rather_than_dropping_it(scored_adata):
    adata, genes = scored_adata
    import piaso

    other = [g for g in adata.var_names[8:16]]
    before = set(adata.obs.columns)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        piaso.tl.score(adata, gene_list={"A": genes, "B": other},
                       layer="infog", key_added="ignored", verbosity=0)

    assert "ignored" not in adata.obs.columns
    assert _key_added_warnings(rec), (
        "with several sets key_added cannot be honoured, and silence is what "
        f"sent the caller to the wrong place; new columns: "
        f"{sorted(set(adata.obs.columns) - before)}")
