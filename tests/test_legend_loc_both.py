"""``legend_loc='both'`` draws the on-data labels *and* the side legend.

Deciding what a cluster is wants the label where the points are; reading a
colour off the plot wants the list. Before this, a tutorial that wanted both
had to render the same embedding twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
anndata = pytest.importorskip("anndata")

import piaso
from piaso.plotting._plotEmbedding import _VALID_LEGEND_LOC


def _toy(n=180, k=4):
    rng = np.random.default_rng(0)
    a = anndata.AnnData(rng.random((n, 5)).astype("float32"))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obs["group"] = pd.Categorical([f"c{i % k}" for i in range(n)])
    return a


def _fig(a, loc):
    """``return_fig=True`` gives back ``(fig, ax)`` for a single-panel plot."""
    out = piaso.pl.embedding(a, basis="X_umap", color="group", legend_loc=loc,
                             show=False, return_fig=True)
    return out[0] if isinstance(out, tuple) else out


def _counts(fig):
    ax = fig.axes[0]
    return len(ax.texts), int(ax.get_legend() is not None)


def test_both_is_a_valid_choice():
    assert "both" in _VALID_LEGEND_LOC


@pytest.mark.parametrize("loc,want_texts,want_legend", [
    ("on_data", True, False),
    ("right", False, True),
    ("both", True, True),
    ("none", False, False),
])
def test_each_mode_draws_what_it_says(loc, want_texts, want_legend):
    a = _toy()
    fig = _fig(a, loc)
    texts, legend = _counts(fig)
    assert (texts > 0) is want_texts, f"{loc}: {texts} on-data labels"
    assert bool(legend) is want_legend, f"{loc}: legend={bool(legend)}"
    plt.close(fig)


def test_both_labels_every_category():
    a = _toy(k=4)
    fig = _fig(a, "both")
    labels = {t.get_text() for t in fig.axes[0].texts}
    assert {"c0", "c1", "c2", "c3"} <= labels
    plt.close(fig)


# ---------------------------------------------------------------------------
# 'both' worked for a release before either docstring mentioned it, so a caller
# reading help() saw three values and had no reason to try the fourth. A value
# the code accepts but the documentation omits is only half-shipped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [piaso.pl.plotEmbedding,
                                piaso.pl.plot_embeddings_split])
@pytest.mark.parametrize("value", _VALID_LEGEND_LOC)
def test_every_accepted_legend_loc_is_documented(fn, value):
    doc = fn.__doc__ or ""
    assert f"``'{value}'``" in doc, (
        f"{fn.__name__} accepts legend_loc={value!r} but never says so; "
        "help() is the only place most callers will look")


@pytest.mark.parametrize("fn", [piaso.pl.plotEmbedding,
                                piaso.pl.plot_embeddings_split])
def test_the_documented_default_is_the_real_one(fn):
    """The split-panel docstring claimed 'right margin', which is not a value
    the validator accepts -- copying it out of the docs raised ValueError."""
    import inspect

    default = inspect.signature(fn).parameters["legend_loc"].default
    assert default in _VALID_LEGEND_LOC
    doc = fn.__doc__ or ""
    head = doc[:doc.index("legend_marker_size")] if "legend_marker_size" in doc else doc
    assert f"'{default}'" in head.split("legend_loc")[1][:400]
