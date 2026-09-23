"""`color=` may be the values themselves, not the name of a column.

Anything computed on the fly -- a regulon activity vector, a per-cell score
that was never written back, a mask -- had to be assigned into `adata.obs`
first. Passing the array directly reached `color in adata.obs.columns` and
came back as `TypeError: unhashable type: 'numpy.ndarray'`, which says nothing
about what was wrong or what to do about it.

Lists and tuples are deliberately still "one panel per colour": that meaning is
far more common than a per-cell Python list, and changing it would break every
multi-panel call. Pass a numpy array or a pandas Series.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def adata():
    import anndata as ad
    rng = np.random.default_rng(0)
    n = 150
    a = ad.AnnData(
        X=rng.random((n, 4)).astype("float32"),
        obs=pd.DataFrame(
            {"batch": np.array(["b1", "b2"])[rng.integers(0, 2, n)],
             "score": rng.random(n)},
            index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(4)]),
    )
    a.obsm["X_umap"] = rng.random((n, 2))
    return a


def _plot(adata, color, **kw):
    import piaso
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, basis="X_umap", color=color, ax=ax,
                           show=False, **kw)
    return ax


def test_float_array_is_drawn_as_a_continuous_colour(adata):
    rng = np.random.default_rng(1)
    ax = _plot(adata, rng.random(adata.n_obs))
    assert ax.get_legend() is None          # a colorbar, not a category legend
    assert ax.collections
    plt.close("all")


def test_string_array_is_drawn_as_categories(adata):
    rng = np.random.default_rng(1)
    labels = np.array(["A", "B", "C"])[rng.integers(0, 3, adata.n_obs)]
    ax = _plot(adata, labels)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B", "C"]
    plt.close("all")


def test_an_array_matches_the_same_values_in_obs(adata):
    """Passing the column and passing its values must draw the same thing."""
    by_name = _plot(adata, "score")
    named = [c.get_offsets().shape[0] for c in by_name.collections]
    by_value = _plot(adata, adata.obs["score"].to_numpy())
    assert [c.get_offsets().shape[0] for c in by_value.collections] == named
    plt.close("all")


def test_a_named_series_titles_the_panel(adata):
    rng = np.random.default_rng(1)
    ax = _plot(adata, pd.Series(rng.random(adata.n_obs), name="pseudotime"))
    assert ax.get_title() == "pseudotime"
    plt.close("all")


def test_an_unnamed_array_gets_a_neutral_title(adata):
    rng = np.random.default_rng(1)
    ax = _plot(adata, rng.random(adata.n_obs))
    assert ax.get_title() == "value"
    plt.close("all")


def test_a_categorical_keeps_its_own_order(adata):
    """Alphabetical would put 'high' before 'low'; the Categorical says otherwise."""
    rng = np.random.default_rng(1)
    vals = pd.Categorical(np.array(["low", "high"])[rng.integers(0, 2, adata.n_obs)],
                          categories=["low", "high"])
    ax = _plot(adata, vals)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["low", "high"]
    plt.close("all")


def test_wrong_length_names_both_numbers(adata):
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match=r"has 155 values but the data has 150 cells"):
        _plot(adata, rng.random(adata.n_obs + 5))
    plt.close("all")


def test_a_two_dimensional_array_is_refused(adata):
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="one value per cell"):
        _plot(adata, rng.random((adata.n_obs, 3)))
    plt.close("all")


def test_a_column_vector_is_accepted(adata):
    """(n, 1) is one value per cell, just shaped awkwardly."""
    rng = np.random.default_rng(1)
    ax = _plot(adata, rng.random((adata.n_obs, 1)))
    assert ax.collections
    plt.close("all")


def test_a_list_still_means_multiple_panels(adata):
    """The existing meaning of a list must not change."""
    import piaso
    piaso.pl.plotEmbedding(adata, basis="X_umap", color=["score", "G0"],
                           show=False)
    assert len([ax for ax in plt.gcf().get_axes() if ax.get_title()]) == 2
    plt.close("all")


def test_split_panels_accept_an_array_too(adata):
    import piaso
    rng = np.random.default_rng(1)
    piaso.pl.plot_embeddings_split(
        adata, basis="X_umap", color=pd.Series(rng.random(adata.n_obs), name="pt"),
        splitby="batch", show=False)
    titles = [ax.get_title() for ax in plt.gcf().get_axes() if ax.get_title()]
    assert titles == ["pt in\nb1", "pt in\nb2"]
    plt.close("all")


def test_groups_still_works_on_an_array(adata):
    rng = np.random.default_rng(1)
    labels = np.array(["A", "B", "C"])[rng.integers(0, 3, adata.n_obs)]
    ax = _plot(adata, labels, groups=["A"])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "other"]
    plt.close("all")


# ── a Series is aligned by name when it can be ───────────────────────────────

def _labels_of(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_a_reordered_series_is_aligned_by_index(adata):
    """A Series whose index is a permutation of obs_names colours the right
    cells; positional use would silently permute them."""
    import piaso
    cats = np.array(["A", "B"])[np.arange(adata.n_obs) % 2]
    s = pd.Series(cats, index=adata.obs_names)[::-1]          # reversed
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, basis="X_umap", color=s, ax=ax, show=False)
    # cell 0 is 'A': the A collection must contain cell 0's coordinate
    coll = {h.get_label(): h for h in ax.collections}
    xy0 = adata.obsm["X_umap"][0]
    assert any(np.allclose(p, xy0) for p in coll["A"].get_offsets())
    plt.close(fig)


def test_a_range_indexed_series_is_positional(adata):
    import piaso
    s = pd.Series(np.arange(adata.n_obs, dtype=float))
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, basis="X_umap", color=s, ax=ax, show=False)
    assert ax.collections
    plt.close(fig)


def test_a_partially_matching_index_raises(adata):
    import piaso
    idx = list(adata.obs_names)
    idx[0] = "not-a-cell"
    s = pd.Series(np.random.default_rng(0).random(adata.n_obs), index=idx)
    with pytest.raises(ValueError, match=r"matches \d+ of its \d+ labels"):
        piaso.pl.plotEmbedding(adata, basis="X_umap", color=s, show=False)
    plt.close("all")


def test_an_unrelated_string_index_is_positional(adata):
    import piaso
    s = pd.Series(np.random.default_rng(0).random(adata.n_obs),
                  index=[f"z{i}" for i in range(adata.n_obs)])
    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(adata, basis="X_umap", color=s, ax=ax, show=False)
    assert ax.collections
    plt.close(fig)


def test_non_unique_obs_names_fall_back_to_position_with_a_warning(adata):
    import anndata as ad
    import piaso
    dup = adata.copy()
    names = list(dup.obs_names); names[1] = names[0]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dup.obs_names = names
    s = pd.Series(np.random.default_rng(0).random(dup.n_obs), index=dup.obs_names)
    with pytest.warns(UserWarning, match="not unique"):
        fig, ax = plt.subplots()
        piaso.pl.plotEmbedding(dup, basis="X_umap", color=s, ax=ax, show=False)
    plt.close("all")
