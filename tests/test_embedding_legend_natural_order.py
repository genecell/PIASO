"""Leiden legends must read 0, 1, 2, 10 -- not 0, 1, 10, 2.

The renderer sorted category names with plain ``sorted``, which is
lexicographic on strings. With ten or more clusters the legend ran
0, 1, 10, 11, 2 while every other PIASO plot -- which resolves order through
``resolve_group_order`` -- ran 0, 1, 2. A cytome round-trip stores the column
as plain strings, so no categorical order survives to fall back on and this
sort is the only thing deciding it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest


def test_render_orders_numeric_labels_naturally():
    from piaso.plotting._plotEmbedding import _render_embedding
    import matplotlib.pyplot as plt

    values = np.array([str(i) for i in range(12)] * 3)
    xy = np.random.RandomState(0).rand(len(values), 2)

    fig, ax = plt.subplots()
    _render_embedding(ax, xy, values, is_cat=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else []
    plt.close(fig)
    if not labels:
        pytest.skip("this renderer draws no legend in isolation")
    assert labels == [str(i) for i in range(12)], labels


def test_natural_key_beats_lexicographic():
    from piaso.plotting._group_order import natural_key

    labels = [str(i) for i in range(12)]
    assert sorted(labels, key=natural_key) == labels
    assert sorted(labels) != labels, "lexicographic order should differ here"


def test_embedding_and_sankey_use_one_order():
    """Both must come from resolve_group_order / natural_key, not local sorts."""
    import inspect
    from piaso.plotting import _plotEmbedding, _plotSankey

    emb = inspect.getsource(_plotEmbedding)
    assert "natural_key" in emb, (
        "the embedding renderer must order categories with natural_key")
    snk = inspect.getsource(_plotSankey)
    assert "resolve_group_order" in snk
