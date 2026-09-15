"""pandas 3 types a string column as ``StringDtype``, not ``object``.

That single change is what pandas 3 does to us. ``future.infer_string`` is on
by default there, so ``adata.obs['cell_type']`` is no longer an object column
and every check that asked **numpy** about its dtype either answered wrongly or
raised:

* ``np.issubdtype(StringDtype, np.number)`` raises ``TypeError`` rather than
  returning False, so the numeric-versus-categorical dispatch crashed and took
  every categorical embedding with it;
* a writer that reached TEXT only via ``is_object_dtype`` fell through to its
  INTEGER default, and cluster labels ``"0"``, ``"1"``, ``"2"`` came back as
  ``0``, ``1``, ``2``.

The point of this file is that **none of it needs pandas 3 to test**: pandas 2
has carried the same opt-in since 2.1, so the behaviour is reachable here and
stays covered on whatever the CI happens to install.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture
def infer_string():
    """pandas 3's default, turned on for one test."""
    try:
        with pd.option_context("future.infer_string", True):
            yield True
    except (KeyError, ValueError):                    # too old to opt in
        pytest.skip("pandas has no future.infer_string option")


def test_a_string_column_is_not_object_under_the_new_default(infer_string):
    """The premise. If this stops holding, the rest is no longer needed."""
    col = pd.DataFrame({"ct": ["a", "b"]})["ct"]
    assert not pd.api.types.is_object_dtype(col), (
        "the whole class of bug depends on string columns leaving object "
        "dtype; they have not")


def test_numpy_cannot_be_asked_about_that_dtype(infer_string):
    """Why the fix is 'ask pandas', not 'ask numpy more carefully'."""
    col = pd.DataFrame({"ct": ["a", "b"]})["ct"]
    with pytest.raises(TypeError):
        np.issubdtype(col.dtype, np.number)


def test_the_dispatch_answers_instead_of_raising(infer_string):
    from piaso.plotting._plotEmbedding import _is_numeric_dtype

    df = pd.DataFrame({
        "ct": ["a", "b", "c"],
        "n_counts": [1, 2, 3],
        "frac": [0.1, 0.2, 0.3],
        "flag": [True, False, True],
    })
    assert _is_numeric_dtype(df["ct"]) is False
    assert _is_numeric_dtype(df["n_counts"]) is True, "an integer metric is continuous"
    assert _is_numeric_dtype(df["frac"]) is True
    assert _is_numeric_dtype(df["flag"]) is False, (
        "a boolean mask is two groups, not a colour bar -- and this is the one "
        "place pandas and numpy disagree, pandas calling bool numeric")


def test_a_categorical_embedding_draws(infer_string):
    """End to end, because the dispatch is only interesting through a plot."""
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import piaso

    rng = np.random.default_rng(0)
    n = 60
    a = ad.AnnData(
        rng.random((n, 2), dtype=np.float32),
        obs=pd.DataFrame({"ct": rng.choice(["a", "b"], n)},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g1", "g2"]))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))

    fig, ax = plt.subplots()
    piaso.pl.plotEmbedding(a, color="ct", basis="X_umap", ax=ax, show=False)
    assert len(ax.collections) >= 2, "expected one scatter per category"
    plt.close("all")


def test_cluster_labels_survive_a_cytome_round_trip(infer_string, tmp_path):
    """The other half: a string label must not come back an integer.

    The fix is in cytome's entity writer (TEXT is the fallback now, not
    INTEGER); this asserts the property PIASO depends on, so if a cytome
    upgrade regresses it we learn here rather than from a plot whose
    categories stopped matching their stored order.
    """
    cytome = pytest.importorskip("cytome")

    ds = cytome.create(str(tmp_path / "t.cytome"))
    try:
        n = 6
        ds.set_entity("cells", pd.DataFrame({
            "cell_idx": np.arange(n),
            "barcode": [f"BC{i}" for i in range(n)],
        }))
        ds.flush()
        labels = np.array([str(i % 3) for i in range(n)])
        ds.cells["leiden"] = labels
        ds.flush()
        got = list(ds.cells["leiden"])
        assert all(isinstance(v, str) for v in got), (
            f"labels came back as {type(got[0]).__name__}; a string column that "
            "round-trips as an integer silently breaks category lookup")
        assert got == list(labels)
    finally:
        ds.close()
