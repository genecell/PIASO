"""calculateScoreParallel scores the same layer on a store as on an AnnData
(audit 2026-10-27, raised by the analysis-tools review, I1 findings 3 and 4).

On a store it scored cytome_layer, default 'counts', and ignored score_layer:
score_layer='infog' scored INFOG on an AnnData and raw counts on the same data
in a store. score_layer is now the one parameter for both; cytome_layer (kept
for old calls) overrides it only when given; neither given means the stored
counts, as None means .X on an AnnData. Every internal caller passes
cytome_layer explicitly, so their results do not change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")


@pytest.fixture
def pair(tmp_path):
    import cytome
    import piaso
    from anndata import AnnData
    rng = np.random.default_rng(2)
    n, g = 400, 120
    lam = np.full((n, g), 0.4)
    lam[: n // 2, :15] *= 8
    X = sp.csr_matrix(rng.poisson(lam * rng.lognormal(0, 0.8, n)[:, None]).astype(np.float32))
    a = AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    ds = cytome.from_anndata(a, modality="RNA", output=str(tmp_path / "s.cytome"))
    piaso.tl.infog(ds, save_layer=True, verbosity=0)
    a.layers["infog"] = ds.RNA.layer("infog").to_memory().tocsr()
    sets = {"A": [f"G{i}" for i in range(10)], "B": [f"G{i}" for i in range(40, 52)]}
    yield a, ds, sets
    ds.close()


def test_a_store_scores_score_layer(pair):
    import piaso
    a, ds, sets = pair
    s_store, names = piaso.tl.calculateScoreParallel(ds, sets, score_layer="infog", max_workers=1)
    s_explicit, _ = piaso.tl.calculateScoreParallel(ds, sets, cytome_layer="infog", max_workers=1)
    np.testing.assert_array_equal(s_store, s_explicit)
    s_ann, _ = piaso.tl.calculateScoreParallel(a, sets, score_layer="infog", max_workers=1)
    np.testing.assert_allclose(s_store, s_ann, rtol=1e-5, atol=1e-6)
    # cytome_layer, when given, still decides
    s_counts, _ = piaso.tl.calculateScoreParallel(ds, sets, score_layer="infog", cytome_layer="counts",
                                                  max_workers=1)
    assert not np.allclose(s_counts, s_store)


def test_the_default_still_scores_the_stored_counts(pair):
    import piaso
    a, ds, sets = pair
    s_default, _ = piaso.tl.calculateScoreParallel(ds, sets, max_workers=1)
    s_counts, _ = piaso.tl.calculateScoreParallel(ds, sets, cytome_layer="counts", max_workers=1)
    np.testing.assert_array_equal(s_default, s_counts)
    s_ann, _ = piaso.tl.calculateScoreParallel(a, sets, max_workers=1)        # .X: the counts
    np.testing.assert_allclose(s_default, s_ann, rtol=1e-5, atol=1e-6)
