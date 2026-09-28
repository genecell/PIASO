"""With a cytome that cannot hand chunks over (< 0.3.4), runSVD still works:
it warns, and the engine reads the store itself, as before."""
import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cytome
import piaso

try:
    from piaso import _piaso
    HAVE = hasattr(_piaso, "svd_pass")
except Exception:  # pragma: no cover
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="Rust extension not built")


def test_runSVD_without_chunk_source_warns_and_gives_the_same_answer(tmp_path, monkeypatch):
    X = sp.random(300, 800, density=0.05, format="csr", random_state=4, dtype=np.float32)
    X.data = np.ceil(X.data * 4)
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(300)]
    A.var_names = [f"chr1:{1000 * j}-{1000 * j + 500}" for j in range(800)]
    p = tmp_path / "old.cytome"
    cytome.from_anndata(A, modality="ATAC", output=str(p)).close()
    ds = cytome.open(str(p))
    kw = dict(modality="ATAC", measurement="counts", auto_tfidf=True, n_components=4, n_iter=4,
              random_state=1, return_svd=True, use_highly_variable=False, verbosity=0)
    e_new, s_new, _ = piaso.tl.runSVD(ds, key_added="X_new", **kw)
    monkeypatch.delattr(type(ds), "chunk_source")
    with pytest.warns(RuntimeWarning, match="cytome >= 0.3.4"):
        e_old, s_old, _ = piaso.tl.runSVD(ds, key_added="X_old", **kw)
    assert np.array_equal(e_new, e_old) and np.array_equal(s_new, s_old)
    ds.cells["after"] = np.arange(300); ds.flush()
    ds.close()


def test_released_store_without_cytome_released_closes_the_given_dataset(tmp_path, monkeypatch):
    import cytome as C
    from piaso.utils._cytome_compat import released_store
    X = sp.random(20, 30, density=0.2, format="csr", random_state=1, dtype=np.float32)
    A = ad.AnnData(X=X)
    A.obs_names = [f"c{i}" for i in range(20)]
    A.var_names = [f"g{j}" for j in range(30)]
    p = tmp_path / "r.cytome"
    C.from_anndata(A, output=str(p)).close()
    ds = C.open(str(p))
    monkeypatch.delattr(C, "released")
    with released_store(ds) as held:
        assert held == [ds] and ds.is_closed
    assert not ds.is_closed
    ds.close()
