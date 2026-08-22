"""INFOG then SVD on a cytome must compose.

On a cytome `infog()` defaults to `save_layer=False`: it records the
parameters and writes no layer. Nothing reads those parameters back yet, so
`runSVD(layer="infog")` then dies with `KeyError: Matrix not found:
RNA_infog` several minutes into a pipeline. These pin the working path and
the shape of the failure, so that if the lazy layer is ever wired up the test
that changes is the one about the default.
"""
import numpy as np
import pytest
import scipy.sparse as sp

cytome = pytest.importorskip("cytome")
anndata = pytest.importorskip("anndata")
import piaso  # noqa: E402


@pytest.fixture
def cyt(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.6, size=(300, 60)).astype(np.float32))
    a = anndata.AnnData(X=X)
    a.obs["batch"] = [f"b{i % 3}" for i in range(a.n_obs)]
    p = tmp_path / "t.cytome"
    ds = cytome.from_anndata(a, output=str(p))
    ds.close()
    return str(p)


def _matrices(path):
    ds = cytome.open(path)
    try:
        return {r[0] for r in ds._conn.execute("SELECT matrix_name FROM matrix_meta")}
    finally:
        ds.close()


def test_default_records_params_and_writes_no_layer(cyt):
    piaso.tl.infog(cyt, n_top_genes=20, verbosity=0)
    assert "RNA_infog" not in _matrices(cyt)
    ds = cytome.open(cyt)
    try:
        assert ds.metadata.get("RNA_infog_params") is not None
    finally:
        ds.close()


def test_save_layer_materialises_it_and_svd_can_read_it(cyt):
    piaso.tl.infog(cyt, n_top_genes=20, save_layer=True, verbosity=0)
    assert "RNA_infog" in _matrices(cyt)

    piaso.tl.runSVD(cyt, layer="infog", n_components=5, key_added="X_svd")
    ds = cytome.open(cyt)
    try:
        # Embeddings are modality-prefixed on a cytome, so key_added="X_svd"
        # is stored as "RNA_svd". Worth pinning: a caller who passes X_svd and
        # then greps the file for X_svd finds nothing.
        assert "RNA_svd" in list(ds.list_embeddings())
    finally:
        ds.close()


def test_svd_normalizes_on_the_fly_when_the_layer_is_absent(cyt):
    """The whole point: infog() records params and writes nothing, and
    runSVD(layer='infog') used to die with KeyError several minutes in."""
    piaso.tl.infog(cyt, n_top_genes=20, verbosity=0)      # no save_layer
    assert "RNA_infog" not in _matrices(cyt)

    piaso.tl.runSVD(cyt, layer="infog", n_components=5, key_added="X_svd")
    ds = cytome.open(cyt)
    try:
        assert "RNA_svd" in list(ds.list_embeddings())
        assert "RNA_infog" not in _matrices(cyt), "should not have materialised it"
    finally:
        ds.close()


def test_compute_on_fly_false_refuses_rather_than_recomputing(cyt):
    piaso.tl.infog(cyt, n_top_genes=20, verbosity=0)
    with pytest.raises(KeyError, match="compute_on_fly"):
        piaso.tl.runSVD(cyt, layer="infog", n_components=5,
                        key_added="X_svd", compute_on_fly=False)


def test_no_layer_and_no_params_says_to_run_infog(cyt):
    """Without infog() there are no HVGs either, so the earlier guard fires
    first. Its message already names the fix, which is what matters."""
    with pytest.raises(KeyError, match="infog first"):
        piaso.tl.runSVD(cyt, layer="infog", n_components=5, key_added="X_svd")


def test_on_the_fly_matches_the_materialised_layer(cyt, tmp_path):
    """An SVD from parameters and one from the written layer must agree."""
    import numpy as np
    import shutil

    other = str(tmp_path / "materialised.cytome")
    shutil.copy2(cyt, other)

    piaso.tl.infog(cyt, n_top_genes=20, verbosity=0)                 # params only
    piaso.tl.infog(other, n_top_genes=20, save_layer=True, verbosity=0)

    piaso.tl.runSVD(cyt, layer="infog", n_components=5, key_added="X_svd",
                    random_state=0)
    piaso.tl.runSVD(other, layer="infog", n_components=5, key_added="X_svd",
                    random_state=0)

    a, b = cytome.open(cyt), cytome.open(other)
    try:
        ea = np.asarray(a.embeddings["RNA_svd"])
        eb = np.asarray(b.embeddings["RNA_svd"])
        # sign of an SVD component is arbitrary; compare magnitudes
        assert np.allclose(np.abs(ea), np.abs(eb), atol=1e-4), \
            np.abs(np.abs(ea) - np.abs(eb)).max()
    finally:
        a.close(); b.close()
