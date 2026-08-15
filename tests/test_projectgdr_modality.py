"""projectGDR must not confuse the query's modality with the reference's.

`modality` describes the QUERY; `state['modality']` describes the REFERENCE. They
were one parameter, so a reference built on one modality could be scored against
a query read from another with nothing to notice.

This is not catchable downstream. projectGDR does guard the gene space — it
raises when over half the marker sets recover <20% of their genes — but RNA and
gene-activity are **both keyed by gene symbol**, so a GA-vs-RNA mix passes that
guard on matching names while the values come from the wrong assay. The result is
a plausible embedding computed against the wrong reference.

Covered here:

1. a query modality differing from the reference's warns (it is legitimate, but
   never something to do by accident);
2. matching modalities are silent;
3. an explicit `reference_modality` contradicting the saved state is an error,
   and it fires before any expensive completion work;
4. `reference_modality` defaults to the recorded value, so existing calls are
   unchanged.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("cytome")
import cytome  # noqa: E402

import piaso  # noqa: E402
from piaso.tools._projectGDR import projectGDR, GDR_REFERENCE_KEY, _get_state  # noqa: E402

N_CELLS, N_GENES, N_TYPES = 240, 120, 4
SEED = 0


def _anndata(n_cells=N_CELLS, n_genes=N_GENES, seed=SEED):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    labels = np.array([f"ct{i % N_TYPES}" for i in range(n_cells)])
    X = rng.poisson(0.6, size=(n_cells, n_genes)).astype(np.float32)
    # give each cell type a block of elevated genes so COSG finds real markers
    for k in range(N_TYPES):
        lo, hi = k * (n_genes // N_TYPES), (k + 1) * (n_genes // N_TYPES)
        X[labels == f"ct{k}", lo:hi] += rng.poisson(6.0, size=((labels == f"ct{k}").sum(), hi - lo))
    a = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"cell_type": labels},
                         index=[f"cell{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)]))
    a.layers["counts"] = a.X
    return a


@pytest.fixture(scope="module")
def reference_cytome(tmp_path_factory):
    """A cytome reference built (and recorded) on modality 'RNA'."""
    out = tmp_path_factory.mktemp("gdr") / "ref.cytome"
    a = _anndata()
    ds = cytome.from_anndata(a, modality="RNA", output=str(out))
    ds.cells["cell_type"] = a.obs.cell_type.astype(str).values
    ds.flush()
    piaso.tl.infog(ds, modality="RNA", layer="counts", key_added="infog",
                   save_layer=True, verbosity=0)
    piaso.tl.runGDR(ds, groupby="cell_type", batch_key=None, layer="infog",
                    n_gene=8, mu=10.0, n_svd_dims=10, resolution=1.0,
                    key_added="X_gdr", max_workers=2, random_seed=SEED,
                    save_reference=True, verbosity=0)
    ds.close()
    return str(out)


@pytest.fixture(scope="module")
def query_cytome(tmp_path_factory):
    out = tmp_path_factory.mktemp("gdr") / "query.cytome"
    a = _anndata(n_cells=160, seed=SEED + 1)
    ds = cytome.from_anndata(a, modality="RNA", output=str(out))
    ds.flush()
    ds.close()
    return str(out)


def test_reference_records_its_modality(reference_cytome):
    ds = cytome.open(reference_cytome)
    try:
        state = _get_state(ds)
        assert state is not None, f"no {GDR_REFERENCE_KEY} written"
        assert state.get("modality") == "RNA"
    finally:
        ds.close()


def test_matching_modalities_are_silent(reference_cytome, query_cytome):
    q = cytome.open(query_cytome)
    try:
        # pytest.warns(None) is gone in modern pytest; record explicitly.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            projectGDR(q, reference_cytome, modality="RNA", counts_layer="counts",
                       key_added="X_p_ok", batch_size=64, verbosity=0)
        msgs = [str(w.message) for w in rec
                if "reference modality" in str(w.message)]
        assert not msgs, msgs
    finally:
        q.close()


def test_query_modality_mismatch_warns(reference_cytome, query_cytome):
    """The dangerous case: names match, values do not. Must be announced.

    The fixture has no 'GA' modality, so the projection itself will fail shortly
    after — that is fine and beside the point. What must hold is that the warning
    is emitted *before* any of that, so a user whose cytome DOES have both
    modalities is told rather than silently given wrong coordinates.
    """
    q = cytome.open(query_cytome)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                projectGDR(q, reference_cytome, modality="GA", counts_layer="counts",
                           key_added="X_p_mismatch", batch_size=64, verbosity=0)
            except Exception:
                pass                       # downstream failure is expected here
        msgs = [str(w.message) for w in rec
                if "differs from the reference modality" in str(w.message)]
        assert msgs, [str(w.message) for w in rec]
        assert "GA" in msgs[0] and "RNA" in msgs[0]
    finally:
        q.close()


def test_explicit_reference_modality_contradicting_state_errors(reference_cytome, query_cytome):
    q = cytome.open(query_cytome)
    try:
        with pytest.raises(ValueError, match="contradicts the modality recorded"):
            projectGDR(q, reference_cytome, modality="RNA", reference_modality="ATAC",
                       counts_layer="counts", key_added="X_p_bad", batch_size=64, verbosity=0)
    finally:
        q.close()


def test_reference_modality_defaults_to_recorded(reference_cytome, query_cytome):
    """Omitting reference_modality must behave exactly as before the change."""
    q = cytome.open(query_cytome)
    try:
        projectGDR(q, reference_cytome, modality="RNA", counts_layer="counts",
                   key_added="X_p_default", batch_size=64, verbosity=0)
        projectGDR(q, reference_cytome, modality="RNA", reference_modality="RNA",
                   counts_layer="counts", key_added="X_p_explicit", batch_size=64, verbosity=0)
        a = np.asarray(q.embeddings["X_p_default"])
        b = np.asarray(q.embeddings["X_p_explicit"])
        assert a.shape == b.shape
        assert np.allclose(a, b), "explicit reference_modality changed the result"
    finally:
        q.close()
