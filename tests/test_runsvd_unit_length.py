"""runSVD(unit_length=True): each cell's embedding vector scaled to length 1 before it is stored, on every path.

The kept components (after drop_first_component, if set) are scaled; each cell's length before the scaling is kept
(adata.obs[key + '_length'] / the cytome cells table's '<embedding>_length'), so the scaling can be undone; S and Vt
are the decomposition's. leiden_local passes it to every group's SVD (svd_unit_length), and the workflow sets it for
every ATAC embedding it computes (svd_unit_length, default true in config.yaml; a dataset config that does not set
it takes that value, as for every parameter, and one that must keep the Euclidean embedding sets it to false).
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import yaml

import piaso


def toy(n_cells=300, n_genes=120, seed=0, zero_rows=()):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.5, 1.0, size=(1, n_genes)) * rng.gamma(2.0, 0.5, size=(n_cells, 1))
    X = rng.poisson(lam).astype(np.float32)
    for r in zero_rows:
        X[r] = 0
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


def norms(E):
    return np.linalg.norm(np.asarray(E, dtype=np.float64), axis=1)


def test_in_memory_stores_unit_vectors_and_their_lengths():
    a, b = toy(), toy()
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, drop_first_component=True, unit_length=True)
    piaso.tl.runSVD(b, use_highly_variable=False, n_components=5, drop_first_component=True)
    assert a.obsm["X_svd"].shape == (300, 5)
    assert np.allclose(norms(a.obsm["X_svd"]), 1.0, atol=1e-12)
    lengths = a.obs["X_svd_length"].to_numpy()
    assert np.allclose(lengths, norms(b.obsm["X_svd"]), rtol=1e-12)
    # reversible: the unit vectors times their lengths are the Euclidean embedding
    assert np.allclose(a.obsm["X_svd"] * lengths[:, None], b.obsm["X_svd"], rtol=1e-10, atol=1e-12)
    rec = a.uns["X_svd_params"]
    assert rec["unit_length"] is True and rec["length_column"] == "X_svd_length"
    assert rec["first_component_dropped"] is True and rec["components"] == [2, 6]
    # a rerun without the option leaves neither the record nor a stale length
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5)
    assert "X_svd_params" not in a.uns and "X_svd_length" not in a.obs.columns


def test_unit_length_alone_is_recorded_without_the_drop():
    a = toy()
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, unit_length=True)
    rec = a.uns["X_svd_params"]
    assert rec == {"unit_length": True, "length_column": "X_svd_length"}
    assert np.allclose(norms(a.obsm["X_svd"]), 1.0, atol=1e-12)


def test_a_cell_with_no_signal_stays_at_the_origin():
    a = toy(zero_rows=(3, 7))
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, unit_length=True)
    E = a.obsm["X_svd"]
    assert np.isfinite(E).all()
    assert np.all(E[[3, 7]] == 0) and np.all(a.obs["X_svd_length"].to_numpy()[[3, 7]] == 0)
    keep = np.setdiff1d(np.arange(300), [3, 7])
    assert np.allclose(norms(E[keep]), 1.0, atol=1e-12)


def test_streaming_returns_unit_vectors_and_the_decomposition_unchanged():
    a, b = toy(), toy()
    e1, s1, v1 = piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, streaming=True,
                                 unit_length=True, method="krylov", n_iter=8)
    e2, s2, v2 = piaso.tl.runSVD(b, use_highly_variable=False, n_components=5, streaming=True,
                                 method="krylov", n_iter=8)
    assert np.allclose(norms(e1), 1.0, atol=1e-12)
    assert np.array_equal(s1, s2) and np.array_equal(v1, v2)
    assert np.allclose(e1 * norms(e2)[:, None], e2, rtol=1e-10, atol=1e-12)


def test_a_subset_is_returned_at_unit_length_and_nothing_is_stored():
    a = toy()
    mask = np.zeros(300, bool)
    mask[::3] = True
    e, s, v = piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, cell_mask=mask, unit_length=True)
    assert e.shape == (100, 5) and np.allclose(norms(e), 1.0, atol=1e-12)
    assert "X_svd" not in a.obsm and "X_svd_length" not in a.obs.columns


def test_a_cytome_stores_unit_vectors_lengths_and_provenance_equal_to_the_anndata(tmp_path):
    import cytome
    a, b = toy(), toy()
    path = str(tmp_path / "toy.cytome")
    cytome.from_anndata(a, modality="RNA", output=path).close()
    piaso.tl.runSVD(path, modality="RNA", measurement="counts", use_highly_variable=False, n_components=5,
                    drop_first_component=True, unit_length=True, key_added="X_svd_unit",
                    method="krylov", n_iter=8, random_state=10)
    piaso.tl.runSVD(path, modality="RNA", measurement="counts", use_highly_variable=False, n_components=5,
                    drop_first_component=True, key_added="X_svd_eucl",
                    method="krylov", n_iter=8, random_state=10)
    piaso.tl.runSVD(b, use_highly_variable=False, n_components=5, drop_first_component=True,
                    unit_length=True, method="krylov", n_iter=8, random_state=10)
    ds = cytome.open(path, mode="r")
    try:
        unit = np.asarray(ds.embeddings["RNA_svd_unit"], dtype=np.float64)
        eucl = np.asarray(ds.embeddings["RNA_svd_eucl"], dtype=np.float64)
        lengths = np.asarray(ds.cells["RNA_svd_unit_length"], dtype=np.float64)
        prov = json.loads(ds.provenance.get_for_object("RNA_svd_unit")[-1]["parameters"])
        prov_e = json.loads(ds.provenance.get_for_object("RNA_svd_eucl")[-1]["parameters"])
    finally:
        ds.close()
    assert np.allclose(norms(unit), 1.0, atol=1e-6)                 # stored in float32
    assert np.allclose(lengths, norms(eucl), rtol=1e-6)
    assert np.allclose(unit * lengths[:, None], eucl, rtol=1e-5, atol=1e-6)
    assert prov["unit_length"] is True and prov["length_column"] == "RNA_svd_unit_length"
    assert prov["first_component_dropped"] is True
    assert "unit_length" not in prov_e
    # the AnnData and the store agree (the store keeps float32)
    assert np.allclose(unit, b.obsm["X_svd"], atol=1e-6)
    assert np.allclose(lengths, b.obs["X_svd_length"].to_numpy(), rtol=1e-6)


def test_leiden_local_passes_it_to_every_group(monkeypatch):
    from piaso.tools import _clustering
    seen = []
    real = _clustering.runSVD

    def spy(*args, **kw):
        seen.append(kw.get("unit_length"))
        return real(*args, **kw)

    monkeypatch.setattr(_clustering, "runSVD", spy)
    a = toy(n_cells=400)
    a.obs["coarse"] = np.where(np.arange(400) < 200, "A", "B")
    piaso.tl.leiden_local(a, groupby="coarse", resolution=0.5, dr_method="X_svd", key_added="ll",
                          n_components=5, svd_unit_length=True, verbosity=0)
    assert seen and all(v is True for v in seen)
    seen.clear()
    piaso.tl.leiden_local(a, groupby="coarse", resolution=0.5, dr_method="X_svd", key_added="ll",
                          n_components=5, svd_unit_length=False, verbosity=0)
    assert seen and all(v is False for v in seen)
    # left unset, each group's runSVD decides by what it factorises (test_runsvd_defaults.py)
    seen.clear()
    piaso.tl.leiden_local(a, groupby="coarse", resolution=0.5, dr_method="X_svd", key_added="ll",
                          n_components=5, verbosity=0)
    assert seen and all(v is None for v in seen)


SNAKEMAKE = os.environ.get("PIASO_SNAKEMAKE") or shutil.which("snakemake")
WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"


@pytest.mark.snakemake
@pytest.mark.skipif(not SNAKEMAKE, reason="snakemake (PIASO_SNAKEMAKE or PATH) not available")
def test_the_workflow_scales_every_atac_embedding(tmp_path):
    cfg = yaml.safe_load(open(WORKFLOW / "config.yaml"))
    assert cfg["params"]["svd_unit_length"] is True
    cfg["outdir"] = str(tmp_path / "out")
    cfg["datasets"] = {"e18_unsup": cfg["datasets"]["e18_unsup"]}
    path = tmp_path / "config.yaml"

    def lines(params):
        c = dict(cfg)
        c["params"] = params
        path.write_text(yaml.safe_dump(c))
        r = subprocess.run([SNAKEMAKE, "-s", str(WORKFLOW / "Snakefile"), "--configfile", str(path), "-np",
                            f"{cfg['outdir']}/e18_unsup/svd/.done"],
                           capture_output=True, text=True, cwd=str(tmp_path))
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        out = (r.stdout + r.stderr).splitlines()
        return ([l for l in out if "select_tfidf_svd.py" in l], [l for l in out if "cluster.py" in l])

    svd_on, cluster_on = lines(dict(cfg["params"]))
    final = [l for l in svd_on if "/svd/metrics.json" in l and "pass1" not in l]
    first = [l for l in svd_on if "pass1" in l]
    assert len(final) == 1 and "--unit-length" in final[0]
    assert first and all("--unit-length" in l for l in first)
    pass1_cluster = [l for l in cluster_on if "preliminary" in l]
    assert pass1_cluster and all("--ll-unit-length" in l for l in pass1_cluster)
    off = dict(cfg["params"], svd_unit_length=False)
    svd_off, cluster_off = lines(off)
    assert all("--unit-length" not in l for l in svd_off)
    assert all("--ll-unit-length" not in l for l in cluster_off)
    # a config that does not set it takes config.yaml's value (the workflow's rule for every parameter);
    # a config that must keep the Euclidean embedding sets svd_unit_length: false
    missing = {k: v for k, v in cfg["params"].items() if k != "svd_unit_length"}
    svd_missing, cluster_missing = lines(missing)
    assert any("--unit-length" in l for l in svd_missing)
    assert any("--ll-unit-length" in l for l in cluster_missing)
