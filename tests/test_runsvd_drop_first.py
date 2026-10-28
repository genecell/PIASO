"""runSVD(drop_first_component=True): components 2 to n_components + 1, on every path (2026-10-22).

The embedding keeps n_components columns, but its first column is the second singular component. The
choice is recorded next to the embedding (adata.uns[key + '_params'] / the cytome embedding's
provenance), the returned S and Vt hold the same components, and the workflow does it for its final ATAC
embedding (svd_drop_first_component, default true) but not for the first pass.
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


def toy(n_cells=300, n_genes=120, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.5, 1.0, size=(1, n_genes)) * rng.gamma(2.0, 0.5, size=(n_cells, 1))
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(lam).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


def test_in_memory_keeps_components_two_onwards_and_says_so():
    a, b = toy(), toy()
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, drop_first_component=True)
    piaso.tl.runSVD(b, use_highly_variable=False, n_components=6)
    assert a.obsm["X_svd"].shape == (300, 5)
    assert np.array_equal(a.obsm["X_svd"], b.obsm["X_svd"][:, 1:])
    rec = a.uns["X_svd_params"]
    assert {k: rec[k] for k in ("first_component_dropped", "components", "n_components")} == \
        {"first_component_dropped": True, "components": [2, 6], "n_components": 5}
    assert rec["dropped_singular_value"] > 0
    piaso.tl.runSVD(a, use_highly_variable=False, n_components=5)     # a rerun without it drops the record
    assert "X_svd_params" not in a.uns and a.obsm["X_svd"].shape == (300, 5)


def test_streaming_returns_the_same_components_in_S_and_Vt():
    a, b = toy(), toy()
    e1, s1, v1 = piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, streaming=True,
                                 drop_first_component=True, method="krylov", n_iter=8)
    e2, s2, v2 = piaso.tl.runSVD(b, use_highly_variable=False, n_components=6, streaming=True,
                                 method="krylov", n_iter=8)
    assert e1.shape == (300, 5) and s1.shape == (5,) and v1.shape[0] == 5
    assert np.allclose(e1, e2[:, 1:]) and np.allclose(s1, s2[1:]) and np.allclose(v1, v2[1:])
    assert a.uns["X_svd_params"]["components"] == [2, 6]


def test_a_cytome_stores_components_two_onwards_with_their_provenance(tmp_path):
    import cytome
    a = toy()
    path = str(tmp_path / "toy.cytome")
    cytome.from_anndata(a, modality="RNA", output=path).close()
    for drop, n, key in ((True, 5, "X_svd_drop"), (False, 6, "X_svd_all")):
        e, s, v = piaso.tl.runSVD(path, modality="RNA", measurement="counts", use_highly_variable=False,
                                  n_components=n,
                                  drop_first_component=drop, return_svd=True, key_added=key,
                                  method="krylov", n_iter=8, random_state=10)
    ds = cytome.open(path, mode="r")
    try:
        drop = np.asarray(ds.embeddings["RNA_svd_drop"])
        full = np.asarray(ds.embeddings["RNA_svd_all"])
        prov = ds.provenance.get_for_object("RNA_svd_drop")
    finally:
        ds.close()
    assert drop.shape == (300, 5)
    assert np.allclose(drop, full[:, 1:], atol=1e-5)
    params = json.loads(prov[-1]["parameters"])
    assert params["first_component_dropped"] is True and params["components"] == "2 to 6"
    assert params["dropped_singular_value"] > 0


SNAKEMAKE = os.environ.get("PIASO_SNAKEMAKE") or shutil.which("snakemake")
WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"


@pytest.mark.snakemake
@pytest.mark.skipif(not SNAKEMAKE, reason="snakemake (PIASO_SNAKEMAKE or PATH) not available")
def test_the_workflow_drops_it_for_the_final_embedding_only(tmp_path):
    cfg = yaml.safe_load(open(WORKFLOW / "config.yaml"))
    assert cfg["params"]["svd_drop_first_component"] is True
    cfg["outdir"] = str(tmp_path / "out")
    cfg["datasets"] = {"e18_unsup": cfg["datasets"]["e18_unsup"]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))

    def lines(flag):
        if flag is not None:
            cfg["params"]["svd_drop_first_component"] = flag
            path.write_text(yaml.safe_dump(cfg))
        r = subprocess.run([SNAKEMAKE, "-s", str(WORKFLOW / "Snakefile"), "--configfile", str(path), "-np",
                            f"{cfg['outdir']}/e18_unsup/svd/.done"],
                           capture_output=True, text=True, cwd=str(tmp_path))
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        return [l for l in (r.stdout + r.stderr).splitlines() if "select_tfidf_svd.py" in l]

    on = lines(None)
    final = [l for l in on if "/svd/metrics.json" in l and "pass1" not in l]
    first = [l for l in on if "pass1" in l]
    assert len(final) == 1 and "--drop-first-component" in final[0]
    assert first and all("--drop-first-component" not in l for l in first)
    off = lines(False)
    assert all("--drop-first-component" not in l for l in off)
