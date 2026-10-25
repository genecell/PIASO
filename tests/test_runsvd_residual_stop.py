"""runSVD(stop_on='residuals'): the solver stops when every computed component's relative residual
||X X^T u - s^2 u|| / s^2 is below the tolerance, instead of when the singular values stop changing.

The residuals cost no pass: block Lanczos has them from its block tridiagonal matrix (the next block's triangular
factor times the last block of each Ritz vector), power iteration from the product its next round needs. These tests
check that the residual the solver computes is the true one, that the rule stops and respects the ceiling, that the
store and the AnnData still agree, and that leiden_local and the workflow pass the rule on.
"""
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
from piaso.tools._rsvd_core import AUTO_KRYLOV_QMAX, AUTO_RESIDUAL_TOL, rsvd_core


class Dense:
    """The operator rsvd_core takes, over a dense matrix."""

    def __init__(self, X):
        self.X = X
        self.n_rows, self.n_cols = X.shape

    def xb(self, B):
        return self.X @ B

    def xty(self, Y):
        return self.X.T @ Y


def slow_matrix(n=1500, m=400, seed=0, decay=0.985):
    """A matrix whose spectrum decays slowly, so that the vectors need several rounds."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, m)))
    V, _ = np.linalg.qr(rng.standard_normal((m, m)))
    return (U * (10.0 * decay ** np.arange(m))) @ V.T


def true_residuals(X, emb, S):
    U = emb / S
    return np.linalg.norm(X @ (X.T @ U) - U * S ** 2, axis=0) / S ** 2


@pytest.mark.parametrize("method", ["krylov", "power"])
@pytest.mark.parametrize("tol", [3e-2, 1e-4])
def test_the_residual_the_solver_stops_on_is_the_true_residual(method, tol):
    X = slow_matrix()
    emb, S, Vt, info = rsvd_core(Dense(X), n_components=10, n_iter=60, random_state=0, tol=tol,
                                 method=method, stop_on="residuals")
    assert info["converged"] is True and info["stop_on"] == "residuals"
    true = true_residuals(X, emb, S)
    # the vectors returned are the Ritz vectors whose residuals were computed
    assert np.isclose(true.max(), info["criterion"], rtol=1e-6)
    assert true.max() < tol


@pytest.mark.parametrize("method", ["krylov", "power"])
def test_it_stops_before_the_ceiling_and_never_after_it(method):
    X = slow_matrix()
    _, _, _, loose = rsvd_core(Dense(X), n_components=10, n_iter=60, random_state=0, tol=3e-2,
                               method=method, stop_on="residuals")
    _, _, _, strict = rsvd_core(Dense(X), n_components=10, n_iter=60, random_state=0, tol=1e-6,
                                method=method, stop_on="residuals")
    assert loose["iterations"] < strict["iterations"] < 60
    _, _, _, capped = rsvd_core(Dense(X), n_components=10, n_iter=3, random_state=0, tol=1e-12,
                                method=method, stop_on="residuals")
    assert capped["converged"] is False and capped["iterations"] == 3
    assert capped["passes"] == 2 + 2 * 3


def test_singular_values_settle_before_the_vectors():
    # the case the rule is for: at the same tolerance the singular-value rule stops earlier, and its vectors are
    # further from converged than the tolerance says
    X = slow_matrix(decay=0.995)
    emb_v, S_v, _, values = rsvd_core(Dense(X), n_components=10, n_iter=60, random_state=0, tol=1e-3,
                                      method="krylov", stop_on="singular_values")
    _, _, _, residuals = rsvd_core(Dense(X), n_components=10, n_iter=60, random_state=0, tol=1e-3,
                                   method="krylov", stop_on="residuals")
    assert values["iterations"] < residuals["iterations"]
    assert true_residuals(X, emb_v, S_v).max() > 1e-3


def test_auto_takes_its_own_tolerance_for_the_rule():
    X = slow_matrix()
    a = rsvd_core(Dense(X), n_components=10, n_iter=None, random_state=0, method="auto", stop_on="residuals")
    b = rsvd_core(Dense(X), n_components=10, n_iter=AUTO_KRYLOV_QMAX, random_state=0, tol=AUTO_RESIDUAL_TOL,
                  method="krylov", stop_on="residuals")
    assert a[3]["iterations"] == b[3]["iterations"] and a[3]["criterion"] < AUTO_RESIDUAL_TOL
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="stop_on"):
        rsvd_core(Dense(slow_matrix(n=200, m=60)), n_components=5, n_iter=None, random_state=0, method="auto",
                  stop_on="vectors")
    a = ad.AnnData(X=sp.csr_matrix(np.abs(slow_matrix(n=200, m=60))))
    with pytest.raises(ValueError, match="stop_on"):
        piaso.tl.runSVD(a, use_highly_variable=False, n_components=5, stop_on="vectors")


def toy(n_cells=300, n_genes=120, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.5, 1.0, size=(1, n_genes)) * rng.gamma(2.0, 0.5, size=(n_cells, 1))
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(lam).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


def test_the_store_and_the_anndata_agree(tmp_path):
    import cytome
    a, b = toy(), toy()
    path = str(tmp_path / "toy.cytome")
    cytome.from_anndata(a, modality="RNA", output=path).close()
    kw = dict(use_highly_variable=False, n_components=5, drop_first_component=True, random_state=10,
              stop_on="residuals")
    piaso.tl.runSVD(path, modality="RNA", measurement="counts", key_added="X_svd_res", **kw)
    piaso.tl.runSVD(b, **kw)
    ds = cytome.open(path, mode="r")
    try:
        stored = np.asarray(ds.embeddings["RNA_svd_res"], dtype=np.float64)
    finally:
        ds.close()
    assert np.allclose(stored, b.obsm["X_svd"], atol=1e-5)          # the store keeps float32
    # the stop's guarantee holds on the embedding stored
    X = b.X.toarray().astype(np.float64)
    e, S, _ = piaso.tl.runSVD(toy(), use_highly_variable=False, n_components=5, random_state=10,
                              stop_on="residuals", return_svd=True)
    assert true_residuals(X, e, S).max() < AUTO_RESIDUAL_TOL


def test_leiden_local_passes_it_to_every_group(monkeypatch):
    from piaso.tools import _clustering
    seen = []
    real = _clustering.runSVD

    def spy(*args, **kw):
        seen.append(kw.get("stop_on"))
        return real(*args, **kw)

    monkeypatch.setattr(_clustering, "runSVD", spy)
    a = toy(n_cells=400)
    a.obs["coarse"] = np.where(np.arange(400) < 200, "A", "B")
    piaso.tl.leiden_local(a, groupby="coarse", resolution=0.5, dr_method="X_svd", key_added="ll",
                          n_components=5, svd_stop_on="residuals", verbosity=0)
    assert seen and all(v == "residuals" for v in seen)
    seen.clear()
    piaso.tl.leiden_local(a, groupby="coarse", resolution=0.5, dr_method="X_svd", key_added="ll",
                          n_components=5, svd_stop_on="singular_values", verbosity=0)
    assert seen and all(v == "singular_values" for v in seen)
    # the default since the residual rule became runSVD's
    seen.clear()
    piaso.tl.leiden_local(a, groupby="coarse", resolution=0.5, dr_method="X_svd", key_added="ll",
                          n_components=5, verbosity=0)
    assert seen and all(v == "residuals" for v in seen)


SNAKEMAKE = os.environ.get("PIASO_SNAKEMAKE") or shutil.which("snakemake")
WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"


@pytest.mark.skipif(not SNAKEMAKE, reason="snakemake (PIASO_SNAKEMAKE or PATH) not available")
def test_the_workflow_stops_every_atac_svd_on_the_residuals(tmp_path):
    cfg = yaml.safe_load(open(WORKFLOW / "config.yaml"))
    assert cfg["params"]["svd_stop_on"] == "residuals" and cfg["params"]["svd_residual_tol"] == 0.01
    cfg["outdir"] = str(tmp_path / "out")
    cfg["datasets"] = {"e18_unsup": cfg["datasets"]["e18_unsup"]}
    path = tmp_path / "config.yaml"

    def lines(params):
        c = dict(cfg)
        c["params"] = params
        path.write_text(yaml.safe_dump(c))
        r = subprocess.run([SNAKEMAKE, "-s", str(WORKFLOW / "Snakefile"), "--configfile", str(path), "-np",
                            f"{cfg['outdir']}/e18_unsup/svd/.done"],
                           capture_output=True, text=True, cwd=WORKFLOW.parent)
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        out = (r.stdout + r.stderr).splitlines()
        svd = [l for l in out if "select_tfidf_svd.py" in l]
        cluster = [l for l in out if "cluster.py" in l and "preliminary" in l]
        assert len(svd) == 2 and len(cluster) == 1          # the first pass's SVD and the final one
        return svd, cluster

    svd, cluster = lines(dict(cfg["params"]))
    assert all("--svd-stop-on residuals --svd-tol 0.01 " in l for l in svd)
    assert "--ll-svd-stop-on residuals --ll-svd-tol 0.01 " in cluster[0]
    # the singular-value rule keeps its own tolerance and the command lines of before the option
    svd, cluster = lines(dict(cfg["params"], svd_stop_on="singular_values"))
    assert all("--svd-stop-on" not in l and "--svd-tol 0.001 " in l for l in svd)
    assert "--ll-svd-stop-on" not in cluster[0] and "--ll-svd-tol 0.001 " in cluster[0]
    # null: the solver's default for the rule
    svd, cluster = lines(dict(cfg["params"], svd_residual_tol=None))
    assert all("--svd-stop-on residuals" in l and "--svd-tol" not in l for l in svd)
    assert "--ll-svd-stop-on residuals" in cluster[0] and "--ll-svd-tol" not in cluster[0]
