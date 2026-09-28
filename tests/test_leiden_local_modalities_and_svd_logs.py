"""leiden_local on every modality of a cytome, and what runSVD says about itself.

- dr_method='X_svd' (default) streams per group for ATAC and tiles through
  the peak chain, and for RNA with a materialised INFOG layer through a
  per-group runSVD on stored values; 'X_svd_full' streams with every feature.
- The per-group subset carries only the matrix being clustered.
- An AnnData without a highly_variable column uses every feature and says so.
- runSVD announces the true pass count, names the solver for every method,
  reports its thread count and the split between passes and numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("piaso._piaso")


def _two_type_counts(rng, n_per, n_feat, n_marker):
    """Two latent types, each with its own marker block, plus a block of
    ubiquitous high-count features. The ubiquitous block matters: the peak
    selection drops the features holding the top share of counts, which in
    real data are housekeeping peaks; without such a block the markers would
    be the most accessible features and be dropped instead."""
    X = (rng.random((2 * n_per, n_feat)) < 0.03).astype(np.float32) * rng.integers(1, 3, (2 * n_per, n_feat))
    X[:n_per, :n_marker] += (rng.random((n_per, n_marker)) < 0.85) * 4
    X[n_per:, n_marker:2 * n_marker] += (rng.random((n_per, n_marker)) < 0.85) * 4
    X[:, -n_marker:] += (rng.random((2 * n_per, n_marker)) < 0.98) * 9      # ubiquitous
    return X.astype(np.float32)


def _store(path, X, modality, extra_matrix=None):
    import cytome
    n, m = X.shape
    ds = cytome.create(str(path))
    ds.set_entity("cells", pd.DataFrame({"cell_idx": np.arange(n), "barcode": [f"b{i}" for i in range(n)],
                                         "coarse": ["A"] * (n // 2) + ["B"] * (n - n // 2)}))
    if modality == "ATAC":
        ds.set_entity("peaks", pd.DataFrame({"peak_idx": np.arange(m), "peak_id": [f"chr1:{i*1000}-{i*1000+500}" for i in range(m)],
                                             "chr": ["chr1"] * m, "start": np.arange(m) * 1000, "end_": np.arange(m) * 1000 + 500}))
    elif modality == "tiles":
        ds.set_entity("tiles", pd.DataFrame({"tile_idx": np.arange(m), "tile_id": [f"chr1:{i*500}-{i*500+500}" for i in range(m)],
                                             "chr": ["chr1"] * m, "start": np.arange(m) * 500, "end_": np.arange(m) * 500 + 500}))
    else:
        ds.set_entity("genes", pd.DataFrame({"gene_idx": np.arange(m), "gene_id": [f"G{i}" for i in range(m)],
                                             "symbol": [f"Gene{i}" for i in range(m)]}))
    ds.add_matrix(f"{modality}_counts", sp.csr_matrix(X))
    if extra_matrix is not None:
        name, Y, entity_cols = extra_matrix
        ds.set_entity(entity_cols[0], entity_cols[1])
        ds.add_matrix(name, sp.csr_matrix(Y))
    ds.flush()
    return ds


def _sub_purity(labels, truth):
    from sklearn.metrics import homogeneity_score
    return homogeneity_score(truth, labels)


@pytest.mark.parametrize("modality", ["ATAC", "tiles"])
def test_streaming_peak_chain_on_atac_and_tiles(tmp_path, modality):
    """Each coarse group hides two latent types; the per-group chain must
    find them on both peak-like modalities, and the default dr_method must
    take this path (no AnnData conversion, labels on the store)."""
    import piaso
    rng = np.random.default_rng(0)
    n_per, m = 60, 400
    X = np.vstack([_two_type_counts(rng, n_per, m, 40), _two_type_counts(rng, n_per, m, 40)])
    ds = _store(tmp_path / f"{modality}.cytome", X, modality)
    truth = np.array(["A1"] * n_per + ["A2"] * n_per + ["B1"] * n_per + ["B2"] * n_per)
    piaso.tl.leiden_local(ds, groupby="coarse", modality=modality, resolution=1.0,
                          n_components=10, min_cells=10, verbosity=0)
    labels = np.asarray(ds.cells["Leiden_local"]).astype(str)
    coarse = np.asarray(ds.cells["coarse"]).astype(str)
    assert all(lab.split("-")[0] == c for lab, c in zip(labels, coarse))
    assert len(set(labels[coarse == "A"])) >= 2 and len(set(labels[coarse == "B"])) >= 2
    assert _sub_purity(labels, truth) > 0.7
    ds.close()


def test_rna_cytome_with_a_materialised_infog_layer_streams_per_group(tmp_path, capsys):
    """RNA must not go through the peak chain: it runs runSVD per group on
    the stored layer, selecting by highly_variable when the store has it."""
    import piaso
    rng = np.random.default_rng(1)
    n_per, m = 60, 300
    X = np.vstack([_two_type_counts(rng, n_per, m, 30), _two_type_counts(rng, n_per, m, 30)])
    ds = _store(tmp_path / "rna.cytome", X, "RNA")
    # a "materialised infog" layer: log1p-scaled values, plus a selection column
    ds.add_matrix("RNA_infog", sp.csr_matrix(np.log1p(X)))
    hv = np.zeros(m, bool); hv[:120] = True
    ds.genes["highly_variable"] = hv; ds.flush()
    truth = np.array(["A1"] * n_per + ["A2"] * n_per + ["B1"] * n_per + ["B2"] * n_per)
    piaso.tl.leiden_local(ds, groupby="coarse", modality="RNA", cytome_layer="infog",
                          resolution=1.0, n_components=10, min_cells=10, verbosity=1)
    out = capsys.readouterr().out
    assert "120 features" in out, out          # the selection, not the peak chain
    labels = np.asarray(ds.cells["Leiden_local"]).astype(str)
    assert _sub_purity(labels, truth) > 0.7
    ds.close()


def test_x_svd_full_streams_with_every_feature_and_the_subset_carries_one_matrix(tmp_path, capsys, monkeypatch):
    """'X_svd_full' is not an alias: every feature, values as stored. And the
    per-group stores (one split of the whole store) carry the one matrix it clusters."""
    import piaso
    import cytome
    rng = np.random.default_rng(2)
    n_per, m = 60, 400
    X = np.vstack([_two_type_counts(rng, n_per, m, 40), _two_type_counts(rng, n_per, m, 40)])
    tiles = pd.DataFrame({"tile_idx": np.arange(50), "tile_id": [f"chr1:{i}-{i+1}" for i in range(50)],
                          "chr": ["chr1"] * 50, "start": np.arange(50), "end_": np.arange(50) + 1})
    Y = (rng.random((X.shape[0], 50)) < 0.2).astype(np.float32)
    ds = _store(tmp_path / "full.cytome", X, "ATAC", extra_matrix=("tiles_counts", Y, ("tiles", tiles)))
    seen = []
    real_split = cytome.CytomeDataset.split
    def spy(self, labels, outputs, matrices, **kw):
        seen.append(list(matrices))
        return real_split(self, labels, outputs, matrices, **kw)
    monkeypatch.setattr(cytome.CytomeDataset, "split", spy)
    piaso.tl.leiden_local(ds, groupby="coarse", modality="ATAC", dr_method="X_svd_full",
                          resolution=1.0, n_components=10, min_cells=10, verbosity=1)
    out = capsys.readouterr().out
    assert seen and all(x == ["ATAC_counts"] for x in seen), seen
    assert "400 features" in out, out           # every feature, no selection
    labels = np.asarray(ds.cells["Leiden_local"]).astype(str)
    assert len(set(labels)) >= 4
    ds.close()


def test_anndata_default_uses_every_feature_when_nothing_is_selected_and_says_so(capsys):
    import anndata as ad
    import piaso
    rng = np.random.default_rng(3)
    n_per, m = 60, 200
    X = np.vstack([_two_type_counts(rng, n_per, m, 30), _two_type_counts(rng, n_per, m, 30)])
    adata = ad.AnnData(sp.csr_matrix(np.log1p(X)))
    adata.obs["coarse"] = ["A"] * (2 * n_per) + ["B"] * (2 * n_per)
    piaso.tl.leiden_local(adata, groupby="coarse", resolution=1.0, n_components=10, verbosity=1)
    out = capsys.readouterr().out
    assert "using every feature" in out and out.count("using every feature") == 1
    assert "Leiden_local" in adata.obs and adata.obs["Leiden_local"].nunique() >= 4
    with pytest.raises(ValueError, match="dr_method"):
        piaso.tl.leiden_local(adata, groupby="coarse", dr_method="X_nonsense")


def test_runSVD_announces_the_true_pass_count_the_solver_and_the_split(tmp_path, capsys):
    import piaso
    rng = np.random.default_rng(4)
    X = ((rng.random((80, 240)) < 0.2) * rng.integers(1, 5, (80, 240))).astype(np.float32)
    ds = _store(tmp_path / "log.cytome", X, "ATAC")
    ds.peaks["highly_variable"] = np.ones(240, dtype=bool); ds.flush()
    common = dict(modality="ATAC", measurement="counts", n_components=3, random_state=0, verbosity=1)
    for method in ("power", "krylov"):
        piaso.tl.runSVD(ds, method=method, n_iter=3, key_added=f"X_{method}", **common)
        out = capsys.readouterr().out
        label = "power iteration" if method == "power" else "block Lanczos"
        assert f"Solver: {label} (3 rounds = 8 passes)" in out, out
        assert f"Solver done: {label}, 3 rounds, 8 passes" in out, out
        assert "SVD engine: " in out and " threads" in out
        assert "Solver time:" in out and "passes over the store" in out
    ds.close()
