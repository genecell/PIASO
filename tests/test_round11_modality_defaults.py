"""Round 11 (2026-05-26) regression tests for modality default fixes.

Two bugs fixed in PIASO Round 11:

A. ``runGDR`` / ``runGDRParallel`` / ``_runGDRParallel_cytome`` defaulted
   to ``modality="ATAC"`` — wrong for the common RNA-runGDR workflow.
   When users called ``piaso.tl.runGDR(rna_cytome)`` without
   specifying modality, the error message asked for ``ATAC_infog``
   instead of ``RNA_infog``. Fixed to default to ``"RNA"``.

B. ``piaso.tl.infog`` had no ``modality=`` parameter — was hardcoded
   to RNA via ``iter_chunks(modality="RNA", ...)``. The runGDR error
   hint told users to pass ``modality='RNA'`` but the function
   rejected it. Round 11 adds the kwarg (default 'RNA') and threads
   it through ``_infog_streaming`` + ``_get_infog_chunk_iterator``
   + the entity-table write (HVG mask / variance go to the modality's
   var entity, not always ``ds.genes``).
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------
# Issue A — runGDR family modality defaults
# ---------------------------------------------------------------------

def test_runGDR_default_modality_is_RNA():
    """Round 11: runGDR defaults to modality='RNA' (was 'ATAC')."""
    import piaso
    sig = inspect.signature(piaso.tl.runGDR)
    assert sig.parameters["modality"].default == "RNA", (
        f"runGDR's modality default should be 'RNA' (Round 11 change); "
        f"got {sig.parameters['modality'].default!r}"
    )


def test_runGDRParallel_default_modality_is_RNA():
    """Round 11: runGDRParallel default to modality='RNA' too."""
    import piaso
    sig = inspect.signature(piaso.tl.runGDRParallel)
    assert sig.parameters["modality"].default == "RNA"


def test_runGDR_error_hint_references_RNA_modality_by_default(tmp_path):
    """When runGDR is called on an RNA-only cytome (no RNA_infog yet)
    with default kwargs, the error message must reference 'RNA_infog'
    (not 'ATAC_infog') — that was the user-reported bug."""
    import cytome
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import piaso

    # Build an RNA-only cytome (no infog materialised)
    p = tmp_path / "rna.cytome"
    ds = cytome.create(p)
    n_obs, n_vars = 30, 10
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
        "leiden": ["a"] * n_obs,
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": np.arange(n_vars),
        "gene_id": [f"g{i}" for i in range(n_vars)],
    }))
    X = sp.csr_matrix(
        np.random.default_rng(0).standard_normal((n_obs, n_vars)).astype(np.float32)
    )
    ds.add_matrix("RNA_counts", X)
    ds.flush()
    ds.close()

    # Call runGDR with default modality. Pre-Round-11 it would have
    # complained about ATAC_infog; Round 11 must complain about RNA_infog.
    with pytest.raises(ValueError) as exc:
        piaso.tl.runGDR(str(p), groupby="leiden", n_gene=5)
    msg = str(exc.value)
    assert "RNA_infog" in msg, (
        f"runGDR(default modality) error must reference RNA_infog "
        f"(Round 11); got:\n{msg}"
    )
    assert "ATAC_infog" not in msg, (
        f"runGDR(default modality) must NOT mention ATAC_infog "
        f"(that was the pre-Round-11 bug); got:\n{msg}"
    )


# ---------------------------------------------------------------------
# Issue B — piaso.tl.infog modality kwarg
# ---------------------------------------------------------------------

def test_infog_signature_has_modality_kwarg():
    """Round 11: piaso.tl.infog gains a modality kwarg (default 'RNA')."""
    import piaso
    sig = inspect.signature(piaso.tl.infog)
    assert "modality" in sig.parameters, (
        f"piaso.tl.infog must accept a 'modality' kwarg (Round 11). "
        f"Found params: {list(sig.parameters)}"
    )
    assert sig.parameters["modality"].default == "RNA"


def test_infog_modality_kwarg_threads_through_iter_chunks(tmp_path):
    """Round 11: passing modality='GA' to infog must read from GA_counts
    (not silently from RNA_counts) and write its HVG/variance to the
    GA_genes entity table (not ds.genes).
    """
    import cytome
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import piaso

    p = tmp_path / "ga.cytome"
    ds = cytome.create(p)
    n_obs, n_vars = 40, 15
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(n_obs),
        "barcode": [f"c{i}" for i in range(n_obs)],
    }))
    ds.set_entity("GA_genes", pd.DataFrame({
        "gene_idx": np.arange(n_vars),
        "gene_id": [f"ga_g{i}" for i in range(n_vars)],
    }))
    # GA modality data (deliberately different from RNA so we can tell).
    # Integer-valued: infog rejects non-count input outright, see
    # tests/test_infog_integer_counts_guard.py.
    GA_X = sp.csr_matrix(
        np.random.default_rng(7).poisson(5.0, size=(n_obs, n_vars))
        .astype(np.float32)
    )
    ds.add_matrix("GA_counts", GA_X)
    ds.flush()
    ds.close()

    # Call infog with modality='GA' — this requires the new kwarg.
    piaso.tl.infog(
        str(p), modality="GA", n_top_genes=5, save_layer=False,
        verbosity=0,
    )

    # Verify: GA_genes table now has the HVG mask + infog_var columns.
    ds = cytome.open(p)
    ga_cols = [r[1] for r in ds._conn.execute(
        "PRAGMA table_info(GA_genes)"
    ).fetchall()]
    assert "highly_variable" in ga_cols, (
        f"infog(modality='GA') must write HVG mask to GA_genes; "
        f"got cols={ga_cols}"
    )
    assert "infog_var" in ga_cols, (
        f"infog(modality='GA') must write infog_var to GA_genes; "
        f"got cols={ga_cols}"
    )

    # And the params metadata must be namespaced by modality
    assert "GA_infog_params" in ds.metadata, (
        f"infog(modality='GA') must write metadata['GA_infog_params']; "
        f"got keys={list(ds.metadata.keys())[:20]}"
    )

    ds.close()
