"""Regression test for ``piaso.pp.normalize_log1p(save_layer=True)`` on a cytome.

Guards the Round-19 bug where ``_normalize_log1p_cytome`` called
``writer.write_chunk(chunk)`` without the required ``row_offset`` (and
``writer.close()`` instead of ``writer.finalize()``), so the cytome
``save_layer=True`` path crashed with::

    TypeError: ChunkedLayerWriter.write_chunk() missing 1 required positional argument: 'row_offset'

The fix passes ``int(row_indices[0])`` as the offset and calls ``finalize()``.
This test builds a small cytome, writes the normalized layer, reads it back and
checks it matches a manual library-size + log1p computation.
"""
import numpy as np
import scipy.sparse as sp
import pytest

import piaso


def _manual_log1p(counts: np.ndarray, target_sum: float) -> np.ndarray:
    depth = counts.sum(axis=1, keepdims=True).astype(np.float64)
    depth[depth == 0] = 1.0
    return np.log1p(counts / depth * target_sum)


@pytest.mark.skipif(
    pytest.importorskip("cytome", reason="cytome not installed") is None,
    reason="cytome not installed",
)
def test_normalize_log1p_cytome_save_layer(tmp_path):
    import cytome
    from anndata import AnnData

    rng = np.random.RandomState(0)
    # 250 cells × 40 genes, mixed depths incl. a zero-count cell to hit that branch.
    counts = rng.poisson(0.6, size=(250, 40)).astype(np.float32)
    counts[7] = 0  # zero-row safety
    adata = AnnData(sp.csr_matrix(counts))
    adata.obs_names = [f"cell{i}" for i in range(counts.shape[0])]
    adata.var_names = [f"g{j}" for j in range(counts.shape[1])]

    out = str(tmp_path / "tiny.cytome")
    ds = cytome.from_anndata(adata, modality="RNA", output=out)
    ds.close()

    ds = cytome.open(out)
    target_sum = 1e4
    # Use a small batch_size so multiple chunks exercise the row_offset path.
    piaso.pp.normalize_log1p(
        ds, target_sum=target_sum, key_added="log1p_batch",
        save_layer=True, batch_size=64,
    )
    ds.flush()

    # Read the written layer back, reassembling by the chunk's global rows.
    written = np.zeros_like(counts, dtype=np.float64)
    for chunk_csr, row_indices in ds.iter_chunks(
        modality="RNA", layer="log1p_batch", batch_size=64
    ):
        arr = chunk_csr.toarray() if sp.issparse(chunk_csr) else np.asarray(chunk_csr)
        written[row_indices] = arr
    ds.close()

    expected = _manual_log1p(counts.astype(np.float64), target_sum)
    np.testing.assert_allclose(written, expected, rtol=1e-5, atol=1e-5)
    # The offset bug would have mis-placed rows; assert the zero-row maps to zeros.
    assert np.allclose(written[7], 0.0)
