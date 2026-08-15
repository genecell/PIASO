"""§2 regression: calculateCellMetrics writes `frip` for ATAC, and
filter_cells's dict-range mask accepts open-ended `(lo, None)`/`(None, hi)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest


def _atac_cytome(path, n=10):
    """Tiny ATAC cytome: per-cell in-peak fragments = (i+1)*50, total
    n_fragments = (i+1)*100, so frip == 0.5 for every cell."""
    import anndata as ad
    import cytome
    X = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        X[i, 0] = (i + 1) * 50
    a = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"n_fragments": (np.arange(n) + 1) * 100},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"chr1:{i*100}-{i*100+50}" for i in range(4)]),
    )
    ds = cytome.from_anndata(a, modality="ATAC", output=str(path))
    ds.flush()
    return ds


def test_calculateCellMetrics_writes_frip_for_atac(tmp_path):
    import piaso
    ds = _atac_cytome(tmp_path / "a.cytome")
    piaso.pp.calculateCellMetrics(ds, modality="ATAC", measurement="counts",
                                  verbose=False)
    cols = list(ds.cells.columns)
    assert "frip" in cols
    assert "n_fragments_in_peak" in cols
    frip = np.asarray(ds.cells["frip"], dtype=float)
    assert np.allclose(frip, 0.5), f"expected frip==0.5, got {frip[:5]}"
    ds.close()


def test_filter_cells_open_ended_ranges(tmp_path):
    import piaso
    ds = _atac_cytome(tmp_path / "a.cytome")
    piaso.pp.calculateCellMetrics(ds, modality="ATAC", measurement="counts",
                                  verbose=False)
    # n_fragments = 100,200,...,1000
    cases = [
        ({"n_fragments": (500, None)}, 6),   # >= 500
        ({"n_fragments": (None, 300)}, 3),   # <= 300
        ({"n_fragments": (300, 600)}, 4),    # closed range unchanged
        ({"frip": (0.4, None)}, 10),         # all frip=0.5 pass
        ({"frip": (0.6, None)}, 0),          # none pass
    ]
    for spec, expected in cases:
        m = piaso.pp.filter_cells(ds, modality="ATAC", mask=spec, inplace=False)
        assert int(m.sum()) == expected, f"{spec}: got {int(m.sum())}, want {expected}"
    ds.close()


def test_filter_cells_two_element_list_still_isin(tmp_path):
    """A 2-element LIST stays an isin (escape hatch), not a range."""
    import piaso
    ds = _atac_cytome(tmp_path / "a.cytome")
    piaso.pp.calculateCellMetrics(ds, modality="ATAC", measurement="counts",
                                  verbose=False)
    # [100, 1000] = isin → exactly the two cells with those values, not a range
    m = piaso.pp.filter_cells(ds, modality="ATAC",
                              mask={"n_fragments": [100, 1000]}, inplace=False)
    assert int(m.sum()) == 2, f"list should isin (2 cells), got {int(m.sum())}"
    ds.close()


def test_as_range_helper():
    """Unit: _as_range recognises open-ended numeric tuples, rejects lists."""
    from piaso.preprocessing._filtering import _as_range
    assert _as_range((1.0, 2.0)) == (1.0, 2.0)
    assert _as_range((1.0, None)) == (1.0, None)
    assert _as_range((None, 2.0)) == (None, 2.0)
    assert _as_range((None, None)) is None      # no constraint → not a range
    assert _as_range([1.0, 2.0]) is None        # list → isin escape hatch
    assert _as_range(("a", "b")) is None        # non-numeric → not a range
    assert _as_range((1, 2, 3)) is None         # wrong length
    assert _as_range(5) is None                 # scalar
