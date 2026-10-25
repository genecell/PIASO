"""piaso.pp.filter_cells(check_integrity=...) on a cytome (2026-10-24, the user's request with cytome's merge of
store-page-size-and-subset-check): SQLite's integrity check of the filtered store only when asked, in place and to a
new file; the argument is passed to cytome only when True, so the default runs on any cytome, and asked of a cytome
without it the call stops with a clear error."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

cytome = pytest.importorskip("cytome")
import cytome.utils.validation as validation_mod  # noqa: E402
import piaso  # noqa: E402
from piaso.preprocessing._filtering import _integrity_kw  # noqa: E402

needs_flag = pytest.mark.skipif(
    "check_integrity" not in __import__("inspect").signature(cytome.CytomeDataset.filter_cells).parameters,
    reason="cytome without check_integrity (older than 0.3.7)")


def _store(path, n=40, g=25):
    rng = np.random.default_rng(0)
    ds = cytome.CytomeDataset(path, mode="w")
    ds.set_entity("cells", pd.DataFrame({"barcode": [f"BC{i:03d}" for i in range(n)]}))
    ds.set_entity("genes", pd.DataFrame({"gene_id": [f"g{j}" for j in range(g)]}))
    ds.flush()
    ds.add_matrix("RNA_counts", sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32)))
    ds.flush()
    ds.close()
    return str(path)


@needs_flag
def test_check_integrity_reaches_cytome(tmp_path, monkeypatch):
    calls = []
    real = validation_mod.validate

    def spy(ds, *args, **kwargs):
        calls.append(kwargs.get("integrity", args[0] if args else True))
        return real(ds, *args, **kwargs)

    monkeypatch.setattr(validation_mod, "validate", spy)
    keep = np.ones(40, dtype=bool)
    keep[[3, 9]] = False
    for flag in (False, True):
        calls.clear()
        p = _store(tmp_path / f"in_{flag}.cytome")
        ds = cytome.open(p, mode="r+")
        n = piaso.pp.filter_cells(ds, mask=keep, check_integrity=flag, verbose=0)
        ds.close()
        assert n == 38
        q = _store(tmp_path / f"src_{flag}.cytome")
        ds = cytome.open(q, mode="r")
        out = piaso.pp.filter_cells(ds, mask=keep, inplace=False, output=tmp_path / f"out_{flag}.cytome",
                                    check_integrity=flag, verbose=0)
        out.close()
        ds.close()
        assert calls == [flag, flag], (flag, calls)


def test_check_integrity_needs_a_cytome_with_it():
    class OldDataset:                     # filter_cells as cytome 0.3.6 has it
        def filter_cells(self, mask, include_fragments=True, include_embeddings=True, copy_annotations=True,
                         keep_original=None, check_free_space=True):
            raise AssertionError("not reached")

    assert _integrity_kw(OldDataset(), False) == {}
    with pytest.raises(ValueError, match="cytome 0.3.7 or later"):
        _integrity_kw(OldDataset(), True)
