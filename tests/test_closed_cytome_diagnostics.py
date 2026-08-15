"""Closed-cytome diagnostics: PIASO functions accepting a cytome Dataset
must surface the closed-state with an actionable RuntimeError, NOT a
cryptic ``sqlite3.ProgrammingError: Cannot operate on a closed database``.

The hook lives in ``piaso.utils._cytome_compat._assert_cytome_open`` and
fires from the public predicates ``is_cytome_input`` and
``_is_cytome_dataset_obj``, which every PIASO function calls when
detecting cytome inputs. Closed cytomes are NOT auto-reopened — the
philosophy mirrors the WAL-on-NFS diagnostic: surface user intent
violations loudly, do not paper over them.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_minimal_cytome(path):
    """Build a tiny cytome with cells + RNA matrix sufficient for the predicate
    tests below."""
    import cytome
    ds = cytome.create(path)
    ds.set_entity("cells", pd.DataFrame({
        "cell_idx": np.arange(5),
        "barcode": [f"AAA-{i}" for i in range(5)],
    }))
    ds.set_entity("genes", pd.DataFrame({
        "gene_idx": [0, 1, 2],
        "gene_id": ["GeneA", "GeneB", "GeneC"],
    }))
    ds.add_matrix(
        "RNA_counts",
        sp.csr_matrix(np.eye(5, 3, dtype=np.float32)),
    )
    ds.flush()
    return ds


def test_is_closed_property_open_then_closed(tmp_path):
    """``ds.is_closed`` flips from False to True after ``ds.close()``."""
    import cytome
    ds = _build_minimal_cytome(tmp_path / "tiny.cytome")
    assert ds.is_closed is False
    ds.close()
    assert ds.is_closed is True


def test_check_open_raises_actionable_error_after_close(tmp_path):
    """``ds._check_open()`` raises ``RuntimeError`` whose message contains
    both fix paths (re-open + pass path string)."""
    import cytome
    ds = _build_minimal_cytome(tmp_path / "tiny.cytome")
    ds.close()

    with pytest.raises(RuntimeError) as exc_info:
        ds._check_open()
    msg = str(exc_info.value)
    assert "closed" in msg.lower()
    assert "cytome.open" in msg, "Re-open hint must be present"
    assert "path string" in msg, "Path-string hint must be present"
    assert str(ds.path) in msg, "Error must name the offending path"


def test_is_cytome_input_raises_on_closed(tmp_path):
    """The piaso public predicate ``is_cytome_input`` raises immediately
    when handed a closed Dataset."""
    import cytome
    from piaso.utils._cytome_compat import is_cytome_input

    ds = _build_minimal_cytome(tmp_path / "tiny.cytome")
    assert is_cytome_input(ds) is True  # open: returns True
    ds.close()

    with pytest.raises(RuntimeError, match="closed"):
        is_cytome_input(ds)


def test_is_cytome_dataset_obj_raises_on_closed(tmp_path):
    """``_is_cytome_dataset_obj`` is the predicate most tools branch on.
    Closed Dataset → actionable RuntimeError, not a silent True."""
    import cytome
    from piaso.utils._cytome_compat import _is_cytome_dataset_obj

    ds = _build_minimal_cytome(tmp_path / "tiny.cytome")
    assert _is_cytome_dataset_obj(ds) is True
    ds.close()

    with pytest.raises(RuntimeError) as exc_info:
        _is_cytome_dataset_obj(ds)
    assert str(ds.path) in str(exc_info.value)


def test_string_paths_unaffected(tmp_path):
    """Path strings are not probed (no I/O). The closed-check only triggers
    when the input is an actual Dataset object."""
    from piaso.utils._cytome_compat import is_cytome_input, _is_cytome_dataset_obj
    p = str(tmp_path / "no_such.cytome")
    assert is_cytome_input(p) is True       # path-shaped
    assert _is_cytome_dataset_obj(p) is False  # not a Dataset object
    # Neither call probes the filesystem; absence is fine.


def test_non_cytome_objects_unaffected(tmp_path):
    """Random objects must not trigger any closed-check magic."""
    from piaso.utils._cytome_compat import is_cytome_input, _is_cytome_dataset_obj
    class NotACytome:
        pass
    obj = NotACytome()
    assert is_cytome_input(obj) is False
    assert _is_cytome_dataset_obj(obj) is False


def test_actionable_error_via_piaso_score_path(tmp_path):
    """End-to-end: a representative PIASO function (`piaso.tl.score`) called
    with a closed cytome must raise the actionable RuntimeError. We don't
    care about the function's downstream behaviour — only that the
    closed-state surfaces at entry."""
    import cytome
    import piaso

    ds = _build_minimal_cytome(tmp_path / "tiny.cytome")
    ds.close()

    # piaso.tl.score reaches `_is_cytome_dataset(source)` early. With the
    # liveness-aware predicate, the closed state must surface as
    # RuntimeError with the actionable message, not as
    # sqlite3.ProgrammingError("Cannot operate on a closed database.").
    with pytest.raises(RuntimeError) as exc_info:
        piaso.tl.score(
            ds,
            gene_list=["GeneA"],
            key_added="x",
        )
    assert "closed" in str(exc_info.value).lower()
    assert "cytome.open" in str(exc_info.value)
