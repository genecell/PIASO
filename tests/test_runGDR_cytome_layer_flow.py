"""`cytome_layer=` / `score_cytome_layer=` are gone; `layer=` / `score_layer=`
are canonical on both backends.

The cytome-specific aliases duplicated `layer=` and `score_layer=` — a layer
name means the same thing whether the data is an AnnData or a cytome — and
letting a deprecated alias override the canonical parameter was the source of
a real bug (`regulonActivity` forwarded a stale `cytome_layer='counts'` that
clobbered `layer='infog'`, surfacing as `KeyError: 'counts'`).

They were removed in 1.2.0 rather than given a deprecation cycle because they
were never present in a released version — `piaso-tools` 1.1.0 does not have
them, so no published code can be relying on them.

This file previously asserted that the aliases warned and forwarded. It now
asserts they are absent, which is the invariant worth keeping: a re-added
alias should fail here rather than quietly resurrect the override bug.
"""
from __future__ import annotations

import inspect

import pytest

import piaso


REMOVED = ("cytome_layer", "score_cytome_layer")
CANONICAL = ("layer", "score_layer")


@pytest.mark.parametrize("func", [piaso.tl.runGDR, piaso.tl.runGDRParallel])
def test_removed_aliases_are_not_in_the_signature(func):
    params = inspect.signature(func).parameters
    present = [a for a in REMOVED if a in params]
    assert not present, (
        f"{func.__name__} re-introduced {present}. Use layer= / score_layer=; "
        "a cytome-specific twin of a parameter that already applies to both "
        "backends is what caused the layer-override bug."
    )


@pytest.mark.parametrize("func", [piaso.tl.runGDR, piaso.tl.runGDRParallel])
def test_canonical_layer_params_are_present(func):
    params = inspect.signature(func).parameters
    missing = [c for c in CANONICAL if c not in params]
    assert not missing, f"{func.__name__} is missing {missing}"


def test_score_has_layer_and_not_the_alias():
    params = inspect.signature(piaso.tl.score).parameters
    assert "layer" in params
    assert "cytome_layer" not in params


@pytest.mark.parametrize("func_name,kwarg", [
    ("runGDR", "cytome_layer"),
    ("runGDR", "score_cytome_layer"),
    ("score", "cytome_layer"),
])
def test_passing_a_removed_alias_raises_typeerror(func_name, kwarg):
    """The failure mode is a plain TypeError from Python, not a silent no-op.

    Worth pinning: if one of these functions ever grows **kwargs, passing the
    old name would be silently ignored and the caller would get results
    computed on the wrong layer.
    """
    func = getattr(piaso.tl, func_name)
    with pytest.raises(TypeError, match=kwarg):
        func(None, **{kwarg: "infog"})
