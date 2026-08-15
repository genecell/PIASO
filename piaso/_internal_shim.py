"""Lazy shims for functions that are not part of the public PIASO package.

PIASO ships two kinds of module. Most are public. A smaller set implements
methods that are not yet published — PICCO peak calling, cospecificity and the
specificity hotspot, ATAC gene activity, the fragment/peak processing chain,
and the genome-browser plots. Those modules are excluded from the public
distribution.

Excluding them cannot mean deleting the imports from ``tools/__init__.py`` and
friends, because then ``piaso.tl.picco`` would simply *not exist* and the error
a user sees is ``AttributeError: module 'piaso.tools' has no attribute
'picco'`` — which reads like a typo or a broken install. These forwarders keep
the names present and make them explain themselves at call time instead.

Same pattern as :mod:`piaso._grn_shim`, which does this for the GRN API that
moved to Cytorete. The difference is only where the implementation lives: GRN
moved to a separate published package, these are simply not published yet.
"""
from __future__ import annotations

import importlib

_HINT = (
    " is not available in the public PIASO distribution. It belongs to a "
    "method that has not been published yet; the module is excluded from the "
    "released package. If you have access to the full source tree, run from "
    "there instead."
)


def _forward(module: str, attr: str, public_name: str):
    """Build a lazy forwarder to ``<module>.<attr>``, resolved on first call."""

    def _fwd(*args, **kwargs):
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            raise ImportError(f"`piaso.{public_name}`{_HINT}") from exc
        return getattr(mod, attr)(*args, **kwargs)

    _fwd.__name__ = attr
    _fwd.__qualname__ = attr
    _fwd.__doc__ = f"Unavailable in the public package: ``{public_name}``.{_HINT}"
    return _fwd


def forward_many(module: str, names, prefix: str):
    """Return ``{name: forwarder}`` for every name in ``names``.

    ``module`` is resolved lazily, so a missing module costs nothing at import
    time — which is the whole point: ``import piaso`` must succeed whether or
    not the unpublished modules are present.
    """
    return {n: _forward(module, n, f"{prefix}.{n}") for n in names}
