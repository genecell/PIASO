"""Open-a-path-or-pass-through for the spatial preprocessing helpers.

The fork itself lives in :mod:`piaso.utils._cytome_compat`. This is the
``(dataset, opened_here)`` shape the spatial helpers want, so they close only
what they opened, expressed in terms of the shared predicate rather than a
private ``isinstance(data, str)`` -- which is what made ``pathlib.Path`` land
in the Dataset branch and fail later, somewhere else.
"""
from __future__ import annotations

from ..utils._cytome_compat import _is_path_like, open_cytome_sync


def _resolve_and_open(data):
    """Return ``(dataset, opened_here)`` for an open cytome or a path."""
    if _is_path_like(data):
        return open_cytome_sync(data), True
    return data, False
