"""Open-a-path-or-pass-through for the spatial preprocessing helpers."""
from __future__ import annotations


def _resolve_and_open(data):
    """Return ``(dataset, opened_here)`` for an open cytome or a path."""
    if isinstance(data, str):
        import cytome
        return cytome.open(data), True
    return data, False
