"""Lazy shims for the GRN API that moved to the **Cytorete** package.

The GRN method — ``inferGRN`` / ``inferRegulon`` / ``inferTFActivity``, regulon
activity/specificity, cistrome & promoter builders, motif-DB loaders, and the
regulon plots — now lives in `Cytorete <https://github.com/genecell/cytorete>`_
(``pip install cytorete``), which depends on ``piaso-tools``.

These thin forwarders keep the historical entry points (``piaso.tl.inferGRN``,
``piaso.pp.build_peak_cistrome``, ``piaso.pl.regulonActivity``,
``piaso.data.load_meme`` …) working for existing notebooks and benchmark
scripts: each resolves Cytorete at *call time* and, if it isn't installed,
raises a clear :class:`ImportError` pointing to ``pip install cytorete``.

Cytorete is deliberately **not** a declared dependency of PIASO — that would be a
packaging cycle (``piaso-tools[grn] → cytorete → piaso-tools``). This is a
zero-cost, import-time-free pointer. The recommended usage is to
``import cytorete`` directly; these shims exist only for backward compatibility.
"""
from __future__ import annotations

import importlib

_HINT = (
    " has moved to the Cytorete package — `pip install cytorete` "
    "(it depends on piaso-tools), then use it via `import cytorete` "
    "(e.g. cytorete.tl.inferGRN)."
)


def _forward(module: str, attr: str, public_name: str):
    """Build a lazy forwarder to ``<module>.<attr>`` (imported on first call)."""

    def _fwd(*args, **kwargs):
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:  # Cytorete not installed
            raise ImportError(f"`piaso.{public_name}`{_HINT}") from exc
        return getattr(mod, attr)(*args, **kwargs)

    _fwd.__name__ = attr
    _fwd.__qualname__ = attr
    _fwd.__doc__ = f"Lazy shim → ``cytorete``: ``{public_name}``.{_HINT}"
    return _fwd


# --- piaso.tl (tools) ---
inferGRN = _forward("cytorete.tools", "inferGRN", "tl.inferGRN")
inferRegulon = _forward("cytorete.tools", "inferRegulon", "tl.inferRegulon")
inferTFActivity = _forward("cytorete.tools", "inferTFActivity", "tl.inferTFActivity")
regulonActivity = _forward("cytorete.tools", "regulonActivity", "tl.regulonActivity")
regulonSpecificity = _forward("cytorete.tools", "regulonSpecificity", "tl.regulonSpecificity")

# --- piaso.pp (preprocessing) ---
build_peak_cistrome = _forward("cytorete.preprocessing", "build_peak_cistrome", "pp.build_peak_cistrome")
bulk_base_cistrome = _forward("cytorete.preprocessing", "bulk_base_cistrome", "pp.bulk_base_cistrome")
build_cistrome = _forward("cytorete.preprocessing", "build_cistrome", "pp.build_cistrome")
extract_promoter_sequences = _forward("cytorete.preprocessing", "extract_promoter_sequences", "pp.extract_promoter_sequences")

# --- piaso.pl (plotting) — note pl.regulonActivity is the PLOT (distinct from tl.regulonActivity) ---
pl_regulonActivity = _forward("cytorete.plotting", "regulonActivity", "pl.regulonActivity")
regulonNetwork = _forward("cytorete.plotting", "regulonNetwork", "pl.regulonNetwork")
regulonEmbedding = _forward("cytorete.plotting", "regulonEmbedding", "pl.regulonEmbedding")
regulonSpecificityScatter = _forward("cytorete.plotting", "regulonSpecificityScatter", "pl.regulonSpecificityScatter")

# The motif-DB loaders and .2bit sequence access are NOT forwarded here.
# They live in PIASO (piaso/data/_motifs.py, piaso/data/_fasta.py) because
# they are the inputs to piaso.pp.scan_motifs, which ships publicly --
# forwarding them to an unpublished package left the scanner with no
# supported way to obtain a sequence or a PWM.
