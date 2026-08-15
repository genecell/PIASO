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

# --- piaso.data (motif DBs + .2bit sequence access; PWM stays in PIASO) ---
load_meme = _forward("cytorete.data", "load_meme", "data.load_meme")
load_jaspar_meme = _forward("cytorete.data", "load_jaspar_meme", "data.load_jaspar_meme")
load_cisbp_meme = _forward("cytorete.data", "load_cisbp_meme", "data.load_cisbp_meme")
load_cisbp = _forward("cytorete.data", "load_cisbp", "data.load_cisbp")
load_tf_list = _forward("cytorete.data", "load_tf_list", "data.load_tf_list")
fetch_jaspar = _forward("cytorete.data", "fetch_jaspar", "data.fetch_jaspar")
resolve_jaspar_path = _forward("cytorete.data", "resolve_jaspar_path", "data.resolve_jaspar_path")
fetch_cisbp = _forward("cytorete.data", "fetch_cisbp", "data.fetch_cisbp")
resolve_cisbp_meme_path = _forward("cytorete.data", "resolve_cisbp_meme_path", "data.resolve_cisbp_meme_path")
fetch_cistarget_motifs = _forward("cytorete.data", "fetch_cistarget_motifs", "data.fetch_cistarget_motifs")
load_cistarget_motifs = _forward("cytorete.data", "load_cistarget_motifs", "data.load_cistarget_motifs")
resolve_cistarget_paths = _forward("cytorete.data", "resolve_cistarget_paths", "data.resolve_cistarget_paths")
write_meme = _forward("cytorete.data", "write_meme", "data.write_meme")
fetch_animaltfdb_tf_list = _forward("cytorete.data", "fetch_animaltfdb_tf_list", "data.fetch_animaltfdb_tf_list")
build_tf_motif_map = _forward("cytorete.data", "build_tf_motif_map", "data.build_tf_motif_map")
fetch_2bit = _forward("cytorete.data", "fetch_2bit", "data.fetch_2bit")
resolve_2bit_path = _forward("cytorete.data", "resolve_2bit_path", "data.resolve_2bit_path")
extract_sequences = _forward("cytorete.data", "extract_sequences", "data.extract_sequences")
revcomp = _forward("cytorete.data", "revcomp", "data.revcomp")
