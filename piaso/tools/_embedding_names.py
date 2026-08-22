"""One place that knows what an embedding is called on a cytome.

Four writers produced three conventions, and the resolver compensated by
guessing:

    runSVD(key_added='X_svd')   ->  RNA_svd
    runGDR(key_added='X_gdr')   ->  X_gdr
    umap(key_added='X_umap')    ->  X_umap
    cytome.from_h5ad conversion ->  RNA_obsm_X_umap

    # the old resolver
    for prefix in ('ATAC', 'RNA'):
        candidates.append(f'{prefix}_{suffix}')

That hardcoded list was not sloppiness: `neighbors`, `umap` and `leiden` had
no `modality` parameter, so with nothing to namespace by, guessing the two
most likely was the only option. With a modality in hand the guessing stops.

The user-facing name and the stored name are deliberately different things.
A caller writes `use_rep='X_svd'` and the same script runs on AnnData, where
that IS the key, and on a cytome, where it resolves to `{modality}_svd`.
"""
from __future__ import annotations

from typing import Iterable, Optional

__all__ = ["short_name", "storage_name", "candidate_names",
           "resolve_embedding_name", "list_embeddings"]


def short_name(name: str, modality: Optional[str] = None) -> str:
    """`X_svd` -> `svd`, `RNA_svd` -> `svd`, `svd` -> `svd`."""
    n = str(name)
    if n.startswith("X_"):
        return n[2:]
    if modality and n.startswith(f"{modality}_"):
        n = n[len(modality) + 1:]
        return n[2:] if n.startswith("X_") else n
    return n


def storage_name(name: str, modality: str) -> str:
    """The name a NEW write should use: ``{modality}_{short}``."""
    return f"{modality}_{short_name(name, modality)}"


def candidate_names(name: str, modality: str) -> list:
    """Every spelling this embedding might already be stored under.

    Ordered most-canonical first so a file holding two of them resolves
    predictably.
    """
    short = short_name(name, modality)
    out = [
        f"{modality}_{short}",            # what new PIASO writes use
        str(name),                        # exactly what the caller asked for
        f"X_{short}",                     # runGDR / umap, historically
        short,                            # bare
        f"{modality}_obsm_X_{short}",     # cytome.from_h5ad conversion
        f"{modality}_obsm_{short}",
        f"{modality}_X_{short}",
    ]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def list_embeddings(ds) -> list:
    """Embedding names actually present, for error messages."""
    try:
        return sorted(r[0] for r in
                      ds._conn.execute("SELECT array_name FROM embedding_meta"))
    except Exception:
        return []


def resolve_embedding_name(ds, name: str, modality: str,
                           present: Optional[Iterable] = None) -> str:
    """Find `name` in `ds` for `modality`, or raise saying what is there.

    The old error listed the three candidates it tried and not the file's
    actual embeddings, which is the information that would have answered the
    question.
    """
    have = set(present) if present is not None else set(list_embeddings(ds))
    for c in candidate_names(name, modality):
        if c in have:
            return c

    # Nothing under the requested modality. Before failing, look under the
    # others actually present: `neighbors`/`umap`/`leiden` default to
    # modality='RNA', and an ATAC-only cytome holding ATAC_svd would otherwise
    # go unfound by a caller that never had to think about modality before.
    # Warn, because resolving across modalities is a guess and the caller
    # should know it happened.
    import warnings as _w
    short = short_name(name, modality)
    others = sorted({n.split("_", 1)[0] for n in have if "_" in n} - {modality})
    for other in others:
        for c in candidate_names(name, other):
            if c in have:
                _w.warn(
                    f"embedding {name!r} not found for modality {modality!r}; "
                    f"using {c!r} from modality {other!r}. Pass "
                    f"modality={other!r} to make this explicit.",
                    stacklevel=3)
                return c
    del short

    raise KeyError(
        f"embedding {name!r} not found for modality {modality!r}. "
        f"Tried: {candidate_names(name, modality)}. "
        f"This cytome has: {sorted(have) or '(none)'}. "
        f"If the embedding belongs to another modality, pass modality=.")
