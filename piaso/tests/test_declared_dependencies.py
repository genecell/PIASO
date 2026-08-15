"""Every module-level import must be satisfiable by a plain ``pip install``.

PIASO 1.2.0 shipped with ``from cytome.utils.modality import ...`` at module
scope in ``piaso/plotting/_plotEmbedding.py``. ``cytome`` is an *optional*
dependency, so a clean ``pip install piaso-tools`` produced a package that
could not be imported at all::

    >>> import piaso
    ModuleNotFoundError: No module named 'cytome'

Nothing caught it because every development environment has cytome installed,
and the test suite runs in one of them. The bug is only visible from the
outside, which is exactly what this file checks from the inside: walk the
package with ``ast`` and assert that no module reaches for a third-party
package at import time unless that package is a hard dependency.

The distinction that matters is *when*, not *whether*. Thirty-odd other cytome
imports live inside function bodies and are fine -- they run only on the cytome
code path, and a user who never touches a ``.cytome`` file never pays for them.
Only module scope is load-bearing, because module scope runs on ``import
piaso``.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

import pytest


PKG = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = PKG.parent / "pyproject.toml"

# Import names that differ from their distribution name.
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
    "array_api_compat": "anndata",     # re-exported by anndata
    "pkg_resources": "setuptools",
}

# Trees the audit does not apply to.
#   _picco_core -- ``publish.sh`` removes it wholesale (HELD_DIRS), so its
#     imports (click, pysam) never reach a released wheel.
#   tests       -- imported by pytest, never by ``import piaso``, so their
#     pytest import is not load-bearing at package import time.
NOT_SHIPPED = ("_picco_core", f"{pathlib.Path('piaso') / 'tests'}")


def _declared_dependencies() -> set[str]:
    """Distribution names from ``[project] dependencies``, normalised."""
    text = PYPROJECT.read_text()
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert block, "could not locate [project] dependencies in pyproject.toml"
    out = set()
    for line in block.group(1).splitlines():
        spec = line.strip().strip(",").strip('"').strip("'")
        if not spec or spec.startswith("#"):
            continue
        out.add(re.split(r"[<>=!\[; ]", spec)[0].replace("_", "-").lower())
    return out


def _shipped_modules() -> list[pathlib.Path]:
    return [
        f for f in sorted(PKG.rglob("*.py"))
        if not any(part in str(f) for part in NOT_SHIPPED)
        and ".ipynb_checkpoints" not in str(f)
    ]


def _module_level_imports(path: pathlib.Path):
    """Yield ``(module_name, lineno)`` for imports that run on ``import piaso``.

    A ``try: import x / except ImportError:`` block at module scope is not
    counted -- that is the documented way to make a dependency optional, and
    the guard means a missing package degrades instead of exploding.
    """
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Try):
            continue                                   # guarded: fine
        for n in [node]:
            if isinstance(n, ast.Import):
                for a in n.names:
                    yield a.name.split(".")[0], n.lineno
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                yield n.module.split(".")[0], n.lineno


def test_no_undeclared_module_level_imports():
    declared = _declared_dependencies()
    stdlib = set(sys.stdlib_module_names)
    offenders = []
    for f in _shipped_modules():
        for name, lineno in _module_level_imports(f):
            if name in stdlib or name == "piaso":
                continue
            dist = IMPORT_TO_DIST.get(name, name).replace("_", "-").lower()
            if dist in declared:
                continue
            offenders.append(f"{f.relative_to(PKG.parent)}:{lineno}: {name}")

    assert not offenders, (
        "These modules import a package that is not a declared dependency, at "
        "module scope, so `import piaso` fails on a clean install:\n  "
        + "\n  ".join(offenders)
        + "\n\nEither add it to [project] dependencies, or move the import "
          "inside the function that needs it (the usual fix -- an optional "
          "backend should cost nothing until it is used)."
    )


def test_cytome_is_not_imported_at_module_scope():
    """The 1.2.0 regression, pinned -- and it still matters after the change.

    cytome is now a *required* dependency, so the audit above no longer flags
    it and an eager import would no longer break a correct install. The rule
    stays anyway, for a different reason: `import piaso` should not pay to
    parse cytome on the AnnData path that never touches it, and the whole
    package is consistent about this (every cytome import is function-local).

    Without this test that consistency would erode silently the moment someone
    finds it convenient -- which is exactly how 1.2.0 shipped unimportable.
    """
    offenders = [
        f"{f.relative_to(PKG.parent)}:{lineno}"
        for f in _shipped_modules()
        for name, lineno in _module_level_imports(f)
        if name == "cytome"
    ]
    assert not offenders, (
        "cytome imported at module scope in: " + ", ".join(offenders)
        + ". Use piaso.plotting._plotEmbedding._cytome_modality() or a "
          "function-local import instead. cytome being a required dependency "
          "is not a reason to import it eagerly."
    )


def test_cytome_is_a_required_dependency():
    """Not an extra.

    As an extra it was undiscoverable -- users read the release notes, ran the
    first cytome example and got ImportError, because nobody installs extras
    they were not told about. Requiring it costs one 124 KB pure-Python wheel
    whose own dependencies PIASO already has.
    """
    assert "cytome" in _declared_dependencies(), (
        "cytome dropped out of [project] dependencies. If it went back to "
        "being an extra, README and the release notes must say so loudly."
    )


def test_piaso_uses_cytomes_public_api_not_its_internals():
    """Depend on the surface cytome promises to keep.

    ``cytome.utils.modality`` is internal; cytome makes no compatibility
    promise about it, and PIASO reading MODALITY_REGISTRY from there meant a
    reshaping in cytome would break PIASO silently. The top-level names are
    covered by cytome's own tests/test_public_api_contract.py.
    """
    # Import statements only -- prose that *mentions* the old path (explaining
    # why we no longer use it) is not a coupling.
    offenders = []
    for f in _shipped_modules():
        for node in ast.walk(ast.parse(f.read_text())):
            mod = None
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = node.names[0].name
            if mod and mod.startswith("cytome.utils"):
                offenders.append(f"{f.relative_to(PKG.parent)}:{node.lineno}")
    assert not offenders, (
        "PIASO reaches into cytome internals at: " + ", ".join(offenders)
        + ". Import the promoted names from `cytome` directly instead."
    )


def test_missing_cytome_raises_an_actionable_error():
    """A broken install explains itself instead of NameError-ing.

    cytome is required now, so this should be unreachable on a correct install
    -- but a partially-installed environment is exactly when a clear message is
    worth the most.
    """
    from piaso.plotting import _plotEmbedding as pe

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name == "cytome" or name.startswith("cytome."):
            raise ImportError("No module named 'cytome'")
        return real_import(name, *args, **kwargs)

    import builtins
    builtins.__import__ = blocked
    try:
        with pytest.raises(ImportError, match="cytome"):
            pe._cytome_modality()
        with pytest.raises(ImportError, match="cytome"):
            pe._plot_modality_registry()
    finally:
        builtins.__import__ = real_import


def test_plot_modality_registry_still_returns_3_tuples():
    """The plot side unpacks 3-tuples; cytome's canonical registry has 4.

    The projection moved into a function when the import went lazy -- this
    pins that the shape did not change with it.
    """
    pytest.importorskip("cytome")
    from piaso.plotting._plotEmbedding import _plot_modality_registry

    registry = _plot_modality_registry()
    assert registry, "registry is empty"
    assert all(len(entry) == 3 for entry in registry)
    assert registry[0][0] == "RNA", "RNA must stay first for auto-detect order"


def test_version_matches_pyproject():
    """``piaso.__version__`` and the packaged version cannot disagree.

    1.2.0 shipped with ``__version__ = "1.1.0"`` hard-coded in
    ``piaso/__init__.py`` -- pyproject was bumped for the release and this was
    not, so `pip show` and `piaso.__version__` reported different versions and
    a bug report would name the wrong one.
    """
    import piaso

    text = PYPROJECT.read_text()
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert declared, "no version in pyproject.toml"
    assert piaso.__version__ == declared.group(1), (
        f"piaso.__version__ is {piaso.__version__!r} but pyproject.toml says "
        f"{declared.group(1)!r}. __version__ is derived from packaging "
        "metadata; a mismatch means a stale installed distribution is "
        "shadowing this checkout."
    )
