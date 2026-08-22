"""Regressions for the hand-built pages' shared CSS and their runtime data.

Three site bugs shipped together, and none of them was the kind a link check
or a build failure catches — the pages built, every URL resolved, and the
result was still wrong in the browser:

* the dark palette lost to each page's own inline ``:root`` on every page
  except index.html, which happened to declare its dark block inline too;
* the theme button drew the sun *and* the moon, because a replaced two-state
  rule set keyed off ``data-theme`` while the new one keys off
  ``data-theme-pref`` — both matched;
* the visitor map fetched a JSON file relative to its own URL that nothing
  copied there, so it rendered "Unable to Load Data".

These assert the shape that makes each impossible rather than the exact
colours, which are free to change.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
CSS = REPO / "docs" / "_static" / "piaso-dark.css"
LANDING = REPO / "web" / "scripts" / "build_landing.py"

pytestmark = pytest.mark.skipif(not CSS.exists(), reason="docs/_static not present")


def _css() -> str:
    """The stylesheet with comments stripped.

    The comments here quote selectors while explaining them, so matching
    selectors against the raw text finds prose.
    """
    return re.sub(r"/\*.*?\*/", "", CSS.read_text(), flags=re.S)


def test_dark_palette_outranks_a_pages_own_root():
    """The token block must carry an element, so it beats a later `:root`.

    `[data-theme="dark"]` and `:root` are both (0,1,0). Every hand-built page
    declares `:root { --bg-primary: #FAF9F7; ... }` in an inline <style> after
    this stylesheet, so a bare attribute selector loses the cascade and the
    page stays cream while the navbar goes dark.
    """
    # `--bg-primary\s*:` is a declaration; `var(--bg-primary)` is a use, and
    # rules that merely consume the token are not what this is about.
    blocks = re.findall(r"([^\n{}]*)\{([^}]*--bg-primary\s*:[^}]*)\}", _css())
    assert blocks, "no block declaring --bg-primary — has the palette moved?"
    for selector, _body in blocks:
        sel = selector.strip()
        assert sel.startswith("html["), (
            f"{sel!r} declares the dark palette at :root specificity; a page's "
            "own inline :root will win. Prefix it with `html`.")


def test_theme_icons_are_driven_only_by_the_preference():
    """One icon at a time means one attribute deciding it."""
    shows = re.findall(r"([^\n{}]*fa-(?:sun|moon|circle-half-stroke)[^\n{}]*)"
                       r"\{\s*display:\s*inline", _css())
    assert len(shows) == 3, f"expected three icon rules, found {len(shows)}"
    for sel in shows:
        assert "data-theme-pref" in sel, (
            f"{sel.strip()!r} shows an icon based on the resolved theme, not "
            "the stored preference — it will match alongside the pref rules "
            "and two icons will render at once.")

    prefs = sorted(re.search(r'data-theme-pref="(\w+)"', s).group(1) for s in shows)
    assert prefs == ["auto", "dark", "light"], prefs


def _landing():
    spec = importlib.util.spec_from_file_location("build_landing", LANDING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not LANDING.exists(), reason="web/ not present")
def test_every_page_relative_fetch_is_published_beside_the_page():
    """A page that fetches `x.json` needs `x.json` at its own URL.

    visitor_map.html is served at /visitor-map/, so `fetch('visitor_data.json')`
    asks for /visitor-map/visitor_data.json. The file lived in docs/ and the
    build only ever copied _static, so the map was empty in production while
    looking fine from the docs directory.
    """
    mod = _landing()
    docs = REPO / "docs"
    for src, dst in mod.PAGES.items():
        text = (docs / src).read_text(errors="replace")
        wanted = set(re.findall(r"""["']([\w.-]+\.json)["']""", text))
        if not wanted:
            continue
        url_dir = str(pathlib.PurePath(dst).parent)
        declared = set(mod.PAGE_DATA.get(url_dir, ()))
        missing = wanted - declared
        assert not missing, (
            f"{src} fetches {sorted(missing)} relative to /{url_dir}/, but "
            f"build_landing.PAGE_DATA[{url_dir!r}] does not publish it there.")
        for name in declared:
            assert (docs / name).exists(), f"docs/{name} is missing"


def test_no_absolute_user_path_reaches_the_published_pages():
    """A workstation path in a published page is a privacy leak, not a typo.

    Five paths belonging to a *colleague* (a workstation ``/dataN/<name>/...``
    mount) shipped into the built KEGG/ChEMBL page and survived every scrub
    run, because scrub_docs_paths.py's rules only covered the HMS cluster
    layout (``/dataN/hms/``). The plain workstation form matched nothing and
    RESIDUAL did not flag it. (The literal pattern is deliberately not written
    out here: publish.sh greps the shipped tree for it, and this docstring
    ships.)

    This checks the *sources* the site is built from, so it fails before a
    build rather than after one.
    """
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    leak = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"
                      r"|/n?/?data\d+/[A-Za-z0-9_.-]+/")
    # Directories whose contents are published verbatim.
    searched = [root / "docs" / "tutorials", root / "docs" / "notebooks",
                root / "web" / "src" / "content"]
    suffixes = {".md", ".mdx", ".html", ".ipynb", ".json", ".js"}

    offenders = []
    for base in searched:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in suffixes:
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            for m in leak.finditer(text):
                offenders.append(f"{p.relative_to(root)}: {m.group(0)}")
                break

    assert not offenders, (
        "absolute user paths in published sources — run "
        "`python scripts/scrub_docs_paths.py docs`:\n  "
        + "\n  ".join(sorted(offenders)[:20]))
