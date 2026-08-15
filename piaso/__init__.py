"""PIASO: Precise Integrative Analysis for Single-cell Omics"""

### Setting short names
from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl
from . import settings
from . import data


def _resolve_version() -> str:
    """Version from installed metadata, falling back to pyproject.toml.

    Hard-coding it here is what let ``__version__`` say 1.1.0 while the wheel
    said 1.2.0: pyproject was bumped for the release and this line was not.
    pyproject.toml is now the single source, and neither branch below can
    disagree with it.
    """
    # A checkout is checked first, deliberately. Running PIASO from a source
    # tree via PYTHONPATH while an older wheel is also installed is the normal
    # development setup, and there importlib.metadata reports the *installed*
    # distribution -- i.e. a version that is not the code being executed. The
    # adjacent pyproject.toml always describes the source actually imported.
    import pathlib
    import re
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
        if m:
            return m.group(1)
    # Installed as a wheel: no pyproject next to the package, so metadata is
    # both correct and the only source available.
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("piaso-tools")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()
