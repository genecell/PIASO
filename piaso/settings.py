"""Global PIASO settings.

Provides ``set_figure_params()`` for configuring matplotlib defaults
and ``figdir`` for controlling where figures are saved.

Usage::

    import piaso

    # One-liner setup
    piaso.settings.set_figure_params(dpi=96, dpi_save=300, fontsize=12, figsize=(4, 4))

    # Set figure output directory
    piaso.settings.figdir = '/path/to/figures'

    # Now save='_leiden' saves to /path/to/figures/plotEmbedding_leiden.png
    piaso.pl.plotEmbedding(adata, color='leiden', save='_leiden')
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Module-level settings (mutable globals)
# ---------------------------------------------------------------------------

figdir: Path = Path('./')

#: Session-level override for the PIASO data root (datasets, genomes, and the
#: registry cache). Resolution order, most specific wins:
#:   1. the ``data_dir=`` argument of ``fetch_dataset`` / ``load_dataset``
#:   2. this setting
#:   3. the ``PIASO_DATA_DIR`` environment variable
#:   4. ``~/.piaso/data`` (the historical default)
#: The dataset store and the genome store share this root, so the two cannot
#: end up configured differently by accident.
data_dir = None

#: Numeric width for matrices, embeddings and graphs that PIASO **computes**
#: and writes to a cytome: INFOG, TF-IDF, log1p, gene activity, SVD/GDR/UMAP
#: embeddings, neighbour graphs.
#:
#: float32 because a normalised expression value is a ratio of int32 counts
#: and carries nothing like 15 significant digits, and because floats do not
#: compress: a float64 INFOG layer measured 6.52 GB against 1.34 GB of int32
#: counts for the same matrix, half of that being width alone. scanpy makes
#: the same choice (int counts through normalize_total + log1p come out
#: float32); Seurat, SingleCellExperiment and ArchR are float64 only because
#: R's Matrix package has no float32 sparse type.
#:
#: This governs what PIASO computes. It deliberately does NOT govern format
#: conversion: `cytome.from_h5ad` preserves the source dtype, and narrowing
#: someone's float64 matrix on import would turn a lossless conversion into a
#: lossy one.
#:
#: Accumulators are a separate matter and stay float64. Storage width and
#: summation precision are different decisions, and conflating them would
#: undo the float64 fix to `_precompute_stats`.
layer_dtype: str = 'float32'


def _resolve_layer_dtype(dtype=None) -> str:
    """Per-call override, else the package default."""
    return str(layer_dtype if dtype is None else dtype)
"""Directory for saving figures. Used when ``save`` is a suffix string
(e.g. ``'_leiden'``) or ``True``."""

file_format_figs: str = 'png'
"""Default file extension for saved figures."""

dpi_save: int = 300
"""DPI for saved figures (separate from display DPI)."""

default_cmap: str = 'Spectral_r'
"""Default colormap for continuous data."""

_frameon: bool = False
"""Whether to show axis frames by default in embedding plots."""

autoshow: bool = True
"""Whether to call ``plt.show()`` by default."""

autosave: bool = False
"""Whether to auto-save every figure (even when ``save`` is not passed)."""

style: str = None
"""Active journal style preset (set by ``set_figure_params(style=...)``), or None."""


# ---------------------------------------------------------------------------
# Journal style presets
# ---------------------------------------------------------------------------
#
# Single-column figure width + base font size + font family per journal,
# from each publisher's author guidelines (widths in mm; sizes in pt). The
# single-column width drives the default ``figsize`` (a square panel) and the
# base font size drives all derived label/tick/legend sizes. ``pdf.fonttype``
# / ``ps.fonttype`` are forced to 42 (TrueType) for every style so vector text
# stays editable — Nature/Cell/Science/etc. all require this.
#
# Widths: most journals quote 1-col ≈ 85-90 mm, 2-col ≈ 174-183 mm.
_JOURNAL_PRESETS = {
    # title                    single_mm  fontsize  font
    'nature':                 {'single_mm': 89.0, 'fontsize': 7, 'font': 'Arial'},
    'nature_methods':         {'single_mm': 89.0, 'fontsize': 7, 'font': 'Arial'},
    'cell':                   {'single_mm': 85.0, 'fontsize': 7, 'font': 'Arial'},
    'science':                {'single_mm': 55.0, 'fontsize': 7, 'font': 'Helvetica'},
    'pnas':                   {'single_mm': 87.0, 'fontsize': 7, 'font': 'Helvetica'},
    'elife':                  {'single_mm': 85.0, 'fontsize': 8, 'font': 'Arial'},
    'bioinformatics':         {'single_mm': 86.0, 'fontsize': 8, 'font': 'Helvetica'},
    'genome_biology':         {'single_mm': 85.0, 'fontsize': 8, 'font': 'Arial'},
    'nucleic_acids_research': {'single_mm': 86.0, 'fontsize': 8, 'font': 'Helvetica'},
}
"""Per-journal single-column width (mm), base font size (pt) and font family."""


# ---------------------------------------------------------------------------
# set_figure_params
# ---------------------------------------------------------------------------

def set_figure_params(
    piaso: bool = True,
    dpi: int = 96,
    dpi_save: int = 300,
    frameon: bool = False,
    fontsize: int = None,
    figsize: tuple = None,
    color_map: str = None,
    facecolor: str = 'white',
    transparent: bool = False,
    font: str = None,
    pdf_fonttype: int = 42,
    format: str = 'png',
    ipython_format: str = 'retina',
    style: str = None,
):
    """Configure global matplotlib figure parameters for PIASO plots.

    Call this once at the top of a notebook or script.  It is a
    scanpy-free replacement for ``sc.set_figure_params()``.

    Parameters
    ----------
    piaso : bool
        If True (default), apply PIASO-specific rcParams defaults
        (font, line widths, legend style, etc.).
    dpi : int
        Display resolution (affects figure size in notebooks).
    dpi_save : int
        Resolution for saved figures (typically higher for publication).
    frameon : bool
        Whether to show axis frames in embedding plots. Default False.
    fontsize : int, optional
        Base font size. Title, labels, ticks, and legend sizes are
        derived from this value. If None, falls back to the ``style`` preset's
        base size (or 12 when no style is given).
    figsize : tuple, optional
        Default figure size ``(width, height)`` in inches.  Primarily
        affects embedding plots (``plotEmbedding``, ``plotUMAP``).
        Data-driven plots (dotplot, heatmap, etc.) auto-size by default.
        If None, falls back to a square panel at the ``style`` preset's
        single-column width (or ``(4, 4)`` when no style is given).
    color_map : str, optional
        Default colormap for continuous data.  If None, keeps current.
    facecolor : str
        Figure and axes background color.
    transparent : bool
        Save figures with transparent background.
    font : str, optional
        Font family name (e.g. ``'Arial'``, ``'Helvetica'``). If None,
        falls back to the ``style`` preset's font (or ``'Arial'``).
    pdf_fonttype : int
        PDF/PS font type (42 = TrueType — embedded as editable vector text,
        required by virtually every journal). Always applied, regardless of
        ``piaso`` / ``style``.
    format : str
        Default file format for saved figures (``'png'``, ``'pdf'``, ``'svg'``).
    ipython_format : str
        Display format in Jupyter notebooks (``'retina'``, ``'png'``, ``'svg'``).
    style : str, optional
        Journal style preset. ``None`` (default) keeps the generic PIASO look.
        Otherwise one of ``'nature'`` / ``'nature_methods'``, ``'cell'``,
        ``'science'``, ``'pnas'``, ``'elife'``, ``'bioinformatics'``,
        ``'genome_biology'``, ``'nucleic_acids_research'`` — sets a square panel
        at that journal's single-column width, its base font size, and its font
        family (any of which an explicit ``figsize`` / ``fontsize`` / ``font``
        argument still overrides). See :data:`_JOURNAL_PRESETS`.

    Examples
    --------
    >>> import piaso
    >>> piaso.settings.set_figure_params(style='nature_methods')   # NM single-col
    >>> piaso.settings.set_figure_params(style='cell', fontsize=8)  # Cell, bigger font
    """
    import piaso.settings as _settings

    # --- Resolve journal style preset (explicit args always win) ---
    preset = None
    if style is not None:
        key = str(style).strip().lower().replace(' ', '_').replace('-', '_')
        if key not in _JOURNAL_PRESETS:
            raise ValueError(
                f"Unknown style {style!r}. Choose from "
                f"{sorted(_JOURNAL_PRESETS)} or None.")
        preset = _JOURNAL_PRESETS[key]
        _settings.style = key
    else:
        _settings.style = None

    if fontsize is None:
        fontsize = preset['fontsize'] if preset else 12
    if font is None:
        font = preset['font'] if preset else 'Arial'
    if figsize is None:
        if preset:
            _in = preset['single_mm'] / 25.4
            figsize = (_in, _in)
        else:
            figsize = (4, 4)

    # Store PIASO-specific settings
    _settings.dpi_save = dpi_save
    _settings._frameon = frameon
    _settings.file_format_figs = format
    if color_map is not None:
        _settings.default_cmap = color_map

    # Set IPython display format
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats(ipython_format)
    except (ImportError, ModuleNotFoundError):
        pass

    # Core rcParams
    mpl.rcParams.update({
        'figure.dpi': dpi,
        'savefig.dpi': dpi_save,
        'savefig.transparent': transparent,
        'figure.facecolor': facecolor,
        'axes.facecolor': facecolor,
        'figure.figsize': figsize,
        'pdf.fonttype': pdf_fonttype,
        'ps.fonttype': pdf_fonttype,
    })

    if color_map is not None:
        mpl.rcParams['image.cmap'] = color_map

    # PIASO-specific style defaults
    if piaso:
        _set_rcParams_piaso(fontsize=fontsize, font=font)


def _set_rcParams_piaso(fontsize: int = 12, font: str = 'Arial'):
    """Set matplotlib rcParams to PIASO defaults.

    Called by :func:`set_figure_params` when ``piaso=True``.
    """
    from .plotting import color as _color_mod

    rcParams = mpl.rcParams

    # Font
    rcParams['font.sans-serif'] = [
        font, 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif',
    ]
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = fontsize
    rcParams['legend.fontsize'] = 0.92 * fontsize
    rcParams['axes.titlesize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    rcParams['xtick.labelsize'] = fontsize
    rcParams['ytick.labelsize'] = fontsize

    # Lines and frame
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 6
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.edgecolor'] = 'black'

    # Legend
    rcParams['legend.numpoints'] = 1
    rcParams['legend.scatterpoints'] = 1
    rcParams['legend.handlelength'] = 0.5
    rcParams['legend.handletextpad'] = 0.4

    # Ticks
    rcParams['xtick.color'] = 'k'
    rcParams['ytick.color'] = 'k'

    # Grid (off by default for clean plots)
    rcParams['axes.grid'] = False

    # Color cycle
    if hasattr(_color_mod, 'd_color4'):
        from cycler import cycler
        rcParams['axes.prop_cycle'] = cycler(color=_color_mod.d_color4)


# ---------------------------------------------------------------------------
# Save helper (used by all plotting functions)
# ---------------------------------------------------------------------------

def _resolve_save_path(save, writekey: str = ''):
    """Resolve save path from user input and current settings.

    Parameters
    ----------
    save : str, bool, Path, or None
        - ``None`` or ``False``: don't save (returns None)
        - ``True``: auto-name using ``figdir / {writekey}.{format}``
        - ``str`` with ``/`` or ``\\``: treat as full path (direct save)
        - ``str`` without path separator: treat as suffix,
          saves to ``figdir / {writekey}{suffix}.{format}``
        - ``Path`` object: treat as full path

    writekey : str
        Function name used for auto-naming (e.g. ``'plotEmbedding'``).

    Returns
    -------
    Path or None
        Resolved file path, or None if save is disabled.
    """
    import piaso.settings as _settings

    if save is None or save is False:
        return None

    if save is True:
        # Auto-name: figdir / writekey.format
        _settings.figdir = Path(_settings.figdir)
        _settings.figdir.mkdir(parents=True, exist_ok=True)
        return _settings.figdir / f'{writekey}.{_settings.file_format_figs}'

    save = str(save)

    # Full path: contains path separator or is absolute
    if '/' in save or '\\' in save or Path(save).is_absolute():
        return Path(save)

    # Suffix mode: figdir / writekey{suffix}.format
    # Detect if save already has an extension
    known_exts = {'.png', '.pdf', '.svg', '.jpg', '.jpeg', '.tiff', '.eps'}
    p = Path(save)
    if p.suffix.lower() in known_exts:
        ext = p.suffix
        stem = save[:-len(ext)]
    else:
        ext = f'.{_settings.file_format_figs}'
        stem = save

    _settings.figdir = Path(_settings.figdir)
    _settings.figdir.mkdir(parents=True, exist_ok=True)
    return _settings.figdir / f'{writekey}{stem}{ext}'


def _savefig(fig, save, writekey: str = '', dpi=None):
    """Save figure using resolved path.

    Parameters
    ----------
    fig : matplotlib Figure
    save : str, bool, Path, or None
    writekey : str
        Function name for auto-naming.
    dpi : int, optional
        Override DPI. If None, uses ``settings.dpi_save``.
    """
    import piaso.settings as _settings

    path = _resolve_save_path(save, writekey)
    if path is None:
        return

    if dpi is None:
        dpi = _settings.dpi_save

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'Figure saved to: {path}')
