"""Figure life cycle for every PIASO plot: create, lay out, save, show, return.

Each plotting function used to decide these itself, and they disagreed:

* ``plt.tight_layout()`` acted on ``plt.gcf()``, the current figure, which is
  not always the figure drawn;
* six functions closed the figure when ``show=False`` -- including a figure
  the *caller* had made and passed in through ``ax=``, so a composed figure
  left pyplot and a later ``plt.savefig`` wrote a blank page;
* ``embedding`` and ``split_embedding`` with ``show=False`` kept the figure
  open and returned nothing, and several functions never closed at all;
* ``show`` defaulted to True even when drawing into someone else's axes;
* saving went through matplotlib's default Type 3 fonts unless the session
  had called ``set_figure_params``.

The contract now, in one place:

``show``
    ``None`` means ``ax is None``: a function draws into a caller's axes
    without calling ``plt.show()``.
Layout
    ``tight_layout`` runs on the figure drawn, and never on a figure that
    already has a layout engine (constrained or compressed layout, the
    caller's or the session's), which it would otherwise replace.
Saving
    Through :func:`savefig`, with TrueType (Type 42) fonts in PDF and PS, so
    the text stays text in Illustrator, as journals ask. A session that set
    ``rcParams['pdf.fonttype']`` to something other than matplotlib's default
    keeps its choice.
Return and close
    Shown: the function returns None (a returned figure would be displayed a
    second time at the end of a notebook cell). Not shown: it returns the
    figure. A figure the function made is closed only once it is saved and
    not shown -- nothing else will use it -- so ``show=False`` followed by
    ``plt.savefig()`` keeps working. A caller's figure is never closed.
"""
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["resolve_show", "begin", "layout", "savefig", "finish"]


def resolve_show(show: Optional[bool], ax=None) -> bool:
    """``show=None`` means "show unless drawing into the caller's axes"."""
    return (ax is None) if show is None else bool(show)


def begin(ax=None, **subplots_kw):
    """``(fig, ax, created)``: the caller's axes and figure, or new ones."""
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(**subplots_kw)
    return fig, ax, True


def layout(fig, *, quiet: bool = True, **kw) -> None:
    """``fig.tight_layout(**kw)``, unless ``fig`` has a layout engine.

    ``quiet`` silences tight_layout's "not compatible" warnings, which a
    figure with a colorbar or a legend panel beside the axes raises routinely
    while still laying out correctly.
    """
    try:
        engine = fig.get_layout_engine()
    except AttributeError:          # matplotlib < 3.6
        engine = None
    if engine is not None:
        return
    if quiet:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                fig.tight_layout(**kw)
            except Exception:
                pass
    else:
        fig.tight_layout(**kw)


def _font_rc():
    """Type 42 in PDF and PS unless the session chose otherwise."""
    rc = {}
    for key in ("pdf.fonttype", "ps.fonttype"):
        current = mpl.rcParams[key]
        default = mpl.rcParamsDefault[key]
        rc[key] = 42 if current == default else current
    return rc


def savefig(fig, save, writekey: str = "", dpi=None, literal: bool = False,
            **savefig_kw):
    """Save ``fig`` where ``save`` says (see ``piaso.settings``); the path or None.

    ``save``: None/False (do not save), True (``figdir/<writekey>.<format>``),
    a suffix (``figdir/<writekey><suffix>``) or a path. ``literal=True`` takes
    any string as a path, for the functions whose ``save``/``save_path`` was
    always documented as one (``'map.png'`` stays in the working directory).
    """
    from pathlib import Path
    from .. import settings as _settings
    from ..settings import _resolve_save_path

    if literal and save is not None and save is not False and save is not True:
        path = Path(save)
    else:
        path = _resolve_save_path(save, writekey)
    if path is None:
        return None
    if dpi is None:
        dpi = _settings.dpi_save
    savefig_kw.setdefault("bbox_inches", "tight")
    with mpl.rc_context(_font_rc()):
        fig.savefig(path, dpi=dpi, **savefig_kw)
    print(f"Figure saved to: {path}")
    return path


def finish(fig, *, created: bool, show: bool, save=None, writekey: str = "",
           dpi=None, savefig_kw: Optional[dict] = None, literal: bool = False):
    """Save, show and return ``fig`` under the module's contract.

    Returns ``fig`` when it is not shown, else None.
    """
    path = savefig(fig, save, writekey=writekey, dpi=dpi, literal=literal,
                   **(savefig_kw or {}))
    if show:
        plt.show()
        return None
    if created and path is not None:
        # saved and not shown: the caller holds the object if it wants it,
        # and pyplot no longer accumulates it across a batch of calls
        plt.close(fig)
    return fig
