"""Backward-compatibility shims for PIASO's public API.

Several top-level functions historically named their first argument ``adata`` even though they
accept an AnnData, a cytome ``Dataset``, or a path to a ``.cytome`` file. The first argument is being
standardised to ``data`` (matching ``leiden`` / ``neighbors``). To avoid breaking existing notebooks,
the old ``adata=`` keyword keeps working as a deprecated alias.
"""
import warnings

# Sentinel distinguishing "argument not supplied" from an explicit ``None``.
_UNSET = object()


def resolve_data_arg(data, func_name, **aliases):
    """Resolve the polymorphic first argument from the new ``data`` param and legacy aliases.

    Parameters
    ----------
    data : object
        Value of the new ``data`` parameter (``_UNSET`` if not supplied).
    func_name : str
        Name of the calling function, used in the deprecation/error messages.
    **aliases :
        Legacy keyword aliases (e.g. ``adata=...``, ``source=...``), each ``_UNSET`` if not
        supplied. Passing one emits a ``FutureWarning`` and is used as ``data``.

    Returns
    -------
    The resolved data object (AnnData, cytome ``Dataset``, or path string).
    """
    given = [(name, val) for name, val in aliases.items() if val is not _UNSET]
    data_given = data is not _UNSET
    if given:
        if data_given or len(given) > 1:
            names = ", ".join(["data"] * data_given + [n for n, _ in given])
            raise TypeError(
                f"{func_name}() received more than one of ({names}); pass only `data`."
            )
        name, val = given[0]
        warnings.warn(
            f"`{name}=` is deprecated in {func_name}(); use `data=` "
            f"(it accepts an AnnData, a cytome Dataset, or a path to a .cytome file).",
            FutureWarning, stacklevel=3,
        )
        return val
    if not data_given:
        raise TypeError(f"{func_name}() missing required argument: 'data'")
    return data
