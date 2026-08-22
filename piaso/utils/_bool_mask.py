"""Reading boolean selection columns out of a cytome entity table."""

def as_bool_mask(values, *, name="column", source=None, warn_all_true=True):
    """Coerce a feature/cell selection column to a boolean mask.

    A cytome entity column can hold a boolean in several spellings depending on
    which writer created it and when:

        1 / 0            INTEGER  -- an int mask, or a column pre-created as INTEGER
        '1' / '0'        TEXT     -- a bool mask written by older files
        'True' / 'False' TEXT     -- a pandas object column, str(bool)
        'true' / 'false' TEXT     -- some JSON / R round-trips
        1.0 / 0.0        REAL     -- a float mask
        NULL / ''                 -- missing

    `np.asarray(col).astype(bool)` is correct for exactly the first and last of
    those and catastrophically wrong for the middle ones, because a non-empty
    string is truthy: '0' reads as True. That is how `highly_variable` silently
    became "every gene" on files where the column had been created as TEXT,
    which made runSVD compute on 32,285 genes instead of 3,000.

    Coerces rather than raising: files already written with the TEXT spelling
    -- including published ones -- must keep opening. Warns once so a user
    whose numbers move knows why.
    """
    import warnings
    import numpy as np

    arr = np.asarray(values)
    if arr.dtype == bool:
        mask = arr
    elif np.issubdtype(arr.dtype, np.number):
        mask = arr != 0
    else:
        flat = np.asarray([("" if v is None else str(v)).strip().lower() for v in arr])
        truthy = {"1", "true", "t", "yes", "y"}
        falsy = {"0", "false", "f", "no", "n", "", "none", "nan"}
        unknown = sorted(set(flat) - truthy - falsy)
        if unknown:
            # A REAL column, or a float array boxed as object, stringifies to
            # '1.0' / '0.0'. Try a numeric read of the leftovers before giving
            # up, so those are handled rather than rejected.
            try:
                numeric = np.asarray([float(v) if v not in ("", "none", "nan") else 0.0
                                      for v in flat])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name}: cannot read as a boolean mask; unrecognised values "
                    f"{unknown[:5]}. Expected 1/0, True/False or true/false."
                ) from None
            mask = numeric != 0
        else:
            mask = np.isin(flat, list(truthy))
        warnings.warn(
            f"{name}: stored as text ('1'/'0' or 'True'/'False')"
            + (f" in {source}" if source else "")
            + ". Reading it with .astype(bool) would select EVERY entry, "
              "because a non-empty string is truthy. Coerced correctly here; "
              "rewrite the column (cytome >= this version stores booleans as "
              "INTEGER) to silence this.",
            UserWarning, stacklevel=2,
        )

    if warn_all_true and mask.size and mask.all():
        warnings.warn(
            f"{name}: every one of {mask.size} entries is selected. If this is "
            f"meant to be a subset (e.g. highly-variable genes) it is the "
            f"symptom of a text-typed boolean column being read with "
            f".astype(bool).",
            UserWarning, stacklevel=2,
        )
    return mask
