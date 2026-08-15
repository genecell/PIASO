import pandas as pd


def _resolve_dataframe(source, required_cols=None):
    """Resolve source to a DataFrame.

    Accepts pd.DataFrame, AnnData (.obs), cytome Dataset (cells table),
    or str path to .cytome file.
    """
    if isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, str):
        # Path to cytome file
        import sqlite3
        conn = sqlite3.connect(source)
        if required_cols:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()}
            missing = [c for c in required_cols if c not in cols]
            if missing:
                conn.close()
                raise ValueError(
                    f"Column(s) {missing} not found in cytome cells table. Available: {sorted(cols)}"
                )
            select = ", ".join(f"[{c}]" for c in required_cols)
            rows = conn.execute(f"SELECT {select} FROM cells").fetchall()
            conn.close()
            df = pd.DataFrame(rows, columns=list(required_cols))
        else:
            rows = conn.execute("SELECT * FROM cells").fetchall()
            col_names = [d[0] for d in conn.description]
            conn.close()
            df = pd.DataFrame(rows, columns=col_names)
    elif hasattr(source, '_conn'):
        # Cytome Dataset object
        conn = source._conn
        if required_cols:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(cells)").fetchall()}
            missing = [c for c in required_cols if c not in cols]
            if missing:
                raise ValueError(
                    f"Column(s) {missing} not found in cytome cells table. Available: {sorted(cols)}"
                )
            select = ", ".join(f"[{c}]" for c in required_cols)
            rows = conn.execute(f"SELECT {select} FROM cells").fetchall()
            df = pd.DataFrame(rows, columns=list(required_cols))
        else:
            rows = conn.execute("SELECT * FROM cells").fetchall()
            col_names = [d[0] for d in conn.description]
            df = pd.DataFrame(rows, columns=col_names)
    elif hasattr(source, 'obs'):
        # AnnData
        df = source.obs
    else:
        raise TypeError(
            "source must be a DataFrame, AnnData, cytome Dataset, or path to .cytome file"
        )
    return df


def getCrossCategories(source, col1, col2, delimiter='@', iterate_by_second_column=True):
    """
    Generates a new categorical column from the cross combinations of two specified columns.

    Accepts pd.DataFrame, AnnData (.obs), cytome Dataset (cells table),
    or str path to .cytome file.

    Parameters
    ----------
    source : pd.DataFrame, AnnData, cytome.Dataset, or str
        The data source containing the columns to be combined.
    col1 : str
        Name of the first column to combine.
    col2 : str
        Name of the second column to combine.
    delimiter : str, optional
        Delimiter used to join the column values. Defaults to '@'.
    iterate_by_second_column : bool, optional
        If set to True, the function iterates by the values of the second column first
        when generating the combined categories. Defaults to True.

    Returns
    -------
    pd.Categorical
        A Pandas Categorical series of the combined columns with a defined order.
    """
    df = _resolve_dataframe(source, required_cols=[col1, col2])

    # Determine the order of values in col1 and col2, respecting categorical order if present
    if pd.api.types.is_categorical_dtype(df[col1]):
        col1_categories = df[col1].cat.categories
    else:
        col1_categories = sorted(df[col1].unique())

    if pd.api.types.is_categorical_dtype(df[col2]):
        col2_categories = df[col2].cat.categories
    else:
        col2_categories = sorted(df[col2].unique())

    # Decide the ordering based on the iterate_by_second_column flag
    if iterate_by_second_column:
        categories = [f"{x}{delimiter}{y}" for x in col1_categories for y in col2_categories]
    else:
        categories = [f"{x}{delimiter}{y}" for y in col2_categories for x in col1_categories]

    # Directly create the combined series without modifying the dataframe
    combined_series = df[col1].astype(str) + delimiter + df[col2].astype(str)

    # Convert to a categorical type with the defined order
    return pd.Categorical(combined_series, categories=categories, ordered=True)
    

from collections import Counter
def table(
    values,
    column: str = None,
    rank: bool = False,
    ascending: bool = False,
    as_dataframe: bool = False
):
    """
    Returns the counts of unique values in the given list or from a data source column.

    Parameters
    ----------
    values : list, AnnData, cytome.Dataset, or str
        A list of values, or a data source (AnnData, cytome Dataset, or path
        to .cytome file). When a data source is provided, ``column`` must also
        be specified.
    column : str, optional
        Column name to read from the data source. Required when ``values`` is
        an AnnData, cytome Dataset, or path.
    rank : bool, optional
        If True, the results are sorted by count. Default is False.
    ascending : bool, optional
        If True and rank is True, the results are sorted in ascending order.
        If False and rank is True, the results are sorted in descending order.
        Default is False.
    as_dataframe : bool, optional
        If True, the result is returned as a pandas DataFrame with columns 'Value' and 'Count'.
        If False, the result is returned as a dictionary. Default is False.

    Returns
    -------
    dict or pandas.DataFrame
        A dictionary (or DataFrame, if `as_dataframe` is True) containing the counts of unique values.
        If rank is True, the dictionary is sorted by count.
    """
    # If values is a data source + column, resolve to list
    if column is not None:
        df = _resolve_dataframe(values, required_cols=[column])
        values = df[column].tolist()

    counts = dict(Counter(values))

    if rank:
        counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=not ascending))

    # Return results as a DataFrame if requested.
    if as_dataframe:
        return pd.DataFrame(list(counts.items()), columns=['Value', 'Count'])

    return counts