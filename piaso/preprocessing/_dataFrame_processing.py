import pandas as pd

def getCrossCategories(df, col1, col2, delimiter='@', iterate_by_second_column=True):
    """
    Generates a new categorical column from the cross combinations of two specified columns in a DataFrame,
    respecting existing categorical orders if present. The iteration order of combination can be controlled,
    and a custom delimiter can be used to join the column values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to be combined.
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
    rank: bool = False,
    ascending: bool = False,
    as_dataframe: bool = False
):
    """
    Returns the counts of unique values in the given list.

    Parameters
    ----------
    values : list
        A list of values for which the counts are to be calculated.
    rank : bool, optional
        If True, the results are sorted by count. Default is False.
    ascending : bool, optional
        If True and rank is True, the results are sorted in ascending order.
        If False and rank is True, the results are sorted in descending order.
        Default is False.
    as_dataframe : bool, optional
        If True, the result is returned as a pandas DataFrame with columns 'value' and 'count'.
        If False, the result is returned as a dictionary. Default is False.

    Returns
    -------
    dict or pandas.DataFrame
        A dictionary (or DataFrame, if `as_dataframe` is True) containing the counts of unique values.
        If rank is True, the dictionary is sorted by count.
    """

    counts = dict(Counter(values))
    
    if rank:
        counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=not ascending))
            
    # Return results as a DataFrame if requested.
    if as_dataframe:
        return pd.DataFrame(list(counts.items()), columns=['Value', 'Count'])

    return counts