"""
BIOMEDIN 215: Assignment 3
features.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""

# Imports - Do not modify
import pandas as pd
from typing import List, Optional


def summarize_by_mean(
    df: pd.DataFrame,
    columns_to_group_by: Optional[List[str]] = None,
    column_to_summarize: str = "valuenum",
):
    """
    Returns version of the input DataFrame where the values in the column titled
    <column_to_summarize> are replaced by the mean value of the column for all
    rows that share the same value in the columns given by <columns_to_group_by>.

    EXAMPLE: If <column_to_summarize> is "valuenum", and <columns_to_group_by> is
    ["subject_id", "hadm_id", "icustay_id", "charttime"], and the input DataFrame
    contains the columns "subject_id", "hadm_id", "icustay_id", "charttime", and
    "valuenum", then the returned DataFrame should contain the mean value of the
    "valuenum" column for all rows that share the same value in the columns
    "subject_id", "hadm_id", "icustay_id", "charttime".

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a version of the input
            DataFrame where the values in the column titled <column_to_summarize> are
            replaced by the mean value of the column for all rows of the same measurement
            for the same timestamp for the same subject.
        - If <columns_to_group_by> is [] or None, group by all columns except
            <column_to_summarize>.
        - If <column_to_summarize> is not in the DataFrame, raise a ValueError.
        - Do NOT modify the original DataFrame.
        - No other error checking is required.

    HINTS:
        - Your implementation should use the Pandas groupby() function and the
            Pandas mean() function.
        - It's good practice to reset the index of the DataFrame after grouping.
            You can do this using the Pandas reset_index() function right before
            returning the DataFrame. If you want to learn more about why this is
            good practice, see the following link:
            https://www.dataquest.io/blog/tutorial-reset-index-in-pandas/

    Parameters:
        df (pd.DataFrame): The DataFrame to be summarized
        columns_to_group_by (List[str]): The columns to group by
        column_to_summarize (str): The column whose values should be summarized
            as the mean

    Returns:
        summarized_df (pd.DataFrame): A version of the input DataFrame where the values
            in the column titled <column_to_summarize> are replaced by the mean value of
            the column for all rows of the same measurement for all rows that share the
            same value in the columns given by <columns_to_group_by>.
    """

    # Set default value for columns_to_group_by
    if columns_to_group_by is None:
        columns_to_group_by = []

    # Overwrite this variable with the return value
    summarized_df = None

    # ==================== YOUR CODE HERE ====================
    
    if column_to_summarize not in df.columns:
        raise ValueError(f"Column {column_to_summarize} not found in DataFrame")

    # If columns_to_group_by is empty, set it to all columns except column_to_summarize
    if columns_to_group_by == []:
        columns_to_group_by = df.columns.tolist()

    summarized_df = df.groupby(columns_to_group_by).mean()[column_to_summarize]

    summarized_df = summarized_df.reset_index() # Reset the index to get a single level of column names

    # ==================== YOUR CODE HERE ====================
    

    # Return the DataFrame with the summarized column
    return summarized_df


def pivot_wide(
    df: pd.DataFrame,
    index_columns: Optional[List[str]] = None,
    columns: str = "vital_id",
    values: str = "valuenum",
):
    """
    Returns a wide version of the input DataFrame where the values in the column titled
    <values> are pivoted to columns with the column titles given in <columns>.

    EXAMPLE: If columns = "vital_id" and values = "valuenum", then the returned
    DataFrame should contain a column for each unique value in the "vital_id" column
    of the input DataFrame. Each of these new columns will contain the corresponding
    values from the "valuenum" column of the input DataFrame. Note that there will
    be many columns that contain NaN values since not all patients have all vital
    measurements at all timestamps.

    In other words, create a new column for each unique measurement type in your
    labs and vitals data frames where the rows are given by unique combinations
    of `subject_id`, `hadm_id`, `icustay_id`, `charttime`, `vital_id`.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a wide version of the
            input DataFrame where the values in the column titled <values> are pivoted
            to columns with the column titles given in <columns>.
        - If needed, use reset_index() to flatten your DataFrame if it is turned
          into a multi-index Dataframe. The returned DataFrame should be single level.
            This means that the column names should be a single level of the form
                <column1>_<column2>...
        - If a measurement is not available in a particular row of the resulting
            dataframe, the value should be NaN. (NaN is a special value in Python
            that represents an empty value aka "Not a Number" in numeric fields.)
        - Do NOT modify the original DataFrame.
        - No other error checking is required.

    HINTS:
        - Your implementation should utilize the Pandas pivot_table() function.

    Parameters:
        df (pd.DataFrame): The DataFrame to be pivoted
        index_columns (list): The columns to use as indices
        columns (str): The column whose values should be used as columns
        values (str): The column whose values should be used as values

    Returns:
        wide_df (pd.DataFrame): A wide version of the input DataFrame where the values
    """

    # Overwrite this variable with the return value
    wide_df = None

    # ==================== YOUR CODE HERE ====================
    
    if index_columns is None:
        index_columns = df.columns.tolist()

    if columns not in df.columns:
        raise ValueError(f"Column {columns} not found in DataFrame")

    if values not in df.columns:
        raise ValueError(f"Column {values} not found in DataFrame")

    wide_df = df.pivot_table(index=index_columns, columns=columns, values=values, aggfunc='mean')
    wide_df = wide_df.reset_index() # Reset the index to get a single level of column names

    # ==================== YOUR CODE HERE ====================
    

    # Return the wide DataFrame
    return wide_df


def merge_dataframes(dataframe_A: pd.DataFrame, dataframe_B: pd.DataFrame):
    """
    Merges two dataframes by `subject_id`, `hadm_id`, `icustay_id`, `charttime`.

    EXAMPLE: If dataframe_A contains the columns `subject_id`, `hadm_id`, `icustay_id`,
    `charttime`, `value_A` and dataframe_B contains the columns `subject_id`, `hadm_id`,
    `icustay_id`, `charttime`, `value_B`, then the returned DataFrame should contain
    the columns `subject_id`, `hadm_id`, `icustay_id`, `charttime`, `value_A`, `value_B`.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a merged version of
            the input DataFrames.
        - Do NOT modify the original DataFrames.
        - No other error checking is required.

    HINTS:
        - Your implementation should use the Pandas merge() function.
            - You will need to select the appropriate merge strategy such that
                all rows from both tables are included in the result, and if a
                matching value is not found in either table for a particular key,
                the result will include NaN (Not a Number).

    Parameters:
        dataframe_A (pd.DataFrame): The first DataFrame to be merged
        dataframe_B (pd.DataFrame): The second DataFrame to be merged

    Returns:
        merged_df (pd.DataFrame): A merged version of the input DataFrames
    """

    # Overwrite this variable with the return value
    merged_df = None

    # ==================== YOUR CODE HERE ====================
    merged_df = pd.merge(
        dataframe_A,
        dataframe_B,
        on=["subject_id", "hadm_id", "icustay_id", "charttime"],
        how="outer",
    )
    # ==================== YOUR CODE HERE ====================
    

    # Return the merged DataFrame
    return merged_df


def impute_missing(dataframe: pd.DataFrame):
    """
    Imputes missing values in the input DataFrame using a last-value-carried-forward
    strategy.

    EXAMPLE: If a patient has a missing value for a particular measurement at a
    particular timestamp, the value of the last measurement of that type for that
    patient should be used for the missing value.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a version of the
            input DataFrame where the missing values have been imputed using the
            last-value-carried-forward strategy.
        - Do NOT modify the original DataFrame.
        - No other error checking is required.

    HINTS:
        - Your implementation should use:
            - Pandas fillna() function
            - Pandas groupby() function
            - Pandas sort_values() function

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to be imputed

    Returns:
        imputed_df (pd.DataFrame): A version of the input DataFrame where the missing
            values have been imputed using the last-value-carried-forward strategy
    """

    # Overwrite this variable with the return value
    imputed_df = None

    # ==================== YOUR CODE HERE ====================
    
    imputed_df = dataframe.copy()
    imputed_df = imputed_df.sort_values(["subject_id", "hadm_id", "icustay_id", "charttime"])
    group_cols = ["subject_id", "hadm_id", "icustay_id"]
    value_cols = [c for c in imputed_df.columns if c not in group_cols + ["charttime"]]
    imputed_df[value_cols] = imputed_df.groupby(group_cols, sort=False)[value_cols].transform(
        lambda s: s.fillna(method="ffill")
    )
    
    # ==================== YOUR CODE HERE ====================
    

    # Return the imputed DataFrame
    return imputed_df
