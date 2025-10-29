"""
BIOMEDIN 215: Assignment 4
utils.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""


import pandas as pd
from typing import Union, List


def preprocess_dates(
    df: pd.DataFrame,
    date_columns: List[str],
    date_formats: List[str],
    inplace: bool = True,
) -> Union[pd.DataFrame, None]:
    """
    Converts the columns in the DataFrame to datetime objects.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to convert the columns in the
            DataFrame to datetime objects.
        - As an in-place opperation, the "admittime" and "dischtime" columns should
            be modified in the input DataFrame and nothing should be returned.
        - As a non in-place operation, the input DataFrame should not be modified
            and a new DataFrame should be returned with the appropriate columns
            converted to datetime objects.
        - All converted `datetime` objects should be timezone-aware and in UTC.

    HINT: Review the Pandas documentation to find a function that can be used to
        convert a column to datetime objects.

    Parameters:
        df (pd.DataFrame): The DataFrame
        date_columns (List[str]): The columns to convert to datetime objects
        date_formats (List[str]): The formats of the dates in each date_column column
            NOTE: date_columns and date_formats should be utilized such that the
            columns and formats in each list correspond 1-to-1, in order.
        inplace (bool): Whether or not to perform the operation in-place
    """
    
    # Validate inputs
    if len(date_columns) != len(date_formats):
        raise ValueError("date_columns and date_formats must have the same length")

    # Work in-place or on a copy
    target_df = df if inplace else df.copy()

    # Convert specified columns to timezone-aware UTC datetimes
    for col, fmt in zip(date_columns, date_formats):
        target_df[col] = pd.to_datetime(target_df[col], format=fmt, utc=True)

    # Return only when not operating in-place
    if not inplace:
        return target_df

    


def join_and_clean_data(diagnosis_features, note_concept_features, heart_rate_features):
    """
    Returns a joined dataframe of all features and cleans the data.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to `outer` join the three
            feature dataframes and clean the data.
        - The returned dataframe should not contain any NaN values.
            -  Replace all NaN values in the `latest_heart_rate` and `time_wt_avg`
                columns with their respective means.
            - Replace all other NaN values with 0.
        - The returned dataframe should be sorted by `subject_id` in ascending order.

    Parameters:
        diagnosis_features (pd.DataFrame): The patient diagnosis features dataframe.
        note_concept_features (pd.DataFrame): The patient note concept features
            dataframe.
        heart_rate_features (pd.DataFrame): The patient heart rate features dataframe.

    Returns:
        X (pd.DataFrame): The joined and cleaned dataframe of all features.
    """

    # Overwrite this variable with the return value for your implementation
    X = None


    # ==================== YOUR CODE HERE ====================
    # Outer-join all feature matrices on subject_id
    X = pd.merge(diagnosis_features, note_concept_features, on="subject_id", how="outer")
    X = pd.merge(X, heart_rate_features, on="subject_id", how="outer")

    # Impute heart rate feature columns with their column means
    for col in ["latest_heart_rate", "time_wt_avg"]:
        if col in X.columns:
            X[col] = X[col].astype(float)
            X[col] = X[col].fillna(X[col].mean())

    # Replace all other NaNs with 0
    X = X.fillna(0)

    # Sort by subject_id ascending
    if "subject_id" in X.columns:
        X = X.sort_values("subject_id").reset_index(drop=True)
    # ==================== YOUR CODE HERE ====================
    
    
    # Return the dataframe
    return X
