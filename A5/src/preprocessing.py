"""
BIOMEDIN 215: Assignment 4
preprocessing.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def split_labels_and_features(
    patient_feature_matrix: pd.DataFrame,
    column_value_map: Dict[str, Dict[str, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts non-numeric features to numeric and splits the patient_feature_matrix
    into two dataFrames: one containing the numerical features and one
    containing the labels.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified.
        - Convert the non-numeric values in the patient_feature_matrix to numeric
            values using the column_value_map. HINT: Look at pandas documentation
            to find a function that can map values in columns to new values.
        - Split the patient_feature_matrix into two dataframes: one containing
            the numerical features and one containing the labels.
            - The features matrix should contain all values from all columns in the
                patient_feature_matrix except for "death_in_stay". Additionally,
                the "subject_id" column should be set as the index of the features.
            - The labels matrix should contain only the "death_in_stay" column values.
                Additionally, the "subject_id" column should be set as the index of
                the labels.
        - Do NOT modify the input dataframe.

    Parameters:
        patient_feature_matrix (pd.DataFrame): A dataframe containing the patient
            feature matrix
        column_value_map (Dict[str, Dict[str, int]]): A dictionary mapping column
            names to a dictionary mapping column values to their corresponding
            numeric values.
                Example Structure:
                    {
                        "column_name_1": {
                            "A": 0,
                            "B": 1,
                            }
                    }
                With this mapping, the value "A" in "column_name_1" would be
                converted to 0, and the value "B" in "column_name_1" would be
                converted to 1.

    Returns:
        features (pd.DataFrame): A pd.DataFrame containing the numerical features
            with the subject_id as the index
        labels (pd.DataFrame): A pd.DataFrame containing the labels with the
            subject_id as the index
    """

    # Overwrite these return variables in your implementation
    features = None
    labels = None


    # ==================== YOUR CODE HERE ====================
    # Work on a copy to avoid mutating the input dataframe
    df = patient_feature_matrix.copy(deep=True)

    # Convert non-numeric/categorical columns using provided mappings
    for col, value_map in column_value_map.items():
        if col in df.columns:
            # replace keeps unmapped values intact (safer than map which yields NaN)
            df[col] = df[col].replace(value_map)

    # Split into features and labels and set index to subject_id
    features = df.drop(columns=["death_in_stay"]).set_index("subject_id")
    labels = df[["subject_id", "death_in_stay"]].set_index("subject_id")[["death_in_stay"]]
    # ==================== YOUR CODE HERE ====================
    

    return features, labels


def feature_variance_threshold(
    features: pd.DataFrame, freq_cut: float = 95 / 5, unique_cut: float = 0.1
) -> pd.DataFrame:
    """
    Removes features from the features dataframe that have low variance utilizing
    the following behavior:
        - If the feature has only one unique value, then the feature should always
            be REMOVED
        - If the ratio of the most common value to the second most common value
            for a particular feature column is LESS THAN freq_cut, then the feature
            should be KEPT
        - If the (number of unique values) / (total number of rows) is GREATER
            than unique_cut for a particular feature column, then the feature
            should be KEPT
        - Otherwise, the feature should be REMOVED

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified.
        - Remove features from the features dataframe that have low variance
            utilizing the behavior described above.
        - Return a version of the features dataframe with the low variance features
            removed.
        - Do NOT modify the input dataframe.


    Parameters:
        features (pd.DataFrame): A dataframe containing the numerical features
        freq_cut (float): The frequency cut-off value to use for removing features
            based on frequency of most common value to second most common value
        unique_cut (int): The unique cut-off value to use for removing features
            based on the ratio of unique values to total number of rows

    Returns:
        filtered_features (pd.DataFrame): A pd.DataFrame containing the filtered
            features
    """

    # Overwrite this return variable in your implementation
    filtered_features = None

    # Do not modify the input dataframe
    df = features.copy(deep=True)

    kept_columns = []
    total_rows = len(df)

    for column_name in df.columns:
        column_series = df[column_name]

        # Work with non-null values for uniqueness and frequency calculations
        non_null_series = column_series.dropna()

        # Rule 1: If the feature has only one unique non-null value -> remove
        unique_non_null_count = non_null_series.nunique()
        if unique_non_null_count <= 1:
            continue

        # Frequency-based rule using most common and second most common
        value_counts = non_null_series.value_counts()
        # At this point we know there are at least 2 unique values
        most_common = value_counts.iloc[0]
        second_most_common = value_counts.iloc[1]
        freq_ratio = most_common / second_most_common

        # Uniqueness ratio relative to total number of rows (including null rows)
        unique_ratio = unique_non_null_count / total_rows if total_rows > 0 else 0.0

        # Keep if either:
        # - freq ratio is less than cutoff, or
        # - unique ratio is greater than cutoff
        if (freq_ratio < freq_cut) or (unique_ratio > unique_cut):
            kept_columns.append(column_name)
        # else: remove (i.e., do not append)

    filtered_features = df[kept_columns]
    
    return filtered_features
