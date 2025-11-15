"""
data.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""

import pandas as pd
from typing import Tuple
import numpy as np


def split_data(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    test_fraction: float,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing sets.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - Set the random seed to the random_state parameter passed into this function.
        - Generate a permutation of indices for the data using np.random.permutation
        - Calculate num_test or the number of samples to include in the test set using
            the test_fraction parameter multiplied by the number of samples in the
            dataset. (rounded to the nearest integer)
        - Split the permutation of indicies into two lists:
            - test_indices: Contains the first num_test permutation indices
            - train_indices: Contains the remaining permutation indices
        - Use the test_indices to create the testing dataframes:
            - X_test: Contains the testing features
            - Y_test: Contains the testing labels
        - Use the train_indices to create the training dataframes:
            - X_train: Contains the training features
            - Y_train: Contains the training labels
        - Do NOT modify the input dataframes.

    Parameters:
        features (pd.DataFrame): dataframe containing the features for each sample
        labels (pd.DataFrame): dataframe containing the labels for each sample
        test_fraction (float): fraction of the total data to use for testing
        random_state (int): random state to use

    Returns:
        pd.DataFrame: dataframe containing the training features    (X_train)
        pd.DataFrame: dataframe containing the testing features     (X_test)
        pd.DataFrame: dataframe containing the training labels      (Y_train)
        pd.DataFrame: dataframe containing the testing labels       (Y_test)
    """

    # Overwrite these variables in your implementation
    X_train, X_test, Y_train, Y_test = None, None, None, None


    # ==================== YOUR CODE HERE ====================
    # Set the random seed
    np.random.seed(random_state)

    # Generate a permutation of indices for the dataset
    num_samples = features.shape[0]
    permuted_indices = np.random.permutation(num_samples)

    # Determine number of test samples (rounded to nearest integer)
    num_test = int(np.round(test_fraction * num_samples))

    # Split indices into test and train
    test_indices = permuted_indices[:num_test]
    train_indices = permuted_indices[num_test:]

    # Create train/test splits without modifying input dataframes
    X_test = features.iloc[test_indices].copy()
    Y_test = labels.iloc[test_indices].copy()
    X_train = features.iloc[train_indices].copy()
    Y_train = labels.iloc[train_indices].copy()
    # ==================== YOUR CODE HERE ====================
    

    return X_train, X_test, Y_train, Y_test
