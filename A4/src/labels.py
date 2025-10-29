"""
BIOMEDIN 215: Assignment 4
labels.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""


import pandas as pd
from datetime import timedelta


def filter_admissions(admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the admissions DataFrame to only include admissions greater than 12 hours in length.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a filtered version
            of the input DataFrame where only rows that contain admissions longer than
            12 hours are retained.
        - The input DataFrame should not be modified.

    HINTS:
        - You may need to refer to the MIMIC III documentation to understand how
            to determine the length of an admission.

    Parameters:
        admissions (pd.DataFrame): The admissions DataFrame to be filtered

    Returns:
        pd.DataFrame: The filtered admissions DataFrame
    """

    # Overwrite this variable with the return value
    filtered_admissions = None


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    

    return filtered_admissions


def merge_and_create_times(
    cohort_labels: pd.DataFrame, admissions: pd.DataFrame
) -> pd.DataFrame:
    """
    Performs a merge with the appropriate strategy on the admissions and chartevents
    DataFrames and creates two new columns in the admissions DataFrame called
    "relative_charttime" and "index_time".

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to perform a merge
            on the cohort_labels and admissions DataFrames and create two new columns
            in the resulting DataFrame called "relative_charttime" and "index_time".
        - Your implementation should call the get_relative_charttime() and
            get_index_time() functions.

    Parameters:
        admissions (pd.DataFrame): The admissions DataFrame to be modified
        chartevents (pd.DataFrame): The chartevents DataFrame to be merged

    Returns:
        pd.DataFrame: The merged DataFrame with the new columns
    """

    # Overwrite this variable with the return value
    merged_df = None


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    

    return merged_df


def get_relative_charttime(admissions: pd.DataFrame) -> None:
    """
    Performs an inplace opperation on the admission dataframe to add a column
    called "relative_charttime" that contains the number of hours between the
    admission time and the chart time. The number of hours should be represented
    as an unrounded floating point number (charttime - admittime).

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to add a column called
            "relative_charttime" to the input DataFrame that contains the number
            of hours (represented as an unrounded floating point number) between
            the admission time and the chart time.
        - The "relative_charttime" column should be added to the input DataFrame
            and nothing should be returned.

    Parameters:
        admissions (pd.DataFrame): The admissions DataFrame to be modified
    """
    # Implement your code in the space provided below.
    pass

    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    


def get_index_time(admissions: pd.DataFrame) -> None:
    """
    Performs an inplace opperation on the admission dataframe to add a column
    called "index_time" that contains a timestamp of the time 12 hours after the
    admission time.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to add a column called
            "index_time" to the input DataFrame that contains a timestamp of the
            time 12 hours after the admission time.
        - The "index_time" column should be added to the input DataFrame and
            nothing should be returned.

    Parameters:
        admissions (pd.DataFrame): The admissions DataFrame to be modified
    """
    # Implement your code in the space provided below.
    pass

    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    


def get_shock_labels(merged_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    This function determines labels for the patients admissions in the cohort
    based on the provided specifications:
        - An admission is assigned a negative label if septic shock does
            not occur at any time during the admission.
        - An admission is assigned a positive label if septic shock occurs
            fifteen hours after admission or later.
        - Admissions where the earliest time of septic shock occurs prior
            to fifteen hours after admission are removed from the study.
        - In the case that a patient has multiple admissions for which a valid
            index time and label may be assigned, we only use the latest one.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to assign labels to
            the admissions in the cohort based on the provided specifications.
        - The input DataFrame should not be modified.
        - A new DataFrame should be returned with the following columns:
            - subject_id: The patient's unique identifier
            - hadm_id: The unique identifier for the admission
            - icustay_id: The unique identifier for the ICU stay
            - admittime: The time of admission
            - dischtime: The time of discharge
            - index_time: The time 12 hours after admission
            - label: The septic shock label for the admission (True or False)
        - The returned DataFrame should only contain one row for each unique
            `subject_id` that past the inclusion criteria

    HINTS:
        - This function is a lot to implement. We highly recommend that you break
            it down into smaller pieces by implementing helper functions.

    Parameters:
        merged_cohort (pd.DataFrame): The merged cohort DataFrame

    Returns:
        pd.DataFrame: The DataFrame containing the shock labels
    """

    # Overwrite this variable with the return value
    label_df = None


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    

    return label_df



# NOTE: For any helper functions you choose to implement, please include a docstring 
#       that briefly describes the function and its parameters/returns.
#       This will help us better understand your code for awarding partial credit.

