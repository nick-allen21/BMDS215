"""
trewscore.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the existing functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""

# Imports - Do not modify
import pandas as pd


def summarize_sepsis(dev_sirs: pd.DataFrame, all_infections: pd.DataFrame):
    """
    Returns a merged dataframe containing all of the columns from the <dev_sirs> and
    <all_infections> DataFrames, as well as a new column called "sepsis_status" that
    indicates whether or not the subject met the TREWScore Sepsis definition at a given
    timestamp.

    EXAMPLE: The returned DataFrame should contain all of the columns from the
    <dev_sirs> and <all_infections> DataFrames, as well as a new column called
    "sepsis_status" that indicates whether or not the subject met the TREWScore
    Sepsis definition at a given charttime.

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a merged dataframe
            containing all of the columns from the <dev_sirs> and <all_infections>
            DataFrames, as well as a new column called "sepsis_status" that indicates
            whether or not the subject met the TREWScore Sepsis definition at a given
            charttime.
            - NOTE: You will need to choose the appropriate join keys for the merge,
                and the appropriate join strategy.
        - Your implementation should use the get_sepsis_status() function.
        - No other error checking is required.

    HINTS:
        - You may find the following functions useful:
            - Pandas: merge()

    Parameters:
        dev_sirs (pd.DataFrame): A DataFrame containing the SIRS criteria for each
            subject at each charttime.
        all_infections (pd.DataFrame): A DataFrame containing the ICD-9 and Note
            infection labels for each subject at each charttime.

    Returns:
        sepsis_summary (pd.DataFrame): A merged dataframe containing all of the columns
            from the <dev_sirs> and <all_infections> DataFrames, as well as a new column
            called "sepsis_status" that indicates whether or not the subject met the
            TREWScore Sepsis definition at a given charttime.
    """

    # Overwrite this variable with the return value
    sepsis_summary = None


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    

    return sepsis_summary


def get_sepsis_status(sepsis_summary: pd.DataFrame) -> None:
    """
    Creates a new column called "sepsis_status" in the <sepsis_summary> DataFrame that
    indicates whether or not the subject met the TREWScore Sepsis definition at a given
    charttime.

    #### TREWScore Sepsis definition:
    At a given charttime, a subject is considered to have sepsis if they:
    - Meet two or more SIRS criteria
    - Have a suspected infection (either from ICD-9 codes OR from notes)

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to perform an inplace
            modification of the <sepsis_summary> DataFrame to add a new column called
            "sepsis_status" that indicates whether or not the subject met the TREWScore
            Sepsis definition at a given charttime.
        - No other error checking is required.

    HINTS:
        - Your implementation should not utilize any loops, and take advantage of
            Pandas' vectorized operations. If you are unfamiliar with vectorization,
            check out the pandas_practice.ipynb for examples.

    Parameters:
        sepsis_summary (pd.DataFrame): A DataFrame containing the SIRS criteria and
            infection labels for each subject at each charttime.
    """


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    


def summarize_severe_sepsis(dev_sepsis: pd.DataFrame, organ_dys: pd.DataFrame):
    """
    Returns a merged dataframe containing all of the columns from the <dev_sepsis> and
    <organ_dys> DataFrames, as well as a new column called "severe_sepsis_status" that
    indicates whether or not the subject met the TREWScore Sepsis definition at a given
    timestamp.

    EXAMPLE: The returned DataFrame should contain all of the columns from the
    <dev_sepsis> and <organ_dys> DataFrames, as well as a new column called
    "severe_sepsis_status" that indicates whether or not the subject met the TREWScore
    Severe Sepsis definition at a given charttime.

    #### TREWScore Severe Sepsis definition:
    At a given charttime, a subject is considered to have severe sepsis if they:
    - Have sepsis (as defined by the TREWScore Sepsis definition)
    - Have organ dysfunction

    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a merged dataframe
            containing all of the columns from the <dev_sepsis> and <organ_dys>
            DataFrames, as well as a new column called "severe_sepsis_status" that
            indicates whether or not the subject met the TREWScore Severe Sepsis
            definition at a given charttime.
            - NOTE: You will need to choose the appropriate join keys for the merge,
                and the appropriate join strategy.
        - No other error checking is required.

    HINTS:
        - You may find the following functions useful:
            - Pandas: merge()

    Parameters:
        dev_sepsis (pd.DataFrame): A DataFrame containing the sepsis_status for each
            subject at each charttime.
        all_infections (pd.DataFrame): A DataFrame containing the ICD-9 and Note
            infection labels for each subject at each charttime.

    Returns:
        severe_sepsis_summary (pd.DataFrame): A merged dataframe containing all
            of the columns from the <dev_sepsis> and <organ_dys> DataFrames,
            as well as a new column called "severe_sepsis_status" that indicates whether
            or not the subject met the TREWScore Sepsis definition at a given
            charttime.
    """

    # Overwrite this variable with the return value
    severe_sepsis_summary = None


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    

    return severe_sepsis_summary


def summarize_septic_shock(
    dev_severe_sepsis: pd.DataFrame,
    hypotension_labels: pd.DataFrame,
    fluids_all: pd.DataFrame,
):
    """
    Determines whether or not a subject met the TREWScore Septic Shock definition at a
    given charttime. Returns a merged dataframe containing the septic_shock_status column.

    Where `septic_shock` is a binary column indicating whether or not the subject met the
    TREWScore Septic Shock definition at a given charttime.

    #### TREWScore Septic Shock definition:
    At a given charttime, a subject is considered to have septic shock if they:
    - Have severe sepsis (as defined by the TREWScore Severe Sepsis definition)
    - Have hypotension
    - Have adequate fluid resuscitation

    Use a full (outer) merge strategy to combine the <dev_severe_sepsis>,
    <hypotension_labels>, and <fluids_all> DataFrames, and then use last-observation-
    carried-forward (LOCF) to fill in missing values WITHIN `subject_id`, `hadm_id`, `icustay_id`
    groups. After performing the LOCF, fill all remaining missing values with False.

    The returned DataFrame should contain the following columns (in addition to
        optional others):
        - `subject_id`, `hadm_id`, `icustay_id`, `charttime`, `septic_shock_status`


    IMPLEMENTATION INSTRUCTIONS:
        - Complete the function below (where indicated) to return a merged dataframe
            containing the following columns (in addition to optional others):
                - `subject_id`, `hadm_id`, `icustay_id`, `charttime`, `septic_shock_status`
            where `septic_shock_status` is a binary column indicating whether or not the subject
            met the TREWScore Septic Shock definition at a given charttime.
            - NOTE: You will need to choose the appropriate join keys for the merge.
        - No other error checking is required.

    HINTS:
        - You may find the following functions useful:
            - Pandas: merge()
        - You may find it helpful to create a helper function that determines whether
            or not a subject met the TREWScore Septic Shock definition at a given
            charttime.

    Parameters:
        dev_severe_sepsis (pd.DataFrame): A DataFrame containing the severe_sepsis_status
            for each subject at each charttime.
        hypotension_labels (pd.DataFrame): A DataFrame containing the hypotension labels
            for each subject at each charttime.
        fluids_all (pd.DataFrame): A DataFrame containing the fluid resuscitation labels
            for each subject at each charttime.

    Returns:
        septic_shock_summary (pd.DataFrame): A merged dataframe containing the
            following columns:
                - `subject_id`, `hadm_id`, `icustay_id`, `charttime`, `septic_shock_status`
            where `septic_shock_status` is a binary column indicating whether or not the subject
            met the TREWScore Septic Shock definition at a given charttime.
    """

    septic_shock_summary = None


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    

    return septic_shock_summary


# NOTE: You may find it helpful to create additional helper functions below this line.
