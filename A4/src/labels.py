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

    # Validate required columns
    required_cols = {"admittime", "dischtime"}
    if not required_cols.issubset(admissions.columns):
        missing = required_cols - set(admissions.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Compute admission durations in hours without modifying input
    admit_dt = pd.to_datetime(admissions["admittime"], utc=True, errors="coerce")
    disc_dt = pd.to_datetime(admissions["dischtime"], utc=True, errors="coerce")
    durations = disc_dt - admit_dt

    # Keep rows with duration > 12 hours
    mask = durations >= timedelta(hours=12)
    filtered_admissions = admissions.loc[mask].copy()

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


    # Merge cohort labels (many rows per admission with charttime + labels)
    # with filtered admissions (one row per admission with admit/discharge).
    # Join on subject_id and hadm_id, keeping relevant columns.
    required_cols_cohort = {"subject_id", "hadm_id", "icustay_id", "charttime"}
    if not required_cols_cohort.issubset(cohort_labels.columns):
        missing = required_cols_cohort - set(cohort_labels.columns)
        raise ValueError(f"Missing required columns in cohort_labels: {missing}")

    required_cols_adm = {"subject_id", "hadm_id", "admittime", "dischtime"}
    if not required_cols_adm.issubset(admissions.columns):
        missing = required_cols_adm - set(admissions.columns)
        raise ValueError(f"Missing required columns in admissions: {missing}")

    merged_df = pd.merge(
        cohort_labels,
        admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    # Add relative_charttime and index_time inplace
    get_relative_charttime(merged_df)
    get_index_time(merged_df)
    
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

    # ==================== YOUR CODE HERE ====================
    # Validate required columns
    required_cols = {"admittime", "charttime"}
    if not required_cols.issubset(admissions.columns):
        missing = required_cols - set(admissions.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime types (should already be preprocessed per notebook)
    adm = pd.to_datetime(admissions["admittime"], utc=True, errors="coerce")
    cht = pd.to_datetime(admissions["charttime"], utc=True, errors="coerce")

    # Unrounded floating point number of hours between times
    admissions["relative_charttime"] = (cht - adm).dt.total_seconds() / 3600.0
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

    # ==================== YOUR CODE HERE ====================
    # Validate required column
    if "admittime" not in admissions.columns:
        raise ValueError("Missing required column: admittime")

    adm = pd.to_datetime(admissions["admittime"], utc=True, errors="coerce")
    admissions["index_time"] = adm + timedelta(hours=12)
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

    # Validate required columns
    required_cols = {
        "subject_id",
        "hadm_id",
        "icustay_id",
        "admittime",
        "dischtime",
        "charttime",
        "index_time",
        "septic_shock",
    }
    if not required_cols.issubset(merged_cohort.columns):
        missing = required_cols - set(merged_cohort.columns)
        raise ValueError(f"Missing required columns in merged_cohort: {missing}")

    df = merged_cohort.copy()

    # Ensure datetime types (should have been preprocessed)
    df["admittime"] = pd.to_datetime(df["admittime"], utc=True, errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], utc=True, errors="coerce")
    df["charttime"] = pd.to_datetime(df["charttime"], utc=True, errors="coerce")
    df["index_time"] = pd.to_datetime(df["index_time"], utc=True, errors="coerce")

    # Consider septic shock events that occur during the admission window
    during_admission = (df["charttime"] > df["admittime"]) & (df["charttime"] < df["dischtime"])
    shock_events = df.loc[during_admission & (df["septic_shock"].astype(str).str.upper() == "TRUE")]

    # Earliest septic shock time per admission
    earliest_shock = (
        shock_events.sort_values("charttime").groupby(["subject_id", "hadm_id", "icustay_id"], as_index=False)[
            "charttime"
        ].first()
    )
    earliest_shock = earliest_shock.rename(columns={"charttime": "earliest_shock_time"})

    # unique admissions
    base_cols = ["subject_id", "hadm_id", "icustay_id", "admittime", "dischtime", "index_time"]
    admissions_unique = df[base_cols].drop_duplicates()

    # Merge earliest shock times
    admissions_with_shock = pd.merge(
        admissions_unique,
        earliest_shock,
        on=["subject_id", "hadm_id", "icustay_id"],
        how="left",
    )

    # Determine label based on earliest shock time
    hours_to_shock = (
        (admissions_with_shock["earliest_shock_time"] - admissions_with_shock["admittime"])
        .dt.total_seconds()
        .div(3600.0)
    )

    # Eligibility and label rules
    # - Positive if shock occurs at > 15 hours after admission
    # - Exclude if shock occurs before 15 hours
    # - Negative if no shock during admission
    admissions_with_shock["label"] = (
        (hours_to_shock > 15).fillna(False)
    )
    eligible_mask = admissions_with_shock["label"] | hours_to_shock.isna()
    eligible = admissions_with_shock.loc[eligible_mask].copy()

    # For subjects with multiple eligible admissions, keep the latest admission (max admittime)
    idx_latest = eligible.groupby("subject_id")["admittime"].idxmax()
    label_df = eligible.loc[idx_latest, [
        "subject_id",
        "hadm_id",
        "icustay_id",
        "admittime",
        "dischtime",
        "index_time",
        "label",
    ]].sort_values("subject_id").reset_index(drop=True)

    return label_df



# NOTE: For any helper functions you choose to implement, please include a docstring 
#       that briefly describes the function and its parameters/returns.
#       This will help us better understand your code for awarding partial credit.

