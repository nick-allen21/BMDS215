"""
BIOMEDIN 215: Assignment 4
features.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""

import pandas as pd
import numpy as np


def get_diagnoses(
    admissions: pd.DataFrame, diagnoses: pd.DataFrame, shock_labels: pd.DataFrame
):
    """
    Utilizing the admissions, diagnoses, and shock_labels dataframes, return a
    dataframe that only contains the diagnoses that occurred BEFORE the index_time
    for patient's in the shock_labels dataframe.

    IMPLEMENTATION INSTRUCTIONS:
       - Implement the function below to return a dataframe that contains
            the diagnoses that occurred BEFORE the index_time for patient's in the
            shock_labels dataframe with the following columns:
                - `subject_id`: the unique identifier for each patient
                - `hadm_id`: the unique identifier for each patient hospital admission
                - `diagnosis_time`: the time of the diagnosis
                - `icd9_code`: the icd9 code for the diagnosis
                - `index_time`: the time of the shock diagnosis
         - Utilize the appropriate columns in admissions, diagnoses, and shock_labels
            dataframes to pull the information needed.

    Parameters:
        admissions (pd.DataFrame): a dataframe containing the admissions information
        diagnoses (pd.DataFrame): a dataframe containing the diagnoses information
        shock_labels (pd.DataFrame): a dataframe containing the shock labels for
            each patient in the cohort

    Returns:
        dx (pd.DataFrame): a dataframe containing the diagnoses that occurred BEFORE
            the index_time for patient's in the shock_labels dataframe
    """

    # Overwrite this variable with the return value in your implementation
    dx = None
   
    # Validate required columns
    required_adm = {"subject_id", "hadm_id", "admittime"}
    if not required_adm.issubset(admissions.columns):
        missing = required_adm - set(admissions.columns)
        raise ValueError(f"Missing required columns in admissions: {missing}")

    required_dx = {"subject_id", "hadm_id", "icd9_code"}
    if not required_dx.issubset(diagnoses.columns):
        missing = required_dx - set(diagnoses.columns)
        raise ValueError(f"Missing required columns in diagnoses: {missing}")

    required_labels = {"subject_id", "hadm_id", "index_time"}
    if not required_labels.issubset(shock_labels.columns):
        missing = required_labels - set(shock_labels.columns)
        raise ValueError(f"Missing required columns in shock_labels: {missing}")

    # Merge diagnoses with admissions to get admittime as diagnosis_time
    dx_adm = pd.merge(
        diagnoses[["subject_id", "hadm_id", "icd9_code"]],
        admissions[["subject_id", "hadm_id", "admittime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    ).rename(columns={"admittime": "diagnosis_time"})

    # Merge with shock_labels to attach each subject's index_time for their index admission
    # Join on subject_id ONLY so we include diagnoses from prior admissions as well
    dx_joined = pd.merge(
        dx_adm,
        shock_labels[["subject_id", "index_time", "hadm_id"]]
        .rename(columns={"hadm_id": "index_hadm_id"}),
        on=["subject_id"],
        how="inner",
    )

    # Ensure datetime types
    dx_joined["diagnosis_time"] = pd.to_datetime(dx_joined["diagnosis_time"], utc=True, errors="coerce")
    dx_joined["index_time"] = pd.to_datetime(dx_joined["index_time"], utc=True, errors="coerce")

    # Keep diagnoses strictly before index_time
    mask = (dx_joined["diagnosis_time"] < dx_joined["index_time"]) & (
        dx_joined["hadm_id"] != dx_joined["index_hadm_id"]
    )
    dx = dx_joined.loc[mask, [
        "subject_id",
        "hadm_id",
        "diagnosis_time",
        "icd9_code",
        "index_time",
    ]].reset_index(drop=True)
    
    # De-duplicate potential repeated coding entries if present
    dx = dx.drop_duplicates(subset=["subject_id", "hadm_id", "icd9_code", "diagnosis_time"]).reset_index(drop=True)
    
    return dx


def calc_ic(dx_features: pd.DataFrame, all_patients_count: int) -> pd.DataFrame:
    """
    Calculate the IC score for each diagnosis in the dx_features dataframe.

    IMPLEMENTATION INSTRUCTIONS:
         - Implement the function below to return a dataframe that contains
            the IC score for each diagnosis in the dx_features dataframe with
            the following columns:
                - `icd9_code`: the icd9 code for the diagnosis
                - `IC`: the IC score for the diagnosis
            - Utilize the appropriate columns in the dx_features dataframe to pull
                the information needed.
            - Your implementation should utilize the all_patients_count variable to
                calculate the IC score for each diagnosis.

    HINTS:
        - You can use numpy functions to opperate on dataframe columns.
        - Check out the .agg function in pandas to determine the number of unique
            patients with each diagnosis icd9_code. (You may find it helpful
            to use this with the .groupby function.)

    Parameters:
        dx_features (pd.DataFrame): a dataframe containing the valid diagnosis
            features for each patient in the cohort
        all_patients_count (int): the number of patients in the cohort

    Returns:
        icd9_ic (pd.DataFrame): a dataframe containing the IC score for each
            diagnosis icd9 code in the dx_features dataframe
    """

    # Overwrite this variable with the return value in your implementation
    icd9_ic = None

    # Validate inputs
    required_cols = {"icd9_code", "subject_id"}
    if not required_cols.issubset(dx_features.columns):
        missing = required_cols - set(dx_features.columns)
        raise ValueError(f"Missing required columns in dx_features: {missing}")

    # Count unique patients per ICD9 code
    counts = (
        dx_features.groupby("icd9_code")["subject_id"]
        .nunique()
        .reset_index(name="patient_count")
    )

    # Compute IC = -log2(p), where p = patient_count / all_patients_count
    probs = counts["patient_count"] / float(all_patients_count)
    icd9_ic = counts.assign(IC=-np.log2(probs))[["icd9_code", "IC"]]
    return icd9_ic


def filter_ic(dx_features: pd.DataFrame, icd9_ic: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dx_features dataframe to only include diagnoses that have an IC
    value between 4 and 9 (inclusive).

    IMPLEMENTATION INSTRUCTIONS:
        - Return a filtered version of the dx_features dataframe (with the same
            columns) that only contains diagnoses that have an IC value between
            4 and 9 (inclusive).

    Parameters:
        dx_features (pd.DataFrame): a dataframe containing the valid diagnosis
            features for each patient in the cohort
        icd9_ic (pd.DataFrame): a dataframe containing the IC score for each
            diagnosis icd9 code in the dx_features dataframe

    Returns:
        dx_filtered (pd.DataFrame): a dataframe containing the filtered diagnosis
            features for each patient in the cohort
    """

    # Overwrite this variable with the return value in your implementation
    dx_filtered = None


    # ==================== YOUR CODE HERE ====================
    # Validate inputs
    if "icd9_code" not in dx_features.columns:
        raise ValueError("dx_features must contain 'icd9_code'")
    required_ic_cols = {"icd9_code", "IC"}
    if not required_ic_cols.issubset(icd9_ic.columns):
        missing = required_ic_cols - set(icd9_ic.columns)
        raise ValueError(f"icd9_ic missing required columns: {missing}")

    # Select codes with 4 <= IC <= 9 (inclusive)
    allowed_codes = set(icd9_ic.loc[(icd9_ic["IC"] >= 4) & (icd9_ic["IC"] <= 9), "icd9_code"])

    # Filter dx_features to those codes, preserve original columns
    dx_filtered = dx_features.loc[dx_features["icd9_code"].isin(allowed_codes)].reset_index(drop=True)
    # ==================== YOUR CODE HERE ====================


    return dx_filtered


def get_diagnosis_features(dx_selected: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a patient-feature matrix that summarizes and organizes diagnoses features.

        Each row represents a patient and each column represents a time-binned
        diagnosis code, where there are two bins (RECENT and PRIOR).
            - RECENT: diagnoses that occurred within 6 months of the index time
            - PRIOR: diagnoses that occurred more than 6 months before the index time

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below to return a dataframe that contains
            the patient-feature matrix with the following columns:
                - `subject_id`: the unique identifier for each patient
                And the following columns for each diagnosis code:
                - `RECENT_<icd9_code>`: the number of times the diagnosis occurred
                    within 6 months of the index time
                - `PRIOR_<icd9_code>`: the number of times the diagnosis occurred
                    more than 6 months before the index time
                - For date calculations, assume that 1 month == 30.44 days

    HINTS:
        - A great way to implement this function is to first create a new column
            called `time_bin` that indicates if each diagnosis should be considered
            RECENT or PRIOR.
        - Then, you can use the pandas groupby function to count the number of times
            each diagnosis occurred for each patient.
        - Your implementation should utilize the pandas_pivot function.
        - You may find it helpful to write helper functions to complete this task.

    Parameters:
        dx_selected (pd.DataFrame): a dataframe containing the filtered diagnosis
            features for each patient in the cohort

    Returns:
        patient_diagnosis_features (pd.DataFrame): a dataframe containing the
            patient-feature matrix for diagnoses

    """

    # Overwrite this variable with the return value in your implementation
    patient_diagnosis_features = None


    # ==================== YOUR CODE HERE ====================
    # Validate inputs
    required_cols = {"subject_id", "diagnosis_time", "icd9_code", "index_time"}
    if not required_cols.issubset(dx_selected.columns):
        missing = required_cols - set(dx_selected.columns)
        raise ValueError(f"dx_selected missing required columns: {missing}")

    df = dx_selected.copy()
    # Time delta in days between index_time and diagnosis_time
    delta_days = (pd.to_datetime(df["index_time"]) - pd.to_datetime(df["diagnosis_time"])) \
        .dt.total_seconds() / 86400.0

    # Define bins: RECENT if within 6 months (<= 6*30.44 days), else PRIOR
    six_months_days = 6 * 30.44
    df["time_bin"] = np.where(delta_days <= six_months_days, "RECENT", "PRIOR")

    # Count occurrences per subject, code, and time_bin
    counts = (
        df.groupby(["subject_id", "icd9_code", "time_bin"])  
          .size()
          .reset_index(name="count")
    )

    # Feature column names: RECENT_<code> and PRIOR_<code>
    counts["feature"] = counts["time_bin"] + "_" + counts["icd9_code"].astype(str)

    # Pivot to wide patient-feature matrix
    pivot = counts.pivot_table(
        index="subject_id",
        columns="feature",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure subject_id is a column and counts are integers
    pivot = pivot.reset_index()
    for col in pivot.columns:
        if col != "subject_id":
            pivot[col] = pivot[col].astype(int)

    patient_diagnosis_features = pivot
    # ==================== YOUR CODE HERE ====================
    
    
    return patient_diagnosis_features


# NOTE: Feel free to add additional helper functions if you wish!



# NOTE: For any helper functions you choose to implement, please include a docstring 
#       that briefly describes the function and its parameters/returns.
#       This will help us better understand your code for awarding partial credit.

