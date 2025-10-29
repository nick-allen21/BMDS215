"""
BIOMEDIN 215: Assignment 4
vitals.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""


import pandas as pd
import numpy as np
from typing import List


def filter_vitals(
    vitals: pd.DataFrame, shock_labels: pd.DataFrame, vitals_to_keep: List[str]
):
    """
    Filters the vitals dataframe to only include the vital measurements that have
    their vital_id include in vitals_to_keep, and only includes the vital measurements
    that were taken before the index_time for each patient. (NOTE: This is strictly
    before, not inclusive of the index_time.)

    IMPLEMENTATION INSTRUCTIONS:
        - Return a filtered version of the vitals dataframe that only includes the
            vital measurements that have their vital_id include in vitals_to_keep, and
            only includes the vital measurements that were taken before the index_time
            for each patient.
        - You will need to select the appropriate merge strategy when merging the
            shock_labels and vitals dataframes.

    Parameters:
        vitals (pd.DataFrame): A dataframe containing the vital measurements for each
            patient.
        shock_labels (pd.DataFrame): A dataframe containing the shock labels for each
            patient, as well as the index_time for each patient.
        vitals_to_keep (List[str]): A list of vital_ids that should be kept in the
            filtered vitals dataframe.

    Returns:
        filtered_vitals (pd.DataFrame): A dataframe containing the filtered vital
            measurements for each patient.
    """

    # Overwrite this variable with the return value in your implementation
    filtered_vitals = None

    # Keep only requested vital_ids
    vitals_keep = vitals.loc[vitals["vital_id"].isin(vitals_to_keep)].copy()

    # Attach index_time and label by joining on subject and the index admission (hadm_id)
    merged = pd.merge(
        vitals_keep,
        shock_labels[["subject_id", "hadm_id", "icustay_id", "index_time", "label"]],
        on=["subject_id", "hadm_id", "icustay_id"],
        how="inner",
    )

    # Ensure datetime types
    merged["charttime"] = pd.to_datetime(merged["charttime"], utc=True, errors="coerce")
    merged["index_time"] = pd.to_datetime(merged["index_time"], utc=True, errors="coerce")

    # Strictly before index_time
    filtered_vitals = merged.loc[merged["charttime"] < merged["index_time"]].reset_index(drop=True)
    

    return filtered_vitals


def get_latest_hr(heart_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe containing the latest heart rate measurement for each patient
    prior to their index_time. The returned dataframe should contain the columns:
        - `subject_id`: The ID of the patient
        - `charttime`: The time at which the heart rate measurement was taken
        - `latest_heart_rate`: The latest heart rate measurement for the patient
        - `index_time`: The time at which the patient was diagnosed with shock
        - `label`: The shock label for the patient

    IMPLEMENTATION INSTRUCTIONS:
        - Return a dataframe containing the latest heart rate measurement for each
            patient prior to their index_time.

    Parameters:
        heart_rates (pd.DataFrame): A dataframe containing the heart rate measurements
            for each patient.

    Returns:
        latest_hr_df (pd.DataFrame): A dataframe containing the latest heart rate
            measurement for each patient prior to their index_time.
    """

    # Overwrite this variable with the return value in your implementation
    latest_hr_df = None


    # ==================== YOUR CODE HERE ====================
    # Ensure datetime types
    df = heart_rates.copy()
    df["charttime"] = pd.to_datetime(df["charttime"], utc=True, errors="coerce")
    df["index_time"] = pd.to_datetime(df["index_time"], utc=True, errors="coerce")

    # Keep only measurements strictly before index_time (safety if not pre-filtered)
    df = df.loc[df["charttime"] < df["index_time"]]

    # For each subject, pick the row with the latest charttime
    idx = df.groupby("subject_id")["charttime"].idxmax()
    latest = df.loc[idx, [
        "subject_id",
        "charttime",
        "valuenum",
        "index_time",
        "label",
    ]].rename(columns={"valuenum": "latest_heart_rate"}).reset_index(drop=True)

    latest_hr_df = latest
    # ==================== YOUR CODE HERE ====================
    

    return latest_hr_df


def get_time_weighted_hr(heart_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe containing the time-weighted average heart rate for each
    patient, using the hr vitals taken prior to their index_time. The returned
    dataframe should contain the columns:
        - `subject_id`: The ID of the patient
        - `time_wt_avg`: The time-weighted average heart rate for the patient


    IMPLEMENTATION INSTRUCTIONS:
        - Return a dataframe containing the time-weighted average heart rate for each
            patient, using the hr vitals taken prior to their index_time.
        - Calculate the weight values for each heart rate measurement using the
            provided formula.
            weight = exp(-|dt| - 1) where dt is the time difference in hours between
            the charttime and the index_time.
        - Calculate the time-weighted average heart rate for each patient as follows:
            time_wt_avg[patient A] = sum(weights[patient A] * heart_rate[patient A]) /
                sum(weights[patient A])
            Notes:
                - The time_wt_avg is a scalar value for each patient.
                - In the formula above, weights[patient A] is a vector of weight values
                    for each heart rate measurement for patient A.
                - heart_rate[patient A] is a vector of heart rate values for each
                    heart rate measurement for patient A.
                - The sum(weights[patient A] * heart_rate[patient A]) is the sum of the
                    element-wise multiplication of the weights and heart_rate vectors.
                    For example, if weights[patient A] = [1, 2, 3] and
                    heart_rate[patient A] = [4, 5, 6], then the sum(weights[patient A] *
                        heart_rate[patient A]) = (1 * 4) + (2 * 5) + (3 * 6) = 32
                - The sum(weight) is the sum of the weight values for all rows in the
                    dataframe that correspond to the same patient.

                - HINT: Whenever you see opperations being performed on a "group"
                    of rows, you should think about using the groupby function.

        - After calculating the time-weighted average heart rate for each patient,
            filter OUT any patients that have a time-weighted average heart rate of
            NaN.
        - The final dataframe should have the following columns:
            - subject_id
            - time_wt_avg
        - Do not modify the input dataframe.

    Parameters:
        heart_rates (pd.DataFrame): A dataframe containing the heart rate measurements
            for each patient.

    Returns:
        time_wt_avg_df (pd.DataFrame): A dataframe containing the time-weighted
            average heart rate for each patient.
    """

    # Overwrite this variable with the return value in your implementation
    result = None


    # ==================== YOUR CODE HERE ====================
    # Ensure datetime and filter to strictly before index_time
    df = heart_rates.copy()
    df["charttime"] = pd.to_datetime(df["charttime"], utc=True, errors="coerce")
    df["index_time"] = pd.to_datetime(df["index_time"], utc=True, errors="coerce")
    df = df.loc[df["charttime"] < df["index_time"]].dropna(subset=["valuenum"]) 

    # Compute weights: w = exp(-(|dt| + 1)), where dt is in hours
    dt_hours = (df["charttime"] - df["index_time"]).dt.total_seconds().abs() / 3600.0
    df["weight"] = np.exp(-(dt_hours + 1.0))
    df["xw"] = df["valuenum"] * df["weight"]

    # Aggregate per subject
    sum_w = df.groupby("subject_id")["weight"].sum()
    sum_xw = df.groupby("subject_id")["xw"].sum()
    result = (
        pd.DataFrame({"subject_id": sum_w.index, "time_wt_avg": (sum_xw / sum_w).values})
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["time_wt_avg"]).reset_index(drop=True)
    )
    # ==================== YOUR CODE HERE ====================
    
    
    return result
