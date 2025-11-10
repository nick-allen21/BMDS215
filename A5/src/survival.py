"""
survival.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""


import pandas as pd
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

TOTAL_SECOND_IN_DAY = 86400.0

def calc_survival_time(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a Dataframe containing the survival time for each patient in the
    cohort dataframe. The survival time is defined as follows:
        - If the patient died during their stay, the survival time is the number
            of days between admission and death.
        - If the patient survived their stay, the survival time is the number of
            days between admission and censorship. Censorship occurs when a patient
            is discharged from the hospital without dying. We use the term censorship
            as it implies that we do not know the exact survival time for that patient,
            only that they survived at least that long.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified.
        - Calculate the survival time for each patient in the cohort dataframe and store
            the results in a column named "survival_time_in_days".
        - In the returned dataframe, only include the columns "subject_id", "death_in_stay",
            "survival_time_days", and "oxy_drop".
        - Do NOT modify the input dataframe.
    """

    # Overwrite this return variable in your implementation
    survival_df = None


    # ==================== YOUR CODE HERE ====================
    # Work on a copy so we don't mutate the input
    df = cohort.copy(deep=True)

    # Parse timestamps
    index_time = pd.to_datetime(df["index_time"], errors="coerce")
    death_time = pd.to_datetime(df["deathtime"], errors="coerce")
    censor_time = pd.to_datetime(df["censor_time"], errors="coerce")

    # Determine end time: death time if died, otherwise censor (discharge) time
    died_mask = df["death_in_stay"].astype(str).str.lower().eq("died")
    end_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    end_time.loc[died_mask] = death_time.loc[died_mask]
    end_time.loc[~died_mask] = censor_time.loc[~died_mask]

    # Compute survival time in days
    survival_time_days = (end_time - index_time).dt.total_seconds() / TOTAL_SECOND_IN_DAY
    # Floor to whole days and store as integers
    df["survival_time_days"] = np.floor(survival_time_days).astype(int)

    # Return only the requested columns
    survival_df = df[["subject_id", "death_in_stay", "survival_time_days", "oxy_drop"]]
    # ==================== YOUR CODE HERE ====================
    

    return survival_df


def display_kaplan_meier_curve(
    survival_df: pd.DataFrame, output_path: str = None
) -> None:
    """
    Creates and displays a Kaplan-Meier curve using the survival data in the
    survival_df dataframe. If output_path is specified, saves the plot to that
    file path.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified.
        - Use the survival data to generate a Kaplan-Meier curve for each oxy_drop
            group. Display both curves on the same plot. The x-axis should be labeled
            "Survival Time Days" and the y-axis should be labeled "Survival Probability".
        - If output_path is specified (not None), save your plot to that file path.
        - You may utilize pandas and/or numpy to help you calculate the survival probability
            for each time point for each oxy_drop group. However, you may not use any
            functions that are specifically designed to calculate a Kaplan-Meier curve.
        - Use the function plt.step() to plot the Kaplan-Meier curve.

    HINTS:
    This function can be tricky to implement! There are multiple ways to implement
    this function, but one way is to follow the steps below:
        - Sort the survival_df in ascending order of survival_time_days.
        - Group the data by oxy_drop and survival_time_days. This gives you the
            number of patients who died and survived at each time point for each
            oxy_drop group.
        - Count the number of patients who died at each time point for
            each oxy_drop group.
        - Count the number of patients who were censored at each time point for
            each oxy_drop group.
        - Sort the dataframe again by oxy_drop and survival_time_days, but this time
            in descending order of survival_time_days. This is important for the
            next step.
        - Using the counts of patients who died and were censored at each time point
            for each oxy_drop group, calculate the number of patients at risk at each
            time point for each oxy_drop group. This is the number of patients who
            either died or were censored **at that time point or later**. Since we
            sorted the dataframe in descending order of survival_time_days, we can
            calculate this by cumulatively summing the number of patients who died
            and were censored up to but not including each time point for each oxy_drop
            group.
            - NOTE: There are multiple ways to do this, but one way is to use the
            pandas.DataFrame.cumsum() function
        - Using the number of patients at risk and the number of patients who died
            at each time point for each oxy_drop group, calculate the survival
            probability at each time point for each oxy_drop group.
            - NOTE: There are multiple ways to do this, but one way is to use the
            pandas.DataFrame.cumprod() function
        - Plot the Kaplan-Meier curve for each oxy_drop group on the same plot.

    Parameters:
        survival_df (pd.DataFrame): A dataframe containing the survival data
        output_path (str): The path to save the plot to. If None, do not save.
    """

    # Implement this function where specified
    def _km_curve_for_group(group_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes x (time) and y (survival probability) for a single group using
        the Kaplan–Meier product-limit estimator without survival-specific libs.
        """
        # Times and event indicator
        times = group_df["survival_time_days"].astype(float)
        events = group_df["death_in_stay"].astype(str).str.lower().eq("died").astype(int)

        # Count events and censoring at each unique time
        counts = (
            pd.DataFrame({"time": times, "event": events})
            .groupby(["time", "event"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "censored", 1: "deaths"})
            .sort_index()
        )

        # Number at risk just prior to each time
        n_total = len(group_df)
        removed_cumulative = counts[["deaths", "censored"]].sum(axis=1).cumsum().shift(fill_value=0)
        n_risk = n_total - removed_cumulative

        # KM step
        hazard = counts["deaths"] / n_risk.replace(0, np.nan)
        step = (1.0 - hazard).fillna(1.0)
        survival = step.cumprod()

        # Build stepwise arrays (start at time 0 with survival 1)
        x = np.r_[0.0, counts.index.values]
        y = np.r_[1.0, survival.values]
        return x, y


    # ==================== YOUR CODE HERE ====================
    # Plot KM curves for each oxy_drop group on the same axes
    plt.figure(figsize=(7, 4))
    for group_name, group_df in survival_df.groupby("oxy_drop"):
        x_vals, y_vals = _km_curve_for_group(group_df)
        plt.step(x_vals, y_vals, where="post", label=str(group_name))

    plt.xlabel("Survival Time Days")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan–Meier Survival by oxy_drop")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    # ==================== YOUR CODE HERE ====================
    



# NOTE: For any helper functions you choose to implement, please include a docstring 
#       that briefly describes the function and its parameters/returns.
#       This will help us better understand your code for awarding partial credit.

