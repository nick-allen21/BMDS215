"""
BIOMEDIN 215: Assignment 4
visualize.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union


def show_diagnosis_hist(
    dx_features: pd.DataFrame, output_file: Union[str, None] = None
) -> None:
    """
    Create a historgram demonstrating the distribution of the number of unique
    diagnositic histories that the codes belong to.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified to create a histogram where:
            - y-axis is labeled "Diagnosis Count":
                The y-axis represents the total count of codes within each
                'number of patient histories' interval.
            - x-axis is labeled "Patient Count":
                The x-axis represents the number of unique patient histories that the
                codes belong to.
        - You may specify other parameters as you see fit to make a good looking plot.
        - If output_file is specified (not None), save your plot to that file path.

        - Your implementation should utilize the function seaborn.histplot()
            - Documentation: https://seaborn.pydata.org/generated/seaborn.histplot.html

    Parameters:
        dx_features (pd.DataFrame): The patient diagnosis features dataframe.
        output_file (str): The path to save the plot to. If None, do not save.
    """

    # This line sets the size of the plot
    plt.figure(figsize=(10, 5))


    # ==================== YOUR CODE HERE ====================
    # Count unique patient histories per ICD9 code
    code_patient_counts = (
        dx_features.groupby("icd9_code")["subject_id"].nunique().rename("patient_count")
    )

    # Histogram of code frequencies by number of unique patient histories
    sns.histplot(code_patient_counts.values, bins="auto")
    plt.xlabel("Patient Count")
    plt.ylabel("Diagnosis Count")
    plt.title("Distribution of Diagnosis Code Patient Counts")

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    # ==================== YOUR CODE HERE ====================
    


def show_hr_plot(
    latest_hr_df: pd.DataFrame, output_file: Union[str, None] = None
) -> None:
    """
    Show a density plot of the latest heart rate measurements for each patient,
    separated by shock label. If output_file is None, do not save the plot. If
    output_file is specified, save the plot to that file path.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified to create a density plot where:
            - y-axis is labeled "Density":
                The y-axis represents the density of the heart rate measurements.
            - x-axis is labeled "Heart Rate (bpm)":
                The x-axis represents the heart rate measurements.
            - The plot should have two different colored lines, one for each shock
                label.
        - You may specify other parameters as you see fit to make a good looking plot.
        - If output_file is specified (not None), save your plot to that file path.
        - Your implementation should utilize the function seaborn.kdeplot()

    Parameters:
        latest_hr_df (pd.DataFrame): The dataframe containing the latest heart rate
            measurements for each patient.
        output_file (str): The path to save the plot to. If None, do not save.
    """

    # This line sets the size of the plot
    plt.figure(figsize=(10, 5))


    # ==================== YOUR CODE HERE ====================
    sns.kdeplot(
        data=latest_hr_df,
        x="latest_heart_rate",
        hue="label",
        common_norm=False,
        fill=False,
        bw_adjust=1.0,
    )
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel("Density")
    plt.title("Latest Heart Rate by Shock Label")
    plt.legend(title="Shock Label")

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    # ==================== YOUR CODE HERE ====================
    


def show_hr_time_plot(
    latest_hr_df: pd.DataFrame, output_file: Union[str, None] = None
) -> None:
    """
    Show a density plot of the latest heart rate measurements for each patient,
    separated by shock label. If output_file is None, do not save the plot. If
    output_file is specified, save the plot to that file path.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified to create a density plot where:
            - y-axis is labeled "Density":
                The y-axis represents the density of the heart rate measurements.
            - x-axis is labeled "Time Difference: Chart Time - Index Time (hours)"
                The x-axis represents the time difference between the chart time and
                the index time in hours.
            - The plot should have two different colored lines, one for each shock
                label.
        - You may specify other parameters as you see fit to make a good looking plot.
        - If output_file is specified (not None), save your plot to that file path.
        - Your implementation should utilize the function seaborn.kdeplot()

    Parameters:
        latest_hr_df (pd.DataFrame): The dataframe containing the latest heart rate
            measurements for each patient.
        output_file (str): The path to save the plot to. If None, do not save.
    """

    # This line sets the size of the plot
    plt.figure(figsize=(10, 6))

    # Time difference (charttime - index_time) in hours
    time_diff_hours = (
        (latest_hr_df["charttime"] - latest_hr_df["index_time"]).dt.total_seconds() / 3600.0
    )

    plot_df = latest_hr_df.copy()
    plot_df["time_diff_hours"] = time_diff_hours

    sns.kdeplot(
        data=plot_df,
        x="time_diff_hours",
        hue="label",
        common_norm=False,
        fill=False,
        bw_adjust=1.0,
    )
    plt.xlabel("Time Difference: Chart Time - Index Time (hours)")
    plt.ylabel("Density")
    plt.title("Latest HR Time Difference by Shock Label")
    plt.legend(title="Shock Label")

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()


def show_hr_scatter(
    latest_hr_df: pd.DataFrame,
    time_weighted_hr_df: pd.DataFrame,
    output_file: Union[str, None] = None,
) -> None:
    """
    Create a scatter plot comparing the latest heart rate measurements to the
    time-weighted average heart rate measurements for each patient. If output_file
    is None, do not save the plot. If output_file is specified, save the plot to
    that file path.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified to create a scatter plot where:
            - y-axis is labeled "Time Weighted Avg Heart Rate (bpm)":
                The y-axis represents the time-weighted average heart rate
                measurements.
            - x-axis is labeled "Latest Heart Rate (bpm)":
                The x-axis represents the latest heart rate measurements.
        - You may specify other parameters as you see fit to make a good looking plot.
        - If output_file is specified (not None), save your plot to that file path.
        - Your implementation should utilize the function seaborn.scatterplot()

    Parameters:
        latest_hr_df (pd.DataFrame): The dataframe containing the latest heart rate
            measurements for each patient.
        time_weighted_hr_df (pd.DataFrame): The dataframe containing the time-weighted
            average heart rate measurements for each patient.

    Returns:
        None
    """

    # This line sets the size of the plot
    plt.figure(figsize=(10, 6))

    merged = latest_hr_df.merge(
        time_weighted_hr_df[["subject_id", "time_wt_avg"]], on="subject_id", how="inner"
    )

    sns.scatterplot(
        data=merged,
        x="latest_heart_rate",
        y="time_wt_avg",
        hue="label",
        alpha=0.7,
        edgecolor=None,
    )
    plt.xlabel("Latest Heart Rate (bpm)")
    plt.ylabel("Time Weighted Avg Heart Rate (bpm)")
    plt.title("Latest vs Time-Weighted Average Heart Rate")
    plt.legend(title="Shock Label")

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
   