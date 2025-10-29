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
    
    # TODO: Implement
    
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
    
    # TODO: Implement
    
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


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    


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


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    
