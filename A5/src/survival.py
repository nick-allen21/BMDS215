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
    
    # TODO: Implement
    
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
    pass


    # ==================== YOUR CODE HERE ====================
    
    # TODO: Implement
    
    # ==================== YOUR CODE HERE ====================
    



# NOTE: For any helper functions you choose to implement, please include a docstring 
#       that briefly describes the function and its parameters/returns.
#       This will help us better understand your code for awarding partial credit.

