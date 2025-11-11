"""
propensity.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""


import pandas as pd
import numpy as np
from typing import List, Union, Tuple
import seaborn as sns
import matplotlib.pyplot as plt


def plot_logit_scores(propensity_scores: pd.DataFrame, output_file: str) -> None:
    """
    Creates a density plot of the logit of the propensity scores, stratified by
    whether the patient had an oxygen drop or not. This plot is useful for
    determining whether the propensity scores have a good degree of overlap.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified.
        - Utilize the function `safe_logit` to calculate the logit of the
            propensity scores for each patient. The function `safe_logit` is
            provided below and does not need to be modified.
        - Use the function `sns.kdeplot` to create a density plot of the logit
            of the propensity scores, stratified by whether the patient had an
            oxygen drop or not.
            - The x-axis should be labeled "logit(P)"
            - The y-axis should be labeled "Density"
            - The title should be "Density Plot of logit(P)"
            - Specify these values for the following parameters (in addition to others as needed):
                - common_norm=False
                - fill=True
                - alpha=0.5
                - linewidth=0
            - Documentation: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
        - If output_file is specified, save your plot to that file path.
            If output_file is None, do not save.
        - Do NOT modify the input dataframe.



    Parameters:
        propensity_scores (pd.DataFrame): The propensity scores for each patient.
        output_file (str): The path to save the plot to.
    """

    # Create a plot
    plt.figure(figsize=(10, 6))


    # ==================== YOUR CODE HERE ====================
    # Work on a copy to avoid mutating the input
    df = propensity_scores.copy(deep=True)
    # Compute logit of propensity scores
    df["logit_p"] = df["propensity_score"].apply(safe_logit)

    # Density plot stratified by oxy_drop status
    sns.kdeplot(
        data=df,
        x="logit_p",
        hue="oxy_drop",
        common_norm=False,
        fill=True,
        alpha=0.5,
        linewidth=0,
    )
    plt.xlabel("logit(P)")
    plt.ylabel("Density")
    plt.title("Density Plot of logit(P)")

    # Save if requested
    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    # ==================== YOUR CODE HERE ====================
    


def safe_logit(p: float) -> float:
    """
    PROVIDED FUNCTION
    This function is provided and does not need to be modified.

    Calculates the logit of p, but ensures that p is between 0.025 and 0.975
    for numerical stability.

    Parameters:
        p (float): The probability to transform

    Returns:
        logit_p (float): The logit of p
    """

    # Ensure p is between 0.025 and 0.975
    if p > 0.975:
        p = 0.975
    elif p < 0.025:
        p = 0.025

    return np.log(p / (1 - p))


def caliper_match(
    propensity_scores: pd.DataFrame, caliper: float = 0.25
) -> List[Tuple[int, int]]:
    """
    Performs a caliper match on the propensity scores. This function should
    return a list of tuples, where each tuple contains the subject_ids of a pair
    of matched patients.

    For each patient with oxy_drop, find the patient in the no_oxy_drop group
    with the closest logit propensity score. Ensure that the difference is within
    the caliper threshold (i.e. the absolute value of the difference between the
    logit propensity scores is less than or equal to the caliper threshold). If
    the difference is within the caliper threshold, add the pair of subject_ids
    to the list of matches in the order (oxy_drop_subject_id, no_oxy_drop_subject_id).

    The caliper threshold is calculated as the standard deviations of the logit
    propensity scores (of all patients) multiplied by the caliper parameter.
        caliper_threshold = std(logit_P_all_patients) * caliper

    The logit propensity scores should be calculated using the function
    `safe_logit`, which is provided.


    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where specified.
        - Utilize the function `safe_logit` to calculate the logit of the
            propensity scores for each patient. The function `safe_logit` is
            provided below and does not need to be modified.
        - Calculate the caliper threshold based on the standard deviation of the
            logit propensity scores and the caliper parameter. (As described above)
        - For each patient with oxy_drop, find the patient in the no_oxy_drop group
            with the closest logit propensity score. Ensure that the difference is
            within the caliper threshold (i.e. the absolute value of the difference
            between the logit propensity scores is less than or equal to the caliper
            threshold). If the difference is within the caliper threshold, add the
            pair of subject_ids to the list of matches in the order
            (oxy_drop_subject_id, no_oxy_drop_subject_id).
        - Ensure that you match between the groups *without replacement*. If a
            patient in the no_oxy_drop group has already been matched, they should
            be removed from consideration for future matches. (No matches should
            share ANY subject_ids)

    """

    # Overwrite this return variable(s) in your implementation
    matches = []


    # ==================== YOUR CODE HERE ====================
    # Copy and compute logit propensity for all patients
    df = propensity_scores.copy(deep=True)
    df["logit_p"] = df["propensity_score"].apply(safe_logit)

    # Caliper threshold based on SD of all logit scores
    caliper_threshold = df["logit_p"].std() * caliper

    # Split by treatment (oxy_drop == 1) vs control (0)
    treated = df[df["oxy_drop"] == 1][["logit_p"]]
    control = df[df["oxy_drop"] == 0][["logit_p"]]

    # Available control ids set for matching without replacement
    available_controls = set(control.index.tolist())

    # For determinism, iterate treated in ascending logit
    for tid, trow in treated.sort_values("logit_p").iterrows():
        if not available_controls:
            break
        # Candidate control pool
        cand = control.loc[list(available_controls)]
        # Find closest by absolute difference in logit
        diffs = (cand["logit_p"] - trow["logit_p"]).abs()
        closest_cid = diffs.idxmin()
        min_diff = diffs.loc[closest_cid]

        # Accept match if within caliper
        if np.isfinite(min_diff) and min_diff <= caliper_threshold:
            matches.append((int(tid), int(closest_cid)))
            available_controls.remove(closest_cid)
        # else: no match for this treated unit
    # ==================== YOUR CODE HERE ====================
    

    return matches
