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
    # Merge SIRS with infections on admission identifiers
    sepsis_summary = pd.merge(
        dev_sirs,
        all_infections,
        on=["subject_id", "hadm_id"],
        how="left",
    )

    # Replace missing infection indicators with 0
    for col in ["has_icd9_infection", "has_note_infection"]:
        if col in sepsis_summary.columns:
            sepsis_summary[col] = (
                sepsis_summary[col]
                .replace({True: 1, False: 0})
                .fillna(0)
                .astype(int)
            )

    # Compute sepsis status inplace
    get_sepsis_status(sepsis_summary)
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
    # Count how many SIRS criteria are met
    criteria_cols = [
        c for c in ["criteria_1", "criteria_2", "criteria_3", "criteria_4"] if c in sepsis_summary.columns
    ]
    num_criteria = sepsis_summary[criteria_cols].fillna(False).astype(bool).sum(axis=1)

    # Infection present if either source indicates infection
    has_icd9 = sepsis_summary.get("has_icd9_infection", 0)
    has_notes = sepsis_summary.get("has_note_infection", 0)
    infection_present = (
        pd.Series(has_icd9).fillna(0).astype(int).gt(0)
        | pd.Series(has_notes).fillna(0).astype(int).gt(0)
    )

    sepsis_summary["sepsis_status"] = (num_criteria >= 2) & infection_present
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
    severe_sepsis_summary = pd.merge(
        dev_sepsis,
        organ_dys,
        on=["subject_id", "hadm_id"],
        how="left",
    )

    # Ensure organ dysfunction indicator present and clean
    if "has_organ_dysfunction" in severe_sepsis_summary.columns:
        severe_sepsis_summary["has_organ_dysfunction"] = (
            severe_sepsis_summary["has_organ_dysfunction"]
            .replace({True: 1, False: 0})
            .fillna(0)
            .astype(int)
        )
    else:
        severe_sepsis_summary["has_organ_dysfunction"] = 0

    severe_sepsis_summary["severe_sepsis_status"] = (
        severe_sepsis_summary.get("sepsis_status", False).astype(bool)
        & severe_sepsis_summary["has_organ_dysfunction"].astype(int).gt(0)
    )
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
    # IDs and time key
    keys = ["subject_id", "hadm_id", "icustay_id", "charttime"]

    # Keep minimal columns and coerce to expected types
    base_cols = keys + ["severe_sepsis_status"]
    base = dev_severe_sepsis[base_cols].copy() if "severe_sepsis_status" in dev_severe_sepsis.columns else dev_severe_sepsis[keys].copy()

    h = hypotension_labels.copy()
    h = h[keys + ["hypotension"]] if "hypotension" in h.columns else h.assign(hypotension=False)[keys + ["hypotension"]]
    # Coerce hypotension to boolean
    if h["hypotension"].dtype != bool:
        h["hypotension"] = (
            h["hypotension"]
            .replace({True: True, False: False, "TRUE": True, "FALSE": False, 1: True, 0: False})
            .fillna(False)
            .astype(bool)
        )

    f = fluids_all.copy()
    f = f[keys + ["adequate_fluid"]] if "adequate_fluid" in f.columns else f.assign(adequate_fluid=False)[keys + ["adequate_fluid"]]
    # Coerce adequate_fluid to boolean
    if f["adequate_fluid"].dtype != bool:
        f["adequate_fluid"] = (
            f["adequate_fluid"]
            .replace({True: True, False: False, "TRUE": True, "FALSE": False, 1: True, 0: False})
            .fillna(False)
            .astype(bool)
        )

    # Drop any exact duplicate key rows to avoid row-multiplication on merge
    base = base.drop_duplicates(subset=keys, keep="last")
    h = h.drop_duplicates(subset=keys, keep="last")
    f = f.drop_duplicates(subset=keys, keep="last")

    # Full outer merge of all three sources
    merged = pd.merge(base, h, on=keys, how="outer")
    merged = pd.merge(merged, f, on=keys, how="outer")

    # Sort for LOCF and forward-fill within subject/hadm/icu groups
    merged = merged.sort_values(keys)
    grp = ["subject_id", "hadm_id", "icustay_id"]

    indicator_cols = []
    if "severe_sepsis_status" in merged.columns:
        indicator_cols.append("severe_sepsis_status")
    indicator_cols += [c for c in ["hypotension", "adequate_fluid"] if c in merged.columns]

    if indicator_cols:
        merged[indicator_cols] = merged.groupby(grp, sort=False)[indicator_cols].ffill()

    # Fill any remaining missing indicators with False
    for c in indicator_cols:
        merged[c] = merged[c].fillna(False).astype(bool)

    # Compute septic shock: severe sepsis AND hypotension AND adequate fluid
    # If severe_sepsis_status isn't present for a row, treat as False
    sev = merged.get("severe_sepsis_status", False)
    hypot = merged.get("hypotension", False)
    fluid = merged.get("adequate_fluid", False)
    merged["septic_shock_status"] = sev.astype(bool) & hypot.astype(bool) & fluid.astype(bool)

    # Return the 8 expected columns, ordered
    cols_out = keys + ["severe_sepsis_status", "hypotension", "adequate_fluid", "septic_shock_status"]
    septic_shock_summary = merged[cols_out].sort_values(keys).reset_index(drop=True)

    # ==================== YOUR CODE HERE ====================
    

    return septic_shock_summary


# NOTE: You may find it helpful to create additional helper functions below this line.
