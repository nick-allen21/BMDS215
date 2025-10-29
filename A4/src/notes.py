"""
BIOMEDIN 215: Assignment 4
notes.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""
import pandas as pd
import numpy as np
from typing import List, Union
from tqdm import tqdm


def filter_by_chartdate(
    shock_labels: pd.DataFrame, notes: pd.DataFrame
) -> pd.DataFrame:
    """
    Filters the notes dataframe to only include notes that have a chartdate
    prior to the index_time for patient.

    IMPLEMENTATION INSTRUCTIONS:
        - Return a filtered version of the notes dataframe (with the same
            columns) that only contains notes that have a chartdate prior to
            the day of the index_time for the patient.
        - Drop the "index_time" and "label" columns from the dataframe before
            returning
        - Do not modify the input dataframe.

    Parameters:
        - shock_labels: dataframe containing the index_time and label for each
            patient
        - notes: dataframe containing all of the notes for each patient

    Returns:
        - notes_filtered: dataframe containing only notes that have a chartdate
            prior to the day of the index_time for the patient
    """

    # Overwrite this variable with the return value in your implementation
    notes_filtered = None


    # ==================== YOUR CODE HERE ====================
    # Validate required columns
    if "subject_id" not in notes.columns or "chartdate" not in notes.columns:
        raise ValueError("notes must contain 'subject_id' and 'chartdate'")
    required_labels = {"subject_id", "index_time", "label"}
    if not required_labels.issubset(shock_labels.columns):
        missing = required_labels - set(shock_labels.columns)
        raise ValueError(f"shock_labels missing required columns: {missing}")

    # Merge to attach each patient's index_time
    merged = pd.merge(
        notes,
        shock_labels[["subject_id", "index_time", "label"]],
        on="subject_id",
        how="inner",
    )

    # Ensure datetime types
    merged["chartdate"] = pd.to_datetime(merged["chartdate"], utc=True, errors="coerce")
    merged["index_time"] = pd.to_datetime(merged["index_time"], utc=True, errors="coerce")

    # Strictly before the day of index_time (exclude same-day notes)
    mask = merged["chartdate"].dt.date < merged["index_time"].dt.date
    notes_filtered = merged.loc[mask].drop(columns=["index_time", "label"]).reset_index(drop=True)
    
    # ==================== YOUR CODE HERE ====================
    
    return notes_filtered


def merge_snomed(
    snomed_ct_isaclosure: pd.DataFrame, snomed_ct_str_cui: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the snomed_ct_isaclosure and snomed_ct_str_cui dataframes to create a
    dataframe that contains a row for the CUIs and all of their descendant terms.
    In other words, the dataframe will contain 1 or more rows for each CUI such
    that each CUI is in a row with each of its descendant terms.

    EXAMPLE:
        - snomed_ct_isaclosure contains ancestor descendant pairs for each CUI
        - snomed_ct_str_cui contains the CUI and the term each CUI represents
        - The resulting dataframe should contain two columns: `cui` and `term`
            where the `cui` column represents the CUI and the `term` column
            represents a descendant term for the CUI. There should be one or more
            rows for each CUI, but each row (CUI, term) pair should be unique.


    IMPLEMENTATION INSTRUCTIONS:
        - Perform the appropriate merge operation to create a dataframe that
            contains a row for each descendant and the corresponding ancestor
            and term.
        - The resulting dataframe should have the following columns:
            - `cui`: represents the 'ancestor', which is the unique identifier
                    for each concept.
            - `term`: represents the 'str', which is the term for the concept.
        - Do not modify the input dataframes.

    Parameters:
        - snomed_ct_isaclosure: dataframe containing ancestor descendant pairs
            for each CUI
        - snomed_ct_str_cui: dataframe containing the CUI and the term each CUI
            represents

    Returns:
        - snomed_ct_concept_string: dataframe containing a row for each descendant
            and the corresponding ancestor and term
    """
    # Overwrite this variable with the return value in your implementation
    snomed_ct_concept_string = None


    # ==================== YOUR CODE HERE ====================
    # Validate required columns
    required_close = {"descendant", "ancestor"}
    if not required_close.issubset(snomed_ct_isaclosure.columns):
        missing = required_close - set(snomed_ct_isaclosure.columns)
        raise ValueError(f"snomed_ct_isaclosure missing required columns: {missing}")

    required_terms = {"CUI", "str"}
    if not required_terms.issubset(snomed_ct_str_cui.columns):
        missing = required_terms - set(snomed_ct_str_cui.columns)
        raise ValueError(f"snomed_ct_str_cui missing required columns: {missing}")

    # Map each descendant CUI to its term, keep ancestor as the concept
    merged = snomed_ct_isaclosure.merge(
        snomed_ct_str_cui[["CUI", "str"]], left_on="descendant", right_on="CUI", how="inner"
    )

    # Select and rename columns
    snomed_ct_concept_string = (
        merged[["ancestor", "str"]].rename(columns={"ancestor": "cui", "str": "term"})
    )
    
    # ==================== YOUR CODE HERE ====================
    

    # Return the dataframe
    return snomed_ct_concept_string


def get_cui_list(
    snomed_ct_concept_string: pd.DataFrame, cui: str, character_limit: int
):
    """
    Returns a list of terms that are descendants of the cui (passed in via the
    parameter) where the term string is shorter than or equal to the character
    limit. The returned list should be sorted primarily by the length of the term
    string in ascending order and then alphabetically for terms with the same length.

    IMPLEMENTATION INSTRUCTIONS:
        - Create a list of terms that are descendants of the CUI that are shorter
            than or equal to the character limit.
        - Return the list of terms sorted primarily by the length of the term
            string (in ascending order) and then alphabetically for terms with the
            same length.
        - Ensure that the returned list does not contain any duplicate terms.
        - Do not modify the input dataframe.

    Parameters:
        - snomed_ct_concept_string: dataframe containing a row for each descendant
            and the corresponding ancestor and term
        - cui: the cui to find descendants for
        - character_limit: the maximum number of characters allowed for a term

    Returns:
        - term_list: sorted list of terms that are descendants of the CUI that
            are shorter than or equal to the character limit
    """

    # Overwrite this variable with the return value in your implementation
    term_list = None


    # ==================== YOUR CODE HERE ====================
    # Validate required columns
    required_cols = {"cui", "term"}
    if not required_cols.issubset(snomed_ct_concept_string.columns):
        missing = required_cols - set(snomed_ct_concept_string.columns)
        raise ValueError(f"snomed_ct_concept_string missing required columns: {missing}")

    # Filter to the specified ancestor CUI and collect descendant terms
    terms = (
        snomed_ct_concept_string.loc[snomed_ct_concept_string["cui"] == cui, "term"]
        .dropna()
        .astype(str)
    )

    # Enforce character limit and uniqueness
    terms = terms[terms.str.len() <= int(character_limit)].drop_duplicates()

    # Sort by length (ascending) then alphabetically
    term_list = sorted(terms.tolist(), key=lambda s: (len(s), s))
    
    # ==================== YOUR CODE HERE ====================
    

    # Return the dataframe
    return term_list


def extract_terms(
    notes_filtered: pd.DataFrame,
    term_list: List[str],
    term_limit: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Determines if any of the terms in the term_list are found in the notes
    dataframe. Returns a dataframe containing all of the columns in the notes_filtered
    dataframe and an additional column terms in the term_list that indicate
    whether or not the term was found in the note with a boolean value.

    If the term_limit paramter `IS` None, then the function should use all terms
    in the term_list.

    If the term_limit parameter is `NOT` None, then the function should use the first
    <term_limit> terms in the term_list.


    IMPLEMENTATION INSTRUCTIONS:
        - Return a dataframe containing all of the columns in the notes_filtered
            dataframe and an additional column for each term in the term_list (
            up to term_limit if specified) that inddicates whether or not the
            term was found in the note with a boolean value.
        - Do not modify the input dataframe.

    Parameters:
        - notes_filtered: dataframe containing only notes that have a chartdate
            prior to the day of the index_time for the patient
        - term_list: list of terms to search for in the notes dataframe
        - term_limit: the maximum number of terms to use from the term_list. If
            None, then use all terms in the term_list. Default: None

    HINT:
        - Your implementation should utilize the str.contains method to determine
            if a term is in a note. This method can be used on a pandas series
            in a vectorized fashion.
        - We highly recommend that you create a temporary dataframe to store
            the columns for each term and then utilize the pd.concat method to
            add the columns to the nx_terms dataframe. This will be much faster
            than adding a column to the nx_terms dataframe at each iteration of
            the for loop.

    Returns:
        - nx_terms: dataframe containing all of the columns in the notes_filtered
            dataframe and an additional column for each term in the term_list that
            inddicates whether or not the term was found in the note with a boolean
            value
    """

    # Overwrite this variable with the return value in your implementation
    nx_terms = None


    # ==================== YOUR CODE HERE ====================
    # Validate inputs
    if "note_text" not in notes_filtered.columns:
        raise ValueError("notes_filtered must contain 'note_text' column")

    # Determine which terms to use (respect term_limit if provided)
    terms_to_use = term_list if term_limit is None else term_list[: int(term_limit)]
    # Remove duplicate terms while preserving order
    seen = set()
    unique_terms = []
    for t in terms_to_use:
        if t not in seen:
            unique_terms.append(t)
            seen.add(t)

    # Pre-compute lowercased note text for case-insensitive literal matching
    note_lower = notes_filtered["note_text"].astype(str).str.lower()

    # Build a temporary DataFrame with boolean matches for each term
    tmp_cols = {}
    for term in unique_terms:
        lt = str(term).lower()
        tmp_cols[term] = note_lower.str.contains(lt, regex=False, na=False)

    term_df = pd.DataFrame(tmp_cols, index=notes_filtered.index)

    # Concatenate with the original notes dataframe
    nx_terms = pd.concat([notes_filtered.reset_index(drop=True), term_df.reset_index(drop=True)], axis=1)
    
    # ==================== YOUR CODE HERE ====================
    

    # Return the resulting DataFrame and its dimensions
    return nx_terms


def normalize_terms(
    nx_terms: pd.DataFrame, snomed_ct_concept_string: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalizes the nx_terms dataframe by converting the term column to the CUI
    for each term. Returns a dataframe containing the `subject_id`, `chartdate`, and
    `concept` (CUI) corresponding to each True value in the nx_terms dataframe.

    IMPLEMENTATION INSTRUCTIONS:
        - Return a dataframe containing the `subject_id`, `chartdate`, and `concept`
            (CUI) corresponding to each True value in the nx_terms dataframe.
        - Do not modify the input dataframes.

    EXAMPLE:
        Lets say that the terms_df dataframe contains a row for a note where the
            column "heart attack" is True. If the CUI for "heart attack" is
            "C0018799", then the resulting dataframe should contain a row that
            contains the subject_id for the note, the chartdate for the note, and
            the CUI "C0018799".

    HINTS:
        - You can use the melt method to reshape the nx_terms dataframe from wide
            to long format. Check out the documentation for more information:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.melt.html

    Parameters:
        - nx_terms: dataframe containing all of the columns in the notes_filtered
            dataframe and an additional column for each term in the term_list that
            inddicates whether or not the term was found in the note with a boolean
            value
        - snomed_ct_concept_string: dataframe containing a row for each descendant
            and the corresponding ancestor and term

    Returns:
        - result_df: dataframe containing the `subject_id`, `chartdate`, and `concept`
            (CUI) corresponding to each True value in the nx_terms dataframe
    """

    # Overwrite this variable with the return value in your implementation
    result_df = None


    # ==================== YOUR CODE HERE ====================
    # Validate required columns
    if not {"subject_id", "chartdate"}.issubset(nx_terms.columns):
        raise ValueError("nx_terms must contain 'subject_id' and 'chartdate'")
    if not {"cui", "term"}.issubset(snomed_ct_concept_string.columns):
        raise ValueError("snomed_ct_concept_string must contain 'cui' and 'term'")

    # Identify boolean term columns (these were created in extract_terms)
    term_cols = [c for c in nx_terms.columns if nx_terms[c].dtype == bool]

    if len(term_cols) == 0:
        # No term columns; return empty frame with expected columns
        return pd.DataFrame(columns=["subject_id", "chartdate", "concept"])

    # Melt to long format: one row per (subject_id, chartdate, term)
    long_df = nx_terms.melt(
        id_vars=["subject_id", "chartdate"],
        value_vars=term_cols,
        var_name="term",
        value_name="present",
    )

    # Keep only rows where the term was present in the note
    long_df = long_df[long_df["present"]].drop(columns=["present"]).reset_index(drop=True)

    # Map each term string to its CUI via the concept string table
    merged = pd.merge(
        long_df,
        snomed_ct_concept_string[["cui", "term"]],
        on="term",
        how="left",
    )

    # Select and rename to requested schema
    result_df = merged[["subject_id", "chartdate", "cui"]].rename(columns={"cui": "concept"})
    # Drop rows where concept could not be mapped
    result_df = result_df.dropna(subset=["concept"]).reset_index(drop=True)
    
    # ==================== YOUR CODE HERE ====================
    

    return result_df


def get_note_concept_features(concept_df: pd.DataFrame):
    """
    Creates a patient level feature matrix that summarizes which concepts were
    found in each note as a binary feature matrix. Each row represents a patient
    and each column represents a concept. A value of 1 indicates that the concept
    was found in at least one note for the patient and a value of 0 indicates that
    the concept was not found in any notes for the patient.

    EXAMPLE:
        If concept_df contains rows that link the subject_id 1 to the ONLY the
        concepts C0018799 and C0007222, then the resulting dataframe should contain
        a row for subject_id 1 with a 1 in the columns for C0018799 and C0007222
        and 0 in all other concept columns.


    HINT: Your implemetation should utilize the pivot_table method, which has a
    argument that allows you to specify an aggregation function to use.

    IMPLEMENTATION INSTRUCTIONS:
        - Return a dataframe containing the patient level feature matrix that
            summarizes which concepts were found in each note as a binary feature
            matrix.
        - Do not modify the input dataframe.

    Parameters:
        - concept_df: dataframe containing the `subject_id`, `chartdate`, and `concept`
            (CUI) corresponding to each True value in the nx_terms dataframe

    Returns:
        - concept_features: dataframe containing the patient level feature matrix
            that summarizes which concepts were found for each patient as a binary
            feature matrix
    """

    # Overwrite this variable with the return value in your implementation
    concept_features = None


    # ==================== YOUR CODE HERE ====================
    # Validate inputs
    required_cols = {"subject_id", "concept"}
    if not required_cols.issubset(concept_df.columns):
        missing = required_cols - set(concept_df.columns)
        raise ValueError(f"concept_df missing required columns: {missing}")

    # Add indicator column and pivot to wide binary matrix per subject
    temp = concept_df.copy()
    temp["present"] = 1
    pivot = temp.pivot_table(
        index="subject_id",
        columns="concept",
        values="present",
        aggfunc="max",
        fill_value=0,
    )

    concept_features = pivot.reset_index()
    # Ensure integer dtype for binary columns
    for col in concept_features.columns:
        if col != "subject_id":
            concept_features[col] = concept_features[col].astype(int)
    # ==================== YOUR CODE HERE ====================
    

    return concept_features
