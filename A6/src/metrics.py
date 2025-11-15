"""
metrics.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""


import pandas as pd
import numpy as np


def accuracy(test_labels: np.ndarray, test_predictions: np.ndarray) -> float:
    """
    Calculates the accuracy of the model's predictions given the true labels.

    Accuracy is defined as the fraction of predictions that are correct.
        acc = correct / total

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - You may use numpy to calculate the accuracy.

    Parameters:
        test_labels (np.ndarray): array containing the true labels
        test_predictions (np.ndarray): array containing the predicted labels

    Returns:
        float: accuracy of the model's predictions
    """

    # Overwrite this variable in your implementation
    acc = None

    # ==================== YOUR CODE HERE ====================
    labels = np.asarray(test_labels).ravel()
    preds = np.asarray(test_predictions).ravel()

    if labels.size == 0:
        acc = 0.0
    else:
        acc = float(np.mean(labels == preds))
    # ==================== YOUR CODE HERE ====================
    
    return acc


def confusion_matrix(
    test_labels: np.ndarray, test_predictions: np.ndarray
) -> np.ndarray:
    """
    Calculates the confusion matrix for the model's predictions given the true
    labels and returns the resulting 2x2 numpy array.

    The confusion matrix is defined as follows:
        - The rows correspond to the true labels
        - The columns correspond to the predicted labels
        - The entry at position (i, j) corresponds to the number of samples with
            true label i and predicted label j.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - Return a 2x2 numpy array containing the confusion matrix as specified above.

    Parameters:
        test_labels (np.ndarray): array containing the true labels
        test_predictions (np.ndarray): array containing the predicted labels

    Returns:
        np.ndarray: confusion matrix for the model's predictions
    """

    # Overwrite this variable in your implementation
    con_mat = np.zeros((2, 2))


    # ==================== YOUR CODE HERE ====================
    y_true = np.asarray(test_labels).ravel()
    y_pred = np.asarray(test_predictions).ravel()

    # Compute counts
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    con_mat[0, 0] = tn  # true 0, pred 0
    con_mat[0, 1] = fp  # true 0, pred 1
    con_mat[1, 0] = fn  # true 1, pred 0
    con_mat[1, 1] = tp  # true 1, pred 1
    # ==================== YOUR CODE HERE ====================
    

    return con_mat


def sensitivity(con_mat: np.ndarray) -> float:
    """
    Calculates the sensitivity of the model's predictions given the confusion matrix.

    Sensitivity is defined as the fraction of positive samples that were correctly
    identified.
        sens = TP / (TP + FN)

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - Your implementation should return a 0 if the denominator is 0.
            - Ensure that no warning is raised in this case.

    Parameters:
        con_mat (np.ndarray): confusion matrix for the model's predictions

    Returns:
        float: sensitivity of the model's predictions
    """

    # Overwrite this variable in your implementation
    sens = None


    # ==================== YOUR CODE HERE ====================
    tp = float(con_mat[1, 1])
    fn = float(con_mat[1, 0])
    denom = tp + fn
    sens = 0.0 if denom == 0 else tp / denom
    # ==================== YOUR CODE HERE ====================
    

    return sens


def specificity(con_mat: np.ndarray) -> float:
    """
    Calculates the specificity of the model's predictions given the confusion matrix.

    Specificity is defined as the fraction of negative samples that were correctly
    identified.
        spec = TN / (TN + FP)

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - Your implementation should return a 0 if the denominator is 0.
            - Ensure that no warning is raised in this case.

    Parameters:
        con_mat (np.ndarray): confusion matrix for the model's predictions

    Returns:
        float: specificity of the model's predictions
    """

    # Overwrite this variable in your implementation
    spec = None


    # ==================== YOUR CODE HERE ====================
    tn = float(con_mat[0, 0])
    fp = float(con_mat[0, 1])
    denom = tn + fp
    spec = 0.0 if denom == 0 else tn / denom
    # ==================== YOUR CODE HERE ====================
    

    return spec


def precision(con_mat: np.ndarray) -> float:
    """
    Calculates the precision of the model's predictions given the confusion matrix.

    Precision is defined as the fraction of positive predictions that were correct.
        prec = TP / (TP + FP)

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - Your implementation should return a 0 if the denominator is 0.
            - Ensure that no warning is raised in this case.

    Parameters:
        con_mat (np.ndarray): confusion matrix for the model's predictions

    Returns:
        float: precision of the model's predictions
    """

    # Overwrite this variable in your implementation
    prec = None


    # ==================== YOUR CODE HERE ====================
    tp = float(con_mat[1, 1])
    fp = float(con_mat[0, 1])
    denom = tp + fp
    prec = 0.0 if denom == 0 else tp / denom
    # ==================== YOUR CODE HERE ====================
    

    return prec
