"""
curve.py

This file contains functions that you must implement.

IMPORTANT INSTRUCTIONS:
    - Do NOT modify the function signatures of the functions in this file.
    - Only make changes inside the specified locations for your implementation.
    - You may add additional helper functions if you wish.
    - Do NOT import anything other than what is already imported below.
"""

import numpy as np
from src.metrics import precision, sensitivity, specificity, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Union
from sklearn.calibration import calibration_curve


def calc_roc_curve(
    test_labels: np.ndarray,
    test_probabilites: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the TPR and FPR values that up the ROC curve for a model's predictions
    given the true labels and predicted probabilites. Returns two numpy arrays
    containing the FPR and TPR values for the ROC curve.

    TPR is the true positive rate, also known as sensitivity or recall.

    FPR is the false positive rate, which is defined as 1 - specificity.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - Your implementation should calculate the FPR and TPR for 1000 equally
            spaced threshold values between 0 and 1.
            - HINT: You may find the np.linspace function useful.
            - HINT: You may find it helpful to use some functions you implemented
                in the metrics.py file! They have been imported for you.
        - Return the resulting FPR and TPR values as numpy arrays.
            - The FPR and TPR values should be ordered in increasing order of the
                threshold values. (So the FPRs and TPRs corresponding to the
                smallest threshold values should be at the beginning of the arrays,
                and the FPRs and TPRs corresponding to the largest threshold values
                should be at the end of the arrays.)

    Parameters:
        test_labels (np.ndarray): array containing the true labels
        test_probabilities (np.ndarray): array containing the probabilites of each example
        filename (str): filename to save the plot to

    Returns:
        np.ndarray: array containing the FPR values
        np.ndarray: array containing the TPR values
    """

    # Overwrite this variable in your implementation
    fprs, tprs = None, None


    # ==================== YOUR CODE HERE ====================
    labels = np.asarray(test_labels).ravel()
    probs = np.asarray(test_probabilites).ravel()

    thresholds = np.linspace(0.0, 1.0, 1000)
    fpr_list = []
    tpr_list = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        con_mat = confusion_matrix(labels, preds)
        tpr_val = sensitivity(con_mat)
        fpr_val = 1.0 - specificity(con_mat)
        tpr_list.append(tpr_val)
        fpr_list.append(fpr_val)

    fprs = np.asarray(fpr_list)
    tprs = np.asarray(tpr_list)
    # ==================== YOUR CODE HERE ====================
    
    return fprs, tprs


def display_roc_curve(
    fpr_tpr_list: List[Tuple[np.ndarray, np.ndarray]],
    model_name_list: List[str],
    filename: Union[str, None] = None,
):
    """
    Displays a plot that contains a ROC curve for each pair of FPR and TPR arrays
    in the fpr_tpr_list. Creates a png file of the plot.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - For each pair of FPR and TPR arrays in the fpr_tpr_list, plot the ROC
            on the same plot with a different color.
        - Using seaborn, your ROC plot should contain the following attributes:
            - The x-axis should be the FPR
            - The y-axis should be the TPR
            - The plot should also contain a diagonal line to represent random
                performance.
            - A legend should be included to indicate which ROC curve corresponds
                to which model.
        - Save the plot to a png file with the given filename if it is not None.

    Parameters:
        fpr_tpr_list (List[np.ndarray, np.ndarray]): list of tuples containing the FPR
            and TPR values for each model
        model_name_list (List[str]): list of strings containing the names of each model
            (should be used as labels in the legend)
    """
    # Plot the ROC curve using seaborn
    sns.set()
    plt.figure()


    # ==================== YOUR CODE HERE ====================
    # Plot each ROC curve
    for (fprs, tprs), name in zip(fpr_tpr_list, model_name_list):
        sns.lineplot(x=fprs, y=tprs, label=name)

    # Diagonal random baseline
    sns.lineplot(x=[0, 1], y=[0, 1], label="Random", linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    # Show the plot for interactive contexts
    plt.show()
    # ==================== YOUR CODE HERE ====================
    


def calc_precision_recall_curve(
    test_labels: np.ndarray,
    test_probabilites: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the precision-recall curve for the model's predictions given the true
    labels and predicted probabilites. Returns two numpy arrays containing the precision
    and recall values for the curve.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - Your implementation should calculate the precision and recall for 1000
            equally spaced threshold values between 0 and 1.
            - HINT: You may find the np.linspace function useful.
            - HINT: You may find it helpful to use some functions you implemented
                in the metrics.py file! They have been imported for you.
        - Return the resulting precision and recall values as numpy arrays.
            - The recall and precision values should be ordered in increasing order of the
                threshold values. (So the recall and precision corresponding to the
                smallest threshold values should be at the beginning of the arrays,
                and the recall and precision corresponding to the largest threshold values
                should be at the end of the arrays.)

    Parameters:
        test_labels (np.ndarray): array containing the true labels
        test_probabilities (np.ndarray): array containing the probabilites of each example

    Return:
        np.ndarray: array containing the precision values
        np.ndarray: array containing the recall values
    """

    # Overwrite this variable in your implementation
    precs, recalls = None, None


    # ==================== YOUR CODE HERE ====================
    labels = np.asarray(test_labels).ravel()
    probs = np.asarray(test_probabilites).ravel()

    thresholds = np.linspace(0.0, 1.0, 1000)
    prec_list = []
    rec_list = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        con_mat = confusion_matrix(labels, preds)
        prec_val = precision(con_mat)
        rec_val = sensitivity(con_mat)  # recall
        prec_list.append(prec_val)
        rec_list.append(rec_val)

    precs = np.asarray(prec_list)
    recalls = np.asarray(rec_list)
    # ==================== YOUR CODE HERE ====================
    

    return precs, recalls


def display_precision_recall_curve(
    p_r_list: List[Tuple[np.ndarray, np.ndarray]],
    model_name_list: List[str],
    filename: Union[str, None] = None,
):
    """
    Displays a plot that contains a Precision Recall curve for each pair of precision
    and recall arrays in p_r_list. Creates a png file of the plot.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - For each pair of precision and recall arrays in the p_r_list, plot the
            precision recall curve on the same plot with a different color.
        - Using seaborn, your precision recall plot should adhere to the following:
            - The x-axis should be the recall
            - The y-axis should be the precision
            - A legend should be included to indicate which PR curve corresponds
                to which model using the names in model_name_list.
        - Save the plot to a png file with the given filename if it is not None.

    Parameters:
        p_r_list (List[np.ndarray, np.ndarray]): list of tuples containing
            the precision and recall values for each model
        model_name_list (List[str]): list of strings containing the names of each model
            (should be used as labels in the legend)
        filename (str): filename to save the plot to
    """
    sns.set()
    plt.figure()


    # ==================== YOUR CODE HERE ====================
    for (precs, recalls), name in zip(p_r_list, model_name_list):
        sns.lineplot(x=recalls, y=precs, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
    # ==================== YOUR CODE HERE ====================
    


def display_calibration_plot(
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    filename: Union[str, None] = None,
):
    """
    Displays a calibration plot and saves it as a png with filename.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below where indicated
        - Your implementation should use the calibration_curve function from
            sklearn.calibration (imported for you).
            - Use the parameters:
                - n_bins=10
                - strategy='uniform'
        - Create the plot using matplotlib.pyplot (imported for you)
            by the parameters
            - Add appropriate X and Y labels to your plot. (NOTE: What are the
                axis of a calibration plot?)
            - Add a line to represent what a perfectly calibrated model would
                look like on this plot.
        - Save the plot to a png file with the given filename if it is not None.


    Parameters:
        test_labels (np.ndarray): array containing the true labels
        test_probabilities (np.ndarray): array containing the probabilites of each example
        filename (str): filename to save the plot to
    """
    # Set the figure size
    plt.figure(figsize=(4, 4))


    # ==================== YOUR CODE HERE ====================
    y_true = np.asarray(test_labels).ravel()
    y_prob = np.asarray(test_probabilities).ravel()

    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    # calibration_curve returns (fraction_of_positives, mean_predicted_value)
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot")
    plt.legend()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
    # ==================== YOUR CODE HERE ====================
    
