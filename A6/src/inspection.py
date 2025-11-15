from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from typing import List, Tuple, Union


def display_partial_dependence(
    model: GradientBoostingClassifier,
    features: pd.DataFrame,
    feature_names: Union[List[str], List[Tuple[str, str]]],
    filename: Union[str, None] = None,
):
    """
    Display the partial dependence plots for a model.

    This function uses the PartialDependenceDisplay function from sklearn to
    display the partial dependence plots for a model. The partial dependence
    plots show the relationship between the features and the predicted
    probability of death in the hospital.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - You should only need to write a few lines of code in this function.
            - Hint: Look at the documentation for PartialDependenceDisplay:
                https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html
        - Do NOT modify the input dataframes.

    Parameters:
        model: The model to use to generate the partial dependence plots
        features: The features to use to generate the partial dependence plots
        feature_names: The names of the features to use to generate the partial
            dependence plots. This should be a list of strings or a list of tuples
            of strings.
        filename: The name of the file to save the plot to. If this is None, the
            plot will not be saved to a file.
    """


    # ==================== YOUR CODE HERE ====================
    sns.set()
    plt.figure()
    PartialDependenceDisplay.from_estimator(
        model,
        X=features,
        features=feature_names,
        kind="average",
    )
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    # ==================== YOUR CODE HERE ====================
    

    plt.show()


def display_age_distribution(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    filename=None,
):
    """
    Display the distribution of ages in the dataset split by whether or not
    the patient died in the hospital.

    IMPLEMENTATION INSTRUCTIONS:
        - Implement the function below in the specified location.
        - You should only need to write a few lines of code in this function.
            - Hint: Look at the documentation for seaborn:
                https://seaborn.pydata.org/generated/seaborn.histplot.html
        - Create a histogram of the ages of the patients in the dataset split
            by whether or not the patient died in the hospital.
            - Feel free to add additional parameters to the histogram if you
                wish!
        - Save the plot to the file specified by `filename` if `filename` is
            not None.
        - Do NOT modify the input dataframes.

    Parameters:
        features: The features to use to generate the partial dependence plots
        labels: The labels to use to generate the partial dependence plots
        filename: The name of the file to save the plot to. If this is None, the
            plot will not be saved to a file.
    """


    # ==================== YOUR CODE HERE ====================
    sns.set()
    plt.figure()

    # Build a small dataframe with age and label for plotting
    plot_df = pd.DataFrame(
        {
            "age_in_days": features["age_in_days"],
            "died": labels["death_in_stay"],
        }
    )

    sns.histplot(
        data=plot_df,
        x="age_in_days",
        hue="died",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.4,
    )
    plt.xlabel("Age (days)")
    plt.ylabel("Density")
    plt.title("Age distribution by in-hospital death")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    # ==================== YOUR CODE HERE ====================
    

    plt.show()
