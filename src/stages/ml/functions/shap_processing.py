# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for SHAP (SHapley Additive exPlanations) model interpretation.

This module provides a function to compute SHAP values using the `shap` library.
It supports different types of SHAP explainers and returns the computed SHAP values,
legacy SHAP values, and the expected value from the explainer.

Imports
--------
pandas as pd
shap : Explainer, explainers
"""

import pandas as pd
from shap import Explainer, explainers


def shap_explain(model, x: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for a given model and dataset.

    This function uses the `shap` library to compute SHAP values based on the provided
    model and dataset. It determines the type of explainer to use and computes both
    standard and legacy SHAP values. It also retrieves the expected value from the explainer.

    Parameters
    ----------
    model : object
        The model to be explained. It should be compatible with the SHAP library's
        explainers (e.g., a scikit-learn model).
    x : pd.DataFrame
        The input features for which to compute the SHAP values. This should be a
        DataFrame with the same features used to train the model.

    Returns
    -------
    tuple
        A tuple containing:
        - shap_values : shap.Explanation
            The computed SHAP values for the input features.
        - shap_values_legacy : shap.Explanation
            The legacy SHAP values for the input features.
        - expected_value : array-like
            The expected value from the explainer, which represents the base value for
            the SHAP explanations.
    """
    explainer = Explainer(model, x)
    if explainer.__class__ == explainers.TreeExplainer:
        shap_values = explainer(x, check_additivity=False)
        shap_values_legacy = explainer.shap_values(x, check_additivity=False)
    else:
        shap_values = explainer(x)
        shap_values_legacy = explainer.shap_values(x)
    expected_value = explainer.expected_value
    return shap_values, shap_values_legacy, expected_value
