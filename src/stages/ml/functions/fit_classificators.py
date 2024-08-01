# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functionality for fitting various machine learning models,
performing dimensionality reduction, and evaluating model performance.

Functions
---------
- `scorer_metrics()`: Returns a dictionary of custom scoring metrics for model evaluation.
- `fit_lda(x_train, y_train, best_params, rnd)`: Fits a Linear Discriminant Analysis model.
- `fit_lr(x_train, y_train, best_params, rnd)`: Fits a Logistic Regression model.
- `fit_svc(x_train, y_train, best_params, rnd)`: Fits a Support Vector Classifier model.
- `fit_dt(x_train, y_train, best_params, rnd)`: Fits a Decision Tree Classifier model.
- `fit_rf(x_train, y_train, best_params, rnd)`: Fits a Random Forest Classifier model.
- `fit_xgboost(x_train, y_train, best_params, rnd)`: Fits an XGBoost Classifier model.
- `fit_pca(x_train, y_train, x_test, y_test)`: Applies PCA for dimensionality reduction.
- `clf_predict(item, x)`: Makes predictions using a provided classifier model.
- `dim_reduction(x_train, x_test, y_train)`: Applies PCA for dimensionality reduction on training
    and test data.
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, \
    log_loss, roc_auc_score
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier


def scorer_metrics() -> dict:
    """
    Returns a dictionary of custom scoring metrics for model evaluation.

    Returns
    -------
    dict
        A dictionary where keys are metric names and values are sklearn `make_scorer` objects.
        Includes 'precision_score', 'recall_score', 'accuracy_score', 'f1_score', 'auc', and
        'log_loss'.
    """
    return {'precision_score': make_scorer(precision_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro'),
            'accuracy_score': make_scorer(accuracy_score, average='micro'),
            'f1_score': make_scorer(f1_score, average='micro'),
            'auc': make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
            'log_loss': make_scorer(log_loss, average='micro'),
            }


def fit_lda(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, rnd: int):
    """
    Fits a Linear Discriminant Analysis (LDA) model to the training data.

    Parameters
    ----------
    x_train : pd.DataFrame
        The features of the training data.
    y_train : pd.Series
        The target values for the training data.
    best_params : dict
        Parameters to initialize the LDA model.
    rnd : int
        Random seed for reproducibility.

    Returns
    -------
    LinearDiscriminantAnalysis
        The fitted LDA model.
    """
    model = LinearDiscriminantAnalysis(**best_params)
    model.fit(x_train, y_train)
    return model


def fit_lr(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, rnd: int):
    """
    Fits a Logistic Regression model to the training data.

    Parameters
    ----------
    x_train : pd.DataFrame
        The features of the training data.
    y_train : pd.Series
        The target values for the training data.
    best_params : dict
        Parameters to initialize the Logistic Regression model.
    rnd : int
        Random seed for reproducibility.

    Returns
    -------
    LogisticRegression
        The fitted Logistic Regression model.
    """
    params = {}
    for key, value in best_params.items():
        if 'solver' in key:
            key = 'solver'
        params[key] = value
    model = LogisticRegression(class_weight='balanced', random_state=rnd, max_iter=1000,
                               **params)
    model.fit(x_train, y_train)
    return model


def fit_svc(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, rnd: int):
    """
    Fits a Support Vector Classifier (SVC) model to the training data.

    Parameters
    ----------
    x_train : pd.DataFrame
        The features of the training data.
    y_train : pd.Series
        The target values for the training data.
    best_params : dict
        Parameters to initialize the SVC model.
    rnd : int
        Random seed for reproducibility.

    Returns
    -------
    SVC
        The fitted SVC model.
    """
    model = SVC(class_weight='balanced', random_state=rnd, probability=True, kernel='linear',
                **best_params)
    model.fit(x_train, y_train)
    return model


def fit_dt(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, rnd: int):
    """
    Fits a Decision Tree Classifier model to the training data.

    Parameters
    ----------
    x_train : pd.DataFrame
        The features of the training data.
    y_train : pd.Series
        The target values for the training data.
    best_params : dict
        Parameters to initialize the Decision Tree model.
    rnd : int
        Random seed for reproducibility.

    Returns
    -------
    DecisionTreeClassifier
        The fitted Decision Tree Classifier model.
    """
    model = DecisionTreeClassifier(class_weight='balanced', random_state=rnd, **best_params)
    model.fit(x_train, y_train)
    return model


def fit_rf(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, rnd: int):
    """
    Fits a Random Forest Classifier model to the training data.

    Parameters
    ----------
    x_train : pd.DataFrame
        The features of the training data.
    y_train : pd.Series
        The target values for the training data.
    best_params : dict
        Parameters to initialize the Random Forest model.
    rnd : int
        Random seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        The fitted Random Forest Classifier model.
    """
    model = RandomForestClassifier(class_weight='balanced', random_state=rnd, **best_params)
    model.fit(x_train, y_train)
    return model


def fit_xgboost(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, rnd: int):
    """
    Fits an XGBoost Classifier model to the training data.

    Parameters
    ----------
    x_train : pd.DataFrame
        The features of the training data.
    y_train : pd.Series
        The target values for the training data.
    best_params : dict
        Parameters to initialize the XGBoost model.
    rnd : int
        Random seed for reproducibility.

    Returns
    -------
    XGBClassifier
        The fitted XGBoost Classifier model.
    """
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    model = XGBClassifier(
        eval_metric="auc",
        random_state=rnd,
        scale_pos_weight=class_weight_dict[0],
        enable_categorical=True,
        device="cpu",
        **best_params
    )
    model.fit(x_train, y_train)
    return model


def fit_pca(x_train: pd.DataFrame, y_train: list[int], x_test: pd.DataFrame, y_test: list[int]) \
        -> dict:
    """
    Applies Principal Component Analysis (PCA) to the combined training and test data.

    Parameters
    ----------
    x_train : pd.DataFrame
        The features of the training data.
    y_train : list of int
        The target values for the training data.
    x_test : pd.DataFrame
        The features of the test data.
    y_test : list of int
        The target values for the test data.

    Returns
    -------
    dict
        A dictionary containing:
            - 'model': The fitted PCA model.
            - 'features_2d': Transformed features in 2D space.
            - 'y_train_test': Combined target values from training and test sets.
            - 'explained_variance_ratio': The amount of variance explained by each PCA component.
    """
    model = PCA(n_components=2)
    x_data = np.concatenate((x_train.values, x_test.values))
    y_data = np.concatenate((y_train, y_test))
    model.fit(x_data, y_data)
    features_2d = scale(model.transform(x_data))
    return {'model': model, 'features_2d': features_2d, 'y_train_test': y_data,
            'explained_variance_ratio': model.explained_variance_ratio_}


def clf_predict(item: tuple, x: pd.DataFrame) -> dict:
    """
    Makes predictions using a provided classifier model.

    Parameters
    ----------
    item : tuple
        A tuple where the first element is the classifier model and the second is the classifier's
        name.
    x : pd.DataFrame
        The features of the data to predict.

    Returns
    -------
    dict
        A dictionary containing:
            - 'y_pred': Predicted labels.
            - 'y_score': Prediction probabilities.
            - 'clf_name': The name of the classifier used.
    """
    model, clf_name = item
    y_pred = model.predict(x)
    y_score = model.predict_proba(x)
    return {'y_pred': y_pred, 'y_score': y_score, 'clf_name': clf_name}


def dim_reduction(x_train, x_test, y_train):
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality of the training and test
    data.

    Parameters
    ----------
    x_train : array-like
        The features of the training data. Can be either a DataFrame or numpy array.
    x_test : array-like
        The features of the test data. Can be either a DataFrame or numpy array.
    y_train : array-like
        The target values for the training data.

    Returns
    -------
    tuple
        A tuple containing:
            - transformed_2d: The training data transformed into 2D space.
            - features_in_2d: Combined training and test data transformed into 2D space.
            - explained_variance_ratio: The amount of variance explained by each PCA component.
    """
    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.values
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values
    pca = PCA(n_components=2)
    pca.fit(x_train, y_train)
    transformed_2d = scale(pca.transform(x_train))
    features_in_2d = scale(pca.transform(np.concatenate((x_train, x_test))))

    return transformed_2d, features_in_2d, pca.explained_variance_ratio_
