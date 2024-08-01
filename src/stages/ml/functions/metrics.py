# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for evaluating and formatting classification model performance metrics.

This module provides functions for calculating classification metrics, parsing
metric reports, inserting tables into text edits, and creating fit data including
metrics, model predictions, and transformed feature spaces.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas import DataFrame
from qtpy.QtGui import QTextCursor
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                             log_loss)
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from src.stages import dim_reduction


def metrics_estimation(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                       y_test: pd.Series) -> tuple[DataFrame, float, float, float, float]:
    """
    Generate a DataFrame with metrics for classification models, including accuracy, ROC AUC,
    precision, recall, F1 score, and log loss.

    Parameters
    ----------
    model : sklearn classifier
        The classification model to be evaluated.
    x_train : pd.DataFrame
        Features of the training dataset.
    y_train : pd.Series
        True labels of the training dataset.
    x_test : pd.DataFrame
        Features of the test dataset.
    y_test : pd.Series
        True labels of the test dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the metrics for both training and test datasets.
    pd.Series
        Predicted labels for the training dataset.
    np.ndarray
        Predicted probabilities for the training dataset.
    pd.Series
        Predicted labels for the test dataset.
    np.ndarray
        Predicted probabilities for the test dataset.
    """
    y_pred_train = model.predict(x_train)
    y_score_train = model.predict_proba(x_train)
    y_pred_test = model.predict(x_test)
    y_score_test = model.predict_proba(x_test)

    df_train = get_metrics(y_train, y_pred_train, y_score_train, 'Train')
    df_test = get_metrics(y_test, y_pred_test, y_score_test, 'Test')
    df = pd.concat([df_train, df_test])
    df.set_index('model', inplace=True)
    auc_train = df.loc['Train']['ROC_AUC']
    auc_test = df.loc['Test']['ROC_AUC']
    df['overfitting, %'] = abs(auc_train - auc_test) / auc_test * 100
    return df, y_pred_train, y_score_train, y_pred_test, y_score_test


def get_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, y_score: np.ndarray,
                name: str) -> pd.DataFrame:
    """
    Compute and return classification metrics for a given set of true labels and predictions.

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels from the classifier.
    y_score : np.ndarray
        Predicted probabilities or decision function scores.
    name : str
        Identifier for the metrics (e.g., 'Train' or 'Test').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed metrics.
    """
    binary = len(np.unique(y_true)) == 2
    average_func = 'binary' if binary else 'micro'
    df_metrics = pd.DataFrame()
    df_metrics['model'] = [name]
    df_metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    df_metrics['ROC_AUC'] = roc_auc_score(
        y_true, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
        multi_class='ovr', average='micro')
    df_metrics['Precision'] = precision_score(y_true, y_pred, average=average_func)
    df_metrics['Recall'] = recall_score(y_true, y_pred, average=average_func)
    df_metrics['F1'] = f1_score(y_true, y_pred, average=average_func)
    df_metrics['Logloss'] = log_loss(y_true, y_score)
    return df_metrics

def parse_table(report: str) -> tuple[list, list]:
    """
    Parse a classification report into headers and rows suitable for table insertion.

    Parameters
    ----------
    report : str
        The classification report as a string.

    Returns
    -------
    tuple of (list, list)
        A tuple containing:
        - A list of headers for the table.
        - A list of rows, where each row is a list of cell values.
    """
    headers = [' ']
    rows = []
    for i in report.split('\n')[0].strip().split(' '):
        if i != '':
            headers.append(i)
    for i, r in enumerate(report.split('\n')):
        new_row = []
        if r == '' or i == 0:
            continue
        rr = r.split('  ')
        for c in rr:
            if c == '':
                continue
            new_row.append(c)
        if new_row[0].strip() == 'accuracy':
            new_row = [new_row[0], '', '', new_row[1], new_row[2]]
        rows.append(new_row)
    return headers, rows


def insert_table_to_text_edit(cursor, headers, rows) -> None:
    """
    Insert a table with the given headers and rows into a QTextCursor.

    Parameters
    ----------
    cursor : QTextCursor
        The cursor into which the table will be inserted.
    headers : list
        The headers of the table.
    rows : list
        The rows of the table.
    """
    cursor.insertTable(len(rows) + 1, len(headers))
    for header in headers:
        cursor.insertText(header)
        cursor.movePosition(QTextCursor.NextCell)
    for row in rows:
        for value in row:
            cursor.insertText(str(value))
            cursor.movePosition(QTextCursor.NextCell)


def create_fit_data(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                    y_test: pd.Series) -> dict:
    """
    Generate a dictionary with fit data, including metrics, predictions, and transformed features.

    Parameters
    ----------
    model : sklearn classifier
        The classification model to be fitted and evaluated.
    x_train : pd.DataFrame
        Features of the training dataset.
    y_train : pd.Series
        True labels of the training dataset.
    x_test : pd.DataFrame
        Features of the test dataset.
    y_test : pd.Series
        True labels of the test dataset.

    Returns
    -------
    dict
        A dictionary containing:
        - Metrics DataFrame and parsed classification report.
        - Predictions and probabilities for both training and test datasets.
        - Transformed feature spaces if applicable.
    """
    result = {}
    metrics, y_pred_train, y_score_train, y_pred_test, result['y_score_test'] \
        = metrics_estimation(model, x_train, y_train, x_test, y_test)

    result['c_r_parsed'] = parse_table(classification_report(y_test, y_pred_test))
    metrics_parsed = parse_table(metrics.T.to_string())
    metrics_parsed[0].remove(' ')
    result['metrics_parsed'] = metrics_parsed

    y_pred_train_test = np.concatenate((y_pred_train, y_pred_test))
    result['misclassified'] = np.concatenate((y_train, y_test)) != y_pred_train_test
    if 'decision_function' in model.__dir__():
        result['y_score_decision'] = model.decision_function(x_test)
    else:
        result['y_score_decision'] = result['y_score_test']
    if 'transform' in model.__dir__():
        transformed_2d = model.transform(x_train)
        if transformed_2d.shape[1] > 1:
            transformed_2d = transformed_2d[:, [0, 1]]
        transformed_2d_test = model.transform(x_test)
        if transformed_2d_test.shape[1] > 1:
            transformed_2d_test = transformed_2d_test[:, [0, 1]]
        result['features_2d'] = np.concatenate((transformed_2d, transformed_2d_test))
    else:
        (transformed_2d, result['features_2d'],
         result['explained_variance_ratio']) = dim_reduction(x_train, x_test, y_train)
    model_2d = deepcopy(model)
    model_2d.fit(transformed_2d, y_train)
    result['model_2d'] = model_2d
    result['y_pred_2d'] = model_2d.predict(result['features_2d'])
    label_binarizer = LabelBinarizer().fit(y_train)
    result['y_onehot_test'] = label_binarizer.transform(y_test)
    return result
