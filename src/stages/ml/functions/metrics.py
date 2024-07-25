import pandas as pd
import numpy as np
from qtpy.QtGui import QTextCursor
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                             log_loss)

def metrics_estimation(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                       y_test: pd.Series) -> pd.DataFrame:
    """Generating tables with metrics for classification.

    Parameters
    ----------
    model: sklearn clf estimator
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    Returns
    -------
    df: pd.DataFrame
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
    return df


def get_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, y_score: np.ndarray,
                name: str) -> pd.DataFrame:
    """
    Generating tables with metrics for classification.

    Parameters
    ----------
    y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    y_score: array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

    name: str
        id of metrics

    Returns
    -------
    df: pd.DataFrame
    """
    df_metrics = pd.DataFrame()
    df_metrics['model'] = [name]
    df_metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    df_metrics['ROC_AUC'] = roc_auc_score(y_true, y_score if len(y_score.shape) == 1 else y_score[:, 1])
    df_metrics['Precision'] = precision_score(y_true, y_pred)
    df_metrics['Recall'] = recall_score(y_true, y_pred)
    df_metrics['F1'] = f1_score(y_true, y_pred)
    df_metrics['Logloss'] = log_loss(y_true, y_score)
    return df_metrics


def lda_coef_equation(model) -> str:
    """
    Формирует уравнение LDA вида
        a = c1*x1 + c2*x2 ... cn*xn
        где cn - коэффиент, xn - признак
    Коэффициенты отсортированы по возрастанию по абсолютной величине.

    Parameters
    ----------
    model : LDA model

    Returns
    -------
    out : str
        Equation
    """
    model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
    for coef in model.coef_:
        feature_names = model.feature_names_in_
        name_coef = {}
        for j, n in enumerate(feature_names):
            name_coef[n] = coef[j]
        name_coef = sorted(name_coef.items(), key=lambda x: abs(x[1]), reverse=True)
        eq_text = f'LD-{coef + 1} ='
        for feature_name, coef_value in name_coef:
            eq_text += f'+ {coef_value} * x({feature_name}) '
        return eq_text


def parse_table(report: str) -> tuple[list, list]:
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
    y_pred = model.predict(x_test)
    c_r = classification_report(y_test, y_pred)
    c_r_parsed = parse_table(c_r)
    metrics = metrics_estimation(model, x_train, y_train, x_test, y_test)
    metrics_parsed = parse_table(metrics.T.to_string())
    metrics_parsed[0].remove(' ')
    return {'c_r_parsed': c_r_parsed, 'metrics_parsed': metrics_parsed}
