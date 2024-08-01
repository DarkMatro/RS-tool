# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for optimization of machine learning models using Optuna.

This module provides functions to perform hyperparameter optimization for various
machine learning models including Logistic Regression, SVC, Decision Tree, Random
Forest, and XGBoost using Optuna. It also includes utility functions for handling
pruner objects and creating Optuna studies.
"""
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier


def get_pruner(path: str) -> SuccessiveHalvingPruner:
    """
    Load a pruner from file or create a new one if it does not exist.

    Parameters
    ----------
    path : str
        Path to the pruner file.

    Returns
    -------
    SuccessiveHalvingPruner
        A SuccessiveHalvingPruner instance loaded from the file or newly created.
    """
    pruner_file = Path(path)
    if pruner_file.exists():
        return pickle.load(open(path, "rb"))
    else:
        return SuccessiveHalvingPruner()


def optuna_opt(cl_type, obj, n_jobs, x_train, y_train, rnd: int):
    """
    Optimize a machine learning model using Optuna.

    Parameters
    ----------
    cl_type : str
        Type of the classifier (e.g., 'XGBoost').
    obj : callable
        Objective function to optimize.
    n_jobs : int
        Number of parallel jobs.
    x_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    rnd : int
        Random seed.

    Returns
    -------
    optuna.study.Study
        The optimized Optuna study.
    """
    if cl_type == 'XGBoost':
        return optuna_opt_xgboost(x_train, y_train, rnd, n_jobs)
    study = get_study(cl_type)
    study.optimize(lambda trial: obj(trial, x_train, y_train, rnd), n_trials=n_jobs * 5,
                   show_progress_bar=True, n_jobs=n_jobs, gc_after_trial=True)
    return study


def optuna_opt_xgboost(x_train, y_train, rnd, n_jobs):
    """
    Optimize an XGBoost model using Optuna.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    rnd : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    optuna.study.Study
        The optimized Optuna study.
    """
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    scale_pos_weight = 1 if len(np.unique(y_train)) != 2 else class_weight_dict[0]
    study1 = get_study('XGBoost_1')
    study1.optimize(lambda trial: objective_xgb_step1(trial, x_train, y_train, rnd,
                                                      scale_pos_weight), n_trials=n_jobs * 5,
                    show_progress_bar=True, n_jobs=n_jobs, gc_after_trial=True)
    n_est = study1.best_params['n_estimators']
    lr = study1.best_params['learning_rate']
    study2 = get_study('XGBoost')
    study2.optimize(lambda trial: objective_xgb(trial, x_train, y_train, rnd,
                                                scale_pos_weight, n_est, lr),
                    n_trials=n_jobs * 10,
                    show_progress_bar=True, n_jobs=n_jobs, gc_after_trial=True)
    return study2


def get_study(cl_type: str) -> optuna.study:
    """
    Create or load an Optuna study.

    Parameters
    ----------
    cl_type : str
        Type of the classifier (e.g., 'XGBoost').

    Returns
    -------
    optuna.study.Study
        The created or loaded Optuna study.
    """
    kwargs = {'storage': f"sqlite:///{cl_type}.db", 'study_name': cl_type,
              'direction': "maximize", 'load_if_exists': True, 'pruner': get_pruner(cl_type)}
    study = optuna.create_study(**kwargs)
    return study


def objective_lda(
        trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series, _: int) -> float:
    """
    Optimization objective function for LDA model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Instance representing a process of evaluating an objective function.
    x : pd.DataFrame
        Training features to split into X_train_cv and X_val_cv.
    y : pd.Series
        Training labels to split into y_train_cv and y_val_cv.

    Returns
    -------
    float
        Mean ROC_AUC score from cross-validation.
    """
    lda_params = {
        "solver": trial.suggest_categorical("solver", ['svd', 'eigen']),
        "shrinkage": trial.suggest_float("shrinkage", 0.0, 1.0)
    }
    if lda_params["solver"] == 'svd':
        lda_params["shrinkage"] = None

    cv = StratifiedKFold(n_splits=3, shuffle=True)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = LinearDiscriminantAnalysis(**lda_params)
        model.fit(
            x_train_cv,
            y_train_cv,
        )
        y_score = model.predict_proba(x_val_cv)
        binary = len(np.unique(y_val_cv)) == 2
        cv_predicts[idx] = roc_auc_score(
            y_val_cv, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
            multi_class='ovr', average='micro')
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    av_auc = np.mean(cv_predicts)
    return av_auc


def objective_lr(trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series, rnd: int) -> float:
    """
    Optimization objective function for Logistic Regression model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Instance representing a process of evaluating an objective function.
    x : pd.DataFrame
        Training features to split into X_train_cv and X_val_cv.
    y : pd.Series
        Training labels to split into y_train_cv and y_val_cv.
    rnd : int
        Random seed.

    Returns
    -------
    float
        Mean ROC_AUC score from cross-validation.
    """
    penalty_solver = {
        'l2': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'l1': ['liblinear', 'saga'],
        'elasticnet': ['saga']
    }
    params = {
        "penalty": trial.suggest_categorical("penalty", ['l1', 'l2', 'elasticnet']),
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
    }
    params["solver"] = trial.suggest_categorical(f"solver_{params['penalty']}",
                                                 penalty_solver[params['penalty']])
    params['l1_ratio'] = trial.suggest_float("l1_ratio", 0., 1.) \
        if params['penalty'] == 'elasticnet' else None

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = LogisticRegression(class_weight='balanced', random_state=rnd, max_iter=1000,
                                   **params)
        model.fit(
            x_train_cv,
            y_train_cv,
        )
        y_score = model.predict_proba(x_val_cv)
        binary = len(np.unique(y_val_cv)) == 2
        cv_predicts[idx] = roc_auc_score(
            y_val_cv, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
            multi_class='ovr', average='micro')
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    av_auc = np.mean(cv_predicts)
    return av_auc


def objective_svc(trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series, rnd: int) -> float:
    """
    Optimization objective function for Support Vector Classification model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Instance representing a process of evaluating an objective function.
    x : pd.DataFrame
        Training features to split into X_train_cv and X_val_cv.
    y : pd.Series
        Training labels to split into y_train_cv and y_val_cv.
    rnd : int
        Random seed.

    Returns
    -------
    float
        Mean ROC_AUC score from cross-validation.
    """
    params = {
        "C": trial.suggest_float("C", 1e-5, 1e4, log=True),
        "tol": trial.suggest_float("tol", 1e-8, 1e-4, log=True),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = SVC(class_weight='balanced', random_state=rnd, probability=True, kernel='linear',
                    **params)
        model.fit(x_train_cv, y_train_cv)
        y_score = model.predict_proba(x_val_cv)
        binary = len(np.unique(y_val_cv)) == 2
        cv_predicts[idx] = roc_auc_score(
            y_val_cv, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
            multi_class='ovr', average='micro')
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    av_auc = np.mean(cv_predicts)
    return av_auc


def objective_dt(trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series, rnd: int) -> float:
    """
    Optimization objective function for Decision Tree model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Instance representing a process of evaluating an objective function.
    x : pd.DataFrame
        Training features to split into X_train_cv and X_val_cv.
    y : pd.Series
        Training labels to split into y_train_cv and y_val_cv.
    rnd : int
        Random seed.

    Returns
    -------
    float
        Mean ROC_AUC score from cross-validation.
    """
    params = {
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "splitter": trial.suggest_categorical("splitter", ['best', 'random']),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 21),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 11),
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2']),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = DecisionTreeClassifier(class_weight='balanced', random_state=rnd, **params)
        model.fit(x_train_cv, y_train_cv)
        y_score = model.predict_proba(x_val_cv)
        binary = len(np.unique(y_val_cv)) == 2
        cv_predicts[idx] = roc_auc_score(
            y_val_cv, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
            multi_class='ovr', average='micro')
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    av_auc = np.mean(cv_predicts)
    return av_auc


def objective_rf(trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series, rnd: int) -> float:
    """
    Optimization objective function for Random Forest model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Instance representing a process of evaluating an objective function.
    x : pd.DataFrame
        Training features to split into X_train_cv and X_val_cv.
    y : pd.Series
        Training labels to split into y_train_cv and y_val_cv.
    rnd : int
        Random seed.

    Returns
    -------
    float
        Mean ROC_AUC score from cross-validation.
    """
    params = {
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "max_samples": trial.suggest_float("max_samples", .1, 1.),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 21),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 11),
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2']),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = RandomForestClassifier(class_weight='balanced', random_state=rnd, **params)
        model.fit(x_train_cv, y_train_cv)
        y_score = model.predict_proba(x_val_cv)
        binary = len(np.unique(y_val_cv)) == 2
        cv_predicts[idx] = roc_auc_score(
            y_val_cv, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
            multi_class='ovr', average='micro')
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    av_auc = np.mean(cv_predicts)
    return av_auc


def objective_xgb_step1(
        trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series, rnd: int, scale_pos_weight: dict
) -> float:
    """
    Optimization objective function for XGBoost model (Step 1) with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Instance representing a process of evaluating an objective function.
    x : pd.DataFrame
        Training features to split into X_train_cv and X_val_cv.
    y : pd.Series
        Training labels to split into y_train_cv and y_val_cv.
    rnd : int
        Random seed.
    scale_pos_weight : dict
        Class weights to adjust imbalance in data.

    Returns
    -------
    float
        Mean ROC_AUC score from cross-validation.
    """
    xgb_params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        model = XGBClassifier(
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=rnd,
            scale_pos_weight=scale_pos_weight,
            device="cpu",
            **xgb_params,
        )
        model.fit(
            x_train_cv,
            y_train_cv,
            eval_set=[(x_val_cv, y_val_cv)],
            verbose=0,
        )
        y_score = model.predict_proba(x_val_cv)
        binary = len(np.unique(y_val_cv)) == 2
        cv_predicts[idx] = roc_auc_score(
            y_val_cv, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
            multi_class='ovr', average='micro')
        # Report the intermediate score for pruning
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    av_auc = np.mean(cv_predicts)
    return av_auc

def objective_xgb(
        trial: optuna.trial.Trial, x: pd.DataFrame, y: pd.Series, rnd: int, scale_pos_weight: dict,
        n_estimators: int, learning_rate: float
) -> float:
    """
    Optimization objective function for XGBoost model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
       instance represents a process of evaluating an objective function
    x: pd.DataFrame
        X_train to split into X_train_cv and X_val_cv
    y: pd.Series
        y_train to split into y_train_cv and y_val_cv

    Returns
    -------
    out : float
       mean ROC_AUC
    """
    xgb_params = {
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "gamma": trial.suggest_int("gamma", 0, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 4, 16),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e4, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1e2, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rnd)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=rnd,
            scale_pos_weight=scale_pos_weight,
            device="cpu",
            **xgb_params,
        )
        model.fit(
            x_train_cv,
            y_train_cv,
            eval_set=[(x_val_cv, y_val_cv)],
            verbose=0,
        )
        y_score = model.predict_proba(x_val_cv)
        binary = len(np.unique(y_val_cv)) == 2
        cv_predicts[idx] = roc_auc_score(
            y_val_cv, y_score if len(y_score.shape) == 1 or not binary else y_score[:, 1],
            multi_class='ovr', average='micro')
        # Report the intermediate score for pruning
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    av_auc = np.mean(cv_predicts)
    return av_auc
