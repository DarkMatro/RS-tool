import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def get_pruner(path: str) -> SuccessiveHalvingPruner:
    """
    Load from file or create new pruner if not exists.

    Parameters
    ----------
    path: str
        to pruner file

    Returns
    -------
    out : SuccessiveHalvingPruner
    """
    pruner_file = Path(path)
    if pruner_file.exists():
        return pickle.load(open(path, "rb"))
    else:
        return SuccessiveHalvingPruner()


def optuna_opt(cl_type, obj, n_trials, x_train, y_train):
    kwargs = {'storage': f"sqlite:///{cl_type}.db", 'study_name': cl_type,
              'direction': "maximize", 'load_if_exists': True, 'pruner': get_pruner(cl_type)}
    study = optuna.create_study(**kwargs)
    study.optimize(lambda trial: obj(trial, x_train, y_train), n_trials=n_trials,
                   show_progress_bar=True, n_jobs=-1, gc_after_trial=True)
    return study


def objective_lda(
        trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optimization objective function for LDA model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
       instance represents a process of evaluating an objective function
    X: pd.DataFrame
        X_train to split into X_train_cv and X_val_cv
    y: pd.Series
        y_train to split into y_train_cv and y_val_cv
    Returns
    -------
    out : float
       mean ROC_AUC
    """
    lda_params = {
        "solver": trial.suggest_categorical("solver", ['svd', 'lsqr', 'eigen']),
        "shrinkage": trial.suggest_float("shrinkage", 0.0, 1.0)
    }
    if lda_params["solver"] == 'svd':
        lda_params["shrinkage"] = None

    cv = StratifiedKFold(n_splits=3, shuffle=True)
    cv_predicts = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        x_train_cv, x_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = LinearDiscriminantAnalysis(**lda_params)
        model.fit(
            x_train_cv,
            y_train_cv,
        )
        y_score = model.predict_proba(x_val_cv)
        cv_predicts[idx] = roc_auc_score(y_val_cv, y_score[:, 1])
        trial.report(cv_predicts[idx], idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    av_auc = np.mean(cv_predicts)
    return av_auc
