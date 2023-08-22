import copy

import numpy as np
from pandas import DataFrame
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, \
    hamming_loss, jaccard_score, classification_report, make_scorer, auc, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from joblib import parallel_backend
from skorch import NeuralNetClassifier
import torch
from torch import nn


def _scorers() -> dict:
    return {'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score)
            }


def do_lda(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """

    parameters = [{'solver': ['svd']},
                  {'solver': ['eigen'], 'shrinkage': np.arange(0, 1, 0.02)}]
    n_classes = len(np.unique(y_train))
    model = LinearDiscriminantAnalysis(store_covariance=True)
    with parallel_backend('multiprocessing', n_jobs=-1):
        if n_classes > 2:
            model = GridSearchCV(model, parameters, n_jobs=-1)
        else:
            model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        model.fit(x_train, y_train)
        transformed_2d = model.transform(x_train)
        if transformed_2d.shape[1] > 1:
            transformed_2d = transformed_2d[:, [0, 1]]
        transformed_2d_test = model.transform(x_test)
        if transformed_2d_test.shape[1] > 1:
            transformed_2d_test = transformed_2d_test[:, [0, 1]]
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_score = model.predict_proba(x_test)
        y_score_dec_func = model.decision_function(x_test)
        accuracy_score_train = model.score(x_train, y_train)
        accuracy_score_train = np.round(accuracy_score_train, 5) * 100.
        _model = copy.deepcopy(model)
        model_2d = make_pipeline(StandardScaler(), _model)
        model_2d.fit(transformed_2d, y_train)
        features_in_2d = np.concatenate((transformed_2d, transformed_2d_test))
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)))
    return {'model': model, 'features_in_2d': features_in_2d, 'y_pred': y_pred, 'y_pred_test': y_pred_test,
            'y_score': y_score, 'y_score_dec_func': y_score_dec_func, 'accuracy_score_train': accuracy_score_train,
            'model_2d': model_2d, 'cv_scores': cv_scores}


def do_qda(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    model = QuadraticDiscriminantAnalysis(store_covariance=True)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    transformed_2d_test = scale(pca.transform(x_test.values))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    y_pred_2d = model_2d.predict(transformed_2d)
    y_pred_2d_test = model_2d.predict(transformed_2d_test)
    y_pred_2d = np.concatenate((y_pred_2d, y_pred_2d_test))
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_score = model.predict_proba(x_test)
    y_score_dec_func = model.decision_function(x_test)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    explained_variance_ratio = pca.explained_variance_ratio_
    features_in_2d = np.concatenate((transformed_2d, transformed_2d_test))
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)))
    return {'model': model, 'features_in_2d': features_in_2d, 'y_pred': y_pred, 'y_pred_test': y_pred_test,
            'y_score': y_score, 'y_score_dec_func': y_score_dec_func, 'accuracy_score_train': accuracy_score_train,
            'y_pred_2d': y_pred_2d, 'model_2d': model_2d, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio}


def do_lr(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) \
        -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    rng = np.random.RandomState(0)
    parameters = [{'penalty': ('l1', 'l2'), 'C': [.01, .1, 1, 10, 100, 1000, 10_000, 100_000],
                   'solver': ['liblinear']},
                  {'penalty': ['l2'], 'C': [.01, .1, 1, 10, 100, 1000, 10_000, 100_000],
                   'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']},
                  {'penalty': ['elasticnet'], 'C': [.01, .1, 1, 10, 100, 1000, 10_000, 100_000],
                   'solver': ['saga']}
                  ]
    n_classes = len(np.unique(y_train))
    with parallel_backend('multiprocessing', n_jobs=-1):
        model = LogisticRegression(max_iter=10000, n_jobs=-1, random_state=rng)
        if n_classes > 2:
            model = GridSearchCV(model, parameters, n_jobs=-1)
        else:
            model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])

        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)

        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        y_score_dec_func = model.decision_function(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d,
            'y_score_dec_func': y_score_dec_func, 'y_pred': y_pred}


def do_svc(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    rng = np.random.RandomState(0)
    parameters = {'nu': np.arange(0, 1, 0.05)}

    model = NuSVC(random_state=rng, probability=True, kernel='linear')
    n_classes = len(np.unique(y_train))
    with parallel_backend('multiprocessing', n_jobs=-1):
        if n_classes > 2:
            model = GridSearchCV(model, parameters, n_jobs=-1)
        else:
            model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        y_score_dec_func = model.decision_function(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d,
            'y_score_dec_func': y_score_dec_func, 'y_pred': y_pred}


def do_nn(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    parameters = {'n_neighbors': np.arange(2, int(len(x_train) / 2), 1), 'weights': ['uniform', 'distance']}
    n_classes = len(np.unique(y_train))
    with parallel_backend('multiprocessing', n_jobs=-1):
        model = KNeighborsClassifier(n_jobs=-1)
        if n_classes > 2:
            model = GridSearchCV(model, parameters, n_jobs=-1)
        else:
            model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_gpc(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    with parallel_backend('multiprocessing', n_jobs=-1):
        model = GaussianProcessClassifier(n_jobs=-1)
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_dt(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    rng = np.random.RandomState(0)
    parameters = {'criterion': ["gini", "entropy", "log_loss"]}
    n_classes = len(np.unique(y_train))
    model = DecisionTreeClassifier(random_state=rng)
    with parallel_backend('multiprocessing', n_jobs=-1):
        if n_classes > 2:
            model = GridSearchCV(model, parameters, n_jobs=-1)
        else:
            model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_nb(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """

    model = GaussianNB()
    model.fit(x_train, y_train)
    transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                             params['use_pca'])
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    with parallel_backend('multiprocessing', n_jobs=-1):
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_rf(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    Random forest
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    rng = np.random.RandomState(0)
    parameters = {'criterion': ["gini", "entropy", "log_loss"], 'min_samples_split': [2, 3, 5, 10],
                  'n_estimators': [100, 200, 300, 400, 500], 'max_features': ["sqrt", "log2", None]}
    n_classes = len(np.unique(y_train))
    with parallel_backend('multiprocessing', n_jobs=-1):
        model = RandomForestClassifier(random_state=rng, n_jobs=-1)
        if n_classes > 2:
            model = GridSearchCV(model, parameters, n_jobs=-1)
        else:
            model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_ab(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    Random forest
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    rng = np.random.RandomState(0)
    model = AdaBoostClassifier(random_state=rng)
    model.fit(x_train, y_train)
    transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                             params['use_pca'])
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    y_score_dec_func = model.decision_function(x_test)
    with parallel_backend('multiprocessing', n_jobs=-1):
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred,
            'y_score_dec_func': y_score_dec_func}


def do_mlp(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) \
        -> dict:
    """
    MLP Multi-layer Perceptron classifier
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    rng = np.random.RandomState(0)
    n_classes = len(np.unique(y_train))
    with parallel_backend('multiprocessing', n_jobs=-1):
        if 'activation' not in params:
            model = MLPClassifier(random_state=rng)
            parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
                          'hidden_layer_sizes': np.arange(2, 100, 2)}
            if n_classes > 2:
                model = GridSearchCV(model, parameters, n_jobs=-1)
            else:
                model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        else:
            model = MLPClassifier(params['hidden_layer_sizes'], params['activation'], solver=params['solver'],
                                  random_state=rng)
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_pca(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    PCA
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    model = PCA(n_components=2)
    x_data = np.concatenate((x_train.values, x_test.values))
    y_data = np.concatenate((y_train, y_test))
    model.fit(x_data, y_data)
    features_in_2d = scale(model.transform(x_data))
    explained_variance_ratio = model.explained_variance_ratio_
    return {'model': model, 'features_in_2d': features_in_2d, 'explained_variance_ratio': explained_variance_ratio}


def do_plsda(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    PLS-DA
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    x_data = np.concatenate((x_train.values, x_test.values))
    y_data = np.concatenate((y_train, y_test))
    dummy = plsda_y_data_trick(y_data)
    model = PLSRegression(n_components=2)
    model.fit(x_data, dummy)
    variance_in_x = np.var(model.x_scores_, axis=0)
    explained_variance_ratio = variance_in_x / np.sum(variance_in_x)
    features_in_2d = scale(model.transform(x_data))
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    return {'model': model, 'features_in_2d': features_in_2d, 'y_pred_test': y_pred_test,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred': y_pred}


def do_xgboost(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    XGBoost
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict

    Parameters
    ----------
    params
    params
    """
    rng = np.random.RandomState(0)
    model = XGBClassifier(random_state=rng, n_jobs=-1)
    parameters = [{'booster': ['gbtree'], 'eta': np.arange(0, 1, 0.1), 'max_depth': np.arange(3, 10, 1),
                   'max_delta_step': np.arange(0, 10, 1)},
                  {'booster': ['dart'], 'normalize_type ': ['tree', 'forest'],
                   'rate_drop ': np.arange(0, 1, 0.1), 'skip_drop': np.arange(0, 1, 0.1)},
                  {'booster': ['gblinear'], 'feature_selector ': ['cyclic', 'shuffle', 'random', 'greedy', 'thrifty']}]
    n_classes = len(np.unique(y_train))
    with parallel_backend('multiprocessing', n_jobs=-1):
        if n_classes > 2:
            model = GridSearchCV(model, parameters, n_jobs=-1)
        else:
            model = GridSearchCV(model, parameters, scoring=_scorers(), n_jobs=-1, refit=params['refit'])
        model.fit(x_train, y_train)
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def model_metrics(y_true: list[int], y_pred: list[int], binary: bool, target_names) -> dict:
    average_func = 'binary' if binary else 'macro'
    if np.max(y_true) != np.max(y_pred) and np.min(y_true) != np.min(y_pred):
        y_true = np.array(y_true)
        y_true -= 1
    c_r = classification_report(y_true, y_pred, target_names=target_names) \
        if len(target_names) == len(np.unique(y_true)) else None
    pos_label = np.sort(np.unique(y_true))[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)
    res = {'accuracy_score': np.round(accuracy_score(y_true, y_pred), 4) * 100,
           'precision_score': np.round(precision_score(y_true, y_pred, average=average_func,
                                                       pos_label=pos_label), 4) * 100,
           'recall_score': np.round(recall_score(y_true, y_pred, average=average_func,
                                                 pos_label=pos_label), 4) * 100,
           'f1_score': np.round(f1_score(y_true, y_pred, average=average_func,
                                         pos_label=pos_label), 4) * 100,
           'fbeta_score': np.round(fbeta_score(y_true, y_pred, average=average_func, beta=0.5,
                                               pos_label=pos_label), 4) * 100,
           'hamming_loss': np.round(hamming_loss(y_true, y_pred), 4) * 100,
           'jaccard_score': np.round(jaccard_score(y_true, y_pred, average=average_func, pos_label=pos_label), 4) * 100,
           'AUC': auc(fpr, tpr),
           'classification_report': c_r}
    return res


def class_labels(row_index, pred, class_count):
    return [f'Class {i + 1} ({pred[row_index, i].round(2):.2f})' for i in range(class_count)]


def clf_predict(X, model, clf_name) -> dict:
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    if clf_name == 'Torch':
        X = np.array(X.values).astype(np.float32)
    predicted = model.predict(X)
    predicted_proba = model.predict_proba(X)
    if clf_name == 'Torch':
        predicted += 1
    return {'predicted': predicted, 'predicted_proba': predicted_proba, 'clf_name': clf_name}


def plsda_y_data_trick(y_data):
    classes = np.unique(y_data)
    lists_trick = []
    for i in classes:
        y_class = np.where(y_data == i, 1, 0)
        lists_trick.append(y_class)
    return np.array(lists_trick).T


def dim_reduction(x_train, x_test, y_train, use_pca):
    if isinstance(x_train, DataFrame):
        x_train = x_train.values
    if isinstance(x_test, DataFrame):
        x_test = x_test.values
    if use_pca:
        pca = PCA(n_components=1)
        pca.fit(x_train, y_train)
        transformed_2d = scale(pca.transform(x_train))
        features_in_2d = scale(pca.transform(np.concatenate((x_train, x_test))))
        explained_variance_ratio = pca.explained_variance_ratio_
    else:
        dummy = plsda_y_data_trick(y_train)
        plsda = PLSRegression(n_components=2)
        plsda.fit(x_train, dummy)
        transformed_2d = scale(plsda.transform(x_train))
        features_in_2d = scale(plsda.transform(np.concatenate((x_train, x_test))))
        variance_in_x = np.var(plsda.x_scores_, axis=0)
        explained_variance_ratio = variance_in_x / np.sum(variance_in_x)
    return transformed_2d, features_in_2d, explained_variance_ratio


def do_torch_nn(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    Pytorch 1 hidden layer classifier
    @param y_test:
    @param x_train: x: array-like of shape (n_samples, n_features)
            Training data.
    @param y_train:  y : array-like of shape (n_samples,)
            Target values.
    @param x_test: array-like of shape (n_samples, n_features)
            Testing data
    @return: dict
    """
    rng = np.random.RandomState(0)
    # device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"Using {device} device")
    n_classes = len(np.unique(y_train))
    input_layer_size = x_train.shape[1]
    loss_fn = nn.CrossEntropyLoss()
    h_sizes = np.arange(1, 11) * input_layer_size
    x_train = np.array(x_train.values).astype(np.float32)
    y_train = np.array(y_train).astype(np.int64)
    y_train -= 1
    x_test = np.array(x_test.values).astype(np.float32)
    y_test = np.array(y_test).astype(np.int64)
    y_test -= 1
    with parallel_backend('multiprocessing', n_jobs=-1):
        if 'activation' not in params:
            model = NeuralNetClassifier(NeuralNetwork, max_epochs=params['max_epoch'], criterion=loss_fn,
                                        iterator_train__shuffle=True, device=device, train_split=False)
            grid_params = {
            #     'lr': [.1, .01, .02, 1e-3],
                'module__hidden_layer_size': list(h_sizes),
                'module__input_size': [input_layer_size],
                'module__output_size': [n_classes],
                # 'optimizer': [torch.optim.SGD, torch.optim.Adam, torch.optim.LBFGS],
                # 'module__activation': [nn.Identity(), nn.LogSigmoid(), nn.Tanh(), nn.ReLU()],
            }
            if n_classes > 2:
                model = GridSearchCV(model, grid_params,  verbose=0, error_score='raise')
            else:
                model = GridSearchCV(model, grid_params,  verbose=0, error_score='raise', refit=params['refit'],
                                     scoring=_scorers())
        else:
            match params['activation']:
                case 'relu':
                    activation_f = nn.ReLU()
                case 'tanh':
                    activation_f = nn.Tanh()
                case 'logistic':
                    activation_f = nn.LogSigmoid()
                case 'identity':
                    activation_f = nn.Identity()
                case _:
                    activation_f = nn.ReLU()
            match params['solver']:
                case 'lbfgs':
                    solver = torch.optim.LBFGS
                case 'sgd':
                    solver = torch.optim.SGD
                case 'adam':
                    solver = torch.optim.Adam
                case _:
                    solver = torch.optim.SGD
            net = NeuralNetwork(input_layer_size, n_classes, params['hidden_layer_sizes'], activation_f).to(device)
            print(net)
            model = NeuralNetClassifier(net, max_epochs=params['max_epoch'], criterion=loss_fn, device=device,
                                        lr=params['learning_rate'], iterator_train__shuffle=True, train_split=False,
                                        optimizer=solver)
        model.fit(x_train, y_train)
        # transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_train,
        #                                                                          params['use_pca'])
        # model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        # model_2d.fit(transformed_2d, y_train)
        y_pred = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_pred = np.concatenate((y_pred, y_pred_test))
        # y_pred_2d = model_2d.predict(features_in_2d)
        accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
        y_score = model.predict_proba(x_test)
        cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                    n_jobs=-1)
    return {'model': model, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores, 'y_pred': y_pred}


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 93, output_size: int = 2, hidden_layer_size: int = 93,
                 activation=nn.ReLU()):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_size),
            activation,
            nn.Linear(self.hidden_layer_size, self.output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

