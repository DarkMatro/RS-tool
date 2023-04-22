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
    hamming_loss, jaccard_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier


def do_lda(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    model = LinearDiscriminantAnalysis(store_covariance=True)
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(x_train, y_train)
    # model.fit(x_train, y_train)
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
    accuracy_score_train = np.round(model.score(x_train, y_train), 5) * 100.
    _model = copy.deepcopy(model)
    # model_2d = _model.fit(transformed_2d, y_train)
    model_2d = make_pipeline(StandardScaler(), _model)
    model_2d.fit(transformed_2d, y_train)
    features_in_2d = np.concatenate((transformed_2d, transformed_2d_test))
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'y_pred': y_pred, 'y_pred_test': y_pred_test,
            'y_score': y_score, 'y_score_dec_func': y_score_dec_func, 'accuracy_score_train': accuracy_score_train,
            'model_2d': model_2d, 'cv_scores': cv_scores}


def do_qda(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'y_pred': y_pred, 'y_pred_test': y_pred_test,
            'y_score': y_score, 'y_score_dec_func': y_score_dec_func, 'accuracy_score_train': accuracy_score_train,
            'y_pred_2d': y_pred_2d, 'model_2d': model_2d, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio}


def do_lr(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    model = LogisticRegression(max_iter=10000, n_jobs=-1, random_state=rng)
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    y_score_dec_func = model.decision_function(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d,
            'y_score_dec_func': y_score_dec_func, 'y_pred': y_pred}


def do_svc(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    y_score_dec_func = model.decision_function(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d,
            'y_score_dec_func': y_score_dec_func, 'y_pred': y_pred}


def do_nn(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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

    model = KNeighborsClassifier(n_jobs=-1)
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_gpc(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    model = GaussianProcessClassifier(n_jobs=-1)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_dt(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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

    model = DecisionTreeClassifier(random_state=rng)
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_nb(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_rf(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    parameters = {'criterion': ["gini", "entropy", "log_loss"]}

    model = RandomForestClassifier(random_state=rng, n_jobs=-1)
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_ab(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    y_score_dec_func = model.decision_function(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred,
            'y_score_dec_func': y_score_dec_func}


def do_mlp(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
                  'hidden_layer_sizes': np.arange(100, 1000, 100)}

    model = MLPClassifier(random_state=rng, max_iter=20000)
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(x_train, y_train)
    pca = PCA(n_components=2)
    pca.fit(x_train.values, y_train)
    transformed_2d = scale(pca.transform(x_train.values))
    features_in_2d = scale(pca.transform(np.concatenate((x_train.values, x_test.values))))
    model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
    model_2d.fit(transformed_2d, y_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred = np.concatenate((y_pred, y_pred_test))
    y_pred_2d = model_2d.predict(features_in_2d)
    accuracy_score_train = np.round(model.score(x_train, y_train) * 100., 5)
    y_score = model.predict_proba(x_test)
    cv_scores = cross_val_score(model, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), n_jobs=-1)
    return {'model': model, 'features_in_2d': features_in_2d, 'model_2d': model_2d, 'y_pred_test': y_pred_test,
            'accuracy_score_train': accuracy_score_train, 'y_score': y_score, 'cv_scores': cv_scores,
            'explained_variance_ratio': explained_variance_ratio, 'y_pred_2d': y_pred_2d, 'y_pred': y_pred}


def do_pca(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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


def do_plsda(x_train: DataFrame, y_train: list[int], x_test: DataFrame, y_test: list[int]) -> dict:
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
    classes = np.unique(y_data)
    lists_trick = []
    for i in classes:
        y_class = np.where(y_data == i, 1, 0)
        lists_trick.append(y_class)
    dummy = np.array(lists_trick).T
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


def model_metrics(y_true: list[int], y_pred: list[int], binary: bool, target_names) -> dict:
    average_func = 'binary' if binary else 'macro'
    c_r = classification_report(y_true, y_pred, target_names=target_names) \
        if len(target_names) == len(np.unique(y_true)) else None
    res = {'accuracy_score': np.round(accuracy_score(y_true, y_pred), 4) * 100,
           'precision_score': np.round(precision_score(y_true, y_pred, average=average_func), 4) * 100,
           'recall_score': np.round(recall_score(y_true, y_pred, average=average_func), 4) * 100,
           'f1_score': np.round(f1_score(y_true, y_pred, average=average_func), 4) * 100,
           'fbeta_score': np.round(fbeta_score(y_true, y_pred, average=average_func, beta=0.5), 4) * 100,
           'hamming_loss': np.round(hamming_loss(y_true, y_pred), 4) * 100,
           'jaccard_score': np.round(jaccard_score(y_true, y_pred, average=average_func), 4) * 100,
           'classification_report': c_r}
    return res


def class_labels(row_index, pred, class_count):
    return [f'Class {i + 1} ({pred[row_index, i].round(2):.2f})' for i in range(class_count)]


def clf_predict(X, model, clf_name) -> dict:
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    predicted = model.predict(X)
    predicted_proba = model.predict_proba(X)
    return {'predicted': predicted, 'predicted_proba': predicted_proba, 'clf_name': clf_name}

