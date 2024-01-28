import copy

import numpy as np
from pandas import DataFrame
from sklearn.cross_decomposition import PLSRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, \
    hamming_loss, jaccard_score, classification_report, make_scorer, auc, roc_curve, \
    balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from hyperopt import STATUS_OK
from joblib import parallel_backend


def scorer_metrics() -> dict:
    return {'precision_score': make_scorer(precision_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro'),
            'accuracy_score': make_scorer(accuracy_score, average='micro'),
            'f1_score': make_scorer(f1_score, average='micro'),
            'jaccard_score': make_scorer(jaccard_score, average='micro'),
            'auc': make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
            'balanced_accuracy_score': make_scorer(balanced_accuracy_score, average='micro'),
            'brier_score_loss': make_scorer(brier_score_loss, average='micro'),
            'log_loss': make_scorer(log_loss, average='micro'),
            }


def scorer_metric_for_halving_grid_search(scorer: str) -> str:
    metrics = {'precision_score': 'precision',
               'recall_score': 'recall',
               'accuracy_score': 'accuracy',
               'f1_score': 'f1',
               'jaccard_score': 'jaccard',
               'auc': 'roc_auc',
               'balanced_accuracy_score': 'balanced_accuracy',
               'brier_score_loss': 'neg_brier_score',
               'log_loss': 'neg_log_loss',
               }
    return metrics[scorer]


def gscv_score(cv_results: dict) -> str:
    """
    Function returns CV metrics result of GridSearchCV for report.
    Parameters
    ----------
    cv_results: dict
        A dict with keys as column headers and values as columns
    Returns
    -------
    text: str
        with precision, recall, accuracy, F1-score results for report
    """
    scorers = scorer_metrics().keys()
    text = ''
    for sc in scorers:
        v_mean = cv_results['mean_test_%s' % sc]
        v_mean = v_mean[~np.isnan(v_mean)].mean()
        sd_mean = cv_results['std_test_%s' % sc]
        sd_mean = sd_mean[~np.isnan(sd_mean)].mean()
        text += "CV test %s = %0.2f ± %0.2f" % (sc, v_mean, sd_mean) + '\n'
    return text


def hgscv_score(cv_results: dict) -> str:
    """
    Function returns CV metrics result of HalvingGridSearchCV for report.
    Parameters
    ----------
    cv_results: dict
        A dict with keys as column headers and values as columns
    Returns
    -------
    text: str
        with 1 metric results for report
    """
    text = ''
    v_mean = cv_results['mean_test_score']
    v_mean = v_mean[~np.isnan(v_mean)].mean()
    sd_mean = cv_results['std_test_score']
    sd_mean = sd_mean[~np.isnan(sd_mean)].mean()
    text += "CV test score = %0.2f ± %0.2f" % (v_mean, sd_mean) + '\n'
    v_mean = cv_results['mean_train_score']
    v_mean = v_mean[~np.isnan(v_mean)].mean()
    sd_mean = cv_results['std_train_score']
    sd_mean = sd_mean[~np.isnan(sd_mean)].mean()
    text += "CV train score = %0.2f ± %0.2f" % (v_mean, sd_mean) + '\n'
    return text


def fit_lda_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int],
                params: dict) -> dict:
    """
    Fit LDA estimator.

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = LinearDiscriminantAnalysis()
    if params['use_GridSearchCV']:
        model = GridSearchCV(model, params['grid_search_parameters'], n_jobs=-1, verbose=3, scoring=scorer_metrics(),
                             refit=params['refit'], cv=3)
    else:
        model = HalvingGridSearchCV(model, params['grid_search_parameters'], n_jobs=-1, verbose=3, cv=3,
                                    scoring=scorer_metric_for_halving_grid_search(params['refit']))

    with parallel_backend('multiprocessing', n_jobs=-1):
        model.fit(x_train, y_true_train)
        if params['use_GridSearchCV']:
            cv_scores = gscv_score(model.cv_results_)
        else:
            cv_scores = hgscv_score(model.cv_results_)

        # 2d_model for DecisionBoundaryDisplay
        transformed_2d = model.transform(x_train)
        if transformed_2d.shape[1] > 1:
            transformed_2d = transformed_2d[:, [0, 1]]
        transformed_2d_test = model.transform(x_test)
        if transformed_2d_test.shape[1] > 1:
            transformed_2d_test = transformed_2d_test[:, [0, 1]]
        model_2d = copy.deepcopy(model)
        features_in_2d = np.concatenate((transformed_2d, transformed_2d_test))
        model_2d.fit(transformed_2d, y_true_train)
        y_predicted_2d = model_2d.predict(features_in_2d)

        y_predicted_train = model.predict(x_train)
        accuracy_score_train = accuracy_score(y_true_train, y_predicted_train)
        accuracy_score_train = np.round(accuracy_score_train, 5) * 100.
        y_predicted_test = model.predict(x_test)
        y_predicted = np.concatenate((y_predicted_train, y_predicted_test))
        y_train_plus_test = np.concatenate((y_true_train, y_test))
        misclassified = y_train_plus_test != y_predicted
        y_score = model.predict_proba(x_test)
        y_score_dec_func = model.decision_function(x_test)
    return {'model': model, 'features_in_2d': features_in_2d, 'misclassified': misclassified,
            'y_score': y_score, 'y_score_dec_func': y_score_dec_func, 'accuracy_score_train': accuracy_score_train,
            'model_2d': model_2d, 'cv_scores': cv_scores, 'y_pred_2d': y_predicted_2d,
            'y_train_plus_test': y_train_plus_test, 'y_pred_test': y_predicted_test}


def fit_classificator(model, x_train, y_true_train, x_test, y_test, params):
    have_decision_function = 'decision_function' in model.__dir__()
    if params['use_GridSearchCV']:
        model = GridSearchCV(model, params['grid_search_parameters'], n_jobs=-1, verbose=3, scoring=scorer_metrics(),
                             refit=params['refit'], cv=3)
    else:
        model = HalvingGridSearchCV(model, params['grid_search_parameters'], n_jobs=-1, verbose=3, cv=3,
                                    scoring=scorer_metric_for_halving_grid_search(params['refit']))

    with parallel_backend('multiprocessing', n_jobs=-1):
        model.fit(x_train, y_true_train)
        if params['use_GridSearchCV']:
            cv_scores = gscv_score(model.cv_results_)
        else:
            cv_scores = hgscv_score(model.cv_results_)

        # 2d_model for DecisionBoundaryDisplay
        transformed_2d, features_in_2d, explained_variance_ratio = dim_reduction(x_train, x_test, y_true_train,
                                                                                 params['use_pca'])
        model_2d = make_pipeline(StandardScaler(), copy.deepcopy(model))
        model_2d.fit(transformed_2d, y_true_train)
        y_predicted_2d = model_2d.predict(features_in_2d)

        y_predicted_train = model.predict(x_train)
        accuracy_score_train = accuracy_score(y_true_train, y_predicted_train)
        accuracy_score_train = np.round(accuracy_score_train, 5) * 100.
        y_predicted_test = model.predict(x_test)
        y_predicted = np.concatenate((y_predicted_train, y_predicted_test))
        y_train_plus_test = np.concatenate((y_true_train, y_test))
        misclassified = y_train_plus_test != y_predicted
        y_score = model.predict_proba(x_test)
        if have_decision_function:
            y_score_dec_func = model.decision_function(x_test)
    res = {'model': model, 'features_in_2d': features_in_2d, 'misclassified': misclassified, 'y_score': y_score,
           'accuracy_score_train': accuracy_score_train, 'model_2d': model_2d,
           'cv_scores': cv_scores, 'y_pred_2d': y_predicted_2d, 'y_train_plus_test': y_train_plus_test,
           'y_pred_test': y_predicted_test, 'explained_variance_ratio': explained_variance_ratio}
    if have_decision_function:
        res['y_score_dec_func'] = y_score_dec_func
    return res


def fit_lr_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) \
        -> dict:
    """
    LogisticRegression

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = LogisticRegression(max_iter=10_000, n_jobs=-1, random_state=params['random_state'], class_weight='balanced')
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_svc_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int],
                params: dict) -> dict:
    """
    NuSVC with linear kernel

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = NuSVC(kernel='linear', probability=True, random_state=params['random_state'], class_weight='balanced')
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_nn_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    Nearest Neighbors

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = KNeighborsClassifier(n_jobs=-1)
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_gpc_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int],
                params: dict) -> dict:
    """
    GaussianProcessClassifier

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = GaussianProcessClassifier(random_state=params['random_state'], n_jobs=-1)
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_dt_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    DecisionTreeClassifier

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = DecisionTreeClassifier(random_state=params['random_state'], class_weight='balanced')
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_nb_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    GaussianNB

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = GaussianNB()
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_rf_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    Random forest
    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = RandomForestClassifier(random_state=params['random_state'], class_weight='balanced', n_jobs=-1)
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_ab_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    AdaBoostClassifier

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = AdaBoostClassifier(random_state=params['random_state'])
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_mlp_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int],
                params: dict) -> dict:
    """
    MLP Multi-layer Perceptron classifier
    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = MLPClassifier(random_state=params['random_state'], max_iter=params['max_epoch'], learning_rate='adaptive')
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_pca(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    PCA
    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = PCA(n_components=2)
    x_data = np.concatenate((x_train.values, x_test.values))
    y_data = np.concatenate((y_true_train, y_test))
    model.fit(x_data, y_data)
    features_in_2d = scale(model.transform(x_data))
    explained_variance_ratio = model.explained_variance_ratio_
    return {'model': model, 'features_in_2d': features_in_2d, 'explained_variance_ratio': explained_variance_ratio}


def fit_plsda(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int], params: dict) -> dict:
    """
    PLS-DA
    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    x_data = np.concatenate((x_train.values, x_test.values))
    y_data = np.concatenate((y_true_train, y_test))
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


def fit_xgboost_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int],
                    params: dict) -> dict:
    """
    XGBoost

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = XGBClassifier(random_state=params['random_state'], n_jobs=-1)
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_voting_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int],
                   params: dict) -> dict:
    """
    VotingClassifier

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = VotingClassifier(estimators=params['estimators'], voting='soft', n_jobs=-1)
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def fit_stacking_clf(x_train: DataFrame, y_true_train: list[int], x_test: DataFrame, y_test: list[int],
                     params: dict) -> dict:
    """
    StackingClassifier

    Parameters
    ----------
    x_train: array-like of shape (n_samples, n_features)
            Training data.
    y_true_train: array-like of shape (n_samples,)
            Target values.
    x_test: array-like of shape (n_samples, n_features)
            Testing data
    y_test: array-like of shape (n_samples,)
    params: dict
    """
    model = StackingClassifier(estimators=params['estimators'], final_estimator=LogisticRegression(), n_jobs=-1)
    return fit_classificator(model, x_train, y_true_train, x_test, y_test, params)


def objective(space):
    X_train = space['X_train']
    X_test = space['X_test']
    y_train = space['y_train']
    y_test = space['y_test']
    clf = XGBClassifier(eta=space['n_estimators'], n_estimators=space['n_estimators'],
                        max_depth=int(space['max_depth']), gamma=space['gamma'],
                        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']))

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train, eval_set=evaluation, verbose=False)

    pred = clf.predict(X_test)
    print(y_test)
    print(pred)
    print(pred > 0.5)
    accuracy = accuracy_score(y_test, pred)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


def model_metrics(y_true: list[int], y_pred: list[int], y_score: list[float], binary: bool, target_names) -> dict:
    average_func = 'binary' if binary else 'micro'
    if np.max(y_true) != np.max(y_pred) and np.min(y_true) != np.min(y_pred):
        y_true = np.array(y_true)
        y_true -= 1
    c_r = classification_report(y_true, y_pred, target_names=target_names) \
        if len(target_names) == len(np.unique(y_true)) else None
    pos_label = np.sort(np.unique(y_true))[0]
    try:
        auc_score = roc_auc_score(y_true, y_score[:, 1] if binary else y_score, multi_class='ovr', average='micro')
    except ValueError:
        auc_score = -1.
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
           'AUC': auc_score,
           'classification_report': c_r}
    return res


def class_labels(row_index, pred, class_count):
    return [f'Class {i + 1} ({pred[row_index, i].round(2):.2f})' for i in range(class_count)]


def clf_predict(X, model, clf_name) -> dict:
    if isinstance(model, GridSearchCV) or isinstance(model, HalvingGridSearchCV):
        model = model.best_estimator_
    predicted = model.predict(X)
    predicted_proba = model.predict_proba(X)
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
        pca = PCA(n_components=2)
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
