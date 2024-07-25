import re
import warnings
from asyncio import gather
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import error, warning
from multiprocessing import Manager
from os import environ

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost
from asyncqtpy import asyncSlot
from hyperopt import Trials, fmin, hp, tpe
from imblearn.over_sampling import RandomOverSampler
from joblib import parallel_backend
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame, Series
from qtpy.QtGui import QColor
from seaborn import histplot, color_palette
from sklearn.calibration import CalibrationDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.inspection import DecisionBoundaryDisplay, permutation_importance, \
    PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, auc, \
    precision_recall_curve, \
    average_precision_score, ConfusionMatrixDisplay, DetCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV, \
    LearningCurveDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data.default_values import classificator_funcs
from src.stages.ml.functions.fit_classificators import objective
from src.undo_redo import CommandAfterFittingStat

warnings.filterwarnings('ignore')


class StatAnalysisLogic:

    def __init__(self, parent):
        self.parent = parent
        self.classificator_funcs = classificator_funcs()
        self.latest_stat_result = {}
        self.top_features = {}
        self.old_labels = None
        self.new_labels = None
        self.params = {'feature_plots': {'LDA', 'Logistic regression', 'NuSVC'}}

    def get_random_state(self) -> None | int:
        main_window = self.parent
        rng = main_window.ui.random_state_sb.value()
        return rng

    @asyncSlot()
    async def do_fit_classificator(self, cl_type: str) -> None:
        """
        Обучение выбранного классификатора, обновление всех графиков.
        @param cl_type: название метода классификации
        @return: None
        """
        main_window = self.parent
        main_window.time_start = datetime.now()
        main_window.ui.statusBar.showMessage('Fitting model...')
        main_window.close_progress_bar()
        main_window.open_progress_bar(max_value=0)
        t = "Fitting %s Classificator..." % cl_type if cl_type not in ['PCA',
                                                                       'PLS-DA'] else "Fitting %s ..." % cl_type
        main_window.open_progress_dialog(t, "Cancel", maximum=0)
        X, Y, feature_names, target_names, _ = self.dataset_for_ml()
        Y = list(Y)
        rnd_state = self.get_random_state()
        filenames = X['Filename']
        X = X.iloc[:, 1:]
        if cl_type == 'XGBoost':
            Y = self.corrected_class_labels(Y)
        test_size = main_window.ui.test_data_ratio_spinBox.value() / 100.

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                            random_state=rnd_state,
                                                            stratify=Y)
        y_test_bin = label_binarize(y_test, classes=list(set(Y))) if len(target_names) > 2 else None
        executor = ProcessPoolExecutor()
        func = self.classificator_funcs[cl_type]
        main_window.ex = executor
        params = {
                  'refit': self.parent.ui.refit_score.currentText(),
                  'random_state': rnd_state}
        self.add_params(params, cl_type, input_layer_size=x_train.shape[1])
        with Manager() as manager:
            main_window.break_event = manager.Event()
            with executor:
                main_window.futures = [
                    main_window.loop.run_in_executor(executor, func, x_train, y_train, x_test,
                                                     y_test, params)]
                for future in main_window.futures:
                    future.add_done_callback(main_window.progress_indicator)
                result = await gather(*main_window.futures)
        if main_window.widgets['stateTooltip'].wasCanceled() or environ['CANCEL'] == '1':
            main_window.close_progress_bar()
            main_window.ui.statusBar.showMessage('Fitting cancelled.')
            return
        result = result[0]
        result['X'] = X
        result['Y'] = Y
        result['y_train'] = y_train
        result['y_test'] = y_test
        result['x_train'] = x_train
        result['x_test'] = x_test
        result['target_names'] = target_names
        result['feature_names'] = feature_names
        result['y_test_bin'] = y_test_bin
        result['filenames'] = filenames
        command = CommandAfterFittingStat(main_window, result, cl_type, "Fit model %s" % cl_type)
        main_window.undoStack.push(command)
        main_window.close_progress_bar()
        main_window.ui.statusBar.showMessage('Fitting completed', 10000)

    def add_params(self, params: dict, cl_type: str, **kwargs) -> None:
        """
        Function adds hyperparameters for currently selected classificator cl_type
        Parameters
        ----------
        params: dict
        cl_type: str classificator name

        Returns
        -------
            None
        """
        parameters = [{}]
        if cl_type == 'LDA':
            parameters = self.create_lda_gridsearch_params()
        elif cl_type == 'Logistic regression':
            parameters = self.create_lr_gridsearch_params()
        elif cl_type == 'NuSVC':
            parameters = self.create_nusvc_gridsearch_params()
        elif cl_type == 'Nearest Neighbors':
            parameters = self.create_nn_gridsearch_params()
        elif cl_type == 'MLP':
            parameters = self.create_mlp_gridsearch_params(**kwargs)
            params['max_epoch'] = self.parent.ui.max_epoch_spinBox.value()
        elif cl_type == 'Decision Tree':
            parameters = self.create_dt_gridsearch_params()
        elif cl_type == 'Random Forest':
            parameters = self.create_rf_gridsearch_params()
        elif cl_type == 'AdaBoost':
            parameters = self.create_ab_gridsearch_params()
        elif cl_type == 'XGBoost':
            parameters = self.create_xgb_gridsearch_params()
        elif cl_type in ['Voting', 'Stacking']:
            params['estimators'] = self.create_voting_gridsearch_params()
        params['grid_search_parameters'] = parameters

    def get_lda_shrinkage_parameter_value(self):
        v_shrinkage = self.parent.ui.lineEdit_lda_shrinkage.text()
        if v_shrinkage.lower() == 'none':
            shrinkage = None
        elif v_shrinkage.lower() == 'auto':
            shrinkage = 'auto'
        else:
            try:
                v = float(v_shrinkage)
                v = .0 if v < .0 else v
                v = 1. if v > 1. else v
                shrinkage = v
            except ValueError:
                shrinkage = None
        return shrinkage

    def create_lda_gridsearch_params(self) -> list[dict]:
        v_solver = self.parent.ui.comboBox_lda_solver.currentText()
        shrinkage = self.get_lda_shrinkage_parameter_value()
        if not self.parent.ui.lda_solver_check_box.isChecked() \
                and not self.parent.ui.lda_shrinkage_check_box.isChecked():
            parameters = [{'solver': ['svd']},
                          {'solver': ['eigen'], 'shrinkage': [None, 'auto']},
                          {'solver': ['eigen'], 'shrinkage': np.arange(0, 1.1, 0.1)}]
        elif self.parent.ui.lda_solver_check_box.isChecked() \
                and not self.parent.ui.lda_shrinkage_check_box.isChecked():
            if v_solver == 'svd':
                parameters = [{'solver': ['svd'], 'shrinkage': [None]}]
            else:
                parameters = [{'solver': [v_solver], 'shrinkage': [None, 'auto']},
                              {'solver': [v_solver], 'shrinkage': np.arange(0, 1.1, 0.1)}]
        elif not self.parent.ui.lda_solver_check_box.isChecked() \
                and self.parent.ui.lda_shrinkage_check_box.isChecked():
            if shrinkage is None:
                parameters = [{'solver': ['svd']},
                              {'solver': ['eigen', 'lsqr'], 'shrinkage': [shrinkage]}]
            else:
                parameters = [{'solver': ['eigen', 'lsqr'], 'shrinkage': [shrinkage]}]
        else:
            parameters = [{'solver': [v_solver], 'shrinkage': [shrinkage]}]
        return parameters

    def create_lr_gridsearch_params(self) -> list[dict]:
        solver_penalty = {'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'],
                          'newton-cg': ['l2', None],
                          'newton-cholesky': ['l2', None], 'sag': ['l2', None],
                          'saga': ['elasticnet', 'l1', 'l2', None]}
        penalty_solver = {
            'l2': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'l1': ['liblinear', 'saga'],
            'elasticnet': ['saga'], None: ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}

        c_range = [.01, .1, 1, 10, 100, 1000, 10_000, 100_000]
        c = self.parent.ui.lr_c_doubleSpinBox.value()
        penalty = self.parent.ui.lr_penalty_comboBox.currentText()
        penalty = None if penalty == 'None' else penalty
        solver = self.parent.ui.lr_solver_comboBox.currentText()
        penalty_checked = self.parent.ui.lr_penalty_checkBox.isChecked()
        solver_checked = self.parent.ui.lr_solver_checkBox.isChecked()
        c_checked = self.parent.ui.lr_c_checkBox.isChecked()
        c = [c] if c_checked else c_range
        if penalty_checked and solver_checked:
            parameters = [{'penalty': [penalty], 'C': c, 'solver': [solver]}]
        elif penalty_checked and not solver_checked:
            parameters = [{'penalty': [penalty], 'C': c, 'solver': penalty_solver[penalty]}]
        elif not penalty_checked and solver_checked:
            parameters = [{'penalty': solver_penalty[solver], 'C': c, 'solver': [solver]}]
        else:
            parameters = [{'penalty': ('l2', None), 'C': c,
                           'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']},
                          {'penalty': ['l2', 'l1'], 'C': c, 'solver': ['liblinear']},
                          {'penalty': ['elasticnet', 'l1', 'l2', None], 'C': c, 'solver': ['saga']}
                          ]
        return parameters

    def create_nusvc_gridsearch_params(self) -> dict:
        nu_checked = self.parent.ui.svc_nu_check_box.isChecked()
        nu = self.parent.ui.svc_nu_doubleSpinBox.value()
        if nu_checked:
            parameters = {'nu': [nu]}
        else:
            parameters = {'nu': np.arange(0.05, 1.05, 0.05)}
        return parameters

    def create_nn_gridsearch_params(self):
        n_neighbors_checked = self.parent.ui.n_neighbors_checkBox.isChecked()
        weights_checked = self.parent.ui.nn_weights_checkBox.isChecked()
        n_neighbors = self.parent.ui.n_neighbors_spinBox.value()
        weights = self.parent.ui.nn_weights_comboBox.currentText()
        if n_neighbors_checked and not weights_checked:
            parameters = {'n_neighbors': [n_neighbors], 'weights': ['uniform', 'distance']}
        elif not n_neighbors_checked and weights_checked:
            parameters = {'n_neighbors': np.arange(2, 11, 1), 'weights': [weights]}
        elif n_neighbors_checked and weights_checked:
            parameters = {'n_neighbors': [n_neighbors], 'weights': [weights]}
        else:
            parameters = {'n_neighbors': np.arange(2, 11, 1), 'weights': ['uniform', 'distance']}
        return parameters

    def create_mlp_gridsearch_params(self, input_layer_size: int):
        activation_checked = self.parent.ui.activation_checkBox.isChecked()
        solver_checked = self.parent.ui.mlp_solve_checkBox.isChecked()
        hidden_layer_sizes_checked = self.parent.ui.mlp_layer_size_checkBox.isChecked()
        learning_rate_init_checked = self.parent.ui.learning_rate_checkBox.isChecked()
        activation = [self.parent.ui.activation_comboBox.currentText()] if activation_checked \
            else ['identity', 'logistic', 'tanh', 'relu']
        solver = self.parent.ui.solver_mlp_combo_box.currentText()
        hidden_layer_sizes = self.parent.ui.mlp_layer_size_spinBox.value()
        learning_rate_init = self.parent.ui.learning_rate_doubleSpinBox.value()
        h_sizes = np.arange(1, 11) * input_layer_size
        h_sizes = list(h_sizes)
        h_sizes = [hidden_layer_sizes] if hidden_layer_sizes_checked else h_sizes
        learning_rate_init_range = [learning_rate_init] if learning_rate_init_checked else [.1, .01,
                                                                                            .02,
                                                                                            1e-3]

        if solver_checked and solver == 'lbfgs':
            parameters = [
                {'solver': [solver], 'activation': activation, 'hidden_layer_sizes': h_sizes}]
        elif solver_checked and solver != 'lbfgs':
            parameters = [
                {'solver': [solver], 'activation': activation, 'hidden_layer_sizes': h_sizes,
                 'learning_rate_init': learning_rate_init_range}]
        else:
            parameters = [
                {'solver': ['lbfgs'], 'activation': activation, 'hidden_layer_sizes': h_sizes},
                {'activation': activation, 'solver': ['sgd', 'adam'],
                 'hidden_layer_sizes': h_sizes, 'learning_rate_init': learning_rate_init_range}]
        return parameters

    def create_dt_gridsearch_params(self):
        criterion_checked = self.parent.ui.criterion_checkBox.isChecked()
        max_depth_checked = self.parent.ui.dt_max_depth_check_box.isChecked()
        min_samples_split_checked = self.parent.ui.dt_min_samples_split_check_box.isChecked()
        criterion = self.parent.ui.criterion_comboBox.currentText()
        max_depth = self.parent.ui.dt_max_depth_spin_box.value()
        max_depth = None if max_depth == 0 else max_depth
        min_samples_split = self.parent.ui.dt_min_samples_split_spin_box.value()
        criterion = [criterion] if criterion_checked else ["gini", "entropy", "log_loss"]
        max_depth = [max_depth] if max_depth_checked else np.arange(1, 11)
        min_samples_split = [min_samples_split] if min_samples_split_checked else np.arange(2, 21)
        parameters = {'criterion': criterion, 'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'splitter': ['best', 'random'], 'min_samples_leaf': list(range(1, 11)),
                      'max_features': ['sqrt', 'log2', None], 'class_weight': ['balanced']}
        return parameters

    def create_rf_gridsearch_params(self):
        criterion_checked = self.parent.ui.rf_criterion_checkBox.isChecked()
        min_samples_split_checked = self.parent.ui.rf_min_samples_split_checkBox.isChecked()
        n_estimators_checked = self.parent.ui.rf_n_estimators_checkBox.isChecked()
        max_features_checked = self.parent.ui.rf_max_features_checkBox.isChecked()
        criterion = self.parent.ui.rf_criterion_comboBox.currentText()
        min_samples_split = self.parent.ui.rf_min_samples_split_spinBox.value()
        n_estimators = self.parent.ui.rf_n_estimators_spinBox.value()
        max_features = self.parent.ui.rf_max_features_comboBox.currentText()
        max_features = None if max_features == 'None' else max_features
        criterion = [criterion] if criterion_checked else ["gini", "entropy", "log_loss"]
        min_samples_split = [min_samples_split] if min_samples_split_checked else [2, 3, 5, 10]
        n_estimators = [n_estimators] if n_estimators_checked else [2, 3, 5, 10]
        max_features = [max_features] if max_features_checked else ["sqrt", "log2", None]
        parameters = {'criterion': criterion, 'min_samples_split': min_samples_split,
                      'n_estimators': n_estimators, 'max_features': max_features}
        return parameters

    def create_ab_gridsearch_params(self) -> dict:
        """
        Hyperparameters for GridSearchCV.
        n_estimators and learning_rate

        Returns
        -------
            Parameters: dict

        """
        n_estimators_checked = self.parent.ui.ab_n_estimators_checkBox.isChecked()
        learning_rate_checked = self.parent.ui.ab_learning_rate_checkBox.isChecked()
        n_estimators = self.parent.ui.ab_n_estimators_spinBox.value()
        learning_rate = self.parent.ui.ab_learning_rate_doubleSpinBox.value()
        n_estimators = [n_estimators] if n_estimators_checked else np.arange(50, 1050, 50)
        learning_rate = [learning_rate] if learning_rate_checked \
            else [.001, .01, .1, 1., 10., 100., 1000., 10_000., 100_000.]
        return {'n_estimators': n_estimators, 'learning_rate': learning_rate}

    def create_xgb_gridsearch_params(self) -> dict:
        """
        Hyperparameters for GridSearchCV.

        Returns
        -------
            Parameters: dict
        """
        eta_checked = self.parent.ui.xgb_eta_checkBox.isChecked()
        gamma_checked = self.parent.ui.xgb_gamma_checkBox.isChecked()
        max_depth_checked = self.parent.ui.xgb_max_depth_checkBox.isChecked()
        min_child_weight_checked = self.parent.ui.xgb_min_child_weight_checkBox.isChecked()
        colsample_bytree_checked = self.parent.ui.xgb_colsample_bytree_checkBox.isChecked()
        lambda_checked = self.parent.ui.xgb_lambda_checkBox.isChecked()
        eta = self.parent.ui.xgb_eta_doubleSpinBox.value()
        gamma = self.parent.ui.xgb_gamma_spinBox.value()
        max_depth = self.parent.ui.xgb_max_depth_spinBox.value()
        min_child_weight = self.parent.ui.xgb_min_child_weight_spinBox.value()
        colsample_bytree = self.parent.ui.xgb_colsample_bytree_doubleSpinBox.value()
        reg_lambda = self.parent.ui.xgb_lambda_doubleSpinBox.value()
        n_estimators = self.parent.ui.xgb_n_estimators_spinBox.value()
        eta = [eta] if eta_checked else [.001, .01, .1, .2, .3, .4, .5]
        gamma = [gamma] if gamma_checked else [0., .25, .5, 1., 2., 3., 5., 7., 10.]
        max_depth = [max_depth] if max_depth_checked else np.arange(3, 18)
        min_child_weight = [min_child_weight] if min_child_weight_checked else [.0, .5, 1., 2., 3.,
                                                                                4., 5., 6., 7., 10.]
        colsample_bytree = [colsample_bytree] if colsample_bytree_checked else [0.5, 0.6, 0.7, 0.8,
                                                                                0.9, 1.]
        reg_lambda = [reg_lambda] if lambda_checked else [.1, .25, .5, .75, 1., 5., 10., 50., 100.,
                                                          1000.]
        return {'n_estimators': [n_estimators], 'eta': eta, 'gamma': gamma, 'max_depth': max_depth,
                'min_child_weight': min_child_weight, 'colsample_bytree': colsample_bytree,
                'reg_lambda': reg_lambda}

    def create_voting_gridsearch_params(self) -> list:
        """
        Hyperparameters for GridSearchCV.

        Returns
        -------
            Parameters: dict
        """
        estimators = []
        if 'LDA' in self.latest_stat_result:
            best_params = self.latest_stat_result['LDA']['model'].best_params_
            shrinkage = best_params['shrinkage'] if 'shrinkage' in best_params \
                else self.get_lda_shrinkage_parameter_value()
            clf = LinearDiscriminantAnalysis(solver=best_params['solver'], shrinkage=shrinkage)
            estimators.append(('LDA', clf))
        if 'Logistic regression' in self.latest_stat_result:
            best_params = self.latest_stat_result['Logistic regression']['model'].best_params_
            clf = LogisticRegression(best_params['penalty'], C=best_params['C'],
                                     solver=best_params['solver'],
                                     max_iter=10_000, n_jobs=-1,
                                     random_state=self.get_random_state())
            estimators.append(('Logistic regression', clf))
        if 'NuSVC' in self.latest_stat_result:
            best_params = self.latest_stat_result['NuSVC']['model'].best_params_
            clf = NuSVC(nu=best_params['nu'], kernel='linear', probability=True,
                        random_state=self.get_random_state())
            estimators.append(('NuSVC', clf))
        if 'Nearest Neighbors' in self.latest_stat_result:
            best_params = self.latest_stat_result['Nearest Neighbors']['model'].best_params_
            clf = KNeighborsClassifier(best_params['n_neighbors'], weights=best_params['weights'],
                                       n_jobs=-1)
            estimators.append(('Nearest Neighbors', clf))
        if 'GPC' in self.latest_stat_result:
            clf = GaussianProcessClassifier(random_state=self.get_random_state(), n_jobs=-1)
            estimators.append(('GPC', clf))
        if 'Naive Bayes' in self.latest_stat_result:
            clf = GaussianNB()
            estimators.append(('Naive Bayes', clf))
        if 'MLP' in self.latest_stat_result:
            best_params = self.latest_stat_result['MLP']['model'].best_params_
            learning_rate_init = best_params[
                'learning_rate_init'] if 'learning_rate_init' in best_params \
                else self.parent.ui.learning_rate_doubleSpinBox.value()
            clf = MLPClassifier(best_params['hidden_layer_sizes'], best_params['activation'],
                                solver=best_params['solver'], learning_rate='adaptive',
                                learning_rate_init=learning_rate_init,
                                max_iter=self.parent.ui.max_epoch_spinBox.value(),
                                random_state=self.get_random_state())
            estimators.append(('MLP', clf))
        if 'Decision Tree' in self.latest_stat_result:
            best_params = self.latest_stat_result['Decision Tree']['model'].best_params_
            clf = DecisionTreeClassifier(criterion=best_params['criterion'],
                                         random_state=self.get_random_state())
            estimators.append(('Decision Tree', clf))
        if 'Random Forest' in self.latest_stat_result:
            best_params = self.latest_stat_result['Random Forest']['model'].best_params_
            clf = RandomForestClassifier(best_params['n_estimators'],
                                         criterion=best_params['criterion'],
                                         min_samples_split=best_params['min_samples_split'],
                                         max_features=best_params['max_features'],
                                         random_state=self.get_random_state(), n_jobs=-1)
            estimators.append(('Random Forest', clf))
        if 'AdaBoost' in self.latest_stat_result:
            best_params = self.latest_stat_result['AdaBoost']['model'].best_params_
            clf = AdaBoostClassifier(n_estimators=best_params['n_estimators'],
                                     learning_rate=best_params['learning_rate'],
                                     random_state=self.get_random_state())
            estimators.append(('AdaBoost', clf))
        if 'XGBoost' in self.latest_stat_result:
            best_params = self.latest_stat_result['XGBoost']['model'].best_params_
            clf = XGBClassifier(n_estimators=best_params['n_estimators'], eta=best_params['eta'],
                                gamma=best_params['gamma'], max_depth=best_params['max_depth'],
                                min_child_weight=best_params['min_child_weight'],
                                colsample_bytree=best_params['colsample_bytree'],
                                reg_lambda=best_params['reg_lambda'],
                                random_state=self.get_random_state(), n_jobs=-1)
            estimators.append(('XGBoost', clf))
        return estimators

    def corrected_class_labels(self, Y: list[int]) -> list[int]:
        uniq = np.unique(Y)
        n_classes = len(uniq)
        self.old_labels = uniq
        self.new_labels = list(np.arange(0, n_classes, 1))
        new_Y = []
        for i in Y:
            idx = np.argwhere(uniq == i)[0][0]
            new_Y.append(self.new_labels[idx])
        return new_Y

    def get_old_class_label(self, i: int) -> int:
        idx = np.argwhere(self.new_labels == i)[0][0]
        return self.old_labels[idx]

    def _get_plot_colors(self, classes: list[int]) -> list[str]:
        plt_colors = []
        for cls in classes:
            clr = self.parent.context.group_table.get_color_by_group_number(cls)
            color_name = clr.name() if self.parent.ui.sun_Btn.isChecked() else clr.lighter().name()
            plt_colors.append(color_name)
        return plt_colors

    def get_current_dataset_type_cb(self):
        ct = self.parent.ui.dataset_type_cb.currentText()
        if ct == 'Smoothed':
            return self.parent.ui.smoothed_dataset_table_view.model()
        elif ct == 'Baseline corrected':
            return self.parent.ui.baselined_dataset_table_view.model()
        elif ct == 'Decomposed':
            return self.parent.ui.deconvoluted_dataset_table_view.model()
        else:
            return None


    # region plots update
    def update_scores_plot(self, features_in_2d: np.ndarray, y: list[int], y_pred: list[int],
                           classes,
                           model_2d, explained_variance_ratio, dr_method: str) -> None:
        """
        Build DecisionBoundaryDisplay if features_in_2d has 2 or more columns

        Parameters
        ----------
        features_in_2d: {ndarray, sparse matrix} of shape (n_samples, n_features)
            `X` transformed in the new space based on the estimator with
            the best found parameters.
        y: list[int]
            targets
        y_pred: list[int]
            Y predicted by estimator
        classes: array-like of shape (n_classes,)
            Unique class labels.
        model_2d:
            model fitted on transformed data features_in_2d
        explained_variance_ratio : ndarray of shape (n_components,)
            Percentage of variance explained by each of the selected components.
            If ``n_components`` is not set then all components are stored and the
            sum of explained variances is equal to 1.0. Only available when eigen
            or svd solver is used.
        dr_method: str
            'LD', 'PC' or 'PLS-DA'

        Returns
        ----------
            None
        """

        plot_widget = self.parent.ui.decision_boundary_plot_widget
        plot_widget.canvas.axes.cla()
        if explained_variance_ratio is None:
            explained_variance_ratio = [0, 0]
        plt_colors = []
        for cls in classes:
            if self.parent.ui.current_classificator_comboBox.currentText() == 'XGBoost':
                cls = self.get_old_class_label(cls)
            clr = self.parent.context.group_table.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter(140).name())
        tp = y == y_pred  # True Positive
        cmap = LinearSegmentedColormap('', None).from_list('', plt_colors)
        DecisionBoundaryDisplay.from_estimator(model_2d, features_in_2d, grid_resolution=1000,
                                               eps=.5,
                                               antialiased=True, cmap=cmap,
                                               ax=plot_widget.canvas.axes)
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P',
                   '*']
        for i, cls in enumerate(classes):
            true_positive_of_class = tp[y == cls]
            x_i = features_in_2d[y == cls]
            x_tp = x_i[true_positive_of_class]
            x_fp = x_i[~true_positive_of_class]
            mrkr = markers[cls]
            if self.parent.ui.current_classificator_comboBox.currentText() == 'XGBoost':
                cls = self.get_old_class_label(cls)
            style = self.parent.context.group_table.table_widget.model().cell_data_by_idx_col_name(cls, 'Style')
            if style is None:
                color = 'orange'
            else:
                color = style['color'].name()
            plot_widget.canvas.axes.scatter(x_tp[:, 0], x_tp[:, 1], marker=mrkr, color=color,
                                            edgecolor='black', s=60)
            plot_widget.canvas.axes.scatter(x_fp[:, 0], x_fp[:, 1], marker="x", s=60, color=color,
                                            edgecolor='black')
        title1 = dr_method + '-1 (%.2f%%)' % (explained_variance_ratio[0] * 100) if \
        explained_variance_ratio[0] != .0 \
            else dr_method + '-1'
        title2 = dr_method + '-2 (%.2f%%)' % (explained_variance_ratio[1] * 100) if \
        explained_variance_ratio[1] != .0 \
            else dr_method + '-2'
        plot_widget.canvas.axes.set_xlabel(title1,
                                           fontsize=int(environ['axis_label_font_size']))
        plot_widget.canvas.axes.set_ylabel(title2,
                                           fontsize=int(environ['axis_label_font_size']))
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plt.close()
        except ValueError:
            pass

    def update_features_plot(self, X_train: DataFrame, model, feature_names: list[str],
                             target_names: list[str],
                             plt_colors) -> DataFrame:
        """
        Update features plot

        @param plot_widget:
        @param plt_colors:
        @param X_train: dataframe with train data
        @param model: model
        @param feature_names: name for features
        @param target_names: classes name
        @return: top features per class
        """

        # learned coefficients weighted by frequency of appearance
        plot_widget = self.parent.ui.features_plot_widget
        feature_names = np.array(feature_names)
        if isinstance(model, GridSearchCV) or isinstance(model, HalvingGridSearchCV):
            model = model.best_estimator_
        average_feature_effects = model.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
        for i, label in enumerate(target_names):
            if i >= len(average_feature_effects):
                break
            top5 = np.argsort(average_feature_effects[i])[-5:][::-1]
            if i == 0:
                top = DataFrame(feature_names[top5], columns=[label])
                top_indices = top5
            else:
                top[label] = feature_names[top5]
                top_indices = np.concatenate((top_indices, top5), axis=None)
        top_indices = np.unique(top_indices)
        predictive_words = feature_names[top_indices]
        # plot feature effects
        bar_size = 0.5
        padding = 0.75
        y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)
        ax = plot_widget.canvas.axes
        ax.cla()
        for i, label in enumerate(target_names):
            if i >= average_feature_effects.shape[0]:
                break
            ax.barh(y_locs + (i - 2) * bar_size, average_feature_effects[i, top_indices],
                    height=bar_size, label=label,
                    color=plt_colors[i])
        ax.set(yticks=y_locs, yticklabels=predictive_words,
               ylim=[0 - 4 * bar_size, len(top_indices) * (4 * bar_size + padding) - 4 * bar_size])
        ax.legend(loc="best")
        ax.set_xlabel("Average feature effect on the original data",
                      fontsize=int(environ['axis_label_font_size']))
        ax.set_ylabel("Feature", fontsize=int(environ['axis_label_font_size']))
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        return top

    def update_roc_plot_bin(self, model, x_test, y_test, target_names) -> None:
        plot_widget = self.parent.ui.roc_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)",
                color=environ['primaryColor'])
        RocCurveDisplay.from_estimator(model, x_test, y_test, name=target_names[0],
                                       color='darkorange', ax=ax)
        ax.axis("square")
        ax.set_xlabel("False Positive Rate", fontsize=int(environ['axis_label_font_size']))
        ax.set_ylabel("True Positive Rate", fontsize=int(environ['axis_label_font_size']))
        ax.set_title("Receiver Operating Characteristic (ROC) curve")
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size'])})
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_roc_plot(self, y_score, y_onehot_test, target_names, plt_colors) -> None:
        plot_widget = self.parent.ui.roc_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        # store the fpr, tpr, and roc_auc for all averaging strategies
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        n_classes = len(target_names)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
                color="deeppink", linestyle=":", linewidth=3, )
        ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC = {roc_auc['macro']:.2f})",
                color="turquoise", linestyle=":", linewidth=3, )
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)",
                color=environ['primaryColor'])
        for class_id, color in zip(range(n_classes), plt_colors):
            RocCurveDisplay.from_predictions(y_onehot_test[:, class_id], y_score[:, class_id],
                                             name=target_names[class_id], color=color, ax=ax, )
        ax.axis("square")
        ax.set_xlabel("False Positive Rate", fontsize=int(environ['axis_label_font_size']))
        ax.set_ylabel("True Positive Rate", fontsize=int(environ['axis_label_font_size']))
        ax.set_title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size'])})
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_pr_plot_bin(self, y_score_dec_func, y_test, pos_label: int, name=None) -> None:
        plot_widget = self.parent.ui.pr_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        if len(y_score_dec_func.shape) > 1 and y_score_dec_func.shape[1] > 1:
            y_score_dec_func = y_score_dec_func[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_score_dec_func, name=name, ax=ax,
                                                color='darkorange',
                                                pos_label=pos_label)
        ax.set_title("2-class Precision-Recall curve")
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size'])})
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_pr_plot(self, classes: list[int], y_test_bin, y_score_dec_func, colors,
                       target_names) -> None:
        plot_widget = self.parent.ui.pr_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        precision = dict()
        recall = dict()
        average_precision = dict()
        n_classes = len(classes)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i],
                                                                y_score_dec_func[:, i])
            average_precision[i] = average_precision_score(y_test_bin[:, i], y_score_dec_func[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(),
                                                                        y_score_dec_func.ravel())
        average_precision["micro"] = average_precision_score(y_test_bin, y_score_dec_func,
                                                             average="micro")
        display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"],
                                         average_precision=average_precision["micro"])
        display.plot(ax=ax, name="Micro-average precision-recall", color="deeppink", linestyle=":",
                     linewidth=3, )
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = ax.plot(x[y >= 0], y[y >= 0], color="deeppink", alpha=0.2)
            ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02), color="deeppink")
        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i],
                                             average_precision=average_precision[i], )
            target_name = target_names[i] if i < len(target_names) else 'NoName'
            display.plot(ax=ax, name=f"Precision-recall for {target_name}", color=color,
                         linewidth=2)
        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best",
                  prop={'size': int(environ['plot_font_size'])})
        ax.set_title("Extension of Precision-Recall curve to multi-class")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_dm_plot(self, y_test, x_test, target_names, model) -> None:
        plot_widget = self.parent.ui.dm_plot_widget
        plot_widget.canvas.axes.cla()
        ax = plot_widget.canvas.axes
        # ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, colorbar=False)
        try:
            ax.xaxis.set_ticklabels(target_names,
                                    fontsize=int(environ['axis_label_font_size']))
            ax.yaxis.set_ticklabels(target_names,
                                    fontsize=int(environ['axis_label_font_size']))
        except ValueError as err:
            error(err)
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        _ = ax.set_title(f"Confusion Matrix for {model.__class__.__name__}\non the test data")

        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def do_update_shap_plots(self, classificator_type):
        if classificator_type not in self.latest_stat_result:
            return
        target_names = self.latest_stat_result[classificator_type]['target_names']
        num = np.where(target_names == self.parent.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_means_plot(i, classificator_type)
        self.update_shap_beeswarm_plot(False, i, classificator_type)
        self.update_shap_scatter_plot(False, i, classificator_type)
        self.update_shap_heatmap_plot(False, i, classificator_type)

    def do_update_shap_plots_by_instance(self, classificator_type):
        if classificator_type not in self.latest_stat_result:
            return
        target_names = self.latest_stat_result[classificator_type]['target_names']
        num = np.where(target_names == self.parent.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_force(i, classificator_type)
        self.update_shap_force(i, classificator_type, True)
        self.update_shap_decision_plot(i, classificator_type)
        self.update_shap_waterfall_plot(i, classificator_type)

    # endregion

    # region SHAP
    def update_shap_means_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        plot_widget = self.parent.ui.shap_means
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass

        if classificator_type not in self.latest_stat_result:
            return
        result = self.latest_stat_result[classificator_type]
        if 'shap_values' not in result:
            return
        shap_values = result['shap_values']
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if isinstance(shap_values, list) and not binary:
            shap_values = shap_values[class_i]
        elif not isinstance(shap_values, list) and len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.bar(shap_values, show=True, max_display=20, ax=fig.gca(), fig=fig)
        self.parent.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_beeswarm_plot(self, binary: bool, class_i: int = 0,
                                  classificator_type: str = 'LDA') -> None:
        plot_widget = self.parent.ui.shap_beeswarm

        if self.parent.ui.sun_Btn.isChecked():
            color = None
            # color = plt.get_cmap("cool")
        else:
            color = plt.get_cmap("cool")
        if binary:
            class_i = 0
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        if classificator_type not in self.latest_stat_result:
            return
        if 'shap_values' not in self.latest_stat_result[classificator_type]:
            return
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.beeswarm(shap_values, show=False, color=color, max_display=20, ax=fig.gca(),
                            fig=fig)
        self.parent.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_scatter_plot(self, binary: bool, class_i: int = 0,
                                 classificator_type: str = 'LDA') -> None:
        plot_widget = self.parent.ui.shap_scatter

        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        current_feature = self.parent.ui.current_feature_comboBox.currentText()
        cmap = None if self.parent.ui.sun_Btn.isChecked() else plt.get_cmap("cool")
        if 'shap_values' not in self.latest_stat_result[classificator_type]:
            return
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if binary:
            class_i = 0
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        if current_feature not in shap_values.feature_names:
            return
        ct = self.parent.ui.coloring_feature_comboBox.currentText()
        color = shap_values if ct == '' else shap_values[:, ct]
        shap.plots.scatter(shap_values[:, current_feature], color=color, show=False, cmap=cmap,
                           ax=ax,
                           axis_color=self.parent.plot_text_color.name())
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
        except ValueError:
            pass

    def update_shap_heatmap_plot(self, binary: bool, class_i: int = 0,
                                 classificator_type: str = 'LDA') -> None:
        plot_widget = self.parent.ui.shap_heatmap

        if binary:
            class_i = 0
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        if classificator_type not in self.latest_stat_result:
            return
        if 'shap_values' not in self.latest_stat_result[classificator_type]:
            return
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.heatmap(shap_values, show=False, max_display=20, ax=fig.gca(), fig=fig)
        self.parent.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)

    def update_shap_force(self, class_i: int = 0, classificator_type: str = 'LDA',
                          full: bool = False) -> None:
        if classificator_type not in self.latest_stat_result:
            warning(classificator_type + ' not in self.latest_stat_result. def update_shap_force')
            return
        table_model = self.get_current_dataset_type_cb()
        if table_model is None:
            return

        current_instance = self.parent.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = table_model.first_index
        else:
            sample_id = table_model.idx_by_column_value('Filename', current_instance)
        result = self.latest_stat_result[classificator_type]
        if 'model' not in result:
            return
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if 'explainer' not in result and 'expected_value' not in result:
            return
        elif 'expected_value' in result:
            expected_value = result['expected_value']
        elif 'explainer' in result:
            expected_value = result['explainer'].expected_value
        else:
            return
        if 'shap_values_legacy' not in result:
            return
        shap_values = result['shap_values_legacy']
        if 'X_display' not in result:
            return
        X_display = result['X_display']
        if isinstance(X_display, DataFrame) and X_display.empty or not isinstance(X_display,
                                                                                  DataFrame):
            return
        if full and isinstance(shap_values, list):
            shap_v = shap_values[class_i]
        elif isinstance(shap_values, list):
            shap_v = shap_values[class_i][sample_id]
        elif full:
            shap_v = shap_values
        else:
            shap_v = shap_values[sample_id]
        if full:
            x_d = X_display.iloc[:, 2:]
        else:
            x_d = X_display.iloc[:, 2:].loc[sample_id]
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[class_i]
        if (not full and shap_v.shape[0] != len(x_d.values)) \
                or (full and shap_v[0].shape[0] != len(x_d.loc[table_model.first_index].values)):
            err = 'Force plot не смог обновиться. Количество shap_values features != количеству X features.' \
                  ' Возможно была изменена таблица с обучающими данными.' \
                  ' Нужно пересчитать %s' % classificator_type
            print(err)
            error(err)
            return
        try:
            force_plot = shap.force_plot(expected_value, shap_v, x_d,
                                         feature_names=result['feature_names'])
        except ValueError:
            error('failed update_shap_force')
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        if full:
            self.latest_stat_result[classificator_type]['shap_html_full'] = shap_html
        else:
            self.latest_stat_result[classificator_type]['shap_html'] = shap_html

    def update_shap_decision_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type not in self.latest_stat_result:
            return
        plot_widget = self.parent.ui.shap_decision

        table_model = self.get_current_dataset_type_cb()
        if table_model is None:
            return
        current_instance = self.parent.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = None
        else:
            sample_id = table_model.idx_by_column_value('Filename', current_instance)
        result = self.latest_stat_result[classificator_type]
        if 'model' not in result:
            return
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if 'explainer' not in result and 'expected_value' not in result:
            return
        elif 'expected_value' in result:
            expected_value = result['expected_value']
        elif 'explainer' in result:
            expected_value = result['explainer'].expected_value
        else:
            return
        if 'shap_values_legacy' not in result:
            return
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        shap_values = result['shap_values_legacy']
        if 'X_display' not in result:
            return
        X_display = result['X_display']
        if isinstance(X_display, DataFrame) and X_display.empty or not isinstance(X_display,
                                                                                  DataFrame):
            return
        misclassified = result['misclassified']
        title = 'all'
        if sample_id is None and isinstance(shap_values, list):
            shap_v = shap_values[class_i]
        elif isinstance(shap_values, list):
            shap_v = shap_values[class_i][sample_id]
        elif sample_id is None:
            shap_v = shap_values
        else:
            shap_v = shap_values[sample_id]

        if sample_id is None:
            x_d = X_display.iloc[:, 2:]
        else:
            x_d = X_display.iloc[:, 2:].loc[sample_id]
            misclassified = misclassified[sample_id]
            title = current_instance
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[class_i]
        if (sample_id is not None and shap_v.shape[0] != len(x_d.values)) \
                or (sample_id is None and shap_v[0].shape[0] != len(
            x_d.loc[table_model.first_index].values)):
            err = 'Decision plot не смог обновиться. Количество shap_values features != количеству X features.' \
                  ' Возможно была изменена таблица с обучающими данными.' \
                  ' Нужно пересчитать %s' % classificator_type
            print(err)
            error(err)
            return
        feature_display_range_max = -self.parent.ui.feature_display_max_spinBox.value() \
            if self.parent.ui.feature_display_max_checkBox.isChecked() else None
        try:
            shap.plots.decision(expected_value, shap_v, x_d, title=title, ignore_warnings=True,
                                feature_display_range=slice(None, feature_display_range_max, -1),
                                highlight=misclassified, ax=ax, fig=fig)
        except ValueError:
            error(classificator_type + ' decision plot trouble. {!s}'.format(ValueError))
            return
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)

    def update_shap_waterfall_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type not in self.latest_stat_result:
            return
        plot_widget = self.parent.ui.shap_waterfall
        model = self.get_current_dataset_type_cb()
        if model.rowCount() == 0:
            return
        q_res = model.dataframe()
        features_names = list(q_res.columns[2:])
        n_features = len(features_names)
        model = self.get_current_dataset_type_cb()
        if model is None:
            return
        current_instance = self.parent.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = 0
        else:
            sample_id = model.idx_by_column_value('Filename', current_instance)
        result = self.latest_stat_result[classificator_type]
        if 'model' not in result:
            return
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        if 'shap_values' not in result:
            return
        shap_values = result['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        max_display = self.parent.ui.feature_display_max_spinBox.value() \
            if self.parent.ui.feature_display_max_checkBox.isChecked() else n_features
        try:
            shap.plots.waterfall(shap_values[sample_id], max_display, ax=ax, fig=fig)
        except ValueError:
            error(classificator_type + ' waterfall plot trouble. {!s}'.format(ValueError))
            return
        # self.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)

    def update_force_single_plots(self, cl_type: str = '') -> None:
        plot_widget = self.parent.ui.force_single

        if cl_type not in self.latest_stat_result:
            return
        if 'shap_html' in self.latest_stat_result[cl_type]:
            shap_html = self.latest_stat_result[cl_type]['shap_html']
            if self.parent.ui.sun_Btn.isChecked():
                shap_html = re.sub(r'#ffe', "#000", shap_html)
                shap_html = re.sub(r'#001', "#fff", shap_html)
            else:
                shap_html = re.sub(r'#000', "#ffe", shap_html)
                shap_html = re.sub(r'#fff', "#001", shap_html)
            plot_widget.setHtml(shap_html)
            plot_widget.page().setBackgroundColor(QColor(self.parent.plot_background_color))
            plot_widget.reload()
            self.latest_stat_result[cl_type]['shap_html'] = shap_html
        else:
            error(cl_type + ' shap_html not in self.latest_stat_result')

    def update_force_full_plots(self, cl_type: str = '') -> None:
        plot_widget = self.parent.ui.force_full
        if cl_type not in self.latest_stat_result:
            return
        if 'shap_html_full' in self.latest_stat_result[cl_type]:
            shap_html = self.latest_stat_result[cl_type]['shap_html_full']
            if self.parent.ui.sun_Btn.isChecked():
                shap_html = re.sub(r'#ffe', "#000", shap_html)
                shap_html = re.sub(r'#001', "#fff", shap_html)
            else:
                shap_html = re.sub(r'#000', "#ffe", shap_html)
                shap_html = re.sub(r'#fff', "#001", shap_html)
            plot_widget.setHtml(shap_html)
            plot_widget.page().setBackgroundColor(QColor(self.parent.plot_background_color_web))
            plot_widget.reload()
            self.latest_stat_result[cl_type]['shap_html_full'] = shap_html
        else:
            error(cl_type + ' shap_html_full not in self.latest_stat_result')

    # endregion

    def update_plots(self, cl_type) -> None:
        if cl_type not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result[cl_type]
        model = model_results['model']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        y_pred_2d = model_results['y_pred_2d']
        model_2d = model_results['model_2d']
        x_test = model_results['x_test']
        x_train = model_results['x_train']
        y_test = model_results['y_test']
        y_train = model_results['y_train']
        target_names = model_results['target_names']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        if 'y_score_dec_func' not in model_results:
            y_score_dec_func = y_score
            y_score_dec_func_x = y_score
        else:
            y_score_dec_func = model_results['y_score_dec_func']
            y_score_dec_func_x = model.decision_function(model_results['X'])
        if cl_type == 'LDA':
            dr_method = 'LD'
        else:
            dr_method = 'PC'
        y_test_bin = model_results['y_test_bin']
        explained_variance_ratio = None
        if 'explained_variance_ratio' in model_results:
            explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)
        binary = len(classes) == 2

        # page 1 of fit results
        if cl_type in ['LDA', 'Logistic regression', 'NuSVC']:
            self.build_decision_score_plot(y_score_dec_func_x, target_names, model_results['Y'])
        else:
            self.parent.ui.decision_score_plot_widget.canvas.axes.axes.cla()
            try:
                self.parent.ui.decision_score_plot_widget.canvas.draw()
            except ValueError:
                pass
        if len(features_in_2d.shape) > 1 and features_in_2d.shape[1] > 1:
            self.update_scores_plot(features_in_2d, y_train_plus_test, y_pred_2d, classes, model_2d,
                                    explained_variance_ratio, dr_method)
        self.update_dm_plot(y_test, x_test, target_names, model)
        if binary:
            self.update_roc_plot_bin(model, x_test, y_test, target_names)
            self.update_pr_plot_bin(y_score_dec_func, y_test, classes[0], cl_type)
        else:
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors)
            self.update_pr_plot(classes, y_test_bin, y_score_dec_func, plt_colors, target_names)
        # page 2 of fit results
        self.build_permutation_importance_plot(model, x_test, y_test, True)
        self.build_permutation_importance_plot(model, x_train, y_train)
        if cl_type in ['LDA', 'Logistic regression', 'NuSVC']:
            top = self.update_features_plot(model_results['x_train'], model,
                                            model_results['feature_names'],
                                            target_names, plt_colors)
            self.top_features[cl_type] = top
        else:
            self.parent.ui.features_plot_widget.canvas.axes.axes.cla()
            try:
                self.parent.ui.features_plot_widget.canvas.draw()
            except ValueError:
                pass
        if cl_type == 'Decision Tree':
            self.update_plot_tree(model.best_estimator_, model_results['feature_names'],
                                  target_names)
            self.update_features_plot_random_forest(model.best_estimator_,
                                                    "Decision Tree Feature Importances (MDI)")
        elif cl_type == 'Random Forest':
            self.update_plot_tree(model.best_estimator_.estimators_[0],
                                  model_results['feature_names'], target_names)
            self.update_features_plot_random_forest(model.best_estimator_,
                                                    "Random Forest Feature Importances (MDI)")
        elif cl_type == 'AdaBoost':
            self.update_plot_tree(model.best_estimator_.estimators_[0],
                                  model_results['feature_names'], target_names)
            self.update_features_plot_random_forest(model.best_estimator_,
                                                    "AdaBoost Feature Importances (MDI)")
        if cl_type == 'XGBoost' and self.parent.ui.dataset_type_cb.currentText() == 'Decomposed':
            self.update_features_plot_xgboost(model.best_estimator_)
        elif cl_type == 'XGBoost':
            self.update_features_plot_random_forest(model.best_estimator_,
                                                    "XGBoost Feature Importances (MDI)")
        if cl_type == 'XGBoost':
            self.update_xgboost_tree_plot(model.best_estimator_)
        # page 3 of fit results
        if binary:
            self.build_calibration_plot(model, x_test, y_test)
            self.build_dev_curve_plot(model, x_test, y_test)

    def build_permutation_importance_plot(self, estimator, x, y, test: bool = False) -> None:
        plot_widget = self.parent.ui.perm_imp_test_plot_widget if test else self.parent.ui.perm_imp_train_plot_widget
        plot_widget.canvas.axes.cla()
        ax = plot_widget.canvas.axes
        binary = len(estimator.classes_) == 2
        scoring = 'f1' if binary else 'f1_micro'
        with parallel_backend('multiprocessing', n_jobs=-1):
            result = permutation_importance(estimator, x, y, scoring=scoring, n_repeats=10,
                                            n_jobs=-1,
                                            random_state=self.get_random_state())
        sorted_importances_idx = result.importances_mean.argsort()
        importances = DataFrame(
            result.importances[sorted_importances_idx].T,
            columns=x.columns[sorted_importances_idx],
        )
        importances.plot.box(vert=False, whis=10, ax=ax)
        t = 'test' if test else 'train'
        ax.set_title("Permutation Importances (%s set)" % t)
        try:
            ax.axvline(x=0, color="k", linestyle="--")
        except np.linalg.LinAlgError:
            pass
        ax.set_xlabel("Decrease in accuracy score")
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def build_partial_dependence_plot(self, clf, X) -> None:
        plot_widget = self.parent.ui.partial_depend_plot_widget
        plot_widget.canvas.figure.clf()
        ax = plot_widget.canvas.axes
        ax.cla()
        feature_names = list(X.columns)
        f1 = self.parent.ui.current_dep_feature1_comboBox.currentText()
        f2 = self.parent.ui.current_dep_feature2_comboBox.currentText()
        try:
            with parallel_backend('multiprocessing', n_jobs=-1):
                PartialDependenceDisplay.from_estimator(clf, X, [f1, (f1, f2)],
                                                        target=clf.classes_[0],
                                                        feature_names=feature_names, ax=ax)
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def build_calibration_plot(self, clf, X_test, y_test) -> None:
        plot_widget = self.parent.ui.calibration_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        with parallel_backend('multiprocessing', n_jobs=-1):
            CalibrationDisplay.from_estimator(clf, X_test, y_test, pos_label=clf.classes_[0], ax=ax)
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def build_dev_curve_plot(self, clf, X_test, y_test) -> None:
        plot_widget = self.parent.ui.det_curve_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        with parallel_backend('multiprocessing', n_jobs=-1):
            DetCurveDisplay.from_estimator(clf, X_test, y_test, pos_label=clf.classes_[0], ax=ax)
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def refresh_learning_curve(self):
        cl_type = self.parent.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result[cl_type]
        model = model_results['model']
        self.build_learning_curve(model, model_results['X'], model_results['Y'])

    def build_learning_curve(self, clf, X, y) -> None:
        plot_widget = self.parent.ui.learning_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with parallel_backend('multiprocessing', n_jobs=-1):
                LearningCurveDisplay.from_estimator(clf, X, y, ax=ax, verbose=0)
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    # region LDA

    def build_decision_score_plot(self, decision_function_values, class_names, Y) -> None:
        main_window = self.parent
        plot_widget = main_window.ui.decision_score_plot_widget
        ax = plot_widget.canvas.axes.axes
        ax.cla()

        dfv = np.array(decision_function_values)
        if len(dfv.shape) > 1 and dfv.shape[1] > 1:
            decision_function_values_1d = []
            for i, v_row in enumerate(dfv):
                try:
                    decision_function_values_1d.append(v_row[Y[i] - 1])
                except IndexError:
                    break
            decision_function_values_1d = np.array(decision_function_values_1d)
        else:
            decision_function_values_1d = decision_function_values
        palette = color_palette(self.parent.context.group_table.table_widget.model().groups_colors)
        c_n = []
        for i in Y:
            class_name = class_names[i - 1] if i <= len(class_names) else 'NoName'
            c_n.append(class_name)
        df = DataFrame({'Decision score value': decision_function_values_1d, 'label': c_n})
        histplot(data=df, x='Decision score value', hue='label', palette=palette, kde=True, ax=ax)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
        except ValueError:
            pass

    # endregion

    # region Random Forest
    def update_features_plot_random_forest(self, model, title: str) -> None:
        plot_widget = self.parent.ui.features_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        if self.parent.ui.dataset_type_cb.currentText() == 'Decomposed':
            mdi_importances = Series(model.feature_importances_,
                                     index=model.feature_names_in_).sort_values(
                ascending=True)
            mdi_importances = mdi_importances[mdi_importances > 0]
            ax.barh(mdi_importances.index, mdi_importances,
                    color=environ['primaryColor'])
            ax.set_xlabel('MDI importance', fontsize=int(environ['axis_label_font_size']))
            ax.set_ylabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=int(environ['axis_label_font_size']))
        else:
            k = []
            for i in model.feature_names_in_:
                k.append(float(i.replace('k', '').replace('_a', '').replace('_x0', '')))
            ser = Series(model.feature_importances_, index=k)
            ax.plot(ser)
            ax.set_xlabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=int(environ['axis_label_font_size']))
            ax.set_ylabel('MDI importance', fontsize=int(environ['axis_label_font_size']))

        ax.set_title(title)
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    # endregion

    # region PCA

    def update_pca_plots(self) -> None:
        """
        Update all 'PCA plots and fields
        @return: None
        """
        if 'PCA' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['PCA']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        explained_variance_ratio = model_results['explained_variance_ratio']

        self.update_scores_plot_pca(features_in_2d, y_train_plus_test,
                                    self.parent.ui.pca_scores_plot_widget,
                                    explained_variance_ratio, model_results['target_names'])
        if 'loadings' in model_results:
            self.parent.ui.pca_features_table_view.model().set_dataframe(model_results['loadings'])
            self.update_pca_loadings_plot(model_results['loadings'], explained_variance_ratio)

    def update_scores_plot_pca(self, features_in_2d: np.ndarray, y: list[int], plot_widget,
                               explained_variance_ratio, target_names) -> None:
        """
        @param target_names:
        @param plot_widget:
        @param features_in_2d: transformed 2d
        @param y: fact labels
        @param explained_variance_ratio:
        @return:
        """
        ax = plot_widget.canvas.axes
        ax.cla()
        classes = np.unique(y)
        plt_colors = []
        for cls in classes:
            clr = self.parent.context.group_table.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter().name())
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P',
                   '*']
        for i, cls in enumerate(classes):
            x_i = features_in_2d[y == cls]
            mrkr = markers[cls]
            color = self.parent.context.group_table.table_widget.model().cell_data_by_idx_col_name(cls, 'Style')[
                'color'].name()
            ax.scatter(x_i[:, 0], x_i[:, 1], marker=mrkr, color=color,
                       edgecolor='black', s=60, label=target_names[i])
        if plot_widget == self.parent.ui.pca_scores_plot_widget:
            ax.set_xlabel('PC-1 (%.2f%%)' % (explained_variance_ratio[0] * 100),
                          fontsize=int(environ['plot_font_size']))
            ax.set_ylabel('PC-2 (%.2f%%)' % (explained_variance_ratio[1] * 100),
                          fontsize=int(environ['plot_font_size']))
        else:
            ax.set_xlabel('PLS-DA-1 (%.2f%%)' % (explained_variance_ratio[0] * 100),
                          fontsize=int(environ['plot_font_size']))
            ax.set_ylabel('PLS-DA-2 (%.2f%%)' % (explained_variance_ratio[1] * 100),
                          fontsize=int(environ['plot_font_size']))
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size']) - 2})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_pca_loadings_plot(self, loadings, explained_variance_ratio) -> None:
        plot_widget = self.parent.ui.pca_loadings_plot_widget
        plot_widget.canvas.axes.cla()
        features_names = list(loadings.axes[0])
        plot_widget.canvas.axes.scatter(loadings['PC1'], loadings['PC2'],
                                        color=environ['primaryColor'], s=60)
        for i, txt in enumerate(features_names):
            plot_widget.canvas.axes.annotate(txt, (loadings['PC1'][i], loadings['PC2'][i]))
        plot_widget.canvas.axes.set_xlabel('PC-1 (%.2f%%)' % (explained_variance_ratio[0] * 100),
                                           fontsize=int(environ['plot_font_size']))
        plot_widget.canvas.axes.set_ylabel('PC-2 (%.2f%%)' % (explained_variance_ratio[1] * 100),
                                           fontsize=int(environ['plot_font_size']))
        plot_widget.canvas.axes.set_title("PCA Loadings")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # endregion

    # region XGBoost
    def update_features_plot_xgboost(self, model) -> None:
        plot_widget = self.parent.ui.features_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        max_num_features = self.parent.ui.feature_display_max_spinBox.value() \
            if self.parent.ui.feature_display_max_checkBox.isChecked() else None
        xgboost.plot_importance(model, ax, max_num_features=max_num_features, grid=False)
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def tune_xgboost_params(self) -> None:
        """
        Bayesian Optimization with HYPEROPT

        Returns
        -------

        """

        current_dataset = self.parent.ui.dataset_type_cb.currentText()
        if current_dataset == 'Smoothed' and self.parent.ui.smoothed_dataset_table_view.model().rowCount() == 0 \
                or current_dataset == 'Baseline corrected' \
                and self.parent.ui.baselined_dataset_table_view.model().rowCount() == 0 \
                or current_dataset == 'Decomposed' \
                and self.parent.ui.deconvoluted_dataset_table_view.model().rowCount() == 0:
            return
        X, Y, _, _, _ = self.dataset_for_ml()
        Y = list(Y)
        Y = list(map(lambda x: x - 1, Y))
        rnd_state = self.get_random_state()
        test_size = self.parent.ui.test_data_ratio_spinBox.value() / 100.
        ros = RandomOverSampler(random_state=rnd_state)
        X, Y = ros.fit_resample(X, Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                            random_state=rnd_state)
        trials = Trials()
        space = {'eta': hp.uniform('eta', 0.01, .5),
                 'max_depth': hp.quniform("max_depth", 3, 18, 1),
                 'gamma': hp.uniform('gamma', 0, 10),
                 'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                 'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                 'n_estimators': 180,
                 'seed': hp.uniform('seed', 0, 100),
                 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                 }
        best_hyperparams = fmin(fn=objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)
        print(best_hyperparams)

    # endregion
