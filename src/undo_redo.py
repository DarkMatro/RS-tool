import warnings
from asyncio import create_task, sleep
from copy import deepcopy
from datetime import datetime
from gc import collect
from logging import info, debug

import numpy as np
from asyncqtpy import asyncSlot
from joblib import parallel_backend
from pandas import DataFrame
from qtpy.QtWidgets import QUndoCommand
from shap import Explainer, KernelExplainer
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from src.mutual_functions.static_functions import \
    calculate_vips
from src.stages.ml.functions.fit_classificators import model_metrics

warnings.filterwarnings('ignore')

class CommandAfterFittingStat(QUndoCommand):
    """
    undo / redo fit classificator model

    Parameters
    ----------
    main_window
        Main window class
    result: of fitting model
    cl_type: type of classificator
        value to write by index
    description : str
        Description to set in tooltip
    """

    def __init__(self, main_window, result: dict, cl_type: str, description: str) -> None:
        super(CommandAfterFittingStat, self).__init__(description)

        self.stat_result_new = None
        info('init FitIntervalChanged {!s}:'.format(str(description)))
        self.setText(description)
        self.result = result
        self.cl_type = cl_type
        if cl_type not in main_window.stat_analysis_logic.latest_stat_result:
            self.stat_result_old = None
        else:
            self.stat_result_old = deepcopy(
                main_window.stat_analysis_logic.latest_stat_result[cl_type])
        self.UndoAction = main_window.action_undo
        self.RedoAction = main_window.action_redo
        self.UndoStack = main_window.undoStack
        self.mw = main_window
        self.stat_result_new = self.create_stat_result()

    def create_stat_result(self) -> dict:
        if self.cl_type == 'LDA':
            stat_result_new = self.create_stat_result_lda()
        elif self.cl_type in ['Logistic regression', 'NuSVC']:
            stat_result_new = self.create_stat_result_lr_svc(self.cl_type)
        elif self.cl_type == 'PCA':
            stat_result_new = self.create_stat_result_pca()
        elif self.cl_type == 'PLS-DA':
            stat_result_new = self.create_stat_result_plsda()
        else:
            stat_result_new = self.create_stat_result_rest(self.cl_type)
        if self.mw.ui.dataset_type_cb.currentText() == 'Smoothed':
            X_display = self.mw.ui.smoothed_dataset_table_view.model().dataframe()
        elif self.mw.ui.dataset_type_cb.currentText() == 'Baseline corrected':
            X_display = self.mw.ui.baselined_dataset_table_view.model().dataframe()
        else:
            X_display = self.mw.ui.deconvoluted_dataset_table_view.model().dataframe()
            ignored_features = self.mw.ui.ignore_dataset_table_view.model().ignored_features
            X_display = X_display.drop(ignored_features, axis=1)
        stat_result_new['X_display'] = X_display
        return stat_result_new

    def create_stat_result_lda(self) -> dict:
        result = deepcopy(self.result)
        if 'y_pred_2d' in result:
            y_pred_2d = result['y_pred_2d']
        else:
            y_pred_2d = None
        result['y_pred_2d'] = y_pred_2d
        X = result['X']
        y_score_dec_func = result['y_score_dec_func']
        label_binarizer = LabelBinarizer().fit(result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(result['y_test'])
        binary = True if len(result['model'].classes_) == 2 else False
        if result['features_in_2d'] is not None:
            with parallel_backend('multiprocessing', n_jobs=-1):
                classifier = OneVsRestClassifier(make_pipeline(StandardScaler(), result['model']))
                classifier.fit(result['x_train'], result['y_train'])
                y_score_dec_func = classifier.decision_function(result['x_test'])
        metrics_result = model_metrics(result['y_test'], result['y_pred_test'], result['y_score'],
                                       binary, result['target_names'])
        metrics_result['accuracy_score_train'] = result['accuracy_score_train']
        result['metrics_result'] = metrics_result

        result['y_score_dec_func'] = y_score_dec_func

        model = result['model']
        model = model.best_estimator_ if isinstance(model, GridSearchCV) or isinstance(model,
                                                                                       HalvingGridSearchCV) \
            else model
        explainer = Explainer(model, X)
        shap_values = explainer(X)
        shap_values_legacy = explainer.shap_values(X)
        result['explainer'] = explainer
        result['shap_values'] = shap_values
        result['shap_values_legacy'] = shap_values_legacy

        return result

    def create_stat_result_lr_svc(self, cl_type: str) -> dict:
        result = deepcopy(self.result)
        y_test = self.result['y_test']
        result['y_train_plus_test'] = np.concatenate((self.result['y_train'], y_test))
        binary = True if len(self.result['model'].classes_) == 2 else False
        metrics_result = model_metrics(y_test, self.result['y_pred_test'], result['y_score'],
                                       binary, self.result['target_names'])
        metrics_result['accuracy_score_train'] = self.result['accuracy_score_train']
        result['metrics_result'] = metrics_result
        with parallel_backend('multiprocessing', n_jobs=-1):
            label_binarizer = LabelBinarizer().fit(self.result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(y_test)
        if cl_type == 'Logistic regression':
            explainer = Explainer(self.result['model'].best_estimator_, self.result['X'])
            result['shap_values'] = explainer(self.result['X'])
            result['explainer'] = explainer
            result['shap_values_legacy'] = explainer.shap_values(self.result['X'])
        else:
            kernel_explainer = KernelExplainer(self.result['model'].best_estimator_.predict_proba,
                                               self.result['X'])
            explainer = Explainer(self.result['model'].best_estimator_, self.result['X'],
                                  max_evals=2 * len(self.result['feature_names']) + 1)
            result['shap_values'] = explainer(self.result['X'])
            result['explainer'] = kernel_explainer
            result['shap_values_legacy'] = kernel_explainer.shap_values(self.result['X'])
        return result

    def create_stat_result_rest(self, cl_type: str) -> dict:
        result = deepcopy(self.result)
        y_test = result['y_test']
        x_train = result['x_train']
        model = result['model']
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        result['y_train_plus_test'] = np.concatenate((result['y_train'], y_test))
        binary = True if len(model.classes_) == 2 else False
        metrics_result = model_metrics(y_test, result['y_pred_test'], result['y_score'], binary,
                                       result['target_names'])
        metrics_result['accuracy_score_train'] = result['accuracy_score_train']
        result['metrics_result'] = metrics_result
        with parallel_backend('multiprocessing', n_jobs=-1):
            label_binarizer = LabelBinarizer().fit(result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(y_test)
        func = lambda x: model.predict_proba(x)[:, 1]
        med = x_train.median().values.reshape((1, x_train.shape[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = Explainer(func, med, feature_names=result['feature_names'])
            kernel_explainer = KernelExplainer(func, med, feature_names=result['feature_names'])
            result['shap_values'] = explainer(result['X'])
            result['expected_value'] = kernel_explainer.expected_value
            result['shap_values_legacy'] = kernel_explainer.shap_values(result['X'])
        return result

    def create_stat_result_pca(self) -> dict:
        result = deepcopy(self.result)
        result['y_train_plus_test'] = np.concatenate(
            (self.result['y_train'], self.result['y_test']))
        result['loadings'] = DataFrame(result['model'].components_.T, columns=['PC1', 'PC2'],
                                       index=result['feature_names'])
        return result

    def create_stat_result_plsda(self) -> dict:
        result = deepcopy(self.result)
        result['y_train_plus_test'] = np.concatenate(
            (self.result['y_train'], self.result['y_test']))
        result['vips'] = calculate_vips(result['model'])
        return result

    @asyncSlot()
    async def redo(self) -> None:
        debug('redo CommandAfterFittingStat')
        if self.stat_result_new is not None:
            self.mw.stat_analysis_logic.latest_stat_result[self.cl_type] = self.stat_result_new
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        info('undo CommandAfterFittingStat')
        self.mw.stat_analysis_logic.latest_stat_result[self.cl_type] = self.stat_result_old
        if self.stat_result_old is None:
            self.mw.clear_selected_step(self.cl_type)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        if self.cl_type == 'PCA' and self.mw.stat_analysis_logic.latest_stat_result[
            'PCA'] is not None:
            self.mw.loop.run_in_executor(None, self.mw.stat_analysis_logic.update_pca_plots)
        elif self.cl_type != 'PCA' and self.cl_type != 'PLS-DA' \
                and self.mw.stat_analysis_logic.latest_stat_result[self.cl_type] is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings('once')
                self.mw.loop.run_in_executor(None, self.mw.stat_analysis_logic.update_plots,
                                             self.cl_type)
        self.mw.stat_analysis_logic.update_force_single_plots(self.cl_type)
        self.mw.stat_analysis_logic.update_force_full_plots(self.cl_type)
        self.mw.context.ml.update_stat_report_text(self.cl_type)
        info('stop CommandAfterFittingStat')
        self.update_undo_redo_tooltips()
        self.mw.context.set_modified()

        time_end = self.mw.time_start
        if not self.mw.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.mw.time_start = None
        self.mw.ui.statusBar.showMessage('Model fitting completed for ' + str(seconds) + ' sec.',
                                         550000)
        collect(2)
        await sleep(0)
