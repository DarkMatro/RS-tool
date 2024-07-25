from asyncio import get_event_loop
from logging import error
from os import environ

import numpy as np
import pandas as pd
import psutil
import xgboost
from qtpy.QtGui import QMouseEvent
from asyncqtpy import asyncSlot
from qtpy.QtCore import QObject, Qt
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, train_test_split
from sklearn.tree import plot_tree
from multiprocessing import cpu_count
from qfluentwidgets import MessageBox
from src import get_parent, get_config, ObservableDict
from src.data.default_values import classificator_funcs, objectives
from src.stages.fitting.classes.undo import CommandAfterFitting
from src.stages.ml.functions.hyperopt import optuna_opt
from src.stages.ml.functions.metrics import lda_coef_equation, insert_table_to_text_edit, \
    create_fit_data


class ML(QObject):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.latest_stat_result = ObservableDict()
        self.studies = ObservableDict()
        self.reset()
        self._set_ui()

    def reset(self):
        mw = get_parent(self.parent, "MainWindow")
        defaults = get_config('defaults')
        mw.ui.dataset_type_cb.setCurrentText(defaults["dataset_type_cb"])
        mw.ui.classes_lineEdit.setText("")
        mw.ui.test_data_ratio_spinBox.setValue(defaults["test_data_ratio_spinBox"])
        mw.ui.random_state_sb.setValue(defaults["random_state_sb"])
        mw.ui.baseline_cb.setChecked(False)
        mw.ui.current_group_shap_comboBox.clear()
        mw.ui.current_feature_comboBox.clear()
        mw.ui.coloring_feature_comboBox.clear()
        mw.ui.current_instance_combo_box.clear()
        mw.ui.feature_display_max_checkBox.setChecked(False)
        mw.ui.feature_display_max_spinBox.setValue(defaults["feature_display_max_spinBox"])
        mw.ui.test_data_ratio_spinBox.mouseDoubleClickEvent = lambda event: (
            self._reset_field(event, 'test_data_ratio_spinBox'))
        mw.ui.random_state_sb.mouseDoubleClickEvent = lambda event: (
            self._reset_field(event, 'random_state_sb'))

    def _reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Change field value to default on double click by MiddleButton.

        Parameters
        -------
        event: QMouseEvent

        field_id: str
            name of field
        """
        if event.buttons() != Qt.MouseButton.MiddleButton:
            return
        mw = get_parent(self.parent, "MainWindow")
        value = get_config('defaults')[field_id]
        match field_id:
            case 'test_data_ratio_spinBox':
                mw.ui.test_data_ratio_spinBox.setValue(value)
            case 'random_state_sb':
                mw.ui.random_state_sb.setValue(value)
            case _:
                return

    def _set_ui(self):
        mw = get_parent(self.parent, "MainWindow")
        self._init_dataset_type_cb()
        self._init_current_classificator_combo_box()
        mw.ui.current_group_shap_comboBox.currentTextChanged.connect(
            self.current_group_shap_changed)
        mw.ui.current_feature_comboBox.currentTextChanged.connect(mw.update_shap_scatters)
        mw.ui.coloring_feature_comboBox.currentTextChanged.connect(mw.update_shap_scatters)
        mw.ui.current_tree_spinBox.valueChanged.connect(self.current_tree_sb_changed)

    def read(self) -> dict:
        """
        Read attributes data.

        Returns
        -------
        dt: dict
            all class attributes data
        """
        mw = get_parent(self.parent, "MainWindow")
        dt = {"dataset_type_cb": mw.ui.dataset_type_cb.currentText(),
              'classes_lineEdit': mw.ui.classes_lineEdit.text(),
              'test_data_ratio_spinBox': mw.ui.test_data_ratio_spinBox.value(),
              'random_state_sb': mw.ui.random_state_sb.value(),
              'baseline_cb': mw.ui.baseline_cb.isChecked(),
              'feature_display_max_checkBox': mw.ui.feature_display_max_checkBox.isChecked(),
              'feature_display_max_spinBox': mw.ui.feature_display_max_spinBox.value(),
              }
        return dt

    def load(self, db: dict):
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.dataset_type_cb.setCurrentText(db["dataset_type_cb"])
        mw.ui.classes_lineEdit.setText(db["classes_lineEdit"])
        mw.ui.test_data_ratio_spinBox.setValue(db["test_data_ratio_spinBox"])
        mw.ui.random_state_sb.setValue(db["random_state_sb"])
        mw.ui.baseline_cb.setChecked(db["baseline_cb"])
        mw.ui.feature_display_max_checkBox.setChecked(db["feature_display_max_checkBox"])
        mw.ui.feature_display_max_spinBox.setValue(db["feature_display_max_spinBox"])

    def _init_dataset_type_cb(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.dataset_type_cb.addItems(["Smoothed", "Baseline corrected", "Decomposed"])
        mw.ui.dataset_type_cb.currentTextChanged.connect(self.dataset_type_cb_current_text_changed)

    def _init_current_classificator_combo_box(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.current_classificator_comboBox.addItems(list(classificator_funcs().keys())[: -2])
        mw.ui.current_classificator_comboBox.currentTextChanged.connect(
            self.update_stat_report_text
        )

    @asyncSlot()
    async def current_group_shap_changed(self, _: str = "") -> None:
        mw = get_parent(self.parent, "MainWindow")
        loop = get_event_loop()
        await loop.run_in_executor(None, mw.update_shap_plots)
        await loop.run_in_executor(None, mw.update_shap_plots_by_instance)
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        mw.update_force_single_plots(cl_type)
        mw.update_force_full_plots(cl_type)

    def current_tree_sb_changed(self, idx: int) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if (
                mw.ui.current_classificator_comboBox.currentText() == "Random Forest"
                and "Random Forest" in self.latest_stat_result
        ):
            model_results = self.latest_stat_result["Random Forest"]
            model = model_results["model"]
            self.update_plot_tree(
                model.best_estimator_.estimators_[idx],
                model_results["feature_names"],
                model_results["target_names"],
            )
        elif (
                mw.ui.current_classificator_comboBox.currentText() == "XGBoost"
                and "XGBoost" in self.latest_stat_result
        ):
            model_results = self.latest_stat_result["XGBoost"]
            model = model_results["model"]
            self.update_xgboost_tree_plot(
                model.best_estimator_, idx
            )

    def dataset_type_cb_current_text_changed(self, ct: str) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if ct == "Smoothed":
            model = mw.ui.smoothed_dataset_table_view.model()
        elif ct == "Baseline corrected":
            model = mw.ui.baselined_dataset_table_view.model()
        elif ct == "Decomposed":
            model = mw.ui.deconvoluted_dataset_table_view.model()
        else:
            return
        if model.rowCount() == 0 or mw.predict_logic.is_production_project:
            mw.ui.dataset_features_n.setText("")
            return
        q_res = model.dataframe()
        features_names = list(q_res.columns[2:])
        n_features = (
            mw.ui.ignore_dataset_table_view.model().n_features
            if ct == "Decomposed"
            else len(features_names)
        )
        mw.ui.dataset_features_n.setText(f"{n_features} features")
        mw.ui.current_feature_comboBox.clear()
        mw.ui.current_dep_feature1_comboBox.clear()
        mw.ui.current_dep_feature2_comboBox.clear()
        mw.ui.coloring_feature_comboBox.clear()
        mw.ui.coloring_feature_comboBox.addItem("")
        mw.ui.current_feature_comboBox.addItems(features_names)
        mw.ui.current_dep_feature1_comboBox.addItems(features_names)
        mw.ui.current_dep_feature2_comboBox.addItems(features_names)
        mw.ui.coloring_feature_comboBox.addItems(features_names)
        mw.ui.current_dep_feature2_comboBox.setCurrentText(features_names[1])
        try:
            mw.ui.current_group_shap_comboBox.currentTextChanged.disconnect(
                self.current_group_shap_changed)
        except:
            error(
                "failed to disconnect currentTextChanged self.current_group_shap_comboBox)"
            )
        mw.ui.current_group_shap_comboBox.clear()
        uniq_classes = np.unique(q_res["Class"].values)
        classes = []
        groups = self.parent.group_table.groups_list()
        for i in uniq_classes:
            if i in groups:
                classes.append(i)
        target_names = self.parent.group_table.target_names(classes)
        mw.ui.current_group_shap_comboBox.addItems(target_names)
        try:
            mw.ui.current_group_shap_comboBox.currentTextChanged.connect(
                self.current_group_shap_changed)
        except:
            error("failed to connect currentTextChanged self.current_group_shap_comboBox)")
        try:
            mw.ui.current_instance_combo_box.currentTextChanged.disconnect(
                mw.current_instance_changed)
        except:
            error("failed to disconnect currentTextChanged self.current_instance_combo_box)")
        mw.ui.current_instance_combo_box.addItem("")
        mw.ui.current_instance_combo_box.addItems(q_res["Filename"])
        try:
            mw.ui.current_instance_combo_box.currentTextChanged.connect(mw.current_instance_changed)
        except:
            error("failed to connect currentTextChanged self.current_instance_changed)")

    def update_stat_report_text(self, cl_type: str):
        mw = get_parent(self.parent, "MainWindow")
        if mw.parent.ui.current_classificator_comboBox.currentText() == 'PCA':
            mw.parent.ui.stat_report_text_edit.setText('')
            return
        classificator_type = cl_type
        if classificator_type not in self.latest_stat_result or 'metrics_result' \
                not in self.latest_stat_result[classificator_type]:
            mw.ui.stat_report_text_edit.setText('')
            return
        if classificator_type in self.top_features:
            top = self.top_features[classificator_type]
        else:
            top = None
        model_results = self.latest_stat_result[classificator_type]
        misclassified_filenames = np.unique(
            model_results['filenames'].values[model_results['misclassified']])
        self.update_report_text(model_results['metrics_result'], model_results['cv_scores'], top,
                                model_results['model'], classificator_type, misclassified_filenames)

    def update_report_text(self, metrics_result: dict, cv_scores=None, top=None, model=None,
                           classificator_type=None, misclassified_filenames=None) -> None:
        """
        Set report text

        Parameters
        ----------
        metrics_result: dict
        cv_scores:
        top:
        model:
        classificator_type
        misclassified_filenames: ndarray
            filenames which was classified wrong and will be shown in --Misclassified-- section

        """
        mw = get_parent(self.parent, "MainWindow")
        text = ''
        if model is not None and not isinstance(model, HalvingGridSearchCV) and not isinstance(
                model, GridSearchCV):
            model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
            evr = model.explained_variance_ratio_
            text += '\n' + f'Explained variance ratio: {evr}' + '\n'
        if classificator_type == 'Random Forest':
            text += f'N Trees: {len(model.best_estimator_.estimators_)}' + '\n'

        if misclassified_filenames is not None and misclassified_filenames.any():
            text += '\n' + '--Misclassified--'
            text += str(list(misclassified_filenames)) + '\n'
        if classificator_type == 'LDA' and model:
            model = model.best_estimator_ if isinstance(model, (GridSearchCV, HalvingGridSearchCV)) \
                else model
            text += lda_coef_equation(model) + '\n'
        mw.ui.stat_report_text_edit.setText(text)


    def update_plot_tree(self, clf, feature_names, class_names) -> None:
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.tree_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        plot_tree(clf, feature_names=feature_names, fontsize=int(environ['plot_font_size']),
                  class_names=class_names, ax=ax)
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_xgboost_tree_plot(self, model, idx: int = 0) -> None:
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.tree_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        xgboost.plot_tree(model, ax=ax, num_trees=idx,
                          yes_color=environ['secondaryDarkColor'],
                          no_color=environ['primaryColor'])
        plot_widget.canvas.draw()
        plot_widget.canvas.figure.tight_layout()

    @asyncSlot()
    async def fit_classificator(self, cl_type=None):
        """
        Проверяем что dataset заполнен.
        Переходим в процедуру создания классификатора
        """
        mw = get_parent(self.parent, "MainWindow")
        current_dataset = mw.ui.dataset_type_cb.currentText()
        if (current_dataset == "Smoothed"
                and mw.ui.smoothed_dataset_table_view.model().rowCount() == 0
                or current_dataset == "Baseline corrected"
                and mw.ui.baselined_dataset_table_view.model().rowCount() == 0
                or current_dataset == "Decomposed"
                and mw.ui.deconvoluted_dataset_table_view.model().rowCount() == 0
        ):
            MessageBox("Classificator Fitting failed.", "No data to train the classifier",
                       mw, {"Ok"})
            return
        if not cl_type:
            cl_type = mw.ui.current_classificator_comboBox.currentText()
        await self.do_fit_classificator(cl_type)

    @asyncSlot()
    async def do_fit_classificator(self, cl_type: str) -> None:
        mw = get_parent(self.parent, "MainWindow")

        x, y, feature_names, target_names, filenames = self.dataset_for_ml()
        rnd_state = mw.ui.random_state_sb.value()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=mw.ui.test_data_ratio_spinBox.value() / 100.,
            stratify=y, random_state=rnd_state)
        best_params = {}
        if not mw.ui.baseline_cb.isChecked():
            cfg = get_config("texty")["optuna"]
            mw.progress.open_progress(cfg, 0)
            kwargs = {'n_trials': psutil.cpu_count(logical=False) * 4, 'x_train': x_train,
                      'y_train': y_train, 'cl_type': cl_type, 'obj': objectives()[cl_type]}
            study = await mw.progress.run_in_executor('optuna', optuna_opt, None, **kwargs)
            if mw.progress.close_progress(cfg):
                return
            if not study:
                mw.ui.statusBar.showMessage(cfg["no_result_msg"])
                return
            self.studies[cl_type] = study[0]
            best_params = study[0].best_params

        cfg = get_config("texty")["fit_model"]
        mw.progress.open_progress(cfg, 0)
        kwargs = {'best_params': best_params, 'x_train': x_train, 'y_train': y_train}
        result = await mw.progress.run_in_executor('fit_model', classificator_funcs()[cl_type],
                                                   None, **kwargs)
        if mw.progress.close_progress(cfg):
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
            return
        model = result[0]
        fit_data = create_fit_data(model, x_train, y_train, x_test, y_test)
        headers, rows = fit_data['c_r_parsed']
        insert_table_to_text_edit(mw.ui.stat_report_text_edit.textCursor(), headers, rows)
        headers, rows = fit_data['metrics_parsed']
        insert_table_to_text_edit(mw.ui.stat_report_text_edit.textCursor(), headers, rows)
        # context = get_parent(self.parent, "Context")
        # command = CommandAfterFitting(model, context, text=f"Fit model {cl_type}",
        #                               **{'stage': self, 'cl_type': cl_type})
        # context.undo_stack.push(command)

    def dataset_for_ml(self) \
            -> tuple[pd.DataFrame, list[int], list[str], np.ndarray, pd.DataFrame] | None:
        """
        Выбор данных для обучения из датасета
        @return:
        X: DataFrame. Columns - fealures, rows - samples
        Y: list[int]. True labels
        feature_names: feature names (lol)
        target_names: classes names
        filenames
        """
        mw = get_parent(self.parent, "MainWindow")
        selected_dataset = mw.ui.dataset_type_cb.currentText()
        match selected_dataset:
            case 'Smoothed':
                model = mw.ui.smoothed_dataset_table_view.model()
            case 'Baseline corrected':
                model = mw.ui.baselined_dataset_table_view.model()
            case 'Decomposed':
                model = mw.ui.deconvoluted_dataset_table_view.model()
            case _:
                return
        df = model.dataframe()
        if mw.ui.classes_lineEdit.text() != '':
            classes = [int(i) for i in list(mw.ui.classes_lineEdit.text().strip().split(','))]
            if len(classes) > 1:
                df = model.query_result_with_list('Class == @input_list', classes)
        if selected_dataset == 'Decomposed':
            ignored_features = mw.ui.ignore_dataset_table_view.model().ignored_features
            df = df.drop(ignored_features, axis=1)
        uniq_classes = df['Class'].unique()
        groups = mw.context.group_table.table_widget.model().groups_list()
        classes = [i for i in uniq_classes if i in groups]
        if mw.predict_logic.is_production_project:
            target_names = None
        else:
            target_names = mw.context.group_table.table_widget.model().dataframe().loc[classes][
                'Group name'].values
        filenames = df.pop('Filename')
        y = df.pop('Class')

        return df, y, list(df.axes[1]), target_names, filenames
