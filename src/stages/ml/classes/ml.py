"""
Module containing the ML class for machine learning model management and UI integration.

This module provides functionality for managing machine learning model settings, updating
UI elements, and handling various interactions related to model configuration and
visualization in the MainWindow.
"""

# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin

from asyncio import get_event_loop
from logging import error
from os import environ

import numpy as np
import pandas as pd
import psutil
from PyQt5.QtWidgets import QLineEdit, QMenu
from asyncqtpy import asyncSlot
from qtpy.QtCore import QObject, Qt
from qtpy.QtGui import QMouseEvent, QColor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from qfluentwidgets import MessageBox
from src import get_parent, get_config, ObservableDict
from src.data.default_values import classificator_funcs, objectives
from src.data.plotting import initial_stat_plot, initial_shap_plot
from src.pandas_tables import PandasModelPCA
from src.stages import fit_pca
from src.stages.ml.classes.plots import Plots
from src.stages.ml.classes.shap_plots import ShapPlots
from src.stages.ml.classes.undo import CommandAfterFittingStat
from src.stages.ml.functions.hyperopt import optuna_opt, get_study
from src.stages.ml.functions.metrics import insert_table_to_text_edit, create_fit_data
from src.stages.ml.functions.shap_processing import shap_explain


class ML(QObject):
    """
    A class to manage machine learning model settings and interact with the UI.

    This class is responsible for initializing UI components, resetting settings to default values,
    updating plots, handling user interactions, and loading/saving model configurations. It
    integrates with the main window and context to ensure proper functionality and display.

    Attributes
    ----------
    parent : QObject
        The parent object, typically the main window or application.
    data : ObservableDict
        A dictionary to store observable data related to the model.
    le : LabelEncoder
        A label encoder instance for encoding categorical labels.
    shap_plotting : ShapPlots
        An instance for handling SHAP (SHapley Additive exPlanations) plot updates.
    plots : Plots
        An instance for managing various plots related to the machine learning model.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the ML class.

        Parameters
        ----------
        parent : QObject
            The parent object, typically the main window or application.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.data = ObservableDict()
        self.le = LabelEncoder()
        self.shap_plotting = ShapPlots(self)
        self.plots = Plots(self)
        self.reset()
        self._set_ui()

    def reset(self):
        """
        Reset UI elements and class attributes to default values.

        This method sets default values for various UI components and resets internal state
        related to data and label encoders.
        """
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
        mw.ui.feature_display_max_checkBox.setChecked(True)
        mw.ui.feature_display_max_spinBox.setValue(defaults["feature_display_max_spinBox"])
        mw.ui.test_data_ratio_spinBox.mouseDoubleClickEvent = lambda event: (
            self._reset_field(event, 'test_data_ratio_spinBox'))
        mw.ui.random_state_sb.mouseDoubleClickEvent = lambda event: (
            self._reset_field(event, 'random_state_sb'))
        mw.ui.current_dep_feature1_comboBox.clear()
        mw.ui.current_dep_feature2_comboBox.clear()
        self.reset_pca_features_table()
        self.data = ObservableDict()
        self.le = LabelEncoder()

    def _reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Change field value to default on double click by MiddleButton.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event triggering the reset.
        field_id : str
            The identifier for the field to reset.
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
        """
        Initialize and connect UI components related to machine learning settings.

        This method sets up connections between UI elements and their corresponding
        slot methods, including initializing dataset type combo box and classifier combo box.
        """
        mw = get_parent(self.parent, "MainWindow")
        self._init_dataset_type_cb()
        self._init_current_classificator_combo_box()
        mw.ui.current_group_shap_comboBox.currentTextChanged.connect(
            self._current_group_shap_changed)
        mw.ui.current_feature_comboBox.currentTextChanged.connect(
            self.shap_plotting.do_update_shap_scatters)
        mw.ui.coloring_feature_comboBox.currentTextChanged.connect(
            self.shap_plotting.do_update_shap_scatters)
        mw.ui.current_tree_spinBox.valueChanged.connect(self._current_tree_sb_changed)
        mw.ui.update_partial_dep_pushButton.clicked.connect(self._partial_btn_clicked)

    def read(self) -> dict:
        """
        Read attributes data from the UI components.

        Returns
        -------
        dict
            A dictionary containing the current state of relevant UI components and internal data.
        """
        mw = get_parent(self.parent, "MainWindow")
        dt = {"dataset_type_cb": mw.ui.dataset_type_cb.currentText(),
              'classes_lineEdit': mw.ui.classes_lineEdit.text(),
              'test_data_ratio_spinBox': mw.ui.test_data_ratio_spinBox.value(),
              'random_state_sb': mw.ui.random_state_sb.value(),
              'baseline_cb': mw.ui.baseline_cb.isChecked(),
              'feature_display_max_checkBox': mw.ui.feature_display_max_checkBox.isChecked(),
              'feature_display_max_spinBox': mw.ui.feature_display_max_spinBox.value(),
              'data': self.data,
              'le': self.le,
              }
        return dt

    def load(self, db: dict):
        """
        Load attributes data into the UI components.

        Parameters
        ----------
        db : dict
            A dictionary containing the state of relevant UI components and internal data to be
            loaded.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.dataset_type_cb.setCurrentText(db["dataset_type_cb"])
        mw.ui.classes_lineEdit.setText(db["classes_lineEdit"])
        mw.ui.test_data_ratio_spinBox.setValue(db["test_data_ratio_spinBox"])
        mw.ui.random_state_sb.setValue(db["random_state_sb"])
        mw.ui.baseline_cb.setChecked(db["baseline_cb"])
        mw.ui.feature_display_max_checkBox.setChecked(db["feature_display_max_checkBox"])
        mw.ui.feature_display_max_spinBox.setValue(db["feature_display_max_spinBox"])
        if 'data' in db:
            self.data = db["data"]
        if 'le' in db:
            self.le = db["le"]

    def clear_model(self, cl_type):
        """
        Clear the model of a specific type and remove associated files.

        Parameters
        ----------
        cl_type : str
            The type of the model to clear.
        """
        mw = get_parent(self.parent, "MainWindow")
        del self.data[cl_type]
        db_files = mw.project.get_db_files()
        for f in db_files:
            if cl_type in f:
                f.unlink()

    def _init_dataset_type_cb(self) -> None:
        """
        Initialize and configure the dataset type combo box.

        This method populates the dataset type combo box with predefined options and sets up
        the connection to handle changes in the selected dataset type.
        """
        mw = get_parent(self.parent, "MainWindow")
        defaults = get_config('defaults')
        mw.ui.dataset_type_cb.addItems(["Smoothed", "Baseline corrected", "Decomposed"])
        mw.ui.dataset_type_cb.setCurrentText(defaults["dataset_type_cb"])
        mw.ui.dataset_type_cb.currentTextChanged.connect(self.dataset_type_cb_current_text_changed)

    def _init_current_classificator_combo_box(self) -> None:
        """
        Initialize and configure the current classifier combo box.

        This method populates the current classifier combo box with available classifiers and sets
        up the connection to handle changes in the selected classifier.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.current_classificator_comboBox.addItems(list(classificator_funcs().keys()))
        mw.ui.current_classificator_comboBox.currentTextChanged.connect(
            self.update_stat_report_text
        )

    @asyncSlot()
    async def _current_group_shap_changed(self, _: str = "") -> None:
        """
        Handle changes in the current group for SHAP plots.

        Parameters
        ----------
        _ : str, optional
            The new text of the current group (not used in this implementation).
        """
        loop = get_event_loop()
        await loop.run_in_executor(None, self.shap_plotting.do_update_shap_plots)
        await loop.run_in_executor(None, self.shap_plotting.do_update_shap_plots_by_instance)

    def _current_tree_sb_changed(self, idx: int) -> None:
        """
        Handle changes in the current tree spin box value.

        Parameters
        ----------
        idx : int
            The new value of the current tree spin box.
        """
        mw = get_parent(self.parent, "MainWindow")
        if (
                mw.ui.current_classificator_comboBox.currentText() == "Random Forest"
                and "Random Forest" in self.data
        ):
            self.plots.update_plot_tree(mw.ui.current_classificator_comboBox.currentText()
                                  )
        elif (
                mw.ui.current_classificator_comboBox.currentText() == "XGBoost"
                and "XGBoost" in self.data
        ):
            self.plots.update_xgboost_tree_plot(idx)

    @asyncSlot()
    async def _partial_btn_clicked(self, _: str = "") -> None:
        """
        Build and display the partial dependence plot based on the current classifier.

        Parameters
        ----------
        _ : str, optional
            The new text of the button (not used in this implementation).
        """
        mw = get_parent(self.parent, "MainWindow")
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.data:
            return
        self.plots.build_partial_dependence_plot(cl_type)

    def reset_pca_features_table(self):
        """
        Reset the PCA features table view to an empty state.

        This method sets up the PCA features table view with an empty DataFrame and updates
        the model to reflect this empty state.
        """
        mw = get_parent(self.parent, "MainWindow")
        df = pd.DataFrame(columns=["feature", "PC-1", "PC-2"])
        model = PandasModelPCA(self, df)
        mw.ui.pca_features_table_view.setModel(model)

    def dataset_type_cb_current_text_changed(self, ct: str) -> None:
        """
        Update UI components based on the selected dataset type.

        Parameters
        ----------
        ct : str
            The selected dataset type from the combo box.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        if ct == "Smoothed":
            model = mw.ui.smoothed_dataset_table_view.model()
            q_res = model.dataframe
            features_names = list(q_res.columns[2:])
        elif ct == "Baseline corrected":
            model = mw.ui.baselined_dataset_table_view.model()
            q_res = model.dataframe
            features_names = list(q_res.columns[2:])
        elif ct == "Decomposed":
            model = mw.ui.deconvoluted_dataset_table_view.model()
            q_res = model.dataframe
            features_names = list(set(q_res.columns[2:])
                                  ^ set(mw.ui.ignore_dataset_table_view.model().ignored_features))
            features_names.sort()
        else:
            return
        if model.rowCount() == 0 or context.predict.is_production_project:
            mw.ui.dataset_features_n.setText("")
            return
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
        mw.ui.current_group_shap_comboBox.clear()
        uniq_classes = np.unique(q_res["Class"].values)
        classes = []
        groups = context.group_table.table_widget.model().groups_list
        for i in uniq_classes:
            if i in groups:
                classes.append(i)
        target_names = self.parent.group_table.target_names(classes)
        mw.ui.current_group_shap_comboBox.addItems(target_names)
        try:
            mw.ui.current_group_shap_comboBox.currentTextChanged.connect(
                self._current_group_shap_changed)
        except:
            error("failed to connect currentTextChanged self.current_group_shap_comboBox)")
        mw.ui.current_instance_combo_box.addItem("")
        mw.ui.current_instance_combo_box.addItems(q_res["Filename"])
        try:
            mw.ui.current_instance_combo_box.currentTextChanged.connect(
                self._current_instance_changed)
        except:
            error("failed to connect currentTextChanged self.current_instance_changed)")

    @asyncSlot()
    async def _current_instance_changed(self, _: str = "") -> None:
        """
        Handle changes in the current instance selection.

        Parameters
        ----------
        _ : str, optional
            The new text of the instance combo box (not used in this implementation).
        """
        loop = get_event_loop()
        await loop.run_in_executor(None, self.shap_plotting.do_update_shap_plots_by_instance)

    def update_stat_report_text(self, cl_type: str):
        """
        Update the statistical report text based on the classifier type.

        Parameters
        ----------
        cl_type : str
            The type of classifier (e.g., 'Random Forest', 'XGBoost', etc.).

        Returns
        -------
        None
            This method does not return a value. It updates the text in the statistics report widget
        """
        mw = get_parent(self.parent, "MainWindow")
        if cl_type not in self.data or cl_type == 'PCA':
            mw.ui.stat_report_text_edit.setText('')
            return
        self._update_report_text(cl_type)

    def _update_report_text(self, classificator_type=None) -> None:
        """
        Update the report text in the statistics report widget.

        Parameters
        ----------
        classificator_type : str, optional
            The type of classifier to generate the report for. If not provided, the method uses the
            current classifier type stored in `self.data`.

        Returns
        -------
        None
            This method does not return a value. It updates the text in the statistics report widget
        """
        mw = get_parent(self.parent, "MainWindow")
        fit_data = self.data[classificator_type]
        text = ''
        # if classificator_type == 'Random Forest':
        #     text += f'N Trees: {len(model.best_estimator_.estimators_)}' + '\n'
        misclassified_filenames = np.unique(
            fit_data['filenames'].values[fit_data['misclassified']])
        if misclassified_filenames is not None and misclassified_filenames.any():
            text += '\n' + '--Misclassified--' + '\n'
            for f in misclassified_filenames:
                text += f + '\n'
        study = get_study(classificator_type)
        if len(study.trials) > 0:
            text += '\n' + '--Best parameters--' + '\n'
            text += str(study.best_params) + '\n'
        mw.ui.stat_report_text_edit.setText(text)
        headers, rows = fit_data['c_r_parsed']
        insert_table_to_text_edit(mw.ui.stat_report_text_edit.textCursor(), headers, rows)
        headers, rows = fit_data['metrics_parsed']
        insert_table_to_text_edit(mw.ui.stat_report_text_edit.textCursor(), headers, rows)

    @asyncSlot()
    async def fit_classificator(self, cl_type=None):
        """
        Fit the selected classifier with the current dataset.

        Parameters
        ----------
        cl_type : str, optional
            The type of classifier to fit. If not provided, the method uses the current classifier
            type selected in the UI.

        Returns
        -------
        None
            This method does not return a value. It performs classifier fitting asynchronously.
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
        if cl_type == 'PCA':
            await self._do_fit_pca()
        else:
            await self._do_fit_classificator(cl_type)

    @asyncSlot()
    async def _do_fit_classificator(self, cl_type: str) -> None:
        """
        Fit the specified classifier type asynchronously.

        Parameters
        ----------
        cl_type : str
            The type of classifier to fit (e.g., 'XGBoost').

        Returns
        -------
        None
            This method does not return a value. It performs classifier fitting and updates the
            context.
        """
        mw = get_parent(self.parent, "MainWindow")
        x, y, feature_names, target_names, filenames = self.dataset_for_ml()
        if cl_type == 'XGBoost':
            y = self.le.fit_transform(y)
        rnd_state = mw.ui.random_state_sb.value()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=mw.ui.test_data_ratio_spinBox.value() / 100.,
            stratify=y, random_state=rnd_state)
        best_params = {}
        if not mw.ui.baseline_cb.isChecked():
            cfg = get_config("texty")["optuna"]
            mw.progress.open_progress(cfg, 0)
            kwargs = {'x_train': x_train,
                      'y_train': y_train, 'cl_type': cl_type, 'obj': objectives()[cl_type],
                      'rnd': rnd_state, 'n_jobs': psutil.cpu_count(logical=False)}
            study = await mw.progress.run_in_executor('optuna', optuna_opt, None, **kwargs)
            if mw.progress.close_progress(cfg):
                return
            if not study:
                mw.ui.statusBar.showMessage(cfg["no_result_msg"])
                return
            best_params = study[0].best_params
        cfg = get_config("texty")["fit_model"]
        mw.progress.open_progress(cfg, 0)
        kwargs = {'best_params': best_params, 'x_train': x_train, 'y_train': y_train,
                  'rnd': rnd_state}
        result = await mw.progress.run_in_executor('fit_model', classificator_funcs()[cl_type],
                                                   None, **kwargs)
        if mw.progress.close_progress(cfg):
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
            return
        fit_data = create_fit_data(result[0], x_train, y_train, x_test, y_test)
        fit_data['model'] = result[0]
        fit_data['x'] = x
        fit_data['y'] = y
        fit_data['target_names'] = target_names
        fit_data['feature_names'] = feature_names
        fit_data['filenames'] = filenames
        fit_data['y_train'] = y_train
        fit_data['y_test'] = y_test
        fit_data['x_test'] = x_test
        fit_data['x_train'] = x_train

        cfg = get_config("texty")["shap"]
        mw.progress.open_progress(cfg, 0)
        kwargs = {'model': result[0], 'x': x}
        result = await mw.progress.run_in_executor('shap', shap_explain, None, **kwargs)
        fit_data['shap_values'] = result[0][0]
        fit_data['shap_values_legacy'] = result[0][1]
        fit_data['expected_value'] = result[0][2]
        fit_data['dataset_type'] = mw.ui.dataset_type_cb.currentText()
        if mw.progress.close_progress(cfg):
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
            return

        context = get_parent(self.parent, "Context")
        command = CommandAfterFittingStat(fit_data, context, text=f"Fit model {cl_type}",
                                          **{'stage': self, 'cl_type': cl_type})
        context.undo_stack.push(command)

    @asyncSlot()
    async def _do_fit_pca(self):
        """
        Fit PCA model asynchronously.

        Returns
        -------
        None
            This method does not return a value. It performs PCA fitting and updates the PCA plots.
        """
        mw = get_parent(self.parent, "MainWindow")
        x, y, feature_names, target_names, _ = self.dataset_for_ml()
        rnd_state = mw.ui.random_state_sb.value()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=mw.ui.test_data_ratio_spinBox.value() / 100.,
            stratify=y, random_state=rnd_state)
        cfg = get_config("texty")["fit_model"]
        mw.progress.open_progress(cfg, 0)
        kwargs = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train,
                  'y_test': y_test}
        result = await mw.progress.run_in_executor('fit_model', fit_pca,
                                                   None, **kwargs)
        if mw.progress.close_progress(cfg):
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
            return
        result = result[0]
        result['loadings'] = pd.DataFrame(result['model'].components_.T, columns=['PC1', 'PC2'],
                                          index=feature_names)
        result['target_names'] = target_names
        self.data['PCA'] = result
        self.plots.update_pca_plots()

    def dataset_for_ml(self) \
            -> tuple[pd.DataFrame, list[int], list[str], np.ndarray, pd.DataFrame] | None:
        """
        Prepare dataset for machine learning.

        Returns
        -------
        tuple of (pd.DataFrame, list[int], list[str], np.ndarray, pd.DataFrame) or None
            - X: DataFrame. Features of the dataset.
            - Y: list[int]. True labels of the dataset.
            - feature_names: list[str]. Names of the features.
            - target_names: np.ndarray. Names of the target classes.
            - filenames: pd.DataFrame. Filenames of the samples.

        None if the dataset type is not recognized.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        selected_dataset = mw.ui.dataset_type_cb.currentText()
        match selected_dataset:
            case 'Smoothed':
                model = mw.ui.smoothed_dataset_table_view.model()
            case 'Baseline corrected':
                model = mw.ui.baselined_dataset_table_view.model()
            case 'Decomposed':
                model = mw.ui.deconvoluted_dataset_table_view.model()
            case _:
                return None
        df = model.dataframe
        if mw.ui.classes_lineEdit.text() != '':
            classes = [int(i) for i in list(mw.ui.classes_lineEdit.text().strip().split(','))]
            if len(classes) > 1:
                df = model.dataframe.query('Class == @input_list', classes)
        if selected_dataset == 'Decomposed':
            ignored_features = mw.ui.ignore_dataset_table_view.model().ignored_features
            df = df.drop(ignored_features, axis=1)
        uniq_classes = df['Class'].unique()
        groups = context.group_table.table_widget.model().groups_list
        classes = [i for i in uniq_classes if i in groups]
        if context.predict.is_production_project:
            target_names = None
        else:
            target_names = context.group_table.table_widget.model().dataframe.loc[classes][
                'Group name'].values
        filenames = df.pop('Filename')
        y = df.pop('Class')
        return df, y, df.axes[1], target_names, filenames

    def refresh_learning_curve(self):
        """
        Refresh the learning curve plot for the current classifier.

        Returns
        -------
        None
            This method does not return a value. It updates the learning curve plot widget.
        """
        mw = get_parent(self.parent, "MainWindow")
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.data:
            return
        self.plots.build_learning_curve(cl_type)

    @asyncSlot()
    async def redraw_stat_plots(self) -> None:
        """
        Redraw the statistical plots asynchronously.

        Returns
        -------
        None
            This method does not return a value. It updates all statistical plots asynchronously.
        """
        mw = get_parent(self.parent, "MainWindow")
        loop = get_event_loop()
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        self.update_stat_report_text(cl_type)
        await loop.run_in_executor(None, self.plots.update_plots, cl_type)

    @asyncSlot()
    async def initial_stat_plots_color(self) -> None:
        """
        Set the initial colors for all statistical plots and plot widgets.

        Returns
        -------
        None
            This method does not return a value. It updates the colors of all statistical plots and
            widgets.
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widgets = [
            mw.ui.decision_score_plot_widget, mw.ui.decision_boundary_plot_widget,
            mw.ui.violin_describe_plot_widget, mw.ui.bootstrap_plot_widget,
            mw.ui.boxplot_describe_plot_widget, mw.ui.dm_plot_widget,
            mw.ui.roc_plot_widget, mw.ui.pr_plot_widget, mw.ui.perm_imp_test_plot_widget,
            mw.ui.perm_imp_train_plot_widget, mw.ui.partial_depend_plot_widget,
            mw.ui.tree_plot_widget, mw.ui.features_plot_widget, mw.ui.calibration_plot_widget,
            mw.ui.det_curve_plot_widget, mw.ui.learning_plot_widget, mw.ui.pca_scores_plot_widget,
            mw.ui.pca_loadings_plot_widget, mw.ui.shap_beeswarm, mw.ui.shap_means,
            mw.ui.shap_heatmap, mw.ui.shap_scatter, mw.ui.shap_decision, mw.ui.shap_waterfall,
        ]
        for pl in plot_widgets:
            self.shap_plotting.set_canvas_colors(pl.canvas)
        for p in (mw.ui.cont_1, mw.ui.cont_2, mw.ui.cont_3, mw.ui.cont_4, mw.ui.cont_5,
                  mw.ui.cont_6, mw.ui.hist_plot, mw.ui.param_imp_plot, mw.ui.force_single,
                  mw.ui.force_full):
            p.page().setBackgroundColor(QColor(mw.attrs.plot_background_color))
        if mw.ui.current_group_shap_comboBox.currentText() == "":
            return
        loop = get_event_loop()
        await loop.run_in_executor(None, self.shap_plotting.do_update_shap_plots)
        await loop.run_in_executor(None, self.shap_plotting.do_update_shap_plots_by_instance)

    def initial_force_single_plot(self) -> None:
        """
        Initialize the single-force plot.

        Returns
        -------
        None
            This method does not return a value. It sets up the single-force plot widget and its
            context menu event.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.force_single.page().setHtml("")
        mw.ui.force_single.contextMenuEvent = self.force_single_context_menu_event

    def force_single_context_menu_event(self, a0) -> None:
        """
        Create a context menu for the single-force plot widget.

        Parameters
        ----------
        a0 : QContextMenuEvent
            The context menu event used to position the menu.

        Returns
        -------
        None
            This method does not return a value. It shows the context menu with options to save or
            refresh the plot.
        """
        mw = get_parent(self.parent, "MainWindow")
        line = QLineEdit(mw)
        menu = QMenu(line)
        menu.addAction(
            "Save .pdf", lambda: self.shap_plotting.web_view_print_pdf(mw.ui.force_single.page())
        )
        menu.addAction("Refresh", lambda: self.shap_plotting.reload_force(mw.ui.force_single))
        menu.move(a0.globalPos())
        menu.show()

    def initial_force_full_plot(self) -> None:
        """
        Initialize the full-force plot.

        Returns
        -------
        None
            This method does not return a value. It sets up the full-force plot widget and its
            context menu event.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.force_full.page().setHtml("")
        mw.ui.force_full.contextMenuEvent = self.force_full_context_menu_event

    def force_full_context_menu_event(self, a0) -> None:
        """
        Create a context menu for the full-force plot widget.

        Parameters
        ----------
        a0 : QContextMenuEvent
            The context menu event used to position the menu.

        Returns
        -------
        None
            This method does not return a value. It shows the context menu with options to save or
            refresh the plot.
        """
        mw = get_parent(self.parent, "MainWindow")
        line = QLineEdit(mw)
        menu = QMenu(line)
        menu.addAction(
            "Save .pdf", lambda: self.shap_plotting.web_view_print_pdf(mw.ui.force_full.page())
        )
        menu.addAction("Refresh", lambda: self.shap_plotting.reload_force(mw.ui.force_full, True))
        menu.move(a0.globalPos())
        menu.show()

    def initial_pca_plots(self) -> None:
        """
        Initialize PCA plots.

        Returns
        -------
        None
            This method does not return a value. It sets up the PCA plots with initial
            configurations.
        """
        mw = get_parent(self.parent, "MainWindow")
        for p in (mw.ui.pca_scores_plot_widget, mw.ui.pca_loadings_plot_widget):
            p.canvas.gca().cla()
            ax = p.canvas.gca()
            ax.set_xlabel('PC-1', fontsize=int(environ["axis_label_font_size"]))
            ax.set_ylabel('PC-2', fontsize=int(environ["axis_label_font_size"]))
            p.canvas.draw()

    def initial_all_stat_plots(self) -> None:
        """
        Initialize all statistical plots.

        Returns
        -------
        None
            This method does not return a value. It sets up all statistical plots with initial
            configurations and colors.
        """
        mw = get_parent(self.parent, "MainWindow")
        for pw in (mw.ui.decision_score_plot_widget, mw.ui.decision_boundary_plot_widget,
                   mw.ui.violin_describe_plot_widget, mw.ui.boxplot_describe_plot_widget,
                   mw.ui.dm_plot_widget, mw.ui.roc_plot_widget, mw.ui.pr_plot_widget,
                   mw.ui.perm_imp_train_plot_widget, mw.ui.perm_imp_test_plot_widget,
                   mw.ui.partial_depend_plot_widget, mw.ui.tree_plot_widget,
                   mw.ui.features_plot_widget, mw.ui.calibration_plot_widget,
                   mw.ui.det_curve_plot_widget, mw.ui.learning_plot_widget,
                   mw.ui.bootstrap_plot_widget):
            initial_stat_plot(pw)
        self.initial_pca_plots()
        self.initial_force_single_plot()
        self.initial_force_full_plot()

        shap_plots = [
            mw.ui.shap_beeswarm,
            mw.ui.shap_means,
            mw.ui.shap_heatmap,
            mw.ui.shap_scatter,
            mw.ui.shap_waterfall,
            mw.ui.shap_decision,
        ]
        for sp in shap_plots:
            initial_shap_plot(sp)
        self.initial_stat_plots_color()

    @asyncSlot()
    async def refresh_shap_push_button_clicked(self) -> None:
        """
        Refresh all SHAP plots for the currently selected classifier.

        Returns
        -------
        None
            This method does not return a value. It updates all SHAP plots asynchronously and
            reloads force plots.
        """
        mw = get_parent(self.parent, "MainWindow")
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        loop = get_event_loop()
        if (cl_type not in self.data or "target_names" not in self.data[cl_type]
                or "shap_values" not in self.data[cl_type]):
            msg = MessageBox(
                "SHAP plots refresh error.", "Selected classificator is not fitted.",
                mw, {"Ok"})
            msg.exec()
            return
        await loop.run_in_executor(None, self.shap_plotting.do_update_shap_plots)
        await loop.run_in_executor(None, self.shap_plotting.do_update_shap_plots_by_instance)
        self.shap_plotting.reload_force(mw.ui.force_single)
        self.shap_plotting.reload_force(mw.ui.force_full, True)
