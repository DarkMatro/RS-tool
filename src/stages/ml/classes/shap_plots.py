# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module contains functionality for interacting with and manipulating
SHAP force plots within a Qt-based GUI application.
"""
import re
from logging import error, info
from os import environ
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from qtpy.QtCore import QMarginsF, QObject
from qtpy.QtGui import QPageSize, QPageLayout
from qtpy.QtWidgets import QFileDialog
from matplotlib import pyplot as plt

from qfluentwidgets import MessageBox
from src import get_parent
from src.pandas_tables import PandasModel


class ShapPlots(QObject):
    """
    A class to handle the creation and updating of SHAP plots in the Qt GUI.

    This class integrates with the SHAP library to generate various plots
    for model interpretability, including beeswarm plots, bar plots, scatter
    plots, heatmaps, decision plots, and waterfall plots. It is designed to
    work within a Qt-based application and utilizes matplotlib for plotting.

    Parameters
    ----------
    parent : QObject
        The parent object, typically a Qt widget or window, to which this
        plotting backend is attached.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the ShapPlots instance.

        Parameters
        ----------
        parent : QObject
            The parent object, typically a Qt widget or window, to which this
            plotting backend is attached.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.parent = parent

    @property
    def data(self):
        """
        Retrieve the data from the parent object.

        Returns
        -------
        dict
            The data associated with the parent object.
        """
        return self.parent.data

    def create_fit_display(self, dt_type) -> pd.DataFrame:
        """
        Create a DataFrame to display based on the specified dataset type.

        Parameters
        ----------
        dt_type : str
            The type of dataset to display. Possible values are 'Smoothed',
            'Baseline corrected', and 'Deconvoluted'.

        Returns
        -------
        pd.DataFrame
            The DataFrame corresponding to the specified dataset type.
        """
        mw = get_parent(self.parent, "MainWindow")
        if dt_type == 'Smoothed':
            x_display = mw.ui.smoothed_dataset_table_view.model().dataframe
        elif dt_type == 'Baseline corrected':
            x_display = mw.ui.baselined_dataset_table_view.model().dataframe
        else:
            x_display = mw.ui.deconvoluted_dataset_table_view.model().dataframe
            ignored_features = mw.ui.ignore_dataset_table_view.model().ignored_features
            x_display = x_display.drop(ignored_features, axis=1)
        return x_display.iloc[:, 2:]

    def get_current_dataset_type_cb(self) -> PandasModel | None:
        """
        Get the current dataset type model based on the selection in the
        dataset type combo box.

        Returns
        -------
        PandasModel | None
            The PandasModel corresponding to the selected dataset type, or None
            if the type is not recognized.
        """
        mw = get_parent(self.parent, "MainWindow")
        ct = mw.ui.dataset_type_cb.currentText()
        if ct == 'Smoothed':
            return mw.ui.smoothed_dataset_table_view.model()
        if ct == 'Baseline corrected':
            return mw.ui.baselined_dataset_table_view.model()
        return mw.ui.deconvoluted_dataset_table_view.model()

    def do_update_shap_plots(self):
        """
        Update all SHAP plots based on the current classifier and group
        selections.

        This method updates the beeswarm, means, scatter, and heatmap plots
        for the currently selected classifier and group.
        """
        mw = get_parent(self.parent, "MainWindow")
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.data:
            return
        target_names = self.data[cl_type]['target_names']
        num = np.where(target_names == mw.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_beeswarm_plot(i, cl_type)
        self.update_shap_means_plot(i, cl_type)
        self.update_shap_scatter_plot(i, cl_type)
        self.update_shap_heatmap_plot(i, cl_type)

    def do_update_shap_plots_by_instance(self):
        """
        Update SHAP plots related to individual instances, including decision,
        waterfall, and force plots.

        This method updates the decision, waterfall, and force plots for the
        currently selected instance.
        """
        mw = get_parent(self.parent, "MainWindow")
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.data:
            return
        target_names = self.data[cl_type]['target_names']
        num = np.where(target_names == mw.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_decision_plot(i, cl_type)
        self.update_shap_waterfall_plot(i, cl_type)
        self.update_shap_force(i, cl_type)
        self.update_shap_force(i, cl_type, True)

    def update_shap_beeswarm_plot(self, class_i: int = 0, cl_type: str = 'LDA') -> None:
        """
        Update the SHAP beeswarm plot for the given class and classifier type.

        Parameters
        ----------
        class_i : int, optional
            The index of the class for which to update the plot. Default is 0.
        cl_type : str, optional
            The type of classifier. Default is 'LDA'.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.shap_beeswarm

        color = None if mw.ui.sun_Btn.isChecked() else plt.get_cmap("cool")
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
        if cl_type not in self.data:
            return
        if 'shap_values' not in self.data[cl_type]:
            return
        model = self.data[cl_type]['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        shap_values = self.data[cl_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        ax = shap.plots.beeswarm(shap_values, show=False, color=color, max_display=20)
        plot_widget.canvas.sca(ax)
        self.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_means_plot(self, class_i: int = 0, cl_type: str = 'LDA') -> None:
        """
        Update the SHAP means plot for the given class and classifier type.

        Parameters
        ----------
        class_i : int, optional
            The index of the class for which to update the plot. Default is 0.
        cl_type : str, optional
            The type of classifier. Default is 'LDA'.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.shap_means
        fig = plot_widget.canvas.figure
        ax = fig.gca()
        try:
            fig.clear()
            ax.clear()
        except ValueError:
            pass
        if cl_type not in self.data:
            return
        if 'shap_values' not in self.data[cl_type]:
            return
        shap_values = self.data[cl_type]['shap_values']
        model = self.data[cl_type]['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if isinstance(shap_values, list) and not binary:
            shap_values = shap_values[class_i]
        elif not isinstance(shap_values, list) and len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.bar(shap_values, show=False, max_display=20, ax=fig.gca())
        self.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def do_update_shap_scatters(self):
        """
        Update the SHAP scatter plots based on the current classifier and group
        selections.

        This method updates scatter plots for the currently selected classifier
        and group.
        """
        mw = get_parent(self.parent, "MainWindow")
        clf = mw.ui.current_classificator_comboBox.currentText()
        if not (clf in self.data and "shap_values" in self.data[clf]):
            return
        target_names = self.data[clf]["target_names"]
        if mw.ui.current_group_shap_comboBox.currentText() not in target_names:
            return
        i = int(np.where(target_names == mw.ui.current_group_shap_comboBox.currentText())[0][0])
        self.update_shap_scatter_plot(i, clf)

    def update_shap_scatter_plot(self, class_i: int = 0, cl_type: str = 'LDA') -> None:
        """
        Update the SHAP scatter plot for the given class and classifier type.

        Parameters
        ----------
        class_i : int
            The index of the class for which to update the plot.
        cl_type : str
            The type of classifier.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.shap_scatter
        fig = plot_widget.canvas.figure
        ax = plot_widget.canvas.gca()
        try:
            fig.clear()
            ax.clear()
        except ValueError:
            pass
        current_feature = mw.ui.current_feature_comboBox.currentText()
        cmap = None if mw.ui.sun_Btn.isChecked() else plt.get_cmap("cool")
        if 'shap_values' not in self.data[cl_type]:
            return
        shap_values = self.data[cl_type]['shap_values']
        model = self.data[cl_type]['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        if current_feature not in shap_values.feature_names:
            return
        ct = mw.ui.coloring_feature_comboBox.currentText()
        color = shap_values if ct == '' else shap_values[:, ct]
        shap.plots.scatter(shap_values[:, current_feature], color=color, show=False, cmap=cmap,
                           axis_color=mw.attrs.plot_text_color.name())
        ax = plt.gca()
        plot_widget.canvas.sca(ax)
        self.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_heatmap_plot(self, class_i: int = 0,
                                 cl_type: str = 'LDA') -> None:
        """
        Update the SHAP heatmap plot for the given class and classifier type.

        Parameters
        ----------
        class_i : int
            The index of the class for which to update the plot.
        cl_type : str
            The type of classifier.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.shap_heatmap
        model = self.data[cl_type]['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        fig = plot_widget.canvas.figure
        ax = plot_widget.canvas.gca()
        try:
            fig.clear()
            ax.clear()
        except ValueError:
            pass
        if cl_type not in self.data or 'shap_values' not in self.data[cl_type]:
            return
        shap_values = self.data[cl_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        ax = shap.plots.heatmap(shap_values, show=False, max_display=20)
        plot_widget.canvas.sca(ax)
        self.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_decision_plot(self, class_i: int = 0, cl_type: str = 'LDA') -> None:
        """
        Update the SHAP decision plot for the given class and classifier type.

        Parameters
        ----------
        class_i : int
            The index of the class for which to update the plot.
        cl_type : str
            The type of classifier.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.shap_decision
        table_model = self.get_current_dataset_type_cb()
        current_instance = mw.ui.current_instance_combo_box.currentText()
        try:
            sample_id = None if not current_instance \
                else table_model.idx_by_column_value('Filename', current_instance)
        except KeyError:
            sample_id = None
        class_i = 0 if len(self.data[cl_type]['model'].classes_) == 2 else class_i
        expected_value = self.data[cl_type]['expected_value']
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[class_i]
        s_v = self.data[cl_type]['shap_values_legacy']
        x_d = self.create_fit_display(self.data[cl_type]['dataset_type'])
        misclassified = self.data[cl_type]['misclassified']
        try:
            plot_widget.canvas.figure.clear()
            plot_widget.canvas.gca().clear()
        except ValueError:
            pass
        if x_d.empty:
            return
        if len(s_v.shape) == 3:
            s_v = s_v[..., class_i]
        if sample_id is not None:
            x_d = x_d.loc[sample_id]
            misclassified = misclassified[sample_id]
            title = current_instance
            s_v = s_v[sample_id, ...]
        else:
            title = 'all'
        if (sample_id is not None and s_v.shape[0] != len(x_d.values)) \
                or (sample_id is None and s_v[0].shape[0]
                    != len(x_d.loc[table_model.first_index].values)):
            err = ("Decision plot failed to update. Number of shap_values features != number of "
                   "X features. The training data table may have been modified. "
                   f"Needs to be recalculated {cl_type}")
            error(err)
            return
        feature_display_range_max = -mw.ui.feature_display_max_spinBox.value() \
            if mw.ui.feature_display_max_checkBox.isChecked() else None
        try:
            shap.plots.decision(expected_value, s_v, x_d,
                                title=title, ignore_warnings=True,
                                feature_display_range=slice(None, feature_display_range_max, -1),
                                highlight=misclassified, show=False)
            plot_widget.canvas.sca(plt.gca())
            self.set_canvas_colors(plot_widget.canvas)
        except ValueError:
            error(cl_type + f' decision plot trouble. {ValueError}')
            return
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_waterfall_plot(self, class_i: int = 0, cl_type: str = 'LDA') -> None:
        """
        Update the SHAP waterfall plot for the given class and classifier type.

        Parameters
        ----------
        class_i : int
            The index of the class for which to update the plot.
        cl_type : str
            The type of classifier.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.shap_waterfall
        table_model = self.get_current_dataset_type_cb()
        if table_model.rowCount() == 0:
            return
        n_features = len(list(table_model.dataframe.columns[2:]))
        current_instance = mw.ui.current_instance_combo_box.currentText()
        try:
            sample_id = 0 if not current_instance \
                else table_model.idx_by_column_value('Filename', current_instance)
        except KeyError:
            sample_id = 0
        class_i = 0 if len(self.data[cl_type]['model'].classes_) == 2 else class_i
        shap_values = self.data[cl_type]['shap_values']
        try:
            plot_widget.canvas.figure.clear()
            plot_widget.canvas.gca().clear()
        except ValueError:
            pass
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap_values = shap_values[sample_id, ...]
        max_display = mw.ui.feature_display_max_spinBox.value() \
            if mw.ui.feature_display_max_checkBox.isChecked() else n_features
        try:
            shap.plots.waterfall(shap_values, max_display, show=False)
            plot_widget.canvas.sca(plt.gca())
            self.set_canvas_colors(plot_widget.canvas)
        except ValueError as err:
            error(cl_type + f' waterfall plot trouble. {ValueError}', err)
            return
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_force(self, class_i: int = 0, cl_type: str = 'LDA', full: bool = False) \
            -> None:
        """
        Update the SHAP force plot for the given class and classifier type.

        Parameters
        ----------
        class_i : int
            The index of the class for which to update the plot.
        cl_type : str
            The type of classifier.
        full : bool, optional
            Whether to generate a force plot. Default is False.

        Returns
        -------
        None
        """
        info(f"update_shap_force for {cl_type} class-i {class_i}, full = {full}")
        mw = get_parent(self.parent, "MainWindow")
        table_model = self.get_current_dataset_type_cb()
        current_instance = mw.ui.current_instance_combo_box.currentText()
        sample_id = table_model.first_index if not current_instance \
            else table_model.idx_by_column_value('Filename', current_instance)
        class_i = 0 if len(self.data[cl_type]['model'].classes_) == 2 else class_i
        expected_value = self.data[cl_type]['expected_value']
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[class_i]
        shap_values = self.data[cl_type]['shap_values_legacy']
        x_d = self.create_fit_display(self.data[cl_type]['dataset_type'])
        if x_d.empty:
            error('update_shap_force x_d.empty return')
            return
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        if not full:
            x_d = x_d.loc[sample_id]
            shap_values = shap_values[sample_id, ...]
        if (not full and shap_values.shape[0] != len(x_d.values)) \
                or (full and shap_values[0].shape[0] != len(x_d.loc[
                                                                table_model.first_index].values)):
            err = ("Decision plot failed to update. Number of shap_values features != number of "
                   "X features. The training data table may have been modified. "
                   f"Needs to be recalculated {cl_type}")
            error(err)
            return
        force_plot = shap.force_plot(expected_value, shap_values, x_d,
                                     feature_names=self.data[cl_type]['feature_names'],
                                     plot_cmap="CyPU")
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        if full:
            self.data[cl_type]['shap_html_full'] = shap_html
        else:
            self.data[cl_type]['shap_html'] = shap_html

    def reload_force(self, plot_widget, full: bool = False):
        """
        Reloads the SHAP force plot in the specified web view widget.

        This method retrieves the SHAP force plot HTML content from the current classifier and
        updates the provided `plot_widget` with this content. The appearance of the plot may be
        adjusted based on the application's theme settings.

        Parameters
        ----------
        plot_widget : qtpy.QtWebEngineWidgets.QWebEngineView
            The Qt web view widget where the SHAP force plot HTML content will be loaded.
        full : bool, optional
            If True, reloads the full SHAP force plot HTML. If False, reloads a partial SHAP force
            plot.
            Default is False.

        Returns
        -------
        None

        Notes
        -----
        If the classifier is not fitted or if the SHAP HTML content is missing, a message box is
        displayed to inform the user. The `sun_Btn` checkbox determines if the plot colors are
        adjusted for dark mode.
        """
        info(f"reload_force, full = {full}")
        mw = get_parent(self.parent, "MainWindow")
        cl_type = mw.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.data:
            msg = MessageBox(
                "SHAP Force plot refresh error.", "Selected classificator was not fitted.",
                mw, {"Ok"})
            msg.exec()
            return
        shap_html = "shap_html_full" if full else "shap_html"
        if shap_html not in self.data[cl_type]:
            msg = MessageBox(
                "Ups",
                "Selected classificator was fitted without Shapley calculation.",
                mw, {"Ok"})
            msg.setInformativeText('Push Fit/Refresh SHAP')
            msg.exec()
            return
        shap_html = self.data[cl_type][shap_html]
        if not mw.ui.sun_Btn.isChecked():
            shap_html = re.sub(r"#000", "#FFFFFe", shap_html)
            shap_html = re.sub(r"#fff", "#000001", shap_html)
            shap_html = re.sub(r"#ccc", "#FFFFFe", shap_html)
            shap_html = re.sub(
                r"font-family: arial;", "font-family: arial; color: white;", shap_html
            )
            shap_html = re.sub(r"background: none;", "background: #212946;", shap_html)
            shap_html = "<div style='background-color:#212946;'>" + shap_html + "</div>"
        plot_widget.setHtml(shap_html)

    def web_view_print_pdf(self, page):
        """
        Prints the specified page to a PDF file using a file dialog for file selection.

        This method opens a file dialog for the user to select a location and filename to save the
        PDF.
        The page is then printed to a PDF file with an A4 page size in landscape orientation.

        Parameters
        ----------
        page : qtpy.QtWebEngineWidgets.QWebEnginePage
            The page to be printed to a PDF.

        Returns
        -------
        None

        Notes
        -----
        If the user cancels the file selection, the method will return without performing any action
        """
        mw = get_parent(self.parent, "MainWindow")
        fd = QFileDialog(mw)
        file_path = fd.getSaveFileName(mw, "Print page to PDF", mw.attrs.latest_file_path,
                                       "PDF (*.pdf)")
        if file_path[0] == "":
            return
        mw.attrs.latest_file_path = str(Path(file_path[0]).parent)
        ps = QPageSize(QPageSize.A4)
        pl = QPageLayout(ps, QPageLayout.Orientation.Landscape, QMarginsF())
        page.printToPdf(file_path[0], pageLayout=pl)

    def set_canvas_colors(self, canvas) -> None:
        """
        Set the colors of the canvas for better visibility and aesthetics.

        Parameters
        ----------
        canvas : FigureCanvasQTAgg
            The canvas of the figure to update.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        ax = canvas.figure.gca()
        plot_background_color = mw.attrs.plot_background_color.name()
        plot_text_color = mw.attrs.plot_text_color.name()
        ax.set_facecolor(plot_background_color)
        canvas.figure.set_facecolor(plot_background_color)
        ax.tick_params(axis="x", colors=plot_text_color)
        ax.tick_params(axis="y", colors=plot_text_color)
        ax.yaxis.label.set_color(plot_text_color)
        ax.xaxis.label.set_color(plot_text_color)
        ax.title.set_color(plot_text_color)
        ax.spines["bottom"].set_color(plot_text_color)
        ax.spines["top"].set_color(plot_text_color)
        ax.spines["right"].set_color(plot_text_color)
        ax.spines["left"].set_color(plot_text_color)
        leg = ax.get_legend()
        if leg is not None:
            ax.legend(
                facecolor=plot_background_color,
                labelcolor=plot_text_color,
                prop={"size": int(environ["plot_font_size"])},
            )
        try:
            canvas.draw()
        except (ValueError, AttributeError):
            pass
