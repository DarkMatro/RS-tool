import warnings
from asyncio import create_task, sleep, wait, get_event_loop
from collections import defaultdict
from datetime import datetime
from gc import collect
from logging import critical, error, info, warning
from os import environ, getenv
from pathlib import Path
from shelve import open as shelve_open
from sys import exit
from traceback import format_exc
from zipfile import ZipFile, ZIP_DEFLATED
from src.ui.MultiLine import MultiLine
import numpy as np
from asyncqtpy import QEventLoop
from asyncqtpy import asyncSlot
from matplotlib import pyplot as plt
from pandas import DataFrame, ExcelWriter
from psutil import cpu_percent
from pyqtgraph import (
    setConfigOption,
    PlotDataItem,
)
from qtpy.QtCore import Qt, QModelIndex, QTimer, QMarginsF
from qtpy.QtGui import QFont, QIcon, QCloseEvent, QColor, QPageLayout, QPageSize
from qtpy.QtWidgets import (
    QUndoStack,
    QMenu,
    QMainWindow,
    QAction,
    QFileDialog,
    QLineEdit,
    QInputDialog,
    QTableView,
    QScrollArea,
)
from qtpy.QtWinExtras import QWinTaskbarButton

from qfluentwidgets import (
    IndeterminateProgressBar,
    ProgressBar,
    StateToolTip,
    MessageBox,
)
from src.mutual_functions.static_functions import show_error_msg
from src.data.config import get_config
from src.files.help import action_help
from src.mutual_functions.static_functions import (
    get_memory_used,
)
from src.mw_page4_stat_analysis import StatAnalysisLogic
from src.mw_page5_predict import PredictLogic
from src.pandas_tables import (
    PandasModelPredictTable,
    PandasModelPCA,
)
from src.ui.ui_main_window import Ui_MainWindow
from src.widgets.setting_window import SettingWindow
from ..backend.context import Context
from ..backend.project import Project
from src.backend.progress import Progress
from ..data.plotting import initial_stat_plot
from ..ui.ui_convert_widget import Ui_ConvertForm
from ..ui.ui_import_widget import Ui_ImportForm
from src.ui.ui_cut_widget import Ui_CutForm
from src.ui.ui_normalize_widget import Ui_NormalizeForm
from src.ui.ui_smooth_widget import Ui_SmoothForm
from src.ui.ui_bl_widget import Ui_BaselineForm
from src.ui.ui_average_widget import Ui_AverageForm
from ..widgets.drag_items import DragWidget, DragItem


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Main window widget with all user interface elements.

    Parameters
    ----------
    event_loop : QEventLoop
        name of primary color
    """

    def __init__(self, event_loop: QEventLoop) -> None:
        super().__init__(None)
        self.loop = event_loop or get_event_loop()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.plot_background_color = QColor(environ["plotBackground"])
        self.plot_text_color_value = environ["plotText"]
        self.plot_text_color = QColor(self.plot_text_color_value)
        self.context = Context(self)
        self.progress = Progress(self)

        self.widgets = {"stateTooltip": None, "taskbar": None}
        self.project = Project(self)
        self.current_futures = []
        self.state = defaultdict(bool)
        self.break_event = None
        self.auto_save_timer = None
        self.cpu_load = None
        self.timer_mem_update = None
        self.action_redo = None
        self.action_undo = None
        self.progressBar = None
        self.plot_background_color_web = None
        self.current_executor = None
        self.time_start = None
        self.task_mem_update = None
        self.export_folder_path = None
        self.dragPos = None
        self.project_path = None
        self.window_maximized = True
        self.latest_file_path = (
                    getenv("APPDATA") + "/RS-tool")  # save folder path for file dialogs

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowFrameSection.TopSection
                            | Qt.WindowType.WindowMinMaxButtonsHint)
        self.keyPressEvent = self.key_press_event
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.widgets["taskbar"] = QWinTaskbarButton()
        self.widgets["taskbar"].setWindow(self.windowHandle())
        self.stat_analysis_logic = StatAnalysisLogic(self)
        self.predict_logic = PredictLogic(self)

        # UNDO/ REDO
        self.undoStack = QUndoStack(self)
        self.undoStack.setUndoLimit(int(environ["undo_limit"]))

        # SET UI DEFINITIONS
        cfg = get_config()
        self.setWindowIcon(QIcon(cfg["logo"]["path"]))
        self.setWindowTitle("Raman Spectroscopy Tool ")
        self.plot_background_color_web = QColor(environ["backgroundMainColor"])
        self.update_icons()
        self.initial_ui_definitions()

        self._initial_menu()
        self._init_left_menu()
        self._init_push_buttons()
        self._initial_all_tables()
        self.initial_right_scrollbar()
        self._initial_plots()
        self.initial_plot_buttons()
        # self.initial_timers()
        self.setAcceptDrops(True)
        self._init_test()
        self.context.set_modified(False)

    def _init_test(self):
        self.ui.chain_layout.setAlignment(Qt.AlignLeft)
        self.drag_widget = DragWidget(orientation=Qt.Orientation.Horizontal)

        form = DragItem(draggable=False)
        import_widget = Ui_ImportForm()
        import_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.input_data)
        self.context.preprocessing.stages.input_data.set_ui(import_widget)
        self.drag_widget.add_item(form)

        form = DragItem(draggable=False)
        convert_widget = Ui_ConvertForm()
        convert_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.convert_data)
        self.context.preprocessing.stages.convert_data.set_ui(convert_widget)
        self.drag_widget.add_item(form)

        form = DragItem(draggable=False)
        cut_widget = Ui_CutForm()
        cut_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.cut_data)
        self.context.preprocessing.stages.cut_data.set_ui(cut_widget)
        self.drag_widget.add_item(form)

        form = DragItem()
        bl_widget = Ui_BaselineForm()
        bl_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.bl_data)
        self.context.preprocessing.stages.bl_data.set_ui(bl_widget)
        self.drag_widget.add_item(form)

        form = DragItem()
        smooth_widget = Ui_SmoothForm()
        smooth_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.smoothed_data)
        self.context.preprocessing.stages.smoothed_data.set_ui(smooth_widget)
        self.drag_widget.add_item(form)

        form = DragItem()
        normalization_widget = Ui_NormalizeForm()
        normalization_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.normalized_data)
        self.context.preprocessing.stages.normalized_data.set_ui(normalization_widget)
        self.drag_widget.add_item(form)

        form = DragItem(draggable=False)
        trim_widget = Ui_CutForm()
        trim_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.trim_data)
        self.context.preprocessing.stages.trim_data.set_ui(trim_widget)
        self.drag_widget.add_item(form)

        form = DragItem(draggable=False)
        av_widget = Ui_AverageForm()
        av_widget.setupUi(form)
        form.set_backend_instance(self.context.preprocessing.stages.av_data)
        self.context.preprocessing.stages.av_data.set_ui(av_widget)
        self.drag_widget.add_item(form)

        self.drag_widget.doubleClickedWidget.connect(self._widget_selected)
        self.ui.chain_layout.addWidget(self.drag_widget)

    def _widget_selected(self, w: DragWidget) -> None:
        self.context.preprocessing.active_stage = w.backend_instance
        self.context.preprocessing.update_plot_item(w.backend_instance.name)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            if self.progress.time_start is not None:
                return
            filenames = [url.toLocalFile() for url in e.mimeData().urls()]
            import_files = [f for f in filenames if Path(f).suffix.lower() in ['.txt', '.asc']]
            if import_files:
                self.context.preprocessing.stages.input_data.import_files(filenames)
            elif len(filenames) == 1 and Path(filenames[0]).suffix.lower() == '.zip':
                if not self.project.can_close_project():
                    return
                self.project.open_project(filenames[0])
            e.accept()
        else:
            pos = e.pos()
            widget = e.source()
            if isinstance(widget, DragItem):
                return
            for n in range(self.ui.chain_layout.count()):
                # Get the widget at each index in turn.
                w = self.ui.chain_layout.itemAt(n).widget()
                if pos.x() < w.x() + w.size().width() // 2:
                    # We didn't drag past this widget. Insert to the left of it.
                    self.ui.chain_layout.insertWidget(n - 1, widget)
                    break
            e.accept()

    def closeEvent(self, a0: QCloseEvent) -> None:
        if not self.can_close_project():
            a0.ignore()
            return

        self.ui.preproc_plot_widget.getPlotItem().close()
        del self.ui.preproc_plot_widget.vertical_line
        del self.ui.preproc_plot_widget.horizontal_line
        exit()

    # region init

    # region plots
    def _initial_preproc_plot_color(self) -> None:
        self.ui.preproc_plot_widget.setBackground(self.plot_background_color)
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item.getAxis("bottom").setPen(self.plot_text_color)
        plot_item.getAxis("left").setPen(self.plot_text_color)
        plot_item.getAxis("bottom").setTextPen(self.plot_text_color)
        plot_item.getAxis("left").setTextPen(self.plot_text_color)

    def initial_scores_plot(self, plot_widget, cl_type=None):
        plot_widget.canvas.axes.cla()
        if cl_type == "LDA":
            function_name = "LD-"
            plot_widget.setVisible(True)
        elif cl_type == "PLS-DA":
            function_name = "PLS-DA-"
        else:
            function_name = "PC-"
        plot_widget.canvas.axes.set_xlabel(
            function_name + "1", fontsize=int(environ["axis_label_font_size"])
        )
        plot_widget.canvas.axes.set_ylabel(
            function_name + "2", fontsize=int(environ["axis_label_font_size"])
        )
        plot_widget.canvas.draw()

    @staticmethod
    def initial_shap_plot(plot_widget) -> None:
        plot_widget.canvas.figure.gca().cla()
        plot_widget.canvas.figure.clf()
        plot_widget.canvas.draw()

    def _initial_force_single_plot(self) -> None:
        self.ui.force_single.page().setHtml("")
        self.ui.force_single.contextMenuEvent = self.force_single_context_menu_event

    def force_single_context_menu_event(self, a0) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction(
            "Save .pdf", lambda: self.web_view_print_pdf(self.ui.force_single.page())
        )
        menu.addAction("Refresh", lambda: self.reload_force(self.ui.force_single))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_force_full_plot(self) -> None:
        self.ui.force_full.page().setHtml("")
        self.ui.force_full.contextMenuEvent = self.force_full_context_menu_event

    def force_full_context_menu_event(self, a0) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction(
            "Save .pdf", lambda: self.web_view_print_pdf(self.ui.force_full.page())
        )
        menu.addAction("Refresh", lambda: self.reload_force(self.ui.force_full, True))
        menu.move(a0.globalPos())
        menu.show()

    def reload_force(self, plot_widget, full: bool = False):
        cl_type = self.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.stat_analysis_logic.latest_stat_result:
            msg = MessageBox(
                "SHAP Force plot refresh error.",
                "Selected classificator was not fitted.",
                self,
                {"Ok"},
            )
            msg.setInformativeText(
                "Try to turn on Use Shapley option before fit classificator."
            )
            msg.exec()
            return
        shap_html = "shap_html_full" if full else "shap_html"
        if shap_html not in self.stat_analysis_logic.latest_stat_result[cl_type]:
            msg = MessageBox(
                "SHAP Force plot refresh error.",
                "Selected classificator was fitted without Shapley calculation.",
                self,
                {"Ok"},
            )
            msg.setInformativeText(
                "Try to turn on Use Shapley option before fit classificator."
            )
            msg.exec()
            return
        plot_widget.setHtml(
            self.stat_analysis_logic.latest_stat_result[cl_type][shap_html]
        )

    def web_view_print_pdf(self, page):
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(
            self, "Print page to PDF", self.latest_file_path, "PDF (*.pdf)"
        )
        if file_path[0] == "":
            return
        self.latest_file_path = str(Path(file_path[0]).parent)
        ps = QPageSize(QPageSize.A4)
        pl = QPageLayout(ps, QPageLayout.Orientation.Landscape, QMarginsF())
        page.printToPdf(file_path[0], pageLayout=pl)

    @asyncSlot()
    async def _initial_stat_plots_color(self) -> None:
        """
            Set colors for all stat plots at app start and at SunBtn pressed event.
        Returns
        -------
            None
        """
        plot_widgets = [
            self.ui.decision_score_plot_widget,
            self.ui.decision_boundary_plot_widget,
            self.ui.violin_describe_plot_widget,
            self.ui.bootstrap_plot_widget,
            self.ui.boxplot_describe_plot_widget,
            self.ui.dm_plot_widget,
            self.ui.roc_plot_widget,
            self.ui.pr_plot_widget,
            self.ui.perm_imp_test_plot_widget,
            self.ui.perm_imp_train_plot_widget,
            self.ui.partial_depend_plot_widget,
            self.ui.tree_plot_widget,
            self.ui.features_plot_widget,
            self.ui.calibration_plot_widget,
            self.ui.det_curve_plot_widget,
            self.ui.learning_plot_widget,
            self.ui.pca_scores_plot_widget,
            self.ui.pca_loadings_plot_widget,
            self.ui.shap_beeswarm,
            self.ui.shap_means,
            self.ui.shap_heatmap,
            self.ui.shap_scatter,
            self.ui.shap_decision,
            self.ui.shap_waterfall,
        ]
        for pl in plot_widgets:
            self.set_canvas_colors(pl.canvas)
        if self.ui.current_group_shap_comboBox.currentText() == "":
            return
        cl_type = self.ui.current_classificator_comboBox.currentText()
        await self.loop.run_in_executor(None, self.update_shap_plots)
        await self.loop.run_in_executor(None, self.update_shap_plots_by_instance)
        self.stat_analysis_logic.update_force_single_plots(cl_type)
        self.stat_analysis_logic.update_force_full_plots(cl_type)

    def update_shap_plots(self) -> None:
        cl_type = self.ui.current_classificator_comboBox.currentText()
        self.stat_analysis_logic.do_update_shap_plots(cl_type)

    def update_shap_plots_by_instance(self) -> None:
        cl_type = self.ui.current_classificator_comboBox.currentText()
        self.stat_analysis_logic.do_update_shap_plots_by_instance(cl_type)

    @asyncSlot()
    async def update_shap_scatters(self) -> None:
        self.loop.run_in_executor(None, self.do_update_shap_scatters)

    def do_update_shap_scatters(self):
        classificator = self.ui.current_classificator_comboBox.currentText()
        if (
                classificator in self.stat_analysis_logic.latest_stat_result
                and "shap_values"
                in self.stat_analysis_logic.latest_stat_result[classificator]
        ):
            target_names = self.stat_analysis_logic.latest_stat_result[classificator][
                "target_names"
            ]
            if self.ui.current_group_shap_comboBox.currentText() not in target_names:
                return
            i = int(
                np.where(
                    target_names == self.ui.current_group_shap_comboBox.currentText()
                )[0][0]
            )
            self.stat_analysis_logic.update_shap_scatter_plot(False, i, classificator)

    def set_canvas_colors(self, canvas) -> None:
        """
        Set plot colors to canvas. Different colors for dark and light theme.
        Parameters
        ----------
        canvas

        Returns
        -------
            None
        """
        ax = canvas.figure.gca()
        ax.set_facecolor(self.plot_background_color.name())
        canvas.figure.set_facecolor(self.plot_background_color.name())
        ax.tick_params(axis="x", colors=self.plot_text_color.name())
        ax.tick_params(axis="y", colors=self.plot_text_color.name())
        ax.yaxis.label.set_color(self.plot_text_color.name())
        ax.xaxis.label.set_color(self.plot_text_color.name())
        ax.title.set_color(self.plot_text_color.name())
        ax.spines["bottom"].set_color(self.plot_text_color.name())
        ax.spines["top"].set_color(self.plot_text_color.name())
        ax.spines["right"].set_color(self.plot_text_color.name())
        ax.spines["left"].set_color(self.plot_text_color.name())
        leg = ax.get_legend()
        if leg is not None:
            ax.legend(
                facecolor=self.plot_background_color.name(),
                labelcolor=self.plot_text_color.name(),
                prop={"size": int(environ["plot_font_size"])},
            )
        try:
            canvas.draw()
        except ValueError | np.linalg.LinAlgError:
            pass

    def _initial_plots(self) -> None:
        setConfigOption("antialias", True)
        self.ui.stackedWidget_mainpages.currentChanged.connect(
            self.stacked_widget_changed
        )
        self._initial_all_stat_plots()

    def _initial_pca_plots(self) -> None:
        self.initial_scores_plot(self.ui.pca_scores_plot_widget)
        self.initial_scores_plot(self.ui.pca_loadings_plot_widget)

    def _initial_all_stat_plots(self) -> None:
        for pw in (self.ui.decision_score_plot_widget, self.ui.decision_boundary_plot_widget,
                   self.ui.violin_describe_plot_widget, self.ui.boxplot_describe_plot_widget,
                   self.ui.dm_plot_widget, self.ui.roc_plot_widget, self.ui.pr_plot_widget,
                   self.ui.perm_imp_train_plot_widget, self.ui.perm_imp_test_plot_widget,
                   self.ui.partial_depend_plot_widget, self.ui.tree_plot_widget,
                   self.ui.features_plot_widget, self.ui.calibration_plot_widget,
                   self.ui.det_curve_plot_widget, self.ui.learning_plot_widget,
                   self.ui.bootstrap_plot_widget):
            initial_stat_plot(pw)
        self._initial_pca_plots()
        self._initial_force_single_plot()
        self._initial_force_full_plot()

        shap_plots = [
            self.ui.shap_beeswarm,
            self.ui.shap_means,
            self.ui.shap_heatmap,
            self.ui.shap_scatter,
            self.ui.shap_waterfall,
            self.ui.shap_decision,
        ]
        for sp in shap_plots:
            self.initial_shap_plot(sp)

        self.initial_plots_set_fonts()
        self.initial_plots_labels()
        self._initial_stat_plots_color()

    def initial_plots_set_fonts(self) -> None:
        plot_font = QFont(
            "AbletonSans", int(environ["plot_font_size"]), QFont.Weight.Normal
        )
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item.getAxis("bottom").setStyle(tickFont=plot_font)
        plot_item.getAxis("left").setStyle(tickFont=plot_font)
        self.ui.deconv_plot_widget.getPlotItem().getAxis("bottom").setStyle(tickFont=plot_font)
        self.ui.deconv_plot_widget.getPlotItem().getAxis("left").setStyle(tickFont=plot_font)

        plt.rcParams.update({"font.size": int(environ["plot_font_size"])})

    def get_plot_label_style(self) -> dict:
        return {
            "color": self.plot_text_color_value,
            "font-size": str(environ["axis_label_font_size"]) + "pt",
            "font-family": "AbletonSans",
        }

    def initial_plots_labels(self) -> None:
        label_style = self.get_plot_label_style()
        print(label_style)
        self.ui.preproc_plot_widget.setLabel(
            "left", "Intensity, rel. un.", units="", **label_style
        )
        self.ui.deconv_plot_widget.setLabel(
            "left", "Intensity, rel. un.", units="", **label_style
        )
        self.ui.deconv_plot_widget.setLabel(
            "bottom",
            "Raman shift, cm\N{superscript minus}\N{superscript one}",
            units="",
            **label_style,
        )

    # endregion

    # region MenuBar

    def _initial_menu(self) -> None:
        """
        Custom menu bar. Add menu for buttons.
        Menus: File, Edit, Preprocessing, Stat Analysis
        """
        self._init_file_menu()
        self._init_edit_menu()
        self._init_stat_analysis_menu()

    def _init_file_menu(self) -> None:
        file_menu = QMenu(self)
        file_menu.addAction("New Project", self.project.action_new_project)
        file_menu.addAction("Open Project", self.project.action_open_project)
        recent_menu = file_menu.addMenu("Open Recent")
        self.project.set_recent_menu(recent_menu)
        file_menu.addSeparator()

        export_menu = file_menu.addMenu("Export")
        export_menu.addAction("Tables to excel", self.action_export_table_excel)
        export_menu.addAction("Production project", self.action_save_production_project)
        export_menu.addAction(
            "Decomposed lines to .csv", self.action_save_decomposed_to_csv
        )

        fit_template_menu = file_menu.addMenu("Fit template")
        fit_template_menu.addAction("Import", self.action_import_fit_template)
        fit_template_menu.addAction("Export", self.action_export_fit_template)

        file_menu.addSeparator()

        file_menu_save_all_action = QAction("Save all", file_menu)
        file_menu_save_all_action.triggered.connect(self.project.action_save_project)
        file_menu_save_all_action.setShortcut("Ctrl+S")
        file_menu_save_as_action = QAction("Save as", file_menu)
        file_menu_save_as_action.triggered.connect(self.action_save_as)
        file_menu_save_as_action.setShortcut("Shift+S")
        actions = [file_menu_save_all_action, file_menu_save_as_action]
        file_menu.addActions(actions)
        file_menu.addSeparator()

        file_menu.addAction("Close Project", self.project.action_close_project)
        file_menu.addSeparator()

        file_menu_help = QAction("Help", file_menu)
        file_menu_help.triggered.connect(action_help)
        file_menu_help.setShortcut("F1")
        file_menu.addAction(file_menu_help)

        self.ui.FileBtn.setMenu(file_menu)

    def _init_edit_menu(self) -> None:
        edit_menu = QMenu(self)
        self.action_undo = QAction("Undo")
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_redo = QAction("Redo")
        self.action_redo.triggered.connect(self.redo)
        self.action_redo.setShortcut("Ctrl+Y")
        actions = [self.action_undo, self.action_redo]
        edit_menu.addActions(actions)
        edit_menu.setToolTipsVisible(True)
        self.action_undo.setToolTip("")
        edit_menu.addSeparator()

        clear_menu = edit_menu.addMenu("Clear")
        clear_menu.addAction("Fitting lines",
                             self.context.decomposition.tables.decomp_lines.clear_all_deconv_lines)
        clear_menu.addAction(
            "All fitting data", self.context.decomposition.reset
        )
        clear_menu.addSeparator()
        clear_menu.addAction("Smoothed dataset", self.context.datasets.reset_smoothed_dataset_table)
        clear_menu.addAction(
            "Baseline corrected dataset", self.context.datasets.reset_baselined_dataset_table
        )
        clear_menu.addAction(
            "Decomposed dataset", self.context.datasets.reset_deconvoluted_dataset_table
        )
        clear_menu.addAction("Ignore features", self.context.datasets.reset_ignore_dataset_table)
        clear_menu.addSeparator()
        clear_menu.addAction("LDA", lambda: self.clear_selected_step("LDA"))
        clear_menu.addAction(
            "Logistic regression",
            lambda: self.clear_selected_step("Logistic regression"),
        )
        clear_menu.addAction("NuSVC", lambda: self.clear_selected_step("NuSVC"))
        clear_menu.addAction(
            "Nearest Neighbors", lambda: self.clear_selected_step("Nearest Neighbors")
        )
        clear_menu.addAction("GPC", lambda: self.clear_selected_step("GPC"))
        clear_menu.addAction(
            "Naive Bayes", lambda: self.clear_selected_step("Naive Bayes")
        )
        clear_menu.addAction("MLP", lambda: self.clear_selected_step("MLP"))
        clear_menu.addAction(
            "Decision Tree", lambda: self.clear_selected_step("Decision Tree")
        )
        clear_menu.addAction(
            "Random Forest", lambda: self.clear_selected_step("Random Forest")
        )
        clear_menu.addAction("AdaBoost", lambda: self.clear_selected_step("AdaBoost"))
        clear_menu.addAction("XGBoost", lambda: self.clear_selected_step("XGBoost"))
        clear_menu.addAction("Voting", lambda: self.clear_selected_step("Voting"))
        clear_menu.addAction("Stacking", lambda: self.clear_selected_step("Stacking"))
        clear_menu.addAction("PCA", lambda: self.clear_selected_step("PCA"))
        clear_menu.addAction("PLS-DA", lambda: self.clear_selected_step("PLS-DA"))
        clear_menu.addSeparator()
        clear_menu.addAction("Predicted", lambda: self.clear_selected_step("Page5"))
        self.ui.EditBtn.setMenu(edit_menu)

    def _init_stat_analysis_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Fit", self.context.ml.fit_classificator)
        menu.addAction("PCA", lambda: self.context.ml.fit_classificator("PCA"))
        menu.addAction("Refresh plots", self.redraw_stat_plots)
        menu.addAction("Refresh SHAP", self.refresh_shap_push_button_clicked)
        menu.addAction("Refresh learning curve", self.stat_analysis_logic.refresh_learning_curve)
        self.ui.stat_analysis_btn.setMenu(menu)

    # endregion

    # region left_side_menu

    def _init_left_menu(self) -> None:
        self.ui.left_side_frame.setFixedWidth(350)
        self.ui.left_hide_frame.hide()
        self.ui.dec_list_btn.setVisible(False)
        self.ui.gt_add_Btn.setToolTip("Add new group")
        self.ui.gt_add_Btn.clicked.connect(self.context.group_table.add_new_group)
        self.ui.gt_dlt_Btn.setToolTip("Delete selected group")
        self.ui.gt_dlt_Btn.clicked.connect(self.context.group_table.dlt_selected_group)
        self._init_params_value_changed()
        self.ui.intervals_gb.toggled.connect(self.intervals_gb_toggled)

    def _init_push_buttons(self) -> None:
        self.ui.page5_predict.clicked.connect(self.predict)

    def _init_params_value_changed(self) -> None:
        self.ui.leftsideBtn.clicked.connect(self.left_side_btn_clicked)
        self.ui.dec_list_btn.clicked.connect(self.dec_list_btn_clicked)
        self.ui.stat_param_btn.clicked.connect(self.stat_param_btn_clicked)
        self.ui.update_partial_dep_pushButton.clicked.connect(self.current_dep_feature_changed)

    def intervals_gb_toggled(self, b: bool) -> None:
        self.context.set_modified()
        self.ui.fit_borders_TableView.setVisible(b)
        if b:
            self.ui.intervals_gb.setMaximumHeight(200)
        else:
            self.ui.intervals_gb.setMaximumHeight(0)

    @asyncSlot()
    async def current_group_shap_changed(self, g: str = "") -> None:
        await self.loop.run_in_executor(None, self.update_shap_plots)
        await self.loop.run_in_executor(None, self.update_shap_plots_by_instance)
        cl_type = self.ui.current_classificator_comboBox.currentText()
        self.stat_analysis_logic.update_force_single_plots(cl_type)
        self.stat_analysis_logic.update_force_full_plots(cl_type)

    @asyncSlot()
    async def current_instance_changed(self, _: str = "") -> None:
        await self.loop.run_in_executor(None, self.update_shap_plots_by_instance)
        cl_type = self.ui.current_classificator_comboBox.currentText()
        self.stat_analysis_logic.update_force_single_plots(cl_type)

    # endregion

    # region Tables

    def _initial_all_tables(self) -> None:
        self._initial_predict_dataset_table()
        self._initial_pca_features_table()

    # region pca plsda  features
    def _initial_pca_features_table(self) -> None:
        self._reset_pca_features_table()

    def _reset_pca_features_table(self) -> None:
        df = DataFrame(columns=["feature", "PC-1", "PC-2"])
        model = PandasModelPCA(self, df)
        self.ui.pca_features_table_view.setModel(model)

    # endregion

    # region predict dataset

    def _initial_predict_dataset_table(self) -> None:
        self._reset_predict_dataset_table()
        self.ui.predict_table_view.verticalScrollBar().valueChanged.connect(
            self.move_side_scrollbar
        )
        self.ui.predict_table_view.verticalHeader().setVisible(False)

    def _reset_predict_dataset_table(self) -> None:
        df = DataFrame(columns=["Filename"])
        model = PandasModelPredictTable(self, df)
        self.ui.predict_table_view.setModel(model)

    # endregion

    # endregion

    # region other

    def initial_right_scrollbar(self) -> None:
        self.ui.verticalScrollBar.setVisible(False)
        self.ui.verticalScrollBar.setMinimum(1)
        self.ui.verticalScrollBar.enterEvent = self.vertical_scroll_bar_enter_event
        self.ui.verticalScrollBar.leaveEvent = self.vertical_scroll_bar_leave_event
        self.ui.verticalScrollBar.valueChanged.connect(
            self.vertical_scroll_bar_value_changed
        )
        self.ui.data_tables_tab_widget.currentChanged.connect(
            self.decide_vertical_scroll_bar_visible
        )
        self.ui.page1Btn.clicked.connect(self.page1_btn_clicked)
        self.ui.page2Btn.clicked.connect(self.page2_btn_clicked)
        self.ui.page3Btn.clicked.connect(self.page3_btn_clicked)
        self.ui.page4Btn.clicked.connect(self.page4_btn_clicked)
        self.ui.page5Btn.clicked.connect(self.page5_btn_clicked)

    def initial_plot_buttons(self) -> None:
        self.ui.crosshairBtn.clicked.connect(self.crosshair_btn_clicked)
        self.ui.by_one_control_button.clicked.connect(
            self.by_one_control_button_clicked
        )
        self.ui.by_group_control_button.clicked.connect(self.by_group_control_button)
        self.ui.by_group_control_button.mouseDoubleClickEvent = self.by_group_control_button_double_clicked

        self.ui.all_control_button.clicked.connect(self.all_control_button)
        self.ui.lr_movableBtn.clicked.connect(self.linear_region_movable_btn_clicked)
        self.ui.lr_showHideBtn.clicked.connect(self.linear_region_show_hide_btn_clicked)
        self.ui.sun_Btn.clicked.connect(self.change_plots_bckgrnd)

    def initial_timers(self) -> None:
        self.timer_mem_update = QTimer(self)
        self.timer_mem_update.timeout.connect(self.set_timer_memory_update)
        self.timer_mem_update.start(1000)
        self.cpu_load = QTimer(self)
        self.cpu_load.timeout.connect(self.set_cpu_load)
        self.cpu_load.start(300)

    def initial_ui_definitions(self) -> None:
        self.ui.maximizeRestoreAppBtn.setToolTip("Maximize")
        self.ui.unsavedBtn.hide()
        self.ui.titlebar.mouseDoubleClickEvent = self.double_click_maximize_restore
        self.ui.titlebar.mouseMoveEvent = self.move_window
        self.ui.titlebar.mouseReleaseEvent = self.titlebar_mouse_release_event
        self.ui.right_buttons_frame.mouseMoveEvent = self.move_window
        self.ui.right_buttons_frame.mouseReleaseEvent = (
            self.titlebar_mouse_release_event
        )

        self.ui.minimizeAppBtn.clicked.connect(lambda: self.showMinimized())
        self.ui.maximizeRestoreAppBtn.clicked.connect(lambda: self.maximize_restore())
        self.ui.closeBtn.clicked.connect(lambda: self.close())
        self.ui.settingsBtn.clicked.connect(lambda: SettingWindow(self).show())

    def titlebar_mouse_release_event(self, _) -> None:
        self.setWindowOpacity(1)

    def move_window(self, mouse_event) -> None:
        # IF MAXIMIZED CHANGE TO NORMAL
        if self.window_maximized:
            self.maximize_restore()
        # MOVE WINDOW
        if mouse_event.buttons() == Qt.MouseButton.LeftButton:
            try:
                new_pos = self.pos() + mouse_event.globalPos() - self.dragPos
                self.setWindowOpacity(0.9)
                self.move(new_pos)
                self.dragPos = mouse_event.globalPos()
                mouse_event.accept()
            except Exception:
                pass

    def double_click_maximize_restore(self, mouse_event) -> None:
        # IF DOUBLE CLICK CHANGE STATUS
        if mouse_event.type() == 4:
            timer = QTimer(self)
            timer.singleShot(250, self.maximize_restore)

    def maximize_restore(self) -> None:
        if not self.window_maximized:
            self.showMaximized()
            self.window_maximized = True
            self.ui.maximizeRestoreAppBtn.setToolTip("Restore")
            self.set_icon_for_restore_button()
        else:
            self.window_maximized = False
            self.showNormal()
            self.ui.maximizeRestoreAppBtn.setToolTip("Maximize")
            self.set_icon_for_restore_button()

    def set_icon_for_restore_button(self) -> None:
        if "Light" in environ["theme"] and self.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down_black.svg")
            )
        elif "Light" in environ["theme"] and not self.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-up_black.svg")
            )
        elif self.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down.svg")
            )
        else:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-up.svg")
            )

    def show_hide_left_menu(self) -> None:
        if self.ui.left_side_main.isHidden():
            self.ui.left_side_frame.setMaximumWidth(350)
            self.ui.left_side_frame.setFixedWidth(350)
            self.ui.left_side_main.show()
            self.ui.left_hide_frame.hide()
            self.ui.main_frame.layout().setSpacing(10)
            self.ui.left_side_up_frame.setMaximumHeight(40)
            self.ui.dec_list_btn.setVisible(False)
            self.ui.stat_param_btn.setVisible(False)
            if "Light" in environ["bckgrnd_theme"]:
                self.ui.leftsideBtn.setIcon(
                    QIcon("material/resources/source/chevron-left_black.svg")
                )
            else:
                self.ui.leftsideBtn.setIcon(
                    QIcon("material/resources/source/chevron-left.svg")
                )
        elif self.ui.left_side_main.isVisible():
            self.ui.left_side_main.hide()
            self.ui.left_hide_frame.show()
            self.ui.left_side_frame.setMaximumWidth(35)
            self.ui.left_side_frame.setFixedWidth(35)
            self.ui.left_side_up_frame.setMaximumHeight(120)
            self.ui.dec_list_btn.setVisible(True)
            self.ui.stat_param_btn.setVisible(True)
            self.ui.main_frame.layout().setSpacing(1)
            if "Light" in environ["bckgrnd_theme"]:
                self.ui.leftsideBtn.setIcon(
                    QIcon("material/resources/source/sliders_black.svg")
                )
                self.ui.dec_list_btn.setIcon(
                    QIcon("material/resources/source/align-justify_black.svg")
                )
                self.ui.stat_param_btn.setIcon(
                    QIcon("material/resources/source/percent_black.svg")
                )
            else:
                self.ui.leftsideBtn.setIcon(
                    QIcon("material/resources/source/sliders.svg")
                )
                self.ui.dec_list_btn.setIcon(
                    QIcon("material/resources/source/align-justify.svg")
                )
                self.ui.stat_param_btn.setIcon(
                    QIcon("material/resources/source/percent.svg")
                )

    def left_side_btn_clicked(self) -> None:
        self.show_hide_left_menu()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_1)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_1)

    def dec_list_btn_clicked(self) -> None:
        self.show_hide_left_menu()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_2)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_2)

    def stat_param_btn_clicked(self) -> None:
        self.show_hide_left_menu()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_3)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_3)

    def change_plots_bckgrnd(self) -> None:
        if self.ui.sun_Btn.isChecked():
            self.plot_text_color_value = environ["inversePlotText"]
            self.plot_text_color = QColor(self.plot_text_color_value)
            self.plot_background_color = QColor(environ["inversePlotBackground"])
            self.plot_background_color_web = QColor(environ["inversePlotBackground"])
            plt.style.use(["default"])
        else:
            self.plot_text_color_value = environ["plotText"]
            self.plot_text_color = QColor(self.plot_text_color_value)
            self.plot_background_color = QColor(environ["plotBackground"])
            self.plot_background_color_web = QColor(environ["backgroundMainColor"])
            plt.style.use(["dark_background"])
        self._initial_preproc_plot_color()
        self.context.decomposition.graph_drawing.initial_deconv_plot_color()
        self._initial_stat_plots_color()
        self.initial_plots_labels()

    # endregion

    # endregion

    # region Plot buttons

    # region crosshair button
    def linear_region_movable_btn_clicked(self) -> None:
        b = not self.ui.lr_movableBtn.isChecked()
        self.context.preprocessing.stages.cut_data.linear_region.setMovable(b)
        self.context.decomposition.plotting.linear_region.setMovable(b)

    def linear_region_show_hide_btn_clicked(self) -> None:
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item_dec = self.ui.deconv_plot_widget.getPlotItem()
        if self.ui.lr_showHideBtn.isChecked():
            plot_item.addItem(self.context.preprocessing.stages.cut_data.linear_region)
            plot_item_dec.addItem(self.context.decomposition.plotting.linear_region)
        else:
            plot_item.removeItem(self.context.preprocessing.stages.cut_data.linear_region)
            plot_item_dec.removeItem(self.context.decomposition.plotting.linear_region)

    def crosshair_btn_clicked(self) -> None:
        """Add crosshair with coordinates at title."""
        if self.ui.stackedWidget_mainpages.currentIndex() == 0:
            self.crosshair_preproc_plot()
        elif self.ui.stackedWidget_mainpages.currentIndex() == 1:
            self.crosshair_btn_clicked_for_deconv_plot()

    def crosshair_preproc_plot(self) -> None:
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item.removeItem(self.ui.preproc_plot_widget.vertical_line)
        plot_item.removeItem(self.ui.preproc_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            plot_item.addItem(
                self.ui.preproc_plot_widget.vertical_line, ignoreBounds=True
            )
            plot_item.addItem(
                self.ui.preproc_plot_widget.horizontal_line, ignoreBounds=True
            )
        elif not self.ui.by_one_control_button.isChecked():
            self.context.preprocessing.set_preproc_title(self.drag_widget.get_current_widget())

    def crosshair_btn_clicked_for_deconv_plot(self) -> None:
        plot_item = self.ui.deconv_plot_widget.getPlotItem()
        plot_item.removeItem(self.ui.deconv_plot_widget.vertical_line)
        plot_item.removeItem(self.ui.deconv_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            plot_item.addItem(self.ui.deconv_plot_widget.vertical_line, ignoreBounds=True)
            plot_item.addItem(self.ui.deconv_plot_widget.horizontal_line, ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            new_title = (
                    '<span style="font-family: AbletonSans; color:'
                    + environ["plotText"]
                    + ';font-size:14pt">'
                    + self.context.decomposition.data.current_spectrum_name
                    + "</span>"
            )
            self.ui.deconv_plot_widget.setTitle(new_title)

    def update_crosshair(self, point_event) -> None:
        """Paint crosshair on mouse"""

        coordinates = point_event[0]
        if (
                self.ui.preproc_plot_widget.sceneBoundingRect().contains(coordinates)
                and self.ui.crosshairBtn.isChecked()
        ):
            mouse_point = self.ui.preproc_plot_widget.plotItem.vb.mapSceneToView(
                coordinates
            )
            self.ui.preproc_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:"
                + environ["secondaryColor"]
                + "'>x=%0.1f,   <span style=>y=%0.1f</span>"
                % (mouse_point.x(), mouse_point.y())
            )
            self.ui.preproc_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.preproc_plot_widget.horizontal_line.setPos(mouse_point.y())


    def update_crosshair_deconv_plot(self, point_event) -> None:
        coordinates = point_event[0]
        if (
                self.ui.deconv_plot_widget.sceneBoundingRect().contains(coordinates)
                and self.ui.crosshairBtn.isChecked()
        ):
            mouse_point = self.ui.deconv_plot_widget.plotItem.vb.mapSceneToView(
                coordinates
            )
            self.ui.deconv_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:"
                + environ["secondaryColor"]
                + "'>x=%0.1f,   <span style=>y=%0.1f</span>"
                % (mouse_point.x(), mouse_point.y())
            )
            self.ui.deconv_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.deconv_plot_widget.horizontal_line.setPos(mouse_point.y())

    # endregion

    # region '1' button

    @asyncSlot()
    async def by_one_control_button_clicked(self) -> None:
        """
        using with button self.ui.by_one_control_button for hide all plot item besides selected
         in input table"""
        if self.ui.by_one_control_button.isChecked():
            self.ui.by_group_control_button.setChecked(False)
            self.ui.all_control_button.setChecked(False)
            tasks = [create_task(self.update_plots_for_single())]
            await wait(tasks)
        else:
            self.ui.by_one_control_button.setChecked(True)

    async def update_plots_for_single(self) -> None:
        """loop to set visible for all plot items"""
        self.ui.statusBar.showMessage("Updating plot...")
        current_index = self.ui.input_table.selectionModel().currentIndex()
        if current_index.row() == -1:
            return
        current_spectrum_name = self.ui.input_table.model().get_filename_by_row(current_index.row())
        group_number = self.ui.input_table.model().cell_data(current_index.row(), 2)
        new_title = (
                '<span style="font-family: AbletonSans; color:'
                + environ["plotText"]
                + ';font-size:14pt">'
                + current_spectrum_name
                + "</span>"
        )
        tasks = [create_task(self.update_single_plot(new_title, current_spectrum_name,
                                                     group_number))]
        await wait(tasks)

        self.ui.statusBar.showMessage("Plot updated", 5000)

    async def update_single_plot(self, new_title: str, current_spectrum_name: str,
                                 group_number: str) -> None:
        data_items = self.ui.preproc_plot_widget.getPlotItem().listDataItems()
        if len(data_items) <= 0:
            return
        self.ui.preproc_plot_widget.setTitle(new_title)
        for i in data_items:
            i.setVisible(False)
        current_widget = self.drag_widget.get_current_widget_name()
        match current_widget:
            case "InputData":
                data = self.context.preprocessing.stages.input_data.data
            case 'ConvertData':
                data = self.context.preprocessing.stages.convert_data.data
            case 'CutData':
                data = self.context.preprocessing.stages.cut_data.data
            case 'NormalizedData':
                data = self.context.preprocessing.stages.normalized_data.data
            case 'SmoothedData':
                data = self.context.preprocessing.stages.smoothed_data.data
            case 'BaselineData':
                data = self.context.preprocessing.stages.bl_data.data
            case 'TrimData':
                data = self.context.preprocessing.stages.trim_data.data
            case 'AvData':
                return
            case _:
                return
        arr = data[current_spectrum_name]
        if self.context.preprocessing.one_curve:
            self.ui.preproc_plot_widget.getPlotItem().removeItem(
                self.context.preprocessing.one_curve
            )
        self.context.preprocessing.one_curve = self.get_curve_plot_data_item(arr, group_number)
        self.ui.preproc_plot_widget.getPlotItem().addItem(
            self.context.preprocessing.one_curve, kargs=["ignoreBounds", "skipAverage"]
        )
        if (
                self.ui.input_table.selectionModel().currentIndex().row() != -1
                and current_widget == "InputData"
                and len(self.context.preprocessing.stages.input_data.before_despike_data) > 0
                and current_spectrum_name
                in self.context.preprocessing.stages.input_data.before_despike_data
        ):
            tasks = [create_task(
                self.context.preprocessing.stages.input_data.despike_history_add_plot(
                    current_spectrum_name))]
            await wait(tasks)
        elif (self.ui.input_table.selectionModel().currentIndex().row() != -1
              and current_widget == "BaselineData"
              and len(self.context.preprocessing.stages.bl_data.baseline_data) > 0
              and current_spectrum_name
              in self.context.preprocessing.stages.bl_data.baseline_data
        ):
            tasks = [create_task(
                self.context.preprocessing.stages.bl_data.baseline_add_plot(current_spectrum_name))]
            await wait(tasks)
        self.ui.preproc_plot_widget.getPlotItem().getViewBox().updateAutoRange()

    # endregion

    # region 'G' button
    @asyncSlot()
    async def by_group_control_button(self) -> None:
        """using with button self.ui.by_group_control_button for hide all plot items besides
         selected in group table"""
        print('by_group_control_button')
        if self.ui.by_group_control_button.isChecked():
            self.ui.by_one_control_button.setChecked(False)
            self.ui.all_control_button.setChecked(False)
            tasks = [create_task(
                self.context.preprocessing.stages.input_data.despike_history_remove_plot())]
            await wait(tasks)
            await self.update_plots_for_group(None)
        else:
            self.ui.by_group_control_button.setChecked(True)

    def by_group_control_button_double_clicked(self, _) -> None:
        if self.context.group_table.table_widget.model().rowCount() < 2:
            return
        input_dialog = QInputDialog(self)
        result = input_dialog.getText(
            self,
            "Choose visible groups",
            "Write groups numbers to show (example: 1, 2, 3):",
        )
        if not result[1]:
            return
        v = list(result[0].strip().split(","))
        self.context.preprocessing.stages.input_data.despike_history_remove_plot()
        groups = [int(x) for x in v]
        self.update_plots_for_group(groups)

    @asyncSlot()
    async def update_plots_for_group(self, current_group: list[int] | None) -> None:
        """loop to set visible for all plot items in group"""
        if not current_group:
            current_row = self.context.group_table.table_widget.selectionModel().currentIndex().row()
            current_group_name = self.context.group_table.table_widget.model().cell_data(
                current_row, 0)
            current_group = [current_row + 1]
            new_title = (
                    '<span style="font-family: AbletonSans; color:'
                    + environ["plotText"]
                    + ';font-size:14pt">'
                    + current_group_name
                    + "</span>"
            )
        else:
            new_title = (
                    '<span style="font-family: AbletonSans; color:'
                    + environ["plotText"]
                    + ';font-size:14pt">'
                    + str(current_group)
                    + "</span>"
            )
        self.ui.preproc_plot_widget.setTitle(new_title)

        self.ui.statusBar.showMessage("Updating plot...")
        tasks = [
            create_task(self.context.preprocessing.stages.input_data.despike_history_remove_plot()),
        ]
        await wait(tasks)
        tasks = [
            create_task(self.update_group_plot(current_group)),
        ]
        await wait(tasks)
        self.ui.statusBar.showMessage("Plot updated", 5000)

    async def update_group_plot(self, current_group: list[int]) -> None:
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        data_items = plot_item.listDataItems()
        if len(data_items) == 0:
            return
        items_matches = (x for x in data_items if isinstance(x, MultiLine)
                         and x.get_group_number() in current_group)
        for i in data_items:
            i.setVisible(False)
            await sleep(0)
        for i in items_matches:
            i.setVisible(True)
        plot_item.getViewBox().updateAutoRange()

    # endregion

    # region 'A' button
    @asyncSlot()
    async def all_control_button(self) -> None:
        """loop to set visible True for all plot items"""
        if self.ui.all_control_button.isChecked():
            self.ui.by_one_control_button.setChecked(False)
            self.ui.by_group_control_button.setChecked(False)
            tasks = [
                create_task(
                    self.context.preprocessing.stages.input_data.despike_history_remove_plot()),
                create_task(self.context.preprocessing.stages.bl_data.baseline_remove_plot()),
                create_task(self.update_plot_all()),
            ]
            await wait(tasks)
        else:
            self.ui.all_control_button.setChecked(True)

    async def update_plot_all(self) -> None:
        self.ui.statusBar.showMessage("Updating plot...")
        self.update_all_plot()
        self.ui.statusBar.showMessage("Plot updated", 5000)

    def update_all_plot(self) -> None:
        self.context.preprocessing.set_preproc_title(self.drag_widget.get_current_widget())
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        data_items = plot_item.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.context.preprocessing.one_curve:
            plot_item.removeItem(self.context.preprocessing.one_curve)
        plot_item.getViewBox().updateAutoRange()

    # endregion

    def stacked_widget_changed(self) -> None:
        self.crosshair_btn_clicked()
        self.ui.deconv_plot_widget.getPlotItem().getViewBox().updateAutoRange()

    def select_plots_by_buttons(self) -> None:
        if self.ui.by_one_control_button.isChecked():
            self.by_one_control_button_clicked()
        elif self.ui.by_group_control_button.isChecked():
            self.by_group_control_button()

    # endregion

    # region VERTICAL SCROLL BAR

    def vertical_scroll_bar_value_changed(self, event: int) -> None:
        if self.ui.stackedWidget_mainpages.currentIndex() == 0:
            self.ui.input_table.verticalScrollBar().setValue(event)
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 0
        ):
            self.ui.smoothed_dataset_table_view.verticalScrollBar().setValue(event)
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 1
        ):
            self.ui.baselined_dataset_table_view.verticalScrollBar().setValue(event)
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 2
        ):
            self.ui.deconvoluted_dataset_table_view.verticalScrollBar().setValue(event)
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 3
        ):
            self.ui.ignore_dataset_table_view.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 4:
            self.ui.predict_table_view.verticalScrollBar().setValue(event)

    def vertical_scroll_bar_enter_event(self, _) -> None:
        self.ui.verticalScrollBar.setStyleSheet(
            "#verticalScrollBar {background: {{scrollLineHovered}};}"
        )

    def vertical_scroll_bar_leave_event(self, _) -> None:
        self.ui.verticalScrollBar.setStyleSheet(
            "#verticalScrollBar {background: transparent;}"
        )

    def move_side_scrollbar(self, idx: int) -> None:
        self.ui.verticalScrollBar.setValue(idx)

    def page1_btn_clicked(self) -> None:
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page1)
        self.ui.page1Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1
        )
        self.ui.page2Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2
        )
        self.ui.page3Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3
        )
        self.ui.page4Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4
        )
        self.ui.page5Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5
        )
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_1)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_1)
        self.decide_vertical_scroll_bar_visible()

    def page2_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(False)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page2)
        self.ui.page1Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1
        )
        self.ui.page2Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2
        )
        self.ui.page3Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3
        )
        self.ui.page4Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4
        )
        self.ui.page5Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5
        )
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_2)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_2)

    def page3_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page3)
        self.ui.page1Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1
        )
        self.ui.page2Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2
        )
        self.ui.page3Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3
        )
        self.ui.page4Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4
        )
        self.ui.page5Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5
        )
        self.decide_vertical_scroll_bar_visible()

    def page4_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page4)
        self.ui.page1Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1
        )
        self.ui.page2Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2
        )
        self.ui.page3Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3
        )
        self.ui.page4Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4
        )
        self.ui.page5Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5
        )
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_3)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_3)
        self.decide_vertical_scroll_bar_visible()

    def page5_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page5)
        self.ui.page1Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1
        )
        self.ui.page2Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2
        )
        self.ui.page3Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3
        )
        self.ui.page4Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4
        )
        self.ui.page5Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5
        )
        self.decide_vertical_scroll_bar_visible()

    # endregion

    def decide_vertical_scroll_bar_visible(
            self, _model_index: QModelIndex = None, _start: int = 0, _end: int = 0
    ) -> None:

        tv = None
        if self.ui.stackedWidget_mainpages.currentIndex() == 0:
            tv = self.ui.input_table
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 0
        ):
            tv = self.ui.smoothed_dataset_table_view
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 1
        ):
            tv = self.ui.baselined_dataset_table_view
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 2
        ):
            tv = self.ui.deconvoluted_dataset_table_view
        elif (
                self.ui.stackedWidget_mainpages.currentIndex() == 2
                and self.ui.data_tables_tab_widget.currentIndex() == 3
        ):
            tv = self.ui.ignore_dataset_table_view
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3:
            self.ui.verticalScrollBar.setVisible(False)
            return
        elif self.ui.stackedWidget_mainpages.currentIndex() == 4:
            tv = self.ui.predict_table_view
        if tv is None:
            return
        if isinstance(tv, QTableView):
            row_count = tv.model().rowCount()
            row_height = tv.rowHeight(0)
            if row_count > 0:
                page_step = tv.verticalScrollBar().pageStep()
                self.ui.verticalScrollBar.setMinimum(tv.verticalScrollBar().minimum())
                self.ui.verticalScrollBar.setVisible(
                    page_step <= row_height * row_count
                )
                self.ui.verticalScrollBar.setPageStep(page_step)
                self.ui.verticalScrollBar.setMaximum(tv.verticalScrollBar().maximum())
            else:
                self.ui.verticalScrollBar.setVisible(False)
            self.ui.verticalScrollBar.setValue(tv.verticalScrollBar().value())
        elif isinstance(tv, QScrollArea):
            self.ui.verticalScrollBar.setValue(tv.verticalScrollBar().value())
            self.ui.verticalScrollBar.setMinimum(0)
            self.ui.verticalScrollBar.setVisible(True)
            self.ui.verticalScrollBar.setMaximum(tv.verticalScrollBar().maximum())

    # region EVENTS

    def key_press_event(self, key_event) -> None:
        match key_event.key():
            case (Qt.Key.Key_Control, Qt.Key.Key_Z):
                self.undo()
            case (Qt.Key.Key_Control, Qt.Key.Key_Y):
                self.redo()
            case (Qt.Key.Key_Control, Qt.Key.Key_S):
                self.action_save_project()
            case (Qt.Key.Key_Shift, Qt.Key.Key_S):
                self.action_save_as()
            case Qt.Key.Key_End:
                self.executor_stop()
            case Qt.Key.Key_F1:
                action_help()
            case Qt.Key.Key_F2:
                from src.stages.fitting.functions.guess_raman_lines import (
                    show_distribution,
                )

                if self.ui.stackedWidget_mainpages.currentIndex() != 1:
                    return
                if self.context.decomposition.plotting.intervals_data is None:
                    return
                for i, v in enumerate(self.context.decomposition.plotting.intervals_data.items()):
                    key, item = v
                    show_distribution(
                        item["x0"],
                        self.context.preprocessing.stages.av_data.data,
                        self.context.decomposition.data.all_ranges_clustered_x0_sd[i],
                        self.context.group_table.table_widget.model().groups_colors,
                    )
            case Qt.Key.Key_F9:
                areas = []
                stage = self.drag_widget.get_latest_active_stage()
                assert stage is not None, 'Cant find latest active stage.'
                data = stage.data
                for i in data.values():
                    y = i[:, 1]
                    y = y[y < 0]
                    area = np.trapz(y)
                    areas.append(area)
                mean_area = np.mean(areas)
                print(mean_area)
            case Qt.Key.Key_F11:
                if not self.isFullScreen():
                    self.showFullScreen()
                else:
                    self.showMaximized()

    def undo(self) -> None:
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.context.undo_stack.undo()

    def redo(self) -> None:
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.context.undo_stack.redo()

    # endregion

    # region ACTIONS FILE MENU

    def can_close_project(self) -> bool:
        if not self.context.modified:
            return True
        msg = MessageBox(
            "You have unsaved changes.",
            "Save changes before exit?",
            self,
            {"Yes", "No", "Cancel"},
        )
        if self.project_path:
            msg.setInformativeText(self.project_path)
        result = msg.exec()
        if result == 1:
            self.action_save_project()
            return True
        elif result == 0:
            return False
        elif result == 2:
            return True

    def action_save_project(self) -> None:
        if self.project_path == "" or self.project_path is None:
            fd = QFileDialog(self)
            file_path = fd.getSaveFileName(
                self, "Create Project File", self.latest_file_path, "ZIP (*.zip)"
            )
            if file_path[0] == "":
                return
            self.latest_file_path = str(Path(file_path[0]).parent)
            self.save_with_shelve(file_path[0])
            self.ui.projectLabel.setText(file_path[0])
            self.setWindowTitle(file_path[0])
            self._add_path_to_recent(file_path[0])
            self.project_path = file_path[0]
        else:
            self.save_with_shelve(self.project_path)

    @asyncSlot()
    async def action_save_production_project(self) -> None:
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(
            self, "Create Production Project File", self.latest_file_path, "ZIP (*.zip)"
        )
        if file_path[0] == "":
            return
        self.latest_file_path = str(Path(file_path[0]).parent)
        self.save_with_shelve(file_path[0], True)

    @asyncSlot()
    async def action_save_as(self) -> None:
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(
            self, "Create Project File", self.latest_file_path, "ZIP (*.zip)"
        )
        if file_path[0] == "":
            return
        self.latest_file_path = str(Path(file_path[0]).parent)
        self.save_with_shelve(file_path[0])
        self.project_path = file_path[0]

    def save_with_shelve(self, path: str, production_export: bool = False) -> None:
        self.ui.statusBar.showMessage("Saving file...")
        self.close_progress_bar()
        self.open_progress_bar()
        filename = str(Path(path).parent) + "/" + str(Path(path).stem)
        with shelve_open(filename, "n") as db:
            db["_y_axis_ref_EMSC"] = self.predict_logic.y_axis_ref_EMSC
            db["old_Y"] = self.stat_analysis_logic.old_labels
            db["new_Y"] = self.stat_analysis_logic.new_labels
            if not self.predict_logic.is_production_project:
                self.predict_logic.stat_models = {}
                for key, v in self.stat_analysis_logic.latest_stat_result.items():
                    self.predict_logic.stat_models[key] = v["model"]
                db["stat_models"] = self.predict_logic.stat_models
                if self.context.preprocessing.stages.input_data.data:
                    db["interp_ref_array"] = next(
                        iter(self.context.preprocessing.stages.input_data.data.values()))
            if not production_export:
                db["predict_df"] = self.ui.predict_table_view.model().dataframe()
                db["latest_stat_result"] = self.stat_analysis_logic.latest_stat_result
                db["is_production_project"] = False
            else:
                db["is_production_project"] = True
            if self.predict_logic.is_production_project:
                db["is_production_project"] = True
                db["interp_ref_array"] = self.predict_logic.interp_ref_array
                db["stat_models"] = self.predict_logic.stat_models
        zf = ZipFile(filename + ".zip", "w", ZIP_DEFLATED, compresslevel=9)
        zf.write(filename + ".dat", "data.dat")
        zf.write(filename + ".dir", "data.dir")
        zf.write(filename + ".bak", "data.bak")
        self.ui.statusBar.showMessage("File saved.   " + filename + ".zip", 10000)
        self.ui.projectLabel.setText(filename + ".zip")
        self.setWindowTitle(filename + ".zip")
        # set_modified(self.context, self.ui, False)
        Path(filename + ".dat").unlink()
        Path(filename + ".dir").unlink()
        Path(filename + ".bak").unlink()
        self.close_progress_bar()

    @asyncSlot()
    async def open_project(self, path: str, new: bool = False) -> None:
        file_name = Path(path).name
        self.ui.projectLabel.setText(file_name)
        self.setWindowTitle(file_name)
        if not new:
            self._add_path_to_recent(path)
        self.project_path = path
        self._clear_all_parameters()

    def _add_path_to_recent(self, _path: str) -> None:
        with Path("recent-files.txt").open() as file:
            lines = [line.rstrip() for line in file]
        lines = list(dict.fromkeys(lines))
        if Path(_path).suffix == "":
            _path += ".zip"

        if len(lines) > 0 and _path in lines:
            lines.remove(_path)

        with Path("recent-files.txt").open("a") as f:
            f.truncate(0)
            lines_fin = []
            recent_limit = int(environ["recent_limit"])
            for idx, i in enumerate(reversed(lines)):
                if idx < recent_limit - 1:
                    lines_fin.append(i)
                idx += 1
            for line in reversed(lines_fin):
                f.write(line + "\n")
            f.write(_path + "\n")

        if not self.recent_menu.isEnabled():
            self.recent_menu.setDisabled(False)


    @asyncSlot()
    async def action_save_decomposed_to_csv(self) -> None:
        """
        Action saves decomposed dataset pandas table into .csv format
        Table consists is like

        Returns
        -------
            None
        """
        if self.ui.deconvoluted_dataset_table_view.model().rowCount() == 0:
            msg = MessageBox("Export failed.", "No data to save", self, {"Ok"})
            msg.setInformativeText("Try to decompose spectra before save.")
            msg.exec()
            return
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(
            self,
            "Save decomposed lines data to csv table",
            self.latest_file_path,
            "CSV (*.csv",
        )
        if not file_path[0]:
            return
        self.latest_file_path = str(Path(file_path[0]).parent)
        self.ui.deconvoluted_dataset_table_view.model().dataframe().to_csv(file_path[0])

    @asyncSlot()
    async def action_export_table_excel(self) -> None:
        if not self.context.preprocessing.stages.input_data.data:
            msg = MessageBox("Export failed.", "Table is empty", self, {"Ok"})
            msg.exec()
            return
        fd = QFileDialog(self)
        folder_path = fd.getExistingDirectory(
            self, "Choose folder to save excel file", self.latest_file_path
        )
        if not folder_path:
            return
        self.latest_file_path = folder_path
        self.ui.statusBar.showMessage("Saving file...")
        self.close_progress_bar()
        self.open_progress_bar(max_value=0)
        self.open_progress_dialog("Exporting Excel...", "Cancel", maximum=0)
        await self.loop.run_in_executor(None, self.excel_write, folder_path)

        self.close_progress_bar()
        self.ui.statusBar.showMessage("Excel file saved to " + folder_path)

    def excel_write(self, folder_path) -> None:
        with ExcelWriter(folder_path + "\output.xlsx") as writer:
            self.ui.input_table.model().dataframe().to_excel(
                writer, sheet_name="Spectrum info"
            )
            if self.ui.deconv_lines_table.model().rowCount() > 0:
                self.ui.deconv_lines_table.model().dataframe().to_excel(
                    writer, sheet_name="Fit lines"
                )
            if self.ui.fit_params_table.model().rowCount() > 0:
                self.ui.fit_params_table.model().dataframe().to_excel(
                    writer, sheet_name="Fit initial params"
                )
            if self.ui.smoothed_dataset_table_view.model().rowCount() > 0:
                self.ui.smoothed_dataset_table_view.model().dataframe().to_excel(
                    writer, sheet_name="Smoothed dataset"
                )
            if self.ui.baselined_dataset_table_view.model().rowCount() > 0:
                self.ui.baselined_dataset_table_view.model().dataframe().to_excel(
                    writer, sheet_name="Pure Raman dataset"
                )
            if self.ui.deconvoluted_dataset_table_view.model().rowCount() > 0:
                self.ui.deconvoluted_dataset_table_view.model().dataframe().to_excel(
                    writer, sheet_name="Deconvoluted dataset"
                )
            if self.ui.ignore_dataset_table_view.model().rowCount() > 0:
                self.ui.ignore_dataset_table_view.model().dataframe().to_excel(
                    writer, sheet_name="Ignored features"
                )
            if self.ui.pca_features_table_view.model().rowCount() > 0:
                self.ui.pca_features_table_view.model().dataframe().to_excel(
                    writer, sheet_name="PCA loadings"
                )
            if self.ui.predict_table_view.model().rowCount() > 0:
                self.ui.predict_table_view.model().dataframe().to_excel(
                    writer, sheet_name="Predicted"
                )

    def clear_selected_step(self, step: str) -> None:
        if step in self.stat_analysis_logic.latest_stat_result:
            del self.stat_analysis_logic.latest_stat_result[step]
        match step:
            case "Stat":
                self._initial_all_stat_plots()
                self.ui.current_dep_feature1_comboBox.clear()
                self.ui.current_dep_feature2_comboBox.clear()
                self.stat_analysis_logic.latest_stat_result = {}
                self._initial_pca_features_table()
            case "PCA":
                self._initial_pca_plots()
                self._initial_pca_features_table()
            case "Page5":
                self._initial_predict_dataset_table()

    def _clear_all_parameters(self) -> None:
        self.predict_logic.is_production_project = False
        self.predict_logic.stat_models = {}
        self.predict_logic.interp_ref_array = None
        self.stat_analysis_logic.top_features = {}
        self.stat_analysis_logic.old_labels = None
        self.stat_analysis_logic.new_labels = None
        self.stat_analysis_logic.latest_stat_result = {}
        self.clear_selected_step("Stat")
        self.clear_selected_step("Page5")

        self.ui.crosshairBtn.setChecked(False)
        self.crosshair_btn_clicked()
        collect()
        self.undoStack.clear()
        # set_modified(self.context, self.ui, False)

    @asyncSlot()
    async def load_params(self, path: str) -> None:
        self.unzip_project_file(path)
        self.stat_analysis_logic.update_force_single_plots("LDA")
        self.stat_analysis_logic.update_force_full_plots("LDA")

        # set_modified(self.context, self.ui, False)
        # await self.redraw_stat_plots()

    def unzip_project_file(self, path: str) -> None:
        with ZipFile(path) as archive:
            directory = Path(path).parent
            archive.extractall(directory)
            if Path(str(directory) + "/data.dat").exists():
                file_name = str(directory) + "/data"
                self.unshelve_project_file(file_name)
                Path(str(directory) + "/data.dat").unlink()
                Path(str(directory) + "/data.dir").unlink()
                Path(str(directory) + "/data.bak").unlink()

    def unshelve_project_file(self, file_name: str) -> None:
        with shelve_open(file_name, "r") as db:
            if "stat_models" in db:
                try:
                    self.predict_logic.stat_models = db["stat_models"]
                except AttributeError as err:
                    print(err)
            if "interp_ref_array" in db:
                self.predict_logic.interp_ref_array = db["interp_ref_array"]
            if "predict_df" in db:
                self.ui.predict_table_view.model().set_dataframe(db["predict_df"])
            if "is_production_project" in db:
                self.predict_logic.is_production_project = db["is_production_project"]
            if "_y_axis_ref_EMSC" in db:
                self.predict_logic.y_axis_ref_EMSC = db["_y_axis_ref_EMSC"]
            if "latest_stat_result" in db:
                try:
                    self.stat_analysis_logic.latest_stat_result = db[
                        "latest_stat_result"
                    ]
                except AttributeError as err:
                    print(err)
                except ModuleNotFoundError:
                    pass
            if "old_Y" in db:
                self.stat_analysis_logic.old_labels = db["old_Y"]
            if "new_Y" in db:
                self.stat_analysis_logic.new_labels = db["new_Y"]

            if self.widgets["stateTooltip"].wasCanceled() or environ["CANCEL"] == "1":
                self.ui.statusBar.showMessage("Import canceled by user.")
                self._clear_all_parameters()
                return

    @asyncSlot()
    async def action_import_fit_template(self):
        fd = QFileDialog(self)
        file_path = fd.getOpenFileName(
            self, "Open fit template file", self.latest_file_path, "ZIP (*.zip)"
        )
        if not file_path[0]:
            return
        path = file_path[0]
        self.latest_file_path = str(Path(path).parent)
        self.ui.statusBar.showMessage("Reading data file...")
        self.close_progress_bar()
        self.open_progress_bar()
        self.open_progress_dialog("Opening template...", "Cancel")
        self.time_start = datetime.now()
        with ZipFile(path) as archive:
            directory = Path(path).parent
            archive.extractall(directory)
        if not Path(str(directory) + "/data.dat").exists():
            Path(str(directory) + "/data.dat").unlink()
            Path(str(directory) + "/data.dir").unlink()
            Path(str(directory) + "/data.bak").unlink()
            return
        file_name = str(directory) + "/data"
        with shelve_open(file_name, "r") as db:
            if "DeconvLinesTableDF" in db:
                df = db["DeconvLinesTableDF"]
                self.ui.deconv_lines_table.model().set_dataframe(df)
            if "DeconvLinesTableChecked" in db:
                checked = db["DeconvLinesTableChecked"]
                self.ui.deconv_lines_table.model().set_checked(checked)
            if "IgnoreTableChecked" in db:
                checked = db["IgnoreTableChecked"]
                self.ui.ignore_dataset_table_view.model().set_checked(checked)
            if "DeconvParamsTableDF" in db:
                df = db["DeconvParamsTableDF"]
                self.ui.fit_params_table.model().set_dataframe(df)
            if "intervals_table_df" in db:
                df = db["intervals_table_df"]
                self.ui.fit_borders_TableView.model().set_dataframe(df)
        Path(str(directory) + "/data.dat").unlink()
        Path(str(directory) + "/data.dir").unlink()
        Path(str(directory) + "/data.bak").unlink()
        self.close_progress_bar()
        seconds = round((datetime.now() - self.time_start).total_seconds())
        # set_modified(self.context, self.ui, False)
        self.ui.statusBar.showMessage(
            "Fit template imported for " + str(seconds) + " sec.", 5000
        )
        if (
                self.ui.fit_params_table.model().rowCount() != 0
                and self.ui.deconv_lines_table.model().rowCount() != 0
        ):
            await self.context.decomposition.graph_drawing.draw_all_curves()

    @asyncSlot()
    async def action_export_fit_template(self):
        if (
                self.ui.deconv_lines_table.model().rowCount() == 0
                and self.ui.fit_params_table.model().rowCount() == 0
        ):
            msg = MessageBox("Export failed.", "Fit template is empty", self, {"Ok"})
            msg.exec()
            return
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(
            self, "Save fit template file", self.latest_file_path, "ZIP (*.zip)"
        )
        if not file_path[0]:
            return
        self.latest_file_path = str(Path(file_path[0]).parent)
        self.ui.statusBar.showMessage("Saving file...")
        self.close_progress_bar()
        self.open_progress_bar()
        filename = file_path[0]
        with shelve_open(filename, "n") as db:
            db["DeconvLinesTableDF"] = self.ui.deconv_lines_table.model().dataframe()
            db["DeconvParamsTableDF"] = self.ui.fit_params_table.model().dataframe()
            db["intervals_table_df"] = self.ui.fit_borders_TableView.model().dataframe()
            db["DeconvLinesTableChecked"] = self.ui.deconv_lines_table.model().checked()
            db["IgnoreTableChecked"] = self.ui.ignore_dataset_table_view.model().checked
        zf = ZipFile(filename, "w", ZIP_DEFLATED, compresslevel=9)
        zf.write(filename + ".dat", "data.dat")
        zf.write(filename + ".dir", "data.dir")
        zf.write(filename + ".bak", "data.bak")
        self.ui.statusBar.showMessage("File saved. " + filename, 10000)
        Path(filename + ".dat").unlink()
        Path(filename + ".dir").unlink()
        Path(filename + ".bak").unlink()
        self.close_progress_bar()

    # endregion

    # region Main window functions
    def show_error(self, err) -> None:
        critical(err)
        tb = format_exc()
        error(tb)
        if self.widgets["stateTooltip"]:
            self.widgets["stateTooltip"].setContent("Error! 😵")
            self.widgets["stateTooltip"].close()
        self.close_progress_bar()
        show_error_msg(err, err, str(tb), parent=self)
        self.executor_stop()

    def get_curve_plot_data_item(
            self,
            n_array: np.ndarray,
            group_number: str = 0,
            color: QColor = None,
            name: str = "",
            style: Qt.PenStyle = Qt.PenStyle.SolidLine,
            width: int = 2,
    ) -> PlotDataItem:
        curve = PlotDataItem(skipFiniteCheck=True, name=name)
        curve.setData(x=n_array[:, 0], y=n_array[:, 1], skipFiniteCheck=True)
        if color is None:
            color = self.context.group_table.get_color_by_group_number(group_number)
        curve.setPen(color, width=width, style=style)
        return curve

    def open_progress_dialog(
            self, text: str, buttons: str = "", maximum: int = 0
    ) -> None:
        environ["CANCEL"] = "0"
        if self.widgets["stateTooltip"] is None:
            self.widgets["stateTooltip"] = StateToolTip(
                text, "Please wait patiently", self, maximum
            )
            self.widgets["stateTooltip"].move(
                self.ui.centralwidget.width() // 2 - 120,
                self.ui.centralwidget.height() // 2 - 50,
            )
            self.widgets["stateTooltip"].closedSignal.connect(self.executor_stop)
            self.widgets["stateTooltip"].show()

    def executor_stop(self) -> None:
        if not self.current_executor or self.break_event is None:
            return
        for f in self.current_futures:
            if not f.done():
                f.cancel()
        try:
            self.break_event.set()
        except FileNotFoundError:
            warning("FileNotFoundError self.break_event.set()")
        environ["CANCEL"] = "1"
        self.current_executor.shutdown(cancel_futures=True, wait=False)
        self.close_progress_bar()
        self.ui.statusBar.showMessage("Operation canceled by user ")

    def progress_indicator(self, _=None) -> None:
        current_value = self.progressBar.value() + 1
        if self.progressBar:
            self.progressBar.setValue(current_value)
        if self.widgets["stateTooltip"] is not None:
            self.widgets["stateTooltip"].setValue(current_value)
        if self.widgets["taskbar"].progress():
            self.widgets["taskbar"].progress().setValue(current_value)
            self.widgets["taskbar"].progress().show()

    def open_progress_bar(self, min_value: int = 0, max_value: int = 0) -> None:
        if max_value == 0:
            self.progressBar = IndeterminateProgressBar(self)
        else:
            self.progressBar = ProgressBar(self)
            self.progressBar.setRange(min_value, max_value)
        self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
        self.widgets["taskbar"] = QWinTaskbarButton()
        self.widgets["taskbar"].progress().setRange(min_value, max_value)
        self.widgets["taskbar"].setWindow(self.windowHandle())
        self.widgets["taskbar"].progress().show()

    def cancelled_by_user(self) -> bool:
        """
        Cancel button was pressed by user?

        Returns
        -------
        out: bool
            True if Cancel button pressed
        """
        if self.widgets["stateTooltip"].wasCanceled() or environ["CANCEL"] == "1":
            self.close_progress_bar()
            self.ui.statusBar.showMessage("Cancelled by user.")
            info("Cancelled by user")
            return True
        else:
            return False

    def close_progress_bar(self) -> None:
        if self.progressBar is not None:
            self.statusBar().removeWidget(self.progressBar)
        if self.widgets["taskbar"].progress() is not None:
            self.widgets["taskbar"].progress().hide()
            self.widgets["taskbar"].progress().stop()
        if self.widgets["stateTooltip"] is not None:
            text = (
                "Completed! 😆"
                if not self.widgets["stateTooltip"].wasCanceled()
                else "Canceled! 🥲"
            )
            self.widgets["stateTooltip"].setContent(text)
            self.widgets["stateTooltip"].setState(True)
            self.widgets["stateTooltip"] = None

    def set_timer_memory_update(self) -> None:
        """
        Prints at statusBar how much RAM memory used at this moment
        Returns
        -------
            None
        """
        try:
            string_selected_files = ""
            n_selected = len(self.ui.input_table.selectionModel().selectedIndexes())
            if n_selected > 0:
                string_selected_files = str(n_selected // 8) + " selected of "
            string_n = ""
            n_spectrum = len(self.context.preprocessing.stages.input_data.data)
            if n_spectrum > 0:
                string_n = str(n_spectrum) + " files. "
            string_mem = str(round(get_memory_used())) + " Mb RAM used"
            usage_string = string_selected_files + string_n + string_mem
            self.memory_usage_label.setText(usage_string)
        except KeyboardInterrupt:
            pass

    def set_cpu_load(self) -> None:
        cpu_perc = int(cpu_percent())
        self.ui.cpuLoadBar.setValue(cpu_perc)

    def mousePressEvent(self, event) -> None:
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

    def update_icons(self) -> None:
        if "Light" in environ["theme"]:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down_black.svg")
            )
            self.ui.minimizeAppBtn.setIcon(
                QIcon("material/resources/source/minus_black.svg")
            )
            self.ui.leftsideBtn.setIcon(
                QIcon("material/resources/source/chevron-left_black.svg")
            )
            self.ui.gt_add_Btn.setIcon(
                QIcon("material/resources/source/plus_black.svg")
            )
            self.ui.gt_dlt_Btn.setIcon(
                QIcon("material/resources/source/minus_black.svg")
            )
        else:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down.svg")
            )
            self.ui.minimizeAppBtn.setIcon(QIcon("material/resources/source/minus.svg"))
            self.ui.leftsideBtn.setIcon(
                QIcon("material/resources/source/chevron-left.svg")
            )
            self.ui.gt_add_Btn.setIcon(QIcon("material/resources/source/plus.svg"))
            self.ui.gt_dlt_Btn.setIcon(QIcon("material/resources/source/minus.svg"))

    # endregion

    # region Stat analysis (machine learning) page4
    @asyncSlot()
    async def redraw_stat_plots(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("once")
            self.context.ml.update_stat_report_text(
                self.ui.current_classificator_comboBox.currentText()
            )
            await self.loop.run_in_executor(
                None,
                self.stat_analysis_logic.update_plots,
                self.ui.current_classificator_comboBox.currentText(),
            )
            await self.loop.run_in_executor(
                None, self.stat_analysis_logic.update_pca_plots
            )

    @asyncSlot()
    async def refresh_shap_push_button_clicked(self) -> None:
        """
            Refresh all shap plots for currently selected classificator
        Returns
        -------
            None
        """
        cl_type = self.ui.current_classificator_comboBox.currentText()
        if (
                cl_type not in self.stat_analysis_logic.latest_stat_result
                or "target_names"
                not in self.stat_analysis_logic.latest_stat_result[cl_type]
                or "shap_values" not in self.stat_analysis_logic.latest_stat_result[cl_type]
        ):
            msg = MessageBox(
                "SHAP plots refresh error.", "Selected classificator is not fitted.",
                self, {"Ok"},
            )
            msg.setInformativeText(
                "Try to turn on Use Shapley option before fit classificator."
            )
            msg.exec()
            return
        await self.loop.run_in_executor(None, self.update_shap_plots)
        await self.loop.run_in_executor(None, self.update_shap_plots_by_instance)
        self.stat_analysis_logic.update_force_single_plots(cl_type)
        self.stat_analysis_logic.update_force_full_plots(cl_type)
        self.reload_force(self.ui.force_single)
        self.reload_force(self.ui.force_full, True)

    @asyncSlot()
    async def current_dep_feature_changed(self, g: str = "") -> None:
        cl_type = self.ui.current_classificator_comboBox.currentText()
        if cl_type not in self.stat_analysis_logic.latest_stat_result:
            return
        model_results = self.stat_analysis_logic.latest_stat_result[cl_type]
        self.stat_analysis_logic.build_partial_dependence_plot(
            model_results["model"], model_results["X"]
        )

    # endregion
    # region Predict page5

    @asyncSlot()
    async def predict(self):
        if self.predict_logic.is_production_project:
            try:
                await self.predict_logic.do_predict_production()
            except Exception as err:
                self.show_error(err)
        else:
            try:
                await self.predict_logic.do_predict()
            except Exception as err:
                self.show_error(err)
    # endregion
