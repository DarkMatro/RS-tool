import warnings
from asyncio import create_task, sleep, wait, get_event_loop
from collections import defaultdict
from datetime import datetime
from gc import get_objects, collect
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
from pandas import DataFrame, MultiIndex, ExcelWriter
from psutil import cpu_percent
from pyqtgraph import (
    setConfigOption,
    PlotDataItem,
    PlotCurveItem,
    SignalProxy,
    InfiniteLine,
    LinearRegionItem,
    mkPen,
    mkBrush,
    FillBetweenItem,
)
from qtpy.QtCore import Qt, QModelIndex, QTimer, QMarginsF, QPointF
from qtpy.QtGui import QFont, QIcon, QCloseEvent, QColor, QPageLayout, QPageSize
from qtpy.QtWidgets import (
    QUndoStack,
    QMenu,
    QMainWindow,
    QAction,
    QHeaderView,
    QAbstractItemView,
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
from src.widgets.curve_properties_window import CurvePropertiesWindow
from src.data.config import get_config
from src.data.default_values import (
    default_values,
    baseline_parameter_defaults,
    peak_shape_names,
    classificators_names,
)
from src.files.help import action_help
from src.mutual_functions.static_functions import (
    get_memory_used,
    curve_pen_brush_by_style,
    set_roi_size,
)
from src.mw_page2_fitting import FittingLogic
from src.mw_page3_datasets import DatasetsManager
from src.mw_page4_stat_analysis import StatAnalysisLogic
from src.mw_page5_predict import PredictLogic
from src.pandas_tables import (
    InputTable,
    PandasModelDeconvTable,
    PandasModelDeconvLinesTable,
    ComboDelegate,
    PandasModelFitParamsTable,
    DoubleSpinBoxDelegate,
    PandasModelFitIntervals,
    IntervalsTableDelegate,
    PandasModelSmoothedDataset,
    PandasModelBaselinedDataset,
    PandasModelDeconvolutedDataset,
    PandasModel,
    PandasModelPredictTable,
    PandasModelPCA,
    PandasModelIgnoreDataset,
    PandasModelDescribeDataset,
)
from src.stages.stat_analysis.functions.fit_classificators import scorer_metrics
from src.ui.ui_main_window import Ui_MainWindow
from src.undo_redo import (
    CommandChangeGroupCell,
    CommandDeleteInputSpectrum,
    CommandAddDeconvLine,
    CommandDeleteDeconvLines,
    CommandUpdateDeconvCurveStyle,
    CommandUpdateDataCurveStyle,
    CommandClearAllDeconvLines,
    CommandFitIntervalAdded,
    CommandFitIntervalDeleted,
    CommandDeleteDatasetRow,
)
from src.widgets.setting_window import SettingWindow
from ..backend.context import Context
from ..backend.project import Project
from ..data.plotting import random_rgb
from src.backend.progress import Progress
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

    def __init__(self, event_loop: QEventLoop, ) -> None:
        super().__init__(None)
        self.loop = event_loop or get_event_loop()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
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
        self.previous_group_of_item = None
        self.plot_text_color_value = None
        self.plot_text_color = None
        self.plot_background_color = None
        self.plot_background_color_web = None
        self.current_executor = None
        self.time_start = None
        self.task_mem_update = None
        self.current_spectrum_despiked_name = None
        self.current_spectrum_baseline_name = None
        self.export_folder_path = None
        self.CommandStartIntervalChanged_allowed = True
        self.CommandEndIntervalChanged_allowed = True
        self.dragPos = None
        self.project_path = None
        self.window_maximized = True
        self._ascending_input_table = False
        self._ascending_ignore_table = False
        self._ascending_deconv_lines_table = False
        self.latest_file_path = (
                getenv("APPDATA") + "/RS-tool"
        )  # save folder path for file dialogs

        # parameters to turn on/off pushing UndoStack command during another redo/undo command executing
        self.setWindowFlags(
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowFrameSection.TopSection
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.keyPressEvent = self.key_press_event
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.widgets["taskbar"] = QWinTaskbarButton()
        self.widgets["taskbar"].setWindow(self.windowHandle())
        self.fitting = FittingLogic(self)
        self.datasets_manager = DatasetsManager(self)
        self.stat_analysis_logic = StatAnalysisLogic(self)
        self.predict_logic = PredictLogic(self)
        self._init_default_values()

        # UNDO/ REDO
        self.undoStack = QUndoStack(self)
        self.undoStack.setUndoLimit(int(environ["undo_limit"]))

        # SET UI DEFINITIONS
        cfg = get_config()
        self.setWindowIcon(QIcon(cfg["logo"]["path"]))
        self.setWindowTitle("Raman Spectroscopy Tool ")
        self.plot_text_color_value = environ["plotText"]
        self.plot_text_color = QColor(self.plot_text_color_value)
        self.plot_background_color = QColor(environ["plotBackground"])
        self.plot_background_color_web = QColor(environ["backgroundMainColor"])
        self.update_icons()
        self.initial_ui_definitions()

        self._initial_menu()
        self._init_left_menu()
        self._init_push_buttons()
        self._init_spin_boxes()
        self._init_combo_boxes()
        self._initial_all_tables()
        self.initial_right_scrollbar()
        self._initial_plots()
        self.initial_plot_buttons()
        self._initial_guess_table_frame()
        # self.initial_timers()
        self._set_parameters_to_default()
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
                    # We didn't drag past this widget.
                    # insert to the left of it.
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

    def _init_default_values(self) -> None:
        """
        Initialize dict default values from src.default_values
        @return: None
        """
        self.default_values = default_values()
        self.old_start_interval_value = self.default_values["interval_start"]
        self.old_end_interval_value = self.default_values["interval_end"]
        self.fitting.data_style = {
            "color": QColor(environ["secondaryColor"]),
            "style": Qt.PenStyle.SolidLine,
            "width": 1.0,
            "fill": False,
            "use_line_color": True,
            "fill_color": QColor().fromRgb(random_rgb()),
            "fill_opacity": 0.0,
        }
        self.data_style_button_style_sheet(self.fitting.data_style["color"].name())
        self.fitting.sum_style = {
            "color": QColor(environ["primaryColor"]),
            "style": Qt.PenStyle.DashLine,
            "width": 1.0,
            "fill": False,
            "use_line_color": True,
            "fill_color": QColor().fromRgb(random_rgb()),
            "fill_opacity": 0.0,
        }
        self.sum_style_button_style_sheet(self.fitting.sum_style["color"].name())
        self.fitting.sigma3_style = {
            "color": QColor(environ["primaryDarkColor"]),
            "style": Qt.PenStyle.SolidLine,
            "width": 1.0,
            "fill": True,
            "use_line_color": True,
            "fill_color": QColor().fromRgb(random_rgb()),
            "fill_opacity": 0.25,
        }
        self.sigma3_style_button_style_sheet(self.fitting.sigma3_style["color"].name())
        self.fitting.residual_style = {
            "color": QColor(environ["secondaryLightColor"]),
            "style": Qt.PenStyle.DotLine,
            "width": 1.0,
            "fill": False,
            "use_line_color": True,
            "fill_color": QColor().fromRgb(random_rgb()),
            "fill_opacity": 0.0,
        }
        self.residual_style_button_style_sheet(
            self.fitting.residual_style["color"].name()
        )

        self.baseline_parameter_defaults = baseline_parameter_defaults()

    def _set_parameters_to_default(self) -> None:
        self.ui.select_percentile_spin_box.setValue(
            self.default_values["select_percentile_spin_box"]
        )
        self.ui.guess_method_cb.setCurrentText(self.default_values["guess_method_cb"])
        self.ui.dataset_type_cb.setCurrentText(self.default_values["dataset_type_cb"])
        self.ui.classes_lineEdit.setText("")
        self.ui.test_data_ratio_spinBox.setValue(
            self.default_values["test_data_ratio_spinBox"]
        )
        self.ui.random_state_sb.setValue(self.default_values["random_state_sb"])
        self.ui.max_noise_level_dsb.setValue(self.default_values["max_noise_level"])
        self.ui.fit_opt_method_comboBox.setCurrentText(
            self.default_values["fit_method"]
        )
        self.ui.intervals_gb.setChecked(True)
        self.ui.use_pca_checkBox.setChecked(False)

        self.ui.interval_start_dsb.setValue(self.default_values["interval_start"])
        self.ui.interval_end_dsb.setMaximum(99_999.0)
        self.ui.interval_end_dsb.setValue(self.default_values["interval_end"])
        self.ui.max_dx_dsb.setValue(self.default_values["max_dx_guess"])
        self.ui.mlp_layer_size_spinBox.setValue(
            self.default_values["mlp_layer_size_spinBox"]
        )
        self.ui.rf_min_samples_split_spinBox.setValue(
            self.default_values["rf_min_samples_split_spinBox"]
        )
        self.ui.ab_n_estimators_spinBox.setValue(
            self.default_values["ab_n_estimators_spinBox"]
        )
        self.ui.ab_learning_rate_doubleSpinBox.setValue(
            self.default_values["ab_learning_rate_doubleSpinBox"]
        )
        self.ui.xgb_eta_doubleSpinBox.setValue(
            self.default_values["xgb_eta_doubleSpinBox"]
        )
        self.ui.xgb_gamma_spinBox.setValue(self.default_values["xgb_gamma_spinBox"])
        self.ui.xgb_max_depth_spinBox.setValue(
            self.default_values["xgb_max_depth_spinBox"]
        )
        self.ui.xgb_min_child_weight_spinBox.setValue(
            self.default_values["xgb_min_child_weight_spinBox"]
        )
        self.ui.xgb_colsample_bytree_doubleSpinBox.setValue(
            self.default_values["xgb_colsample_bytree_doubleSpinBox"]
        )
        self.ui.xgb_lambda_doubleSpinBox.setValue(
            self.default_values["xgb_lambda_doubleSpinBox"]
        )
        self.ui.xgb_n_estimators_spinBox.setValue(
            self.default_values["xgb_n_estimators_spinBox"]
        )
        self.ui.rf_n_estimators_spinBox.setValue(
            self.default_values["rf_n_estimators_spinBox"]
        )
        self.ui.max_epoch_spinBox.setValue(self.default_values["max_epoch_spinBox"])
        self.ui.dt_min_samples_split_spin_box.setValue(
            self.default_values["dt_min_samples_split_spin_box"]
        )
        self.ui.dt_max_depth_spin_box.setValue(
            self.default_values["dt_max_depth_spin_box"]
        )
        self.ui.learning_rate_doubleSpinBox.setValue(
            self.default_values["learning_rate_doubleSpinBox"]
        )
        self.ui.feature_display_max_spinBox.setValue(
            self.default_values["feature_display_max_spinBox"]
        )
        self.ui.l_ratio_doubleSpinBox.setValue(self.default_values["l_ratio"])
        self.ui.comboBox_lda_solver.setCurrentText(
            self.default_values["comboBox_lda_solver"]
        )
        self.ui.lda_solver_check_box.setChecked(True)
        self.ui.lda_shrinkage_check_box.setChecked(True)
        self.ui.lr_c_checkBox.setChecked(True)
        self.ui.lr_solver_checkBox.setChecked(True)
        self.ui.svc_nu_check_box.setChecked(True)
        self.ui.n_neighbors_checkBox.setChecked(True)
        self.ui.nn_weights_checkBox.setChecked(True)
        self.ui.lr_penalty_checkBox.setChecked(True)
        self.ui.learning_rate_checkBox.setChecked(True)
        self.ui.mlp_layer_size_checkBox.setChecked(True)
        self.ui.mlp_solve_checkBox.setChecked(True)
        self.ui.activation_checkBox.setChecked(True)
        self.ui.criterion_checkBox.setChecked(True)
        self.ui.dt_min_samples_split_check_box.setChecked(True)
        self.ui.dt_max_depth_check_box.setChecked(True)
        self.ui.rf_max_features_checkBox.setChecked(True)
        self.ui.rf_n_estimators_checkBox.setChecked(True)
        self.ui.rf_min_samples_split_checkBox.setChecked(True)
        self.ui.rf_criterion_checkBox.setChecked(True)
        self.ui.ab_learning_rate_checkBox.setChecked(True)
        self.ui.ab_n_estimators_checkBox.setChecked(True)
        self.ui.xgb_eta_checkBox.setChecked(True)
        self.ui.xgb_gamma_checkBox.setChecked(True)
        self.ui.xgb_max_depth_checkBox.setChecked(True)
        self.ui.xgb_min_child_weight_checkBox.setChecked(True)
        self.ui.xgb_colsample_bytree_checkBox.setChecked(True)
        self.ui.xgb_lambda_checkBox.setChecked(True)
        self.ui.lineEdit_lda_shrinkage.setText(
            self.default_values["lineEdit_lda_shrinkage"]
        )
        self.ui.lr_c_doubleSpinBox.setValue(self.default_values["lr_c_doubleSpinBox"])
        self.ui.svc_nu_doubleSpinBox.setValue(
            self.default_values["svc_nu_doubleSpinBox"]
        )
        self.ui.n_neighbors_spinBox.setValue(self.default_values["n_neighbors_spinBox"])
        self.ui.lr_solver_comboBox.setCurrentText(
            self.default_values["lr_solver_comboBox"]
        )
        self.ui.nn_weights_comboBox.setCurrentText(
            self.default_values["nn_weights_comboBox"]
        )
        self.ui.lr_penalty_comboBox.setCurrentText(
            self.default_values["lr_penalty_comboBox"]
        )

    # region plots
    def _initial_preproc_plot(self) -> None:
        cfg = get_config('plots')['preproc']
        self.ui.crosshair_update_preproc = SignalProxy(
            self.ui.preproc_plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.update_crosshair,
        )
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        self.ui.preproc_plot_widget.setAntialiasing(1)
        plot_item.enableAutoRange()
        plot_item.showGrid(True, True, cfg['alpha'])
        self.ui.preproc_plot_widget.vertical_line = InfiniteLine()
        self.ui.preproc_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.preproc_plot_widget.vertical_line.setPen(
            QColor(environ["secondaryColor"])
        )
        self.ui.preproc_plot_widget.horizontal_line.setPen(
            QColor(environ["secondaryColor"])
        )
        current_stage = self.context.preprocessing.active_stage.name
        title = ('<span style="font-family: AbletonSans; color:' + environ["plotText"]
                 + ';font-size: ' + str(environ['plot_font_size'])
                 + '">' + cfg[current_stage] + '</span>')
        self.ui.preproc_plot_widget.setTitle(title)
        items_matches = (
            x
            for x in plot_item.listDataItems()
            if not x.name()
        )
        for i in items_matches:
            plot_item.removeItem(i)
        self._initial_plot_color()

    def _initial_plot_color(self) -> None:
        self.ui.preproc_plot_widget.setBackground(self.plot_background_color)
        self.ui.preproc_plot_widget.getPlotItem().getAxis("bottom").setPen(
            self.plot_text_color
        )
        self.ui.preproc_plot_widget.getPlotItem().getAxis("left").setPen(
            self.plot_text_color
        )
        self.ui.preproc_plot_widget.getPlotItem().getAxis("bottom").setTextPen(
            self.plot_text_color
        )
        self.ui.preproc_plot_widget.getPlotItem().getAxis("left").setTextPen(
            self.plot_text_color
        )

    def _initial_preproc_plot_color(self) -> None:
        self.ui.preproc_plot_widget.setBackground(self.plot_background_color)
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item.getAxis("bottom").setPen(self.plot_text_color)
        plot_item.getAxis("left").setPen(self.plot_text_color)
        plot_item.getAxis("bottom").setTextPen(self.plot_text_color)
        plot_item.getAxis("left").setTextPen(self.plot_text_color)

    def _initial_deconvolution_plot(self) -> None:
        self.deconvolution_plotItem = self.ui.deconv_plot_widget.getPlotItem()
        self.ui.deconv_plot_widget.scene().sigMouseClicked.connect(
            self.fit_plot_mouse_clicked
        )
        self.crosshair_update_deconv = SignalProxy(
            self.ui.deconv_plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.update_crosshair_deconv_plot,
        )
        self.ui.deconv_plot_widget.setAntialiasing(1)
        self.deconvolution_plotItem.enableAutoRange()
        self.deconvolution_plotItem.showGrid(True, True, 0.5)
        self.ui.deconv_plot_widget.vertical_line = InfiniteLine()
        self.ui.deconv_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.deconv_plot_widget.vertical_line.setPen(
            QColor(environ["secondaryColor"])
        )
        self.ui.deconv_plot_widget.horizontal_line.setPen(
            QColor(environ["secondaryColor"])
        )
        items_matches = (
            x for x in self.deconvolution_plotItem.listDataItems() if not x.name()
        )
        for i in items_matches:
            self.deconvolution_plotItem.removeItem(i)
        self._initial_deconv_plot_color()
        self.linearRegionDeconv = LinearRegionItem(
            [self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value()],
            bounds=[
                self.ui.interval_start_dsb.minimum(),
                self.ui.interval_end_dsb.maximum(),
            ],
            swapMode="push",
        )
        self.deconvolution_plotItem.addItem(self.linearRegionDeconv)
        color_for_lr = QColor(environ["secondaryDarkColor"])
        color_for_lr.setAlpha(20)
        color_for_lr_hover = QColor(environ["secondaryDarkColor"])
        color_for_lr_hover.setAlpha(40)
        self.linearRegionDeconv.setBrush(color_for_lr)
        self.linearRegionDeconv.setHoverBrush(color_for_lr_hover)
        self.linearRegionDeconv.sigRegionChangeFinished.connect(
            self.lr_deconv_region_changed
        )
        self.linearRegionDeconv.setMovable(not self.ui.lr_movableBtn.isChecked())
        self.fitting.sigma3_top = PlotCurveItem(name="sigma3_top")
        self.fitting.sigma3_bottom = PlotCurveItem(name="sigma3_bottom")
        color = self.fitting.sigma3_style["color"]
        color.setAlphaF(0.25)
        pen = mkPen(color=color, style=Qt.PenStyle.SolidLine)
        brush = mkBrush(color)
        self.fitting.sigma3_fill = FillBetweenItem(
            self.fitting.sigma3_top, self.fitting.sigma3_bottom, brush, pen
        )
        self.deconvolution_plotItem.addItem(self.fitting.sigma3_fill)
        self.fitting.sigma3_fill.setVisible(False)

    def fit_plot_mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.fitting.deselect_selected_line()

    def _initial_deconv_plot_color(self) -> None:
        self.ui.deconv_plot_widget.setBackground(self.plot_background_color)
        self.deconvolution_plotItem.getAxis("bottom").setPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis("left").setPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis("bottom").setTextPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis("left").setTextPen(self.plot_text_color)

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
    def initial_stat_plot(plot_widget) -> None:
        plot_widget.canvas.axes.cla()
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
            self.ui.plsda_scores_plot_widget,
            self.ui.plsda_vip_plot_widget,
            self.ui.roc_comparsion_plot_widget,
            self.ui.shap_beeswarm,
            self.ui.shap_means,
            self.ui.shap_heatmap,
            self.ui.shap_scatter,
            self.ui.shap_decision,
            self.ui.shap_waterfall,
            self.ui.roc_comparsion_plot_widget,
        ]
        for pl in plot_widgets:
            self.set_canvas_colors(pl.canvas)
        if self.ui.current_group_shap_comboBox.currentText() == "":
            return
        cl_type = self.ui.current_classificator_comboBox.currentText()
        await self.loop.run_in_executor(None, self._update_shap_plots)
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        self.stat_analysis_logic.update_force_single_plots(cl_type)
        self.stat_analysis_logic.update_force_full_plots(cl_type)

    def _update_shap_plots(self) -> None:
        cl_type = self.ui.current_classificator_comboBox.currentText()
        self.stat_analysis_logic.do_update_shap_plots(cl_type)

    def _update_shap_plots_by_instance(self) -> None:
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
        self._initial_preproc_plot()
        self._initial_deconvolution_plot()
        self._initial_all_stat_plots()

    def _initial_pca_plots(self) -> None:
        self.initial_scores_plot(self.ui.pca_scores_plot_widget)
        self.initial_scores_plot(self.ui.pca_loadings_plot_widget)

    def _initial_plsda_plots(self) -> None:
        self.initial_scores_plot(self.ui.plsda_scores_plot_widget, "PLS-DA")
        self.initial_scores_plot(self.ui.plsda_vip_plot_widget, "PLS-DA")

    def _initial_all_stat_plots(self) -> None:
        self.initial_stat_plot(self.ui.decision_score_plot_widget)
        self.initial_stat_plot(self.ui.decision_boundary_plot_widget)
        self.initial_stat_plot(self.ui.violin_describe_plot_widget)
        self.initial_stat_plot(self.ui.boxplot_describe_plot_widget)
        self.initial_stat_plot(self.ui.dm_plot_widget)
        self.initial_stat_plot(self.ui.roc_plot_widget)
        self.initial_stat_plot(self.ui.pr_plot_widget)
        self.initial_stat_plot(self.ui.perm_imp_train_plot_widget)
        self.initial_stat_plot(self.ui.perm_imp_test_plot_widget)
        self.initial_stat_plot(self.ui.partial_depend_plot_widget)
        self.initial_stat_plot(self.ui.tree_plot_widget)
        self.initial_stat_plot(self.ui.features_plot_widget)
        self.initial_stat_plot(self.ui.calibration_plot_widget)
        self.initial_stat_plot(self.ui.det_curve_plot_widget)
        self.initial_stat_plot(self.ui.learning_plot_widget)
        self.initial_stat_plot(self.ui.roc_comparsion_plot_widget)
        self._initial_pca_plots()
        self._initial_plsda_plots()
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
        self.deconvolution_plotItem.getAxis("bottom").setStyle(tickFont=plot_font)
        self.deconvolution_plotItem.getAxis("left").setStyle(tickFont=plot_font)

        plt.rcParams.update({"font.size": int(environ["plot_font_size"])})

    def get_plot_label_style(self) -> dict:
        return {
            "color": self.plot_text_color_value,
            "font-size": str(environ["axis_label_font_size"]) + "pt",
            "font-family": "AbletonSans",
        }
    def initial_plots_labels(self) -> None:
        label_style = self.get_plot_label_style()

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
        clear_menu.addAction("Fitting lines", self.clear_all_deconv_lines)
        clear_menu.addAction(
            "All fitting data", lambda: self.clear_selected_step("Deconvolution")
        )
        clear_menu.addSeparator()
        clear_menu.addAction("Smoothed dataset", self._initial_smoothed_dataset_table)
        clear_menu.addAction(
            "Baseline corrected dataset", self._initial_baselined_dataset_table
        )
        clear_menu.addAction(
            "Decomposed dataset", self._initial_deconvoluted_dataset_table
        )
        clear_menu.addAction("Ignore features", self._initial_ignore_dataset_table)
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
        print('_init_stat_analysis_menu')
        stat_analysis_menu = QMenu(self)
        action_fit = QAction("Fit")
        action_fit.triggered.connect(self.fit_classificator)
        action_fit_pca = QAction("PCA")
        action_fit_pca.triggered.connect(lambda: self.fit_classificator("PCA"))
        action_fit_plsda = QAction("PLS-DA")
        action_fit_plsda.triggered.connect(lambda: self.fit_classificator("PLS-DA"))
        action_refresh_plots = QAction("Refresh fit result plots")
        action_refresh_plots.triggered.connect(self.redraw_stat_plots)
        action_refresh_shap = QAction("Refresh SHAP")
        action_refresh_shap.triggered.connect(self.refresh_shap_push_button_clicked)
        action_refresh_learning_curve = QAction("Refresh learning curve")
        action_refresh_learning_curve.triggered.connect(
            self.stat_analysis_logic.refresh_learning_curve
        )
        actions = [
            action_fit,
            action_fit_pca,
            action_fit_plsda,
            action_refresh_plots,
            action_refresh_shap,
            action_refresh_learning_curve,
        ]
        stat_analysis_menu.addActions(actions)
        self.ui.stat_analysis_btn.setMenu(stat_analysis_menu)

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
        self._init_current_classificator_combo_box()
        self._init_fit_opt_method_combo_box()
        self._init_guess_method_cb()
        self._init_params_mouse_double_click_event()
        self._init_dataset_type_cb()
        self._init_refit_score()
        self._init_solver_mlp_combo_box()
        self._init_activation_combo_box()
        self._init_criterion_combo_box()
        self._init_rf_max_features_combo_box()
        self._init_current_feature_cb()
        self._init_coloring_feature_cb()
        self._init_current_tree_sb()
        self._init_use_pca_cb()
        self._init_include_x0_chb()
        self._init_combo_box_lda_solver()
        self._init_lr_penalty_combo_box()
        self._init_lr_solver_combo_box()
        self._init_weights_combo_box()
        self.ui.use_grid_search_checkBox.stateChanged.connect(
            self.use_grid_search_check_box_change_event
        )
        self.ui.edit_template_btn.clicked.connect(
            lambda: self.fitting.switch_to_template()
        )
        self.ui.template_combo_box.currentTextChanged.connect(
            lambda: self.fitting.switch_to_template()
        )
        self.ui.current_group_shap_comboBox.currentTextChanged.connect(
            self.current_group_shap_changed
        )
        self.ui.intervals_gb.toggled.connect(self.intervals_gb_toggled)

    def _init_push_buttons(self) -> None:
        self.ui.select_percentile_push_button.clicked.connect(
            self.stat_analysis_logic.feature_select_percentile
        )
        self.ui.check_all_push_button.clicked.connect(
            lambda: self.ui.ignore_dataset_table_view.model().set_checked({})
        )
        self.ui.update_describe_push_button.clicked.connect(
            self.datasets_manager.update_describe_tables
        )
        self.ui.violin_box_plots_update_push_button.clicked.connect(
            self.datasets_manager.update_violin_boxplot
        )
        self.ui.page5_predict.clicked.connect(self.predict)

    def _init_spin_boxes(self) -> None:
        self.ui.describe_1_SpinBox.valueChanged.connect(
            lambda: self.ui.describe_1_SpinBox.setMaximum(
                self.context.group_table.rowCount
            )
        )
        self.ui.describe_2_SpinBox.valueChanged.connect(
            lambda: self.ui.describe_2_SpinBox.setMaximum(
                self.context.group_table.rowCount
            )
        )

    def _init_combo_boxes(self) -> None:
        self._init_current_filename_combobox()

    def _init_current_filename_combobox(self) -> None:
        if not self.ui.deconvoluted_dataset_table_view.model():
            return
        q_res = self.ui.deconvoluted_dataset_table_view.model().dataframe()
        self.ui.current_filename_combobox.addItem(None)
        self.ui.current_filename_combobox.addItems(q_res["Filename"])

    def _init_current_classificator_combo_box(self) -> None:
        self.ui.current_classificator_comboBox.addItems(classificators_names())
        self.ui.current_classificator_comboBox.currentTextChanged.connect(
            self.stat_analysis_logic.update_stat_report_text
        )


    def _init_fit_opt_method_combo_box(self) -> None:
        self.ui.fit_opt_method_comboBox.addItems(self.fitting.fitting_methods)

    def _init_params_mouse_double_click_event(self) -> None:
        self.ui.interval_start_dsb.mouseDoubleClickEvent = (
            self._interval_start_mouse_dce
        )
        self.ui.interval_end_dsb.mouseDoubleClickEvent = self._interval_end_mouse_dce
        self.ui.max_epoch_spinBox.mouseDoubleClickEvent = (
            self._max_epoch_spin_box_mouse_dce
        )
        self.ui.learning_rate_doubleSpinBox.mouseDoubleClickEvent = (
            self._learning_rate_double_spin_box_mouse_dce
        )

    def _init_params_value_changed(self) -> None:
        self.ui.leftsideBtn.clicked.connect(self.leftside_btn_clicked)
        self.ui.dec_list_btn.clicked.connect(self.dec_list_btn_clicked)
        self.ui.stat_param_btn.clicked.connect(self.stat_param_btn_clicked)
        # self.ui.interval_start_dsb.valueChanged.connect(set_modified)
        # self.ui.interval_end_dsb.valueChanged.connect(set_modified)
        # self.ui.select_percentile_spin_box.valueChanged.connect(set_modified)

        # self.ui.max_noise_level_dsb.valueChanged.connect(set_modified)
        # self.ui.l_ratio_doubleSpinBox.valueChanged.connect(set_modified)
        # self.ui.dataset_type_cb.currentTextChanged.connect(set_modified)
        # self.ui.classes_lineEdit.textChanged.connect(set_modified)
        # self.ui.test_data_ratio_spinBox.valueChanged.connect(set_modified)
        # self.ui.updateTrimRangebtn.clicked.connect(self.update_trim_range_btn_clicked)
        # self.ui.update_partial_dep_pushButton.clicked.connect(self.current_dep_feature_changed)
        # self.ui.guess_method_cb.currentTextChanged.connect(self.guess_method_cb_changed)
        # self.ui.comboBox_lda_solver.currentTextChanged.connect(set_modified)
        # self.ui.lr_penalty_comboBox.currentTextChanged.connect(set_modified)
        # self.ui.lr_solver_comboBox.currentTextChanged.connect(set_modified)
        # self.ui.nn_weights_comboBox.currentTextChanged.connect(set_modified)
        # self.ui.lr_c_doubleSpinBox.valueChanged.connect(set_modified)
        # self.ui.svc_nu_doubleSpinBox.valueChanged.connect(set_modified)
        # self.ui.n_neighbors_spinBox.valueChanged.connect(set_modified)
        # self.ui.lineEdit_lda_shrinkage.textChanged.connect(set_modified)
        # self.ui.lr_penalty_checkBox.stateChanged.connect(set_modified)
        # self.ui.lr_solver_checkBox.stateChanged.connect(set_modified)
        # self.ui.svc_nu_check_box.stateChanged.connect(set_modified)
        # self.ui.nn_weights_checkBox.stateChanged.connect(set_modified)
        # self.ui.n_neighbors_checkBox.stateChanged.connect(set_modified)
        # self.ui.activation_checkBox.stateChanged.connect(set_modified)
        # self.ui.criterion_checkBox.stateChanged.connect(set_modified)
        # self.ui.dt_max_depth_check_box.stateChanged.connect(set_modified)
        # self.ui.dt_min_samples_split_check_box.stateChanged.connect(set_modified)
        # self.ui.rf_criterion_checkBox.stateChanged.connect(set_modified)
        # self.ui.ab_n_estimators_checkBox.stateChanged.connect(set_modified)
        # self.ui.xgb_lambda_checkBox.stateChanged.connect(set_modified)
        # self.ui.xgb_colsample_bytree_checkBox.stateChanged.connect(set_modified)
        # self.ui.xgb_min_child_weight_checkBox.stateChanged.connect(set_modified)
        # self.ui.xgb_max_depth_checkBox.stateChanged.connect(set_modified)
        # self.ui.xgb_gamma_checkBox.stateChanged.connect(set_modified)
        # self.ui.xgb_eta_checkBox.stateChanged.connect(set_modified)
        # self.ui.ab_learning_rate_checkBox.stateChanged.connect(set_modified)
        # self.ui.rf_min_samples_split_checkBox.stateChanged.connect(set_modified)
        # self.ui.rf_n_estimators_checkBox.stateChanged.connect(set_modified)
        # self.ui.rf_max_features_checkBox.stateChanged.connect(set_modified)
        # self.ui.mlp_solve_checkBox.stateChanged.connect(set_modified)
        # self.ui.mlp_layer_size_checkBox.stateChanged.connect(set_modified)
        # self.ui.learning_rate_checkBox.stateChanged.connect(set_modified)
        # self.ui.lr_c_checkBox.stateChanged.connect(set_modified)
        # self.ui.compare_roc_push_button.clicked.connect(self.stat_analysis_logic.compare_models_roc)


    def _init_guess_method_cb(self) -> None:
        self.ui.guess_method_cb.addItems(["Average", "Average groups", "All"])

    def _init_refit_score(self) -> None:
        self.ui.refit_score.addItems(scorer_metrics().keys())

    def _init_dataset_type_cb(self) -> None:
        self.ui.dataset_type_cb.addItems(
            ["Smoothed", "Baseline corrected", "Decomposed"]
        )
        self.ui.dataset_type_cb.currentTextChanged.connect(
            self.dataset_type_cb_current_text_changed
        )

    def _init_current_feature_cb(self) -> None:
        self.ui.current_feature_comboBox.currentTextChanged.connect(
            self.update_shap_scatters
        )

    def _init_coloring_feature_cb(self) -> None:
        self.ui.coloring_feature_comboBox.currentTextChanged.connect(
            self.update_shap_scatters
        )

    def dataset_type_cb_current_text_changed(self, ct: str) -> None:
        if ct == "Smoothed":
            model = self.ui.smoothed_dataset_table_view.model()
        elif ct == "Baseline corrected":
            model = self.ui.baselined_dataset_table_view.model()
        elif ct == "Decomposed":
            model = self.ui.deconvoluted_dataset_table_view.model()
        else:
            return
        if model.rowCount() == 0 or self.predict_logic.is_production_project:
            self.ui.dataset_features_n.setText("")
            return
        q_res = model.dataframe()
        features_names = list(q_res.columns[2:])
        n_features = (
            self.ui.ignore_dataset_table_view.model().n_features
            if ct == "Decomposed"
            else len(features_names)
        )
        self.ui.dataset_features_n.setText("%s features" % n_features)
        self.ui.current_feature_comboBox.clear()
        self.ui.current_dep_feature1_comboBox.clear()
        self.ui.current_dep_feature2_comboBox.clear()
        self.ui.coloring_feature_comboBox.clear()
        self.ui.coloring_feature_comboBox.addItem("")
        self.ui.current_feature_comboBox.addItems(features_names)
        self.ui.current_dep_feature1_comboBox.addItems(features_names)
        self.ui.current_dep_feature2_comboBox.addItems(features_names)
        self.ui.coloring_feature_comboBox.addItems(features_names)
        self.ui.current_dep_feature2_comboBox.setCurrentText(features_names[1])
        try:
            self.ui.current_group_shap_comboBox.currentTextChanged.disconnect(
                self.current_group_shap_changed
            )
        except:
            error(
                "failed to disconnect currentTextChanged self.current_group_shap_comboBox)"
            )
        self.ui.current_group_shap_comboBox.clear()

        uniq_classes = np.unique(q_res["Class"].values)
        classes = []
        groups = self.context.group_table.groups_list()
        for i in uniq_classes:
            if i in groups:
                classes.append(i)
        target_names = self.context.group_table.target_names(classes)
        self.ui.current_group_shap_comboBox.addItems(target_names)

        try:
            self.ui.current_group_shap_comboBox.currentTextChanged.connect(
                self.current_group_shap_changed
            )
        except:
            error(
                "failed to connect currentTextChanged self.current_group_shap_comboBox)"
            )

        try:
            self.ui.current_instance_combo_box.currentTextChanged.disconnect(
                self.current_instance_changed
            )
        except:
            error(
                "failed to disconnect currentTextChanged self.current_instance_combo_box)"
            )
        self.ui.current_instance_combo_box.addItem("")
        self.ui.current_instance_combo_box.addItems(q_res["Filename"])
        try:
            self.ui.current_instance_combo_box.currentTextChanged.connect(
                self.current_instance_changed
            )
        except:
            error("failed to connect currentTextChanged self.current_instance_changed)")

    def intervals_gb_toggled(self, b: bool) -> None:
        self.ui.fit_borders_TableView.setVisible(b)
        if b:
            self.ui.intervals_gb.setMaximumHeight(200)
        else:
            self.ui.intervals_gb.setMaximumHeight(1)

    def guess_method_cb_changed(self, value: str) -> None:
        pass
        # set_modified(self.context, self.ui)

    @asyncSlot()
    async def current_group_shap_changed(self, g: str = "") -> None:
        await self.loop.run_in_executor(None, self._update_shap_plots)
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        cl_type = self.ui.current_classificator_comboBox.currentText()
        self.stat_analysis_logic.update_force_single_plots(cl_type)
        self.stat_analysis_logic.update_force_full_plots(cl_type)

    @asyncSlot()
    async def current_instance_changed(self, _: str = "") -> None:
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        cl_type = self.ui.current_classificator_comboBox.currentText()
        self.stat_analysis_logic.update_force_single_plots(cl_type)

    def _init_activation_combo_box(self) -> None:
        self.ui.activation_comboBox.addItems(["identity", "logistic", "tanh", "relu"])

    def _init_criterion_combo_box(self) -> None:
        items = ["gini", "entropy", "log_loss"]
        self.ui.criterion_comboBox.addItems(items)
        self.ui.rf_criterion_comboBox.addItems(items)

    def _init_rf_max_features_combo_box(self) -> None:
        self.ui.rf_max_features_comboBox.addItems(["sqrt", "log2", "None"])

    def _init_solver_mlp_combo_box(self) -> None:
        self.ui.solver_mlp_combo_box.addItems(["lbfgs", "sgd", "adam"])

    def _init_current_tree_sb(self) -> None:
        self.ui.current_tree_spinBox.valueChanged.connect(self.current_tree_sb_changed)

    def _init_use_pca_cb(self) -> None:
        self.ui.use_pca_checkBox.stateChanged.connect(self.use_pca_cb_changed)

    def use_pca_cb_changed(self, b: bool):
        if b:
            self.ui.use_pca_checkBox.setText("PCA dimensional reduction")
        else:
            self.ui.use_pca_checkBox.setText("PLS-DA dimensional reduction")

    def _init_include_x0_chb(self) -> None:
        self.ui.include_x0_checkBox.stateChanged.connect(self.include_x0_chb_changed)

    def include_x0_chb_changed(self, _: bool) -> None:
        if not self.state["loading_params"]:
            self.set_deconvoluted_dataset()

    def _init_combo_box_lda_solver(self) -> None:
        self.ui.comboBox_lda_solver.addItems(["svd", "eigen", "lsqr"])

    def _init_lr_penalty_combo_box(self) -> None:
        self.ui.lr_penalty_comboBox.addItems(["l2", "l1", "elasticnet", "None"])

    def _init_lr_solver_combo_box(self) -> None:
        self.ui.lr_solver_comboBox.addItems(
            ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        )

    def _init_weights_combo_box(self) -> None:
        self.ui.nn_weights_comboBox.addItems(["uniform", "distance"])

    def use_grid_search_check_box_change_event(self, state: int):
        t = "Use GridSearchCV" if state == 2 else "Use HalvingGridSearchCV"
        self.ui.use_grid_search_checkBox.setText(t)

    # region mouse double clicked


    def _interval_start_mouse_dce(self, event) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.interval_start_dsb.setValue(self.default_values["interval_start"])

    def _interval_end_mouse_dce(self, event) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.interval_end_dsb.setValue(self.default_values["interval_end"])


    # endregion

    # region baseline params mouseDCE



    def _learning_rate_double_spin_box_mouse_dce(self, event) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.learning_rate_doubleSpinBox.setValue(
                self.default_values["learning_rate_doubleSpinBox"]
            )

    def _max_epoch_spin_box_mouse_dce(self, event) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.max_epoch_spinBox.setValue(self.default_values["max_epoch_spinBox"])
    # endregion

    # endregion

    # region Tables

    def _initial_all_tables(self) -> None:
        self._initial_input_table()
        self._initial_dec_table()
        self._initial_deconv_lines_table()
        self._initial_deconv_params_table()
        self._initial_fit_intervals_table()
        self._initial_smoothed_dataset_table()
        self._initial_baselined_dataset_table()
        self._initial_deconvoluted_dataset_table()
        self._initial_ignore_dataset_table()
        self._initial_describe_dataset_tables()
        self._initial_predict_dataset_table()
        self._initial_pca_features_table()
        self._initial_plsda_vip_table()

    # region input_table
    def _initial_input_table(self) -> None:
        self._reset_input_table()
        self.ui.input_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.ui.input_table.horizontalHeader().resizeSection(0, 80)
        self.ui.input_table.horizontalHeader().setMinimumWidth(10)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        self.ui.input_table.horizontalHeader().resizeSection(1, 80)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Interactive
        )
        self.ui.input_table.horizontalHeader().resizeSection(2, 50)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.ui.input_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Interactive
        )
        self.ui.input_table.horizontalHeader().resizeSection(4, 130)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Interactive
        )
        self.ui.input_table.horizontalHeader().resizeSection(5, 90)
        self.ui.input_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        self.ui.input_table.selectionModel().selectionChanged.connect(
            self.input_table_selection_changed
        )
        self.ui.input_table.verticalScrollBar().valueChanged.connect(
            self.move_side_scrollbar
        )
        self.ui.input_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ui.input_table.moveEvent = (
            self.input_table_vertical_scrollbar_value_changed
        )
        self.ui.input_table.model().dataChanged.connect(self.input_table_item_changed)
        self.ui.input_table.rowCountChanged = self.decide_vertical_scroll_bar_visible
        self.ui.input_table.horizontalHeader().sectionClicked.connect(
            self._input_table_header_clicked
        )
        self.ui.input_table.contextMenuEvent = self._input_table_context_menu_event
        self.ui.input_table.keyPressEvent = self._input_table_key_pressed

    def _input_table_key_pressed(self, key_event) -> None:
        if (
                key_event.key() == Qt.Key.Key_Delete
                and self.ui.input_table.selectionModel().currentIndex().row() > -1
                and len(self.ui.input_table.selectionModel().selectedIndexes())
        ):
            self.time_start = datetime.now()
            command = CommandDeleteInputSpectrum(self, "Delete files")
            self.undoStack.push(command)

    def _reset_input_table(self) -> None:
        df = DataFrame(
            columns=[
                "Min, nm",
                "Max, nm",
                "Group",
                "Despiked, nm",
                "Rayleigh line, nm",
                "FWHM, nm",
                "FWHM, cm\N{superscript minus}\N{superscript one}",
                "SNR",
            ]
        )
        model = InputTable(df)
        self.ui.input_table.setSortingEnabled(True)
        self.ui.input_table.setModel(model)

    def _input_table_context_menu_event(self, a0) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction(
            "Sort by index ascending", lambda: self.ui.input_table.model().sort_index()
        )
        menu.addAction(
            "Sort by index descending",
            lambda: self.ui.input_table.model().sort_index(ascending=False),
        )
        menu.move(a0.globalPos())
        menu.show()

    # endregion

    # region dec table

    def _initial_dec_table(self) -> None:
        self._reset_dec_table()
        self.ui.dec_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.ui.dec_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.ui.dec_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.ui.dec_table.doubleClicked.connect(
            lambda: self.fitting.dec_table_double_clicked()
        )

    def _reset_dec_table(self) -> None:
        df = DataFrame(columns=["Filename"])
        model = PandasModelDeconvTable(df)
        self.ui.dec_table.setModel(model)

    def _dec_table_context_menu_event(self, a0) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction("To template", self.to_template_clicked)
        menu.addAction(
            "Copy spectrum lines parameters from template",
            self.fitting.copy_spectrum_lines_parameters_from_template,
        )
        menu.move(a0.globalPos())
        menu.show()

    @asyncSlot()
    async def to_template_clicked(self) -> None:
        selected_rows = self.ui.dec_table.selectionModel().selectedRows()
        if len(selected_rows) == 0:
            return
        selected_filename = self.ui.dec_table.model().cell_data_by_index(
            selected_rows[0]
        )
        self.fitting.update_single_deconvolution_plot(selected_filename, True)

    # endregion

    # region deconv_lines_table
    def _initial_deconv_lines_table(self) -> None:
        self._reset_deconv_lines_table()
        self.ui.deconv_lines_table.verticalHeader().setSectionsMovable(True)
        self.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.ui.deconv_lines_table.horizontalHeader().resizeSection(0, 110)
        self.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        self.ui.deconv_lines_table.horizontalHeader().resizeSection(1, 150)
        self.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.ui.deconv_lines_table.horizontalHeader().resizeSection(2, 150)
        self.ui.deconv_lines_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.ui.deconv_lines_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.ui.deconv_lines_table.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.ui.deconv_lines_table.setDragDropOverwriteMode(False)
        self.ui.deconv_lines_table.horizontalHeader().sectionClicked.connect(
            self._deconv_lines_table_header_clicked
        )
        self.ui.deconv_lines_table.keyPressEvent = self.deconv_lines_table_key_pressed
        self.ui.deconv_lines_table.contextMenuEvent = (
            self._deconv_lines_table_context_menu_event
        )
        self.ui.deconv_lines_table.clicked.connect(self._deconv_lines_table_clicked)
        # self.ui.deconv_lines_table.verticalHeader().setVisible(False)

    def _deconv_lines_table_clicked(self) -> None:
        selected_indexes = self.ui.deconv_lines_table.selectionModel().selectedIndexes()
        if len(selected_indexes) == 0:
            return
        self.fitting.set_rows_visibility()
        row = selected_indexes[0].row()
        idx = self.ui.deconv_lines_table.model().index_by_row(row)
        if (
                self.fitting.updating_fill_curve_idx is not None
                and self.fitting.updating_fill_curve_idx
                in self.ui.deconv_lines_table.model().dataframe().index
        ):
            curve_style = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(
                self.fitting.updating_fill_curve_idx, "Style"
            )
            self.fitting.update_curve_style(
                self.fitting.updating_fill_curve_idx, curve_style
            )
        self.fitting.start_fill_timer(idx)

    def deconv_lines_table_key_pressed(self, key_event) -> None:
        if (
                key_event.key() == Qt.Key.Key_Delete
                and self.ui.deconv_lines_table.selectionModel().currentIndex().row() > -1
                and len(self.ui.deconv_lines_table.selectionModel().selectedIndexes())
                and self.fitting.is_template
        ):
            self.time_start = datetime.now()
            command = CommandDeleteDeconvLines(self, "Delete line")
            self.undoStack.push(command)

    def _reset_deconv_lines_table(self) -> None:
        df = DataFrame(columns=["Legend", "Type", "Style"])
        model = PandasModelDeconvLinesTable(self, df, [])
        self.ui.deconv_lines_table.setSortingEnabled(True)
        self.ui.deconv_lines_table.setModel(model)
        combobox_delegate = ComboDelegate(peak_shape_names())
        self.ui.deconv_lines_table.setItemDelegateForColumn(1, combobox_delegate)
        self.ui.deconv_lines_table.model().sigCheckedChanged.connect(
            self.fitting.show_hide_curve
        )
        combobox_delegate.sigLineTypeChanged.connect(
            lambda: self.fitting.curve_type_changed()
        )
        self.ui.deconv_lines_table.clicked.connect(self.deconv_lines_table_clicked)

    @asyncSlot()
    async def deconv_lines_table_clicked(self) -> None:
        current_index = self.ui.deconv_lines_table.selectionModel().currentIndex()
        current_column = current_index.column()
        current_row = current_index.row()
        row_data = self.ui.deconv_lines_table.model().row_data(current_row)
        idx = row_data.name
        style = row_data["Style"]
        if current_column != 2:
            return
        for obj in get_objects():
            if (
                    isinstance(obj, CurvePropertiesWindow)
                    and obj.idx() == idx
                    and obj.isVisible()
            ):
                return
        window_cp = CurvePropertiesWindow(self, style, idx)
        window_cp.sigStyleChanged.connect(self._update_deconv_curve_style)
        window_cp.show()

    def _deconv_lines_table_header_clicked(self, idx: int):
        df = self.ui.deconv_lines_table.model().dataframe()
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending_deconv_lines_table = not self._ascending_deconv_lines_table
        if current_name != "Style":
            self.ui.deconv_lines_table.model().sort_values(
                current_name, self._ascending_deconv_lines_table
            )
        self.fitting.deselect_selected_line()

    def _deconv_lines_table_context_menu_event(self, a0) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        # noinspection PyTypeChecker
        menu.addAction("Delete line", self.delete_line_clicked)
        menu.addAction("Clear table", self.clear_all_deconv_lines)
        menu.move(a0.globalPos())
        menu.show()

    @asyncSlot()
    async def delete_line_clicked(self) -> None:
        if not self.fitting.is_template:
            msg = MessageBox(
                "Warning.",
                "Deleting lines is only possible in template mode.",
                self,
                {"Ok"},
            )
            msg.setInformativeText("Press the Template button")
            msg.exec()
            return
        selected_indexes = self.ui.deconv_lines_table.selectionModel().selectedIndexes()
        if len(selected_indexes) == 0:
            return
        command = CommandDeleteDeconvLines(self, "Delete line")
        self.undoStack.push(command)

    # endregion

    # region deconv_params_table
    def _initial_deconv_params_table(self) -> None:
        self._reset_deconv_params_table()
        dsb_delegate = DoubleSpinBoxDelegate(self)
        for i in range(1, 4):
            self.ui.fit_params_table.setItemDelegateForColumn(i, dsb_delegate)
        dsb_delegate.sigLineParamChanged.connect(self.curve_parameter_changed)
        self.ui.fit_params_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.ui.fit_params_table.horizontalHeader().resizeSection(0, 70)
        self.ui.fit_params_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        self.ui.fit_params_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.ui.fit_params_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.ui.fit_params_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.ui.fit_params_table.contextMenuEvent = (
            self._deconv_params_table_context_menu_event
        )
        self.ui.fit_params_table.verticalHeader().setVisible(False)

    def _reset_deconv_params_table(self) -> None:
        tuples = [("", 0, "a")]
        multi_index = MultiIndex.from_tuples(
            tuples, names=("filename", "line_index", "param_name")
        )
        df = DataFrame(
            columns=["Parameter", "Value", "Min value", "Max value"], index=multi_index
        )
        model = PandasModelFitParamsTable(self, df)
        self.ui.fit_params_table.setModel(model)
        self.ui.fit_params_table.model().clear_dataframe()

    def _deconv_params_table_context_menu_event(self, a0) -> None:
        if self.fitting.is_template:
            return
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction(
            "Copy line parameters from template",
            lambda: self.fitting.copy_line_parameters_from_template(),
        )
        menu.move(a0.globalPos())
        menu.show()

    # endregion

    # region pca plsda  features
    def _initial_pca_features_table(self) -> None:
        self._reset_pca_features_table()

    def _reset_pca_features_table(self) -> None:
        df = DataFrame(columns=["feature", "PC-1", "PC-2"])
        model = PandasModelPCA(self, df)
        self.ui.pca_features_table_view.setModel(model)

    def _initial_plsda_vip_table(self) -> None:
        self._reset_plsda_vip_table()

    def _reset_plsda_vip_table(self) -> None:
        df = DataFrame(columns=["feature", "VIP"])
        model = PandasModel(df)
        self.ui.plsda_vip_table_view.setModel(model)

    # endregion

    # region fit intervals
    def _initial_fit_intervals_table(self) -> None:
        self._reset_fit_intervals_table()
        self.ui.fit_borders_TableView.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.ui.fit_borders_TableView.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.ui.fit_borders_TableView.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.ui.fit_borders_TableView.contextMenuEvent = (
            self._fit_intervals_table_context_menu_event
        )
        self.ui.fit_borders_TableView.keyPressEvent = (
            self._fit_intervals_table_key_pressed
        )
        dsb_delegate = IntervalsTableDelegate(self.ui.fit_borders_TableView, self)
        self.ui.fit_borders_TableView.setItemDelegateForColumn(0, dsb_delegate)
        self.ui.fit_borders_TableView.verticalHeader().setVisible(False)
        self.ui.fit_borders_TableView.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

    def _reset_fit_intervals_table(self) -> None:
        df = DataFrame(columns=["Border"])
        model = PandasModelFitIntervals(df)
        self.ui.fit_borders_TableView.setModel(model)

    def _fit_intervals_table_context_menu_event(self, a0) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction("Add border", self._fit_intervals_table_add)
        menu.addAction(
            "Delete selected", self._fit_intervals_table_delete_by_context_menu
        )
        menu.addAction("Auto-search borders", self.fitting.auto_search_borders)
        menu.move(a0.globalPos())
        menu.show()

    def _fit_intervals_table_add(self) -> None:
        command = CommandFitIntervalAdded(self, "Add new interval border")
        self.undoStack.push(command)

    def _fit_intervals_table_delete_by_context_menu(self) -> None:
        self._fit_intervals_table_delete()

    def _fit_intervals_table_key_pressed(self, key_event) -> None:
        if (
                key_event.key() == Qt.Key.Key_Delete
                and self.ui.fit_borders_TableView.selectionModel().currentIndex().row() > -1
                and len(self.ui.fit_borders_TableView.selectionModel().selectedIndexes())
        ):
            self._fit_intervals_table_delete()

    def _fit_intervals_table_delete(self) -> None:
        selection = self.ui.fit_borders_TableView.selectionModel()
        row = selection.currentIndex().row()
        interval_number = self.ui.fit_borders_TableView.model().row_data(row).name
        command = CommandFitIntervalDeleted(
            self, interval_number, "Delete selected border"
        )
        self.undoStack.push(command)

    # endregion

    # region smoothed dataset
    def _initial_smoothed_dataset_table(self) -> None:
        self._reset_smoothed_dataset_table()
        self.ui.smoothed_dataset_table_view.verticalScrollBar().valueChanged.connect(
            self.move_side_scrollbar
        )
        self.ui.smoothed_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    def _reset_smoothed_dataset_table(self) -> None:
        df = DataFrame(columns=["Class", "Filename"])
        model = PandasModelSmoothedDataset(self, df)
        self.ui.smoothed_dataset_table_view.setModel(model)

    # endregion

    # region baselined corrected dataset
    def _initial_baselined_dataset_table(self) -> None:
        self._reset_baselined_dataset_table()
        self.ui.baselined_dataset_table_view.verticalScrollBar().valueChanged.connect(
            self.move_side_scrollbar
        )
        self.ui.baselined_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    def _reset_baselined_dataset_table(self) -> None:
        df = DataFrame(columns=["Class", "Filename"])
        model = PandasModelBaselinedDataset(self, df)
        self.ui.baselined_dataset_table_view.setModel(model)

    # endregion

    # region deconvoluted dataset

    def _initial_deconvoluted_dataset_table(self) -> None:
        self._reset_deconvoluted_dataset_table()
        self.ui.deconvoluted_dataset_table_view.verticalScrollBar().valueChanged.connect(
            self.move_side_scrollbar
        )
        self.ui.deconvoluted_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ui.deconvoluted_dataset_table_view.keyPressEvent = (
            self.decomp_table_key_pressed
        )
        self.ui.deconvoluted_dataset_table_view.model().modelReset.connect(
            lambda: self._init_current_filename_combobox()
        )

    def _reset_deconvoluted_dataset_table(self) -> None:
        df = DataFrame(columns=["Class", "Filename"])
        model = PandasModelDeconvolutedDataset(self, df)
        self.ui.deconvoluted_dataset_table_view.setModel(model)

    def decomp_table_key_pressed(self, key_event) -> None:
        if (
                key_event.key() == Qt.Key.Key_Delete
                and self.ui.deconvoluted_dataset_table_view.selectionModel()
                .currentIndex()
                .row()
                > -1
                and len(
            self.ui.deconvoluted_dataset_table_view.selectionModel().selectedIndexes()
        )
        ):
            self.time_start = datetime.now()
            command = CommandDeleteDatasetRow(self, "Delete row")
            self.undoStack.push(command)

    # endregion

    # region ignore features dataset

    def _initial_ignore_dataset_table(self) -> None:
        self._reset_ignore_dataset_table()
        self.ui.ignore_dataset_table_view.verticalScrollBar().valueChanged.connect(
            self.move_side_scrollbar
        )
        # self.ui.ignore_dataset_table_view.verticalHeader().setVisible(False)
        self.ui.ignore_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ui.ignore_dataset_table_view.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.ui.ignore_dataset_table_view.horizontalHeader().resizeSection(0, 220)
        self.ui.ignore_dataset_table_view.setSortingEnabled(True)
        self.ui.ignore_dataset_table_view.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        self.ui.ignore_dataset_table_view.horizontalHeader().resizeSection(1, 200)
        self.ui.ignore_dataset_table_view.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.ui.ignore_dataset_table_view.horizontalHeader().resizeSection(2, 200)
        self.ui.ignore_dataset_table_view.horizontalHeader().sectionClicked.connect(
            self._ignore_dataset_table_header_clicked
        )

    def _reset_ignore_dataset_table(self) -> None:
        df = DataFrame(columns=["Feature", "Score", "P value"])
        model = PandasModelIgnoreDataset(self, df, {})
        self.ui.ignore_dataset_table_view.setModel(model)

    def _ignore_dataset_table_header_clicked(self, idx: int):
        df = self.ui.ignore_dataset_table_view.model().dataframe()
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending_ignore_table = not self._ascending_ignore_table
        self.ui.ignore_dataset_table_view.model().sort_values(
            current_name, self._ascending_ignore_table
        )

    # endregion

    # region describe dataset

    def _initial_describe_dataset_tables(self) -> None:
        self._reset_describe_dataset_tables()
        self.ui.describe_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ui.describe_2nd_group.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ui.describe_1st_group.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    def _reset_describe_dataset_tables(self) -> None:
        model = PandasModelDescribeDataset(self, DataFrame())
        self.ui.describe_dataset_table_view.setModel(model)

        model = PandasModelDescribeDataset(self, DataFrame())
        self.ui.describe_1st_group.setModel(model)
        model = PandasModelDescribeDataset(self, DataFrame())
        self.ui.describe_2nd_group.setModel(model)

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

    # region deconv_buttons_frame

    def _initial_guess_table_frame(self) -> None:
        """right side frame with lines table, parameters table and report field"""
        self._initial_add_line_button()
        self._initial_guess_button()
        # self._initial_batch_button()
        self.ui.fit_pushButton.clicked.connect(self.fit)
        self.ui.batch_button.clicked.connect(self.batch_fit)
        self.ui.data_checkBox.stateChanged.connect(self.data_cb_state_changed)
        self.ui.sum_checkBox.stateChanged.connect(self.sum_cb_state_changed)
        self.ui.sigma3_checkBox.stateChanged.connect(self.sigma3_cb_state_changed)
        self.ui.residual_checkBox.stateChanged.connect(self.residual_cb_state_changed)
        self.ui.data_pushButton.clicked.connect(self.data_pb_clicked)
        self.ui.sum_pushButton.clicked.connect(self.sum_pb_clicked)
        self.ui.residual_pushButton.clicked.connect(self.residual_pb_clicked)
        self.ui.sigma3_pushButton.clicked.connect(self.sigma3_push_button_clicked)
        self.ui.interval_start_dsb.valueChanged.connect(
            self.fitting.interval_start_dsb_change_event
        )
        self.ui.interval_end_dsb.valueChanged.connect(
            self.fitting.interval_end_dsb_change_event
        )
        self.ui.interval_checkBox.stateChanged.connect(self.interval_cb_state_changed)
        self.linearRegionDeconv.setVisible(False)

    def _initial_guess_button(self) -> None:
        guess_menu = QMenu()
        line_type: str
        for line_type in peak_shape_names():
            action = guess_menu.addAction(line_type)
            action.triggered.connect(
                lambda checked=None, line=line_type: self.guess(line_type=line)
            )
        self.ui.guess_button.setMenu(guess_menu)
        self.ui.guess_button.menu()

    # def _initial_batch_button(self) -> None:
    #     batch_menu = QMenu()
    #     action = batch_menu.addAction('All')
    #     action.triggered.connect(lambda: self.batch_fit('All'))
    #     action = batch_menu.addAction('Unfitted')
    #     action.triggered.connect(lambda: self.batch_fit('Unfitted'))
    #     self.ui.batch_button.setMenu(batch_menu)
    #     self.ui.batch_button.menu()

    def _initial_add_line_button(self) -> None:
        add_lines_menu = QMenu()
        line_type: str
        for line_type in peak_shape_names():
            action = add_lines_menu.addAction(line_type)
            action.triggered.connect(
                lambda checked=None, line=line_type: self.add_deconv_line(
                    line_type=line
                )
            )
        self.ui.add_line_button.setMenu(add_lines_menu)
        self.ui.add_line_button.menu()

    def data_cb_state_changed(self, a0: int) -> None:
        """
        Set data_curve visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None

        """
        if self.fitting.data_curve is None:
            return
        self.fitting.data_curve.setVisible(a0 == 2)

    def sum_cb_state_changed(self, a0: int) -> None:
        """
        Set sum_curve visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None

        """
        if self.fitting.sum_curve is None:
            return
        self.fitting.sum_curve.setVisible(a0 == 2)

    def sigma3_cb_state_changed(self, a0: int) -> None:
        """
        Set sigma3 fill visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None
        """
        if self.fitting.sigma3_fill is None:
            return
        self.fitting.sigma3_fill.setVisible(a0 == 2)

    def residual_cb_state_changed(self, a0: int) -> None:
        """
        Set residual_curve visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None

        """
        if self.fitting.residual_curve is None:
            return
        self.fitting.residual_curve.setVisible(a0 == 2)

    def data_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.data_pushButton.setStyleSheet(
            f"""*{{background-color: {hex_color};}}"""
        )

    def sum_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.sum_pushButton.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def sigma3_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.sigma3_pushButton.setStyleSheet(
            f"""*{{background-color: {hex_color};}}"""
        )

    def residual_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.residual_pushButton.setStyleSheet(
            f"""*{{background-color: {hex_color};}}"""
        )

    def _update_data_curve_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(
            self, style, old_style, "data", "Update style for data curve"
        )
        self.undoStack.push(command)

    def _update_sum_curve_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(
            self, style, old_style, "sum", "Update style for data curve"
        )
        self.undoStack.push(command)

    def _update_residual_curve_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(
            self, style, old_style, "residual", "Update style for data curve"
        )
        self.undoStack.push(command)

    def _update_sigma3_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(
            self, style, old_style, "sigma3", "Update style for sigma3"
        )
        self.undoStack.push(command)

    def data_pb_clicked(self) -> None:
        if self.fitting.data_curve is None:
            return
        for obj in get_objects():
            if (
                    isinstance(obj, CurvePropertiesWindow)
                    and obj.idx() == 999
                    and obj.isVisible()
            ):
                return
        data_curve_prop_window = CurvePropertiesWindow(
            self, self.fitting.data_style, 999, False
        )
        data_curve_prop_window.sigStyleChanged.connect(self._update_data_curve_style)
        data_curve_prop_window.show()

    def sum_pb_clicked(self) -> None:
        if self.fitting.sum_curve is None:
            return
        for obj in get_objects():
            if (
                    isinstance(obj, CurvePropertiesWindow)
                    and obj.idx() == 998
                    and obj.isVisible()
            ):
                return
        sum_curve_prop_window = CurvePropertiesWindow(
            self, self.fitting.sum_style, 998, False
        )
        sum_curve_prop_window.sigStyleChanged.connect(self._update_sum_curve_style)
        sum_curve_prop_window.show()

    def residual_pb_clicked(self) -> None:
        if self.fitting.residual_curve is None:
            return
        for obj in get_objects():
            if (
                    isinstance(obj, CurvePropertiesWindow)
                    and obj.idx() == 997
                    and obj.isVisible()
            ):
                return
        residual_curve_prop_window = CurvePropertiesWindow(
            self, self.fitting.residual_style, 997, False
        )
        residual_curve_prop_window.sigStyleChanged.connect(
            self._update_residual_curve_style
        )
        residual_curve_prop_window.show()

    def sigma3_push_button_clicked(self) -> None:
        for obj in get_objects():
            if (
                    isinstance(obj, CurvePropertiesWindow)
                    and obj.idx() == 996
                    and obj.isVisible()
            ):
                return
        prop_window = CurvePropertiesWindow(self, self.fitting.sigma3_style, 996, True)
        prop_window.sigStyleChanged.connect(self._update_sigma3_style)
        prop_window.show()

    def lr_deconv_region_changed(self) -> None:
        current_region = self.linearRegionDeconv.getRegion()
        self.ui.interval_start_dsb.setValue(current_region[0])
        self.ui.interval_end_dsb.setValue(current_region[1])

    def interval_cb_state_changed(self, a0: int) -> None:
        """a0 = 0 is False, a0 = 2 if True"""
        self.linearRegionDeconv.setVisible(a0 == 2)
        if a0 == 2:
            self.cut_data_sum_residual_interval()
        else:
            self.uncut_data_sum_residual()

    @asyncSlot()
    async def cut_data_sum_residual_interval(self) -> None:
        self.fitting.cut_data_interval()
        self.fitting.redraw_curves_for_filename()
        self.fitting.draw_sum_curve()
        self.fitting.draw_residual_curve()

    @asyncSlot()
    async def uncut_data_sum_residual(self) -> None:
        self.uncut_data()
        self.fitting.redraw_curves_for_filename()
        self.fitting.draw_sum_curve()
        self.fitting.draw_residual_curve()

    def uncut_data(self) -> None:
        n_array = self.fitting.array_of_current_filename_in_deconvolution()
        if n_array is None:
            return
        self.fitting.data_curve.setData(x=n_array[:, 0], y=n_array[:, 1])

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

    def scroll_area_stat_value_changed(self, event: int):
        self.ui.verticalScrollBar.setValue(event)

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

    def leftside_btn_clicked(self) -> None:
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
        self._initial_deconv_plot_color()
        self._initial_stat_plots_color()
        self.initial_plots_labels()

    # endregion

    # endregion

    # region Plot buttons

    # region crosshair button
    def linear_region_movable_btn_clicked(self) -> None:
        b = not self.ui.lr_movableBtn.isChecked()
        self.context.preprocessing.stages.cut_data.linear_region.setMovable(b)
        self.linearRegionDeconv.setMovable(b)

    def linear_region_show_hide_btn_clicked(self) -> None:
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        if self.ui.lr_showHideBtn.isChecked():
            plot_item.addItem(self.context.preprocessing.stages.cut_data.linear_region)
            self.deconvolution_plotItem.addItem(self.linearRegionDeconv)
        else:
            plot_item.removeItem(self.context.preprocessing.stages.cut_data.linear_region)
            self.deconvolution_plotItem.removeItem(self.linearRegionDeconv)

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
        self.deconvolution_plotItem.removeItem(self.ui.deconv_plot_widget.vertical_line)
        self.deconvolution_plotItem.removeItem(
            self.ui.deconv_plot_widget.horizontal_line
        )
        if self.ui.crosshairBtn.isChecked():
            self.deconvolution_plotItem.addItem(
                self.ui.deconv_plot_widget.vertical_line, ignoreBounds=True
            )
            self.deconvolution_plotItem.addItem(
                self.ui.deconv_plot_widget.horizontal_line, ignoreBounds=True
            )
        elif not self.ui.by_one_control_button.isChecked():
            new_title = (
                    '<span style="font-family: AbletonSans; color:'
                    + environ["plotText"]
                    + ';font-size:14pt">'
                    + self.fitting.current_spectrum_deconvolution_name
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

    def update_crosshair_averaged_plot(self, point_event) -> None:
        """Paint crosshair on mouse"""
        coordinates = point_event[0]
        if (
                self.ui.average_plot_widget.sceneBoundingRect().contains(coordinates)
                and self.ui.crosshairBtn.isChecked()
        ):
            mouse_point = self.ui.average_plot_widget.plotItem.vb.mapSceneToView(
                coordinates
            )
            self.ui.average_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:"
                + environ["secondaryColor"]
                + "'>x=%0.1f,   <span style=>y=%0.1f</span>"
                % (mouse_point.x(), mouse_point.y())
            )
            self.ui.average_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.average_plot_widget.horizontal_line.setPos(mouse_point.y())

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
            current_group_name = self.context.group_table.table_widget.model().cell_data(current_row, 0)
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

    def update_all_average_plot(self) -> None:
        if len(self.context.preprocessing.stages.av_data.data) > 0:
            self.ui.average_plot_widget.setTitle(
                '<span style="font-family: AbletonSans; color:'
                + environ["plotText"]
                + ';font-size:14pt">Average</span>'
            )
        data_items = self.averaged_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        self.averaged_plotItem.getViewBox().updateAutoRange()

    # endregion

    def stacked_widget_changed(self) -> None:
        self.crosshair_btn_clicked()
        self.deconvolution_plotItem.getViewBox().updateAutoRange()

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

    # region input_table

    def input_table_vertical_scrollbar_value_changed(self, _) -> None:
        self.decide_vertical_scroll_bar_visible()

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

    def input_table_item_changed(
            self, top_left: QModelIndex = None, _: QModelIndex = None
    ) -> None:
        """You can change only group column"""
        if self.ui.input_table.selectionModel().currentIndex().column() == 2:
            try:
                new_value = int(
                    self.ui.input_table.model().cell_data_by_index(top_left)
                )
            except ValueError:
                self.ui.input_table.model().setData(
                    top_left, self.previous_group_of_item, Qt.EditRole
                )
                return
            if self.context.group_table.table_widget.model().rowCount() >= new_value >= 0:
                filename = self.ui.input_table.model().cell_data(top_left.row(), 0)
                command = CommandChangeGroupCell(
                    self,
                    top_left,
                    new_value,
                    "Change group number for (%s)" % str(filename),
                )
                self.undoStack.push(command)
            else:
                self.ui.input_table.model().setData(
                    top_left, self.previous_group_of_item, Qt.EditRole
                )

    @asyncSlot()
    async def input_table_selection_changed(self):
        """
        When selected item in input_table.
        Saving self.previous_group_of_item
        or update import_plot
        """
        current_index = self.ui.input_table.selectionModel().currentIndex()
        if current_index:
            self.ui.input_table.scrollTo(current_index)
        if current_index.column() == 2:  # groups
            self.previous_group_of_item = int(
                self.ui.input_table.model().cell_data_by_index(current_index)
            )
        elif self.ui.by_one_control_button.isChecked():  # names
            tasks = [create_task(
                self.context.preprocessing.stages.input_data.despike_history_remove_plot())]
            tasks.append(create_task(self.update_plots_for_single()))
            await wait(tasks)

    def _input_table_header_clicked(self, idx: int):
        df = self.ui.input_table.model().dataframe()
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending_input_table = not self._ascending_input_table
        self.ui.input_table.model().sort_values(
            current_name, self._ascending_input_table
        )

    # endregion

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
            case Qt.Key.Key_F2:
                print(self.drag_widget.get_widgets_order())
            case Qt.Key.Key_End:
                self.executor_stop()
            # case Qt.Key.Key_F2:
            # print('F1')
            # self.preprocessing.baseline_corrected_dict = self.preprocessing.NormalizedDict
            # self.preprocessing.baseline_corrected_not_trimmed_dict = self.preprocessing.NormalizedDict
            # action_help()
            case Qt.Key.Key_F2:
                from src.stages.fitting.functions.guess_raman_lines import (
                    show_distribution,
                )

                if self.ui.stackedWidget_mainpages.currentIndex() != 1:
                    return
                if self.fitting.intervals_data is None:
                    return
                for i, v in enumerate(self.fitting.intervals_data.items()):
                    key, item = v
                    show_distribution(
                        item["x0"],
                        self.context.preprocessing.stages.av_data.data,
                        self.fitting.all_ranges_clustered_x0_sd[i],
                        self.context.group_table.table_widget.model().groups_colors,
                    )
            case Qt.Key.Key_F7:
                self.ui.input_table.model().dataframe()
            case Qt.Key.Key_F5:
                ovcs = list(
                    self.fitting.overlapping_coefficients_for_each_line().values()
                )
                print(ovcs)
                # ovcs = ovcs[~np.isnan(ovcs)]
                # print(ovcs)
                print(np.mean(ovcs))
                print(np.median(ovcs))
                print(np.min(ovcs))
                print(np.max(ovcs))
            case Qt.Key.Key_F10:
                self.fitting.deriv()
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
        # self.update_undo_redo_tooltips()

    def redo(self) -> None:
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.context.undo_stack.redo()
        # self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        return
        if self.undoStack.canUndo():
            self.action_undo.setToolTip(self.undoStack.undoText())
        else:
            self.action_undo.setToolTip("")

        if self.undoStack.canRedo():
            self.action_redo.setToolTip(self.undoStack.redoText())
        else:
            self.action_redo.setToolTip("")

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
            db["DeconvLinesTableDF"] = self.ui.deconv_lines_table.model().dataframe()
            db["DeconvParamsTableDF"] = self.ui.fit_params_table.model().dataframe()
            db["intervals_table_df"] = self.ui.fit_borders_TableView.model().dataframe()
            db["DeconvLinesTableChecked"] = self.ui.deconv_lines_table.model().checked()
            db["IgnoreTableChecked"] = self.ui.ignore_dataset_table_view.model().checked
            db["select_percentile_spin_box"] = (
                self.ui.select_percentile_spin_box.value()
            )
            db["interval_start_cm"] = self.ui.interval_start_dsb.value()
            db["interval_end_cm"] = self.ui.interval_end_dsb.value()
            db["guess_method_cb"] = self.ui.guess_method_cb.currentText()
            db["dataset_type_cb"] = self.ui.dataset_type_cb.currentText()
            db["classes_lineEdit"] = self.ui.classes_lineEdit.text()
            db["test_data_ratio_spinBox"] = self.ui.test_data_ratio_spinBox.value()
            db["max_noise_level"] = self.ui.max_noise_level_dsb.value()
            db["l_ratio_doubleSpinBox"] = self.ui.l_ratio_doubleSpinBox.value()
            db["data_style"] = self.fitting.data_style.copy()
            db["data_curve_checked"] = self.ui.data_checkBox.isChecked()
            db["sum_style"] = self.fitting.sum_style.copy()
            db["sigma3_style"] = self.fitting.sigma3_style.copy()
            # db["params_stderr"] = self.fitting.params_stderr.copy()
            db["sum_curve_checked"] = self.ui.sum_checkBox.isChecked()
            db["sigma3_checked"] = self.ui.sigma3_checkBox.isChecked()
            db["residual_style"] = self.fitting.residual_style.copy()
            db["residual_curve_checked"] = self.ui.residual_checkBox.isChecked()
            db["interval_checkBox_checked"] = self.ui.interval_checkBox.isChecked()
            db["fit_method"] = self.ui.fit_opt_method_comboBox.currentText()
            db["use_fit_intervals"] = self.ui.intervals_gb.isChecked()
            db["use_shapley_cb"] = self.ui.use_shapley_cb.isChecked()
            db["random_state_cb"] = self.ui.random_state_cb.isChecked()
            db["random_state_sb"] = self.ui.random_state_sb.value()
            db["max_dx_guess"] = self.ui.max_dx_dsb.value()
            db["_y_axis_ref_EMSC"] = self.predict_logic.y_axis_ref_EMSC
            db["activation_comboBox"] = self.ui.activation_comboBox.currentText()
            db["criterion_comboBox"] = self.ui.criterion_comboBox.currentText()
            db["rf_criterion_comboBox"] = self.ui.rf_criterion_comboBox.currentText()
            db["rf_max_features_comboBox"] = (
                self.ui.rf_max_features_comboBox.currentText()
            )
            db["refit_score"] = self.ui.refit_score.currentText()
            db["solver_mlp_combo_box"] = self.ui.solver_mlp_combo_box.currentText()
            db["mlp_layer_size_spinBox"] = self.ui.mlp_layer_size_spinBox.value()
            db["rf_min_samples_split_spinBox"] = (
                self.ui.rf_min_samples_split_spinBox.value()
            )
            db["ab_n_estimators_spinBox"] = self.ui.ab_n_estimators_spinBox.value()
            db["ab_learning_rate_doubleSpinBox"] = (
                self.ui.ab_learning_rate_doubleSpinBox.value()
            )
            db["xgb_eta_doubleSpinBox"] = self.ui.xgb_eta_doubleSpinBox.value()
            db["xgb_gamma_spinBox"] = self.ui.xgb_gamma_spinBox.value()
            db["xgb_max_depth_spinBox"] = self.ui.xgb_max_depth_spinBox.value()
            db["xgb_min_child_weight_spinBox"] = (
                self.ui.xgb_min_child_weight_spinBox.value()
            )
            db["xgb_colsample_bytree_doubleSpinBox"] = (
                self.ui.xgb_colsample_bytree_doubleSpinBox.value()
            )
            db["xgb_lambda_doubleSpinBox"] = self.ui.xgb_lambda_doubleSpinBox.value()
            db["xgb_n_estimators_spinBox"] = self.ui.xgb_n_estimators_spinBox.value()
            db["rf_n_estimators_spinBox"] = self.ui.rf_n_estimators_spinBox.value()
            db["max_epoch_spinBox"] = self.ui.max_epoch_spinBox.value()
            db["dt_min_samples_split_spin_box"] = (
                self.ui.dt_min_samples_split_spin_box.value()
            )
            db["dt_max_depth_spin_box"] = self.ui.dt_max_depth_spin_box.value()
            db["learning_rate_doubleSpinBox"] = (
                self.ui.learning_rate_doubleSpinBox.value()
            )
            db["feature_display_max_checkBox"] = (
                self.ui.feature_display_max_checkBox.isChecked()
            )
            db["include_x0_checkBox"] = self.ui.include_x0_checkBox.isChecked()
            db["feature_display_max_spinBox"] = (
                self.ui.feature_display_max_spinBox.value()
            )
            db["old_Y"] = self.stat_analysis_logic.old_labels
            db["new_Y"] = self.stat_analysis_logic.new_labels
            db["use_pca_checkBox"] = self.ui.use_pca_checkBox.isChecked()
            db["intervals_data"] = self.fitting.intervals_data
            db["all_ranges_clustered_lines_x0"] = (
                self.fitting.all_ranges_clustered_x0_sd
            )
            db["comboBox_lda_solver"] = self.ui.comboBox_lda_solver.currentText()
            db["lda_solver_check_box"] = self.ui.lda_solver_check_box.isChecked()
            db["lda_shrinkage_check_box"] = self.ui.lda_shrinkage_check_box.isChecked()
            db["lr_penalty_checkBox"] = self.ui.lr_penalty_checkBox.isChecked()
            db["lr_solver_checkBox"] = self.ui.lr_solver_checkBox.isChecked()
            db["svc_nu_check_box"] = self.ui.svc_nu_check_box.isChecked()
            db["n_neighbors_checkBox"] = self.ui.n_neighbors_checkBox.isChecked()
            db["learning_rate_checkBox"] = self.ui.learning_rate_checkBox.isChecked()
            db["mlp_layer_size_checkBox"] = self.ui.mlp_layer_size_checkBox.isChecked()
            db["mlp_solve_checkBox"] = self.ui.mlp_solve_checkBox.isChecked()
            db["activation_checkBox"] = self.ui.activation_checkBox.isChecked()
            db["criterion_checkBox"] = self.ui.criterion_checkBox.isChecked()
            db["dt_min_samples_split_check_box"] = (
                self.ui.dt_min_samples_split_check_box.isChecked()
            )
            db["dt_max_depth_check_box"] = self.ui.dt_max_depth_check_box.isChecked()
            db["rf_max_features_checkBox"] = (
                self.ui.rf_max_features_checkBox.isChecked()
            )
            db["rf_n_estimators_checkBox"] = (
                self.ui.rf_n_estimators_checkBox.isChecked()
            )
            db["rf_min_samples_split_checkBox"] = (
                self.ui.rf_min_samples_split_checkBox.isChecked()
            )
            db["rf_criterion_checkBox"] = self.ui.rf_criterion_checkBox.isChecked()
            db["ab_learning_rate_checkBox"] = (
                self.ui.ab_learning_rate_checkBox.isChecked()
            )
            db["ab_n_estimators_checkBox"] = (
                self.ui.ab_n_estimators_checkBox.isChecked()
            )
            db["xgb_eta_checkBox"] = self.ui.xgb_eta_checkBox.isChecked()
            db["xgb_gamma_checkBox"] = self.ui.xgb_gamma_checkBox.isChecked()
            db["xgb_max_depth_checkBox"] = self.ui.xgb_max_depth_checkBox.isChecked()
            db["xgb_min_child_weight_checkBox"] = (
                self.ui.xgb_min_child_weight_checkBox.isChecked()
            )
            db["xgb_colsample_bytree_checkBox"] = (
                self.ui.xgb_colsample_bytree_checkBox.isChecked()
            )
            db["xgb_lambda_checkBox"] = self.ui.xgb_lambda_checkBox.isChecked()
            db["nn_weights_checkBox"] = self.ui.nn_weights_checkBox.isChecked()
            db["lr_penalty_comboBox"] = self.ui.lr_penalty_comboBox.currentText()
            db["lr_solver_comboBox"] = self.ui.lr_solver_comboBox.currentText()
            db["nn_weights_comboBox"] = self.ui.nn_weights_comboBox.currentText()
            db["lr_c_doubleSpinBox"] = self.ui.lr_c_doubleSpinBox.value()
            db["svc_nu_doubleSpinBox"] = self.ui.svc_nu_doubleSpinBox.value()
            db["n_neighbors_spinBox"] = self.ui.n_neighbors_spinBox.value()
            db["lr_c_checkBox"] = self.ui.lr_c_checkBox.isChecked()
            db["lineEdit_lda_shrinkage"] = self.ui.lineEdit_lda_shrinkage.text()
            if not self.predict_logic.is_production_project:
                self.predict_logic.stat_models = {}
                for key, v in self.stat_analysis_logic.latest_stat_result.items():
                    self.predict_logic.stat_models[key] = v["model"]
                db["stat_models"] = self.predict_logic.stat_models
                if self.context.preprocessing.stages.input_data.data:
                    db["interp_ref_array"] = next(iter(self.context.preprocessing.stages.input_data.data.values()))
            if not production_export:
                db["InputTable"] = self.ui.input_table.model().dataframe()
                db["smoothed_dataset_df"] = (
                    self.ui.smoothed_dataset_table_view.model().dataframe()
                )
                db["baselined_dataset_df"] = (
                    self.ui.baselined_dataset_table_view.model().dataframe()
                )
                db["deconvoluted_dataset_df"] = (
                    self.ui.deconvoluted_dataset_table_view.model().dataframe()
                )
                db["ignore_dataset_df"] = (
                    self.ui.ignore_dataset_table_view.model().dataframe()
                )
                db["predict_df"] = self.ui.predict_table_view.model().dataframe()
                db["report_result"] = self.fitting.report_result.copy()
                db["sigma3"] = self.fitting.sigma3.copy()
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

    def action_close_project(self) -> None:
        self.setWindowTitle(" ")
        self.ui.projectLabel.setText("")
        self.project_path = ""
        self._clear_all_parameters()
        self.set_buttons_ability()
        self.set_forms_ability()
        self.decide_vertical_scroll_bar_visible()

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
            recent_limite = int(environ["recent_limit"])
            for idx, i in enumerate(reversed(lines)):
                if idx < recent_limite - 1:
                    lines_fin.append(i)
                idx += 1
            for line in reversed(lines_fin):
                f.write(line + "\n")
            f.write(_path + "\n")

        if not self.recent_menu.isEnabled():
            self.recent_menu.setDisabled(False)

    def _export_files_av(self, item: tuple[str, np.ndarray]) -> None:
        filename = self.context.group_table.table_widget.model().get_group_name_by_int(item[0])
        np.savetxt(
            fname=self.export_folder_path + "/" + filename + ".asc",
            X=item[1],
            fmt="%10.5f",
        )

    def _export_files(self, item: tuple[str, np.ndarray]) -> None:
        np.savetxt(
            fname=self.export_folder_path + "/" + item[0], X=item[1], fmt="%10.5f"
        )

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
            if self.ui.plsda_vip_table_view.model().rowCount() > 0:
                self.ui.plsda_vip_table_view.model().dataframe().to_excel(
                    writer, sheet_name="PLS-DA VIP"
                )
            if self.ui.predict_table_view.model().rowCount() > 0:
                self.ui.predict_table_view.model().dataframe().to_excel(
                    writer, sheet_name="Predicted"
                )

    def clear_selected_step(self, step: str) -> None:
        if step in self.stat_analysis_logic.latest_stat_result:
            del self.stat_analysis_logic.latest_stat_result[step]
        match step:
            case "Deconvolution":
                self.deconvolution_plotItem.clear()
                self.fitting.report_result.clear()
                self.fitting.sigma3.clear()
                del self.fitting.sigma3_fill
                self._initial_deconvolution_plot()
                self.fitting.data_curve = None
                self.fitting.current_spectrum_deconvolution_name = ""
                try:
                    self.ui.template_combo_box.currentTextChanged.disconnect(
                        self.fitting.switch_to_template
                    )
                except:
                    error(
                        "failed to disconnect currentTextChanged self.switch_to_template)"
                    )
                self.ui.template_combo_box.clear()
                self.fitting.is_template = False
                self.fitting.all_ranges_clustered_x0_sd = None
                self.ui.data_checkBox.setChecked(True)
                self.ui.sum_checkBox.setChecked(False)
                self.ui.residual_checkBox.setChecked(False)
                self.ui.include_x0_checkBox.setChecked(False)
                self.ui.sigma3_checkBox.setChecked(False)
                self.data_style_button_style_sheet(
                    self.fitting.data_style["color"].name()
                )
                self.sum_style_button_style_sheet(
                    self.fitting.sum_style["color"].name()
                )
                self.sigma3_style_button_style_sheet(
                    self.fitting.sigma3_style["color"].name()
                )
                self.residual_style_button_style_sheet(
                    self.fitting.residual_style["color"].name()
                )
                self.ui.interval_checkBox.setChecked(False)
                self.linearRegionDeconv.setVisible(False)
                self.ui.deconvoluted_dataset_table_view.model().clear_dataframe()
                self.ui.ignore_dataset_table_view.model().clear_dataframe()
                self.fitting.update_template_combo_box()
                self.ui.current_filename_combobox.clear()
            case "Stat":
                self._initial_all_stat_plots()
                self.ui.current_group_shap_comboBox.clear()
                self.ui.current_feature_comboBox.clear()
                self.ui.current_dep_feature1_comboBox.clear()
                self.ui.current_dep_feature2_comboBox.clear()
                self.ui.coloring_feature_comboBox.clear()
                self.ui.current_instance_combo_box.clear()
                self.ui.feature_display_max_checkBox.setChecked(False)
                self.ui.random_state_cb.setChecked(False)
                self.ui.use_shapley_cb.setChecked(True)
                self.stat_analysis_logic.latest_stat_result = {}
                self._initial_pca_features_table()
                self._initial_plsda_vip_table()
                self.ui.refit_score.setCurrentText("recall_score")
            case "PCA":
                self._initial_pca_plots()
                self._initial_pca_features_table()
            case "PLS-DA":
                self._initial_plsda_plots()
                self._initial_plsda_vip_table()
            case "Page5":
                self._initial_predict_dataset_table()

    def _clear_all_parameters(self) -> None:

        self.CommandEndIntervalChanged_allowed = False
        self.CommandStartIntervalChanged_allowed = False
        self._init_default_values()

        self.ui.deconv_lines_table.model().clear_dataframe()
        self.ui.fit_params_table.model().clear_dataframe()
        self.ui.fit_borders_TableView.model().clear_dataframe()
        self.ui.smoothed_dataset_table_view.model().clear_dataframe()
        self._reset_smoothed_dataset_table()
        self.ui.baselined_dataset_table_view.model().clear_dataframe()
        self._reset_baselined_dataset_table()
        self.ui.deconvoluted_dataset_table_view.model().clear_dataframe()
        self.ui.ignore_dataset_table_view.model().clear_dataframe()
        self._reset_deconvoluted_dataset_table()
        self.predict_logic.is_production_project = False
        self.predict_logic.stat_models = {}
        self.predict_logic.interp_ref_array = None
        self.stat_analysis_logic.top_features = {}
        self.stat_analysis_logic.old_labels = None
        self.stat_analysis_logic.new_labels = None
        self.stat_analysis_logic.latest_stat_result = {}
        self.clear_selected_step("Deconvolution")
        self.clear_selected_step("Stat")
        self.clear_selected_step("Page5")

        self.ui.crosshairBtn.setChecked(False)
        self.crosshair_btn_clicked()

        self.ui.stat_report_text_edit.setText("")
        self._set_parameters_to_default()
        self.set_buttons_ability()
        collect()
        self.undoStack.clear()
        self.CommandEndIntervalChanged_allowed = True
        self.CommandStartIntervalChanged_allowed = True
        # set_modified(self.context, self.ui, False)

    def set_forms_ability(self) -> None:
        b = not self.predict_logic.is_production_project
        self.ui.left_side_head_stackedWidget.setEnabled(b)
        self.ui.deconv_lines_table.setEnabled(b)
        self.ui.deconv_buttons_frame_top.setEnabled(b)
        self.ui.fit_params_table.setEnabled(b)
        self.ui.left_page_1.setEnabled(b)
        self.ui.left_page_3.setEnabled(b)
        self.ui.dec_param_frame.setEnabled(b)
        self.ui.intervals_gb.setEnabled(b)

    @asyncSlot()
    async def load_params(self, path: str) -> None:
        self.fitting.deselect_selected_line()

        self.unzip_project_file(path)
        self.fitting.update_template_combo_box()
        await self.fitting.switch_to_template()
        self.fitting.update_deconv_intervals_limits()
        self.dataset_type_cb_current_text_changed(self.ui.dataset_type_cb.currentText())
        if (
                self.ui.fit_params_table.model().rowCount() != 0
                and self.ui.deconv_lines_table.model().rowCount() != 0
        ):
            await self.fitting.draw_all_curves()
        self.stat_analysis_logic.update_force_single_plots("LDA")
        self.stat_analysis_logic.update_force_full_plots("LDA")
        self.set_buttons_ability()
        self.set_forms_ability()

        # set_modified(self.context, self.ui, False)
        self.decide_vertical_scroll_bar_visible()
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
            if "InputTable" in db:
                df = db["InputTable"]
                self.ui.input_table.model().set_dataframe(df)
                names = df.index
                self.ui.dec_table.model().concat_deconv_table(filename=names)
            if "DeconvLinesTableDF" in db:
                try:
                    df = db["DeconvLinesTableDF"]
                    self.ui.deconv_lines_table.model().set_dataframe(df)
                except ModuleNotFoundError:
                    pass
            if "DeconvLinesTableChecked" in db:
                checked = db["DeconvLinesTableChecked"]
                self.ui.deconv_lines_table.model().set_checked(checked)
            if "IgnoreTableChecked" in db:
                checked = db["IgnoreTableChecked"]
                self.ui.ignore_dataset_table_view.model().set_checked(checked)
            if "DeconvParamsTableDF" in db:
                try:
                    df = db["DeconvParamsTableDF"]
                    self.ui.fit_params_table.model().set_dataframe(df)
                except ModuleNotFoundError:
                    pass
            if "intervals_table_df" in db:
                try:
                    df = db["intervals_table_df"]
                    self.ui.fit_borders_TableView.model().set_dataframe(df)
                except ModuleNotFoundError:
                    pass
            if "smoothed_dataset_df" in db:
                self.ui.smoothed_dataset_table_view.model().set_dataframe(
                    db["smoothed_dataset_df"]
                )
            if "stat_models" in db:
                try:
                    self.predict_logic.stat_models = db["stat_models"]
                except AttributeError as err:
                    print(err)
            if "baselined_dataset_df" in db:
                self.ui.baselined_dataset_table_view.model().set_dataframe(
                    db["baselined_dataset_df"]
                )
            if "deconvoluted_dataset_df" in db:
                self.ui.deconvoluted_dataset_table_view.model().set_dataframe(
                    db["deconvoluted_dataset_df"]
                )
                self._init_current_filename_combobox()
            if "ignore_dataset_df" in db:
                self.ui.ignore_dataset_table_view.model().set_dataframe(
                    db["ignore_dataset_df"]
                )
            if "interp_ref_array" in db:
                self.predict_logic.interp_ref_array = db["interp_ref_array"]
            if "predict_df" in db:
                self.ui.predict_table_view.model().set_dataframe(db["predict_df"])
            if "is_production_project" in db:
                self.predict_logic.is_production_project = db["is_production_project"]
            if "_y_axis_ref_EMSC" in db:
                self.predict_logic.y_axis_ref_EMSC = db["_y_axis_ref_EMSC"]
            if "interval_start_cm" in db:
                self.ui.interval_start_dsb.setValue(db["interval_start_cm"])
            if "interval_end_cm" in db:
                self.ui.interval_end_dsb.setValue(db["interval_end_cm"])
            if "select_percentile_spin_box" in db:
                self.ui.select_percentile_spin_box.setValue(
                    db["select_percentile_spin_box"]
                )
            if "report_result" in db:
                self.fitting.report_result = db["report_result"]
            if "sigma3" in db:
                self.fitting.sigma3 = db["sigma3"]
            if "guess_method_cb" in db:
                self.ui.guess_method_cb.setCurrentText(db["guess_method_cb"])
            if "dataset_type_cb" in db:
                self.ui.dataset_type_cb.setCurrentText(db["dataset_type_cb"])
            if "classes_lineEdit" in db:
                self.ui.classes_lineEdit.setText(db["classes_lineEdit"])
            if "test_data_ratio_spinBox" in db:
                self.ui.test_data_ratio_spinBox.setValue(db["test_data_ratio_spinBox"])
            if "max_noise_level" in db:
                self.ui.max_noise_level_dsb.setValue(db["max_noise_level"])
            if "l_ratio_doubleSpinBox" in db:
                self.ui.l_ratio_doubleSpinBox.setValue(db["l_ratio_doubleSpinBox"])
            if "comboBox_lda_solver" in db:
                self.ui.comboBox_lda_solver.setCurrentText(db["comboBox_lda_solver"])
            if "lr_penalty_comboBox" in db:
                self.ui.lr_penalty_comboBox.setCurrentText(db["lr_penalty_comboBox"])
            if "lr_solver_comboBox" in db:
                self.ui.lr_solver_comboBox.setCurrentText(db["lr_solver_comboBox"])
            if "nn_weights_comboBox" in db:
                self.ui.nn_weights_comboBox.setCurrentText(db["nn_weights_comboBox"])
            if "lr_c_doubleSpinBox" in db:
                self.ui.lr_c_doubleSpinBox.setValue(db["lr_c_doubleSpinBox"])
            if "svc_nu_doubleSpinBox" in db:
                self.ui.svc_nu_doubleSpinBox.setValue(db["svc_nu_doubleSpinBox"])
            if "n_neighbors_spinBox" in db:
                self.ui.n_neighbors_spinBox.setValue(db["n_neighbors_spinBox"])
            if "lineEdit_lda_shrinkage" in db:
                self.ui.lineEdit_lda_shrinkage.setText(db["lineEdit_lda_shrinkage"])
            if "lda_solver_check_box" in db:
                self.ui.lda_solver_check_box.setChecked(db["lda_solver_check_box"])
            if "lda_shrinkage_check_box" in db:
                self.ui.lda_shrinkage_check_box.setChecked(
                    db["lda_shrinkage_check_box"]
                )
            if "lr_penalty_checkBox" in db:
                self.ui.lr_penalty_checkBox.setChecked(db["lr_penalty_checkBox"])
            if "lr_solver_checkBox" in db:
                self.ui.lr_solver_checkBox.setChecked(db["lr_solver_checkBox"])
            if "svc_nu_check_box" in db:
                self.ui.svc_nu_check_box.setChecked(db["svc_nu_check_box"])
            if "nn_weights_checkBox" in db:
                self.ui.nn_weights_checkBox.setChecked(db["nn_weights_checkBox"])
            if "n_neighbors_checkBox" in db:
                self.ui.n_neighbors_checkBox.setChecked(db["n_neighbors_checkBox"])
            if "activation_checkBox" in db:
                self.ui.activation_checkBox.setChecked(db["activation_checkBox"])
            if "criterion_checkBox" in db:
                self.ui.criterion_checkBox.setChecked(db["criterion_checkBox"])
            if "dt_max_depth_check_box" in db:
                self.ui.dt_max_depth_check_box.setChecked(db["dt_max_depth_check_box"])
            if "dt_min_samples_split_check_box" in db:
                self.ui.dt_min_samples_split_check_box.setChecked(
                    db["dt_min_samples_split_check_box"]
                )
            if "rf_criterion_checkBox" in db:
                self.ui.rf_criterion_checkBox.setChecked(db["rf_criterion_checkBox"])
            if "ab_n_estimators_checkBox" in db:
                self.ui.ab_n_estimators_checkBox.setChecked(
                    db["ab_n_estimators_checkBox"]
                )
            if "xgb_lambda_checkBox" in db:
                self.ui.xgb_lambda_checkBox.setChecked(db["xgb_lambda_checkBox"])
            if "xgb_colsample_bytree_checkBox" in db:
                self.ui.xgb_colsample_bytree_checkBox.setChecked(
                    db["xgb_colsample_bytree_checkBox"]
                )
            if "xgb_min_child_weight_checkBox" in db:
                self.ui.xgb_min_child_weight_checkBox.setChecked(
                    db["xgb_min_child_weight_checkBox"]
                )
            if "xgb_max_depth_checkBox" in db:
                self.ui.xgb_max_depth_checkBox.setChecked(db["xgb_max_depth_checkBox"])
            if "xgb_gamma_checkBox" in db:
                self.ui.xgb_gamma_checkBox.setChecked(db["xgb_gamma_checkBox"])
            if "xgb_eta_checkBox" in db:
                self.ui.xgb_eta_checkBox.setChecked(db["xgb_eta_checkBox"])
            if "ab_learning_rate_checkBox" in db:
                self.ui.ab_learning_rate_checkBox.setChecked(
                    db["ab_learning_rate_checkBox"]
                )
            if "rf_min_samples_split_checkBox" in db:
                self.ui.rf_min_samples_split_checkBox.setChecked(
                    db["rf_min_samples_split_checkBox"]
                )
            if "rf_n_estimators_checkBox" in db:
                self.ui.rf_n_estimators_checkBox.setChecked(
                    db["rf_n_estimators_checkBox"]
                )
            if "rf_max_features_checkBox" in db:
                self.ui.rf_max_features_checkBox.setChecked(
                    db["rf_max_features_checkBox"]
                )
            if "mlp_solve_checkBox" in db:
                self.ui.mlp_solve_checkBox.setChecked(db["mlp_solve_checkBox"])
            if "mlp_layer_size_checkBox" in db:
                self.ui.mlp_layer_size_checkBox.setChecked(
                    db["mlp_layer_size_checkBox"]
                )
            if "learning_rate_checkBox" in db:
                self.ui.learning_rate_checkBox.setChecked(db["learning_rate_checkBox"])
            if "lr_c_checkBox" in db:
                self.ui.lr_c_checkBox.setChecked(db["lr_c_checkBox"])
            if "fit_method" in db:
                self.ui.fit_opt_method_comboBox.setCurrentText(db["fit_method"])
            if "data_style" in db:
                self.fitting.data_style.clear()
                self.fitting.data_style = db["data_style"].copy()
                self.data_style_button_style_sheet(
                    self.fitting.data_style["color"].name()
                )
            if "data_curve_checked" in db:
                self.ui.data_checkBox.setChecked(db["data_curve_checked"])
            if "sum_style" in db:
                self.fitting.sum_style.clear()
                self.fitting.sum_style = db["sum_style"].copy()
                self.sum_style_button_style_sheet(
                    self.fitting.sum_style["color"].name()
                )
            if "sigma3_style" in db:
                self.fitting.sigma3_style.clear()
                self.fitting.sigma3_style = db["sigma3_style"].copy()
                self.sigma3_style_button_style_sheet(
                    self.fitting.sigma3_style["color"].name()
                )
                pen, brush = curve_pen_brush_by_style(self.fitting.sigma3_style)
                self.fitting.sigma3_fill.setPen(pen)
                self.fitting.sigma3_fill.setBrush(brush)
            if "params_stderr" in db:
                self.fitting.params_stderr = db["params_stderr"].copy()
            if "sum_curve_checked" in db:
                self.ui.sum_checkBox.setChecked(db["sum_curve_checked"])
            if "sigma3_checked" in db:
                self.ui.sigma3_checkBox.setChecked(db["sigma3_checked"])
            if "residual_style" in db:
                self.fitting.residual_style.clear()
                self.fitting.residual_style = db["residual_style"].copy()
                self.residual_style_button_style_sheet(
                    self.fitting.residual_style["color"].name()
                )
            if "residual_curve_checked" in db:
                self.ui.residual_checkBox.setChecked(db["residual_curve_checked"])
            if "interval_checkBox_checked" in db:
                self.ui.interval_checkBox.setChecked(db["interval_checkBox_checked"])
            if "use_fit_intervals" in db:
                self.ui.intervals_gb.setChecked(db["use_fit_intervals"])
            if "use_shapley_cb" in db:
                self.ui.use_shapley_cb.setChecked(db["use_shapley_cb"])
            if "random_state_cb" in db:
                self.ui.random_state_cb.setChecked(db["random_state_cb"])
            if "random_state_sb" in db:
                self.ui.random_state_sb.setValue(db["random_state_sb"])
            if "max_dx_guess" in db:
                self.ui.max_dx_dsb.setValue(db["max_dx_guess"])
            if "latest_stat_result" in db:
                try:
                    self.stat_analysis_logic.latest_stat_result = db[
                        "latest_stat_result"
                    ]
                except AttributeError as err:
                    print(err)
                except ModuleNotFoundError:
                    pass
            if "activation_comboBox" in db:
                self.ui.activation_comboBox.setCurrentText(db["activation_comboBox"])
            if "criterion_comboBox" in db:
                self.ui.criterion_comboBox.setCurrentText(db["criterion_comboBox"])
            if "rf_criterion_comboBox" in db:
                self.ui.rf_criterion_comboBox.setCurrentText(
                    db["rf_criterion_comboBox"]
                )
            if "rf_max_features_comboBox" in db:
                self.ui.rf_max_features_comboBox.setCurrentText(
                    db["rf_max_features_comboBox"]
                )
            if "solver_mlp_combo_box" in db:
                self.ui.solver_mlp_combo_box.setCurrentText(db["solver_mlp_combo_box"])
            if "mlp_layer_size_spinBox" in db:
                self.ui.mlp_layer_size_spinBox.setValue(db["mlp_layer_size_spinBox"])
            if "max_epoch_spinBox" in db:
                self.ui.max_epoch_spinBox.setValue(db["max_epoch_spinBox"])
            if "dt_max_depth_spin_box" in db:
                self.ui.dt_max_depth_spin_box.setValue(db["dt_max_depth_spin_box"])
            if "dt_min_samples_split_spin_box" in db:
                self.ui.dt_min_samples_split_spin_box.setValue(
                    db["dt_min_samples_split_spin_box"]
                )
            if "rf_min_samples_split_spinBox" in db:
                self.ui.rf_min_samples_split_spinBox.setValue(
                    db["rf_min_samples_split_spinBox"]
                )
            if "ab_learning_rate_doubleSpinBox" in db:
                self.ui.ab_learning_rate_doubleSpinBox.setValue(
                    db["ab_learning_rate_doubleSpinBox"]
                )
            if "xgb_n_estimators_spinBox" in db:
                self.ui.xgb_n_estimators_spinBox.setValue(
                    db["xgb_n_estimators_spinBox"]
                )
            if "xgb_lambda_doubleSpinBox" in db:
                self.ui.xgb_lambda_doubleSpinBox.setValue(
                    db["xgb_lambda_doubleSpinBox"]
                )
            if "xgb_colsample_bytree_doubleSpinBox" in db:
                self.ui.xgb_colsample_bytree_doubleSpinBox.setValue(
                    db["xgb_colsample_bytree_doubleSpinBox"]
                )
            if "xgb_min_child_weight_spinBox" in db:
                self.ui.xgb_min_child_weight_spinBox.setValue(
                    db["xgb_min_child_weight_spinBox"]
                )
            if "xgb_max_depth_spinBox" in db:
                self.ui.xgb_max_depth_spinBox.setValue(db["xgb_max_depth_spinBox"])
            if "xgb_gamma_spinBox" in db:
                self.ui.xgb_gamma_spinBox.setValue(db["xgb_gamma_spinBox"])
            if "xgb_eta_doubleSpinBox" in db:
                self.ui.xgb_eta_doubleSpinBox.setValue(db["xgb_eta_doubleSpinBox"])
            if "ab_n_estimators_spinBox" in db:
                self.ui.ab_n_estimators_spinBox.setValue(db["ab_n_estimators_spinBox"])
            if "rf_n_estimators_spinBox" in db:
                self.ui.rf_n_estimators_spinBox.setValue(db["rf_n_estimators_spinBox"])
            if "learning_rate_doubleSpinBox" in db:
                self.ui.learning_rate_doubleSpinBox.setValue(
                    db["learning_rate_doubleSpinBox"]
                )
            if "refit_score" in db:
                self.ui.refit_score.setCurrentText(db["refit_score"])
            if "feature_display_max_checkBox" in db:
                self.ui.feature_display_max_checkBox.setChecked(
                    db["feature_display_max_checkBox"]
                )
            if "include_x0_checkBox" in db:
                self.ui.include_x0_checkBox.setChecked(db["include_x0_checkBox"])
            if "feature_display_max_spinBox" in db:
                self.ui.feature_display_max_spinBox.setValue(
                    db["feature_display_max_spinBox"]
                )
            if "use_pca_checkBox" in db:
                self.ui.use_pca_checkBox.setChecked(db["use_pca_checkBox"])
            if "intervals_data" in db:
                self.fitting.intervals_data = db["intervals_data"]
            if "all_ranges_clustered_lines_x0" in db:
                self.fitting.all_ranges_clustered_x0_sd = db[
                    "all_ranges_clustered_lines_x0"
                ]

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
            await self.fitting.draw_all_curves()

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

    def set_buttons_ability(self) -> None:
        # TODO переделать
        pass
        # self.action_despike.setDisabled(len(self.ImportedArray) == 0)
        # self.action_convert.setDisabled(len(self.ImportedArray) == 0)
        # self.action_cut.setDisabled(len(self.preprocessing.ConvertedDict) == 0)
        # self.action_normalize.setDisabled(len(self.preprocessing.CuttedFirstDict) == 0)
        # self.action_smooth.setDisabled(len(self.preprocessing.CuttedFirstDict) == 0)
        # self.action_baseline_correction.setDisabled(len(self.preprocessing.smoothed_spectra) == 0)
        # self.action_trim.setDisabled(len(self.preprocessing.baseline_corrected_not_trimmed_dict) == 0)
        # self.action_average.setDisabled(len(self.preprocessing.baseline_corrected_not_trimmed_dict) == 0)

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

    def disable_buttons(self, b: bool) -> None:
        self.set_buttons_ability()
        self.ui.EditBtn.setDisabled(b)
        self.ui.FileBtn.setDisabled(b)
        self.ui.ProcessBtn.setDisabled(b)
        self.ui.gt_add_Btn.setDisabled(b)
        self.ui.gt_dlt_Btn.setDisabled(b)
        self.ui.input_table.setDisabled(b)
        self.ui.updateTrimRangebtn.setDisabled(b)

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

    def open_demo_project(self) -> None:
        """
        1. Open Demo project
        Returns
        -------
        None
        """
        path = "examples/demo_project.zip"
        self.open_project(path)
        self.load_params(path)

    # endregion

    # region CONVERT
    def update_cm_min_max_range(self) -> None:
        if not self.preprocessing.ConvertedDict:
            return
        first_x = []
        last_x = []
        for v in self.preprocessing.ConvertedDict.values():
            x_axis = v[:, 0]
            first_x.append(x_axis[0])
            last_x.append(x_axis[-1])
        min_cm = max(first_x)
        max_cm = min(last_x)

        self.ui.trim_start_cm.setMinimum(min_cm)
        self.ui.trim_start_cm.setMaximum(max_cm)
        self.ui.trim_end_cm.setMinimum(min_cm)
        self.ui.trim_end_cm.setMaximum(max_cm)

        current_value_trim_start = self.ui.trim_start_cm.value()
        current_value_trim_end = self.ui.trim_end_cm.value()
        if current_value_trim_start < min_cm or current_value_trim_start > max_cm:
            self.ui.trim_start_cm.setValue(min_cm)
        if current_value_trim_end < min_cm or current_value_trim_end > max_cm:
            self.ui.trim_end_cm.setValue(max_cm)
        self.linearRegionDeconv.setBounds((min_cm, max_cm))


    # endregion

    # region Fitting page 2

    # region Add line
    @asyncSlot()
    async def add_deconv_line(self, line_type: str):
        stage = self.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if not self.fitting.is_template:
            msg = MessageBox(
                "Add line failed.",
                "Switch to Template mode to add new line",
                self,
                {"Ok"},
            )
            msg.exec()
            return
        elif not data:
            msg = MessageBox(
                "Add line failed.", "No baseline corrected spectrum", self, {"Ok"}
            )
            msg.exec()
            return
        try:
            await self.do_add_deconv_line(line_type)
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def do_add_deconv_line(self, line_type: str) -> None:
        idx = self.ui.deconv_lines_table.model().free_index()
        command = CommandAddDeconvLine(self, idx, line_type, "Add " + line_type)
        self.undoStack.push(command)

    # endregion

    @asyncSlot()
    async def batch_fit(self) -> None:
        """
        Check conditions when Fit button pressed, if all ok - go do_batch_fit
        For fitting must be more than 0 lines to fit

        """
        stage = self.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if self.ui.deconv_lines_table.model().rowCount() == 0:
            msg = MessageBox(
                "Fitting failed.", "Add some new lines before fitting", self, {"Ok"}
            )
            msg.exec()
            return
        elif (
                not data
                or len(data) == 0
        ):
            MessageBox(
                "Fitting failed.", "There is No any data to fit", self, {"Ok"}
            ).exec()
            return
        try:
            await self.fitting.do_batch_fit()
        except Exception as err:
            self.show_error(err)

    def set_deconvoluted_dataset(self) -> None:
        if (
                self.ui.input_table.model().rowCount() == 0
                or self.ui.fit_params_table.model().batch_unfitted()
        ):
            return
        df = self.fitting.create_deconvoluted_dataset_new()
        self.ui.deconvoluted_dataset_table_view.model().set_dataframe(df)
        self._init_current_filename_combobox()
        self.fitting.update_ignore_features_table()

    @asyncSlot()
    async def fit(self):
        """
        Check conditions when Fit button pressed, if all ok - go do_fit
        For fitting must be more than 0 lines to fit and current spectrum in plot must also be
        Должна быть хотя бы 1 линия и выделен текущий спектр
        """
        if self.ui.deconv_lines_table.model().rowCount() == 0:
            MessageBox(
                "Fitting failed.", "Add some new lines before fitting", self, {"Ok"}
            ).exec()
            return
        elif self.fitting.array_of_current_filename_in_deconvolution is None:
            MessageBox(
                "Fitting failed.", "There is No any data to fit", self, {"Ok"}
            ).exec()
            return
        try:
            await self.fitting.do_fit()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def guess(self, line_type: str) -> None:
        """
        Auto guess lines, finds number of lines and positions x0
        """
        stage = self.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if not data:
            MessageBox(
                "Guess failed.",
                "Do baseline correction before guessing peaks",
                self,
                {"Ok"},
            ).exec()
            return
        if self.ui.interval_checkBox.isChecked() and self.ui.intervals_gb.isChecked():
            MessageBox(
                "Guess failed.",
                "'Split spectrum' and 'Interval' both active. Leave only one of them turned on",
                self,
                {"Ok"},
            ).exec()
            return
        if (
                self.ui.intervals_gb.isChecked()
                and self.ui.fit_borders_TableView.model().rowCount() == 0
        ):
            MessageBox(
                "Guess failed.",
                "If 'Split spectrum' table is active, you must fill the table",
                self,
                {"Ok"},
            ).exec()
            return
        if self.ui.guess_method_cb.currentText() == "Average groups" and (
                not self.context.preprocessing.stages.av_data.data
                or len(self.context.preprocessing.stages.av_data.data) < 2
        ):
            MessageBox(
                "Guess failed.",
                "Если выбран метод 'Average groups', то должно быть больше 1 группы",
                self,
                {"Ok"},
            ).exec()
            return
        if (
                not self.ui.interval_checkBox.isChecked()
                and not self.ui.intervals_gb.isChecked()
        ):
            msg = MessageBox(
                "Warning!",
                "Авто определение линий в полном диапазоне может занимать очень много "
                "времени" + "\n" + "Точно продолжить?",
                self,
                {"Yes", "No", "Cancel"},
            )
            msg.setInformativeText(
                "Разделение спектра на диапазоны может сократить время анализа на 2-3 порядка. "
                "А без разделения вычисление может идти часами"
            )
            if self.project_path:
                msg.setInformativeText(self.project_path)
            result = msg.exec()
            if result == 1:
                msg = MessageBox(
                    "Warning!",
                    "Последний шанс передумать" + "\n" + "Точно продолжить?",
                    self,
                    {"Yes", "No", "Cancel"},
                )
                if not msg.exec() == 1:
                    return
            else:
                return
        try:
            await self.fitting.do_auto_guess(line_type)
        except Exception as err:
            self.show_error(err)

    def _update_deconv_curve_style(
            self, style: dict, old_style: dict, index: int
    ) -> None:
        command = CommandUpdateDeconvCurveStyle(
            self, index, style, old_style, "Update style for curve idx %s" % index
        )
        self.undoStack.push(command)

    def clear_all_deconv_lines(self) -> None:
        if self.fitting.timer_fill is not None:
            self.fitting.timer_fill.stop()
            self.fitting.timer_fill = None
            self.fitting.updating_fill_curve_idx = None
        command = CommandClearAllDeconvLines(self, "Remove all deconvolution lines")
        self.undoStack.push(command)

    def curve_parameter_changed(
            self, value: float, line_index: int, param_name: str
    ) -> None:
        self.fitting.CommandDeconvLineDraggedAllowed = True
        items_matches = self.fitting.deconvolution_data_items_by_idx(line_index)
        if items_matches is None:
            return
        curve, roi = items_matches
        line_type = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(
            line_index, "Type"
        )
        if param_name == "x0":
            new_pos = QPointF()
            new_pos.setX(value)
            new_pos.setY(roi.pos().y())
            roi.setPos(new_pos)
        elif param_name == "a":
            set_roi_size(roi.size().x(), value, roi, [0, 0])
        elif param_name == "dx":
            set_roi_size(value, roi.size().y(), roi)
        params = {
            "a": value if param_name == "a" else roi.size().y(),
            "x0": roi.pos().x(),
            "dx": roi.size().x(),
        }
        filename = (
            ""
            if self.fitting.is_template
            else self.fitting.current_spectrum_deconvolution_name
        )
        model = self.ui.fit_params_table.model()
        if "add_params" not in self.fitting.peak_shapes_params[line_type]:
            self.fitting.redraw_curve(params, curve, line_type)
            return
        add_params = self.fitting.peak_shapes_params[line_type]["add_params"]
        for s in add_params:
            if param_name == s:
                param = value
            else:
                param = model.get_parameter_value(filename, line_index, s, "Value")
            params[s] = param
        self.fitting.redraw_curve(params, curve, line_type)

    # endregion

    # region Stat analysis (machine learning) page4
    @asyncSlot()
    async def fit_classificator(self, cl_type=None):
        """
        Проверяем что dataset заполнен.
        Переходим в процедуру создания классификатора
        """
        current_dataset = self.ui.dataset_type_cb.currentText()
        if (
                current_dataset == "Smoothed"
                and self.ui.smoothed_dataset_table_view.model().rowCount() == 0
                or current_dataset == "Baseline corrected"
                and self.ui.baselined_dataset_table_view.model().rowCount() == 0
                or current_dataset == "Decomposed"
                and self.ui.deconvoluted_dataset_table_view.model().rowCount() == 0
        ):
            MessageBox(
                "Classificator Fitting failed.",
                "Нет данных для обучения классификатора",
                self,
                {"Ok"},
            )
            return
        if not cl_type:
            cl_type = self.ui.current_classificator_comboBox.currentText()
        try:
            await self.stat_analysis_logic.do_fit_classificator(cl_type)
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def redraw_stat_plots(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("once")
            self.stat_analysis_logic.update_stat_report_text(
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
            await self.loop.run_in_executor(
                None, self.stat_analysis_logic.update_plsda_plots
            )

    def current_tree_sb_changed(self, idx: int) -> None:
        if (
                self.ui.current_classificator_comboBox.currentText() == "Random Forest"
                and "Random Forest" in self.stat_analysis_logic.latest_stat_result
        ):
            model_results = self.stat_analysis_logic.latest_stat_result["Random Forest"]
            model = model_results["model"]
            self.stat_analysis_logic.update_plot_tree(
                model.best_estimator_.estimators_[idx],
                model_results["feature_names"],
                model_results["target_names"],
            )
        elif (
                self.ui.current_classificator_comboBox.currentText() == "XGBoost"
                and "XGBoost" in self.stat_analysis_logic.latest_stat_result
        ):
            model_results = self.stat_analysis_logic.latest_stat_result["XGBoost"]
            model = model_results["model"]
            self.stat_analysis_logic.update_xgboost_tree_plot(
                model.best_estimator_, idx
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
                "SHAP plots refresh error.",
                "Selected classificator is not fitted.",
                self,
                {"Ok"},
            )
            msg.setInformativeText(
                "Try to turn on Use Shapley option before fit classificator."
            )
            msg.exec()
            return
        await self.loop.run_in_executor(None, self._update_shap_plots)
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
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
