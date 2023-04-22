import asyncio
import copy
import re
import sys
from asyncio import create_task, gather, sleep, wait
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from gc import get_objects, collect
from logging import basicConfig, critical, error, DEBUG, info
from os import environ, startfile
from pathlib import Path
from shelve import open as shelve_open
from threading import Event
from traceback import format_exc, format_exception
from zipfile import ZipFile, ZIP_DEFLATED

import lmfit.model
import matplotlib.pyplot as plt
import numpy as np
import pyperclip
import shap
import winsound
from asyncqtpy import asyncSlot, QEventLoop
from lmfit import Parameters, Model
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pandas import DataFrame, MultiIndex, ExcelWriter, Series, concat
from psutil import cpu_percent
from pyqtgraph import setConfigOption, PlotDataItem, PlotCurveItem, ROI, SignalProxy, InfiniteLine, LinearRegionItem, \
    mkPen, mkBrush, FillBetweenItem, BarGraphItem, ArrowItem
from qtpy.QtCore import Qt, QPoint, QEvent, QModelIndex, QTimer, QMarginsF, QAbstractItemModel
from qtpy.QtGui import QPixmap, QFont, QIcon, QCloseEvent, QMouseEvent, QKeyEvent, QContextMenuEvent, QEnterEvent, \
    QMoveEvent, QColor, QPageLayout, QPageSize
from qtpy.QtWidgets import QMessageBox, QApplication, QSplashScreen, QMainWindow, QColorDialog, QUndoStack, QMenu, \
    QAction, QHeaderView, QAbstractItemView, QLabel, QProgressBar, QProgressDialog, QPushButton, QFileDialog, \
    QLineEdit, QInputDialog, QTableView, QScrollArea
from qtpy.QtWinExtras import QWinTaskbarButton
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc, PrecisionRecallDisplay, \
    precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree

from modules.classificators import clf_predict
from modules.customClasses import MultiLine, PandasModelGroupsTable, PandasModelInputTable, PandasModelDeconvTable, \
    PandasModelDeconvLinesTable, ComboDelegate, CurvePropertiesWindow, DialogListBox, SettingsDialog, \
    PandasModelFitParamsTable, DoubleSpinBoxDelegate, PandasModelFitIntervals, IntervalsTableDelegate, \
    PandasModelSmoothedDataset, PandasModelBaselinedDataset, PandasModelDeconvolutedDataset, PandasModel, \
    PandasModelPredictTable
from modules.default_values import default_values, peak_shapes_params, peak_shape_params_limits, fitting_methods, \
    baseline_methods, baseline_parameter_defaults, optimize_extended_range_methods, smoothing_methods, \
    normalize_methods, classificator_funcs
from modules.gui import Ui_MainWindow
from modules.init import QtStyleTools, get_theme, opacity
from modules.normalize_functions import get_emsc_average_spectrum
from modules.spec_functions import check_preferences_file, read_preferences, random_rgb, check_recent_files, \
    get_memory_used, check_rs_tool_folder, import_spectrum, curve_pen_brush_by_style, convert, \
    find_fluorescence_beginning, interpolate, find_nearest_idx, find_nearest, find_nearest_by_idx, \
    subtract_cosmic_spikes_moll, find_first_right_local_minimum, find_first_left_local_minimum, cut_spectrum, \
    cut_full_spectrum, cut_axis, get_average_spectrum, set_roi_size_pos, set_roi_size, get_curve_for_deconvolution, \
    packed_current_line_parameters, fitting_model, fit_model, fit_model_batch, guess_peaks, update_fit_parameters, \
    split_by_borders, models_params_splitted, models_params_splitted_batch, find_interval_key, \
    process_data_by_intervals, legend_by_float, process_wavenumbers_interval, eval_uncert, insert_table_to_text_edit
from modules.undo_classes import CommandImportFiles, CommandAddGroup, CommandDeleteGroup, CommandChangeGroupCell, \
    CommandChangeGroupCellsBatch, CommandDeleteInputSpectrum, CommandChangeGroupStyle, CommandUpdateInterpolated, \
    CommandUpdateDespike, CommandConvert, CommandCutFirst, CommandNormalize, CommandSmooth, CommandBaselineCorrection, \
    CommandTrim, CommandAddDeconvLine, CommandDeleteDeconvLines, CommandDeconvLineTypeChanged, \
    CommandUpdateDeconvCurveStyle, CommandUpdateDataCurveStyle, CommandDeconvLineDragged, \
    CommandClearAllDeconvLines, CommandStartIntervalChanged, CommandEndIntervalChanged, CommandAfterFitting, \
    CommandAfterBatchFitting, CommandAfterGuess, CommandFitIntervalAdded, CommandFitIntervalDeleted, \
    CommandAfterFittingStat

plt.style.use(['dark_background'])
plt.set_loglevel("info")
shap.initjs()
environ['OPENBLAS_NUM_THREADS'] = '1'


# noinspection PyUnresolvedReferences,PyTypeChecker
class RuntimeStylesheets(QMainWindow, QtStyleTools):

    def __init__(self, event_loop) -> None:
        super().__init__()
        self.plt_style = None
        self.stat_models = {}
        self.interp_ref_array = None
        self.is_production_project = False
        self.top_features = {}
        self.latest_stat_result = {}
        self.taskbar_progress = None
        self.taskbar_button = None
        self.updating_fill_curve_idx = None
        self.timer_fill = None
        self.rad = None
        self.lda_1d_inf_lines = []
        self.current_futures = []
        self.baseline_method = None
        self.smooth_method = None
        self.normalization_method = None
        self.residual_style = None
        self.data_style = None
        self.auto_save_timer = None
        self.cpu_load = None
        self.timer_mem_update = None
        self.sum_style = None
        self.sigma3_style = None
        self.action_trim = None
        self.action_baseline_correction = None
        self.action_smooth = None
        self.action_normalize = None
        self.action_normalize = None
        self.action_cut = None
        self.action_convert = None
        self.action_despike = None
        self.action_interpolate = None
        self.process_menu = None
        self.clear_menu = None
        self.action_redo = None
        self.action_undo = None
        self.edit_menu = None
        self.file_menu_save_as_action = None
        self.file_menu_save_all_action = None
        self.export_menu = None
        self.file_menu_import_action = None
        self.recent_menu = None
        self.file_menu = None
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing main window...', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self.theme_bckgrnd = theme_bckgrnd
        self.theme_color = theme_color
        self.plot_font_size = plot_font_size
        self.axis_label_font_size = axis_label_font_size
        self.isTemplate = False
        self._y_axis_ref_EMSC = None
        self.current_executor = None
        self.curveBaseline = None
        self.curveOneCutPlot = None
        self.curveOneConvertPlot = None
        self.curveOneInputPlot = None
        self.curveOneNormalPlot = None
        self.curveOneSmoothPlot = None
        self.curve_one_baseline_plot = None
        self.data_curve = None
        self.sum_curve = None
        self.residual_curve = None
        self.curveDespikedHistory = None
        self.beforeTime = None
        self.currentProgress = None
        self.task_mem_update = None
        self.current_spectrum_despiked_name = None
        self.current_spectrum_baseline_name = None
        self.export_folder_path = None
        self.loop = event_loop or get_event_loop()
        self.progressBar = None
        self.previous_group_of_item = None
        self.plot_text_color_value = None
        self.plot_text_color = None
        self.plot_background_color = None
        self.plot_background_color_web = None
        self.averaged_array = None
        self.dragged_line_parameters = None
        self.prev_dragged_line_parameters = None
        # parameters to turn on/off pushing UndoStack command during another redo/undo command executing
        self.CommandDeconvLineDraggedAllowed = True
        self.CommandStartIntervalChanged_allowed = True
        self.CommandEndIntervalChanged_allowed = True

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowFrameSection.TopSection |
                            Qt.WindowType.WindowMinMaxButtonsHint)
        self.current_spectrum_deconvolution_name = ''
        self.ImportedArray = dict()
        self.BeforeDespike = dict()
        self.ConvertedDict = dict()
        self.CuttedFirstDict = dict()
        self.NormalizedDict = dict()
        self.SmoothedDict = dict()
        self.baseline_corrected_dict = dict()
        self.baseline_corrected_not_trimmed_dict = dict()
        self.baseline_dict = dict()
        self.averaged_dict = dict()
        self.report_result = dict()
        self.sigma3 = dict()
        self.project_path = None
        self.recent_limit = recent_limit
        self.auto_save_minutes = int(auto_save_minutes)
        self.dragPos = None
        self.keyPressEvent = self.key_press_event
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.set_extra(extra)
        self.modified = False
        self.window_maximized = True
        self._ascending_input_table = False
        self._ascending_deconv_lines_table = False
        self.laser_wl_value_old = self.ui.laser_wl_spinbox.value()
        self.theme_colors = get_theme(theme)
        self.taskbar_button = QWinTaskbarButton()
        self.taskbar_button.setWindow(self.windowHandle())
        self._init_default_values()

        # UNDO/ REDO
        self.undoStack = QUndoStack(self)
        self.undoStack.setUndoLimit(int(undo_limit))

        # SET UI DEFINITIONS
        self.setWindowIcon(QIcon(logo))
        self.setWindowTitle('Raman Spectroscopy Tool ')

        self.plot_text_color_value = self.theme_colors['plotText']
        self.plot_text_color = QColor(self.plot_text_color_value)
        self.plot_background_color = QColor(self.theme_colors['plotBackground'])
        self.plot_background_color_web = QColor(self.theme_colors['backgroundMainColor'])
        self.update_icons()
        basicConfig(level=DEBUG, filename='log.log', filemode='w',
                    format="%(asctime)s %(levelname)s %(message)s")
        info('Logging started.')
        self.initial_ui_definitions()
        try:
            check_recent_files()
        except Exception as err:
            self.show_error(err)
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing menu...', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self.initial_menu()
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing left side menu...', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self._init_left_menu()
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing tables...', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self._initial_all_tables()
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing scrollbar...', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self.initial_right_scrollbar()
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing figures...', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self._initial_plots()
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing plot buttons...', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self.initial_plot_buttons()
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing fitting right side frame...',
                           Qt.AlignmentFlag.AlignBottom, splash_color)
        self._initial_guess_table_frame()
        self.initial_timers()
        self.set_buttons_ability()
        self.ui.stat_tab_widget.currentChanged.connect(self.stat_tab_widget_tab_changed)
        self.ui.page5_predict.clicked.connect(self.predict)
        self.ui.splitter_page1.moveSplitter(100, 1)
        self.ui.splitter_page2.moveSplitter(100, 1)
        self.memory_usage_label = QLabel(self)
        # self.init_cuda()
        self.statusBar().addPermanentWidget(self.memory_usage_label)

        self.ui.baseline_correction_method_comboBox.setCurrentText(self.default_values['baseline_method_comboBox'])
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Setting default parameters', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        self._set_parameters_to_default()
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Checking RS-tool folder', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        try:
            check_rs_tool_folder()
        except Exception as err:
            self.show_error(err)
        self.set_modified(False)
        self.ui.statusBar.showMessage('Ready', 2000)
        splash.showMessage('ver. 1.0.00 ' + '\n' + 'Initializing finished', Qt.AlignmentFlag.AlignBottom,
                           splash_color)
        splash.finish(self)

    def closeEvent(self, a0: QCloseEvent) -> None:
        if self.modified:
            msg = QMessageBox(QMessageBox.Icon.Question, "Close", 'You have unsaved changes. '
                              + '\n' + 'Save changes before exit?', QMessageBox.StandardButton.Yes
                              | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if self.project_path:
                msg.setInformativeText(self.project_path)
            result = msg.exec()
            if result == QMessageBox.StandardButton.Yes:
                self.action_save_project()
            elif result == QMessageBox.StandardButton.Cancel:
                a0.ignore()
                return

        self.input_plot_widget_plot_item.close()
        self.converted_cm_widget_plot_item.close()
        self.cut_cm_plotItem.close()
        self.normalize_plotItem.close()
        self.smooth_plotItem.close()
        self.baseline_corrected_plotItem.close()
        self.averaged_plotItem.close()
        del self.ui.input_plot_widget.vertical_line
        del self.ui.input_plot_widget.horizontal_line
        del self.ui.converted_cm_plot_widget.vertical_line
        del self.ui.converted_cm_plot_widget.horizontal_line
        del self.ui.cut_cm_plot_widget.vertical_line
        del self.ui.cut_cm_plot_widget.horizontal_line
        del self.ui.normalize_plot_widget.vertical_line
        del self.ui.normalize_plot_widget.horizontal_line
        del self.ui.smooth_plot_widget.vertical_line
        del self.ui.smooth_plot_widget.horizontal_line
        del self.ui.baseline_plot_widget.vertical_line
        del self.ui.baseline_plot_widget.horizontal_line
        del self.ui.average_plot_widget.vertical_line
        del self.ui.average_plot_widget.horizontal_line
        sys.exit()

    # region init

    def _init_default_values(self) -> None:
        """
        Initialize dict default values from modules.default_values
        @return: None
        """
        self.default_values = default_values()
        self.old_start_interval_value = self.default_values['interval_start']
        self.old_end_interval_value = self.default_values['interval_end']
        self.peak_shapes_params = peak_shapes_params()
        self.peak_shape_params_limits = peak_shape_params_limits()
        self.data_style = {'color': QColor(self.theme_colors['secondaryColor']),
                           'style': Qt.PenStyle.SolidLine,
                           'width': 1.0,
                           'fill': False,
                           'use_line_color': True,
                           'fill_color': QColor().fromRgb(random_rgb()),
                           'fill_opacity': 0.0}
        self.data_style_button_style_sheet(self.data_style['color'].name())
        self.sum_style = {'color': QColor(self.theme_colors['primaryColor']),
                          'style': Qt.PenStyle.DashLine,
                          'width': 1.0,
                          'fill': False,
                          'use_line_color': True,
                          'fill_color': QColor().fromRgb(random_rgb()),
                          'fill_opacity': 0.0}
        self.sum_style_button_style_sheet(self.sum_style['color'].name())
        self.sigma3_style = {'color': QColor(self.theme_colors['primaryDarkColor']),
                             'style': Qt.PenStyle.SolidLine,
                             'width': 1.0,
                             'fill': True,
                             'use_line_color': True,
                             'fill_color': QColor().fromRgb(random_rgb()),
                             'fill_opacity': 0.25}
        self.sigma3_style_button_style_sheet(self.sigma3_style['color'].name())
        self.residual_style = {'color': QColor(self.theme_colors['secondaryLightColor']),
                               'style': Qt.PenStyle.DotLine,
                               'width': 1.0,
                               'fill': False,
                               'use_line_color': True,
                               'fill_color': QColor().fromRgb(random_rgb()),
                               'fill_opacity': 0.0}
        self.residual_style_button_style_sheet(self.residual_style['color'].name())
        self.fitting_methods = fitting_methods()
        self.baseline_methods = baseline_methods()
        self.baseline_parameter_defaults = baseline_parameter_defaults()
        self.smoothing_methods = smoothing_methods()
        self.normalize_methods = normalize_methods()
        self.classificator_funcs = classificator_funcs()

    def _set_parameters_to_default(self) -> None:
        self.ui.cm_range_start.setValue(self.default_values['cm_range_start'])
        self.ui.cm_range_end.setValue(self.default_values['cm_range_end'])
        self.ui.trim_start_cm.setValue(self.default_values['trim_start_cm'])
        self.ui.trim_end_cm.setValue(self.default_values['trim_end_cm'])
        self.ui.maxima_count_despike_spin_box.setValue(self.default_values['maxima_count_despike'])
        self.ui.laser_wl_spinbox.setValue(self.default_values['laser_wl'])
        self.ui.despike_fwhm_width_doubleSpinBox.setValue(self.default_values['despike_fwhm'])
        self.ui.neg_grad_factor_spinBox.setValue(self.default_values['neg_grad_factor_spinBox'])
        self.ui.normalizing_method_comboBox.setCurrentText(self.default_values['normalizing_method_comboBox'])
        self.ui.smoothing_method_comboBox.setCurrentText(self.default_values['smoothing_method_comboBox'])
        self.ui.guess_method_cb.setCurrentText(self.default_values['guess_method_cb'])
        self.ui.average_method_cb.setCurrentText(self.default_values['average_function'])
        self.ui.dataset_type_cb.setCurrentText(self.default_values['dataset_type_cb'])
        self.ui.classes_lineEdit.setText('')
        self.ui.test_data_ratio_spinBox.setValue(self.default_values['test_data_ratio_spinBox'])
        self.ui.n_lines_detect_method_cb.setCurrentText(self.default_values['n_lines_method'])
        self.ui.max_noise_level_dsb.setValue(self.default_values['max_noise_level'])
        self.ui.baseline_correction_method_comboBox.setCurrentText(self.default_values['baseline_method_comboBox'])
        self.ui.window_length_spinBox.setValue(self.default_values['window_length_spinBox'])
        self.ui.smooth_polyorder_spinBox.setValue(self.default_values['smooth_polyorder_spinBox'])
        self.ui.whittaker_lambda_spinBox.setValue(self.default_values['whittaker_lambda_spinBox'])
        self.ui.kaiser_beta_doubleSpinBox.setValue(self.default_values['kaiser_beta'])
        self.ui.emd_noise_modes_spinBox.setValue(self.default_values['EMD_noise_modes'])
        self.ui.eemd_trials_spinBox.setValue(self.default_values['EEMD_trials'])
        self.ui.sigma_spinBox.setValue(self.default_values['sigma'])
        self.ui.lambda_spinBox.setValue(self.default_values['lambda_spinBox'])
        self.ui.p_doubleSpinBox.setValue(self.default_values['p_doubleSpinBox'])
        self.ui.eta_doubleSpinBox.setValue(self.default_values['eta'])
        self.ui.n_iterations_spinBox.setValue(self.default_values['N_iterations'])
        self.ui.polynome_degree_spinBox.setValue(self.default_values['polynome_degree'])
        self.ui.grad_doubleSpinBox.setValue(self.default_values['grad'])
        self.ui.quantile_doubleSpinBox.setValue(self.default_values['quantile'])
        self.ui.alpha_factor_doubleSpinBox.setValue(self.default_values['alpha_factor'])
        self.ui.peak_ratio_doubleSpinBox.setValue(self.default_values['peak_ratio'])
        self.ui.spline_degree_spinBox.setValue(self.default_values['spline_degree'])
        self.ui.num_std_doubleSpinBox.setValue(self.default_values['num_std'])
        self.ui.interp_half_window_spinBox.setValue(self.default_values['interp_half_window'])
        self.ui.sections_spinBox.setValue(self.default_values['sections'])
        self.ui.min_length_spinBox.setValue(self.default_values['min_length'])
        self.ui.fill_half_window_spinBox.setValue(self.default_values['fill_half_window'])
        self.ui.scale_doubleSpinBox.setValue(self.default_values['scale'])
        self.ui.fraction_doubleSpinBox.setValue(self.default_values['fraction'])
        self.ui.cost_func_comboBox.setCurrentText(self.default_values['cost_function'])
        self.ui.opt_method_oer_comboBox.setCurrentText(self.default_values['opt_method_oer'])
        self.ui.fit_opt_method_comboBox.setCurrentText(self.default_values['fit_method'])
        self.ui.intervals_gb.setChecked(False)
        self.ui.emsc_pca_n_spinBox.setValue(self.default_values['EMSC_N_PCA'])
        self.ui.max_CCD_value_spinBox.setValue(self.default_values['max_CCD_value'])
        self.ui.interval_start_dsb.setValue(self.default_values['interval_start'])
        self.ui.interval_end_dsb.setMaximum(99_999.0)
        self.ui.interval_end_dsb.setValue(self.default_values['interval_end'])
        self.ui.max_dx_dsb.setValue(self.default_values['max_dx_guess'])

    # region plots

    def _initial_input_plot(self) -> None:
        self.input_plot_widget_plot_item = self.ui.input_plot_widget.getPlotItem()
        self.crosshair_update = SignalProxy(self.ui.input_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                            slot=self.update_crosshair_input_plot)
        self.ui.input_plot_widget.setAntialiasing(1)
        self.input_plot_widget_plot_item.enableAutoRange()
        self.input_plot_widget_plot_item.showGrid(True, True, 0.75)
        self.ui.input_plot_widget.vertical_line = InfiniteLine()
        self.ui.input_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.input_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.input_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.input_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Imported Raman spectra</span>")
        items_matches = (x for x in self.input_plot_widget_plot_item.listDataItems() if not x.name())
        for i in items_matches:
            self.input_plot_widget_plot_item.removeItem(i)
        self._initial_input_plot_color()

    def _initial_input_plot_color(self) -> None:
        self.ui.input_plot_widget.setBackground(self.plot_background_color)
        self.input_plot_widget_plot_item.getAxis('bottom').setPen(self.plot_text_color)
        self.input_plot_widget_plot_item.getAxis('left').setPen(self.plot_text_color)
        self.input_plot_widget_plot_item.getAxis('bottom').setTextPen(self.plot_text_color)
        self.input_plot_widget_plot_item.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_converted_cm_plot(self) -> None:
        self.converted_cm_widget_plot_item = self.ui.converted_cm_plot_widget.getPlotItem()
        self.crosshair_update_cm = SignalProxy(self.ui.converted_cm_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                               slot=self.update_crosshair_converted_cm_plot)
        self.linearRegionCmConverted = LinearRegionItem(
            [self.ui.cm_range_start.value(), self.ui.cm_range_end.value()],
            bounds=[self.ui.cm_range_start.minimum(), self.ui.cm_range_end.maximum()], swapMode='push')
        self.ui.converted_cm_plot_widget.setAntialiasing(1)
        self.converted_cm_widget_plot_item.enableAutoRange()
        self.converted_cm_widget_plot_item.showGrid(True, True, 0.75)
        self.ui.converted_cm_plot_widget.vertical_line = InfiniteLine()
        self.ui.converted_cm_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.converted_cm_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.converted_cm_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.converted_cm_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Converted to cm\N{superscript minus}\N{superscript one}</span>")
        items_matches = (x for x in self.converted_cm_widget_plot_item.listDataItems() if not x.name())
        for i in items_matches:
            self.converted_cm_widget_plot_item.removeItem(i)
        self.converted_cm_widget_plot_item.addItem(self.linearRegionCmConverted)
        color_for_lr = QColor(self.theme_colors['secondaryDarkColor'])
        color_for_lr.setAlpha(20)
        color_for_lr_hover = QColor(self.theme_colors['secondaryDarkColor'])
        color_for_lr_hover.setAlpha(40)
        self.linearRegionCmConverted.setBrush(color_for_lr)
        self.linearRegionCmConverted.setHoverBrush(color_for_lr_hover)
        self.linearRegionCmConverted.sigRegionChangeFinished.connect(self.lr_cm_region_changed)
        self.linearRegionCmConverted.setMovable(not self.ui.lr_movableBtn.isChecked())
        self._initial_converted_cm_plot_color()

    def _initial_converted_cm_plot_color(self) -> None:
        self.ui.converted_cm_plot_widget.setBackground(self.plot_background_color)
        self.converted_cm_widget_plot_item.getAxis('bottom').setPen(self.plot_text_color)
        self.converted_cm_widget_plot_item.getAxis('left').setPen(self.plot_text_color)
        self.converted_cm_widget_plot_item.getAxis('bottom').setTextPen(self.plot_text_color)
        self.converted_cm_widget_plot_item.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_cut_cm_plot(self) -> None:
        self.cut_cm_plotItem = self.ui.cut_cm_plot_widget.getPlotItem()
        self.crosshair_update_cut = SignalProxy(self.ui.cut_cm_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                                slot=self.update_crosshair_cut_plot)
        self.ui.cut_cm_plot_widget.setAntialiasing(1)
        self.cut_cm_plotItem.enableAutoRange()
        self.cut_cm_plotItem.showGrid(True, True, 0.75)
        self.ui.cut_cm_plot_widget.vertical_line = InfiniteLine()
        self.ui.cut_cm_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.cut_cm_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.cut_cm_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.cut_cm_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Cutted plots</span>")
        items_matches = (x for x in self.cut_cm_plotItem.listDataItems() if not x.name())
        for i in items_matches:
            self.cut_cm_plotItem.removeItem(i)
        self._initial_cut_cm_plot_color()

    def _initial_cut_cm_plot_color(self) -> None:
        self.ui.cut_cm_plot_widget.setBackground(self.plot_background_color)
        self.cut_cm_plotItem.getAxis('bottom').setPen(self.plot_text_color)
        self.cut_cm_plotItem.getAxis('left').setPen(self.plot_text_color)
        self.cut_cm_plotItem.getAxis('bottom').setTextPen(self.plot_text_color)
        self.cut_cm_plotItem.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_normalize_plot(self) -> None:
        self.normalize_plotItem = self.ui.normalize_plot_widget.getPlotItem()
        self.crosshair_update_normalize = SignalProxy(self.ui.normalize_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                                      slot=self.update_crosshair_normalized_plot)
        self.ui.normalize_plot_widget.setAntialiasing(1)
        self.normalize_plotItem.enableAutoRange()
        self.normalize_plotItem.showGrid(True, True, 0.75)
        self.ui.normalize_plot_widget.vertical_line = InfiniteLine()
        self.ui.normalize_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.normalize_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.normalize_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.normalize_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Normalized plots</span>")
        items_matches = (x for x in self.normalize_plotItem.listDataItems() if not x.name())
        for i in items_matches:
            self.normalize_plotItem.removeItem(i)
        self._initial_normalize_plot_color()

    def _initial_normalize_plot_color(self) -> None:
        self.ui.normalize_plot_widget.setBackground(self.plot_background_color)
        self.normalize_plotItem.getAxis('bottom').setPen(self.plot_text_color)
        self.normalize_plotItem.getAxis('left').setPen(self.plot_text_color)
        self.normalize_plotItem.getAxis('bottom').setTextPen(self.plot_text_color)
        self.normalize_plotItem.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_smooth_plot(self) -> None:
        self.smooth_plotItem = self.ui.smooth_plot_widget.getPlotItem()
        self.crosshair_update_smooth = SignalProxy(self.ui.smooth_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                                   slot=self.update_crosshair_smoothed_plot)
        self.ui.smooth_plot_widget.setAntialiasing(1)
        self.smooth_plotItem.enableAutoRange()
        self.smooth_plotItem.showGrid(True, True, 0.75)
        self.ui.smooth_plot_widget.vertical_line = InfiniteLine()
        self.ui.smooth_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.smooth_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.smooth_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.smooth_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Smoothed plots.</span>")
        items_matches = (x for x in self.smooth_plotItem.listDataItems() if not x.name())
        for i in items_matches:
            self.smooth_plotItem.removeItem(i)
        self._initial_smooth_plot_color()

    def _initial_smooth_plot_color(self) -> None:
        self.ui.smooth_plot_widget.setBackground(self.plot_background_color)
        self.smooth_plotItem.getAxis('bottom').setPen(self.plot_text_color)
        self.smooth_plotItem.getAxis('left').setPen(self.plot_text_color)
        self.smooth_plotItem.getAxis('bottom').setTextPen(self.plot_text_color)
        self.smooth_plotItem.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_baseline_plot(self) -> None:
        self.baseline_corrected_plotItem = self.ui.baseline_plot_widget.getPlotItem()
        self.crosshair_update_baseline = SignalProxy(self.ui.baseline_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                                     slot=self.update_crosshair_baseline_plot)
        self.ui.baseline_plot_widget.setAntialiasing(1)
        self.baseline_corrected_plotItem.enableAutoRange()
        self.baseline_corrected_plotItem.showGrid(True, True, 0.75)
        self.ui.baseline_plot_widget.vertical_line = InfiniteLine()
        self.ui.baseline_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.baseline_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.baseline_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.baseline_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Baseline corrected plots.</span>")
        items_matches = (x for x in self.baseline_corrected_plotItem.listDataItems() if not x.name())
        for i in items_matches:
            self.baseline_corrected_plotItem.removeItem(i)
        self._initial_baseline_plot_color()
        self.linearRegionBaseline = LinearRegionItem(
            [self.ui.trim_start_cm.value(), self.ui.trim_end_cm.value()],
            bounds=[self.ui.trim_start_cm.minimum(), self.ui.trim_end_cm.maximum()], swapMode='push')
        self.baseline_corrected_plotItem.addItem(self.linearRegionBaseline)
        color_for_lr = QColor(self.theme_colors['secondaryDarkColor'])
        color_for_lr.setAlpha(20)
        color_for_lr_hover = QColor(self.theme_colors['secondaryDarkColor'])
        color_for_lr_hover.setAlpha(40)
        self.linearRegionBaseline.setBrush(color_for_lr)
        self.linearRegionBaseline.setHoverBrush(color_for_lr_hover)
        self.linearRegionBaseline.sigRegionChangeFinished.connect(self.lr_baseline_region_changed)
        self.linearRegionBaseline.setMovable(not self.ui.lr_movableBtn.isChecked())

    def _initial_baseline_plot_color(self) -> None:
        self.ui.baseline_plot_widget.setBackground(self.plot_background_color)
        self.baseline_corrected_plotItem.getAxis('bottom').setPen(self.plot_text_color)
        self.baseline_corrected_plotItem.getAxis('left').setPen(self.plot_text_color)
        self.baseline_corrected_plotItem.getAxis('bottom').setTextPen(self.plot_text_color)
        self.baseline_corrected_plotItem.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_averaged_plot(self) -> None:
        self.averaged_plotItem = self.ui.average_plot_widget.getPlotItem()
        self.crosshair_update_averaged = SignalProxy(self.ui.average_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                                     slot=self.update_crosshair_averaged_plot)
        self.ui.average_plot_widget.setAntialiasing(1)
        self.averaged_plotItem.enableAutoRange()
        self.averaged_plotItem.showGrid(True, True, 0.75)
        self.ui.average_plot_widget.vertical_line = InfiniteLine()
        self.ui.average_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.average_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.average_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.average_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Averaged</span>")
        items_matches = (x for x in self.averaged_plotItem.listDataItems() if not x.name())
        for i in items_matches:
            self.averaged_plotItem.removeItem(i)
        self._initial_averaged_plot_color()

    def _initial_averaged_plot_color(self) -> None:
        self.ui.average_plot_widget.setBackground(self.plot_background_color)
        self.averaged_plotItem.getAxis('bottom').setPen(self.plot_text_color)
        self.averaged_plotItem.getAxis('left').setPen(self.plot_text_color)
        self.averaged_plotItem.getAxis('bottom').setTextPen(self.plot_text_color)
        self.averaged_plotItem.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_deconvolution_plot(self) -> None:
        self.deconvolution_plotItem = self.ui.deconv_plot_widget.getPlotItem()
        self.ui.deconv_plot_widget.scene().sigMouseClicked.connect(self.fit_plot_mouse_clicked)
        self.crosshair_update_deconv = SignalProxy(self.ui.deconv_plot_widget.scene().sigMouseMoved, rateLimit=60,
                                                   slot=self.update_crosshair_deconv_plot)
        self.ui.deconv_plot_widget.setAntialiasing(1)
        self.deconvolution_plotItem.enableAutoRange()
        self.deconvolution_plotItem.showGrid(True, True, 0.5)
        self.ui.deconv_plot_widget.vertical_line = InfiniteLine()
        self.ui.deconv_plot_widget.horizontal_line = InfiniteLine(angle=0)
        self.ui.deconv_plot_widget.vertical_line.setPen(QColor(self.theme_colors['secondaryColor']))
        self.ui.deconv_plot_widget.horizontal_line.setPen(QColor(self.theme_colors['secondaryColor']))
        items_matches = (x for x in self.deconvolution_plotItem.listDataItems() if not x.name())
        for i in items_matches:
            self.deconvolution_plotItem.removeItem(i)
        self._initial_deconv_plot_color()
        self.linearRegionDeconv = LinearRegionItem(
            [self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value()],
            bounds=[self.ui.interval_start_dsb.minimum(), self.ui.interval_end_dsb.maximum()], swapMode='push')
        self.deconvolution_plotItem.addItem(self.linearRegionDeconv)
        color_for_lr = QColor(self.theme_colors['secondaryDarkColor'])
        color_for_lr.setAlpha(20)
        color_for_lr_hover = QColor(self.theme_colors['secondaryDarkColor'])
        color_for_lr_hover.setAlpha(40)
        self.linearRegionDeconv.setBrush(color_for_lr)
        self.linearRegionDeconv.setHoverBrush(color_for_lr_hover)
        self.linearRegionDeconv.sigRegionChangeFinished.connect(self.lr_deconv_region_changed)
        self.linearRegionDeconv.setMovable(not self.ui.lr_movableBtn.isChecked())
        self.sigma3_top = PlotCurveItem(name='sigma3_top')
        self.sigma3_bottom = PlotCurveItem(name='sigma3_bottom')
        color = self.sigma3_style['color']
        color.setAlphaF(0.25)
        pen = mkPen(color=color, style=Qt.PenStyle.SolidLine)
        brush = mkBrush(color)
        self.fill = FillBetweenItem(self.sigma3_top, self.sigma3_bottom, brush, pen)
        self.deconvolution_plotItem.addItem(self.fill)
        self.fill.setVisible(False)

    def fit_plot_mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.deselect_selected_line()

    def _initial_deconv_plot_color(self) -> None:
        self.ui.deconv_plot_widget.setBackground(self.plot_background_color)
        self.deconvolution_plotItem.getAxis('bottom').setPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis('left').setPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis('bottom').setTextPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis('left').setTextPen(self.plot_text_color)
        # self.update_single_deconvolution_plot(self.current_spectrum_deconvolution_name)

    def _initial_lda_scores_1d_plot(self) -> None:
        self.lda_scores_1d_plot_item = self.ui.lda_scores_1d_plot_widget.getPlotItem()
        self.ui.lda_scores_1d_plot_widget.setAntialiasing(1)
        self.lda_scores_1d_plot_item.enableAutoRange()
        items_matches = self.lda_scores_1d_plot_item.listDataItems()
        for i in items_matches:
            self.lda_scores_1d_plot_item.removeItem(i)
        for i in self.lda_1d_inf_lines:
            self.lda_scores_1d_plot_item.removeItem(i)
        self._initial_lda_scores_1d_plot_color()
        self.ui.lda_scores_1d_plot_widget.setVisible(False)

    def _initial_lda_scores_1d_plot_color(self) -> None:
        self.ui.lda_scores_1d_plot_widget.setBackground(self.plot_background_color)
        self.lda_scores_1d_plot_item.getAxis('bottom').setPen(self.plot_text_color)
        self.lda_scores_1d_plot_item.getAxis('left').setPen(self.plot_text_color)
        self.lda_scores_1d_plot_item.getAxis('bottom').setTextPen(self.plot_text_color)
        self.lda_scores_1d_plot_item.getAxis('left').setTextPen(self.plot_text_color)

    def _initial_lda_scores_2d_plot(self) -> None:
        self.ui.lda_scores_2d_plot_widget.canvas.axes.cla()
        self.ui.lda_scores_2d_plot_widget.canvas.axes.set_xlabel('LD-1')
        self.ui.lda_scores_2d_plot_widget.canvas.axes.set_ylabel('LD-2')
        self.ui.lda_scores_2d_plot_widget.setVisible(True)
        self.ui.lda_scores_2d_plot_widget.canvas.draw()

    def _initial_lda_dm_plot(self) -> None:
        self.ui.lda_dm_plot.canvas.axes.cla()
        self.ui.lda_dm_plot.canvas.draw()

    def _initial_lda_features_plot(self) -> None:
        self.ui.lda_features_plot_widget.canvas.axes.cla()
        self.ui.lda_features_plot_widget.canvas.draw()

    def _initial_lda_roc_plot(self) -> None:
        self.ui.lda_roc_plot.canvas.axes.cla()
        self.ui.lda_roc_plot.canvas.draw()

    def _initial_lda_pr_plot(self) -> None:
        self.ui.lda_pr_plot.canvas.axes.cla()
        self.ui.lda_pr_plot.canvas.draw()

    def _initial_lda_shap_means_plot(self) -> None:
        self.ui.lda_shap_means.canvas.figure.gca().cla()
        self.ui.lda_shap_means.canvas.draw()

    def _initial_lda_shap_beeswarm_plot(self) -> None:
        self.ui.lda_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.lda_shap_beeswarm.canvas.draw()

    def _initial_lda_shap_heatmap_plot(self) -> None:
        self.ui.lda_shap_heatmap.canvas.figure.gca().cla()
        self.ui.lda_shap_heatmap.canvas.draw()

    def _initial_lda_shap_scatter_plot(self) -> None:
        self.ui.lda_shap_scatter.canvas.figure.gca().cla()
        self.ui.lda_shap_scatter.canvas.draw()

    def _initial_lda_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.lda_force_single.setHtml(html_code)
        self.ui.lda_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.lda_force_single.contextMenuEvent = self.lda_force_single_context_menu_event

    def lda_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lda_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_lda_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.lda_force_full.setHtml(html_code)
        self.ui.lda_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.lda_force_full.contextMenuEvent = self.lda_force_full_context_menu_event

    def lda_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lda_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_lda_shap_decision_plot(self) -> None:
        self.ui.lda_shap_decision.canvas.figure.gca().cla()
        self.ui.lda_shap_decision.canvas.figure.clf()
        self.ui.lda_shap_decision.canvas.draw()

    def _initial_lda_shap_waterfall_plot(self) -> None:
        self.ui.lda_shap_waterfall.canvas.figure.gca().cla()
        self.ui.lda_shap_waterfall.canvas.figure.clf()
        self.ui.lda_shap_waterfall.canvas.draw()

    def _initial_qda_scores_2d_plot(self) -> None:
        self.ui.qda_scores_plot_widget.canvas.axes.cla()
        self.ui.qda_scores_plot_widget.canvas.axes.set_xlabel('QD-1')
        self.ui.qda_scores_plot_widget.canvas.axes.set_ylabel('QD-2')
        self.ui.qda_scores_plot_widget.canvas.draw()

    def _initial_qda_dm_plot(self) -> None:
        self.ui.qda_dm_plot.canvas.axes.cla()
        self.ui.qda_dm_plot.canvas.draw()

    def _initial_qda_pr_plot(self) -> None:
        self.ui.qda_pr_plot.canvas.axes.cla()
        self.ui.qda_pr_plot.canvas.draw()

    def _initial_qda_roc_plot(self) -> None:
        self.ui.qda_roc_plot.canvas.axes.cla()
        self.ui.qda_roc_plot.canvas.draw()

    def _initial_qda_shap_means_plot(self) -> None:
        self.ui.qda_shap_means.canvas.figure.gca().cla()
        self.ui.qda_shap_means.canvas.draw()

    def _initial_qda_shap_beeswarm_plot(self) -> None:
        self.ui.qda_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.qda_shap_beeswarm.canvas.draw()

    def _initial_qda_shap_heatmap_plot(self) -> None:
        self.ui.qda_shap_heatmap.canvas.figure.gca().cla()
        self.ui.qda_shap_heatmap.canvas.draw()

    def _initial_qda_shap_scatter_plot(self) -> None:
        self.ui.qda_shap_scatter.canvas.figure.gca().cla()
        self.ui.qda_shap_scatter.canvas.draw()

    def _initial_qda_shap_waterfall_plot(self) -> None:
        self.ui.qda_shap_waterfall.canvas.figure.gca().cla()
        self.ui.qda_shap_waterfall.canvas.figure.clf()
        self.ui.qda_shap_waterfall.canvas.draw()

    def _initial_lr_scores_plot(self) -> None:
        self.ui.lr_scores_plot_widget.canvas.axes.cla()
        self.ui.lr_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.lr_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.lr_scores_plot_widget.canvas.draw()

    def _initial_lr_features_plot(self) -> None:
        self.ui.lr_features_plot_widget.canvas.axes.cla()
        self.ui.lr_features_plot_widget.canvas.draw()

    def _initial_lr_dm_plot(self) -> None:
        self.ui.lr_dm_plot.canvas.axes.cla()
        self.ui.lr_dm_plot.canvas.draw()

    def _initial_lr_pr_plot(self) -> None:
        self.ui.lr_pr_plot.canvas.axes.cla()
        self.ui.lr_pr_plot.canvas.draw()

    def _initial_lr_roc_plot(self) -> None:
        self.ui.lr_roc_plot.canvas.axes.cla()
        self.ui.lr_roc_plot.canvas.draw()

    def _initial_lr_shap_means_plot(self) -> None:
        self.ui.lr_shap_means.canvas.figure.gca().cla()
        self.ui.lr_shap_means.canvas.draw()

    def _initial_lr_shap_beeswarm_plot(self) -> None:
        self.ui.lr_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.lr_shap_beeswarm.canvas.draw()

    def _initial_lr_shap_heatmap_plot(self) -> None:
        self.ui.lr_shap_heatmap.canvas.figure.gca().cla()
        self.ui.lr_shap_heatmap.canvas.draw()

    def _initial_lr_shap_scatter_plot(self) -> None:
        self.ui.lr_shap_scatter.canvas.figure.gca().cla()
        self.ui.lr_shap_scatter.canvas.figure.clf()
        self.ui.lr_shap_scatter.canvas.draw()

    def _initial_lr_shap_decision_plot(self) -> None:
        self.ui.lr_shap_decision.canvas.figure.gca().cla()
        self.ui.lr_shap_decision.canvas.figure.clf()
        self.ui.lr_shap_decision.canvas.draw()

    def _initial_lr_shap_waterfall_plot(self) -> None:
        self.ui.lr_shap_waterfall.canvas.figure.gca().cla()
        self.ui.lr_shap_waterfall.canvas.figure.clf()
        self.ui.lr_shap_waterfall.canvas.draw()

    def _initial_lr_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.lr_force_single.setHtml(html_code)
        self.ui.lr_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.lr_force_single.contextMenuEvent = self.lr_force_single_context_menu_event

    def lr_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lr_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_lr_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.lr_force_full.setHtml(html_code)
        self.ui.lr_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.lr_force_full.contextMenuEvent = self.lr_force_full_context_menu_event

    def lr_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lr_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_svc_scores_plot(self) -> None:
        self.ui.svc_scores_plot_widget.canvas.axes.cla()
        self.ui.svc_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.svc_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.svc_scores_plot_widget.canvas.draw()

    def _initial_svc_features_plot(self) -> None:
        self.ui.svc_features_plot_widget.canvas.axes.cla()
        self.ui.svc_features_plot_widget.canvas.draw()

    def _initial_svc_dm_plot(self) -> None:
        self.ui.svc_dm_plot.canvas.axes.cla()
        self.ui.svc_dm_plot.canvas.draw()

    def _initial_svc_pr_plot(self) -> None:
        self.ui.svc_pr_plot.canvas.axes.cla()
        self.ui.svc_pr_plot.canvas.draw()

    def _initial_svc_roc_plot(self) -> None:
        self.ui.svc_roc_plot.canvas.axes.cla()
        self.ui.svc_roc_plot.canvas.draw()

    def _initial_svc_shap_means_plot(self) -> None:
        self.ui.svc_shap_means.canvas.figure.gca().cla()
        self.ui.svc_shap_means.canvas.figure.clf()
        self.ui.svc_shap_means.canvas.draw()

    def _initial_svc_shap_beeswarm_plot(self) -> None:
        self.ui.svc_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.svc_shap_beeswarm.canvas.figure.clf()
        self.ui.svc_shap_beeswarm.canvas.draw()

    def _initial_svc_shap_heatmap_plot(self) -> None:
        self.ui.svc_shap_heatmap.canvas.figure.gca().cla()
        self.ui.svc_shap_heatmap.canvas.figure.clf()
        self.ui.svc_shap_heatmap.canvas.draw()

    def _initial_svc_shap_scatter_plot(self) -> None:
        self.ui.svc_shap_scatter.canvas.figure.gca().cla()
        self.ui.svc_shap_scatter.canvas.figure.clf()
        self.ui.svc_shap_scatter.canvas.draw()

    def _initial_svc_shap_decision_plot(self) -> None:
        self.ui.svc_shap_decision.canvas.figure.gca().cla()
        self.ui.svc_shap_decision.canvas.figure.clf()
        self.ui.svc_shap_decision.canvas.draw()

    def _initial_svc_shap_waterfall_plot(self) -> None:
        self.ui.svc_shap_waterfall.canvas.figure.gca().cla()
        self.ui.svc_shap_waterfall.canvas.figure.clf()
        self.ui.svc_shap_waterfall.canvas.draw()

    def _initial_svc_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.svc_force_single.setHtml(html_code)
        self.ui.svc_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.svc_force_single.contextMenuEvent = self.svc_force_single_context_menu_event

    def svc_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.svc_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_svc_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.svc_force_full.setHtml(html_code)
        self.ui.svc_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.svc_force_full.contextMenuEvent = self.svc_force_full_context_menu_event

    def svc_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.svc_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nearest_scores_plot(self) -> None:
        self.ui.nearest_scores_plot_widget.canvas.axes.cla()
        self.ui.nearest_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.nearest_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.nearest_scores_plot_widget.canvas.draw()

    def _initial_nearest_dm_plot(self) -> None:
        self.ui.nearest_dm_plot.canvas.axes.cla()
        self.ui.nearest_dm_plot.canvas.draw()

    def _initial_nearest_pr_plot(self) -> None:
        self.ui.nearest_pr_plot.canvas.axes.cla()
        self.ui.nearest_pr_plot.canvas.draw()

    def _initial_nearest_roc_plot(self) -> None:
        self.ui.nearest_roc_plot.canvas.axes.cla()
        self.ui.nearest_roc_plot.canvas.draw()

    def _initial_nearest_shap_means_plot(self) -> None:
        self.ui.nearest_shap_means.canvas.figure.gca().cla()
        self.ui.nearest_shap_means.canvas.figure.clf()
        self.ui.nearest_shap_means.canvas.draw()

    def _initial_nearest_shap_beeswarm_plot(self) -> None:
        self.ui.nearest_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.nearest_shap_beeswarm.canvas.figure.clf()
        self.ui.nearest_shap_beeswarm.canvas.draw()

    def _initial_nearest_shap_heatmap_plot(self) -> None:
        self.ui.nearest_shap_heatmap.canvas.figure.gca().cla()
        self.ui.nearest_shap_heatmap.canvas.figure.clf()
        self.ui.nearest_shap_heatmap.canvas.draw()

    def _initial_nearest_shap_scatter_plot(self) -> None:
        self.ui.nearest_shap_scatter.canvas.figure.gca().cla()
        self.ui.nearest_shap_scatter.canvas.figure.clf()
        self.ui.nearest_shap_scatter.canvas.draw()

    def _initial_nearest_shap_decision_plot(self) -> None:
        self.ui.nearest_shap_decision.canvas.figure.gca().cla()
        self.ui.nearest_shap_decision.canvas.figure.clf()
        self.ui.nearest_shap_decision.canvas.draw()

    def _initial_nearest_shap_waterfall_plot(self) -> None:
        self.ui.nearest_shap_waterfall.canvas.figure.gca().cla()
        self.ui.nearest_shap_waterfall.canvas.figure.clf()
        self.ui.nearest_shap_waterfall.canvas.draw()

    def _initial_nearest_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.nearest_force_single.setHtml(html_code)
        self.ui.nearest_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.nearest_force_single.contextMenuEvent = self.nearest_force_single_context_menu_event

    def nearest_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nearest_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nearest_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.nearest_force_full.setHtml(html_code)
        self.ui.nearest_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.nearest_force_full.contextMenuEvent = self.nearest_force_full_context_menu_event

    def nearest_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nearest_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_gpc_scores_plot(self) -> None:
        self.ui.gpc_scores_plot_widget.canvas.axes.cla()
        self.ui.gpc_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.gpc_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.gpc_scores_plot_widget.canvas.draw()

    def _initial_gpc_dm_plot(self) -> None:
        self.ui.gpc_dm_plot.canvas.axes.cla()
        self.ui.gpc_dm_plot.canvas.draw()

    def _initial_gpc_pr_plot(self) -> None:
        self.ui.gpc_pr_plot.canvas.axes.cla()
        self.ui.gpc_pr_plot.canvas.draw()

    def _initial_gpc_roc_plot(self) -> None:
        self.ui.gpc_roc_plot.canvas.axes.cla()
        self.ui.gpc_roc_plot.canvas.draw()

    def _initial_gpc_shap_means_plot(self) -> None:
        self.ui.gpc_shap_means.canvas.figure.gca().cla()
        self.ui.gpc_shap_means.canvas.figure.clf()
        self.ui.gpc_shap_means.canvas.draw()

    def _initial_gpc_shap_beeswarm_plot(self) -> None:
        self.ui.gpc_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.gpc_shap_beeswarm.canvas.figure.clf()
        self.ui.gpc_shap_beeswarm.canvas.draw()

    def _initial_gpc_shap_heatmap_plot(self) -> None:
        self.ui.gpc_shap_heatmap.canvas.figure.gca().cla()
        self.ui.gpc_shap_heatmap.canvas.figure.clf()
        self.ui.gpc_shap_heatmap.canvas.draw()

    def _initial_gpc_shap_scatter_plot(self) -> None:
        self.ui.gpc_shap_scatter.canvas.figure.gca().cla()
        self.ui.gpc_shap_scatter.canvas.figure.clf()
        self.ui.gpc_shap_scatter.canvas.draw()

    def _initial_gpc_shap_decision_plot(self) -> None:
        self.ui.gpc_shap_decision.canvas.figure.gca().cla()
        self.ui.gpc_shap_decision.canvas.figure.clf()
        self.ui.gpc_shap_decision.canvas.draw()

    def _initial_gpc_shap_waterfall_plot(self) -> None:
        self.ui.gpc_shap_waterfall.canvas.figure.gca().cla()
        self.ui.gpc_shap_waterfall.canvas.figure.clf()
        self.ui.gpc_shap_waterfall.canvas.draw()

    def _initial_gpc_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.gpc_force_single.setHtml(html_code)
        self.ui.gpc_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.gpc_force_single.contextMenuEvent = self.gpc_force_single_context_menu_event

    def gpc_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.gpc_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_gpc_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.gpc_force_full.setHtml(html_code)
        self.ui.gpc_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.gpc_force_full.contextMenuEvent = self.gpc_force_full_context_menu_event

    def gpc_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.gpc_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_dt_scores_plot(self) -> None:
        self.ui.dt_scores_plot_widget.canvas.axes.cla()
        self.ui.dt_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.dt_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.dt_scores_plot_widget.setVisible(True)
        self.ui.dt_scores_plot_widget.canvas.draw()

    def _initial_dt_features_plot(self) -> None:
        self.ui.dt_features_plot_widget.canvas.axes.cla()
        self.ui.dt_features_plot_widget.canvas.draw()

    def _initial_dt_dm_plot(self) -> None:
        self.ui.dt_dm_plot.canvas.axes.cla()
        self.ui.dt_dm_plot.canvas.draw()

    def _initial_dt_pr_plot(self) -> None:
        self.ui.dt_pr_plot.canvas.axes.cla()
        self.ui.dt_pr_plot.canvas.draw()

    def _initial_dt_roc_plot(self) -> None:
        self.ui.dt_roc_plot.canvas.axes.cla()
        self.ui.dt_roc_plot.canvas.draw()

    def _initial_dt_shap_means_plot(self) -> None:
        self.ui.dt_shap_means.canvas.figure.gca().cla()
        self.ui.dt_shap_means.canvas.figure.clf()
        self.ui.dt_shap_means.canvas.draw()

    def _initial_dt_shap_beeswarm_plot(self) -> None:
        self.ui.dt_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.dt_shap_beeswarm.canvas.figure.clf()
        self.ui.dt_shap_beeswarm.canvas.draw()

    def _initial_dt_shap_heatmap_plot(self) -> None:
        self.ui.dt_shap_heatmap.canvas.figure.gca().cla()
        self.ui.dt_shap_heatmap.canvas.figure.clf()
        self.ui.dt_shap_heatmap.canvas.draw()

    def _initial_dt_shap_scatter_plot(self) -> None:
        self.ui.dt_shap_scatter.canvas.figure.gca().cla()
        self.ui.dt_shap_scatter.canvas.figure.clf()
        self.ui.dt_shap_scatter.canvas.draw()

    def _initial_dt_shap_decision_plot(self) -> None:
        self.ui.dt_shap_decision.canvas.figure.gca().cla()
        self.ui.dt_shap_decision.canvas.figure.clf()
        self.ui.dt_shap_decision.canvas.draw()

    def _initial_dt_shap_waterfall_plot(self) -> None:
        self.ui.dt_shap_waterfall.canvas.figure.gca().cla()
        self.ui.dt_shap_waterfall.canvas.figure.clf()
        self.ui.dt_shap_waterfall.canvas.draw()

    def _initial_dt_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.dt_force_single.setHtml(html_code)
        self.ui.dt_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.dt_force_single.contextMenuEvent = self.dt_force_single_context_menu_event

    def dt_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.dt_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_dt_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.dt_force_full.setHtml(html_code)
        self.ui.dt_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.dt_force_full.contextMenuEvent = self.dt_force_full_context_menu_event

    def dt_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.dt_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nb_scores_plot(self) -> None:
        self.ui.nb_scores_plot_widget.canvas.axes.cla()
        self.ui.nb_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.nb_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.nb_scores_plot_widget.setVisible(True)
        self.ui.nb_scores_plot_widget.canvas.draw()

    def _initial_nb_dm_plot(self) -> None:
        self.ui.nb_dm_plot.canvas.axes.cla()
        self.ui.nb_dm_plot.canvas.draw()

    def _initial_nb_pr_plot(self) -> None:
        self.ui.nb_pr_plot.canvas.axes.cla()
        self.ui.nb_pr_plot.canvas.draw()

    def _initial_nb_roc_plot(self) -> None:
        self.ui.nb_roc_plot.canvas.axes.cla()
        self.ui.nb_roc_plot.canvas.draw()

    def _initial_nb_shap_means_plot(self) -> None:
        self.ui.nb_shap_means.canvas.figure.gca().cla()
        self.ui.nb_shap_means.canvas.figure.clf()
        self.ui.nb_shap_means.canvas.draw()

    def _initial_nb_shap_beeswarm_plot(self) -> None:
        self.ui.nb_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.nb_shap_beeswarm.canvas.figure.clf()
        self.ui.nb_shap_beeswarm.canvas.draw()

    def _initial_nb_shap_heatmap_plot(self) -> None:
        self.ui.nb_shap_heatmap.canvas.figure.gca().cla()
        self.ui.nb_shap_heatmap.canvas.figure.clf()
        self.ui.nb_shap_heatmap.canvas.draw()

    def _initial_nb_shap_scatter_plot(self) -> None:
        self.ui.nb_shap_scatter.canvas.figure.gca().cla()
        self.ui.nb_shap_scatter.canvas.figure.clf()
        self.ui.nb_shap_scatter.canvas.draw()

    def _initial_nb_shap_decision_plot(self) -> None:
        self.ui.nb_shap_decision.canvas.figure.gca().cla()
        self.ui.nb_shap_decision.canvas.figure.clf()
        self.ui.nb_shap_decision.canvas.draw()

    def _initial_nb_shap_waterfall_plot(self) -> None:
        self.ui.nb_shap_waterfall.canvas.figure.gca().cla()
        self.ui.nb_shap_waterfall.canvas.figure.clf()
        self.ui.nb_shap_waterfall.canvas.draw()

    def _initial_nb_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.nb_force_single.setHtml(html_code)
        self.ui.nb_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.nb_force_single.contextMenuEvent = self.dt_force_single_context_menu_event

    def nb_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nb_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nb_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.nb_force_full.setHtml(html_code)
        self.ui.nb_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.nb_force_full.contextMenuEvent = self.nb_force_full_context_menu_event

    def nb_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nb_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_rf_scores_plot(self) -> None:
        self.ui.rf_scores_plot_widget.canvas.axes.cla()
        self.ui.rf_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.rf_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.rf_scores_plot_widget.setVisible(True)
        self.ui.rf_scores_plot_widget.canvas.draw()

    def _initial_rf_features_plot(self) -> None:
        self.ui.rf_features_plot_widget.canvas.axes.cla()
        self.ui.rf_features_plot_widget.canvas.draw()

    def _initial_rf_dm_plot(self) -> None:
        self.ui.rf_dm_plot.canvas.axes.cla()
        self.ui.rf_dm_plot.canvas.draw()

    def _initial_rf_pr_plot(self) -> None:
        self.ui.rf_pr_plot.canvas.axes.cla()
        self.ui.rf_pr_plot.canvas.draw()

    def _initial_rf_roc_plot(self) -> None:
        self.ui.rf_roc_plot.canvas.axes.cla()
        self.ui.rf_roc_plot.canvas.draw()

    def _initial_rf_shap_means_plot(self) -> None:
        self.ui.rf_shap_means.canvas.figure.gca().cla()
        self.ui.rf_shap_means.canvas.figure.clf()
        self.ui.rf_shap_means.canvas.draw()

    def _initial_rf_shap_beeswarm_plot(self) -> None:
        self.ui.rf_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.rf_shap_beeswarm.canvas.figure.clf()
        self.ui.rf_shap_beeswarm.canvas.draw()

    def _initial_rf_shap_heatmap_plot(self) -> None:
        self.ui.rf_shap_heatmap.canvas.figure.gca().cla()
        self.ui.rf_shap_heatmap.canvas.figure.clf()
        self.ui.rf_shap_heatmap.canvas.draw()

    def _initial_rf_shap_scatter_plot(self) -> None:
        self.ui.rf_shap_scatter.canvas.figure.gca().cla()
        self.ui.rf_shap_scatter.canvas.figure.clf()
        self.ui.rf_shap_scatter.canvas.draw()

    def _initial_rf_shap_decision_plot(self) -> None:
        self.ui.rf_shap_decision.canvas.figure.gca().cla()
        self.ui.rf_shap_decision.canvas.figure.clf()
        self.ui.rf_shap_decision.canvas.draw()

    def _initial_rf_shap_waterfall_plot(self) -> None:
        self.ui.rf_shap_waterfall.canvas.figure.gca().cla()
        self.ui.rf_shap_waterfall.canvas.figure.clf()
        self.ui.rf_shap_waterfall.canvas.draw()

    def _initial_rf_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.rf_force_single.setHtml(html_code)
        self.ui.rf_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.rf_force_single.contextMenuEvent = self.rf_force_single_context_menu_event

    def rf_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.rf_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_rf_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.rf_force_full.setHtml(html_code)
        self.ui.rf_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.rf_force_full.contextMenuEvent = self.rf_force_full_context_menu_event

    def rf_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.rf_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_ab_scores_plot(self) -> None:
        self.ui.ab_scores_plot_widget.canvas.axes.cla()
        self.ui.ab_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.ab_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.ab_scores_plot_widget.setVisible(True)
        self.ui.ab_scores_plot_widget.canvas.draw()

    def _initial_ab_features_plot(self) -> None:
        self.ui.ab_features_plot_widget.canvas.axes.cla()
        self.ui.ab_features_plot_widget.canvas.draw()

    def _initial_ab_dm_plot(self) -> None:
        self.ui.ab_dm_plot.canvas.axes.cla()
        self.ui.ab_dm_plot.canvas.draw()

    def _initial_ab_pr_plot(self) -> None:
        self.ui.ab_pr_plot.canvas.axes.cla()
        self.ui.ab_pr_plot.canvas.draw()

    def _initial_ab_roc_plot(self) -> None:
        self.ui.ab_roc_plot.canvas.axes.cla()
        self.ui.ab_roc_plot.canvas.draw()

    def _initial_ab_shap_means_plot(self) -> None:
        self.ui.ab_shap_means.canvas.figure.gca().cla()
        self.ui.ab_shap_means.canvas.figure.clf()
        self.ui.ab_shap_means.canvas.draw()

    def _initial_ab_shap_beeswarm_plot(self) -> None:
        self.ui.ab_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.ab_shap_beeswarm.canvas.figure.clf()
        self.ui.ab_shap_beeswarm.canvas.draw()

    def _initial_ab_shap_heatmap_plot(self) -> None:
        self.ui.ab_shap_heatmap.canvas.figure.gca().cla()
        self.ui.ab_shap_heatmap.canvas.figure.clf()
        self.ui.ab_shap_heatmap.canvas.draw()

    def _initial_ab_shap_scatter_plot(self) -> None:
        self.ui.ab_shap_scatter.canvas.figure.gca().cla()
        self.ui.ab_shap_scatter.canvas.figure.clf()
        self.ui.ab_shap_scatter.canvas.draw()

    def _initial_ab_shap_decision_plot(self) -> None:
        self.ui.ab_shap_decision.canvas.figure.gca().cla()
        self.ui.ab_shap_decision.canvas.figure.clf()
        self.ui.ab_shap_decision.canvas.draw()

    def _initial_ab_shap_waterfall_plot(self) -> None:
        self.ui.ab_shap_waterfall.canvas.figure.gca().cla()
        self.ui.ab_shap_waterfall.canvas.figure.clf()
        self.ui.ab_shap_waterfall.canvas.draw()

    def _initial_ab_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.ab_force_single.setHtml(html_code)
        self.ui.ab_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.ab_force_single.contextMenuEvent = self.ab_force_single_context_menu_event

    def ab_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.ab_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_ab_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.ab_force_full.setHtml(html_code)
        self.ui.ab_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.ab_force_full.contextMenuEvent = self.ab_force_full_context_menu_event

    def ab_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.ab_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_mlp_scores_plot(self) -> None:
        self.ui.mlp_scores_plot_widget.canvas.axes.cla()
        self.ui.mlp_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.mlp_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.mlp_scores_plot_widget.setVisible(True)
        self.ui.mlp_scores_plot_widget.canvas.draw()

    def _initial_mlp_dm_plot(self) -> None:
        self.ui.mlp_dm_plot.canvas.axes.cla()
        self.ui.mlp_dm_plot.canvas.draw()

    def _initial_mlp_pr_plot(self) -> None:
        self.ui.mlp_pr_plot.canvas.axes.cla()
        self.ui.mlp_pr_plot.canvas.draw()

    def _initial_mlp_roc_plot(self) -> None:
        self.ui.mlp_roc_plot.canvas.axes.cla()
        self.ui.mlp_roc_plot.canvas.draw()

    def _initial_mlp_shap_means_plot(self) -> None:
        self.ui.mlp_shap_means.canvas.figure.gca().cla()
        self.ui.mlp_shap_means.canvas.figure.clf()
        self.ui.mlp_shap_means.canvas.draw()

    def _initial_mlp_shap_beeswarm_plot(self) -> None:
        self.ui.mlp_shap_beeswarm.canvas.figure.gca().cla()
        self.ui.mlp_shap_beeswarm.canvas.figure.clf()
        self.ui.mlp_shap_beeswarm.canvas.draw()

    def _initial_mlp_shap_heatmap_plot(self) -> None:
        self.ui.mlp_shap_heatmap.canvas.figure.gca().cla()
        self.ui.mlp_shap_heatmap.canvas.figure.clf()
        self.ui.mlp_shap_heatmap.canvas.draw()

    def _initial_mlp_shap_scatter_plot(self) -> None:
        self.ui.mlp_shap_scatter.canvas.figure.gca().cla()
        self.ui.mlp_shap_scatter.canvas.figure.clf()
        self.ui.mlp_shap_scatter.canvas.draw()

    def _initial_mlp_shap_decision_plot(self) -> None:
        self.ui.mlp_shap_decision.canvas.figure.gca().cla()
        self.ui.mlp_shap_decision.canvas.figure.clf()
        self.ui.mlp_shap_decision.canvas.draw()

    def _initial_mlp_shap_waterfall_plot(self) -> None:
        self.ui.mlp_shap_waterfall.canvas.figure.gca().cla()
        self.ui.mlp_shap_waterfall.canvas.figure.clf()
        self.ui.mlp_shap_waterfall.canvas.draw()

    def _initial_mlp_force_single_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.mlp_force_single.setHtml(html_code)
        self.ui.mlp_force_single.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.mlp_force_single.contextMenuEvent = self.mlp_force_single_context_menu_event

    def mlp_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.mlp_force_single.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_mlp_force_full_plot(self) -> None:
        html_code = "<html><body style=""background-color:black;};></body></html>"
        self.ui.mlp_force_full.setHtml(html_code)
        self.ui.mlp_force_full.page().setBackgroundColor(QColor(self.plot_background_color))
        self.ui.mlp_force_full.contextMenuEvent = self.mlp_force_full_context_menu_event

    def mlp_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.mlp_force_full.page()))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_pca_scores_plot(self) -> None:
        self.ui.pca_scores_plot_widget.canvas.axes.cla()
        self.ui.pca_scores_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.pca_scores_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.pca_scores_plot_widget.canvas.draw()

    def _initial_pca_loadings_plot(self) -> None:
        self.ui.pca_loadings_plot_widget.canvas.axes.cla()
        self.ui.pca_loadings_plot_widget.canvas.axes.set_xlabel('PC-1')
        self.ui.pca_loadings_plot_widget.canvas.axes.set_ylabel('PC-2')
        self.ui.pca_loadings_plot_widget.canvas.draw()

    def _initial_plsda_scores_plot(self) -> None:
        self.ui.plsda_scores_plot_widget.canvas.axes.cla()
        self.ui.plsda_scores_plot_widget.canvas.axes.set_xlabel('PLS-DA-1')
        self.ui.plsda_scores_plot_widget.canvas.axes.set_ylabel('PLS-DA-2')
        self.ui.plsda_scores_plot_widget.canvas.draw()

    def _initial_plsda_vip_plot(self) -> None:
        self.ui.plsda_vip_plot_widget.canvas.axes.cla()
        self.ui.plsda_vip_plot_widget.canvas.axes.set_xlabel('PLS-DA-1')
        self.ui.plsda_vip_plot_widget.canvas.axes.set_ylabel('PLS-DA-2')
        self.ui.plsda_vip_plot_widget.canvas.draw()

    def web_view_print_pdf(self, page):
        fd = QFileDialog()
        file_path = fd.getSaveFileName(self, 'Print page to PDF',
                                       '/users/' + str(environ.get('USERNAME')) + '/Documents/RS-tool', "PDF (*.pdf)")
        if file_path[0] == '':
            return
        ps = QPageSize(QPageSize.A4)
        pl = QPageLayout(ps, QPageLayout.Orientation.Landscape, QMarginsF())
        page.printToPdf(file_path[0], pageLayout=pl)

    @asyncSlot()
    async def _initial_stat_plots_color(self) -> None:
        plot_widgets = [self.ui.lda_roc_plot, self.ui.lda_dm_plot, self.ui.lda_features_plot_widget,
                        self.ui.lda_scores_2d_plot_widget, self.ui.lda_pr_plot, self.ui.lda_shap_means,
                        self.ui.lda_shap_beeswarm, self.ui.lda_shap_scatter, self.ui.lda_shap_heatmap,
                        self.ui.lda_shap_decision, self.ui.lda_shap_waterfall,
                        self.ui.qda_roc_plot, self.ui.qda_dm_plot, self.ui.qda_scores_plot_widget, self.ui.qda_pr_plot,
                        self.ui.qda_shap_beeswarm, self.ui.qda_shap_heatmap, self.ui.qda_shap_means,
                        self.ui.qda_shap_scatter, self.ui.qda_shap_waterfall,
                        self.ui.lr_roc_plot, self.ui.lr_pr_plot, self.ui.lr_scores_plot_widget,
                        self.ui.lr_features_plot_widget, self.ui.lr_dm_plot, self.ui.lr_shap_beeswarm,
                        self.ui.lr_shap_heatmap, self.ui.lr_shap_means, self.ui.lr_shap_scatter,
                        self.ui.lr_shap_waterfall,  self.ui.lr_shap_decision,
                        self.ui.svc_pr_plot, self.ui.svc_dm_plot, self.ui.svc_roc_plot,
                        self.ui.svc_features_plot_widget, self.ui.svc_scores_plot_widget, self.ui.svc_shap_beeswarm,
                        self.ui.svc_shap_decision, self.ui.svc_shap_heatmap, self.ui.svc_shap_means,
                        self.ui.svc_shap_scatter, self.ui.svc_shap_waterfall,
                        self.ui.nearest_pr_plot, self.ui.nearest_dm_plot, self.ui.nearest_roc_plot,
                        self.ui.nearest_scores_plot_widget, self.ui.nearest_shap_beeswarm,
                        self.ui.nearest_shap_decision, self.ui.nearest_shap_heatmap, self.ui.nearest_shap_means,
                        self.ui.nearest_shap_scatter, self.ui.nearest_shap_waterfall,
                        self.ui.gpc_dm_plot, self.ui.gpc_pr_plot, self.ui.gpc_roc_plot, self.ui.gpc_scores_plot_widget,
                        self.ui.gpc_shap_beeswarm, self.ui.gpc_shap_decision, self.ui.gpc_shap_heatmap,
                        self.ui.gpc_shap_means, self.ui.gpc_shap_scatter, self.ui.gpc_shap_waterfall,
                        self.ui.dt_dm_plot, self.ui.dt_features_plot_widget, self.ui.dt_pr_plot, self.ui.dt_roc_plot,
                        self.ui.dt_scores_plot_widget, self.ui.dt_shap_beeswarm, self.ui.dt_shap_decision,
                        self.ui.dt_shap_heatmap, self.ui.dt_shap_means, self.ui.dt_shap_scatter,
                        self.ui.dt_shap_waterfall,
                        self.ui.nb_dm_plot, self.ui.nb_pr_plot, self.ui.nb_roc_plot,
                        self.ui.nb_scores_plot_widget, self.ui.nb_shap_beeswarm, self.ui.nb_shap_decision,
                        self.ui.nb_shap_heatmap, self.ui.nb_shap_means, self.ui.nb_shap_scatter,
                        self.ui.nb_shap_waterfall,
                        self.ui.rf_dm_plot, self.ui.rf_features_plot_widget, self.ui.rf_pr_plot, self.ui.rf_roc_plot,
                        self.ui.rf_scores_plot_widget, self.ui.rf_shap_beeswarm, self.ui.rf_shap_decision,
                        self.ui.rf_shap_heatmap, self.ui.rf_shap_means, self.ui.rf_shap_scatter,
                        self.ui.rf_shap_waterfall,
                        self.ui.ab_dm_plot, self.ui.ab_features_plot_widget, self.ui.ab_pr_plot, self.ui.ab_roc_plot,
                        self.ui.ab_scores_plot_widget, self.ui.ab_shap_beeswarm, self.ui.ab_shap_decision,
                        self.ui.ab_shap_heatmap, self.ui.ab_shap_means, self.ui.ab_shap_scatter,
                        self.ui.ab_shap_waterfall,
                        self.ui.mlp_dm_plot, self.ui.mlp_pr_plot, self.ui.mlp_roc_plot,
                        self.ui.mlp_scores_plot_widget, self.ui.mlp_shap_beeswarm, self.ui.mlp_shap_decision,
                        self.ui.mlp_shap_heatmap, self.ui.mlp_shap_means, self.ui.mlp_shap_scatter,
                        self.ui.mlp_shap_waterfall,
                        self.ui.pca_scores_plot_widget, self.ui.pca_loadings_plot_widget
                        ]
        for pl in plot_widgets:
            self.set_canvas_colors(pl.canvas)
        if self.ui.current_group_shap_comboBox.currentText() == '':
            return
        await self.loop.run_in_executor(None, self._update_shap_plots)
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        self.update_force_single_plots()
        self.update_force_full_plots()

    def _update_shap_plots(self) -> None:
        self.do_update_shap_plots('LDA')
        self.do_update_shap_plots('QDA')
        self.do_update_shap_plots('Logistic regression')
        self.do_update_shap_plots('NuSVC')
        self.do_update_shap_plots('Nearest Neighbors')
        self.do_update_shap_plots('GPC')
        self.do_update_shap_plots('Decision Tree')
        self.do_update_shap_plots('Naive Bayes')
        self.do_update_shap_plots('Random Forest')
        self.do_update_shap_plots('AdaBoost')
        self.do_update_shap_plots('MLP')

    def _update_shap_plots_by_instance(self) -> None:
        self.do_update_shap_plots_by_instance('LDA')
        self.do_update_shap_plots_by_instance('QDA')
        self.do_update_shap_plots_by_instance('Logistic regression')
        self.do_update_shap_plots_by_instance('NuSVC')
        self.do_update_shap_plots_by_instance('Nearest Neighbors')
        self.do_update_shap_plots_by_instance('GPC')
        self.do_update_shap_plots_by_instance('Decision Tree')
        self.do_update_shap_plots_by_instance('Naive Bayes')
        self.do_update_shap_plots_by_instance('Random Forest')
        self.do_update_shap_plots_by_instance('AdaBoost')
        self.do_update_shap_plots_by_instance('MLP')

    def do_update_shap_plots(self, classificator_type):
        if classificator_type not in self.latest_stat_result \
                or 'target_names' not in self.latest_stat_result[classificator_type]:
            return
        target_names = self.latest_stat_result[classificator_type]['target_names']
        num = np.where(target_names == self.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_means_plot(False, i, classificator_type)
        self.update_shap_beeswarm_plot(False, i, classificator_type)
        self.update_shap_scatter_plot(False, i, classificator_type)
        self.update_shap_heatmap_plot(False, i, classificator_type)

    def do_update_shap_plots_by_instance(self, classificator_type):
        if classificator_type not in self.latest_stat_result \
                or 'target_names' not in self.latest_stat_result[classificator_type]:
            return
        target_names = self.latest_stat_result[classificator_type]['target_names']
        num = np.where(target_names == self.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_force(i, classificator_type)
        self.update_shap_force(i, classificator_type, True)
        self.update_shap_decision_plot(i, classificator_type)
        self.update_shap_waterfall_plot(i, classificator_type)

    @asyncSlot()
    async def update_shap_scatters(self) -> None:
        print('update_shap_scatters')
        self.loop.run_in_executor(None, self.do_update_shap_scatters)

    def do_update_shap_scatters(self):
        if self.ui.stat_tab_widget.currentIndex() == 0 and 'LDA' in self.latest_stat_result:
            target_names = self.latest_stat_result['LDA']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'LDA')
        elif self.ui.stat_tab_widget.currentIndex() == 1 and 'QDA' in self.latest_stat_result:
            target_names = self.latest_stat_result['QDA']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'QDA')
        elif self.ui.stat_tab_widget.currentIndex() == 2 and 'Logistic regression' in self.latest_stat_result:
            target_names = self.latest_stat_result['Logistic regression']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'Logistic regression')
        elif self.ui.stat_tab_widget.currentIndex() == 3 and 'NuSVC' in self.latest_stat_result:
            target_names = self.latest_stat_result['NuSVC']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'NuSVC')
        elif self.ui.stat_tab_widget.currentIndex() == 4 and 'Nearest Neighbors' in self.latest_stat_result:
            target_names = self.latest_stat_result['Nearest Neighbors']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'Nearest Neighbors')
        elif self.ui.stat_tab_widget.currentIndex() == 5 and 'GPC' in self.latest_stat_result:
            target_names = self.latest_stat_result['GPC']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'GPC')
        elif self.ui.stat_tab_widget.currentIndex() == 6 and 'Decision Tree' in self.latest_stat_result:
            target_names = self.latest_stat_result['Decision Tree']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'Decision Tree')
        elif self.ui.stat_tab_widget.currentIndex() == 7 and 'Naive Bayes' in self.latest_stat_result:
            target_names = self.latest_stat_result['Naive Bayes']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'Naive Bayes')
        elif self.ui.stat_tab_widget.currentIndex() == 8 and 'Random Forest' in self.latest_stat_result:
            target_names = self.latest_stat_result['Random Forest']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'Random Forest')
        elif self.ui.stat_tab_widget.currentIndex() == 9 and 'AdaBoost' in self.latest_stat_result:
            target_names = self.latest_stat_result['AdaBoost']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'AdaBoost')
        elif self.ui.stat_tab_widget.currentIndex() == 10 and 'MLP' in self.latest_stat_result:
            target_names = self.latest_stat_result['MLP']['target_names']
            i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
            self.update_shap_scatter_plot(False, i, 'MLP')
        else:
            return

    def set_canvas_colors(self, canvas) -> None:
        ax = canvas.figure.gca()
        ax.set_facecolor(self.plot_background_color.name())
        canvas.figure.set_facecolor(self.plot_background_color.name())
        ax.tick_params(axis='x', colors=self.plot_text_color.name())
        ax.tick_params(axis='y', colors=self.plot_text_color.name())
        ax.yaxis.label.set_color(self.plot_text_color.name())
        ax.xaxis.label.set_color(self.plot_text_color.name())
        ax.title.set_color(self.plot_text_color.name())
        ax.spines['bottom'].set_color(self.plot_text_color.name())
        ax.spines['top'].set_color(self.plot_text_color.name())
        ax.spines['right'].set_color(self.plot_text_color.name())
        ax.spines['left'].set_color(self.plot_text_color.name())
        leg = ax.get_legend()
        if leg is not None:
            ax.legend(facecolor=self.plot_background_color.name(),
                      labelcolor=self.plot_text_color.name(), prop={'size': 8})
        try:
            canvas.draw()
        except ValueError:
            pass

    def _initial_plots(self) -> None:
        setConfigOption('antialias', True)
        self.ui.plots_tabWidget.setTabText(0, 'Imported, nm')
        self.ui.plots_tabWidget.setTabText(1, 'Converted, cm\N{superscript minus}\N{superscript one}')
        self.ui.plots_tabWidget.tabBar().setUsesScrollButtons(False)
        self.ui.plots_tabWidget.tabBar().currentChanged.connect(self.plots_tab_changed)
        self.ui.stackedWidget_mainpages.currentChanged.connect(self.stacked_widget_changed)
        self._initial_input_plot()
        self._initial_converted_cm_plot()
        self._initial_cut_cm_plot()
        self._initial_normalize_plot()
        self._initial_smooth_plot()
        self._initial_baseline_plot()
        self._initial_averaged_plot()
        self._initial_deconvolution_plot()
        self._initial_all_stat_plots()

    def _initial_lda_plots(self) -> None:
        self._initial_lda_scores_1d_plot()
        self._initial_lda_scores_2d_plot()
        self._initial_lda_features_plot()
        self._initial_lda_roc_plot()
        self._initial_lda_dm_plot()
        self._initial_lda_pr_plot()
        self._initial_lda_shap_means_plot()
        self._initial_lda_shap_beeswarm_plot()
        self._initial_lda_shap_heatmap_plot()
        self._initial_lda_shap_scatter_plot()
        self._initial_lda_shap_waterfall_plot()
        self._initial_lda_shap_decision_plot()
        self._initial_lda_force_single_plot()
        self._initial_lda_force_full_plot()

    def _initial_qda_plots(self) -> None:
        self._initial_qda_scores_2d_plot()
        self._initial_qda_dm_plot()
        self._initial_qda_pr_plot()
        self._initial_qda_roc_plot()
        self._initial_qda_shap_beeswarm_plot()
        self._initial_qda_shap_heatmap_plot()
        self._initial_qda_shap_means_plot()
        self._initial_qda_shap_scatter_plot()
        self._initial_qda_shap_waterfall_plot()

    def _initial_lr_plots(self) -> None:
        self._initial_lr_dm_plot()
        self._initial_lr_features_plot()
        self._initial_lr_pr_plot()
        self._initial_lr_roc_plot()
        self._initial_lr_scores_plot()
        self._initial_lr_shap_beeswarm_plot()
        self._initial_lr_shap_heatmap_plot()
        self._initial_lr_shap_means_plot()
        self._initial_lr_shap_scatter_plot()
        self._initial_lr_force_single_plot()
        self._initial_lr_force_full_plot()
        self._initial_lr_shap_decision_plot()
        self._initial_lr_shap_waterfall_plot()

    def _initial_svc_plots(self) -> None:
        self._initial_svc_dm_plot()
        self._initial_svc_features_plot()
        self._initial_svc_pr_plot()
        self._initial_svc_roc_plot()
        self._initial_svc_scores_plot()
        self._initial_svc_shap_beeswarm_plot()
        self._initial_svc_shap_heatmap_plot()
        self._initial_svc_shap_means_plot()
        self._initial_svc_shap_scatter_plot()
        self._initial_svc_force_single_plot()
        self._initial_svc_force_full_plot()
        self._initial_svc_shap_decision_plot()
        self._initial_svc_shap_waterfall_plot()

    def _initial_nearest_plots(self) -> None:
        self._initial_nearest_dm_plot()
        self._initial_nearest_pr_plot()
        self._initial_nearest_roc_plot()
        self._initial_nearest_scores_plot()
        self._initial_nearest_shap_beeswarm_plot()
        self._initial_nearest_shap_heatmap_plot()
        self._initial_nearest_shap_means_plot()
        self._initial_nearest_shap_scatter_plot()
        self._initial_nearest_force_single_plot()
        self._initial_nearest_force_full_plot()
        self._initial_nearest_shap_decision_plot()
        self._initial_nearest_shap_waterfall_plot()

    def _initial_gpc_plots(self) -> None:
        self._initial_gpc_dm_plot()
        self._initial_gpc_pr_plot()
        self._initial_gpc_roc_plot()
        self._initial_gpc_scores_plot()
        self._initial_gpc_shap_beeswarm_plot()
        self._initial_gpc_shap_heatmap_plot()
        self._initial_gpc_shap_means_plot()
        self._initial_gpc_shap_scatter_plot()
        self._initial_gpc_force_single_plot()
        self._initial_gpc_force_full_plot()
        self._initial_gpc_shap_decision_plot()
        self._initial_gpc_shap_waterfall_plot()

    def _initial_dt_plots(self) -> None:
        self._initial_dt_dm_plot()
        self._initial_dt_pr_plot()
        self._initial_dt_roc_plot()
        self._initial_dt_scores_plot()
        self._initial_dt_features_plot()
        self._initial_dt_shap_beeswarm_plot()
        self._initial_dt_shap_heatmap_plot()
        self._initial_dt_shap_means_plot()
        self._initial_dt_shap_scatter_plot()
        self._initial_dt_force_single_plot()
        self._initial_dt_force_full_plot()
        self._initial_dt_shap_decision_plot()
        self._initial_dt_shap_waterfall_plot()

    def _initial_nb_plots(self) -> None:
        self._initial_nb_dm_plot()
        self._initial_nb_pr_plot()
        self._initial_nb_roc_plot()
        self._initial_nb_scores_plot()
        self._initial_nb_shap_beeswarm_plot()
        self._initial_nb_shap_heatmap_plot()
        self._initial_nb_shap_means_plot()
        self._initial_nb_shap_scatter_plot()
        self._initial_nb_force_single_plot()
        self._initial_nb_force_full_plot()
        self._initial_nb_shap_decision_plot()
        self._initial_nb_shap_waterfall_plot()

    def _initial_rf_plots(self) -> None:
        self._initial_rf_dm_plot()
        self._initial_rf_pr_plot()
        self._initial_rf_roc_plot()
        self._initial_rf_scores_plot()
        self._initial_rf_features_plot()
        self._initial_rf_shap_beeswarm_plot()
        self._initial_rf_shap_heatmap_plot()
        self._initial_rf_shap_means_plot()
        self._initial_rf_shap_scatter_plot()
        self._initial_rf_force_single_plot()
        self._initial_rf_force_full_plot()
        self._initial_rf_shap_decision_plot()
        self._initial_rf_shap_waterfall_plot()

    def _initial_ab_plots(self) -> None:
        self._initial_ab_dm_plot()
        self._initial_ab_pr_plot()
        self._initial_ab_roc_plot()
        self._initial_ab_scores_plot()
        self._initial_ab_features_plot()
        self._initial_ab_shap_beeswarm_plot()
        self._initial_ab_shap_heatmap_plot()
        self._initial_ab_shap_means_plot()
        self._initial_ab_shap_scatter_plot()
        self._initial_ab_force_single_plot()
        self._initial_ab_force_full_plot()
        self._initial_ab_shap_decision_plot()
        self._initial_ab_shap_waterfall_plot()

    def _initial_mlp_plots(self) -> None:
        self._initial_mlp_dm_plot()
        self._initial_mlp_pr_plot()
        self._initial_mlp_roc_plot()
        self._initial_mlp_scores_plot()
        self._initial_mlp_shap_beeswarm_plot()
        self._initial_mlp_shap_heatmap_plot()
        self._initial_mlp_shap_means_plot()
        self._initial_mlp_shap_scatter_plot()
        self._initial_mlp_force_single_plot()
        self._initial_mlp_force_full_plot()
        self._initial_mlp_shap_decision_plot()
        self._initial_mlp_shap_waterfall_plot()

    def _initial_pca_plots(self) -> None:
        self._initial_pca_scores_plot()
        self._initial_pca_loadings_plot()

    def _initial_plsda_plots(self) -> None:
        self._initial_plsda_scores_plot()
        self._initial_plsda_vip_plot()

    def _initial_all_stat_plots(self) -> None:
        self._initial_lda_plots()
        self._initial_qda_plots()
        self._initial_lr_plots()
        self._initial_svc_plots()
        self._initial_nearest_plots()
        self._initial_gpc_plots()
        self._initial_dt_plots()
        self._initial_nb_plots()
        self._initial_rf_plots()
        self._initial_ab_plots()
        self._initial_mlp_plots()
        self._initial_pca_plots()
        self._initial_plsda_plots()

        self._initial_plots_set_fonts()
        self._initial_plots_set_labels_font()
        self._initial_stat_plots_color()

        self.ui.splitter_h_lda_plots_stats.setStretchFactor(1, 10)
        self.ui.splitter_h_qda_plots_stats.setStretchFactor(1, 10)
        self.ui.splitter_6.setStretchFactor(1, 10)
        self.ui.splitter_16.setStretchFactor(1, 10)
        self.ui.splitter_25.setStretchFactor(1, 10)
        self.ui.splitter_34.setStretchFactor(1, 10)
        self.ui.splitter_43.setStretchFactor(1, 10)
        self.ui.splitter_52.setStretchFactor(1, 10)
        self.ui.splitter_61.setStretchFactor(1, 10)
        self.ui.splitter_70.setStretchFactor(1, 10)
        self.ui.splitter_79.setStretchFactor(1, 10)

        self.ui.splitter_15.setStretchFactor(3, 2)
        self.ui.splitter_3.setStretchFactor(3, 2)
        self.ui.splitter_10.setStretchFactor(3, 2)
        self.ui.splitter_20.setStretchFactor(3, 2)
        self.ui.splitter_29.setStretchFactor(3, 2)
        self.ui.splitter_38.setStretchFactor(3, 2)
        self.ui.splitter_47.setStretchFactor(3, 2)
        self.ui.splitter_56.setStretchFactor(3, 2)
        self.ui.splitter_65.setStretchFactor(3, 2)
        self.ui.splitter_74.setStretchFactor(3, 2)
        self.ui.splitter_83.setStretchFactor(3, 2)

        self.ui.lda_scores_1d_plot_widget.setVisible(False)

    def _initial_plots_set_fonts(self) -> None:
        plot_font = QFont("AbletonSans", self.plot_font_size, QFont.Weight.Normal)
        self.input_plot_widget_plot_item.getAxis("bottom").setStyle(tickFont=plot_font)
        self.input_plot_widget_plot_item.getAxis("left").setStyle(tickFont=plot_font)
        self.converted_cm_widget_plot_item.getAxis("bottom").setStyle(tickFont=plot_font)
        self.converted_cm_widget_plot_item.getAxis("left").setStyle(tickFont=plot_font)
        self.cut_cm_plotItem.getAxis("bottom").setStyle(tickFont=plot_font)
        self.cut_cm_plotItem.getAxis("left").setStyle(tickFont=plot_font)
        self.normalize_plotItem.getAxis("bottom").setStyle(tickFont=plot_font)
        self.normalize_plotItem.getAxis("left").setStyle(tickFont=plot_font)
        self.smooth_plotItem.getAxis("bottom").setStyle(tickFont=plot_font)
        self.smooth_plotItem.getAxis("left").setStyle(tickFont=plot_font)
        self.baseline_corrected_plotItem.getAxis("bottom").setStyle(tickFont=plot_font)
        self.baseline_corrected_plotItem.getAxis("left").setStyle(tickFont=plot_font)
        self.averaged_plotItem.getAxis("bottom").setStyle(tickFont=plot_font)
        self.averaged_plotItem.getAxis("left").setStyle(tickFont=plot_font)
        self.deconvolution_plotItem.getAxis("bottom").setStyle(tickFont=plot_font)
        self.deconvolution_plotItem.getAxis("left").setStyle(tickFont=plot_font)
        self.lda_scores_1d_plot_item.getAxis("bottom").setStyle(tickFont=plot_font)
        self.lda_scores_1d_plot_item.getAxis("left").setStyle(tickFont=plot_font)

    def _initial_plots_set_labels_font(self) -> None:
        font_size = self.axis_label_font_size + 'pt'
        label_style = {'color': self.plot_text_color_value, 'font-size': font_size, 'font-family': 'AbletonSans'}
        self.ui.input_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.input_plot_widget.setLabel('bottom', 'Wavelength, nm', units='', **label_style)
        self.ui.converted_cm_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.converted_cm_plot_widget.setLabel('bottom', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                                  units='', **label_style)
        self.ui.cut_cm_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.cut_cm_plot_widget.setLabel('bottom', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                            units='', **label_style)
        self.ui.normalize_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.normalize_plot_widget.setLabel('bottom', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                               units='', **label_style)
        self.ui.smooth_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.smooth_plot_widget.setLabel('bottom', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                            units='', **label_style)
        self.ui.baseline_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.baseline_plot_widget.setLabel('bottom', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                              units='', **label_style)
        self.ui.average_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.average_plot_widget.setLabel('bottom', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                             units='', **label_style)
        self.ui.deconv_plot_widget.setLabel('left', 'Intensity, rel. un.', units='', **label_style)
        self.ui.deconv_plot_widget.setLabel('bottom', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                            units='', **label_style)
        self.ui.lda_scores_1d_plot_widget.setLabel('left', 'Samples number', units='', **label_style)
        self.ui.lda_scores_1d_plot_widget.setLabel('bottom', 'LD-1', units='', **label_style)

    # endregion

    # region init menu

    def initial_menu(self) -> None:
        self._init_file_menu()
        self._init_edit_menu()
        self._init_process_menu()
        self._init_stat_analysis_menu()

    def _init_file_menu(self) -> None:
        self.file_menu = QMenu(self)
        self.file_menu.addAction('New Project', self.action_new_project)
        self.file_menu.addAction('Open Project', self.action_open_project)
        self.recent_menu = self.file_menu.addMenu('Open Recent')
        self.update_recent_list()
        self.file_menu.addSeparator()

        self.file_menu_import_action = QAction('Import files')
        self.file_menu_import_action.triggered.connect(self.importfile_clicked)
        self.file_menu_import_action.setShortcut("Ctrl+I")
        actions = [self.file_menu_import_action]
        self.file_menu.addActions(actions)
        self.file_menu.addSeparator()

        self.export_menu = self.file_menu.addMenu('Export')
        self.export_menu.addAction('Files, nm', self.action_export_files_nm)
        self.export_menu.addAction('Files, cm\N{superscript minus}\N{superscript one}', self.action_export_files_cm)
        self.export_menu.addAction('Average, cm\N{superscript minus}\N{superscript one}', self.action_export_average)
        self.export_menu.addAction('Tables to excel', self.action_export_table_excel)
        self.export_menu.addAction('Production project', self.action_save_production_project)

        self.fit_template_menu = self.file_menu.addMenu('Fit template')
        self.fit_template_menu.addAction('Import', self.action_import_fit_template)
        self.fit_template_menu.addAction('Export', self.action_export_fit_template)

        self.file_menu.addSeparator()
        self.file_menu_save_all_action = QAction('Save all')
        self.file_menu_save_all_action.triggered.connect(self.action_save_project)
        self.file_menu_save_all_action.setShortcut("Ctrl+S")
        self.file_menu_save_as_action = QAction('Save as')
        self.file_menu_save_as_action.triggered.connect(self.action_save_as)
        self.file_menu_save_as_action.setShortcut("Shift+S")
        actions = [self.file_menu_save_all_action, self.file_menu_save_as_action]
        self.file_menu.addActions(actions)
        self.file_menu.addSeparator()

        self.file_menu.addAction('Close Project', self.action_close_project)
        self.file_menu.addSeparator()
        self.file_menu_help = QAction('Help')
        self.file_menu_help.triggered.connect(self.action_help)
        self.file_menu_help.setShortcut("F1")
        self.file_menu.addActions([self.file_menu_help])
        self.ui.FileBtn.setMenu(self.file_menu)

    def _init_edit_menu(self) -> None:
        self.edit_menu = QMenu(self)
        self.action_undo = QAction('Undo')
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_redo = QAction('Redo')
        self.action_redo.triggered.connect(self.redo)
        self.action_redo.setShortcut("Ctrl+Y")
        actions = [self.action_undo, self.action_redo]
        self.edit_menu.addActions(actions)
        self.edit_menu.setToolTipsVisible(True)
        self.action_undo.setToolTip('')
        self.edit_menu.addSeparator()

        self.clear_menu = self.edit_menu.addMenu('Clear')
        self.clear_menu.addAction('Converted', lambda: self.clear_selected_step('Converted'))
        self.clear_menu.addAction('Cutted', lambda: self.clear_selected_step('Cutted'))
        self.clear_menu.addAction('Normalized', lambda: self.clear_selected_step('Normalized'))
        self.clear_menu.addAction('Smoothed', lambda: self.clear_selected_step('Smoothed'))
        self.clear_menu.addAction('Baseline corrected', lambda: self.clear_selected_step('Baseline'))
        self.clear_menu.addAction('Averaged', lambda: self.clear_selected_step('Averaged'))
        self.clear_menu.addSeparator()
        self.clear_menu.addAction('Fit', self.clear_all_deconv_lines)
        self.clear_menu.addSeparator()
        self.clear_menu.addAction('Smoothed dataset', self._initial_smoothed_dataset_table)
        self.clear_menu.addAction('Baseline corrected dataset', self._initial_baselined_dataset_table)
        self.clear_menu.addAction('Deconvoluted dataset', self.clear_deconvoluted_dataset_table)
        self.clear_menu.addSeparator()
        self.clear_menu.addAction('LDA', lambda: self.clear_selected_step('LDA'))
        self.clear_menu.addAction('QDA', lambda: self.clear_selected_step('QDA'))
        self.clear_menu.addAction('Logistic regression', lambda: self.clear_selected_step('Logistic regression'))
        self.clear_menu.addAction('NuSVC', lambda: self.clear_selected_step('NuSVC'))
        self.clear_menu.addAction('Nearest Neighbors', lambda: self.clear_selected_step('Nearest Neighbors'))
        self.clear_menu.addAction('GPC', lambda: self.clear_selected_step('GPC'))
        self.clear_menu.addAction('Decision Tree', lambda: self.clear_selected_step('Decision Tree'))
        self.clear_menu.addAction('Naive Bayes', lambda: self.clear_selected_step('Naive Bayes'))
        self.clear_menu.addAction('Random Forest', lambda: self.clear_selected_step('Random Forest'))
        self.clear_menu.addAction('AdaBoost', lambda: self.clear_selected_step('AdaBoost'))
        self.clear_menu.addAction('MLP', lambda: self.clear_selected_step('MLP'))
        self.clear_menu.addAction('PCA', lambda: self.clear_selected_step('PCA'))
        self.clear_menu.addAction('PLS-DA', lambda: self.clear_selected_step('PLS-DA'))
        self.clear_menu.addSeparator()
        self.clear_menu.addAction('Predicted', lambda: self.clear_selected_step('Page5'))
        self.ui.EditBtn.setMenu(self.edit_menu)

    def _init_process_menu(self) -> None:
        self.process_menu = QMenu(self)
        self.action_interpolate = QAction('Interpolate')
        self.action_interpolate.triggered.connect(self.interpolate)
        self.action_interpolate.setShortcut("Alt+I")
        self.action_despike = QAction('Despike')
        self.action_despike.triggered.connect(self.despike)
        self.action_despike.setShortcut("Alt+D")
        self.action_convert = QAction('Convert to cm\N{superscript minus}\N{superscript one}')
        self.action_convert.triggered.connect(self.convert)
        self.action_convert.setShortcut("Alt+C")
        self.action_cut = QAction('Cut spectrum')
        self.action_cut.triggered.connect(self.cut_first)
        self.action_cut.setShortcut("Alt+U")
        self.action_normalize = QAction('Normalization')
        self.action_normalize.triggered.connect(self.normalize)
        self.action_normalize.setShortcut("Alt+N")
        self.action_smooth = QAction('Smooth')
        self.action_smooth.triggered.connect(self.smooth)
        self.action_smooth.setShortcut("Alt+S")
        self.action_baseline_correction = QAction('Baseline correction')
        self.action_baseline_correction.triggered.connect(self.baseline_correction)
        self.action_baseline_correction.setShortcut("Alt+B")
        self.action_trim = QAction('Final trim')
        self.action_trim.triggered.connect(self.trim)
        self.action_trim.setShortcut("Alt+T")
        self.action_average = QAction('Average')
        self.action_average.triggered.connect(self.update_averaged)
        self.action_average.setShortcut("Alt+A")
        actions = [self.action_interpolate, self.action_despike, self.action_convert, self.action_cut,
                   self.action_normalize, self.action_smooth, self.action_baseline_correction, self.action_trim,
                   self.action_average]
        self.process_menu.addActions(actions)
        self.ui.ProcessBtn.setMenu(self.process_menu)

    def _init_stat_analysis_menu(self) -> None:
        self.stat_analysis_menu = QMenu(self)
        self.action_fit_lda = QAction('LDA')
        self.action_fit_lda.triggered.connect(lambda: self.fit_classificator('LDA'))
        self.action_fit_qda = QAction('QDA')
        self.action_fit_qda.triggered.connect(lambda: self.fit_classificator('QDA'))
        self.action_fit_lr = QAction('Logistic regression')
        self.action_fit_lr.triggered.connect(lambda: self.fit_classificator('Logistic regression'))
        self.action_fit_svc = QAction('NuSVC')
        self.action_fit_svc.triggered.connect(lambda: self.fit_classificator('NuSVC'))
        self.action_fit_sgd = QAction('Nearest Neighbors')
        self.action_fit_sgd.triggered.connect(lambda: self.fit_classificator('Nearest Neighbors'))
        self.action_fit_gpc = QAction('GPC')
        self.action_fit_gpc.triggered.connect(lambda: self.fit_classificator('GPC'))
        self.action_fit_dt = QAction('Decision Tree')
        self.action_fit_dt.triggered.connect(lambda: self.fit_classificator('Decision Tree'))
        self.action_fit_nb = QAction('Naive Bayes')
        self.action_fit_nb.triggered.connect(lambda: self.fit_classificator('Naive Bayes'))
        self.action_fit_rf = QAction('Random Forest')
        self.action_fit_rf.triggered.connect(lambda: self.fit_classificator('Random Forest'))
        self.action_fit_ab = QAction('AdaBoost')
        self.action_fit_ab.triggered.connect(lambda: self.fit_classificator('AdaBoost'))
        self.action_fit_mlp = QAction('MLP')
        self.action_fit_mlp.triggered.connect(lambda: self.fit_classificator('MLP'))
        self.action_fit_pca = QAction('PCA')
        self.action_fit_pca.triggered.connect(lambda: self.fit_classificator('PCA'))
        self.action_fit_plsda = QAction('PLS-DA')
        self.action_fit_plsda.triggered.connect(lambda: self.fit_classificator('PLS-DA'))
        self.action_redraw_plots = QAction('Redraw plots')
        self.action_redraw_plots.triggered.connect(self.redraw_stat_plots)
        actions = [self.action_fit_lda, self.action_fit_qda, self.action_fit_lr, self.action_fit_svc,
                   self.action_fit_sgd, self.action_fit_gpc, self.action_fit_dt, self.action_fit_nb,
                   self.action_fit_rf, self.action_fit_ab, self.action_fit_mlp, self.action_fit_pca,
                   self.action_fit_plsda, self.action_redraw_plots]
        self.stat_analysis_menu.addActions(actions)
        self.ui.stat_analysis_btn.setMenu(self.stat_analysis_menu)

    # endregion

    # region left_side_menu

    def _init_left_menu(self) -> None:
        self.ui.left_hide_frame.hide()
        self.ui.dec_list_btn.setVisible(False)
        self.ui.gt_add_Btn.setToolTip('Add new group')
        self.ui.gt_add_Btn.clicked.connect(self.add_new_group)
        self.ui.gt_dlt_Btn.setToolTip('Delete selected group')
        self.ui.gt_dlt_Btn.clicked.connect(self.dlt_selected_group)
        self._init_params_value_changed()
        self._init_baseline_correction_method_combo_box()
        self._init_cost_func_combo_box()
        self._init_normalizing_method_combo_box()
        self._init_opt_method_oer_combo_box()
        self._init_fit_opt_method_combo_box()
        self._init_guess_method_cb()
        self._init_params_mouse_double_click_event()
        self._init_n_lines_method()
        self._init_average_function_cb()
        self._init_dataset_type_cb()
        self._init_current_feature_cb()
        self._init_coloring_feature_cb()
        self.ui.edit_template_btn.clicked.connect(self.switch_to_template)
        self.ui.template_combo_box.currentTextChanged.connect(self.switch_to_template)
        self.ui.current_group_shap_comboBox.currentTextChanged.connect(self.current_group_shap_changed)
        self.ui.intervals_gb.toggled.connect(self.intervals_gb_toggled)
        self._init_smoothing_method_combo_box()
        self.normalization_method = self.default_values['normalizing_method_comboBox']
        self.smooth_method = ''
        self.baseline_method = ''

    def _init_baseline_correction_method_combo_box(self) -> None:
        for i in self.baseline_methods.keys():
            self.ui.baseline_correction_method_comboBox.addItem(i)

    def _init_cost_func_combo_box(self) -> None:
        self.ui.cost_func_comboBox.addItem('asymmetric_truncated_quadratic')
        self.ui.cost_func_comboBox.addItem('symmetric_truncated_quadratic')
        self.ui.cost_func_comboBox.addItem('asymmetric_huber')
        self.ui.cost_func_comboBox.addItem('symmetric_huber')
        self.ui.cost_func_comboBox.addItem('asymmetric_indec')
        self.ui.cost_func_comboBox.addItem('symmetric_indec')

    def _init_normalizing_method_combo_box(self) -> None:
        for i in self.normalize_methods.keys():
            self.ui.normalizing_method_comboBox.addItem(i)

    def _init_opt_method_oer_combo_box(self) -> None:
        for i in optimize_extended_range_methods():
            self.ui.opt_method_oer_comboBox.addItem(i)

    def _init_fit_opt_method_combo_box(self) -> None:
        for key in self.fitting_methods:
            self.ui.fit_opt_method_comboBox.addItem(key)

    def _init_params_mouse_double_click_event(self) -> None:
        self.ui.laser_wl_spinbox.mouseDoubleClickEvent = self._laser_wl_spinbox_mouse_dce
        self.ui.max_CCD_value_spinBox.mouseDoubleClickEvent = self._max_ccd_value_sb_mouse_dce
        self.ui.maxima_count_despike_spin_box.mouseDoubleClickEvent = self._maxima_count_despike_spin_box_mouse_dce
        self.ui.despike_fwhm_width_doubleSpinBox.mouseDoubleClickEvent = self._despike_fwhm_dsb_mouse_dce
        self.ui.neg_grad_factor_spinBox.mouseDoubleClickEvent = self._neg_grad_factor_sb_mouse_dce
        self.ui.cm_range_start.mouseDoubleClickEvent = self._cm_range_start_mouse_dce
        self.ui.cm_range_end.mouseDoubleClickEvent = self._cm_range_end_mouse_dce
        self.ui.interval_start_dsb.mouseDoubleClickEvent = self._interval_start_mouse_dce
        self.ui.interval_end_dsb.mouseDoubleClickEvent = self._interval_end_mouse_dce
        self.ui.trim_start_cm.mouseDoubleClickEvent = self._trim_start_cm_mouse_dce
        self.ui.trim_end_cm.mouseDoubleClickEvent = self._trim_end_cm_mouse_dce
        self.ui.emsc_pca_n_spinBox.mouseDoubleClickEvent = self._emsc_pca_n_sb_mouse_dce
        self.ui.window_length_spinBox.mouseDoubleClickEvent = self._window_length_sb_mouse_dce
        self.ui.smooth_polyorder_spinBox.mouseDoubleClickEvent = self._smooth_polyorder_sb_mouse_dce
        self.ui.whittaker_lambda_spinBox.mouseDoubleClickEvent = self._whittaker_lambda_sb_mouse_dce
        self.ui.kaiser_beta_doubleSpinBox.mouseDoubleClickEvent = self._kaiser_beta_dsb_mouse_dce
        self.ui.emd_noise_modes_spinBox.mouseDoubleClickEvent = self._emd_noise_modes_sb_mouse_dce
        self.ui.eemd_trials_spinBox.mouseDoubleClickEvent = self._eemd_trials_sb_mouse_dce
        self.ui.sigma_spinBox.mouseDoubleClickEvent = self._sigma_sb_mouse_dce
        self.ui.lambda_spinBox.mouseDoubleClickEvent = self._lambda_sb_mouse_dce
        self.ui.p_doubleSpinBox.mouseDoubleClickEvent = self._p_dsb_mouse_dce
        self.ui.eta_doubleSpinBox.mouseDoubleClickEvent = self._eta_dsb_mouse_dce
        self.ui.n_iterations_spinBox.mouseDoubleClickEvent = self._n_iterations_sb_mouse_dce
        self.ui.polynome_degree_spinBox.mouseDoubleClickEvent = self._polynome_degree_sb_mouse_dce
        self.ui.grad_doubleSpinBox.mouseDoubleClickEvent = self._grad_dsb_mouse_dce
        self.ui.quantile_doubleSpinBox.mouseDoubleClickEvent = self._quantile_dsb_mouse_dce
        self.ui.alpha_factor_doubleSpinBox.mouseDoubleClickEvent = self._alpha_factor_dsb_mouse_dce
        self.ui.fraction_doubleSpinBox.mouseDoubleClickEvent = self._fraction_dsb_mouse_dce
        self.ui.scale_doubleSpinBox.mouseDoubleClickEvent = self._scale_dsb_mouse_dce
        self.ui.peak_ratio_doubleSpinBox.mouseDoubleClickEvent = self._peak_ratio_dsb_mouse_dce
        self.ui.spline_degree_spinBox.mouseDoubleClickEvent = self._spline_degree_sb_mouse_dce
        self.ui.num_std_doubleSpinBox.mouseDoubleClickEvent = self._num_std_dsb_mouse_dce
        self.ui.interp_half_window_spinBox.mouseDoubleClickEvent = self._interp_half_window_dsb_mouse_dce
        self.ui.min_length_spinBox.mouseDoubleClickEvent = self._min_length_mouse_dce
        self.ui.sections_spinBox.mouseDoubleClickEvent = self._sections_mouse_dce
        self.ui.fill_half_window_spinBox.mouseDoubleClickEvent = self._fill_half_window_mouse_dce

    def _init_params_value_changed(self) -> None:
        self.ui.alpha_factor_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.baseline_correction_method_comboBox.currentTextChanged.connect(self.set_baseline_parameters_disabled)
        self.ui.cm_range_start.valueChanged.connect(self.cm_range_start_change_event)
        self.ui.cm_range_end.valueChanged.connect(self.cm_range_end_change_event)
        self.ui.cost_func_comboBox.currentTextChanged.connect(self.set_modified)
        self.ui.despike_fwhm_width_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.eemd_trials_spinBox.valueChanged.connect(self.set_modified)
        self.ui.emd_noise_modes_spinBox.valueChanged.connect(self.set_modified)
        self.ui.eta_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.emsc_pca_n_spinBox.valueChanged.connect(self.set_modified)
        self.ui.fill_half_window_spinBox.valueChanged.connect(self.set_modified)
        self.ui.grad_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.interp_half_window_spinBox.valueChanged.connect(self.set_modified)
        self.ui.interval_start_dsb.valueChanged.connect(self.set_modified)
        self.ui.interval_end_dsb.valueChanged.connect(self.set_modified)
        self.ui.kaiser_beta_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.lambda_spinBox.valueChanged.connect(self.set_modified)
        self.ui.laser_wl_spinbox.valueChanged.connect(self.set_modified)
        self.ui.leftsideBtn.clicked.connect(self.leftside_btn_clicked)
        self.ui.dec_list_btn.clicked.connect(self.dec_list_btn_clicked)
        self.ui.stat_param_btn.clicked.connect(self.stat_param_btn_clicked)
        self.ui.max_CCD_value_spinBox.valueChanged.connect(self.set_modified)
        self.ui.maxima_count_despike_spin_box.valueChanged.connect(self.set_modified)
        self.ui.min_length_spinBox.valueChanged.connect(self.set_modified)
        self.ui.n_iterations_spinBox.valueChanged.connect(self.set_modified)
        self.ui.neg_grad_factor_spinBox.valueChanged.connect(self.set_modified)
        self.ui.normalizing_method_comboBox.currentTextChanged.connect(self.set_modified)
        self.ui.num_std_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.opt_method_oer_comboBox.currentTextChanged.connect(self.set_modified)
        self.ui.p_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.peak_ratio_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.polynome_degree_spinBox.valueChanged.connect(self.set_modified)
        self.ui.quantile_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.sections_spinBox.valueChanged.connect(self.set_modified)
        self.ui.scale_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.max_noise_level_dsb.valueChanged.connect(self.set_modified)
        self.ui.l_ratio_doubleSpinBox.valueChanged.connect(self.set_modified)
        self.ui.sigma_spinBox.valueChanged.connect(self.set_modified)
        self.ui.smooth_polyorder_spinBox.valueChanged.connect(self.set_modified)
        self.ui.smoothing_method_comboBox.currentTextChanged.connect(self.set_smooth_parameters_disabled)
        self.ui.average_method_cb.currentTextChanged.connect(self.set_modified)
        self.ui.dataset_type_cb.currentTextChanged.connect(self.set_modified)
        self.ui.classes_lineEdit.textChanged.connect(self.set_modified)
        self.ui.test_data_ratio_spinBox.valueChanged.connect(self.set_modified)
        self.ui.n_lines_detect_method_cb.currentTextChanged.connect(self.set_modified)
        self.ui.spline_degree_spinBox.valueChanged.connect(self.set_modified)
        self.ui.trim_start_cm.valueChanged.connect(self._trim_start_change_event)
        self.ui.trim_end_cm.valueChanged.connect(self._trim_end_change_event)
        self.ui.updateRangebtn.clicked.connect(self.update_range_btn_clicked)
        self.ui.updateTrimRangebtn.clicked.connect(self.update_trim_range_btn_clicked)
        self.ui.whittaker_lambda_spinBox.valueChanged.connect(self.set_modified)
        self.ui.window_length_spinBox.valueChanged.connect(self.set_smooth_polyorder_bound)
        self.ui.guess_method_cb.currentTextChanged.connect(self.guess_method_cb_changed)

    def _init_smoothing_method_combo_box(self) -> None:
        for i in self.smoothing_methods.keys():
            self.ui.smoothing_method_comboBox.addItem(i)

    def _init_guess_method_cb(self) -> None:
        self.ui.guess_method_cb.addItem('Average')
        self.ui.guess_method_cb.addItem('Average groups')
        self.ui.guess_method_cb.addItem('All')

    def _init_n_lines_method(self) -> None:
        self.ui.n_lines_detect_method_cb.addItem('Min')
        self.ui.n_lines_detect_method_cb.addItem('Max')
        self.ui.n_lines_detect_method_cb.addItem('Mean')
        self.ui.n_lines_detect_method_cb.addItem('Median')
        self.ui.n_lines_detect_method_cb.addItem('Mode')

    def _init_average_function_cb(self) -> None:
        self.ui.average_method_cb.addItem('Mean')
        self.ui.average_method_cb.addItem('Median')

    def _init_dataset_type_cb(self) -> None:
        self.ui.dataset_type_cb.addItem('Smoothed')
        self.ui.dataset_type_cb.addItem('Baseline corrected')
        self.ui.dataset_type_cb.addItem('Deconvoluted')
        self.ui.dataset_type_cb.currentTextChanged.connect(self.dataset_type_cb_current_text_changed)

    def _init_current_feature_cb(self) -> None:
        self.ui.current_feature_comboBox.currentTextChanged.connect(self.update_shap_scatters)

    def _init_coloring_feature_cb(self) -> None:
        self.ui.coloring_feature_comboBox.currentTextChanged.connect(self.update_shap_scatters)

    def dataset_type_cb_current_text_changed(self, ct: str) -> None:
        if ct == 'Smoothed':
            model = self.ui.smoothed_dataset_table_view.model()
        elif ct == 'Baseline corrected':
            model = self.ui.baselined_dataset_table_view.model()
        elif ct == 'Deconvoluted':
            model = self.ui.deconvoluted_dataset_table_view.model()
        else:
            return
        if model.rowCount() == 0 or self.is_production_project:
            return
        q_res = model.dataframe()
        features_names = list(q_res.columns[2:])
        self.ui.current_feature_comboBox.clear()
        self.ui.coloring_feature_comboBox.clear()
        self.ui.coloring_feature_comboBox.addItem('')
        for i in features_names:
            self.ui.current_feature_comboBox.addItem(i)
            self.ui.coloring_feature_comboBox.addItem(i)
        try:
            self.ui.current_group_shap_comboBox.currentTextChanged.disconnect(self.current_group_shap_changed)
        except:
            error('failed to disconnect currentTextChanged self.current_group_shap_comboBox)')
        self.ui.current_group_shap_comboBox.clear()

        classes = np.unique(q_res['Class'].values)
        target_names = self.ui.GroupsTable.model().dataframe().loc[classes]['Group name'].values
        for i in target_names:
            self.ui.current_group_shap_comboBox.addItem(i)
        try:
            self.ui.current_group_shap_comboBox.currentTextChanged.connect(self.current_group_shap_changed)
        except:
            error('failed to connect currentTextChanged self.current_group_shap_comboBox)')

        try:
            self.ui.current_instance_combo_box.currentTextChanged.disconnect(self.current_instance_changed)
        except:
            error('failed to disconnect currentTextChanged self.current_instance_combo_box)')
        self.ui.current_instance_combo_box.addItem('')
        for i in q_res['Filename']:
            self.ui.current_instance_combo_box.addItem(i)
        try:
            self.ui.current_instance_combo_box.currentTextChanged.connect(self.current_instance_changed)
        except:
            error('failed to connect currentTextChanged self.current_instance_changed)')

    def intervals_gb_toggled(self, b: bool) -> None:
        self.ui.fit_intervals_table_view.setVisible(b)
        if b:
            self.ui.intervals_gb.setMaximumHeight(200)
        else:
            self.ui.intervals_gb.setMaximumHeight(1)

    def guess_method_cb_changed(self, value: str) -> None:
        self.ui.n_lines_detect_method_cb.setEnabled(value != 'Average')
        self.set_modified()

    @asyncSlot()
    async def current_group_shap_changed(self, g: str = '') -> None:
        print('current_group_shap_changed')
        before = datetime.now()
        await self.loop.run_in_executor(None, self._update_shap_plots)
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        self.update_force_single_plots()
        self.update_force_full_plots()
        print(datetime.now() - before)

    @asyncSlot()
    async def current_instance_changed(self, _: str = '') -> None:
        before = datetime.now()
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        self.update_force_single_plots()
        print(datetime.now() - before)

    # region mouse double clicked

    def _laser_wl_spinbox_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.laser_wl_spinbox.setValue(self.default_values['laser_wl'])

    def _max_ccd_value_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.max_CCD_value_spinBox.setValue(self.default_values['max_CCD_value'])

    def _maxima_count_despike_spin_box_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.maxima_count_despike_spin_box.setValue(self.default_values['maxima_count_despike'])

    def _despike_fwhm_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.despike_fwhm_width_doubleSpinBox.setValue(self.default_values['despike_fwhm'])

    def _neg_grad_factor_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.neg_grad_factor_spinBox.setValue(self.default_values['neg_grad_factor_spinBox'])

    def _cm_range_start_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.cm_range_start.setValue(self.default_values['cm_range_start'])

    def _cm_range_end_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.cm_range_end.setValue(self.default_values['cm_range_end'])

    def _interval_start_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.interval_start_dsb.setValue(self.default_values['interval_start'])

    def _interval_end_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.interval_end_dsb.setValue(self.default_values['interval_end'])

    def _trim_start_cm_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.trim_start_cm.setValue(self.default_values['trim_start_cm'])

    def _trim_end_cm_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.trim_end_cm.setValue(self.default_values['trim_end_cm'])

    def _emsc_pca_n_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.emsc_pca_n_spinBox.setValue(self.default_values['EMSC_N_PCA'])

    def _window_length_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.window_length_spinBox.setValue(self.default_values['window_length_spinBox'])

    def _smooth_polyorder_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.smooth_polyorder_spinBox.setValue(self.default_values['smooth_polyorder_spinBox'])

    def _whittaker_lambda_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.whittaker_lambda_spinBox.setValue(self.default_values['whittaker_lambda_spinBox'])

    def _kaiser_beta_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.kaiser_beta_doubleSpinBox.setValue(self.default_values['kaiser_beta'])

    def _emd_noise_modes_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.emd_noise_modes_spinBox.setValue(self.default_values['EMD_noise_modes'])

    def _eemd_trials_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.eemd_trials_spinBox.setValue(self.default_values['EEMD_trials'])

    def _sigma_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.sigma_spinBox.setValue(self.default_values['sigma'])

    # endregion

    # region baseline params mouseDCE

    def _alpha_factor_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['alpha_factor']
                self.ui.alpha_factor_doubleSpinBox.setValue(value)
            else:
                self.ui.alpha_factor_doubleSpinBox.setValue(self.default_values['alpha_factor'])

    def _eta_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['eta']
                self.ui.eta_doubleSpinBox.setValue(value)
            else:
                self.ui.eta_doubleSpinBox.setValue(self.default_values['eta'])

    def _fill_half_window_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['fill_half_window']
                self.ui.fill_half_window_spinBox.setValue(value)
            else:
                self.ui.fill_half_window_spinBox.setValue(self.default_values['fill_half_window'])

    def _fraction_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['fraction']
                self.ui.fraction_doubleSpinBox.setValue(value)
            else:
                self.ui.fraction_doubleSpinBox.setValue(self.default_values['fraction'])

    def _grad_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['grad']
                self.ui.grad_doubleSpinBox.setValue(value)
            else:
                self.ui.grad_doubleSpinBox.setValue(self.default_values['grad'])

    def _interp_half_window_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['interp_half_window']
                self.ui.interp_half_window_spinBox.setValue(value)
            else:
                self.ui.interp_half_window_spinBox.setValue(self.default_values['interp_half_window'])

    def _lambda_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = int(self.baseline_parameter_defaults[method]['lam'])
                self.ui.lambda_spinBox.setValue(value)
            else:
                self.ui.lambda_spinBox.setValue(self.default_values['lambda_spinBox'])

    def _min_length_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['min_length']
                self.ui.min_length_spinBox.setValue(value)
            else:
                self.ui.min_length_spinBox.setValue(self.default_values['min_length'])

    def _n_iterations_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['n_iter']
                self.ui.n_iterations_spinBox.setValue(value)
            else:
                self.ui.n_iterations_spinBox.setValue(self.default_values['N_iterations'])

    def _num_std_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['num_std']
                self.ui.num_std_doubleSpinBox.setValue(value)
            else:
                self.ui.num_std_doubleSpinBox.setValue(self.default_values['num_std'])

    def _p_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['p']
                self.ui.p_doubleSpinBox.setValue(value)
            else:
                self.ui.p_doubleSpinBox.setValue(self.default_values['p_doubleSpinBox'])

    def _peak_ratio_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['peak_ratio']
                self.ui.peak_ratio_doubleSpinBox.setValue(value)
            else:
                self.ui.peak_ratio_doubleSpinBox.setValue(self.default_values['peak_ratio'])

    def _polynome_degree_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['poly_deg']
                self.ui.polynome_degree_spinBox.setValue(value)
            else:
                self.ui.polynome_degree_spinBox.setValue(self.default_values['polynome_degree'])

    def _quantile_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['quantile']
                self.ui.quantile_doubleSpinBox.setValue(value)
            else:
                self.ui.quantile_doubleSpinBox.setValue(self.default_values['quantile'])

    def _sections_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['sections']
                self.ui.sections_spinBox.setValue(value)
            else:
                self.ui.sections_spinBox.setValue(self.default_values['sections'])

    def _scale_dsb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['scale']
                self.ui.scale_doubleSpinBox.setValue(value)
            else:
                self.ui.scale_doubleSpinBox.setValue(self.default_values['scale'])

    def _spline_degree_sb_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            method = self.ui.baseline_correction_method_comboBox.currentText()
            if method in self.baseline_parameter_defaults:
                value = self.baseline_parameter_defaults[method]['spl_deg']
                self.ui.spline_degree_spinBox.setValue(value)
            else:
                self.ui.spline_degree_spinBox.setValue(self.default_values['spline_degree'])

    # endregion

    # endregion

    # region Tables

    def _initial_all_tables(self) -> None:
        self._initial_group_table()
        self._initial_input_table()
        self._initial_dec_table()
        self._initial_deconv_lines_table()
        self._initial_deconv_params_table()
        self._initial_fit_intervals_table()
        self._initial_smoothed_dataset_table()
        self._initial_baselined_dataset_table()
        self._initial_deconvoluted_dataset_table()
        self._initial_predict_dataset_table()
        self._initial_pca_features_table()
        self._initial_plsda_vip_table()

    # region input_table
    def _initial_input_table(self) -> None:
        self._reset_input_table()
        self.ui.input_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.ui.input_table.horizontalHeader().resizeSection(0, 60)
        self.ui.input_table.horizontalHeader().setMinimumWidth(10)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.ui.input_table.horizontalHeader().resizeSection(1, 60)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.ui.input_table.horizontalHeader().resizeSection(2, 11)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        self.ui.input_table.horizontalHeader().resizeSection(4, 115)
        self.ui.input_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.ui.input_table.horizontalHeader().resizeSection(5, 80)
        self.ui.input_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.ui.input_table.verticalHeader().setDefaultSectionSize(9)
        self.ui.input_table.selectionModel().selectionChanged.connect(self.input_table_selection_changed)
        self.ui.input_table.verticalScrollBar().valueChanged.connect(self.move_side_scrollbar)
        self.ui.input_table.moveEvent = self.input_table_vertical_scrollbar_value_changed
        self.ui.input_table.model().dataChanged.connect(self.input_table_item_changed)
        self.ui.input_table.rowCountChanged = self.decide_vertical_scroll_bar_visible
        self.ui.input_table.horizontalHeader().sectionClicked.connect(self._input_table_header_clicked)
        self.ui.input_table.contextMenuEvent = self._input_table_context_menu_event
        self.ui.input_table.keyPressEvent = self._input_table_key_pressed

    def _input_table_key_pressed(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Delete and self.ui.input_table.selectionModel().currentIndex().row() > -1 \
                and len(self.ui.input_table.selectionModel().selectedIndexes()):
            self.beforeTime = datetime.now()
            command = CommandDeleteInputSpectrum(self, "Delete files")
            self.undoStack.push(command)

    def _reset_input_table(self) -> None:
        df = DataFrame(columns=['Min, nm', 'Max, nm', 'Group', 'Despiked, nm', 'Rayleigh line, nm',
                                'FWHM, nm', 'FWHM, cm\N{superscript minus}\N{superscript one}', 'SNR'])
        model = PandasModelInputTable(df)
        self.ui.input_table.setSortingEnabled(True)
        self.ui.input_table.setModel(model)

    def _input_table_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Sort by index ascending', lambda: self.ui.input_table.model().sort_index())
        menu.addAction('Sort by index descending', lambda: self.ui.input_table.model().sort_index(ascending=False))
        menu.move(a0.globalPos())
        menu.show()

    # endregion

    # region group table
    def _initial_group_table(self) -> None:
        self._reset_group_table()
        self.ui.GroupsTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ui.GroupsTable.setColumnWidth(1, 150)
        self.ui.GroupsTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.ui.GroupsTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.GroupsTable.clicked.connect(self.group_table_cell_clicked)

    def _reset_group_table(self) -> None:
        df = DataFrame(columns=['Group name', 'Style'])
        model = PandasModelGroupsTable(self, df)
        self.ui.GroupsTable.setModel(model)

    # endregion

    # region dec table

    def _initial_dec_table(self) -> None:
        self._reset_dec_table()
        self.ui.dec_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ui.dec_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.ui.dec_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.dec_table.doubleClicked.connect(self.dec_table_double_clicked)
        # self.ui.dec_table.contextMenuEvent = self._dec_table_context_menu_event

    def _reset_dec_table(self) -> None:
        df = DataFrame(columns=['Filename'])
        model = PandasModelDeconvTable(df)
        self.ui.dec_table.setModel(model)

    def _dec_table_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('To template', self.to_template_clicked)
        menu.addAction('Copy spectrum lines parameters from template',
                       self.copy_spectrum_lines_parameters_from_template)
        menu.move(a0.globalPos())
        menu.show()

    @asyncSlot()
    async def to_template_clicked(self) -> None:
        selected_rows = self.ui.dec_table.selectionModel().selectedRows()
        if len(selected_rows) == 0:
            return
        selected_filename = self.ui.dec_table.model().cell_data_by_index(selected_rows[0])
        self.update_single_deconvolution_plot(selected_filename, True)

    # endregion

    # region deconv_lines_table
    def _initial_deconv_lines_table(self) -> None:
        self._reset_deconv_lines_table()
        self.ui.deconv_lines_table.verticalHeader().setSectionsMovable(True)
        self.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.ui.deconv_lines_table.horizontalHeader().resizeSection(0, 110)
        self.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.ui.deconv_lines_table.horizontalHeader().resizeSection(1, 150)
        self.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.ui.deconv_lines_table.horizontalHeader().resizeSection(2, 150)
        self.ui.deconv_lines_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.ui.deconv_lines_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.deconv_lines_table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.ui.deconv_lines_table.setDragDropOverwriteMode(False)
        self.ui.deconv_lines_table.horizontalHeader().sectionClicked.connect(self._deconv_lines_table_header_clicked)
        self.ui.deconv_lines_table.keyPressEvent = self.deconv_lines_table_key_pressed
        self.ui.deconv_lines_table.contextMenuEvent = self._deconv_lines_table_context_menu_event
        self.ui.deconv_lines_table.clicked.connect(self._deconv_lines_table_clicked)
        # self.ui.deconv_lines_table.verticalHeader().setVisible(False)

    def _deconv_lines_table_clicked(self) -> None:
        selected_indexes = self.ui.deconv_lines_table.selectionModel().selectedIndexes()
        if len(selected_indexes) == 0:
            return
        self.set_rows_visibility()
        row = selected_indexes[0].row()
        idx = self.ui.deconv_lines_table.model().index_by_row(row)
        print(self.updating_fill_curve_idx)
        if self.updating_fill_curve_idx is not None:
            curve_style = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(self.updating_fill_curve_idx,
                                                                                       'Style')
            self.update_curve_style(self.updating_fill_curve_idx, curve_style)
        self.start_fill_timer(idx)

    def deconv_lines_table_key_pressed(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Delete and self.ui.deconv_lines_table.selectionModel().currentIndex().row() > -1 \
                and len(self.ui.deconv_lines_table.selectionModel().selectedIndexes()) and self.isTemplate:
            self.beforeTime = datetime.now()
            command = CommandDeleteDeconvLines(self, "Delete line")
            self.undoStack.push(command)

    def _reset_deconv_lines_table(self) -> None:
        df = DataFrame(columns=['Legend', 'Type', 'Style'])
        model = PandasModelDeconvLinesTable(self, df, [])
        self.ui.deconv_lines_table.setSortingEnabled(True)
        self.ui.deconv_lines_table.setModel(model)
        combobox_delegate = ComboDelegate(self.peak_shapes_params.keys())
        self.ui.deconv_lines_table.setItemDelegateForColumn(1, combobox_delegate)
        self.ui.deconv_lines_table.model().sigCheckedChanged.connect(self.show_hide_curve)
        combobox_delegate.sigLineTypeChanged.connect(self.curve_type_changed)
        self.ui.deconv_lines_table.clicked.connect(self.deconv_lines_table_clicked)

    @asyncSlot()
    async def deconv_lines_table_clicked(self) -> None:
        current_index = self.ui.deconv_lines_table.selectionModel().currentIndex()
        current_column = current_index.column()
        current_row = current_index.row()
        row_data = self.ui.deconv_lines_table.model().row_data(current_row)
        idx = row_data.name
        style = row_data['Style']
        if current_column != 2:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == idx and obj.isVisible():
                return
        window_cp = CurvePropertiesWindow(self, style, idx)
        window_cp.sigStyleChanged.connect(self._update_deconv_curve_style)
        window_cp.show()

    def _deconv_lines_table_header_clicked(self, idx: int):
        df = self.ui.deconv_lines_table.model().dataframe()
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending_deconv_lines_table = not self._ascending_deconv_lines_table
        if current_name != 'Style':
            self.ui.deconv_lines_table.model().sort_values(current_name, self._ascending_deconv_lines_table)
        self.deselect_selected_line()

    def _deconv_lines_table_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        # noinspection PyTypeChecker
        menu.addAction('Delete line', self.delete_line_clicked)
        menu.addAction('Clear table', self.clear_all_deconv_lines)
        # menu.addAction('Copy template to all spectra', self.copy_template_to_all_spectra)
        menu.move(a0.globalPos())
        menu.show()

    @asyncSlot()
    async def delete_line_clicked(self) -> None:
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
        self.ui.fit_params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.ui.fit_params_table.horizontalHeader().resizeSection(0, 70)
        self.ui.fit_params_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.ui.fit_params_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.ui.fit_params_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.fit_params_table.contextMenuEvent = self._deconv_params_table_context_menu_event
        self.ui.fit_params_table.verticalHeader().setVisible(False)

    def _reset_deconv_params_table(self) -> None:
        tuples = [('', 0, 'a')]
        multi_index = MultiIndex.from_tuples(tuples, names=('filename', 'line_index', 'param_name'))
        df = DataFrame(columns=['Parameter', 'Value', 'Min value', 'Max value'], index=multi_index)
        model = PandasModelFitParamsTable(self, df)
        self.ui.fit_params_table.setModel(model)
        self.ui.fit_params_table.model().clear_dataframe()

    def _deconv_params_table_context_menu_event(self, a0: QContextMenuEvent) -> None:
        if self.isTemplate:
            return
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Copy line parameters from template', self.copy_line_parameters_from_template)
        menu.move(a0.globalPos())
        menu.show()

    def set_rows_visibility(self) -> None:
        """
        Show only rows of selected curve in fit_params_table. Other rows hiding.
        If not selected - show first curve's params
        """
        row_count = self.ui.fit_params_table.model().rowCount()
        if row_count == 0:
            return
        filename = '' if self.isTemplate else self.current_spectrum_deconvolution_name
        row_line = self.ui.deconv_lines_table.selectionModel().currentIndex().row()
        row = row_line if row_line != -1 else 0
        idx = self.ui.deconv_lines_table.model().row_data(row).name
        row_id_to_show = self.ui.fit_params_table.model().row_number_for_filtering((filename, idx))
        if row_id_to_show is None:
            return
        for i in range(row_count):
            self.ui.fit_params_table.setRowHidden(i, True)
        for i in row_id_to_show:
            self.ui.fit_params_table.setRowHidden(i, False)

    def show_current_report_result(self) -> None:
        """
        Show report result of currently showing spectrum name
        """
        filename = '' if self.isTemplate else self.current_spectrum_deconvolution_name
        report = self.report_result[filename] if filename in self.report_result else ''
        self.ui.report_text_edit.setText(report)

    # endregion

    # region pca plsda  features
    def _initial_pca_features_table(self) -> None:
        self._reset_pca_features_table()

    def _reset_pca_features_table(self) -> None:
        df = DataFrame(columns=['feature', 'PC-1', 'PC-2'])
        model = PandasModel(df)
        self.ui.pca_features_table_view.setModel(model)

    def _initial_plsda_vip_table(self) -> None:
        self._reset_plsda_vip_table()

    def _reset_plsda_vip_table(self) -> None:
        df = DataFrame(columns=['feature', 'VIP'])
        model = PandasModel(df)
        self.ui.plsda_vip_table_view.setModel(model)
    # endregion

    # region fit intervals
    def _initial_fit_intervals_table(self) -> None:
        self._reset_fit_intervals_table()
        self.ui.fit_intervals_table_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ui.fit_intervals_table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.fit_intervals_table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.ui.fit_intervals_table_view.contextMenuEvent = self._fit_intervals_table_context_menu_event
        self.ui.fit_intervals_table_view.keyPressEvent = self._fit_intervals_table_key_pressed
        dsb_delegate = IntervalsTableDelegate(self)
        self.ui.fit_intervals_table_view.setItemDelegateForColumn(0, dsb_delegate)
        # self.ui.fit_intervals_table_view.verticalHeader().setVisible(False)

    def _reset_fit_intervals_table(self) -> None:
        df = DataFrame(columns=['Border'])
        model = PandasModelFitIntervals(df)
        self.ui.fit_intervals_table_view.setModel(model)

    def _fit_intervals_table_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Add border', self._fit_intervals_table_add)
        menu.addAction('Delete selected', self._fit_intervals_table_delete_by_context_menu)
        menu.move(a0.globalPos())
        menu.show()

    def _fit_intervals_table_add(self) -> None:
        command = CommandFitIntervalAdded(self, 'Add new interval border')
        self.undoStack.push(command)

    def _fit_intervals_table_delete_by_context_menu(self) -> None:
        self._fit_intervals_table_delete()

    def _fit_intervals_table_key_pressed(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Delete \
                and self.ui.fit_intervals_table_view.selectionModel().currentIndex().row() > -1 \
                and len(self.ui.fit_intervals_table_view.selectionModel().selectedIndexes()):
            self._fit_intervals_table_delete()

    def _fit_intervals_table_delete(self) -> None:
        selection = self.ui.fit_intervals_table_view.selectionModel()
        row = selection.currentIndex().row()
        interval_number = self.ui.fit_intervals_table_view.model().row_data(row).name
        command = CommandFitIntervalDeleted(self, interval_number, 'Delete selected border')
        self.undoStack.push(command)

    # endregion

    # region smoothed dataset
    def _initial_smoothed_dataset_table(self) -> None:
        self._reset_smoothed_dataset_table()
        self.ui.smoothed_dataset_table_view.verticalScrollBar().valueChanged.connect(self.move_side_scrollbar)

    def _reset_smoothed_dataset_table(self) -> None:
        df = DataFrame(columns=['Class', 'Filename'])
        model = PandasModelSmoothedDataset(self, df)
        self.ui.smoothed_dataset_table_view.setModel(model)

    # endregion

    # region baselined corrected dataset
    def _initial_baselined_dataset_table(self) -> None:
        self._reset_baselined_dataset_table()
        self.ui.baselined_dataset_table_view.verticalScrollBar().valueChanged.connect(self.move_side_scrollbar)

    def _reset_baselined_dataset_table(self) -> None:
        df = DataFrame(columns=['Class', 'Filename'])
        model = PandasModelBaselinedDataset(self, df)
        self.ui.baselined_dataset_table_view.setModel(model)

    # endregion

    # region deconvoluted dataset
    def clear_deconvoluted_dataset_table(self) -> None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText('Точно удалить таблицу с разделенными линиями?')
        msg.setInformativeText('На новое перезаполнение таблицы может уйти время.')
        msg.setWindowTitle("Achtung!")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self._initial_deconvoluted_dataset_table()

    def _initial_deconvoluted_dataset_table(self) -> None:
        self._reset_deconvoluted_dataset_table()
        self.ui.deconvoluted_dataset_table_view.verticalScrollBar().valueChanged.connect(self.move_side_scrollbar)

    def _reset_deconvoluted_dataset_table(self) -> None:
        df = DataFrame(columns=['Class', 'Filename'])
        model = PandasModelDeconvolutedDataset(self, df)
        self.ui.deconvoluted_dataset_table_view.setModel(model)

    def get_current_dataset_type_cb(self) -> None | QAbstractItemModel:
        ct = self.ui.dataset_type_cb.currentText()
        if ct == 'Smoothed':
            return self.ui.smoothed_dataset_table_view.model()
        elif ct == 'Baseline corrected':
            return self.ui.baselined_dataset_table_view.model()
        elif ct == 'Deconvoluted':
            return self.ui.deconvoluted_dataset_table_view.model()
        else:
            return None

    # endregion

    # region predict dataset

    def _initial_predict_dataset_table(self) -> None:
        self._reset_predict_dataset_table()
        self.ui.predict_table_view.verticalScrollBar().valueChanged.connect(self.move_side_scrollbar)
        self.ui.predict_table_view.verticalHeader().setVisible(False)

    def _reset_predict_dataset_table(self) -> None:
        df = DataFrame(columns=['Filename'])
        model = PandasModelPredictTable(self, df)
        self.ui.predict_table_view.setModel(model)


    # endregion

    # endregion

    # region deconv_buttons_frame

    def _initial_guess_table_frame(self) -> None:
        """right side frame with lines table, parameters table and report field"""
        self._initial_add_line_button()
        self._initial_guess_button()
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
        self.ui.interval_start_dsb.valueChanged.connect(self._interval_start_dsb_change_event)
        self.ui.interval_end_dsb.valueChanged.connect(self._interval_end_dsb_change_event)
        self.ui.interval_checkBox.stateChanged.connect(self.interval_cb_state_changed)
        self.linearRegionDeconv.setVisible(False)

    def _initial_guess_button(self) -> None:
        guess_menu = QMenu()
        line_type: str
        for line_type in self.peak_shapes_params.keys():
            action = guess_menu.addAction(line_type)
            action.triggered.connect(lambda checked=None, line=line_type: self.guess(line_type=line))
        self.ui.guess_button.setMenu(guess_menu)
        self.ui.guess_button.menu()  # some kind of magik

    def _initial_add_line_button(self) -> None:
        add_lines_menu = QMenu()
        line_type: str
        for line_type in self.peak_shapes_params.keys():
            action = add_lines_menu.addAction(line_type)
            action.triggered.connect(lambda checked=None, line=line_type: self.add_deconv_line(line_type=line))
        self.ui.add_line_button.setMenu(add_lines_menu)
        self.ui.add_line_button.menu()  # some kind of magik

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
        if self.data_curve is None:
            return
        self.data_curve.setVisible(a0 == 2)

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
        if self.sum_curve is None:
            return
        self.sum_curve.setVisible(a0 == 2)

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
        if self.fill is None:
            return
        self.fill.setVisible(a0 == 2)

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
        if self.residual_curve is None:
            return
        self.residual_curve.setVisible(a0 == 2)

    def data_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.data_pushButton.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def sum_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.sum_pushButton.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def sigma3_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.sigma3_pushButton.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def residual_style_button_style_sheet(self, hex_color: str) -> None:
        self.ui.residual_pushButton.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def _update_data_curve_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(self, style, old_style, 'data', 'Update style for data curve')
        self.undoStack.push(command)

    def _update_sum_curve_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(self, style, old_style, 'sum', 'Update style for data curve')
        self.undoStack.push(command)

    def _update_residual_curve_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(self, style, old_style, 'residual', 'Update style for data curve')
        self.undoStack.push(command)

    def _update_sigma3_style(self, style: dict, old_style: dict) -> None:
        command = CommandUpdateDataCurveStyle(self, style, old_style, 'sigma3', 'Update style for sigma3')
        self.undoStack.push(command)

    def data_pb_clicked(self) -> None:
        if self.data_curve is None:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 999 and obj.isVisible():
                return
        data_curve_prop_window = CurvePropertiesWindow(self, self.data_style, 999, False)
        data_curve_prop_window.sigStyleChanged.connect(self._update_data_curve_style)
        data_curve_prop_window.show()

    def sum_pb_clicked(self) -> None:
        if self.sum_curve is None:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 998 and obj.isVisible():
                return
        sum_curve_prop_window = CurvePropertiesWindow(self, self.sum_style, 998, False)
        sum_curve_prop_window.sigStyleChanged.connect(self._update_sum_curve_style)
        sum_curve_prop_window.show()

    def residual_pb_clicked(self) -> None:
        if self.residual_curve is None:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 997 and obj.isVisible():
                return
        residual_curve_prop_window = CurvePropertiesWindow(self, self.residual_style, 997, False)
        residual_curve_prop_window.sigStyleChanged.connect(self._update_residual_curve_style)
        residual_curve_prop_window.show()

    def sigma3_push_button_clicked(self) -> None:
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 996 and obj.isVisible():
                return
        prop_window = CurvePropertiesWindow(self, self.sigma3_style, 996, True)
        prop_window.sigStyleChanged.connect(self._update_sigma3_style)
        prop_window.show()

    def update_deconv_intervals_limits(self) -> None:
        if not self.averaged_dict or 1 not in self.averaged_dict:
            return
        x_axis = self.averaged_dict[1][:, 0]
        if x_axis.size == 0:
            return
        min_cm = np.min(x_axis)
        max_cm = np.max(x_axis)
        if self.ui.interval_start_dsb.value() < min_cm or self.ui.interval_start_dsb.value() > max_cm:
            self.ui.interval_start_dsb.setValue(min_cm)
        if self.ui.interval_end_dsb.value() < min_cm or self.ui.interval_end_dsb.value() > max_cm:
            self.ui.interval_end_dsb.setValue(max_cm)
        self.ui.interval_start_dsb.setMinimum(min_cm)
        self.ui.interval_start_dsb.setMaximum(max_cm)
        self.ui.interval_end_dsb.setMinimum(min_cm)
        self.ui.interval_end_dsb.setMaximum(max_cm)

    def lr_deconv_region_changed(self) -> None:
        current_region = self.linearRegionDeconv.getRegion()
        self.ui.interval_start_dsb.setValue(current_region[0])
        self.ui.interval_end_dsb.setValue(current_region[1])

    def _interval_start_dsb_change_event(self, new_value: float) -> None:
        """
        executing when value of self.ui.interval_start_dsb was changed by user or by code
        self.old_start_interval_value contains previous value

        Parameters
        ----------
        new_value : float
            New value of self.ui.interval_start_dsb
        """
        self.set_modified()
        corrected_value = None
        # correct value - take nearest from x_axis
        if self.averaged_dict != {} and self.averaged_dict:
            x_axis = next(iter(self.averaged_dict.values()))[:, 0]
            corrected_value = find_nearest(x_axis, new_value)
        if corrected_value is not None and round(corrected_value, 5) != new_value:
            self.ui.interval_start_dsb.setValue(corrected_value)
            return
        if new_value >= self.ui.interval_end_dsb.value():
            self.ui.interval_start_dsb.setValue(self.ui.interval_start_dsb.minimum())
            return
        if self.CommandStartIntervalChanged_allowed:
            command = CommandStartIntervalChanged(self, new_value, self.old_start_interval_value,
                                                  'Change start-interval value')
            self.undoStack.push(command)
        self.linearRegionDeconv.setRegion((self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value()))
        # if interval checked - change cut interval
        if self.ui.interval_checkBox.isChecked():
            self.cut_data_sum_residual_interval()

    def _interval_end_dsb_change_event(self, new_value: float) -> None:
        """
        executing when value of self.ui.interval_end_dsb was changed by user or by code
        self.old_end_interval_value contains previous value

        Parameters
        ----------
        new_value : float
            New value of self.ui.interval_end_dsb
        """
        self.set_modified()
        corrected_value = None
        # correct value - take nearest from x_axis
        if self.averaged_dict != {} and self.averaged_dict:
            x_axis = next(iter(self.averaged_dict.values()))[:, 0]
            corrected_value = find_nearest(x_axis, new_value)
        if corrected_value is not None and round(corrected_value, 5) != new_value:
            self.ui.interval_end_dsb.setValue(corrected_value)
            return
        if new_value <= self.ui.interval_start_dsb.value():
            self.ui.interval_end_dsb.setValue(self.ui.interval_end_dsb.minimum())
            return
        if self.CommandEndIntervalChanged_allowed:
            command = CommandEndIntervalChanged(self, new_value, self.old_end_interval_value,
                                                'Change end-interval value')
            self.undoStack.push(command)
        self.linearRegionDeconv.setRegion((self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value()))
        # if interval checked - change cut interval
        if self.ui.interval_checkBox.isChecked():
            self.cut_data_sum_residual_interval()

    def interval_cb_state_changed(self, a0: int) -> None:
        """ a0 = 0 is False, a0 = 2 if True"""
        self.linearRegionDeconv.setVisible(a0 == 2)
        if a0 == 2:
            self.cut_data_sum_residual_interval()
        else:
            self.uncut_data_sum_residual()

    @asyncSlot()
    async def cut_data_sum_residual_interval(self) -> None:
        self.cut_data_interval()
        print('cut_data_sum_residual_interval')
        self.redraw_curves_for_filename()
        self.draw_sum_curve()
        self.draw_residual_curve()

    @asyncSlot()
    async def uncut_data_sum_residual(self) -> None:
        self.uncut_data()
        print('uncut_data_sum_residual')
        self.redraw_curves_for_filename()
        self.draw_sum_curve()
        self.draw_residual_curve()

    def cut_data_interval(self) -> None:
        n_array = self.array_of_current_filename_in_deconvolution()
        if n_array is None:
            return
        n_array = cut_full_spectrum(n_array, self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value())
        self.data_curve.setData(x=n_array[:, 0], y=n_array[:, 1])

    def uncut_data(self) -> None:
        n_array = self.array_of_current_filename_in_deconvolution()
        if n_array is None:
            return
        self.data_curve.setData(x=n_array[:, 0], y=n_array[:, 1])

    # endregion

    # region other

    def stat_tab_widget_tab_changed(self, i: int):
        self.update_stat_report_text()
        self.decide_vertical_scroll_bar_visible()

    def initial_right_scrollbar(self) -> None:
        self.ui.verticalScrollBar.setVisible(False)
        self.ui.verticalScrollBar.setMinimum(1)
        self.ui.verticalScrollBar.enterEvent = self.vertical_scroll_bar_enter_event
        self.ui.verticalScrollBar.leaveEvent = self.vertical_scroll_bar_leave_event
        self.ui.verticalScrollBar.valueChanged.connect(self.vertical_scroll_bar_value_changed)
        self.ui.data_tables_tab_widget.currentChanged.connect(self.decide_vertical_scroll_bar_visible)
        self.ui.page1Btn.clicked.connect(self.page1_btn_clicked)
        self.ui.page2Btn.clicked.connect(self.page2_btn_clicked)
        self.ui.page3Btn.clicked.connect(self.page3_btn_clicked)
        self.ui.page4Btn.clicked.connect(self.page4_btn_clicked)
        self.ui.page5Btn.clicked.connect(self.page5_btn_clicked)
        self.ui.scrollArea_lda.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_qda.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_lr.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_svc.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_nearest.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_gpc.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_dt.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_nb.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_rf.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_ab.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)
        self.ui.scrollArea_mlp.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)

    def scroll_area_stat_value_changed(self, event: int):
        self.ui.verticalScrollBar.setValue(event)

    def initial_plot_buttons(self) -> None:
        self.ui.crosshairBtn.clicked.connect(self.crosshair_btn_clicked)
        self.ui.by_one_control_button.clicked.connect(self.by_one_control_button_clicked)
        self.ui.by_group_control_button.clicked.connect(self.by_group_control_button)
        self.ui.by_group_control_button.mouseDoubleClickEvent = self.by_group_control_button_double_clicked
        self.ui.all_control_button.clicked.connect(self.all_control_button)
        self.ui.despike_history_Btn.clicked.connect(self.despike_history_btn_clicked)
        self.ui.lr_movableBtn.clicked.connect(self.lr_movable_btn_clicked)
        self.ui.lr_showHideBtn.clicked.connect(self.lr_show_hide_btn_clicked)
        self.ui.sun_Btn.clicked.connect(self.change_plots_bckgrnd)

    def initial_timers(self) -> None:
        self.timer_mem_update = QTimer()
        self.timer_mem_update.timeout.connect(self.set_timer_memory_update)
        self.timer_mem_update.start(1000)
        self.cpu_load = QTimer()
        self.cpu_load.timeout.connect(self.set_cpu_load)
        self.cpu_load.start(200)
        # есть подозрения что таймер сбрасывает процесс fit
        # self.auto_save_timer = QTimer()
        # self.auto_save_timer.timeout.connect(self.auto_save)
        # self.auto_save_timer.start(1000 * 60 * self.auto_save_minutes)

    def initial_ui_definitions(self) -> None:
        self.ui.maximizeRestoreAppBtn.setToolTip("Maximize")
        self.ui.unsavedBtn.hide()
        self.ui.titlebar.mouseDoubleClickEvent = self.double_click_maximize_restore
        # self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.ui.titlebar.mouseMoveEvent = self.move_window
        self.ui.titlebar.mouseReleaseEvent = self.titlebar_mouse_release_event
        self.ui.minimizeAppBtn.clicked.connect(lambda: self.showMinimized())
        self.ui.maximizeRestoreAppBtn.clicked.connect(lambda: self.maximize_restore())
        self.ui.closeBtn.clicked.connect(lambda: self.close())
        self.ui.settingsBtn.clicked.connect(self.settings_show)

    def titlebar_mouse_release_event(self, _) -> None:
        self.setWindowOpacity(1)

    def move_window(self, event: QMouseEvent) -> None:
        # IF MAXIMIZED CHANGE TO NORMAL
        if self.window_maximized:
            self.maximize_restore()
        # MOVE WINDOW
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.setWindowOpacity(0.9)
            self.dragPos = event.globalPos()
            event.accept()

    def double_click_maximize_restore(self, event: QMouseEvent) -> None:
        # IF DOUBLE CLICK CHANGE STATUS
        if event.type() == 4:
            QTimer.singleShot(250, lambda: self.maximize_restore())

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
        if 'Light' in self.theme_bckgrnd and self.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon("material/resources/source/chevron-down_black.svg"))
        elif 'Light' in self.theme_bckgrnd and not self.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon("material/resources/source/chevron-up_black.svg"))
        elif self.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon("material/resources/source/chevron-down.svg"))
        else:
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon("material/resources/source/chevron-up.svg"))

    def show_hide_left_menu(self) -> None:
        if self.ui.left_side_main.isHidden():
            self.ui.left_side_frame.setMaximumWidth(350)
            self.ui.left_side_main.show()
            self.ui.left_hide_frame.hide()
            self.ui.main_frame.layout().setSpacing(10)
            self.ui.left_side_up_frame.setMaximumHeight(40)
            self.ui.dec_list_btn.setVisible(False)
            self.ui.stat_param_btn.setVisible(False)
            if 'Light' in theme:
                self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/chevron-left_black.svg"))
            else:
                self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/chevron-left.svg"))
        elif self.ui.left_side_main.isVisible():
            self.ui.left_side_main.hide()
            self.ui.left_hide_frame.show()
            self.ui.left_side_frame.setMaximumWidth(35)
            self.ui.left_side_up_frame.setMaximumHeight(120)
            self.ui.dec_list_btn.setVisible(True)
            self.ui.stat_param_btn.setVisible(True)
            self.ui.main_frame.layout().setSpacing(1)
            if 'Light' in theme:
                self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/sliders_black.svg"))
                self.ui.dec_list_btn.setIcon(QIcon("material/resources/source/align-justify_black.svg"))
                self.ui.stat_param_btn.setIcon(QIcon("material/resources/source/percent_black.svg"))
            else:
                self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/sliders.svg"))
                self.ui.dec_list_btn.setIcon(QIcon("material/resources/source/align-justify.svg"))
                self.ui.stat_param_btn.setIcon(QIcon("material/resources/source/percent.svg"))

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
            self.plot_text_color_value = self.theme_colors['inversePlotText']
            self.plot_text_color = QColor(self.plot_text_color_value)
            self.plot_background_color = QColor(self.theme_colors['inversePlotBackground'])
            self.plot_background_color_web = QColor(self.theme_colors['inversePlotBackground'])
            self.plt_style = ['default']
        else:
            self.plot_text_color_value = self.theme_colors['plotText']
            self.plot_text_color = QColor(self.plot_text_color_value)
            self.plot_background_color = QColor(self.theme_colors['plotBackground'])
            self.plot_background_color_web = QColor(self.theme_colors['backgroundMainColor'])
            self.plt_style = ['dark_background']
        plt.style.use(self.plt_style)
        self._initial_input_plot_color()
        self._initial_converted_cm_plot_color()
        self._initial_cut_cm_plot_color()
        self._initial_normalize_plot_color()
        self._initial_smooth_plot_color()
        self._initial_baseline_plot_color()
        self._initial_averaged_plot_color()
        self._initial_deconv_plot_color()
        self._initial_lda_scores_1d_plot_color()
        self._initial_stat_plots_color()
        self._initial_plots_set_labels_font()

    # endregion

    # endregion

    # region Plot buttons

    # region crosshair button
    def lr_movable_btn_clicked(self) -> None:
        b = not self.ui.lr_movableBtn.isChecked()
        self.linearRegionCmConverted.setMovable(b)
        self.linearRegionBaseline.setMovable(b)
        self.linearRegionDeconv.setMovable(b)

    def lr_show_hide_btn_clicked(self) -> None:
        if self.ui.lr_showHideBtn.isChecked():
            self.converted_cm_widget_plot_item.addItem(self.linearRegionCmConverted)
            self.baseline_corrected_plotItem.addItem(self.linearRegionBaseline)
        else:
            self.converted_cm_widget_plot_item.removeItem(self.linearRegionCmConverted)
            self.baseline_corrected_plotItem.removeItem(self.linearRegionBaseline)

    def crosshair_btn_clicked(self) -> None:
        """Add crosshair for self.ui.input_plot with coordinates at title."""
        if self.ui.plots_tabWidget.tabBar().currentIndex() == 0:
            self.crosshair_btn_clicked_for_input_plot()
        elif self.ui.plots_tabWidget.tabBar().currentIndex() == 1:
            self.crosshair_btn_clicked_for_converted_plot()
        elif self.ui.plots_tabWidget.tabBar().currentIndex() == 2:
            self.crosshair_btn_clicked_for_cut_plot()
        elif self.ui.plots_tabWidget.tabBar().currentIndex() == 3:
            self.crosshair_btn_clicked_for_normal_plot()
        elif self.ui.plots_tabWidget.tabBar().currentIndex() == 4:
            self.crosshair_btn_clicked_for_smooth_plot()
        elif self.ui.plots_tabWidget.tabBar().currentIndex() == 5:
            self.crosshair_btn_clicked_for_baseline_plot()
        elif self.ui.plots_tabWidget.tabBar().currentIndex() == 6:
            self.crosshair_btn_clicked_for_averaged_plot()

        if self.ui.stackedWidget_mainpages.currentIndex() == 1:
            self.crosshair_btn_clicked_for_deconv_plot()

    def crosshair_btn_clicked_for_input_plot(self) -> None:
        self.input_plot_widget_plot_item.removeItem(self.ui.input_plot_widget.vertical_line)
        self.input_plot_widget_plot_item.removeItem(self.ui.input_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.input_plot_widget_plot_item.addItem(self.ui.input_plot_widget.vertical_line, ignoreBounds=True)
            self.input_plot_widget_plot_item.addItem(self.ui.input_plot_widget.horizontal_line, ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            self.ui.input_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Imported Raman spectra</span>")

    def crosshair_btn_clicked_for_converted_plot(self) -> None:
        self.converted_cm_widget_plot_item.removeItem(self.ui.converted_cm_plot_widget.vertical_line)
        self.converted_cm_widget_plot_item.removeItem(self.ui.converted_cm_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.converted_cm_widget_plot_item.addItem(self.ui.converted_cm_plot_widget.vertical_line,
                                                       ignoreBounds=True)
            self.converted_cm_widget_plot_item.addItem(self.ui.converted_cm_plot_widget.horizontal_line,
                                                       ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            text_for_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors['plotText'] + \
                             ";font-size:14pt\">Converted to cm\N{superscript minus}\N{superscript one}</span>"
            self.ui.converted_cm_plot_widget.setTitle(text_for_title)

    def crosshair_btn_clicked_for_cut_plot(self) -> None:
        self.cut_cm_plotItem.removeItem(self.ui.cut_cm_plot_widget.vertical_line)
        self.cut_cm_plotItem.removeItem(self.ui.cut_cm_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.cut_cm_plotItem.addItem(self.ui.cut_cm_plot_widget.vertical_line,
                                         ignoreBounds=True)
            self.cut_cm_plotItem.addItem(self.ui.cut_cm_plot_widget.horizontal_line,
                                         ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            self.ui.cut_cm_plot_widget.setTitle(
                "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                    'plotText'] + ";font-size:14pt\">Cutted plots</span>")

    def crosshair_btn_clicked_for_normal_plot(self) -> None:
        self.normalize_plotItem.removeItem(self.ui.normalize_plot_widget.vertical_line)
        self.normalize_plotItem.removeItem(self.ui.normalize_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.normalize_plotItem.addItem(self.ui.normalize_plot_widget.vertical_line,
                                            ignoreBounds=True)
            self.normalize_plotItem.addItem(self.ui.normalize_plot_widget.horizontal_line,
                                            ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            if len(self.NormalizedDict) > 0:
                text_for_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors['plotText'] \
                                 + ";font-size:14pt\">Normalized plots. Method " \
                                 + self.normalization_method + "</span>"
                self.ui.normalize_plot_widget.setTitle(text_for_title)
                info('normalize_plot_widget title is %s',
                     'Normalized plots. Method '
                     + self.normalization_method)
            else:
                if len(self.NormalizedDict) > 0:
                    self.ui.normalize_plot_widget.setTitle(
                        "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                            'plotText'] + ";font-size:14pt\">Normalized plots</span>")

    def crosshair_btn_clicked_for_smooth_plot(self) -> None:
        self.smooth_plotItem.removeItem(self.ui.smooth_plot_widget.vertical_line)
        self.smooth_plotItem.removeItem(self.ui.smooth_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.smooth_plotItem.addItem(self.ui.smooth_plot_widget.vertical_line,
                                         ignoreBounds=True)
            self.smooth_plotItem.addItem(self.ui.smooth_plot_widget.horizontal_line,
                                         ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            if len(self.SmoothedDict) > 0:
                text_for_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors['plotText'] \
                                 + ";font-size:14pt\">Smoothed plots. Method " + self.smooth_method + "</span>"
                self.ui.smooth_plot_widget.setTitle(text_for_title)

            elif len(self.SmoothedDict) > 0:
                self.ui.smooth_plot_widget.setTitle(
                    "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                        'plotText'] + ";font-size:14pt\">Smoothed plots</span>")

    def crosshair_btn_clicked_for_baseline_plot(self) -> None:
        self.baseline_corrected_plotItem.removeItem(self.ui.baseline_plot_widget.vertical_line)
        self.baseline_corrected_plotItem.removeItem(self.ui.baseline_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.baseline_corrected_plotItem.addItem(self.ui.baseline_plot_widget.vertical_line,
                                                     ignoreBounds=True)
            self.baseline_corrected_plotItem.addItem(self.ui.baseline_plot_widget.horizontal_line,
                                                     ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            if len(self.baseline_corrected_dict) > 0:
                text_for_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors['plotText'] \
                                 + ";font-size:14pt\">Baseline corrected plots. Method " + \
                                 self.baseline_method + "</span>"
                self.ui.baseline_plot_widget.setTitle(text_for_title)
            else:
                self.ui.baseline_plot_widget.setTitle(
                    "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                        'plotText'] + ";font-size:14pt\">Baseline corrected plots</span>")

    def crosshair_btn_clicked_for_averaged_plot(self) -> None:
        self.averaged_plotItem.removeItem(self.ui.average_plot_widget.vertical_line)
        self.averaged_plotItem.removeItem(self.ui.average_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.averaged_plotItem.addItem(self.ui.average_plot_widget.vertical_line, ignoreBounds=True)
            self.averaged_plotItem.addItem(self.ui.average_plot_widget.horizontal_line, ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            text_for_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors['plotText'] \
                             + ";font-size:14pt\">Averaged</span>"
            self.ui.average_plot_widget.setTitle(text_for_title)

    def crosshair_btn_clicked_for_deconv_plot(self) -> None:
        self.deconvolution_plotItem.removeItem(self.ui.deconv_plot_widget.vertical_line)
        self.deconvolution_plotItem.removeItem(self.ui.deconv_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            self.deconvolution_plotItem.addItem(self.ui.deconv_plot_widget.vertical_line, ignoreBounds=True)
            self.deconvolution_plotItem.addItem(self.ui.deconv_plot_widget.horizontal_line, ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            new_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">" + self.current_spectrum_deconvolution_name + "</span>"
            self.ui.deconv_plot_widget.setTitle(new_title)

    def update_crosshair_input_plot(self, event: QPoint) -> None:
        """Paint crosshair on mouse"""

        coordinates = event[0]
        if self.ui.input_plot_widget.sceneBoundingRect().contains(coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.input_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.input_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.input_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.input_plot_widget.horizontal_line.setPos(mouse_point.y())

    def update_crosshair_converted_cm_plot(self, event: QPoint) -> None:
        """Paint crosshair on mouse"""

        coordinates = event[0]
        if self.ui.converted_cm_plot_widget.sceneBoundingRect().contains(
                coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.converted_cm_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.converted_cm_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.converted_cm_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.converted_cm_plot_widget.horizontal_line.setPos(mouse_point.y())

    def update_crosshair_cut_plot(self, event: QPoint) -> None:
        """Paint crosshair on mouse"""

        coordinates = event[0]
        if self.ui.cut_cm_plot_widget.sceneBoundingRect().contains(
                coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.cut_cm_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.cut_cm_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.cut_cm_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.cut_cm_plot_widget.horizontal_line.setPos(mouse_point.y())

    def update_crosshair_normalized_plot(self, event: QPoint) -> None:
        """Paint crosshair on mouse"""
        coordinates = event[0]
        if self.ui.normalize_plot_widget.sceneBoundingRect().contains(
                coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.normalize_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.normalize_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.normalize_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.normalize_plot_widget.horizontal_line.setPos(mouse_point.y())

    def update_crosshair_smoothed_plot(self, event: QPoint) -> None:
        """Paint crosshair on mouse"""
        coordinates = event[0]
        if self.ui.smooth_plot_widget.sceneBoundingRect().contains(coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.smooth_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.smooth_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.smooth_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.smooth_plot_widget.horizontal_line.setPos(mouse_point.y())

    def update_crosshair_baseline_plot(self, event: QPoint) -> None:
        """Paint crosshair on mouse"""
        coordinates = event[0]
        if self.ui.baseline_plot_widget.sceneBoundingRect().contains(coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.baseline_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.baseline_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.baseline_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.baseline_plot_widget.horizontal_line.setPos(mouse_point.y())

    def update_crosshair_averaged_plot(self, event: QPoint) -> None:
        """Paint crosshair on mouse"""
        coordinates = event[0]
        if self.ui.average_plot_widget.sceneBoundingRect().contains(coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.average_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.average_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.average_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.average_plot_widget.horizontal_line.setPos(mouse_point.y())

    def update_crosshair_deconv_plot(self, event: QPoint) -> None:
        coordinates = event[0]
        if self.ui.deconv_plot_widget.sceneBoundingRect().contains(coordinates) and self.ui.crosshairBtn.isChecked():
            mouse_point = self.ui.deconv_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            self.ui.deconv_plot_widget.setTitle(
                "<span style='font-size: 12pt; font-family: AbletonSans; color:" + self.theme_colors[
                    'secondaryColor'] + "'>x=%0.1f,   <span style=>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.ui.deconv_plot_widget.vertical_line.setPos(mouse_point.x())
            self.ui.deconv_plot_widget.horizontal_line.setPos(mouse_point.y())

    # endregion

    # region '1' button

    @asyncSlot()
    async def by_one_control_button_clicked(self) -> None:
        """using with button self.ui.by_one_control_button for hide all plot item besides selected in input table"""

        is_checked = self.ui.by_one_control_button.isChecked()
        if is_checked:
            self.ui.by_group_control_button.setChecked(False)
            self.ui.all_control_button.setChecked(False)
            if self.ui.input_table.selectionModel().currentIndex().column() == 0:
                self.current_futures = [self.loop.run_in_executor(None, self.update_plots_for_single)]
                await gather(*self.current_futures)

        else:
            self.ui.by_one_control_button.setChecked(True)

    async def update_plots_for_single(self) -> None:
        """loop to set visible for all plot items"""
        self.ui.statusBar.showMessage('Updating plot...')
        current_index = self.ui.input_table.selectionModel().currentIndex()
        if current_index.row() == -1:
            return
        current_spectrum_name = self.ui.input_table.model().get_filename_by_row(current_index.row())
        group_number = self.ui.input_table.model().cell_data(current_index.row(), 2)
        new_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">" + current_spectrum_name + "</span>"

        tasks = [create_task(self.update_single_input_plot(new_title, current_spectrum_name, group_number)),
                 create_task(self.update_single_converted_plot(new_title, current_spectrum_name, group_number)),
                 create_task(self.update_single_cut_plot(new_title, current_spectrum_name, group_number)),
                 create_task(self.update_single_normal_plot(new_title, current_spectrum_name, group_number)),
                 create_task(self.update_single_smooth_plot(new_title, current_spectrum_name, group_number)),
                 create_task(self.update_single_baseline_plot(new_title, current_spectrum_name, group_number))]
        await wait(tasks)

        self.ui.statusBar.showMessage('Plot updated', 5000)

    async def update_single_input_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_input_plot = self.input_plot_widget_plot_item.listDataItems()
        if len(data_items_input_plot) <= 0:
            return
        self.ui.input_plot_widget.setTitle(new_title)
        for i in data_items_input_plot:
            i.setVisible(False)
        arr = self.ImportedArray[current_spectrum_name]
        if self.curveOneInputPlot:
            self.input_plot_widget_plot_item.removeItem(self.curveOneInputPlot)
        self.curveOneInputPlot = self.get_curve_plot_data_item(arr, group_number)
        self.input_plot_widget_plot_item.addItem(self.curveOneInputPlot, kargs=['ignoreBounds', 'skipAverage'])
        if self.ui.despike_history_Btn.isChecked() and self.ui.input_table.selectionModel().currentIndex().row() != -1 \
                and self.BeforeDespike and len(self.BeforeDespike) > 0 \
                and current_spectrum_name in self.BeforeDespike:
            self.current_spectrum_despiked_name = current_spectrum_name
            tasks = [create_task(self.despike_history_add_plot())]
            await wait(tasks)
        self.input_plot_widget_plot_item.getViewBox().updateAutoRange()

    async def update_single_converted_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_converted_plot = self.converted_cm_widget_plot_item.listDataItems()
        if len(data_items_converted_plot) <= 0:
            return
        self.ui.converted_cm_plot_widget.setTitle(new_title)
        for i in data_items_converted_plot:
            i.setVisible(False)
        arr = self.ConvertedDict[current_spectrum_name]
        if self.curveOneConvertPlot:
            self.converted_cm_widget_plot_item.removeItem(self.curveOneConvertPlot)
        self.curveOneConvertPlot = self.get_curve_plot_data_item(arr, group_number)
        self.converted_cm_widget_plot_item.addItem(self.curveOneConvertPlot, kargs=['ignoreBounds', 'skipAverage'])
        self.converted_cm_widget_plot_item.getViewBox().updateAutoRange()

    async def update_single_cut_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_cut_plot = self.cut_cm_plotItem.listDataItems()
        if len(data_items_cut_plot) <= 0:
            return
        self.ui.cut_cm_plot_widget.setTitle(new_title)
        for i in data_items_cut_plot:
            i.setVisible(False)
        arr = self.CuttedFirstDict[current_spectrum_name]
        if self.curveOneCutPlot:
            self.cut_cm_plotItem.removeItem(self.curveOneCutPlot)
        self.curveOneCutPlot = self.get_curve_plot_data_item(arr, group_number)
        self.cut_cm_plotItem.addItem(self.curveOneCutPlot, kargs=['ignoreBounds', 'skipAverage'])
        self.cut_cm_plotItem.getViewBox().updateAutoRange()

    async def update_single_normal_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_normal_plot = self.normalize_plotItem.listDataItems()
        if len(data_items_normal_plot) <= 0:
            return
        self.ui.normalize_plot_widget.setTitle(new_title)
        for i in data_items_normal_plot:
            i.setVisible(False)
        arr = self.NormalizedDict[current_spectrum_name]
        if self.curveOneNormalPlot:
            self.normalize_plotItem.removeItem(self.curveOneNormalPlot)
        self.curveOneNormalPlot = self.get_curve_plot_data_item(arr, group_number)
        self.normalize_plotItem.addItem(self.curveOneNormalPlot, kargs=['ignoreBounds', 'skipAverage'])
        self.normalize_plotItem.getViewBox().updateAutoRange()

    async def update_single_smooth_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_smooth_plot = self.smooth_plotItem.listDataItems()
        if len(data_items_smooth_plot) <= 0:
            return
        self.ui.smooth_plot_widget.setTitle(new_title)
        for i in data_items_smooth_plot:
            i.setVisible(False)
        arr = self.SmoothedDict[current_spectrum_name]
        if self.curveOneSmoothPlot:
            self.smooth_plotItem.removeItem(self.curveOneSmoothPlot)
        self.curveOneSmoothPlot = self.get_curve_plot_data_item(arr, group_number)
        self.smooth_plotItem.addItem(self.curveOneSmoothPlot, kargs=['ignoreBounds', 'skipAverage'])
        self.smooth_plotItem.getViewBox().updateAutoRange()

    async def update_single_baseline_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_baseline_plot = self.baseline_corrected_plotItem.listDataItems()
        if len(data_items_baseline_plot) <= 0:
            return
        self.ui.baseline_plot_widget.setTitle(new_title)
        for i in data_items_baseline_plot:
            i.setVisible(False)
        arr = self.baseline_corrected_dict[current_spectrum_name]
        if self.curve_one_baseline_plot:
            self.baseline_corrected_plotItem.removeItem(self.curve_one_baseline_plot)
        self.curve_one_baseline_plot = self.get_curve_plot_data_item(arr, group_number)
        self.baseline_corrected_plotItem.addItem(self.curve_one_baseline_plot, kargs=['ignoreBounds', 'skipAverage'])
        if self.baseline_dict and len(self.baseline_dict) > 0 \
                and self.ui.input_table.selectionModel().currentIndex().row() != -1 \
                and current_spectrum_name in self.baseline_dict:
            self.current_spectrum_baseline_name = current_spectrum_name
            tasks = [create_task(self.baseline_add_plot())]
            await wait(tasks)
        self.baseline_corrected_plotItem.getViewBox().updateAutoRange()

    # endregion

    # region 'G' button
    @asyncSlot()
    async def by_group_control_button(self) -> None:
        """using with button self.ui.by_group_control_button for hide all plot items besides selected in group table"""

        is_checked = self.ui.by_group_control_button.isChecked()
        if is_checked:
            self.ui.by_one_control_button.setChecked(False)
            self.ui.all_control_button.setChecked(False)
            self.ui.despike_history_Btn.setChecked(False)
            tasks = [create_task(self.despike_history_remove_plot()),
                     create_task(self.baseline_remove_plot())]
            await wait(tasks)
            if self.ui.GroupsTable.selectionModel().currentIndex().column() == 0:
                await self.update_plots_for_group(None)
        else:
            self.ui.by_group_control_button.setChecked(True)

    @asyncSlot()
    async def by_group_control_button_double_clicked(self, _=None) -> None:
        if self.ui.GroupsTable.model().rowCount() < 2:
            return
        input_dialog = QInputDialog()
        result = input_dialog.getText(self, "Choose visible groups", 'Write groups numbers to show (example: 1, 2, 3):')
        if not result[1]:
            return
        v = list(result[0].strip().split(','))
        tasks = [create_task(self.despike_history_remove_plot()),
                 create_task(self.baseline_remove_plot())]
        await wait(tasks)
        groups = [int(x) for x in v]
        await self.update_plots_for_group(groups)

    @asyncSlot()
    async def update_plots_for_group(self, current_group: list[int] | None) -> None:
        """loop to set visible for all plot items in group"""
        if not current_group:
            current_row = self.ui.GroupsTable.selectionModel().currentIndex().row()
            current_group_name = self.ui.GroupsTable.model().cell_data(current_row, 0)
            current_group = [current_row + 1]
            new_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">" + current_group_name + "</span>"
        else:
            new_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">" + str(current_group) + "</span>"
        self.ui.input_plot_widget.setTitle(new_title)
        self.ui.converted_cm_plot_widget.setTitle(new_title)
        self.ui.cut_cm_plot_widget.setTitle(new_title)
        self.ui.normalize_plot_widget.setTitle(new_title)
        self.ui.smooth_plot_widget.setTitle(new_title)
        self.ui.baseline_plot_widget.setTitle(new_title)
        self.ui.average_plot_widget.setTitle(new_title)

        self.ui.statusBar.showMessage('Updating plot...')
        tasks = [create_task(self.update_group_input_plot(current_group)),
                 create_task(self.update_group_converted_plot(current_group)),
                 create_task(self.update_group_cut_plot(current_group)),
                 create_task(self.update_group_normal_plot(current_group)),
                 create_task(self.update_group_smooth_plot(current_group)),
                 create_task(self.update_group_baseline_plot(current_group)),
                 create_task(self.update_group_average_plot(current_group))]
        await wait(tasks)
        self.ui.statusBar.showMessage('Plot updated', 5000)

    async def update_group_input_plot(self, current_group: int | list[int]) -> None:
        data_items = self.input_plot_widget_plot_item.listDataItems()
        if len(data_items) > 0:
            items_matches = (x for x in data_items if x.get_group_number() in current_group)
            for i in data_items:
                i.setVisible(False)
                await sleep(0)
            for i in items_matches:
                i.setVisible(True)
            self.input_plot_widget_plot_item.getViewBox().updateAutoRange()

    async def update_group_converted_plot(self, current_group: int | list[int]) -> None:
        data_items = self.converted_cm_widget_plot_item.listDataItems()
        if len(data_items) > 0:
            items_matches = (x for x in data_items if x.get_group_number() in current_group)
            for i in data_items:
                i.setVisible(False)
                await sleep(0)
            for i in items_matches:
                i.setVisible(True)
            self.converted_cm_widget_plot_item.getViewBox().updateAutoRange()

    async def update_group_cut_plot(self, current_group: int | list[int]) -> None:
        data_items = self.cut_cm_plotItem.listDataItems()
        if len(data_items) > 0:
            items_matches = (x for x in data_items if x.get_group_number() in current_group)
            for i in data_items:
                i.setVisible(False)
                await sleep(0)
            for i in items_matches:
                i.setVisible(True)
            self.cut_cm_plotItem.getViewBox().updateAutoRange()

    async def update_group_normal_plot(self, current_group: int | list[int]) -> None:
        data_items = self.normalize_plotItem.listDataItems()
        if len(data_items) > 0:
            items_matches = (x for x in data_items if x.get_group_number() in current_group)
            for i in data_items:
                i.setVisible(False)
                await sleep(0)
            for i in items_matches:
                i.setVisible(True)
            self.normalize_plotItem.getViewBox().updateAutoRange()

    async def update_group_smooth_plot(self, current_group: int | list[int]) -> None:
        data_items = self.smooth_plotItem.listDataItems()
        if len(data_items) > 0:
            items_matches = (x for x in data_items if x.get_group_number() in current_group)
            for i in data_items:
                i.setVisible(False)
                await sleep(0)
            for i in items_matches:
                i.setVisible(True)
            self.smooth_plotItem.getViewBox().updateAutoRange()

    async def update_group_baseline_plot(self, current_group: int | list[int]) -> None:
        data_items = self.baseline_corrected_plotItem.listDataItems()
        if len(data_items) > 0:
            items_matches = (x for x in data_items if x.get_group_number() in current_group)
            for i in data_items:
                i.setVisible(False)
                await sleep(0)
            for i in items_matches:
                i.setVisible(True)
            self.baseline_corrected_plotItem.getViewBox().updateAutoRange()

    async def update_group_average_plot(self, current_group: int | list[int]) -> None:
        data_items = self.averaged_plotItem.listDataItems()
        if len(data_items) > 0:
            items_matches = (x for x in data_items if x.get_group_number() in current_group)
            for i in data_items:
                i.setVisible(False)
                await sleep(0)
            for i in items_matches:
                i.setVisible(True)
            self.averaged_plotItem.getViewBox().updateAutoRange()

    # endregion

    # region 'A' button
    @asyncSlot()
    async def all_control_button(self) -> None:
        """loop to set visible True for all plot items"""
        is_checked = self.ui.all_control_button.isChecked()
        if is_checked:
            self.ui.by_one_control_button.setChecked(False)
            self.ui.by_group_control_button.setChecked(False)
            self.ui.despike_history_Btn.setChecked(False)
            tasks = [create_task(self.despike_history_remove_plot()),
                     create_task(self.baseline_remove_plot()),
                     create_task(self.update_plot_all())]
            await wait(tasks)
        else:
            self.ui.all_control_button.setChecked(True)

    async def update_plot_all(self) -> None:
        self.ui.statusBar.showMessage('Updating plot...')
        self.update_all_input_plot()
        self.update_all_converted_plot()
        self.update_all_cut_plot()
        self.update_all_normal_plot()
        self.update_all_smooth_plot()
        self.update_all_baseline_plot()
        self.update_all_average_plot()
        self.ui.statusBar.showMessage('Plot updated', 5000)

    def update_all_input_plot(self) -> None:
        self.ui.input_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Imported Raman spectra</span>")
        data_items = self.input_plot_widget_plot_item.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.curveOneInputPlot:
            self.input_plot_widget_plot_item.removeItem(self.curveOneInputPlot)
        self.input_plot_widget_plot_item.getViewBox().updateAutoRange()

    def update_all_converted_plot(self) -> None:
        self.ui.converted_cm_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Converted to cm\N{superscript minus}\N{superscript one}</span>")
        data_items = self.converted_cm_widget_plot_item.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.curveOneConvertPlot:
            self.converted_cm_widget_plot_item.removeItem(self.curveOneConvertPlot)
        self.converted_cm_widget_plot_item.getViewBox().updateAutoRange()

    def update_all_cut_plot(self) -> None:
        self.ui.cut_cm_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Cutted plots</span>")
        data_items = self.cut_cm_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.curveOneCutPlot:
            self.cut_cm_plotItem.removeItem(self.curveOneCutPlot)
        self.cut_cm_plotItem.getViewBox().updateAutoRange()

    def update_all_normal_plot(self) -> None:
        if len(self.NormalizedDict) > 0:
            self.ui.normalize_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Normalized plots. Method " + self.normalization_method + "</span>")
        else:
            if len(self.NormalizedDict) > 0:
                self.ui.normalize_plot_widget.setTitle(
                    "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                        'plotText'] + ";font-size:14pt\">Normalized plots</span>")
        data_items = self.normalize_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.curveOneNormalPlot:
            self.normalize_plotItem.removeItem(self.curveOneNormalPlot)
        self.normalize_plotItem.getViewBox().updateAutoRange()

    def update_all_smooth_plot(self) -> None:
        if len(self.SmoothedDict) > 0:
            self.ui.smooth_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Smoothed plots. Method " + self.smooth_method + "</span>")
        else:
            if len(self.SmoothedDict) > 0:
                self.ui.smooth_plot_widget.setTitle(
                    "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                        'plotText'] + ";font-size:14pt\">Smoothed plots</span>")
        data_items = self.smooth_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.curveOneSmoothPlot:
            self.smooth_plotItem.removeItem(self.curveOneSmoothPlot)
        self.smooth_plotItem.getViewBox().updateAutoRange()

    def update_all_baseline_plot(self) -> None:
        if len(self.baseline_corrected_dict) > 0:
            self.ui.baseline_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Baseline corrected. Method " + self.baseline_method + "</span>")
        else:
            self.ui.smooth_plot_widget.setTitle(
                "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                    'plotText'] + ";font-size:14pt\">Smoothed </span>")
        data_items = self.baseline_corrected_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.curve_one_baseline_plot:
            self.baseline_corrected_plotItem.removeItem(self.curve_one_baseline_plot)
        self.baseline_corrected_plotItem.getViewBox().updateAutoRange()

    def update_all_average_plot(self) -> None:
        if len(self.averaged_dict) > 0:
            self.ui.average_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Average</span>")
        data_items = self.averaged_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        self.averaged_plotItem.getViewBox().updateAutoRange()

    # endregion

    @asyncSlot()
    async def despike_history_btn_clicked(self) -> None:

        async def light_by_one_button() -> None:
            """
            Change css parameter for (1) button for 0.5 sec hint
            """
            self.ui.by_one_control_button.setStyleSheet(f"""*{{border-color: {self.theme_colors['primaryColor']};}}""")
            await sleep(0.5)

        async def light_by_one_input_table() -> None:
            """
            Change css parameter for input_table_frame for 0.5 sec hint
            """
            color_hint = opacity(self.theme_colors['primaryColor'], 0.1)
            self.ui.input_table_frame.setStyleSheet(f"""*{{background-color: {color_hint}}}""")
            await sleep(0.5)

        # (1) button checked and any row selected
        current_row = self.ui.input_table.selectionModel().currentIndex().row()
        if self.ui.by_one_control_button.isChecked() and current_row != -1:
            await self.show_hide_before_despike_spectrum()
        # (1) button checked but no row selected
        elif self.ui.by_one_control_button.isChecked() and current_row == -1:
            self.ui.despike_history_Btn.setChecked(False)
            tasks = [create_task(light_by_one_input_table())]
            await wait(tasks)
            self.ui.input_table_frame.setStyleSheet(
                f"""*{{background-color: {self.theme_colors['backgroundInsideColor']}}}""")
        # (1) button NOT checked
        else:
            self.ui.despike_history_Btn.setChecked(False)
            tasks = [create_task(light_by_one_button())]
            await wait(tasks)
            self.ui.by_one_control_button.setStyleSheet('border-color: transparent')

    async def show_hide_before_despike_spectrum(self) -> None:
        current_index = self.ui.input_table.selectionModel().currentIndex()
        self.current_spectrum_despiked_name = self.ui.input_table.model().get_filename_by_row(current_index.row())
        if self.ui.despike_history_Btn.isChecked() and self.current_spectrum_despiked_name in self.BeforeDespike:
            tasks = [create_task(self.despike_history_add_plot())]
            await wait(tasks)
        else:
            tasks = [create_task(self.despike_history_remove_plot())]
            await wait(tasks)
            self.ui.despike_history_Btn.setChecked(False)

    async def despike_history_add_plot(self) -> None:
        """
        Add arrows and BeforeDespike plot item to imported_plot for compare
        """
        # selected spectrum despiked
        current_index = self.ui.input_table.selectionModel().currentIndex()
        group_number = self.ui.input_table.model().cell_data(current_index.row(), 2)
        arr = self.BeforeDespike[self.current_spectrum_despiked_name]
        if self.curveDespikedHistory:
            self.input_plot_widget_plot_item.removeItem(self.curveDespikedHistory)
        self.curveDespikedHistory = self.get_curve_plot_data_item(arr, group_number)
        self.input_plot_widget_plot_item.addItem(self.curveDespikedHistory, kargs=['ignoreBounds', 'skipAverage'])

        all_peaks = self.ui.input_table.model().cell_data(current_index.row(), 3)
        all_peaks = all_peaks.split()
        text_peaks = []
        for i in all_peaks:
            i = i.replace(',', '')
            i = i.replace(' ', '')
            text_peaks.append(i)
        list_peaks = [float(s) for s in text_peaks]
        for i in list_peaks:
            idx = find_nearest_idx(arr[:, 0], i)
            y_peak = arr[:, 1][idx]
            arrow = ArrowItem(pos=(i, y_peak), angle=-45)
            self.input_plot_widget_plot_item.addItem(arrow)

    async def despike_history_remove_plot(self) -> None:
        """
        remove old history _BeforeDespike plot item and arrows
        """
        if self.curveDespikedHistory:
            self.input_plot_widget_plot_item.removeItem(self.curveDespikedHistory)

        arrows = []
        for x in self.input_plot_widget_plot_item.items:
            if isinstance(x, ArrowItem):
                arrows.append(x)
        for i in reversed(arrows):
            self.input_plot_widget_plot_item.removeItem(i)

    async def baseline_remove_plot(self) -> None:
        if self.curveBaseline:
            self.smooth_plotItem.removeItem(self.curveBaseline)

    async def baseline_add_plot(self) -> None:
        # selected spectrum baseline
        current_index = self.ui.input_table.selectionModel().currentIndex()
        group_number = self.ui.input_table.model().cell_data(current_index.row(), 2)
        arr = self.baseline_dict[self.current_spectrum_baseline_name]
        if self.curveBaseline:
            self.smooth_plotItem.removeItem(self.curveBaseline)
        self.curveBaseline = self.get_curve_plot_data_item(arr, group_number, color=self.theme_colors['primaryColor'])
        self.smooth_plotItem.addItem(self.curveBaseline, kargs=['ignoreBounds', 'skipAverage'])

    def plots_tab_changed(self) -> None:
        self.crosshair_btn_clicked()
        self.input_plot_widget_plot_item.getViewBox().updateAutoRange()
        self.converted_cm_widget_plot_item.getViewBox().updateAutoRange()
        self.cut_cm_plotItem.getViewBox().updateAutoRange()
        self.normalize_plotItem.getViewBox().updateAutoRange()
        self.smooth_plotItem.getViewBox().updateAutoRange()
        self.baseline_corrected_plotItem.getViewBox().updateAutoRange()
        self.averaged_plotItem.getViewBox().updateAutoRange()

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
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 0:
            self.ui.smoothed_dataset_table_view.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 1:
            self.ui.baselined_dataset_table_view.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 2:
            self.ui.deconvoluted_dataset_table_view.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 4:
            self.ui.predict_table_view.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 0:
            self.ui.scrollArea_lda.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 1:
            self.ui.scrollArea_qda.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 2:
            self.ui.scrollArea_lr.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 3:
            self.ui.scrollArea_svc.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 4:
            self.ui.scrollArea_nearest.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 5:
            self.ui.scrollArea_gpc.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 6:
            self.ui.scrollArea_dt.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 7:
            self.ui.scrollArea_nb.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 8:
            self.ui.scrollArea_rf.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 9:
            self.ui.scrollArea_ab.verticalScrollBar().setValue(event)
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 10:
            self.ui.scrollArea_mlp.verticalScrollBar().setValue(event)

    def vertical_scroll_bar_enter_event(self, _: QEnterEvent) -> None:
        self.ui.verticalScrollBar.setStyleSheet("#verticalScrollBar {background: {{scrollLineHovered}};}")

    def vertical_scroll_bar_leave_event(self, _: QEvent) -> None:
        self.ui.verticalScrollBar.setStyleSheet("#verticalScrollBar {background: transparent;}")

    def move_side_scrollbar(self, idx: int) -> None:
        self.ui.verticalScrollBar.setValue(idx)

    def page1_btn_clicked(self) -> None:
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page1)
        self.ui.page1Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1)
        self.ui.page2Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2)
        self.ui.page3Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3)
        self.ui.page4Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4)
        self.ui.page5Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5)
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_1)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_1)
        self.decide_vertical_scroll_bar_visible()

    def page2_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(False)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page2)
        self.ui.page1Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1)
        self.ui.page2Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2)
        self.ui.page3Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3)
        self.ui.page4Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4)
        self.ui.page5Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5)
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_2)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_2)

    def page3_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page3)
        self.ui.page1Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1)
        self.ui.page2Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2)
        self.ui.page3Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3)
        self.ui.page4Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4)
        self.ui.page5Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5)
        self.decide_vertical_scroll_bar_visible()

    def page4_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page4)
        self.ui.page1Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1)
        self.ui.page2Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2)
        self.ui.page3Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3)
        self.ui.page4Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4)
        self.ui.page5Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5)
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_3)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_3)
        self.decide_vertical_scroll_bar_visible()

    def page5_btn_clicked(self) -> None:
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page5)
        self.ui.page1Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1)
        self.ui.page2Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2)
        self.ui.page3Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3)
        self.ui.page4Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4)
        self.ui.page5Btn.setChecked(self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5)
        self.decide_vertical_scroll_bar_visible()

    # endregion

    # region IMPORT FILES

    # noinspection PyTypeChecker
    @asyncSlot()
    async def importfile_clicked(self) -> None:
        """
        importfile_clicked for import files with Raman data
        1. select files with QFileDialog
        2. delete selected files which existing in memory already
        3. read files to 2d array by numpy
        4. using ThreadPoolExecutor add arrays to self.ImportedArray and new row to input_table
        5. update plot
        """

        username = environ.get('USERNAME')
        fd = QFileDialog()
        file_path = fd.getOpenFileNames(parent=self, caption='Select files with Raman data',
                                        directory='/users/' + str(username) + '/Documents/RS-tool',
                                        filter="Text files (*.txt *.asc)")
        for i in reversed(file_path[0]):
            if Path(i).name in self.ImportedArray:
                file_path[0].remove(i)  # exclude filename existing in ImportedArray
        if file_path[0] == '':
            self.ui.statusBar.showMessage('Не было выбрано ни одного файлы')
            return
        try:
            await self.import_files(file_path[0])
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def import_files(self, path_list: list[str]) -> None:
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Importing...')
        self.close_progress_bar()
        n_files = len(path_list)
        self._open_progress_dialog("Importing files...", "Cancel", maximum=n_files)
        self._open_progress_bar(max_value=n_files)
        laser_wl = self.ui.laser_wl_spinbox.value()
        executor = ThreadPoolExecutor()
        if n_files >= 10_000:
            executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [self.loop.run_in_executor(executor, import_spectrum, i, laser_wl) for i in
                                    path_list]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)

        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Import canceled...No one new file was selected.')
            return
        self.close_progress_bar()
        if result:
            command = CommandImportFiles(self, result, "Import")
            self.undoStack.push(command)
        else:
            self.ui.statusBar.showMessage('Import canceled...No one file with Raman data was read.')

    # endregion

    # region input_table

    def input_table_vertical_scrollbar_value_changed(self, _: QMoveEvent) -> None:
        self.decide_vertical_scroll_bar_visible()

    def decide_vertical_scroll_bar_visible(self, _model_index: QModelIndex = None, _start: int = 0,
                                           _end: int = 0) -> None:
        tv = None
        if self.ui.stackedWidget_mainpages.currentIndex() == 0:
            tv = self.ui.input_table
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 0:
            tv = self.ui.smoothed_dataset_table_view
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 1:
            tv = self.ui.baselined_dataset_table_view
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 2:
            tv = self.ui.deconvoluted_dataset_table_view
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 0:
            tv = self.ui.scrollArea_lda
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 1:
            tv = self.ui.scrollArea_qda
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 2:
            tv = self.ui.scrollArea_lr
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 3:
            tv = self.ui.scrollArea_svc
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 4:
            tv = self.ui.scrollArea_nearest
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 5:
            tv = self.ui.scrollArea_gpc
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 6:
            tv = self.ui.scrollArea_dt
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 7:
            tv = self.ui.scrollArea_nb
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 8:
            tv = self.ui.scrollArea_rf
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 9:
            tv = self.ui.scrollArea_ab
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 10:
            tv = self.ui.scrollArea_mlp
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 \
                and (self.ui.stat_tab_widget.currentIndex() == 11 or self.ui.stat_tab_widget.currentIndex() == 12):
            self.ui.verticalScrollBar.setVisible(False)
            return
        elif self.ui.stackedWidget_mainpages.currentIndex() == 4:
            tv = self.ui.predict_table_view
        if tv is None:
            return
        if isinstance(tv, QTableView):
            row_count = tv.model().rowCount()
            table_height = tv.height()
            row_height = tv.rowHeight(0)
            self.ui.verticalScrollBar.setValue(tv.verticalScrollBar().value())
            if row_count > 0:
                page_step = (table_height // row_height)
                self.ui.verticalScrollBar.setMinimum(0)
                self.ui.verticalScrollBar.setVisible(page_step <= row_count)
                self.ui.verticalScrollBar.setPageStep(page_step)
                self.ui.verticalScrollBar.setMaximum(row_count - page_step + 1)
            else:
                self.ui.verticalScrollBar.setVisible(False)
        elif isinstance(tv, QScrollArea):
            self.ui.verticalScrollBar.setValue(tv.verticalScrollBar().value())
            self.ui.verticalScrollBar.setMinimum(0)
            self.ui.verticalScrollBar.setVisible(True)
            self.ui.verticalScrollBar.setMaximum(tv.verticalScrollBar().maximum())

    def input_table_item_changed(self, top_left: QModelIndex = None, _: QModelIndex = None) \
            -> None:
        """ You can change only group column"""
        if self.ui.input_table.selectionModel().currentIndex().column() == 2:
            new_value = self.ui.input_table.model().cell_data_by_index(top_left)
            if self.ui.GroupsTable.model().rowCount() >= int(new_value) >= 0:
                filename = self.ui.input_table.model().cell_data(top_left.row(), 0)
                command = CommandChangeGroupCell(self, top_left, new_value,
                                                 "Change group number for (%s)" % str(filename))
                self.undoStack.push(command)
            else:
                self.ui.input_table.model().setData(top_left, self.previous_group_of_item, Qt.EditRole)

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
            self.previous_group_of_item = int(self.ui.input_table.model().cell_data_by_index(current_index))
        elif self.ui.by_one_control_button.isChecked():  # names
            tasks = []
            if self.current_spectrum_despiked_name:
                tasks.append(create_task(self.despike_history_remove_plot()))
            if self.current_spectrum_baseline_name:
                tasks.append(create_task(self.baseline_remove_plot()))
            tasks.append(create_task(self.update_plots_for_single()))
            await wait(tasks)

    def _input_table_header_clicked(self, idx: int):
        df = self.ui.input_table.model().dataframe()
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending_input_table = not self._ascending_input_table
        self.ui.input_table.model().sort_values(current_name, self._ascending_input_table)

    # endregion

    # region EVENTS

    def key_press_event(self, event: QKeyEvent) -> None:
        match event.key():
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
                self.action_help()

    def undo(self) -> None:
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.beforeTime = datetime.now()
        self.undoStack.undo()
        self.update_undo_redo_tooltips()

    def redo(self) -> None:
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.beforeTime = datetime.now()
        self.undoStack.redo()
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        if self.undoStack.canUndo():
            self.action_undo.setToolTip(self.undoStack.undoText())
        else:
            self.action_undo.setToolTip('')

        if self.undoStack.canRedo():
            self.action_redo.setToolTip(self.undoStack.redoText())
        else:
            self.action_redo.setToolTip('')

    # endregion

    # region Groups table
    def add_new_group(self) -> None:
        this_row = self.ui.GroupsTable.model().rowCount()
        init_color = QColor(self.theme_colors['secondaryColor'])
        color_dialog = self.color_dialog(init_color)
        color = color_dialog.getColor(init_color)
        if color.isValid():
            command = CommandAddGroup(self, this_row, color, "Add group (%s)" % str(this_row + 1))
            self.undoStack.push(command)

    @asyncSlot()
    async def group_table_cell_clicked(self) -> None:
        # change color
        current_column = self.ui.GroupsTable.selectionModel().currentIndex().column()
        selected_indexes = self.ui.input_table.selectionModel().selectedIndexes()
        if current_column == 1:
            await self.change_style_of_group()
        # change group for many rows
        elif current_column == 0 and not self.ui.by_group_control_button.isChecked() and len(selected_indexes) > 1:
            await self.change_group_for_many_rows(selected_indexes)
        # show group's spectrum only in plot
        elif current_column == 0 and self.ui.by_group_control_button.isChecked():
            await self.update_plots_for_group(None)

    async def change_style_of_group(self) -> None:
        current_row = self.ui.GroupsTable.selectionModel().currentIndex().row()
        row_data = self.ui.GroupsTable.model().row_data(current_row)
        idx = row_data.name
        style = row_data['Style']
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == idx and obj.isVisible():
                return
        window_cp = CurvePropertiesWindow(self, style, idx, fill_enabled=False)
        window_cp.sigStyleChanged.connect(self._group_style_changed)
        window_cp.show()

    def _group_style_changed(self, style: dict, old_style: dict, idx: int) -> None:
        command = CommandChangeGroupStyle(self, style, old_style, idx, "Change group (%s) style" % idx)
        self.undoStack.push(command)

    async def change_group_for_many_rows(self, selected_indexes: list[QModelIndex]) -> None:
        undo_dict = dict()
        for i in selected_indexes:
            idx = i.row()
            current_row_group = self.ui.GroupsTable.selectionModel().currentIndex().row()
            new_value = current_row_group + 1
            old_value = self.ui.input_table.model().cell_data(i.row(), 2)
            if new_value != old_value:
                undo_dict[idx] = (new_value, old_value)
        if undo_dict:
            command = CommandChangeGroupCellsBatch(self, undo_dict, 'Change group numbers for cells')
            self.undoStack.push(command)

    def dlt_selected_group(self) -> None:
        selection = self.ui.GroupsTable.selectionModel()
        row = selection.currentIndex().row()
        if row == -1:
            return
        group_number = self.ui.GroupsTable.model().row_data(row).name - 1
        if row > -1:
            name, style = self.ui.GroupsTable.model().row_data(row)
            command = CommandDeleteGroup(self, group_number, name, style, "Delete group (%s)" % str(group_number))
            self.undoStack.push(command)

    def change_plot_color_for_group(self, group_number: int, style: dict) -> None:
        plot_items = [self.input_plot_widget_plot_item,
                      self.converted_cm_widget_plot_item,
                      self.cut_cm_plotItem,
                      self.normalize_plotItem,
                      self.smooth_plotItem,
                      self.baseline_corrected_plotItem,
                      self.averaged_plotItem]
        for plot_item in plot_items:
            list_data_items = plot_item.listDataItems()
            items_matches = (x for x in list_data_items if x.get_group_number() == group_number)
            for i in items_matches:
                color = style['color']
                color.setAlphaF(1.0)
                pen = mkPen(color=color, style=style['style'], width=style['width'])
                i.setPen(pen)

    def get_names_of_group(self, group_number: int) -> list[str]:
        df = self.ui.input_table.model().dataframe()
        rows = df.loc[df['Group'] == group_number]
        filenames = rows.index
        return filenames

    def get_color_by_group_number(self, group_number: str) -> QColor:
        if group_number != '' and self.ui.GroupsTable.model().rowCount() > 0 \
                and int(group_number) <= self.ui.GroupsTable.model().rowCount():
            color = self.ui.GroupsTable.model().cell_data(int(group_number) - 1, 1)['color']
            return color
        else:
            return QColor(self.theme_colors['secondaryColor'])

    # endregion

    # region Deconvolution list
    @asyncSlot()
    async def dec_table_double_clicked(self):
        """
        When selected item in list.
        Change current spectrum in deconv_plot_widget
        """
        current_index = self.ui.dec_table.selectionModel().currentIndex()
        current_spectrum_name = self.ui.dec_table.model().cell_data(current_index.row())
        self.current_spectrum_deconvolution_name = current_spectrum_name
        self.update_single_deconvolution_plot(current_spectrum_name)
        self.redraw_curves_for_filename()
        self.set_rows_visibility()
        self.draw_sum_curve()
        self.draw_residual_curve()
        self.show_current_report_result()
        self.update_sigma3_curves(current_spectrum_name)

    def update_single_deconvolution_plot(self, current_spectrum_name: str, is_template: bool = False,
                                         is_averaged_or_group: bool = False) -> None:
        """
        Change current data spectrum in deconv_plot_widget
        set self.isTemplate

        isTemplate - True if current spectrum is averaged or group's averaged
        isAveraged_or_Group
        """
        if not self.baseline_corrected_dict:
            return
        self.isTemplate = is_template
        data_items = self.deconvolution_plotItem.listDataItems()
        arr = None
        if is_template and is_averaged_or_group:
            self.current_spectrum_deconvolution_name = ''
            if current_spectrum_name == 'Average':
                arrays_list = [self.baseline_corrected_dict[x] for x in self.baseline_corrected_dict]
                arr = get_average_spectrum(arrays_list, self.ui.average_method_cb.currentText())
                self.averaged_array = arr
                self.ui.max_noise_level_dsb.setValue(np.max(arr[:, 1]) / 100.)
            elif self.averaged_dict:
                arr = self.averaged_dict[int(current_spectrum_name.split('.')[0])]
        else:
            arr = self.baseline_corrected_dict[current_spectrum_name]
            self.current_spectrum_deconvolution_name = current_spectrum_name

        if arr is None:
            return
        if self.ui.interval_checkBox.isChecked():
            arr = cut_full_spectrum(arr, self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value())
        title_text = current_spectrum_name
        if is_template:
            title_text = 'Template. ' + current_spectrum_name

        new_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">" + title_text + "</span>"
        self.ui.deconv_plot_widget.setTitle(new_title)

        for i in data_items:
            if isinstance(i, PlotDataItem):
                i.setVisible(False)

        if self.data_curve:
            self.deconvolution_plotItem.removeItem(self.data_curve)
        if self.sum_curve:
            self.deconvolution_plotItem.removeItem(self.sum_curve)
        if self.residual_curve:
            self.deconvolution_plotItem.removeItem(self.residual_curve)
        # if self.ui.sun_Btn.isChecked():
        #     color = self.theme_colors['inversePlotText']
        # else:
        #     color = self.theme_colors['secondaryColor']

        self.data_curve = self.get_curve_plot_data_item(arr, color=self.data_style['color'], name='data')
        x_axis, y_axis = self.sum_array()
        self.sum_curve = self.get_curve_plot_data_item(np.vstack((x_axis, y_axis)).T, color=self.sum_style['color'],
                                                       name='sum')
        x_res, y_res = self.residual_array()
        self.residual_curve = self.get_curve_plot_data_item(np.vstack((x_res, y_res)).T,
                                                            color=self.residual_style['color'], name='sum')

        self.data_curve.setVisible(self.ui.data_checkBox.isChecked())
        self.sum_curve.setVisible(self.ui.sum_checkBox.isChecked())
        self.residual_curve.setVisible(self.ui.residual_checkBox.isChecked())
        self.deconvolution_plotItem.addItem(self.data_curve, kargs=['ignoreBounds', 'skipAverage'])
        self.deconvolution_plotItem.addItem(self.sum_curve, kargs=['ignoreBounds', 'skipAverage'])
        self.deconvolution_plotItem.addItem(self.residual_curve, kargs=['ignoreBounds', 'skipAverage'])
        self.deconvolution_plotItem.getViewBox().updateAutoRange()

    def add_line_params_from_template_batch(self, keys: list[str]) -> None:
        key = keys[0]
        self.add_line_params_from_template(key)
        df_a = self.ui.fit_params_table.model().dataframe()
        tuples = []
        for i in df_a.index:
            if i[0] == '':
                tuples.append(i)
        df_a = df_a.loc['']
        mi = MultiIndex.from_tuples(tuples, names=('filename', 'line_index', 'param_name'))
        df_a.index = mi
        df_c = df_a.copy(deep=True)
        for key in keys:
            tuples_b = []
            for filename, line_idx, param_name in tuples:
                tuples_b.append((key, line_idx, param_name))
            df_b = df_a.copy(deep=True)
            mi = MultiIndex.from_tuples(tuples_b, names=('filename', 'line_index', 'param_name'))
            df_b.index = mi
            df_c = concat([df_c, df_b])
        self.ui.fit_params_table.model().set_dataframe(df_c)
        self.ui.fit_params_table.model().sort_index()

    def add_line_params_from_template(self, filename: str | None = None) -> None:
        """
        When selected item in list.
        update parameters for lines of current spectrum filename
        if no parameters for filename - copy from template and update limits of amplitude parameter
        """
        model = self.ui.fit_params_table.model()
        model.delete_rows_by_filenames(filename)
        if filename is None:
            filename = self.current_spectrum_deconvolution_name
        df_a = self.ui.fit_params_table.model().dataframe()
        tuples = []
        for i in df_a.index:
            if i[0] == '':
                tuples.append(i)
        df_a = df_a.loc['']
        tuples_b = []
        for _, line_idx, param_name in tuples:
            tuples_b.append((filename, line_idx, param_name))
        df_b = df_a.copy(deep=True)
        mi = MultiIndex.from_tuples(tuples_b, names=('filename', 'line_index', 'param_name'))
        df_b.index = mi
        model.concat_df(df_b)
        # df_b = model.get_df_by_filename(filename)
        # if df_b is None:
        #     df_c = df_a
        # else:
        #     df_c = merge(df_a, df_b, on=['line_index', 'param_name'], how="outer",
        #                  indicator=True).query('_merge=="left_only"')
        # if df_c.empty:
        #     for i in range(len(df_a)):
        #         ser = df_a.iloc[i]
        #         model.append_row(ser.name[0], ser.name[1], ser.Value, ser['Min value'], ser['Max value'],
        #                          filename)
        #     return
        # for i in range(len(df_c)):
        #     ser = df_c.iloc[i]
        #     model.append_row(ser.name[0], ser.name[1], ser.Value_x, ser['Min value_x'], ser['Max value_x'], filename)
        # df_d = merge(df_b, df_a, on=['line_index', 'param_name'], how="outer",
        #              indicator=True).query('_merge=="left_only"')
        # for i in range(len(df_d)):
        #     ser = df_d.iloc[i]
        #     model.delete_rows_multiindex((filename, ser.name[0], ser.name[1]))

    # endregion

    # region ACTIONS FILE MENU

    def action_new_project(self) -> None:
        username = environ.get('USERNAME')
        fd = QFileDialog()
        file_path = fd.getSaveFileName(self, 'Create Project File',
                                       '/users/' + str(username) + '/Documents/RS-tool', "ZIP (*.zip)")
        if file_path[0] != '':
            try:
                f = shelve_open(file_path[0], 'n')
                f.close()
                self.open_project(file_path[0], new=True)
            except BaseException:
                raise

    def action_open_project(self) -> None:
        username = environ.get('USERNAME')
        fd = QFileDialog()
        file_path = fd.getOpenFileName(self, 'Select RS-tool project file to open',
                                       '/users/' + str(username) + '/Documents/RS-tool', "(*.zip)")
        if file_path[0] != '':
            if not self.ask_before_close_project():
                return
            self.open_project(file_path[0])
            self.load_params(file_path[0])


    def ask_before_close_project(self):
        if self.modified:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setText('You have unsaved changes. ' + '\n' + 'Save changes before exit?')
            if self.project_path:
                msg.setInformativeText(self.project_path)
            msg.setWindowTitle("Close")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            result = msg.exec()
            if result == QMessageBox.StandardButton.Yes:
                self.action_save_project()
                return True
            elif result == QMessageBox.StandardButton.Cancel:
                return False
            elif result == QMessageBox.StandardButton.No:
                return True
        else:
            return True

    def action_open_recent(self, path: str) -> None:
        if Path(path).exists():
            if not self.ask_before_close_project():
                return
            self.open_project(path)
            self.load_params(path)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Selected file doesn't exists")
            msg.setInformativeText(path)
            msg.setWindowTitle("Recent file open error")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

    def action_save_project(self) -> None:
        if self.project_path == '' or self.project_path is None:
            username = environ.get('USERNAME')
            fd = QFileDialog()
            file_path = fd.getSaveFileName(self, 'Create Project File',
                                           '/users/' + str(username) + '/Documents/RS-tool', "ZIP (*.zip)")
            if file_path[0] != '':
                self.save_by_shelve(file_path[0])
                self.ui.projectLabel.setText(file_path[0])
                self.setWindowTitle(file_path[0])
                self._add_path_to_recent(file_path[0])
                self.update_recent_list()
                self.project_path = file_path[0]
        else:
            self.save_by_shelve(self.project_path)

    @asyncSlot()
    async def action_save_production_project(self) -> None:
        username = environ.get('USERNAME')
        fd = QFileDialog()
        file_path = fd.getSaveFileName(self, 'Create Production Project File',
                                       '/users/' + str(username) + '/Documents/RS-tool', "ZIP (*.zip)")
        if file_path[0] != '':
            self.save_by_shelve(file_path[0], True)

    @asyncSlot()
    async def action_save_as(self) -> None:
        username = environ.get('USERNAME')
        fd = QFileDialog()
        file_path = fd.getSaveFileName(self, 'Create Project File',
                                       '/users/' + str(username) + '/Documents/RS-tool', "ZIP (*.zip)")
        if file_path[0] != '':
            self.save_by_shelve(file_path[0])
            self.project_path = file_path[0]

    def save_by_shelve(self, path: str, production_export: bool = False) -> None:
        self.ui.statusBar.showMessage('Saving file...')
        self.close_progress_bar()
        self._open_progress_bar()
        filename = str(Path(path).parent) + '/' + str(Path(path).stem)
        with shelve_open(filename, 'n') as db:
            db["GroupsTable"] = self.ui.GroupsTable.model().dataframe()
            db["DeconvLinesTableDF"] = self.ui.deconv_lines_table.model().dataframe()
            db["DeconvParamsTableDF"] = self.ui.fit_params_table.model().dataframe()
            db["intervals_table_df"] = self.ui.fit_intervals_table_view.model().dataframe()
            db["DeconvLinesTableChecked"] = self.ui.deconv_lines_table.model().checked()
            db["LaserWL"] = self.ui.laser_wl_spinbox.value()
            db["Maxima_count_despike"] = self.ui.maxima_count_despike_spin_box.value()
            db["Despike_fwhm_width"] = self.ui.despike_fwhm_width_doubleSpinBox.value()
            db["cm_CutRangeStart"] = self.ui.cm_range_start.value()
            db["cm_CutRangeEnd"] = self.ui.cm_range_end.value()
            db["trim_start_cm"] = self.ui.trim_start_cm.value()
            db["trim_end_cm"] = self.ui.trim_end_cm.value()
            db["interval_start_cm"] = self.ui.interval_start_dsb.value()
            db["interval_end_cm"] = self.ui.interval_end_dsb.value()
            db["neg_grad_factor_spinBox"] = self.ui.neg_grad_factor_spinBox.value()
            db["normalizing_method_comboBox"] = self.ui.normalizing_method_comboBox.currentText()
            db["normalizing_method_used"] = self.normalization_method
            db["smoothing_method_comboBox"] = self.ui.smoothing_method_comboBox.currentText()
            db["guess_method_cb"] = self.ui.guess_method_cb.currentText()
            db["average_function"] = self.ui.average_method_cb.currentText()
            db["dataset_type_cb"] = self.ui.dataset_type_cb.currentText()
            db["classes_lineEdit"] = self.ui.classes_lineEdit.text()
            db["test_data_ratio_spinBox"] = self.ui.test_data_ratio_spinBox.value()
            db["n_lines_method"] = self.ui.n_lines_detect_method_cb.currentText()
            db["max_noise_level"] = self.ui.max_noise_level_dsb.value()
            db["l_ratio_doubleSpinBox"] = self.ui.l_ratio_doubleSpinBox.value()
            db["smooth_method"] = self.smooth_method
            db["window_length_spinBox"] = self.ui.window_length_spinBox.value()
            db["smooth_polyorder_spinBox"] = self.ui.smooth_polyorder_spinBox.value()
            db["whittaker_lambda_spinBox"] = self.ui.whittaker_lambda_spinBox.value()
            db["kaiser_beta"] = self.ui.kaiser_beta_doubleSpinBox.value()
            db['EMD_noise_modes'] = self.ui.emd_noise_modes_spinBox.value()
            db['EEMD_trials'] = self.ui.eemd_trials_spinBox.value()
            db['sigma'] = self.ui.sigma_spinBox.value()
            db['lambda_spinBox'] = self.ui.lambda_spinBox.value()
            db['eta'] = self.ui.eta_doubleSpinBox.value()
            db['N_iterations'] = self.ui.n_iterations_spinBox.value()
            db['p_doubleSpinBox'] = self.ui.p_doubleSpinBox.value()
            db['polynome_degree'] = self.ui.polynome_degree_spinBox.value()
            db['grad'] = self.ui.grad_doubleSpinBox.value()
            db['quantile'] = self.ui.quantile_doubleSpinBox.value()
            db['alpha_factor'] = self.ui.alpha_factor_doubleSpinBox.value()
            db['peak_ratio'] = self.ui.peak_ratio_doubleSpinBox.value()
            db['spline_degree'] = self.ui.spline_degree_spinBox.value()
            db['num_std'] = self.ui.num_std_doubleSpinBox.value()
            db['interp_half_window'] = self.ui.interp_half_window_spinBox.value()
            db['sections'] = self.ui.sections_spinBox.value()
            db['min_length'] = self.ui.min_length_spinBox.value()
            db['fill_half_window'] = self.ui.fill_half_window_spinBox.value()
            db['scale'] = self.ui.scale_doubleSpinBox.value()
            db['cost_function'] = self.ui.cost_func_comboBox.currentText()
            db['opt_method_oer'] = self.ui.opt_method_oer_comboBox.currentText()
            db['fraction'] = self.ui.fraction_doubleSpinBox.value()
            db['EMSC_N_PCA'] = self.ui.emsc_pca_n_spinBox.value()
            db['max_CCD_value'] = self.ui.max_CCD_value_spinBox.value()
            db['baseline_method_comboBox'] = self.ui.baseline_correction_method_comboBox.currentText()
            db["baseline_method"] = self.baseline_method
            db["data_style"] = self.data_style.copy()
            db["data_curve_checked"] = self.ui.data_checkBox.isChecked()
            db["sum_style"] = self.sum_style.copy()
            db["sigma3_style"] = self.sigma3_style.copy()
            db["sum_curve_checked"] = self.ui.sum_checkBox.isChecked()
            db["sigma3_checked"] = self.ui.sigma3_checkBox.isChecked()
            db["residual_style"] = self.residual_style.copy()
            db["residual_curve_checked"] = self.ui.residual_checkBox.isChecked()
            db["interval_checkBox_checked"] = self.ui.interval_checkBox.isChecked()
            db["fit_method"] = self.ui.fit_opt_method_comboBox.currentText()
            db["use_fit_intervals"] = self.ui.intervals_gb.isChecked()
            db["max_dx_guess"] = self.ui.max_dx_dsb.value()
            db["_y_axis_ref_EMSC"] = self._y_axis_ref_EMSC
            db['baseline_corrected_dict'] = self.baseline_corrected_dict

            if not self.is_production_project:
                self.stat_models = {}
                for key, v in self.latest_stat_result.items():
                    self.stat_models[key] = v['model']
                db["stat_models"] = self.stat_models
                if self.ImportedArray:
                    db['interp_ref_array'] = next(iter(self.ImportedArray.values()))
            db["averaged_dict"] = self.averaged_dict.copy()
            if not production_export:
                db["InputTable"] = self.ui.input_table.model().dataframe()
                db["smoothed_dataset_df"] = self.ui.smoothed_dataset_table_view.model().dataframe()
                db["baselined_dataset_df"] = self.ui.baselined_dataset_table_view.model().dataframe()
                db["deconvoluted_dataset_df"] = self.ui.deconvoluted_dataset_table_view.model().dataframe()
                db["predict_df"] = self.ui.predict_table_view.model().dataframe()
                db["ImportedArray"] = self.ImportedArray
                db["ConvertedDict"] = self.ConvertedDict
                db["BeforeDespike"] = self.BeforeDespike
                db["NormalizedDict"] = self.NormalizedDict
                db["CuttedFirstDict"] = self.CuttedFirstDict
                db["SmoothedDict"] = self.SmoothedDict
                db['baseline_corrected_not_trimmed_dict'] = self.baseline_corrected_not_trimmed_dict
                db["baseline_dict"] = self.baseline_dict

                db["report_result"] = self.report_result.copy()
                db["sigma3"] = self.sigma3.copy()
                db["latest_stat_result"] = self.latest_stat_result
                db["is_production_project"] = False
            else:
                db["is_production_project"] = True
            if self.is_production_project:
                db["is_production_project"] = True
                db['interp_ref_array'] = self.interp_ref_array
                db['stat_models'] = self.stat_models
        zf = ZipFile(filename + '.zip', "w", ZIP_DEFLATED, compresslevel=9)
        zf.write(filename + '.dat', "data.dat")
        zf.write(filename + '.dir', "data.dir")
        zf.write(filename + '.bak', "data.bak")
        self.ui.statusBar.showMessage('File saved.   ' + filename + '.zip', 10000)
        self.ui.projectLabel.setText(filename + '.zip')
        self.setWindowTitle(filename + '.zip')
        self.set_modified(False)
        Path(filename + '.dat').unlink()
        Path(filename + '.dir').unlink()
        Path(filename + '.bak').unlink()
        self.close_progress_bar()

    def action_close_project(self) -> None:
        self.setWindowTitle(' ')
        self.ui.projectLabel.setText('')
        self.project_path = ''
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
        self.update_recent_list()
        self.project_path = path
        self._clear_all_parameters()

    def _add_path_to_recent(self, _path: str) -> None:
        with Path('recent-files.txt').open() as file:
            lines = [line.rstrip() for line in file]
        lines = list(dict.fromkeys(lines))
        if Path(_path).suffix == '':
            _path += '.zip'

        if len(lines) > 0 and _path in lines:
            lines.remove(_path)

        with Path('recent-files.txt').open('a') as f:
            f.truncate(0)
            lines_fin = []
            recent_limite = int(self.recent_limit)
            for idx, i in enumerate(reversed(lines)):
                if idx < recent_limite - 1:
                    lines_fin.append(i)
                idx += 1
            for line in reversed(lines_fin):
                f.write(line + '\n')
            f.write(_path + '\n')

        if not self.recent_menu.isEnabled():
            self.recent_menu.setDisabled(False)

    def _export_files_av(self, item: tuple[str, np.ndarray]) -> None:
        filename = self.ui.GroupsTable.model().get_group_name_by_int(item[0])
        np.savetxt(fname=self.export_folder_path + '/' + filename + '.asc', X=item[1], fmt='%10.5f')

    def _export_files(self, item: tuple[str, np.ndarray]) -> None:
        np.savetxt(fname=self.export_folder_path + '/' + item[0], X=item[1], fmt='%10.5f')

    @asyncSlot()
    async def action_export_files_nm(self) -> None:
        if not self.ImportedArray:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No files to save")
            msg.setWindowTitle("Export failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        username = environ.get('USERNAME')
        fd = QFileDialog()
        folder_path = fd.getExistingDirectory(self, 'Choose folder to export files in nm',
                                              '/users/' + str(username) + '/Documents/RS-tool')
        if folder_path:
            self.ui.statusBar.showMessage('Exporting files...')
            self.close_progress_bar()
            self.progressBar = QProgressBar()
            self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.export_folder_path = folder_path + '/nm'
            if not Path(self.export_folder_path).exists():
                Path(self.export_folder_path).mkdir(parents=True)
            with ThreadPoolExecutor() as executor:
                self.current_futures = [loop.run_in_executor(executor, self._export_files, i)
                                        for i in self.ImportedArray.items()]
                await gather(*self.current_futures)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Export completed. {} new files created.'.format(len(self.ImportedArray)),
                                          50_000)

    @asyncSlot()
    async def action_export_files_cm(self) -> None:

        if not self.baseline_corrected_dict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No files to save")
            msg.setWindowTitle("Export failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        username = environ.get('USERNAME')
        fd = QFileDialog()
        folder_path = fd.getExistingDirectory(self, 'Choose folder to export files in cm-1',
                                              '/users/' + str(username) + '/Documents/RS-tool')
        if folder_path:
            self.ui.statusBar.showMessage('Exporting files...')
            self.close_progress_bar()
            self.progressBar = QProgressBar()
            self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.export_folder_path = folder_path + '/cm-1'
            if not Path(self.export_folder_path).exists():
                Path(self.export_folder_path).mkdir(parents=True)
            with ThreadPoolExecutor() as executor:
                self.current_futures = [loop.run_in_executor(executor, self._export_files, i)
                                        for i in self.baseline_corrected_dict.items()]
                await gather(*self.current_futures)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Export completed. {} new files '
                                          'created.'.format(len(self.baseline_corrected_dict)), 50_000)

    @asyncSlot()
    async def action_export_average(self) -> None:

        if not self.averaged_dict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No files to save")
            msg.setWindowTitle("Export failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        username = environ.get('USERNAME')
        fd = QFileDialog()
        folder_path = fd.getExistingDirectory(self, 'Choose folder to export files in cm-1',
                                              '/users/' + str(username) + '/Documents/RS-tool')
        if folder_path:
            self.ui.statusBar.showMessage('Exporting files...')
            self.close_progress_bar()
            self.progressBar = QProgressBar()
            self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.export_folder_path = folder_path + '/average'
            if not Path(self.export_folder_path).exists():
                Path(self.export_folder_path).mkdir(parents=True)
            with ThreadPoolExecutor() as executor:
                self.current_futures = [loop.run_in_executor(executor, self._export_files_av, i)
                                        for i in self.averaged_dict.items()]
                await gather(*self.current_futures)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Export completed', 50_000)

    @asyncSlot()
    async def action_export_table_excel(self) -> None:
        if not self.ImportedArray:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Table is empty")
            msg.setWindowTitle("Export failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        username = environ.get('USERNAME')
        fd = QFileDialog()
        folder_path = fd.getExistingDirectory(self, 'Choose folder to save excel file', '/users/'
                                              + str(username) + '/Documents/RS-tool')
        if not folder_path:
            return
        self.ui.statusBar.showMessage('Saving file...')
        self.close_progress_bar()
        self._open_progress_bar(max_value=0)
        self._open_progress_dialog("Exporting Excel...", "Cancel", maximum=0)
        await self.loop.run_in_executor(None, self.excel_write, folder_path)

        self.close_progress_bar()
        self.ui.statusBar.showMessage('Excel file saved to ' + folder_path)

    def excel_write(self, folder_path) -> None:
        with ExcelWriter(folder_path + '\output.xlsx') as writer:
            self.ui.input_table.model().dataframe().to_excel(writer, sheet_name='Spectrum info')
            if self.ui.deconv_lines_table.model().rowCount() > 0:
                self.ui.deconv_lines_table.model().dataframe().to_excel(writer, sheet_name='Fit lines')
            if self.ui.fit_params_table.model().rowCount() > 0:
                self.ui.fit_params_table.model().dataframe().to_excel(writer,
                                                                      sheet_name='Fit initial params')
            if self.ui.smoothed_dataset_table_view.model().rowCount() > 0:
                self.ui.smoothed_dataset_table_view.model().dataframe().to_excel(writer,
                                                                                 sheet_name='Smoothed dataset')
            if self.ui.baselined_dataset_table_view.model().rowCount() > 0:
                self.ui.baselined_dataset_table_view.model().dataframe().to_excel(writer,
                                                                                  sheet_name='Pure Raman dataset')
            if self.ui.deconvoluted_dataset_table_view.model().rowCount() > 0:
                self.ui.deconvoluted_dataset_table_view.model().dataframe().to_excel(writer,
                                                                                     sheet_name='Deconvoluted dataset')
            if self.ui.pca_features_table_view.model().rowCount() > 0:
                self.ui.pca_features_table_view.model().dataframe().to_excel(writer, sheet_name='PCA loadings')
            if self.ui.plsda_vip_table_view.model().rowCount() > 0:
                self.ui.plsda_vip_table_view.model().dataframe().to_excel(writer, sheet_name='PLS-DA VIP')
            if self.ui.predict_table_view.model().rowCount() > 0:
                self.ui.predict_table_view.model().dataframe().to_excel(writer, sheet_name='Predicted')

    def clear_selected_step(self, step: str) -> None:
        match step:
            case 'Converted':
                self.ConvertedDict.clear()
                self.curveOneConvertPlot = None
                self.converted_cm_widget_plot_item.clear()
                self._initial_converted_cm_plot()
            case 'Cutted':
                self.CuttedFirstDict.clear()
                self.curveOneCutPlot = None
                self.cut_cm_plotItem.clear()
                self._initial_cut_cm_plot()
            case 'Normalized':
                self.NormalizedDict.clear()
                self.curveOneNormalPlot = None
                self.normalize_plotItem.clear()
                self._initial_normalize_plot()
            case 'Smoothed':
                self.SmoothedDict.clear()
                self.curveOneSmoothPlot = None
                self.smooth_plotItem.clear()
                self._initial_smooth_plot()
                self.ui.smoothed_dataset_table_view.model().clear_dataframe()
            case 'Baseline':
                self.baseline_corrected_dict.clear()
                self.baseline_corrected_not_trimmed_dict.clear()
                self.baseline_dict.clear()
                self.curve_one_baseline_plot = None
                self.baseline_corrected_plotItem.clear()
                self.ui.baselined_dataset_table_view.model().clear_dataframe()
                self._initial_baseline_plot()
            case 'Averaged':
                self.averaged_dict.clear()
                self.averaged_plotItem.clear()
                self._initial_averaged_plot()
            case 'Deconvolution':
                self.deconvolution_plotItem.clear()
                self.report_result.clear()
                self.sigma3.clear()
                del self.fill
                self._initial_deconvolution_plot()
                self.data_curve = None
                self.current_spectrum_deconvolution_name = ''
                try:
                    self.ui.template_combo_box.currentTextChanged.disconnect(self.switch_to_template)
                except:
                    error('failed to disconnect currentTextChanged self.switch_to_template)')
                self.ui.template_combo_box.clear()
                self.isTemplate = False
                self.ui.data_checkBox.setChecked(True)
                self.ui.sum_checkBox.setChecked(False)
                self.ui.residual_checkBox.setChecked(False)
                self.ui.sigma3_checkBox.setChecked(False)
                self.data_style_button_style_sheet(self.data_style['color'].name())
                self.sum_style_button_style_sheet(self.sum_style['color'].name())
                self.sigma3_style_button_style_sheet(self.sigma3_style['color'].name())
                self.residual_style_button_style_sheet(self.residual_style['color'].name())
                self.ui.interval_checkBox.setChecked(False)
                self.linearRegionDeconv.setVisible(False)
                self.ui.deconvoluted_dataset_table_view.model().clear_dataframe()
                self.update_template_combo_box()
            case 'Stat':
                self._initial_all_stat_plots()
                self.ui.current_group_shap_comboBox.clear()
                self.latest_stat_result = {}
                self._initial_pca_features_table()
                self._initial_plsda_vip_table()
            case 'LDA':
                self._initial_lda_plots()
                del self.latest_stat_result['LDA']
            case 'QDA':
                self._initial_qda_plots()
                del self.latest_stat_result['QDA']
            case 'Logistic regression':
                self._initial_lr_plots()
                del self.latest_stat_result['Logistic regression']
            case 'NuSVC':
                self._initial_svc_plots()
                del self.latest_stat_result['NuSVC']
            case 'Nearest Neighbors':
                self._initial_nearest_plots()
                del self.latest_stat_result['Nearest Neighbors']
            case 'GPC':
                self._initial_gpc_plots()
                del self.latest_stat_result['GPC']
            case 'Decision Tree':
                self._initial_dt_plots()
                del self.latest_stat_result['Decision Tree']
            case 'Naive Bayes':
                self._initial_nb_plots()
                del self.latest_stat_result['Naive Bayes']
            case 'Random Forest':
                self._initial_rf_plots()
                del self.latest_stat_result['Random Forest']
            case 'AdaBoost':
                self._initial_ab_plots()
                del self.latest_stat_result['AdaBoost']
            case 'MLP':
                self._initial_mlp_plots()
                del self.latest_stat_result['MLP']
            case 'PCA':
                self._initial_pca_plots()
                self._initial_pca_features_table()
                del self.latest_stat_result['PCA']
            case 'PLS-DA':
                self._initial_plsda_plots()
                self._initial_plsda_vip_table()
                del self.latest_stat_result['PLS-DA']
            case 'Page5':
                self._initial_predict_dataset_table()

    def _clear_all_parameters(self) -> None:
        before = datetime.now()
        self.close_progress_bar()
        self._open_progress_bar()
        self.executor_stop()
        self.CommandEndIntervalChanged_allowed = False
        self.CommandStartIntervalChanged_allowed = False
        self._init_default_values()
        self.ui.GroupsTable.model().clear_dataframe()
        self.ui.input_table.model().clear_dataframe()
        self.ui.dec_table.model().clear_dataframe()
        self.ui.deconv_lines_table.model().clear_dataframe()
        self.ui.fit_params_table.model().clear_dataframe()
        self.ui.fit_intervals_table_view.model().clear_dataframe()
        self.ui.smoothed_dataset_table_view.model().clear_dataframe()
        self._reset_smoothed_dataset_table()
        self.ui.baselined_dataset_table_view.model().clear_dataframe()
        self._reset_baselined_dataset_table()
        self.ui.deconvoluted_dataset_table_view.model().clear_dataframe()
        self._reset_deconvoluted_dataset_table()
        self.is_production_project = False
        self.stat_models = {}
        self.interp_ref_array = None
        self.is_production_project = False
        self.top_features = {}
        self.latest_stat_result = {}
        self.clear_selected_step('Converted')
        self.clear_selected_step('Cutted')
        self.clear_selected_step('Normalized')
        self.clear_selected_step('Smoothed')
        self.clear_selected_step('Baseline')
        self.clear_selected_step('Averaged')
        self.clear_selected_step('Deconvolution')
        self.clear_selected_step('Stat')
        self.clear_selected_step('Page5')

        self.ImportedArray.clear()
        self.BeforeDespike.clear()
        self.baseline_dict.clear()
        self.baseline_corrected_dict.clear()
        self.baseline_corrected_not_trimmed_dict.clear()
        self.averaged_dict.clear()
        self.curveOneInputPlot = None
        self.curveDespikedHistory = None
        self.curveBaseline = None
        self.input_plot_widget_plot_item.clear()
        self.ui.crosshairBtn.setChecked(False)
        self.crosshair_btn_clicked()
        self.ui.by_one_control_button.setChecked(False)
        self.ui.by_group_control_button.setChecked(False)
        self.ui.all_control_button.setChecked(True)
        self.ui.despike_history_Btn.setChecked(False)
        self.ui.stat_report_text_edit.setText('')
        self._initial_input_plot()
        self._set_parameters_to_default()
        self.set_buttons_ability()
        collect(2)
        self.undoStack.clear()
        self.CommandEndIntervalChanged_allowed = True
        self.CommandStartIntervalChanged_allowed = True
        self.set_modified(False)
        self.close_progress_bar()
        seconds = round((datetime.now() - before).total_seconds())
        self.ui.statusBar.showMessage('Closed for ' + str(seconds) + ' sec.', 5000)

    def set_forms_ability(self) -> None:
        self.ui.left_side_head_stackedWidget.setEnabled(not self.is_production_project)
        self.ui.stackedWidget_left.setEnabled(not self.is_production_project)
        self.ui.deconv_lines_table.setEnabled(not self.is_production_project)
        self.ui.deconv_buttons_frame_top.setEnabled(not self.is_production_project)
        self.ui.fit_params_table.setEnabled(not self.is_production_project)

    @asyncSlot()
    async def load_params(self, path: str) -> None:
        self.ui.statusBar.showMessage('Reading data file...')
        self.close_progress_bar()
        self._open_progress_bar()
        self._open_progress_dialog("Opening project...", "Cancel")
        self.beforeTime = datetime.now()
        self.unzip_project_file(path)
        await self.update_all_plots()
        self.update_template_combo_box()
        await self.switch_to_template()
        self.update_deconv_intervals_limits()
        self.dataset_type_cb_current_text_changed(self.ui.dataset_type_cb.currentText())
        if self.ui.fit_params_table.model().rowCount() != 0 \
                and self.ui.deconv_lines_table.model().rowCount() != 0:
            await self.draw_all_curves()
        await self.redraw_stat_plots()
        self.update_force_single_plots()
        self.update_force_full_plots()
        # if self.ui.interval_checkBox.isChecked():
        #     self.interval_cb_state_changed(2 if self.ui.interval_checkBox.isChecked() else 0)
        self.currentProgress.setMaximum(1)
        self.currentProgress.setValue(1)
        self.close_progress_bar()
        self.set_buttons_ability()
        self.set_forms_ability()
        seconds = round((datetime.now() - self.beforeTime).total_seconds())
        self.set_modified(False)
        self.decide_vertical_scroll_bar_visible()
        self.ui.statusBar.showMessage('Project opened for ' + str(seconds) + ' sec.', 5000)

    def unzip_project_file(self, path: str) -> None:
        with ZipFile(path) as archive:
            directory = Path(path).parent
            archive.extractall(directory)
            if Path(str(directory) + '/data.dat').exists():
                file_name = str(directory) + '/data'
                self.unshelve_project_file(file_name)
                Path(str(directory) + '/data.dat').unlink()
                Path(str(directory) + '/data.dir').unlink()
                Path(str(directory) + '/data.bak').unlink()

    def unshelve_project_file(self, file_name: str) -> None:
        with shelve_open(file_name, 'r') as db:
            if "GroupsTable" in db:
                df = db["GroupsTable"]
                self.ui.GroupsTable.model().set_dataframe(df)
            if "InputTable" in db:
                df = db["InputTable"]
                self.ui.input_table.model().set_dataframe(df)
                names = df.index
                self.ui.dec_table.model().append_row_deconv_table(filename=names)
            if "DeconvLinesTableDF" in db:
                df = db["DeconvLinesTableDF"]
                self.ui.deconv_lines_table.model().set_dataframe(df)
            if "DeconvLinesTableChecked" in db:
                checked = db["DeconvLinesTableChecked"]
                self.ui.deconv_lines_table.model().set_checked(checked)
            if "DeconvParamsTableDF" in db:
                df = db["DeconvParamsTableDF"]
                self.ui.fit_params_table.model().set_dataframe(df)
            if "intervals_table_df" in db:
                df = db["intervals_table_df"]
                self.ui.fit_intervals_table_view.model().set_dataframe(df)
            if "smoothed_dataset_df" in db:
                self.ui.smoothed_dataset_table_view.model().set_dataframe(db["smoothed_dataset_df"])
            if "stat_models" in db:
                self.stat_models = db["stat_models"]
            if "baselined_dataset_df" in db:
                self.ui.baselined_dataset_table_view.model().set_dataframe(db["baselined_dataset_df"])
            if "deconvoluted_dataset_df" in db:
                self.ui.deconvoluted_dataset_table_view.model().set_dataframe(db["deconvoluted_dataset_df"])
            if "interp_ref_array" in db:
                self.interp_ref_array = db["interp_ref_array"]
            if "predict_df" in db:
                self.ui.predict_table_view.model().set_dataframe(db["predict_df"])
            if "is_production_project" in db:
                print("is_production_project in db", db["is_production_project"])
                self.is_production_project = db["is_production_project"]
            if "ImportedArray" in db:
                self.ImportedArray = db["ImportedArray"]
            if "cm_CutRangeStart" in db:
                self.ui.cm_range_start.setValue(db["cm_CutRangeStart"])
            if "cm_CutRangeEnd" in db:
                self.ui.cm_range_end.setValue(db["cm_CutRangeEnd"])
            if "trim_start_cm" in db:
                self.ui.trim_start_cm.setValue(db["trim_start_cm"])
            if "trim_end_cm" in db:
                self.ui.trim_end_cm.setValue(db["trim_end_cm"])
            if "_y_axis_ref_EMSC" in db:
                self._y_axis_ref_EMSC = db["_y_axis_ref_EMSC"]
            if "interval_start_cm" in db:
                self.ui.interval_start_dsb.setValue(db["interval_start_cm"])
            if "interval_end_cm" in db:
                self.ui.interval_end_dsb.setValue(db["interval_end_cm"])
            if "ConvertedDict" in db:
                self.ConvertedDict = db["ConvertedDict"]
                self.update_cm_min_max_range()
            if "LaserWL" in db:
                self.ui.laser_wl_spinbox.setValue(db["LaserWL"])
            if "BeforeDespike" in db:
                self.BeforeDespike = db["BeforeDespike"]
            if "Maxima_count_despike" in db:
                self.ui.maxima_count_despike_spin_box.setValue(db["Maxima_count_despike"])
            if "Despike_fwhm_width" in db:
                self.ui.despike_fwhm_width_doubleSpinBox.setValue(db["Despike_fwhm_width"])
            if "CuttedFirstDict" in db:
                self.CuttedFirstDict = db["CuttedFirstDict"]
            if "averaged_dict" in db:
                self.averaged_dict = db["averaged_dict"]
            if "report_result" in db:
                self.report_result = db["report_result"]
            if "sigma3" in db:
                self.sigma3 = db["sigma3"]
            if "neg_grad_factor_spinBox" in db:
                self.ui.neg_grad_factor_spinBox.setValue(db["neg_grad_factor_spinBox"])
            if "NormalizedDict" in db:
                self.NormalizedDict = db["NormalizedDict"]
            if "normalizing_method_comboBox" in db:
                self.ui.normalizing_method_comboBox.setCurrentText(db["normalizing_method_comboBox"])
            if "normalizing_method_used" in db:
                self.read_normalizing_method_used(db["normalizing_method_used"])
            if "SmoothedDict" in db:
                self.SmoothedDict = db["SmoothedDict"]
            if "baseline_corrected_dict" in db:
                self.baseline_corrected_dict = db["baseline_corrected_dict"]
            if "baseline_corrected_not_trimmed_dict" in db:
                self.baseline_corrected_not_trimmed_dict = db["baseline_corrected_not_trimmed_dict"]
            if "smoothing_method_comboBox" in db:
                self.ui.smoothing_method_comboBox.setCurrentText(db["smoothing_method_comboBox"])
            if "guess_method_cb" in db:
                self.ui.guess_method_cb.setCurrentText(db["guess_method_cb"])
            if "average_function" in db:
                self.ui.average_method_cb.setCurrentText(db["average_function"])
            if "dataset_type_cb" in db:
                self.ui.dataset_type_cb.setCurrentText(db["dataset_type_cb"])
            if "classes_lineEdit" in db:
                self.ui.classes_lineEdit.setText(db["classes_lineEdit"])
            if "test_data_ratio_spinBox" in db:
                self.ui.test_data_ratio_spinBox.setValue(db["test_data_ratio_spinBox"])
            if "n_lines_method" in db:
                self.ui.n_lines_detect_method_cb.setCurrentText(db["n_lines_method"])
            if "max_noise_level" in db:
                self.ui.max_noise_level_dsb.setValue(db["max_noise_level"])
            if "l_ratio_doubleSpinBox" in db:
                self.ui.l_ratio_doubleSpinBox.setValue(db["l_ratio_doubleSpinBox"])
            if "smooth_method" in db:
                self.read_smoothing_method_used(db["smooth_method"])
            if "window_length_spinBox" in db:
                self.ui.window_length_spinBox.setValue(db["window_length_spinBox"])
            if "smooth_polyorder_spinBox" in db:
                self.ui.smooth_polyorder_spinBox.setValue(db["smooth_polyorder_spinBox"])
            if "whittaker_lambda_spinBox" in db:
                self.ui.whittaker_lambda_spinBox.setValue(db["whittaker_lambda_spinBox"])
            if "kaiser_beta" in db:
                self.ui.kaiser_beta_doubleSpinBox.setValue(db["kaiser_beta"])
            if "EMD_noise_modes" in db:
                self.ui.emd_noise_modes_spinBox.setValue(db["EMD_noise_modes"])
            if "EEMD_trials" in db:
                self.ui.eemd_trials_spinBox.setValue(db["EEMD_trials"])
            if "sigma" in db:
                self.ui.sigma_spinBox.setValue(db["sigma"])
            if "lambda_spinBox" in db:
                self.ui.lambda_spinBox.setValue(db["lambda_spinBox"])
            if "p_doubleSpinBox" in db:
                self.ui.p_doubleSpinBox.setValue(db["p_doubleSpinBox"])
            if "eta" in db:
                self.ui.eta_doubleSpinBox.setValue(db["eta"])
            if "N_iterations" in db:
                self.ui.n_iterations_spinBox.setValue(db["N_iterations"])
            if "polynome_degree" in db:
                self.ui.polynome_degree_spinBox.setValue(db["polynome_degree"])
            if "grad" in db:
                self.ui.grad_doubleSpinBox.setValue(db["grad"])
            if "quantile" in db:
                self.ui.quantile_doubleSpinBox.setValue(db["quantile"])
            if "alpha_factor" in db:
                self.ui.alpha_factor_doubleSpinBox.setValue(db["alpha_factor"])
            if "peak_ratio" in db:
                self.ui.peak_ratio_doubleSpinBox.setValue(db["peak_ratio"])
            if "spline_degree" in db:
                self.ui.spline_degree_spinBox.setValue(db["spline_degree"])
            if "num_std" in db:
                self.ui.num_std_doubleSpinBox.setValue(db["num_std"])
            if "interp_half_window" in db:
                self.ui.interp_half_window_spinBox.setValue(db["interp_half_window"])
            if "fill_half_window" in db:
                self.ui.fill_half_window_spinBox.setValue(db["fill_half_window"])
            if "min_length" in db:
                self.ui.min_length_spinBox.setValue(db["min_length"])
            if "sections" in db:
                self.ui.sections_spinBox.setValue(db["sections"])
            if "scale" in db:
                self.ui.scale_doubleSpinBox.setValue(db["scale"])
            if "fraction" in db:
                self.ui.fraction_doubleSpinBox.setValue(db["fraction"])
            if "cost_function" in db:
                self.ui.cost_func_comboBox.setCurrentText(db["cost_function"])
            if "opt_method_oer" in db:
                self.ui.opt_method_oer_comboBox.setCurrentText(db["opt_method_oer"])
            if "fit_method" in db:
                self.ui.fit_opt_method_comboBox.setCurrentText(db["fit_method"])
            if 'EMSC_N_PCA' in db:
                self.ui.emsc_pca_n_spinBox.setValue(db['EMSC_N_PCA'])
            if "max_CCD_value" in db:
                self.ui.max_CCD_value_spinBox.setValue(db["max_CCD_value"])
            if "baseline_method_comboBox" in db:
                self.ui.baseline_correction_method_comboBox.setCurrentText(db["baseline_method_comboBox"])
            if "baseline_dict" in db:
                self.baseline_dict = db["baseline_dict"]
            if "baseline_method" in db:
                self.read_baseline_method_used(db["baseline_method"])
            if "data_style" in db:
                self.data_style.clear()
                self.data_style = db["data_style"].copy()
                self.data_style_button_style_sheet(self.data_style['color'].name())
            if "data_curve_checked" in db:
                self.ui.data_checkBox.setChecked(db["data_curve_checked"])
            if "sum_style" in db:
                self.sum_style.clear()
                self.sum_style = db["sum_style"].copy()
                self.sum_style_button_style_sheet(self.sum_style['color'].name())
            if "sigma3_style" in db:
                self.sigma3_style.clear()
                self.sigma3_style = db["sigma3_style"].copy()
                self.sigma3_style_button_style_sheet(self.sigma3_style['color'].name())
                pen, brush = curve_pen_brush_by_style(self.sigma3_style)
                self.fill.setPen(pen)
                self.fill.setBrush(brush)
            if "sum_curve_checked" in db:
                self.ui.sum_checkBox.setChecked(db["sum_curve_checked"])
            if "sigma3_checked" in db:
                self.ui.sigma3_checkBox.setChecked(db["sigma3_checked"])
            if "residual_style" in db:
                self.residual_style.clear()
                self.residual_style = db["residual_style"].copy()
                self.residual_style_button_style_sheet(self.residual_style['color'].name())
            if "residual_curve_checked" in db:
                self.ui.residual_checkBox.setChecked(db["residual_curve_checked"])
            if "interval_checkBox_checked" in db:
                self.ui.interval_checkBox.setChecked(db["interval_checkBox_checked"])
            if "use_fit_intervals" in db:
                self.ui.intervals_gb.setChecked(db["use_fit_intervals"])
            if 'max_dx_guess' in db:
                self.ui.max_dx_dsb.setValue(db['max_dx_guess'])
            if 'latest_stat_result' in db:
                self.latest_stat_result = db['latest_stat_result']

            if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
                self.ui.statusBar.showMessage('Import canceled by user.')
                self._clear_all_parameters()
                return

    def read_normalizing_method_used(self, method: str) -> None:
        self.normalization_method = method
        if len(self.NormalizedDict) > 0:
            text_for_title = "<span style=\"font-family: AbletonSans; color:" \
                             + self.theme_colors['plotText'] \
                             + ";font-size:14pt\">Normalized plots. Method " \
                             + method + "</span>"
            self.ui.normalize_plot_widget.setTitle(text_for_title)

    def read_smoothing_method_used(self, method: str) -> None:
        self.smooth_method = method
        if len(self.SmoothedDict) > 0:
            text_for_title = "<span style=\"font-family: AbletonSans; color:" \
                             + self.theme_colors['plotText'] \
                             + ";font-size:14pt\">Smoothed plots. Method " \
                             + method + "</span>"
            self.ui.smooth_plot_widget.setTitle(text_for_title)

    def read_baseline_method_used(self, method: str) -> None:
        self.baseline_method = method
        if len(self.baseline_dict) > 0:
            text_for_title = "<span style=\"font-family: AbletonSans; color:" \
                             + self.theme_colors['plotText'] \
                             + ";font-size:14pt\">Baseline corrected plots. Method " \
                             + method + "</span>"
            self.ui.baseline_plot_widget.setTitle(text_for_title)

    def update_recent_list(self) -> None:
        self.recent_menu.clear()
        try:
            with Path('recent-files.txt').open() as file:
                lines = [line.rstrip() for line in file]

            lines = list(dict.fromkeys(lines))
            for line in reversed(lines):
                action = self.recent_menu.addAction(line)
                action.triggered.connect(lambda checked=None, line=line: self.action_open_recent(path=line))
        except Exception:
            self.recent_menu.setDisabled(True)
            raise

    def set_modified(self, b: bool = True) -> None:
        if not b:
            b = False
        else:
            b = True
        self.modified = b
        if b:
            self.ui.unsavedBtn.show()
        else:
            self.ui.unsavedBtn.hide()

    def set_smooth_polyorder_bound(self, a0: int) -> None:
        self.ui.smooth_polyorder_spinBox.setMaximum(a0 - 1)
        self.set_modified()

    def set_baseline_parameters_disabled(self, value: str) -> None:
        self.set_modified()
        self.ui.alpha_factor_doubleSpinBox.setVisible(False)
        self.ui.label_alpha_factor.setVisible(False)
        self.ui.cost_func_comboBox.setVisible(False)
        self.ui.label_cost_func.setVisible(False)
        self.ui.eta_doubleSpinBox.setVisible(False)
        self.ui.label_eta.setVisible(False)
        self.ui.fill_half_window_spinBox.setVisible(False)
        self.ui.label_fill_half_window.setVisible(False)
        self.ui.fraction_doubleSpinBox.setVisible(False)
        self.ui.label_fraction.setVisible(False)
        self.ui.grad_doubleSpinBox.setVisible(False)
        self.ui.label_grad.setVisible(False)
        self.ui.interp_half_window_spinBox.setVisible(False)
        self.ui.label_interp_half_window.setVisible(False)
        self.ui.lambda_spinBox.setVisible(False)
        self.ui.label_lambda.setVisible(False)
        self.ui.min_length_spinBox.setVisible(False)
        self.ui.label_min_length.setVisible(False)
        self.ui.n_iterations_spinBox.setVisible(False)
        self.ui.label_n_iterations.setVisible(False)
        self.ui.num_std_doubleSpinBox.setVisible(False)
        self.ui.label_num_std.setVisible(False)
        self.ui.opt_method_oer_comboBox.setVisible(False)
        self.ui.label_opt_method_oer.setVisible(False)
        self.ui.p_doubleSpinBox.setVisible(False)
        self.ui.label_p.setVisible(False)
        self.ui.polynome_degree_spinBox.setVisible(False)
        self.ui.polynome_degree_label.setVisible(False)
        self.ui.quantile_doubleSpinBox.setVisible(False)
        self.ui.label_quantile.setVisible(False)
        self.ui.sections_spinBox.setVisible(False)
        self.ui.label_sections.setVisible(False)
        self.ui.scale_doubleSpinBox.setVisible(False)
        self.ui.label_scale.setVisible(False)
        self.ui.spline_degree_spinBox.setVisible(False)
        self.ui.label_spline_degree.setVisible(False)
        self.ui.peak_ratio_doubleSpinBox.setVisible(False)
        self.ui.label_peak_ratio.setVisible(False)

        match value:
            case 'Poly':
                self.ui.polynome_degree_spinBox.setVisible(True)
                self.ui.polynome_degree_label.setVisible(True)
            case 'ModPoly' | 'iModPoly' | 'iModPoly+':
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.polynome_degree_spinBox.setVisible(True)
                self.ui.polynome_degree_label.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
            case 'Penalized poly':
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.polynome_degree_spinBox.setVisible(True)
                self.ui.polynome_degree_label.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
                self.ui.alpha_factor_doubleSpinBox.setVisible(True)
                self.ui.label_alpha_factor.setVisible(True)
                self.ui.cost_func_comboBox.setVisible(True)
                self.ui.label_cost_func.setVisible(True)
            case 'LOESS':
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.polynome_degree_spinBox.setVisible(True)
                self.ui.polynome_degree_label.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
                self.ui.fraction_doubleSpinBox.setVisible(True)
                self.ui.label_fraction.setVisible(True)
                self.ui.scale_doubleSpinBox.setVisible(True)
                self.ui.label_scale.setVisible(True)
            case 'Quantile regression':
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.polynome_degree_spinBox.setVisible(True)
                self.ui.polynome_degree_label.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
                self.ui.quantile_doubleSpinBox.setVisible(True)
                self.ui.label_quantile.setVisible(True)
            case 'Goldindec':
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.polynome_degree_spinBox.setVisible(True)
                self.ui.polynome_degree_label.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
                self.ui.alpha_factor_doubleSpinBox.setVisible(True)
                self.ui.label_alpha_factor.setVisible(True)
                self.ui.cost_func_comboBox.setVisible(True)
                self.ui.label_cost_func.setVisible(True)
                self.ui.peak_ratio_doubleSpinBox.setVisible(True)
                self.ui.label_peak_ratio.setVisible(True)
            case 'AsLS' | 'arPLS' | 'airPLS' | 'iAsLS' | 'psaLSA' | 'DerPSALSA' | 'MPLS' | 'iarPLS' | 'asPLS':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.label_lambda.setVisible(True)
                self.ui.p_doubleSpinBox.setVisible(True)
                self.ui.label_p.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
            case 'drPLS':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.label_lambda.setVisible(True)
                self.ui.p_doubleSpinBox.setVisible(True)
                self.ui.label_p.setVisible(True)
                self.ui.eta_doubleSpinBox.setVisible(True)
                self.ui.label_eta.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
            case 'iMor' | 'MorMol' | 'AMorMol' | 'JBCD':
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
            case 'MPSpline':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.label_lambda.setVisible(True)
                self.ui.p_doubleSpinBox.setVisible(True)
                self.ui.label_p.setVisible(True)
                self.ui.spline_degree_spinBox.setVisible(True)
                self.ui.label_spline_degree.setVisible(True)
            case 'Mixture Model':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.label_lambda.setVisible(True)
                self.ui.p_doubleSpinBox.setVisible(True)
                self.ui.label_p.setVisible(True)
                self.ui.spline_degree_spinBox.setVisible(True)
                self.ui.label_spline_degree.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
            case 'IRSQR':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.label_lambda.setVisible(True)
                self.ui.quantile_doubleSpinBox.setVisible(True)
                self.ui.label_quantile.setVisible(True)
                self.ui.spline_degree_spinBox.setVisible(True)
                self.ui.label_spline_degree.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
            case 'Corner-Cutting':
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
            case 'RIA':
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
            case 'Dietrich':
                self.ui.num_std_doubleSpinBox.setVisible(True)
                self.ui.label_num_std.setVisible(True)
                self.ui.polynome_degree_spinBox.setVisible(True)
                self.ui.polynome_degree_label.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.label_grad.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
                self.ui.interp_half_window_spinBox.setVisible(True)
                self.ui.label_interp_half_window.setVisible(True)
                self.ui.min_length_spinBox.setVisible(True)
                self.ui.label_min_length.setVisible(True)
            case 'Golotvin':
                self.ui.num_std_doubleSpinBox.setVisible(True)
                self.ui.label_num_std.setVisible(True)
                self.ui.interp_half_window_spinBox.setVisible(True)
                self.ui.label_interp_half_window.setVisible(True)
                self.ui.min_length_spinBox.setVisible(True)
                self.ui.label_min_length.setVisible(True)
                self.ui.sections_spinBox.setVisible(True)
                self.ui.label_sections.setVisible(True)
            case 'Std Distribution':
                self.ui.num_std_doubleSpinBox.setVisible(True)
                self.ui.label_num_std.setVisible(True)
                self.ui.interp_half_window_spinBox.setVisible(True)
                self.ui.label_interp_half_window.setVisible(True)
                self.ui.fill_half_window_spinBox.setVisible(True)
                self.ui.label_fill_half_window.setVisible(True)
            case 'FastChrom':
                self.ui.interp_half_window_spinBox.setVisible(True)
                self.ui.label_interp_half_window.setVisible(True)
                self.ui.min_length_spinBox.setVisible(True)
                self.ui.label_min_length.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.label_n_iterations.setVisible(True)
            case 'FABC':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.label_lambda.setVisible(True)
                self.ui.num_std_doubleSpinBox.setVisible(True)
                self.ui.label_num_std.setVisible(True)
                self.ui.min_length_spinBox.setVisible(True)
                self.ui.label_min_length.setVisible(True)
            case 'OER' | 'Adaptive MinMax':
                self.ui.opt_method_oer_comboBox.setVisible(True)
                self.ui.label_opt_method_oer.setVisible(True)

    def set_smooth_parameters_disabled(self, value: str) -> None:
        self.set_modified()
        self.ui.window_length_spinBox.setVisible(False)
        self.ui.window_length_label.setVisible(False)
        self.ui.smooth_polyorder_spinBox.setVisible(False)
        self.ui.smooth_polyorder_label.setVisible(False)
        self.ui.whittaker_lambda_spinBox.setVisible(False)
        self.ui.whittaker_lambda_label.setVisible(False)
        self.ui.kaiser_beta_doubleSpinBox.setVisible(False)
        self.ui.kaiser_beta_label.setVisible(False)
        self.ui.emd_noise_modes_spinBox.setVisible(False)
        self.ui.emd_noise_modes_label.setVisible(False)
        self.ui.eemd_trials_spinBox.setVisible(False)
        self.ui.eemd_trials_label.setVisible(False)
        self.ui.sigma_spinBox.setVisible(False)
        self.ui.sigma_label.setVisible(False)
        match value:
            case 'Savitsky-Golay filter':
                self.ui.window_length_spinBox.setVisible(True)
                self.ui.window_length_label.setVisible(True)
                self.ui.smooth_polyorder_spinBox.setVisible(True)
                self.ui.smooth_polyorder_label.setVisible(True)
            case 'MLESG':
                self.ui.sigma_spinBox.setVisible(True)
                self.ui.sigma_label.setVisible(True)
            case 'Whittaker smoother':
                self.ui.whittaker_lambda_spinBox.setVisible(True)
                self.ui.whittaker_lambda_label.setVisible(True)
            case 'Flat window' | 'hanning' | 'hamming' | 'bartlett' | 'blackman' | 'Median filter' | 'Wiener filter':
                self.ui.window_length_spinBox.setVisible(True)
                self.ui.window_length_label.setVisible(True)
            case 'kaiser':
                self.ui.window_length_spinBox.setVisible(True)
                self.ui.window_length_label.setVisible(True)
                self.ui.kaiser_beta_doubleSpinBox.setVisible(True)
                self.ui.kaiser_beta_label.setVisible(True)
            case 'EMD':
                self.ui.emd_noise_modes_spinBox.setVisible(True)
                self.ui.emd_noise_modes_label.setVisible(True)
            case 'EEMD' | 'CEEMDAN':
                self.ui.emd_noise_modes_spinBox.setVisible(True)
                self.ui.emd_noise_modes_label.setVisible(True)
                self.ui.eemd_trials_spinBox.setVisible(True)
                self.ui.eemd_trials_label.setVisible(True)

    @asyncSlot()
    async def action_import_fit_template(self):
        username = environ.get('USERNAME')
        fd = QFileDialog()
        file_path = fd.getOpenFileName(self, 'Open fit template file',
                                       '/users/' + str(username) + '/Documents/RS-tool', "ZIP (*.zip)")
        if not file_path[0]:
            return
        path = file_path[0]
        self.ui.statusBar.showMessage('Reading data file...')
        self.close_progress_bar()
        self._open_progress_bar()
        self._open_progress_dialog("Opening template...", "Cancel")
        self.beforeTime = datetime.now()
        with ZipFile(path) as archive:
            directory = Path(path).parent
            archive.extractall(directory)
        if not Path(str(directory) + '/data.dat').exists():
            Path(str(directory) + '/data.dat').unlink()
            Path(str(directory) + '/data.dir').unlink()
            Path(str(directory) + '/data.bak').unlink()
            return
        file_name = str(directory) + '/data'
        with shelve_open(file_name, 'r') as db:
            if "DeconvLinesTableDF" in db:
                df = db["DeconvLinesTableDF"]
                self.ui.deconv_lines_table.model().set_dataframe(df)
            if "DeconvLinesTableChecked" in db:
                checked = db["DeconvLinesTableChecked"]
                self.ui.deconv_lines_table.model().set_checked(checked)
            if "DeconvParamsTableDF" in db:
                df = db["DeconvParamsTableDF"]
                self.ui.fit_params_table.model().set_dataframe(df)
            if "intervals_table_df" in db:
                df = db["intervals_table_df"]
                self.ui.fit_intervals_table_view.model().set_dataframe(df)
        Path(str(directory) + '/data.dat').unlink()
        Path(str(directory) + '/data.dir').unlink()
        Path(str(directory) + '/data.bak').unlink()
        self.currentProgress.setMaximum(1)
        self.currentProgress.setValue(1)
        self.close_progress_bar()
        seconds = round((datetime.now() - self.beforeTime).total_seconds())
        self.set_modified(False)
        self.ui.statusBar.showMessage('Fit tenplate imported for ' + str(seconds) + ' sec.', 5000)
        if self.ui.fit_params_table.model().rowCount() != 0 \
                and self.ui.deconv_lines_table.model().rowCount() != 0:
            await self.draw_all_curves()

    @asyncSlot
    async def action_export_fit_template(self):
        if self.ui.deconv_lines_table.model().rowCount() == 0 \
                and self.ui.fit_params_table.model().rowCount() == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Fit template is empty.")
            msg.setWindowTitle("Export failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        username = environ.get('USERNAME')
        fd = QFileDialog()
        file_path = fd.getSaveFileName(self, 'Save fit template file',
                                       '/users/' + str(username) + '/Documents/RS-tool', "ZIP (*.zip)")
        if not file_path[0]:
            return
        self.ui.statusBar.showMessage('Saving file...')
        self.close_progress_bar()
        self._open_progress_bar()
        filename = file_path[0]
        with shelve_open(filename, 'n') as db:
            db["DeconvLinesTableDF"] = self.ui.deconv_lines_table.model().dataframe()
            db["DeconvParamsTableDF"] = self.ui.fit_params_table.model().dataframe()
            db["intervals_table_df"] = self.ui.fit_intervals_table_view.model().dataframe()
            db["DeconvLinesTableChecked"] = self.ui.deconv_lines_table.model().checked()
        zf = ZipFile(filename, "w", ZIP_DEFLATED, compresslevel=9)
        zf.write(filename + '.dat', "data.dat")
        zf.write(filename + '.dir', "data.dir")
        zf.write(filename + '.bak', "data.bak")
        self.ui.statusBar.showMessage('File saved. ' + filename, 10000)
        Path(filename + '.dat').unlink()
        Path(filename + '.dir').unlink()
        Path(filename + '.bak').unlink()
        self.close_progress_bar()

    # endregion

    # region RS

    def action_help(self) -> None:
        startfile('help\index.htm')

    def color_dialog(self, initial: QColor) -> QColorDialog:
        color_dialog = QColorDialog(initial)
        color_dialog.setCustomColor(0, QColor(self.theme_colors['primaryColor']))
        color_dialog.setCustomColor(1, QColor(self.theme_colors['primaryDarker']))
        color_dialog.setCustomColor(2, QColor(self.theme_colors['primaryDarkColor']))
        color_dialog.setCustomColor(3, QColor(self.theme_colors['secondaryColor']))
        color_dialog.setCustomColor(4, QColor(self.theme_colors['secondaryLightColor']))
        color_dialog.setCustomColor(5, QColor(self.theme_colors['secondaryDarkColor']))
        return color_dialog

    def show_error(self, err) -> None:
        critical(err)
        tb = format_exc()
        error(tb)
        self.executor_stop()
        if self.currentProgress:
            self.currentProgress.setMaximum(1)
            self.currentProgress.setValue(1)
        self.close_progress_bar()
        show_error(type(err), err, str(tb))

    # def init_cuda(self):
    #     mlesg = open("modules/mlesg_not_used.cu").read()
    #     mod = SourceModule(mlesg)
    #     self.func_arrange = mod.get_function('arrange')
    #     self.func_le_mle = mod.get_function('le_mle')

    def get_curve_plot_data_item(self, n_array: np.ndarray, group_number: str = 0, color: QColor = None, name: str = '',
                                 style: Qt.PenStyle = Qt.PenStyle.SolidLine, width: int = 2) -> PlotDataItem:
        curve = PlotDataItem(skipFiniteCheck=True, name=name)
        curve.setData(x=n_array[:, 0], y=n_array[:, 1], skipFiniteCheck=True)
        if color is None:
            color = self.get_color_by_group_number(group_number)
        curve.setPen(color, width=width, style=style)
        return curve

    @asyncSlot()
    async def update_all_plots(self) -> None:
        await self.update_plot_item(self.ImportedArray.items())
        await self.update_plot_item(self.ConvertedDict.items(), 1)
        await self.update_plot_item(self.CuttedFirstDict.items(), 2)
        await self.update_plot_item(self.NormalizedDict.items(), 3)
        await self.update_plot_item(self.SmoothedDict.items(), 4)
        await self.update_plot_item(self.baseline_corrected_dict.items(), 5)
        await self.update_plot_item(self.averaged_dict.items(), 6)

    @asyncSlot()
    async def update_plot_item(self, items: dict[str, np.ndarray], plot_item_id: int = 0) -> None:
        self.clear_plots_before_update(plot_item_id)
        if plot_item_id == 6:
            group_styles = [x for x in self.ui.GroupsTable.model().column_data(1)]
            i = 0
            for key, arr in items:
                if len(arr) != 0:
                    self.add_lines(np.array([arr[:, 0]]), np.array([arr[:, 1]]), group_styles[i], key, plot_item_id)
                i += 1
        else:
            arrays = self.get_arrays_by_group(items)
            self.combine_arrays_by_groups(arrays, plot_item_id)
        self.after_updating_data(plot_item_id)

    def clear_plots_before_update(self, plot_item_id: int) -> None:
        match plot_item_id:
            case 0:
                self.input_plot_widget_plot_item.clear()
                if self.curveOneCutPlot:
                    self.input_plot_widget_plot_item.removeItem(self.curveOneCutPlot)
                if self.curveDespikedHistory:
                    self.input_plot_widget_plot_item.removeItem(self.curveDespikedHistory)
            case 1:
                self.converted_cm_widget_plot_item.clear()
                if self.curveOneConvertPlot:
                    self.converted_cm_widget_plot_item.removeItem(self.curveOneConvertPlot)
            case 2:
                self.cut_cm_plotItem.clear()
                if self.curveOneCutPlot:
                    self.cut_cm_plotItem.removeItem(self.curveOneCutPlot)
            case 3:
                self.normalize_plotItem.clear()
                if self.curveOneNormalPlot:
                    self.normalize_plotItem.removeItem(self.curveOneNormalPlot)
            case 4:
                self.smooth_plotItem.clear()
                if self.curveOneSmoothPlot:
                    self.smooth_plotItem.removeItem(self.curveOneSmoothPlot)
            case 5:
                self.baseline_corrected_plotItem.clear()
                if self.curve_one_baseline_plot:
                    self.baseline_corrected_plotItem.removeItem(self.curve_one_baseline_plot)
                if self.curveBaseline:
                    self.baseline_corrected_plotItem.removeItem(self.curveBaseline)
            case 6:
                self.averaged_plotItem.clear()

    def after_updating_data(self, plot_item_id: int) -> None:
        match plot_item_id:
            case 0:
                self.input_plot_widget_plot_item.getViewBox().updateAutoRange()
                self.input_plot_widget_plot_item.updateParamList()
                self.input_plot_widget_plot_item.recomputeAverages()
            case 1:
                self.converted_cm_widget_plot_item.addItem(self.linearRegionCmConverted)
                self.converted_cm_widget_plot_item.getViewBox().updateAutoRange()
                self.converted_cm_widget_plot_item.updateParamList()
                self.converted_cm_widget_plot_item.recomputeAverages()
            case 2:
                self.cut_cm_plotItem.getViewBox().updateAutoRange()
                self.cut_cm_plotItem.updateParamList()
                self.cut_cm_plotItem.recomputeAverages()
            case 3:
                self.normalize_plotItem.getViewBox().updateAutoRange()
                self.normalize_plotItem.updateParamList()
                self.normalize_plotItem.recomputeAverages()
            case 4:
                self.smooth_plotItem.getViewBox().updateAutoRange()
                self.smooth_plotItem.updateParamList()
                self.smooth_plotItem.recomputeAverages()
            case 5:
                self.baseline_corrected_plotItem.addItem(self.linearRegionBaseline)
                self.baseline_corrected_plotItem.getViewBox().updateAutoRange()
                self.baseline_corrected_plotItem.updateParamList()
                self.baseline_corrected_plotItem.recomputeAverages()
            case 6:
                self.linearRegionDeconv.setVisible(self.ui.interval_checkBox.isChecked())
                self.averaged_plotItem.getViewBox().updateAutoRange()
                self.averaged_plotItem.updateParamList()
                self.averaged_plotItem.recomputeAverages()

    def add_lines(self, x: np.ndarray, y: np.ndarray, style: dict, _group: int, plot_item_id: int) -> None:
        curve = MultiLine(x, y, style, _group)
        if plot_item_id == 0:
            self.input_plot_widget_plot_item.addItem(curve, kargs={'ignoreBounds': False})
        elif plot_item_id == 1:
            self.converted_cm_widget_plot_item.addItem(curve)
        elif plot_item_id == 2:
            self.cut_cm_plotItem.addItem(curve)
        elif plot_item_id == 3:
            self.normalize_plotItem.addItem(curve)
        elif plot_item_id == 4:
            self.smooth_plotItem.addItem(curve)
        elif plot_item_id == 5:
            self.baseline_corrected_plotItem.addItem(curve)
        elif plot_item_id == 6:
            self.averaged_plotItem.addItem(curve)

    def combine_arrays_by_groups(self, arrays: list[tuple[dict, list]], plot_item_id: int) -> None:
        for idx, item in enumerate(arrays):
            style = item[0]
            xy_arrays = item[1]
            if len(xy_arrays) == 0:
                continue
            if plot_item_id == 0:
                self.process_input_plots_by_different_ranges(style, xy_arrays, idx, plot_item_id)
            else:
                self.process_plots_by_different_ranges(style, xy_arrays, idx, plot_item_id)

    def process_plots_by_different_ranges(self, style: dict, xy_arrays: list, group_idx: int,
                                          plot_item_id: int) -> None:
        x_axes = []
        y_axes = []

        for j in xy_arrays:
            x_axes.append(j[:, 0])
            y_axes.append(j[:, 1])
        x_arrays = np.array(x_axes)
        y_arrays = np.array(y_axes)
        self.add_lines(x_arrays, y_arrays, style, group_idx, plot_item_id)

    def process_input_plots_by_different_ranges(self, style: dict, xy_arrays: list, group_idx: int,
                                                plot_item_id: int) -> None:
        arrays_by_ranges = dict()
        for j in xy_arrays:
            array_len = j.shape[0]
            x_axis = j[:, 0]
            y_axis = j[:, 1]
            if array_len not in arrays_by_ranges:
                arrays_by_ranges[array_len] = ([x_axis], [y_axis])
            elif array_len in arrays_by_ranges:
                xy_axes_tuple = arrays_by_ranges[array_len]
                xy_axes_tuple[0].append(x_axis)
                xy_axes_tuple[1].append(y_axis)
        for _, xy_axes_list in arrays_by_ranges.items():
            x_arrays = np.array(xy_axes_list[0])
            y_arrays = np.array(xy_axes_list[1])
            self.add_lines(x_arrays, y_arrays, style, group_idx, plot_item_id)

    def get_arrays_by_group(self, items: dict[str, np.ndarray]) -> list[tuple[dict, list]]:
        styles = self.ui.GroupsTable.model().column_data(1)
        std_style = {'color': QColor(self.theme_colors['secondaryColor']),
                     'style': Qt.PenStyle.SolidLine,
                     'width': 1.0,
                     'fill': False,
                     'use_line_color': True,
                     'fill_color': QColor().fromRgb(random_rgb()),
                     'fill_opacity': 0.0}
        arrays = [(std_style, [], 1)]
        idx = 0
        for style in styles:
            arrays.append((style, []))
            idx += 1

        for i in items:
            name = i[0]
            arr = i[1]
            group_number = int(self.ui.input_table.model().get_group_by_name(name))
            if group_number > len(styles):
                group_number = 0  # in case when have group number, but there is no corresponding group actually
            arrays[group_number][1].append(arr)
        return arrays

    def update_averaged_dict(self) -> None:
        groups_count = self.ui.GroupsTable.model().rowCount()
        self.averaged_dict.clear()
        method = self.ui.average_method_cb.currentText()
        for i in range(groups_count):
            group = i + 1
            names = self.ui.input_table.model().names_of_group(group)
            if len(names) == 0:
                continue
            arrays_list = [self.baseline_corrected_dict[x] for x in names]
            arrays_list_av = get_average_spectrum(arrays_list, method)
            self.averaged_dict[group] = arrays_list_av

    def _open_progress_dialog(self, text: str, buttons: str, maximum: int = 0) -> None:
        self.currentProgress = QProgressDialog(text, buttons, 0, maximum)
        self.currentProgress.setWindowFlags(Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowTitleHint)
        self.currentProgress.setWindowModality(Qt.WindowModality.WindowModal)
        self.currentProgress.setWindowTitle(' ')
        self.currentProgress.open()
        # self.currentProgress = QProgressDialog(text, buttons, 0, maximum, self)
        # self.currentProgress.open()
        # self.currentProgress.setWindowModality(Qt.WindowModality.WindowModal)
        # self.currentProgress.setWindowTitle(' ')
        # self.currentProgress.setWindowFlags(Qt.WindowType.Dialog)
        cancel_button = self.currentProgress.findChild(QPushButton)
        environ['CANCEL'] = '0'
        cancel_button.clicked.connect(self.executor_stop)

    def executor_stop(self) -> None:
        if self.current_executor:
            event = Event()
            for f in self.current_futures:
                if not f.done():
                    f.cancel()
            event.set()
            environ['CANCEL'] = '1'
            self.current_executor.shutdown(cancel_futures=True, wait=False)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Operation canceled by user ')

    def progress_indicator(self, _=None) -> None:
        current_value = self.progressBar.value() + 1
        self.progressBar.setValue(current_value)
        self.currentProgress.setValue(current_value)
        self.taskbar_progress.setValue(current_value)
        self.taskbar_progress.show()

    def _open_progress_bar(self, min_value: int = 0, max_value: int = 0) -> None:
        self.progressBar = QProgressBar()
        self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
        self.progressBar.setRange(min_value, max_value)
        self.taskbar_button = QWinTaskbarButton()
        self.taskbar_progress = self.taskbar_button.progress()
        self.taskbar_progress.setRange(min_value, max_value)
        self.taskbar_button.setWindow(self.windowHandle())
        self.taskbar_progress.show()

    def close_progress_bar(self) -> None:
        if self.currentProgress is not None:
            self.currentProgress.close()
        if self.progressBar is not None:
            self.statusBar().removeWidget(self.progressBar)
        if self.taskbar_progress is not None:
            self.taskbar_progress.hide()
        # winsound.MessageBeep()

    def set_buttons_ability(self) -> None:
        self.action_despike.setDisabled(len(self.ImportedArray) == 0)
        self.action_interpolate.setDisabled(len(self.ImportedArray) < 2)
        self.action_convert.setDisabled(len(self.ImportedArray) == 0)
        self.action_cut.setDisabled(len(self.ConvertedDict) == 0)
        self.action_normalize.setDisabled(len(self.CuttedFirstDict) == 0)
        self.action_smooth.setDisabled(len(self.NormalizedDict) == 0)
        self.action_baseline_correction.setDisabled(len(self.SmoothedDict) == 0)
        self.action_trim.setDisabled(len(self.baseline_corrected_not_trimmed_dict) == 0)
        self.action_average.setDisabled(len(self.baseline_corrected_not_trimmed_dict) == 0)

    def set_timer_memory_update(self) -> None:
        try:
            string_selected_files = ''
            n_selected = len(self.ui.input_table.selectionModel().selectedIndexes())
            if n_selected > 0:
                string_selected_files = str(n_selected) + ' selected of '
            string_n = ''
            n_spectrum = len(self.ImportedArray)
            if n_spectrum > 0:
                string_n = str(n_spectrum) + ' files. '
            string_mem = str(round(get_memory_used())) + ' Mb used'
            usage_string = string_selected_files + string_n + string_mem
            self.memory_usage_label.setText(usage_string)
        except KeyboardInterrupt:
            pass

    def set_cpu_load(self) -> None:
        cpu_perc = int(cpu_percent())
        self.ui.cpuLoadBar.setValue(cpu_perc)

    def auto_save(self) -> None:
        if self.modified and self.project_path and self.project_path != '':
            self.save_by_shelve(self.project_path)

    def disable_buttons(self, b: bool) -> None:
        self.set_buttons_ability()
        self.ui.EditBtn.setDisabled(b)
        self.ui.FileBtn.setDisabled(b)
        self.ui.ProcessBtn.setDisabled(b)
        self.ui.gt_add_Btn.setDisabled(b)
        self.ui.gt_dlt_Btn.setDisabled(b)
        self.ui.GroupsTable.setDisabled(b)
        self.ui.input_table.setDisabled(b)
        self.ui.all_control_button.setDisabled(b)
        self.ui.by_one_control_button.setDisabled(b)
        self.ui.by_group_control_button.setDisabled(b)
        self.ui.updateRangebtn.setDisabled(b)
        self.ui.updateTrimRangebtn.setDisabled(b)

    def mousePressEvent(self, event) -> None:
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

    def update_icons(self) -> None:
        if 'Light' in self.theme_bckgrnd:
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon("material/resources/source/chevron-down_black.svg"))
            self.ui.minimizeAppBtn.setIcon(QIcon("material/resources/source/minus_black.svg"))
            self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/chevron-left_black.svg"))
            self.ui.gt_add_Btn.setIcon(QIcon("material/resources/source/plus_black.svg"))
            self.ui.gt_dlt_Btn.setIcon(QIcon("material/resources/source/minus_black.svg"))
        else:
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon("material/resources/source/chevron-down.svg"))
            self.ui.minimizeAppBtn.setIcon(QIcon("material/resources/source/minus.svg"))
            self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/chevron-left.svg"))
            self.ui.gt_add_Btn.setIcon(QIcon("material/resources/source/plus.svg"))
            self.ui.gt_dlt_Btn.setIcon(QIcon("material/resources/source/minus.svg"))

    # endregion

    # region INTERPOLATION

    @asyncSlot()
    async def interpolate(self) -> None:

        self.beforeTime = datetime.now()
        if len(self.ImportedArray) < 2:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Нужно хотя бы 2 спектра с разными диапазонами для интерполяции.")
            msg.setWindowTitle("Interpolation error")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        result = self.check_ranges()
        different_shapes, _, _ = self.check_arrays_shape()
        if len(result) <= 1 and not different_shapes and not self.is_production_project:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Все спектры в одинаковых диапазонах длин волн")
            msg.setWindowTitle("Interpolation didn't started")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        elif len(result) <= 1 and different_shapes and not self.is_production_project:
            await self.interpolate_shapes()
            return

        dialog = DialogListBox(title='RS-tool', checked_ranges=result)
        dialog_code = dialog.exec()

        if dialog_code == 0:
            return
        range_nm = dialog.get_result()
        filenames = self.get_filenames_of_this_range(range_nm)
        ref_file = self.get_ref_file(range_nm)
        interpolated = None
        try:
            interpolated = await self.get_interpolated(filenames, ref_file)
        except Exception as err:
            self.show_error(err)

        if interpolated:
            command = CommandUpdateInterpolated(self, interpolated, "Interpolate files")
            self.undoStack.push(command)
        else:
            await self.interpolate_shapes()

    async def interpolate_shapes(self) -> None:
        different_shapes, ref_array, filenames = self.check_arrays_shape()
        if not different_shapes:
            return
        interpolated = None
        try:
            interpolated = await self.get_interpolated(filenames, ref_array)
        except Exception as err:
            self.show_error(err)
        if interpolated:
            command = CommandUpdateInterpolated(self, interpolated, "Interpolate files")
            self.undoStack.push(command)

    async def get_interpolated(self, filenames: list[str], ref_file: np.ndarray) -> list[tuple[str, np.ndarray]]:
        self.ui.statusBar.showMessage('Interpolating...')
        self.close_progress_bar()
        n_files = len(filenames)
        self._open_progress_bar(max_value=n_files)
        self._open_progress_dialog("Interpolating...", "Cancel", maximum=n_files)
        executor = ThreadPoolExecutor()
        if n_files >= 10_000:
            executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(executor, interpolate, self.ImportedArray[i], i, ref_file)
                                    for i in filenames]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            interpolated = await gather(*self.current_futures)

        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        self.close_progress_bar()
        return interpolated

    def get_filenames_of_this_range(self, target_range_nm: tuple[int, int]) -> list[str]:
        filenames = []
        for i in self.ImportedArray:
            spectrum = self.ImportedArray[i]
            max_nm = spectrum.max(axis=0)[0]
            min_nm = spectrum.min(axis=0)[0]
            file_range = (min_nm, max_nm)
            if file_range != target_range_nm:
                filenames.append(i)
        return filenames

    def get_ref_file(self, target_range_nm: tuple[int, int]) -> np.ndarray:
        for i in self.ImportedArray:
            spectrum = self.ImportedArray[i]
            max_nm = spectrum.max(axis=0)[0]
            min_nm = spectrum.min(axis=0)[0]
            new_range = (min_nm, max_nm)
            if new_range == target_range_nm:
                return spectrum

    def check_ranges(self) -> list[int, int]:
        ranges = []
        for i in self.ImportedArray.items():
            arr = i[1]
            max_nm = arr.max(axis=0)[0]
            min_nm = arr.min(axis=0)[0]
            new_range = (min_nm, max_nm)
            if new_range not in ranges:
                ranges.append(new_range)

        return ranges

    def check_arrays_shape(self) -> tuple[bool, np.ndarray, list[str]]:
        shapes = dict()
        for key, arr in self.ImportedArray.items():
            shapes[key] = arr.shape[0]
        list_shapes = list(shapes.values())
        min_shape = np.min(list_shapes)
        max_shape = np.max(list_shapes)
        different_shapes = min_shape != max_shape
        ref_array = None
        filenames = []
        if different_shapes:
            counts = np.bincount(list_shapes)
            most_shape = np.argmax(counts)
            keys = [k for k, v in shapes.items() if v == most_shape]
            key = keys[0]
            ref_array = self.ImportedArray[key]
            filenames = [k for k, v in shapes.items() if v != most_shape]
        return different_shapes, ref_array, filenames

    def check_array_len(self) -> list[int]:
        lens = []
        for _, arr in self.ImportedArray.items():
            x_len = np.shape(arr)[0]
            if x_len not in lens:
                lens.append(x_len)
        return lens

    # endregion

    # region DESPIKE

    @asyncSlot()
    async def despike(self):
        print('despike')
        try:
            await self._do_despike()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def _do_despike(self) -> None:
        if len(self.ImportedArray) <= 0:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Icon.Warning)
            message_box.setText("Import spectra first")
            message_box.setWindowTitle("Despike error")
            message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            message_box.exec()
            return
        laser_wavelength = self.ui.laser_wl_spinbox.value()
        maxima_count = self.ui.maxima_count_despike_spin_box.value()
        fwhm_width = self.ui.despike_fwhm_width_doubleSpinBox.value()
        self.ui.statusBar.showMessage('Despiking...')
        self.disable_buttons(True)
        self.close_progress_bar()
        items_to_despike = self.get_items_to_despike()
        n_files = len(items_to_despike)
        self._open_progress_bar(max_value=n_files)
        self._open_progress_dialog("Despiking...", "Cancel", maximum=n_files)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            return
        self.beforeTime = datetime.now()
        self.current_executor = ThreadPoolExecutor()
        if n_files > 1000:
            self.current_executor = ProcessPoolExecutor()
        fwhm_nm_df = self.ui.input_table.model().get_column('FWHM, nm')
        with self.current_executor as executor:
            self.current_futures = [loop.run_in_executor(executor, subtract_cosmic_spikes_moll, i, fwhm_nm_df[i[0]],
                                                         laser_wavelength, maxima_count, fwhm_width)
                                    for i in items_to_despike]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result_of_despike = await gather(*self.current_futures)
        # result_of_despike = []
        # for i in items_to_despike:
        #     res = subtract_cosmic_spikes_moll(i, fwhm_nm_df[i[0]],
        #                                     laser_wavelength, maxima_count, fwhm_width)
        #     result_of_despike.append(res)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        despiked_list = [i for i in result_of_despike if i]
        self.disable_buttons(False)
        self.close_progress_bar()
        if despiked_list:
            command = CommandUpdateDespike(self, despiked_list, "Despike")
            self.undoStack.push(command)
        elif not self.is_production_project:
            seconds = round((datetime.now() - self.beforeTime).total_seconds())
            self.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Icon.Warning)
            message_box.setText("No peaks found")
            message_box.setWindowTitle("Despike finished")
            message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            message_box.exec()

    def get_items_to_despike(self) -> list[tuple[str, np.ndarray]]:
        current_row = self.ui.input_table.selectionModel().currentIndex().row()
        if self.ui.by_one_control_button.isChecked() and current_row != -1:
            current_spectrum_name = self.ui.input_table.model().get_filename_by_row(current_row)
            items_to_despike = [(current_spectrum_name, self.ImportedArray[current_spectrum_name])]
        elif self.ui.by_group_control_button.isChecked() \
                and self.ui.GroupsTable.selectionModel().currentIndex().row() != -1:
            current_group = self.ui.GroupsTable.selectionModel().currentIndex().row() + 1
            filenames = self.get_names_of_group(current_group)
            items_to_despike = []
            for i in self.ImportedArray.items():
                if i[0] in filenames:
                    items_to_despike.append(i)
        else:
            items_to_despike = self.ImportedArray.items()
        return items_to_despike

    # endregion

    # region CONVERT

    @asyncSlot()
    async def convert(self):
        try:
            await self._do_convert()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def _do_convert(self) -> None:
        if len(self.ImportedArray) > 1 and len(self.check_ranges()) > 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Spectra must be interpolated before convert")
            msg.setWindowTitle("Convert stopped")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        different_shapes, _, _ = self.check_arrays_shape()
        if different_shapes:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Files have different shapes, interpolation required")
            msg.setWindowTitle("Convert stopped")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Converting...')
        self.close_progress_bar()
        n_files = len(self.ImportedArray)
        self._open_progress_dialog("Converting nm to cm\N{superscript minus}\N{superscript one}...", "Cancel",
                                   maximum=n_files)
        self._open_progress_bar(max_value=n_files)

        x_axis = np.zeros(1)
        for _, arr in self.ImportedArray.items():
            x_axis = arr[:, 0]
            break
        laser_nm = self.ui.laser_wl_spinbox.value()
        nearest_idx = find_nearest_idx(x_axis, laser_nm + 5)
        max_ccd_value = self.ui.max_CCD_value_spinBox.value()
        executor = ThreadPoolExecutor()
        if n_files >= 12_000:
            executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(executor, convert, i, nearest_idx, laser_nm, max_ccd_value)
                                    for i in self.ImportedArray.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            converted = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        command = CommandConvert(self, converted, "Convert to cm\N{superscript minus}\N{superscript one}")
        self.undoStack.push(command)
        self.close_progress_bar()
        if not self.is_production_project:
            await self.update_range_cm()

    def update_cm_min_max_range(self) -> None:
        if not self.ConvertedDict:
            return
        first_x = []
        last_x = []
        for v in self.ConvertedDict.values():
            x_axis = v[:, 0]
            first_x.append(x_axis[0])
            last_x.append(x_axis[-1])
        min_cm = max(first_x)
        max_cm = min(last_x)
        self.ui.cm_range_start.setMinimum(min_cm)
        self.ui.cm_range_start.setMaximum(max_cm)
        self.ui.cm_range_end.setMinimum(min_cm)
        self.ui.cm_range_end.setMaximum(max_cm)

        self.ui.trim_start_cm.setMinimum(min_cm)
        self.ui.trim_start_cm.setMaximum(max_cm)
        self.ui.trim_end_cm.setMinimum(min_cm)
        self.ui.trim_end_cm.setMaximum(max_cm)
        current_value_start = self.ui.cm_range_start.value()
        current_value_end = self.ui.cm_range_end.value()
        if current_value_start < min_cm or current_value_start > max_cm:
            self.ui.cm_range_start.setValue(min_cm)
        if current_value_end < min_cm or current_value_end > max_cm:
            self.ui.cm_range_start.setValue(max_cm)
        self.linearRegionCmConverted.setBounds((min_cm, max_cm))

        current_value_trim_start = self.ui.trim_start_cm.value()
        current_value_trim_end = self.ui.trim_end_cm.value()
        if current_value_trim_start < min_cm or current_value_trim_start > max_cm:
            self.ui.trim_start_cm.setValue(min_cm)
        if current_value_trim_end < min_cm or current_value_trim_end > max_cm:
            self.ui.trim_end_cm.setValue(max_cm)
        self.linearRegionBaseline.setBounds((min_cm, max_cm))
        self.linearRegionDeconv.setBounds((min_cm, max_cm))

    def cm_range_start_change_event(self, new_value: float) -> None:
        self.set_modified()
        if self.ConvertedDict:
            x_axis = next(iter(self.ConvertedDict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.cm_range_start.setValue(new_value)
        if new_value >= self.ui.cm_range_end.value():
            self.ui.cm_range_start.setValue(self.ui.cm_range_start.minimum())
        self.linearRegionCmConverted.setRegion((self.ui.cm_range_start.value(), self.ui.cm_range_end.value()))

    def cm_range_end_change_event(self, new_value: float) -> None:
        self.set_modified()
        if self.ConvertedDict:
            x_axis = next(iter(self.ConvertedDict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.cm_range_end.setValue(new_value)
        if new_value <= self.ui.cm_range_start.value():
            self.ui.cm_range_end.setValue(self.ui.cm_range_end.maximum())
        self.linearRegionCmConverted.setRegion((self.ui.cm_range_start.value(), self.ui.cm_range_end.value()))

    def lr_cm_region_changed(self) -> None:
        current_region = self.linearRegionCmConverted.getRegion()
        self.ui.cm_range_start.setValue(current_region[0])
        self.ui.cm_range_end.setValue(current_region[1])

    def update_range_btn_clicked(self) -> None:
        if self.ConvertedDict:
            self.update_range_cm()
        else:
            self.ui.statusBar.showMessage('Range update failed because there are no any converted plot ', 15000)

    @asyncSlot()
    async def update_range_cm(self) -> None:
        if not self.ConvertedDict:
            return
        self.ui.statusBar.showMessage('Updating range...')
        self.close_progress_bar()
        self._open_progress_bar(max_value=len(self.ConvertedDict))
        time_before = datetime.now()
        x_axis = next(iter(self.ConvertedDict.values()))[:, 0]  # any of dict
        value_right = find_nearest(x_axis, self.ui.cm_range_end.value())
        self.ui.cm_range_end.setValue(value_right)
        factor = self.ui.neg_grad_factor_spinBox.value()
        with ThreadPoolExecutor() as executor:
            self.current_futures = [loop.run_in_executor(executor, find_fluorescence_beginning, i, factor)
                                    for i in self.ConvertedDict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        idx = max(result)
        value_left = find_nearest_by_idx(x_axis, idx)
        self.ui.cm_range_start.setValue(value_left)
        seconds = round((datetime.now() - time_before).total_seconds())
        self.ui.statusBar.showMessage('Range updated for ' + str(seconds) + ' sec.', 5000)
        self.close_progress_bar()

    # endregion

    # region cut_first
    @asyncSlot()
    async def cut_first(self):
        try:
            await self._do_cut_first()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def _do_cut_first(self) -> None:
        if not self.ConvertedDict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No converted plots")
            msg.setWindowTitle("Cut failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        x_axis = next(iter(self.ConvertedDict.values()))[:, 0]
        value_start = self.ui.cm_range_start.value()
        value_end = self.ui.cm_range_end.value()
        if round(value_start, 5) == round(x_axis[0], 5) \
                and round(value_end, 5) == round(x_axis[-1], 5):
            msg = QMessageBox(QMessageBox.Icon.Information, 'Cut failed', 'Cut range is equal to actual spectrum range.'
                                                                          ' No need to cut.',
                              QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        self.ui.statusBar.showMessage('Cut in progress...')
        self.close_progress_bar()
        n_files = len(self.ConvertedDict)
        self._open_progress_dialog("Cut in progress...", "Cancel",
                                   maximum=n_files)
        self._open_progress_bar(max_value=n_files)
        self.beforeTime = datetime.now()
        executor = ThreadPoolExecutor()
        if n_files >= 16_000:
            executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(executor, cut_spectrum, i, value_start, value_end)
                                    for i in self.ConvertedDict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            cutted_arrays = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        if cutted_arrays:
            command = CommandCutFirst(self, cutted_arrays, "Cut spectrum")
            self.undoStack.push(command)
        self.close_progress_bar()

    # endregion

    # region Normalizing
    @asyncSlot()
    async def normalize(self):
        try:
            await self._do_normalize()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def _do_normalize(self) -> None:
        if not self.CuttedFirstDict or len(self.CuttedFirstDict) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No cutted spectra for normalization")
            msg.setWindowTitle("Normalization stopped")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Normalization...')
        self.close_progress_bar()
        n_files = len(self.CuttedFirstDict)
        self._open_progress_bar(max_value=n_files)
        self._open_progress_dialog("Normalization...", "Cancel", maximum=n_files)
        method = self.ui.normalizing_method_comboBox.currentText()
        func = self.normalize_methods[method][0]
        params = None
        n_limit = self.normalize_methods[method][1]
        if method == 'EMSC':
            if self.is_production_project:
                print(self._y_axis_ref_EMSC.shape)
                params = self._y_axis_ref_EMSC, self.ui.emsc_pca_n_spinBox.value()
            else:
                np_y_axis = get_emsc_average_spectrum(self.CuttedFirstDict.values())
                params = np_y_axis, self.ui.emsc_pca_n_spinBox.value()
                self._y_axis_ref_EMSC = np_y_axis
        executor = ThreadPoolExecutor()
        if n_files >= n_limit:
            executor = ProcessPoolExecutor()
        self.current_executor = executor
        with self.current_executor as executor:
            self.current_futures = [loop.run_in_executor(executor, func, i, params)
                                    for i in self.CuttedFirstDict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            normalized = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        if normalized:
            command = CommandNormalize(self, normalized, method, "Normalize")
            self.undoStack.push(command)
            # arrays_list = [normalized[i][1] for i in range(len(normalized))]
            # y_mean = get_average_spectrum(arrays_list)
            # residuals_sd = []
            # for i in range(len(normalized)):
            #     residual = y_mean - normalized[i][1]
            #     sd = np.mean(np.abs(residual))
            #     residuals_sd.append(sd)
        self.close_progress_bar()

    # endregion

    # region Smoothing
    @asyncSlot()
    async def smooth(self):
        try:
            await self.do_smooth()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def do_smooth(self) -> None:
        if not self.NormalizedDict or len(self.NormalizedDict) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No normalized spectra for smoothing")
            msg.setWindowTitle("Smoothing stopped")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Smoothing...')
        self.close_progress_bar()
        n_files = len(self.NormalizedDict)
        self._open_progress_bar(max_value=n_files)
        self._open_progress_dialog("Smoothing...", "Cancel", maximum=n_files)
        snr_df = self.ui.input_table.model().get_column('SNR')
        method = self.ui.smoothing_method_comboBox.currentText()
        executor = ThreadPoolExecutor()
        func = self.smoothing_methods[method][0]
        n_samples_limit = self.smoothing_methods[method][1]
        params = self.smoothing_params(method)
        if n_files >= n_samples_limit:
            executor = ProcessPoolExecutor()
        self.current_executor = executor

        # smoothed = []
        # for i in self.NormalizedDict.items():
        #     smoothed.append(func(i, (params[0], params[1], snr_df.at[i[0]])))

        with executor:
            if method == 'MLESG':
                self.current_futures = [loop.run_in_executor(executor, func, i, (params[0], params[1],
                                                                                 snr_df.at[i[0]]))
                                        for i in self.NormalizedDict.items()]
            else:
                self.current_futures = [loop.run_in_executor(executor, func, i, params)
                                        for i in self.NormalizedDict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            smoothed = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        if smoothed:
            command = CommandSmooth(self, smoothed, method, params, "Smooth")
            self.undoStack.push(command)
        self.close_progress_bar()

    def smoothing_params(self, method: str) -> int | tuple[int, int | str] | tuple[float, int, int]:
        params = None
        match method:
            case 'EMD':
                params = self.ui.emd_noise_modes_spinBox.value()
            case 'EEMD' | 'CEEMDAN':
                params = self.ui.emd_noise_modes_spinBox.value(), self.ui.eemd_trials_spinBox.value()
            case 'MLESG':
                fwhm_cm_df = self.ui.input_table.model().get_column('FWHM, cm\N{superscript minus}\N{superscript one}')
                distance = np.max(fwhm_cm_df.values)
                sigma = self.ui.sigma_spinBox.value()
                params = (distance, sigma, 0)
            case 'Savitsky-Golay filter':
                params = self.ui.window_length_spinBox.value(), self.ui.smooth_polyorder_spinBox.value()
            case 'Whittaker smoother':
                params = self.ui.whittaker_lambda_spinBox.value()
            case 'Flat window':
                params = self.ui.window_length_spinBox.value()
            case 'hanning' | 'hamming' | 'bartlett' | 'blackman':
                params = self.ui.window_length_spinBox.value(), method
            case 'kaiser':
                params = self.ui.window_length_spinBox.value(), self.ui.kaiser_beta_doubleSpinBox.value()
            case 'Median filter' | 'Wiener filter':
                params = self.ui.window_length_spinBox.value()
        return params

    # endregion

    # region baseline_correction
    @asyncSlot()
    async def baseline_correction(self):
        if not self.SmoothedDict or len(self.SmoothedDict) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No spectra for baseline correction")
            msg.setWindowTitle("Baseline correction stopped")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        try:
            await self.do_baseline_correction()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def do_baseline_correction(self) -> None:
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Baseline correction...')
        self.close_progress_bar()
        n_files = len(self.SmoothedDict)
        self._open_progress_bar(max_value=n_files)
        self._open_progress_dialog("Baseline correction...", "Cancel", maximum=n_files)
        method = self.ui.baseline_correction_method_comboBox.currentText()
        executor = ThreadPoolExecutor()
        func = self.baseline_methods[method][0]
        n_samples_limit = self.baseline_methods[method][1]
        params = self.baseline_correction_params(method)
        if n_files >= n_samples_limit:
            executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(executor, func, i, params)
                                    for i in self.SmoothedDict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            baseline_corrected = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        if baseline_corrected:
            command = CommandBaselineCorrection(self, baseline_corrected, method, params, "Baseline correction")
            self.undoStack.push(command)
        self.close_progress_bar()
        if not self.is_production_project:
            await self.update_range_baseline_corrected()

    def baseline_correction_params(self, method: str) -> float | list:
        params = None
        match method:
            case 'Poly':
                params = self.ui.polynome_degree_spinBox.value()
            case 'ModPoly' | 'iModPoly' | 'iModPoly+':
                params = [self.ui.polynome_degree_spinBox.value(), self.ui.grad_doubleSpinBox.value(),
                          self.ui.n_iterations_spinBox.value()]
            case 'Penalized poly':
                params = [self.ui.polynome_degree_spinBox.value(), self.ui.grad_doubleSpinBox.value(),
                          self.ui.n_iterations_spinBox.value(), self.ui.alpha_factor_doubleSpinBox.value(),
                          self.ui.cost_func_comboBox.currentText()]
            case 'LOESS':
                poly_order = min(6, self.ui.polynome_degree_spinBox.value())
                params = [poly_order, self.ui.grad_doubleSpinBox.value(), self.ui.n_iterations_spinBox.value(),
                          self.ui.fraction_doubleSpinBox.value(), self.ui.scale_doubleSpinBox.value()]
            case 'Goldindec':
                params = [self.ui.polynome_degree_spinBox.value(), self.ui.grad_doubleSpinBox.value(),
                          self.ui.n_iterations_spinBox.value(), self.ui.cost_func_comboBox.currentText(),
                          self.ui.peak_ratio_doubleSpinBox.value(), self.ui.alpha_factor_doubleSpinBox.value()]
            case 'Quantile regression':
                params = [self.ui.polynome_degree_spinBox.value(), self.ui.grad_doubleSpinBox.value(),
                          self.ui.n_iterations_spinBox.value(), self.ui.quantile_doubleSpinBox.value()]
            case 'AsLS' | 'iAsLS' | 'arPLS' | 'airPLS' | 'iarPLS' | 'asPLS' | 'psaLSA' | 'DerPSALSA' | 'MPLS':
                params = [self.ui.lambda_spinBox.value(), self.ui.p_doubleSpinBox.value(),
                          self.ui.n_iterations_spinBox.value()]
            case 'drPLS':
                params = [self.ui.lambda_spinBox.value(), self.ui.p_doubleSpinBox.value(),
                          self.ui.n_iterations_spinBox.value(), self.ui.eta_doubleSpinBox.value()]
            case 'iMor' | 'MorMol' | 'AMorMol' | 'JBCD':
                params = [self.ui.n_iterations_spinBox.value(), self.ui.grad_doubleSpinBox.value()]
            case 'MPSpline':
                params = [self.ui.lambda_spinBox.value(), self.ui.p_doubleSpinBox.value(),
                          self.ui.spline_degree_spinBox.value()]
            case 'Mixture Model':
                params = [self.ui.lambda_spinBox.value(), self.ui.p_doubleSpinBox.value(),
                          self.ui.spline_degree_spinBox.value(), self.ui.n_iterations_spinBox.value(),
                          self.ui.grad_doubleSpinBox.value()]
            case 'IRSQR':
                params = [self.ui.lambda_spinBox.value(), self.ui.quantile_doubleSpinBox.value(),
                          self.ui.spline_degree_spinBox.value(), self.ui.n_iterations_spinBox.value()]
            case 'Corner-Cutting':
                params = self.ui.n_iterations_spinBox.value()
            case 'RIA':
                params = self.ui.grad_doubleSpinBox.value()
            case 'Dietrich':
                params = [self.ui.num_std_doubleSpinBox.value(), self.ui.polynome_degree_spinBox.value(),
                          self.ui.grad_doubleSpinBox.value(), self.ui.n_iterations_spinBox.value(),
                          self.ui.interp_half_window_spinBox.value(), self.ui.min_length_spinBox.value()]
            case 'Golotvin':
                params = [self.ui.num_std_doubleSpinBox.value(), self.ui.interp_half_window_spinBox.value(),
                          self.ui.min_length_spinBox.value(), self.ui.sections_spinBox.value()]
            case 'Std Distribution':
                params = [self.ui.num_std_doubleSpinBox.value(), self.ui.interp_half_window_spinBox.value(),
                          self.ui.fill_half_window_spinBox.value()]
            case 'FastChrom':
                params = [self.ui.interp_half_window_spinBox.value(), self.ui.n_iterations_spinBox.value(),
                          self.ui.min_length_spinBox.value()]
            case 'FABC':
                params = [self.ui.lambda_spinBox.value(), self.ui.num_std_doubleSpinBox.value(),
                          self.ui.min_length_spinBox.value()]
            case 'OER' | 'Adaptive MinMax':
                params = self.ui.opt_method_oer_comboBox.currentText()
        return params

    def lr_baseline_region_changed(self) -> None:
        current_region = self.linearRegionBaseline.getRegion()
        self.ui.trim_start_cm.setValue(current_region[0])
        self.ui.trim_end_cm.setValue(current_region[1])

    def _trim_start_change_event(self, new_value: float) -> None:
        self.set_modified()
        if self.baseline_corrected_not_trimmed_dict:
            x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.trim_start_cm.setValue(new_value)
        if new_value >= self.ui.trim_end_cm.value():
            self.ui.trim_start_cm.setValue(self.ui.trim_start_cm.minimum())
        self.linearRegionBaseline.setRegion((self.ui.trim_start_cm.value(), self.ui.trim_end_cm.value()))

    def _trim_end_change_event(self, new_value: float) -> None:
        self.set_modified()
        if self.baseline_corrected_not_trimmed_dict:
            x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.trim_end_cm.setValue(new_value)
        if new_value <= self.ui.trim_start_cm.value():
            self.ui.trim_end_cm.setValue(self.ui.trim_end_cm.maximum())
        self.linearRegionBaseline.setRegion((self.ui.trim_start_cm.value(), self.ui.trim_end_cm.value()))

    def update_trim_range_btn_clicked(self) -> None:
        if self.baseline_corrected_not_trimmed_dict:
            self.update_range_baseline_corrected()
        else:
            self.ui.statusBar.showMessage('Range update failed because there are no any baseline corrected plot ',
                                          15_000)

    @asyncSlot()
    async def update_range_baseline_corrected(self) -> None:
        if not self.baseline_corrected_not_trimmed_dict:
            return
        self.ui.statusBar.showMessage('Updating range...')
        self.close_progress_bar()
        self._open_progress_bar(max_value=len(self.baseline_corrected_not_trimmed_dict))
        # time_before = datetime.now()
        x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]  # any of dict
        with ThreadPoolExecutor() as executor:
            self.current_futures = [loop.run_in_executor(executor, find_first_right_local_minimum, i)
                                    for i in self.baseline_corrected_not_trimmed_dict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        idx = int(np.percentile(result, 0.95))
        value_right = x_axis[idx]
        self.ui.trim_end_cm.setValue(value_right)

        with ThreadPoolExecutor() as executor:
            self.current_futures = [loop.run_in_executor(executor, find_first_left_local_minimum, i)
                                    for i in self.baseline_corrected_not_trimmed_dict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        idx = np.max(result)
        value_left = x_axis[idx]
        self.ui.trim_start_cm.setValue(value_left)
        # seconds = round((datetime.now() - time_before).total_seconds())
        # self.ui.statusBar.showMessage('Range updated for ' + str(seconds) + ' sec.', 5000)
        self.close_progress_bar()

    # endregion

    # region final trim
    @asyncSlot()
    async def trim(self):
        try:
            await self._do_trim()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def _do_trim(self) -> None:
        if not self.baseline_corrected_not_trimmed_dict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No baseline corrected plots")
            msg.setWindowTitle("Trimming failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]
        value_start = self.ui.trim_start_cm.value()
        value_end = self.ui.trim_end_cm.value()
        if round(value_start, 5) == round(x_axis[0], 5) \
                and round(value_end, 5) == round(x_axis[-1], 5):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Trim range is equal to actual spectrum range. No need to cut.")
            msg.setWindowTitle("Trim failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        self.ui.statusBar.showMessage('Trimming in progress...')
        self.close_progress_bar()
        n_files = len(self.baseline_corrected_not_trimmed_dict)
        self._open_progress_dialog("Trimming in progress...", "Cancel",
                                   maximum=n_files)
        self._open_progress_bar(max_value=n_files)
        self.beforeTime = datetime.now()
        executor = ThreadPoolExecutor()
        if n_files >= 16_000:
            executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(executor, cut_spectrum, i, value_start, value_end)
                                    for i in self.baseline_corrected_not_trimmed_dict.items()]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            cutted_arrays = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            return
        if cutted_arrays:
            command = CommandTrim(self, cutted_arrays, "Trim spectrum")
            self.undoStack.push(command)
        self.close_progress_bar()

    @asyncSlot()
    async def update_averaged(self) -> None:
        if self.baseline_corrected_dict:
            self.update_averaged_dict()
            self.update_template_combo_box()
            await self.update_plot_item(self.averaged_dict.items(), 6)
            if not self.is_production_project:
                self.update_deconv_intervals_limits()

    # endregion

    # region Fitting

    # region Add line
    @asyncSlot()
    async def add_deconv_line(self, line_type: str):
        if not self.isTemplate:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Switch to Template mode to add new line")
            msg.setWindowTitle("Add line failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        elif not self.baseline_corrected_dict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No baseline corrected spectrum")
            msg.setWindowTitle("Add line failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
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

    # region BATCH FIT
    @asyncSlot()
    async def batch_fit(self):
        """
        Check conditions when Fit button pressed, if all ok - go do_batch_fit
        For fitting must be more than 0 lines to fit
        Должна быть хотя бы 1 линия и есть спектры
        """
        if self.ui.deconv_lines_table.model().rowCount() == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Add some new lines before fitting")
            msg.setWindowTitle("Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        elif not self.baseline_corrected_dict or len(self.baseline_corrected_dict) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("There is No any data to fit")
            msg.setWindowTitle("Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        try:
            await self.do_batch_fit()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def do_batch_fit(self) -> None:
        """
        Fitting line's parameters to all spectrum files
        1. Get x, y axes of current spectrum
        2. Prepare data before creating model and parameters
        3. Create params
        4. Create model
        5. Fit model to y_data
        """
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Batch fitting...')
        self.close_progress_bar()
        arrays = {}

        if self.is_production_project and self.ui.deconvoluted_dataset_table_view.model().rowCount() != 0:
            filenames_in_dataset = list(self.ui.deconvoluted_dataset_table_view.model().column_data(1).values)
            for key, arr in self.baseline_corrected_dict.items():
                if key not in filenames_in_dataset:
                    arrays[key] = [arr]
        else:
            for key, arr in self.baseline_corrected_dict.items():
                arrays[key] = [arr]
        splitted_arrays = self.split_array_for_fitting(arrays)
        idx_type_param_count_legend_func = self._prepare_data_fitting()
        list_params_full = self._fitting_params_batch(idx_type_param_count_legend_func, arrays)
        x_y_models_params = models_params_splitted_batch(splitted_arrays, list_params_full,
                                                         idx_type_param_count_legend_func)
        method_full_name = self.ui.fit_opt_method_comboBox.currentText()
        method = self.fitting_methods[method_full_name]
        executor = ProcessPoolExecutor()
        self.current_executor = executor
        key_x_y_models_params = []
        for key, item in x_y_models_params.items():
            for x_axis, y_axis, model, interval_params in item:
                key_x_y_models_params.append((key, x_axis, y_axis, model, interval_params))
        intervals = len(key_x_y_models_params)
        self._open_progress_bar(max_value=intervals if intervals > 1 else 0)
        self._open_progress_dialog("Batch Fitting...", "Cancel", maximum=intervals if intervals > 1 else 0)
        with executor:
            self.current_futures = [loop.run_in_executor(executor, fit_model_batch, model, y, params, x, method, key)
                                    for key, x, y, model, params in key_x_y_models_params]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Fitting cancelled.')
            return
        self.close_progress_bar()
        self.ui.statusBar.showMessage('Calculating uncertaintes...')
        n_files = len(result)
        self._open_progress_bar(max_value=n_files if n_files > 1 else 0)
        self._open_progress_dialog("Calculating uncertaintes...", "Cancel", maximum=n_files if n_files > 1 else 0)
        executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(self.current_executor, eval_uncert, i)
                                    for i in result]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            dely = await gather(*self.current_futures)
        self.close_progress_bar()
        self._open_progress_bar(max_value=0)
        self._open_progress_dialog("updating data...", "Cancel", maximum=0)
        command = CommandAfterBatchFitting(self, result, idx_type_param_count_legend_func, dely,
                                           "Fitted spectrum in batch ")
        self.undoStack.push(command)
        self.close_progress_bar()

    def _fitting_params_batch(self, idx_type_param_count_legend_func: list[tuple[int, str, int, str, callable]],
                              arrays: dict[str, list[np.ndarray]]) -> dict:
        x_axis = next(iter(arrays.values()))[0][:, 0]
        params_mutual = self._fitting_params(idx_type_param_count_legend_func, 1000., x_axis[0], x_axis[-1])
        list_params_full = {}
        for key, item in arrays.items():
            this_key_params = copy.deepcopy(params_mutual)
            y_axis = item[0][:, 1]
            for par in this_key_params:
                par_splitted = par.split('_', 2)
                if par_splitted[-1] != 'a':
                    continue
                dx_right_param_name = par_splitted[0] + '_' + par_splitted[1] + '_dx'
                dx_right_max = this_key_params[dx_right_param_name].max
                dx_left_param_name = par_splitted[0] + '_' + par_splitted[1] + '_dx_left'
                if dx_left_param_name in this_key_params:
                    dx_left_max = this_key_params[dx_left_param_name].max
                else:
                    dx_left_max = this_key_params[dx_right_param_name].max
                x0_min = this_key_params[par_splitted[0] + '_' + par_splitted[1] + '_x0'].min
                x0_max = this_key_params[par_splitted[0] + '_' + par_splitted[1] + '_x0'].max
                x0_left = x0_min - dx_left_max
                x0_right = x0_max + dx_right_max
                arg_x0_left = find_nearest_idx(x_axis, x0_left)
                arg_x0_right = find_nearest_idx(x_axis, x0_right)
                y_max_in_range = np.amax(y_axis[arg_x0_left:arg_x0_right])
                this_key_params[par].max = y_max_in_range
                this_key_params[par].value = this_key_params[par].init_value = y_max_in_range / 2.
            list_params_full[key] = this_key_params
        return list_params_full

    # endregion

    # region FIT
    @asyncSlot()
    async def fit(self):
        """
        Check conditions when Fit button pressed, if all ok - go do_fit
        For fitting must be more than 0 lines to fit and current spectrum in plot must also be
        Должна быть хотя бы 1 линия и выделен текущий спектр
        """
        if self.ui.deconv_lines_table.model().rowCount() == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Add some new lines before fitting")
            msg.setWindowTitle("Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        elif self.array_of_current_filename_in_deconvolution is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("There is No any data to fit")
            msg.setWindowTitle("Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        try:
            await self.do_fit()
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def do_fit(self) -> None:
        """
        Fitting line's parameters to current spectrum data
        1. Get x, y axes of current spectrum
        2. Prepare data before creating model and parameters
        3. Create params
        4. Create model
        5. Fit model to y_data
        """
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Fitting...')
        self.close_progress_bar()

        spec_name = self.current_spectrum_deconvolution_name
        # if spec_name != '':
        #     self.add_line_params_from_template(spec_name)
        arr = self.array_of_current_filename_in_deconvolution()
        splitted_arrays = self.split_array_for_fitting({spec_name: [arr]})[spec_name]
        intervals = len(splitted_arrays)
        self._open_progress_bar(max_value=intervals if intervals > 1 else 0)
        self._open_progress_dialog("Fitting...", "Cancel", maximum=intervals if intervals > 1 else 0)
        idx_type_param_count_legend_func = self._prepare_data_fitting()
        params = self._fitting_params(idx_type_param_count_legend_func, np.max(arr[:, 1]), arr[:, 0][0], arr[:, 0][-1])
        x_y_models_params, _ = models_params_splitted(splitted_arrays, params, idx_type_param_count_legend_func)
        method_full_name = self.ui.fit_opt_method_comboBox.currentText()
        method = self.fitting_methods[method_full_name]
        executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(executor, fit_model, i[2], i[1], i[3], i[0], method)
                                    for i in x_y_models_params]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Fitting cancelled.')
            return
        command = CommandAfterFitting(self, result, idx_type_param_count_legend_func, spec_name,
                                      "Fitted spectrum %s" % spec_name)
        self.undoStack.push(command)
        self.close_progress_bar()
        self.ui.statusBar.showMessage('Fitting completed', 10000)

    # endregion

    def _prepare_data_fitting(self) -> list[tuple[int, str, int, str, callable]]:
        """
        Get prepared data to create model and parameters

        Returns
        -------
        func_legend : list[tuple[func, str]]
            using to create fit Model. func is a line function from spec_functions/ peak_shapes, str is curve legend
        idx_type_paramcount_legend : list[tuple[int, str, int, str]]
            using to create Parameters of model. idx - curve index, type - curve type, param_count - number of
             parameters of line_type, legend - curve legend by index
        """
        # get dataframe with active lines - idx | Legend, Type, Style
        line_types = self.ui.deconv_lines_table.model().get_visible_line_types()
        idx_type_param_count_legend = []
        for i in line_types.index:
            ser = line_types.loc[i]
            if 'add_params' in self.peak_shapes_params[ser.Type]:
                param_count = 3 + len(self.peak_shapes_params[ser.Type]['add_params'])
            else:
                param_count = 3
            legend = 'Curve_%s_' % i
            # ser.Legend.replace(' ', '_') + '_'
            idx_type_param_count_legend.append((i, ser.Type, param_count, legend,
                                                self.peak_shapes_params[ser.Type]['func']))
        return idx_type_param_count_legend

    def _fitting_params(self, list_idx_type: list[tuple[int, str, int, str, callable]], bound_max_a: float,
                        bound_min_x0: float, bound_max_x0: float) -> Parameters:
        """
        Set parameters for fit model

        Parameters
        ---------
        list_idx_type : list[tuple[int, str, int, str, callable]
            idx - line index
            line_type - 'Gaussian' for example
            param_count - number of parameters for line type. example for Gaussian = 3, pearson4 = 5
            legend - for parameter name
            _callable - not used here

        bound_max_a : float
            maximal Intensity of y_axis spectrum

        bound_min_x0 : float
            minimal x of x_axis (first value of x_axis)

        bound_max_x0 : float
            maximal x of x_axis (last value of x_axis)

        Returns
        -------
        params : Parameters()
            initial values of fit parameters
        """
        bound_min_a = 0.
        if self.current_spectrum_deconvolution_name == '':
            bound_min_dx = np.max(self.ui.input_table.model().column_data(6)) / 2
        else:
            row_data = self.ui.input_table.model().row_data_by_index(self.current_spectrum_deconvolution_name)
            bound_min_dx = row_data['FWHM, cm\N{superscript minus}\N{superscript one}'] / 2
        params = Parameters()
        i = 0
        for idx, line_type, param_count, legend, _ in list_idx_type:
            add_params_len = param_count - 3
            params_from_table = self.current_line_parameters(idx, '')
            param_names = ['a', 'x0', 'dx']
            for j in range(add_params_len):
                param_names.append(self.peak_shapes_params[line_type]['add_params'][j])
            # value must be between bounds
            for param_name in param_names:
                v = params_from_table[param_name]
                # bounds correction
                bound_min_v = None
                bound_max_v = None
                if param_name == 'a':
                    bound_min_v = bound_min_a
                    bound_max_v = bound_max_a
                elif param_name == 'x0':
                    bound_min_v = bound_min_x0
                    bound_max_v = bound_max_x0
                elif param_name == 'dx' or param_name == 'dx_left':
                    bound_min_v = bound_min_dx
                    bound_max_v = (bound_max_x0 - bound_min_x0) / 2
                elif param_name in self.peak_shape_params_limits:
                    bound_min_v = self.peak_shape_params_limits[param_name][0]
                    bound_max_v = self.peak_shape_params_limits[param_name][1]
                min_v = params_from_table['min_' + param_name]
                max_v = params_from_table['max_' + param_name]
                if bound_min_v is not None:
                    min_v = bound_min_v if min_v < bound_min_v else min_v
                if bound_max_v is not None:
                    max_v = bound_max_v if max_v > bound_max_v else max_v
                v = min_v if v < min_v else v
                v = max_v if v > max_v else v
                if min_v == max_v:
                    max_v += 0.001
                if param_name == 'a':
                    max_v = bound_max_a
                params.add(legend + param_name, v, min=min_v, max=max_v)
            i += param_count
        return params

    def set_parameters_after_fit_for_spectrum(self, fit_result: lmfit.model.ModelResult, filename: str) -> None:
        """
        Set fitted parameters value after fitting

        Parameters
        ---------
        fit_result : lmfit.model.ModelResult

        filename : str
            filename of spectrum model was fitted to
        """
        for key, param in fit_result.params.items():
            idx_param_name = key.split('_', 2)
            idx = int(idx_param_name[1])
            param_name = idx_param_name[2]

            self.ui.fit_params_table.model().set_parameter_value(filename, idx, param_name, 'Value',
                                                                 param.value, False)
            self.ui.fit_params_table.model().set_parameter_value(filename, idx, param_name, 'Max value',
                                                                 param.max, False)
            self.ui.fit_params_table.model().set_parameter_value(filename, idx, param_name, 'Min value',
                                                                 param.min, False)

    # region GUESS
    @asyncSlot()
    async def guess(self, line_type: str) -> None:
        """
        Auto guess lines, finds number of lines and positions x0
        """
        if not self.baseline_corrected_dict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Do baseline correction before guessing peaks")
            msg.setWindowTitle("Guess failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        if self.ui.interval_checkBox.isChecked() and self.ui.intervals_gb.isChecked():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("'Split by intervals' и 'Interval' активны. Нужно оставить один из них")
            msg.setWindowTitle("Guess failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        if self.ui.intervals_gb.isChecked() and self.ui.fit_intervals_table_view.model().rowCount() == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Если 'Split by intervals' активно, то нужно заполнить таблицу")
            msg.setWindowTitle("Guess failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        if self.ui.guess_method_cb.currentText() == 'Average groups' and \
                (not self.averaged_dict or len(self.averaged_dict) < 2):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Если выбран метод 'Average groups', то должно быть больше 1 группы")
            msg.setWindowTitle("Guess failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        if not self.ui.interval_checkBox.isChecked() and not self.ui.intervals_gb.isChecked():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setText('Авто определение линий в полном диапазоне может занимать очень много времени' + '\n'
                        + 'Точно продолжить?')
            msg.setInformativeText('Разделение спектра на диапазоны может сократить время анализа на 2-3 порядка. '
                                   'А без разделения вычисление может идти часами')
            msg.setWindowTitle("Achtung!")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if msg.exec() == QMessageBox.StandardButton.Yes:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Question)
                msg.setText('Последний шанс передумать' + '\n' + 'Точно продолжить?')
                msg.setWindowTitle("Achtung!")
                msg.setStandardButtons(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                if not msg.exec() == QMessageBox.StandardButton.Yes:
                    return
            else:
                return
        try:
            await self.do_auto_guess(line_type)
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def do_auto_guess(self, line_type: str) -> None:
        """
        Автоматический подбор линий к модели. Результат отображается только для шаблона.
        Peaks are added at position of global extremum of data-baseline with previous peaks subtracted.
        Таблица параметров для всех спектров очищается. Результаты прошлых fit очищаются.
        Если на момент начала анализа уже есть линии в таблице линий, то поиск линий начнется не с нуля,
         а с этого состава.
        Спектр делится на интервалы.
        Guess method:
            Если выбран 'Average', то анализируется только усредненный спектр по всем группам.
            Если выбран 'Average groups', то анализируются усредненные по каждой отдельной группе
            Если выбран 'All', то анализируются усредненные по всем спектрам из левой таблицы.
        Для 'Average groups' и 'All' собираются линии x0. По каждому интервалу анализируется количество линий.
        Итоговое количество линий для каждого интервала определяется методом кластеризации k-means.
        Метод определения количества задается в N lines method
        N lines method:
            После определения состава линий автоматическим Guess по каждому спектру формируется список x0.
             По каждому интервалу количество линий может для разных спектров может быть разным.
              Выбор метода влияет на количество линий в итоговой модели. Это количество дальше используется в k-means.
            Пример для одного интервала из 5 спектров: [5, 5, 6, 6, 7]
            Min - будет выбрано минимальное количество линий (5).
            Max - максимальное из всех (7).
            Mean - 5.8, округляется до 6
            Median - 6
        После k-means полученный состав линий опять прогоняется через Fit и мы получаем итоговую модель.

        В процессе анализа на каждый параметр линий накладывается ряд ограничений.
         a - амплитуда от 0 до максимального значения в интервале. Интервал зависит от x0 и dx
         x0 - положение максимума линии. Изначально определяется по положению максимума residual +- 1 см-1. Или после
            k-means границы задаются мин-макс значением в кластере
         dx, dx_left - полуширина линии левая/правая, максимальное значение задается в Max peak HWHM:, минимальное
            определяется из наименьшего FWHM CM-1 / 2 в таблице
        Остальные параметры имеют границы указанные в peak_shape_params_limits

        @param line_type: str
        @return: None
        """
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Guessing peaks...')
        self.close_progress_bar()
        parameters_to_guess = self.parameters_to_guess(line_type)
        mean_snr = parameters_to_guess['mean_snr']
        noise_level = np.max(self.averaged_array[:, 1]) / mean_snr
        noise_level = max(noise_level, self.ui.max_noise_level_dsb.value())
        parameters_to_guess['noise_level'] = noise_level
        arrays_for_guess = self.arrays_for_peak_guess().values()
        splitted_arrays = []
        for i in arrays_for_guess:
            for j in i:
                splitted_arrays.append(j)
        n_files = len(splitted_arrays)
        if n_files == 1:
            n_files = 0
        self._open_progress_bar(max_value=n_files)
        self._open_progress_dialog("Analyze...", "Cancel", maximum=n_files)
        executor = ProcessPoolExecutor()
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(self.current_executor, guess_peaks, arr, parameters_to_guess)
                                    for arr in splitted_arrays]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        info('result {!s}'.format(result))
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            info('Cancelled')
            return
        if self.ui.guess_method_cb.currentText() != 'Average':
            x_y_models_params = self.analyze_guess_results(result, parameters_to_guess['param_names'], line_type)
            self.close_progress_bar()
            self.progressBar.setValue(0)
            intervals = len(x_y_models_params)
            self._open_progress_bar(max_value=intervals if intervals > 1 else 0)
            self._open_progress_dialog("Fitting...", "Cancel", maximum=intervals if intervals > 1 else 0)
            executor = ThreadPoolExecutor()
            self.current_executor = executor
            with executor:
                self.current_futures = [loop.run_in_executor(executor, fit_model, i[2], i[1], i[3], i[0],
                                                             parameters_to_guess['method'])
                                        for i in x_y_models_params]
                for future in self.current_futures:
                    future.add_done_callback(self.progress_indicator)
                result = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled.')
            info('Cancelled')
            return
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        info('to CommandAfterGuess')
        command = CommandAfterGuess(self, result, line_type, parameters_to_guess['param_names'], "Auto guess")
        self.undoStack.push(command)
        self.close_progress_bar()

    def analyze_guess_results(self, result: list[lmfit.model.ModelResult],
                              param_names: list[str], line_type: str) -> list[tuple]:
        method = self.ui.n_lines_detect_method_cb.currentText()
        if self.ui.interval_checkBox.isChecked():
            intervals = [(self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value())]
        else:
            intervals = self.intervals_by_borders_values()
        data_by_intervals = {}
        for start, end in intervals:
            key = str(round(start)) + '_' + str(round(end))
            data_by_intervals[key] = {'interval': (start, end), 'x0': [], 'lines_count': []}
        params_count = len(param_names)
        for fit_result in result:
            parameters = fit_result.params
            lines_count = int(len(parameters) / params_count)
            interval_key = None
            for j, par in enumerate(parameters):
                str_split = par.split('_', 1)
                param_name = str_split[1]
                if j == len(parameters) - 1 and interval_key is not None:
                    data_by_intervals[interval_key]['lines_count'].append(lines_count)
                if param_name != 'x0':
                    continue
                x0 = parameters[par].value
                interval_key = find_interval_key(x0, data_by_intervals)
                if interval_key is None:
                    continue
                data_by_intervals[interval_key]['x0'].append(x0)
        key_clustered_x0 = process_data_by_intervals(data_by_intervals, method)
        x_y_models_params = self.models_params_splitted_after_guess(key_clustered_x0, param_names, line_type)
        return x_y_models_params

    def models_params_splitted_after_guess(self, key_clustered_x0: list[tuple],
                                           param_names: list[str], line_type: str) \
            -> list[tuple[np.ndarray, np.ndarray, Model, Parameters]]:
        """
        Подготовка порезанных x, y, fit model и параметров из списка волновых чисел, разделенных на диапазоны.
        @param line_type: str
        @param param_names: str - Ex: ['a', 'x0', 'dx', ...]
        @param key_clustered_x0: list[tuple[np.ndarray, float]]
        @return: x_y_model_params list[tuple[np.ndarray, np.ndarray, Model, Parameters]]
        """
        splitted_arrays = self.split_array_for_fitting({'Average': [self.averaged_array]})['Average']
        x_y_model_params = []
        init_params = self.get_initial_parameters_for_line(line_type)
        func = self.peak_shapes_params[line_type]['func']
        max_dx = self.ui.max_dx_dsb.value()
        min_fwhm = np.min(self.ui.input_table.model().get_column('FWHM,'
                                                                 ' cm\N{superscript minus}\N{superscript one}').values)
        min_hwhm = min_fwhm / 2.
        static_params = init_params, max_dx, min_hwhm, func, self.peak_shape_params_limits

        for i, item in enumerate(key_clustered_x0):
            wavenumbers, sd = item
            n_array = splitted_arrays[i]
            x_axis = n_array[:, 0]
            y_axis = n_array[:, 1]
            params, func_legend = process_wavenumbers_interval(wavenumbers, n_array, param_names, sd, static_params)
            model = fitting_model(func_legend)
            x_y_model_params.append((x_axis, y_axis, model, params))
        return x_y_model_params

    def arrays_for_peak_guess(self) -> dict[str, list[np.ndarray]]:
        guess_method = self.ui.guess_method_cb.currentText()
        arrays = {'Average': [self.averaged_array]}

        if guess_method == 'Average groups':
            arrays = {}
            for key, arr in self.averaged_dict.items():
                arrays[key] = [arr]
        elif guess_method == 'All':
            arrays = {}
            for key, arr in self.baseline_corrected_dict.items():
                arrays[key] = [arr]
        splitted_arrays = self.split_array_for_fitting(arrays)

        return splitted_arrays

    # endregion

    def split_array_for_fitting(self, arrays: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
        """
        На входе список массивов, на выходе порезанный на интервалы список массивов
        @param arrays: dict[str, list[np.ndarray]] key | list[2D array x|y]
        @return: dict[str, list[np.ndarray]] key | splitted spectrum
        """
        splitted_arrays = {}
        if self.ui.interval_checkBox.isChecked():
            for key, arr in arrays.items():
                splitted_arrays[key] = [cut_full_spectrum(arr[0], self.ui.interval_start_dsb.value(),
                                                          self.ui.interval_end_dsb.value())]
        elif self.ui.intervals_gb.isChecked():
            intervals = self.intervals_by_borders()
            for key, arr in arrays.items():
                splitted_arrays[key] = split_by_borders(arr[0], intervals)
        else:
            splitted_arrays = arrays
        return splitted_arrays

    def intervals_by_borders(self) -> list[tuple[int, int]]:
        """
        Indexes of intervals in x_axis
        @return:
        list[tuple[int, int]]
        Example: [(0, 57), (58, 191), (192, 257), (258, 435), (436, 575), (576, 799)]
        """
        borders = list(self.ui.fit_intervals_table_view.model().column_data(0))
        x_axis = next(iter(self.baseline_corrected_dict.values()))[:, 0]
        idx_in_range = []
        for i in borders:
            if x_axis[0] < i < x_axis[-1]:
                idx = find_nearest_idx(x_axis, i)
                idx_in_range.append(idx)
        intervals_by_borders = [(0, idx_in_range[0])]
        for i in range(len(idx_in_range) - 1):
            intervals_by_borders.append((idx_in_range[i], idx_in_range[i + 1]))
        intervals_by_borders.append((idx_in_range[-1], x_axis.shape[0] - 1))
        return intervals_by_borders

    def intervals_by_borders_values(self) -> list[tuple[float, float]]:
        """
        Values of intervals in x_axis
        @return:
        list[tuple[float, float]]
        Example:
        [(0, 409.4864599363591), (409.4864599363591, 660.819089227205), (660.819089227205, 780.8823589338135),
         (780.8823589338135, 1093.1046703298653), (1093.1046703298653, 1327.4528952748951),
          (1327.4528952748951, 1683.4121245292645)]
        """
        borders = list(self.ui.fit_intervals_table_view.model().column_data(0))
        x_axis = next(iter(self.baseline_corrected_dict.values()))[:, 0]
        v_in_range = []
        for i in borders:
            if x_axis[0] < i < x_axis[-1]:
                v = find_nearest(x_axis, i)
                v_in_range.append(v)
        intervals_by_borders = [(0, v_in_range[0])]
        for i in range(len(v_in_range) - 1):
            intervals_by_borders.append((v_in_range[i], v_in_range[i + 1]))
        intervals_by_borders.append((v_in_range[-1], x_axis[-1]))
        return intervals_by_borders

    def parameters_to_guess(self, line_type: str) -> dict:
        func = self.peak_shapes_params[line_type]['func']
        param_names = ['a', 'x0', 'dx']
        if line_type in self.peak_shapes_params and 'add_params' in self.peak_shapes_params[line_type]:
            for i in self.peak_shapes_params[line_type]['add_params']:
                param_names.append(i)
        init_params = self.get_initial_parameters_for_line(line_type)
        init_model_params = []
        for i, j in init_params.items():
            if i != 'x_axis':
                init_model_params.append(j)
        min_fwhm = np.min(self.ui.input_table.model().get_column('FWHM,'
                                                                 ' cm\N{superscript minus}\N{superscript one}').values)
        snr_df = self.ui.input_table.model().get_column('SNR')
        mean_snr = np.mean(snr_df.values)
        method_full_name = self.ui.fit_opt_method_comboBox.currentText()
        method = self.fitting_methods[method_full_name]
        max_dx = self.ui.max_dx_dsb.value()
        visible_lines = self.ui.deconv_lines_table.model().get_visible_line_types()
        func_legend = []
        params = Parameters()
        prev_k = []
        zero_under_05_hwhm_curve_k = {}
        if len(visible_lines) > 0 and not self.ui.interval_checkBox.isChecked():
            static_parameters = param_names, min_fwhm, self.peak_shape_params_limits
            func_legend, params, prev_k, zero_under_05_hwhm_curve_k = self.initial_guess(visible_lines, func,
                                                                                         static_parameters,
                                                                                         init_model_params)
        params_limits = self.peak_shape_params_limits
        params_limits['l_ratio'] = (0., self.ui.l_ratio_doubleSpinBox.value())
        return {'func': func, 'param_names': param_names, 'init_model_params': init_model_params, 'min_fwhm': min_fwhm,
                'method': method, 'params_limits': params_limits, 'mean_snr': mean_snr,
                'max_dx': max_dx, 'func_legend': func_legend, 'params': params, 'prev_k': prev_k,
                'zero_under_05_hwhm_curve_k': zero_under_05_hwhm_curve_k}

    def initial_guess(self, visible_lines: DataFrame, func, static_parameters: tuple[list[str], float, dict],
                      init_model_params: list[str]) -> tuple[list[tuple], Parameters, list[int], dict]:
        func_legend = []
        params = Parameters()
        prev_k = []
        x_axis = next(iter(self.baseline_corrected_dict.values()))[:, 0]
        max_dx = self.ui.max_dx_dsb.value()
        zero_under_05_hwhm_curve_k = {}
        for i in visible_lines.index:
            x0_series = self.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'x0'))
            a_series = self.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'a'))
            dx_series = self.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'dx'))
            legend = legend_by_float(x0_series.Value)
            func_legend.append((func, legend))
            init_params = init_model_params.copy()
            init_params[0] = a_series.Value
            init_params[1] = x0_series.Value
            dx_right = min(float(dx_series.Value), max_dx)
            if 'dx_left' in static_parameters[0]:
                dx_left_series = self.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'dx_left'))
                dx_left = min(dx_left_series.Value, max_dx)
            else:
                dx_left = dx_right
            prev_k.append(legend)
            y_max_in_range = a_series['Max value']
            x0_arg = find_nearest_idx(x_axis, x0_series.Value)
            x0_arg_dx_l = find_nearest_idx(x_axis, x0_series.Value - dx_left / 2.)
            x0_arg_dx_r = find_nearest_idx(x_axis, x0_series.Value + dx_right / 2.)
            zero_under_05_hwhm_curve_k[legend] = (x0_arg - x0_arg_dx_l, x0_arg_dx_r - x0_arg)
            dynamic_parameters = legend, init_params, y_max_in_range, dx_left, dx_right
            params = update_fit_parameters(params, static_parameters, dynamic_parameters)
        return func_legend, params, prev_k, zero_under_05_hwhm_curve_k

    def get_initial_parameters_for_line(self, line_type: str) -> dict[np.ndarray, float, float, float]:
        x_axis = np.array(range(920, 1080))
        a = 100.0
        x0 = 1000.0
        dx = 10.0
        arr = None
        if self.ui.template_combo_box.currentText() == 'Average':
            arr = self.averaged_array
            dx = np.max(self.ui.input_table.model().column_data(6)) * np.pi / 2
        elif self.current_spectrum_deconvolution_name != '':
            arr = self.baseline_corrected_dict[self.current_spectrum_deconvolution_name]
            row_data = self.ui.input_table.model().row_data_by_index(self.current_spectrum_deconvolution_name)
            dx = row_data['FWHM, cm\N{superscript minus}\N{superscript one}'] * np.pi / 2
        elif self.current_spectrum_deconvolution_name == '' and self.ui.template_combo_box.currentText() != 'Average':
            array_id = int(self.ui.template_combo_box.currentText().split('.')[0])
            arr = self.averaged_dict[array_id]
            dx = np.max(self.ui.input_table.model().column_data(6)) * np.pi / 2

        if arr is not None:
            x_axis = arr[:, 0]
            a = np.max(arr[:, 1]) / 2
            x0 = np.mean(x_axis)

        result = {'x_axis': x_axis,
                  'a': np.round(a, 5),
                  'x0': np.round(x0, 5),
                  'dx': np.round(dx, 5)}
        if 'add_params' not in self.peak_shapes_params[line_type]:
            return result
        add_params = self.peak_shapes_params[line_type]['add_params']
        for param_name in add_params:
            match param_name:
                case 'dx_left':
                    result[param_name] = np.round(dx - dx * 0.1, 5)
                case 'gamma':
                    result[param_name] = 0.5
                case 'skew':
                    result[param_name] = 0.0
                case 'l_ratio':
                    result[param_name] = 0.6
                case 'expon' | 'beta':
                    result[param_name] = 1.0
                case 'alpha' | 'q':
                    result[param_name] = 0.1
        return result

    def add_deconv_curve_to_plot(self, params: dict, idx: int, style: dict, line_type: str) \
            -> None:
        x0 = params['x0']
        dx = params['dx']
        if 'x_axis' not in params:
            params['x_axis'] = self.x_axis_for_line(x0, dx)
        x_axis = params['x_axis']
        full_amp_line, x_axis, _ = self.full_amp_line(line_type, params, x_axis, idx)
        self.create_roi_curve_add_to_plot(full_amp_line, x_axis, idx, style, params)

    def create_roi_curve_add_to_plot(self, full_amp_line: np.ndarray | None, x_axis: np.ndarray | None, idx: int,
                                     style: dict, params: dict) -> None:
        a = params['a']
        x0 = params['x0']
        dx = params['dx']
        if full_amp_line is None:
            return
        # n_array = get_deconv_curve_for_plot(full_amp_line, x_axis)
        n_array = np.vstack((x_axis, full_amp_line)).T
        curve = get_curve_for_deconvolution(n_array, idx, style)
        curve.sigClicked.connect(self.curve_clicked)
        self.deconvolution_plotItem.addItem(curve)
        roi = ROI([x0, 0], [dx, a], resizable=False, removable=True, rotatable=False, movable=False, pen='transparent')

        roi.addTranslateHandle([0, 1])
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        self.deconvolution_plotItem.addItem(roi)
        curve.setParentItem(roi)
        curve.setPos(-x0, 0)
        if not self.ui.deconv_lines_table.model().checked()[idx] and roi is not None:
            roi.setVisible(False)
        roi.sigRegionChangeStarted.connect(lambda checked=None, index=idx: self.curve_roi_pos_change_started(index,
                                                                                                             roi))
        roi.sigRegionChangeFinished.connect(
            lambda checked=None, index=idx: self.curve_roi_pos_change_finished(index, roi))
        roi.sigRegionChanged.connect(lambda checked=None, index=idx: self.curve_roi_pos_changed(index, roi, curve))

    def curve_clicked(self, curve: PlotCurveItem, _event: QMouseEvent) -> None:
        if self.updating_fill_curve_idx is not None:
            curve_style = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(self.updating_fill_curve_idx,
                                                                                       'Style')
            self.update_curve_style(self.updating_fill_curve_idx, curve_style)
        idx = int(curve.name())
        self.select_curve(idx)

    def select_curve(self, idx: int) -> None:
        row = self.ui.deconv_lines_table.model().row_by_index(idx)
        self.ui.deconv_lines_table.selectRow(row)
        self.set_rows_visibility()
        self.start_fill_timer(idx)

    def start_fill_timer(self, idx: int) -> None:
        self.rad = 0.
        self.updating_fill_curve_idx = idx
        self.timer_fill = QTimer()
        self.timer_fill.timeout.connect(self.update_curve_fill_realtime)
        self.timer_fill.start(10)

    def update_curve_fill_realtime(self):
        self.rad += 0.02
        idx = self.updating_fill_curve_idx
        if idx not in list(self.ui.deconv_lines_table.model().dataframe().index):
            self.deselect_selected_line()
            return
        sin_v = np.abs(np.sin(self.rad))
        if self.ui.deconv_lines_table.model().rowCount() == 0:
            return
        curve_style = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Style')
        fill_color = curve_style['color'] if curve_style['use_line_color'] else curve_style['fill_color']
        fill_color.setAlphaF(sin_v)
        brush = mkBrush(fill_color)
        curve, _ = self.deconvolution_data_items_by_idx(idx)
        curve.setBrush(brush)

    def deselect_selected_line(self) -> None:
        if self.timer_fill is not None:
            self.timer_fill.stop()
            if self.updating_fill_curve_idx is not None:
                curve_style = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(self.updating_fill_curve_idx,
                                                                                           'Style')
                self.update_curve_style(self.updating_fill_curve_idx, curve_style)
                self.updating_fill_curve_idx = None

    def delete_deconv_curve(self, idx: int) -> None:
        items_matches = self.deconvolution_data_items_by_idx(idx)
        if items_matches is None:
            return
        curve, roi = items_matches
        self.deconvolution_plotItem.removeItem(roi)
        self.deconvolution_plotItem.removeItem(curve)
        self.deconvolution_plotItem.getViewBox().updateAutoRange()

    def curve_roi_pos_change_started(self, index: int, roi: ROI) -> None:
        params = self.current_line_parameters(index)
        if not params:
            return
        a = params['a']
        x0 = params['x0']
        dx = params['dx']
        color = QColor(self.theme_colors['secondaryColor'])
        color.setAlphaF(0.5)
        roi.setPen(color)
        self.dragged_line_parameters = a, x0, dx

    def curve_roi_pos_change_finished(self, index: int, roi: ROI) -> None:
        params = self.current_line_parameters(index)
        if not params:
            return
        a = params['a']
        x0 = params['x0']
        dx = params['dx']
        roi_a = roi.size().y()
        roi_x0 = roi.pos().x()
        roi_dx = roi.size().x()

        if (a, x0, dx) != self.dragged_line_parameters and (a, x0, dx) != self.prev_dragged_line_parameters and \
                (a, x0, dx) != (roi_a, roi_x0, roi_dx) and self.CommandDeconvLineDraggedAllowed:
            command = CommandDeconvLineDragged(self, (a, x0, dx), self.dragged_line_parameters,
                                               roi, "Edit line %s" % index)
            roi.setPen('transparent')
            self.undoStack.push(command)
            self.prev_dragged_line_parameters = self.dragged_line_parameters
            self.dragged_line_parameters = a, x0, dx

    def current_line_parameters(self, index: int, filename: str | None = None) -> dict | None:
        if filename is None:
            filename = "" if self.isTemplate or self.current_spectrum_deconvolution_name == '' \
                else self.current_spectrum_deconvolution_name
        df_params = self.ui.fit_params_table.model().get_df_by_multiindex((filename, index))
        line_type = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(index, 'Type')
        if df_params.empty:
            return None
        return packed_current_line_parameters(df_params, line_type, self.peak_shapes_params)

    def current_filename_lines_parameters(self, indexes: list[int], filename, line_types: Series) -> dict | None:
        df_params = self.ui.fit_params_table.model().get_df_by_multiindex(filename)
        if df_params.empty:
            return None
        params = {}
        for idx in indexes:
            params[idx] = packed_current_line_parameters(df_params.loc[idx], line_types.loc[idx],
                                                         self.peak_shapes_params)
        return params

    def curve_roi_pos_changed(self, index: int, roi: ROI, curve: PlotCurveItem) -> None:
        dx = roi.size().x()
        x0 = roi.pos().x()
        new_height = roi.pos().y() + roi.size().y()
        params = self.current_line_parameters(index)

        if not params:
            return
        model = self.ui.fit_params_table.model()
        filename = '' if self.isTemplate else self.current_spectrum_deconvolution_name
        line_type = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(index, 'Type')

        if x0 < params['min_x0']:
            model.set_parameter_value(filename, index, 'x0', 'Min value', x0 - 1)
        if x0 > params['max_x0']:
            model.set_parameter_value(filename, index, 'x0', 'Max value', x0 + 1)

        if new_height < params['min_a']:
            print('curve_roi_pos_changed change values min_a', new_height, params['min_a'], index)
            model.set_parameter_value(filename, index, 'a', 'Min value', new_height)
        if new_height > params['max_a']:
            print('curve_roi_pos_changed change values max_a', new_height, params['max_a'])
            model.set_parameter_value(filename, index, 'a', 'Max value', new_height)

        if dx < params['min_dx']:
            model.set_parameter_value(filename, index, 'dx', 'Min value', dx - dx / 2)
        if dx > params['max_dx']:
            model.set_parameter_value(filename, index, 'dx', 'Max value', dx + dx / 2)

        if np.round(new_height, 5) == np.round(params['a'], 5) and np.round(dx, 5) == np.round(params['dx'], 5) \
                and np.round(params['x0'], 5) == np.round(x0, 5):
            return
        model.set_parameter_value(filename, index, 'dx', 'Value', dx)
        model.set_parameter_value(filename, index, 'a', 'Value', new_height)
        model.set_parameter_value(filename, index, 'x0', 'Value', x0)
        set_roi_size(roi.size().x(), new_height, roi)
        params = {'a': new_height, 'x0': x0, 'dx': dx}
        if 'add_params' not in self.peak_shapes_params[line_type]:
            self.redraw_curve(params, curve, line_type)
            return
        add_params = self.peak_shapes_params[line_type]['add_params']
        for param_name in add_params:
            params[param_name] = model.get_parameter_value(filename, index, param_name, 'Value')
        self.redraw_curve(params, curve, line_type)

    def x_axis_for_line(self, x0: float, dx: float) -> np.ndarray:
        if self.baseline_corrected_dict:
            return next(iter(self.baseline_corrected_dict.values()))[:, 0]
        else:
            return np.array(range(int(x0 - dx * 20), int(x0 + dx * 20)))

    def full_amp_line(self, line_type: str, params: dict | None, x_axis: np.ndarray = None, idx: int = 0) \
            -> tuple[np.ndarray, np.ndarray, int]:
        """Returns y-axis and x_axis of deconvolution line

        Inputs
        ------
        line_type : str
            name of line from self.deconv_line_params.keys()
        params : dict
            with parameters value - a, x0, dx and etc.
        x_axis : np.ndarray
            corresponding x_axis of y_axis line

        Returns
        -------
        out : tuple[np.ndarray, np.ndarray]
            y, x axis signal
        """
        a = params['a']
        dx = params['dx']
        x0 = params['x0']
        if x_axis is None:
            x_axis = self.x_axis_for_line(x0, dx)

        full_amp_line = None
        func = self.peak_shapes_params[line_type]['func']
        func_param = []
        if 'add_params' in self.peak_shapes_params[line_type]:
            add_params = self.peak_shapes_params[line_type]['add_params']
            for i in add_params:
                if i in params:
                    func_param.append(params[i])
        if not func_param:
            full_amp_line = func(x_axis, a, x0, dx)
        elif len(func_param) == 1:
            full_amp_line = func(x_axis, a, x0, dx, func_param[0])
        elif len(func_param) == 2:
            full_amp_line = func(x_axis, a, x0, dx, func_param[0], func_param[1])
        elif len(func_param) == 3:
            full_amp_line = func(x_axis, a, x0, dx, func_param[0], func_param[1], func_param[2])
        if self.ui.interval_checkBox.isChecked() and full_amp_line is not None:
            x_axis, idx_start, idx_end = cut_axis(x_axis, self.ui.interval_start_dsb.value(),
                                                  self.ui.interval_end_dsb.value())
            full_amp_line = full_amp_line[idx_start: idx_end + 1]
        return full_amp_line, x_axis, idx

    def show_hide_curve(self, idx: int, b: bool) -> None:
        """
        Find curves by name = idx, and setVisible for it by bool param
        """
        items_matches = self.deconvolution_data_items_by_idx(idx)
        if items_matches is None:
            return
        curve, roi = items_matches
        if curve is None or roi is None:
            return
        self.redraw_curve_by_index(idx)
        curve.setVisible(b)
        roi.setVisible(b)
        self.draw_sum_curve()
        self.draw_residual_curve()
        self.deconvolution_plotItem.getViewBox().updateAutoRange()

    @asyncSlot()
    async def curve_type_changed(self, line_type_new: str, line_type_old: str, row: int) -> None:
        """
        Find curves by name = idx, and redraw it
        add/delete parameters
        """
        idx = self.ui.deconv_lines_table.model().row_data(row).name
        await self.switch_to_template()
        command = CommandDeconvLineTypeChanged(self, line_type_new, line_type_old, idx,
                                               "Change line type for curve idx %s" % idx)
        self.undoStack.push(command)

    def deconvolution_data_items_by_idx(self, idx: int) -> tuple[PlotCurveItem, ROI]:
        data_items = self.deconvolution_plotItem.listDataItems()
        if len(data_items) == 0:
            return None
        curve = None
        roi = None
        for i in data_items:
            if i.name() == idx:
                curve = i
                roi = i.parentItem()
                break
        return curve, roi

    def deconvolution_data_items_by_indexes(self, indexes: list[int]) -> dict | None:
        result = {}
        data_items = self.deconvolution_plotItem.listDataItems()
        if len(data_items) == 0:
            return None
        for i in data_items:
            if i.name() in indexes:
                curve = i
                roi = i.parentItem()
                result[i.name()] = curve, roi
        return result

    def remove_all_lines_from_plot(self) -> None:
        data_items = self.deconvolution_plotItem.listDataItems()
        if len(data_items) == 0:
            return
        items_matches = (x for x in data_items if isinstance(x.name(), int))
        for i in items_matches:
            self.deconvolution_plotItem.removeItem(i.parentItem())
            self.deconvolution_plotItem.removeItem(i)
        self.deconvolution_plotItem.addItem(self.linearRegionDeconv)
        self.deconvolution_plotItem.addItem(self.fill)
        self.deconvolution_plotItem.getViewBox().updateAutoRange()

    def _update_deconv_curve_style(self, style: dict, old_style: dict, index: int) -> None:
        command = CommandUpdateDeconvCurveStyle(self, index, style, old_style, "Update style for curve idx %s" % index)
        self.undoStack.push(command)

    def update_curve_style(self, idx: int, style: dict) -> None:
        pen, brush = curve_pen_brush_by_style(style)
        items_matches = self.deconvolution_data_items_by_idx(idx)
        if items_matches is None:
            return
        curve, _ = items_matches
        curve.setPen(pen)
        curve.setBrush(brush)

    def update_template_combo_box(self) -> None:
        self.ui.template_combo_box.clear()
        self.ui.template_combo_box.addItem('Average')
        if not self.averaged_dict:
            return
        for i in self.averaged_dict:
            group_name = self.ui.GroupsTable.model().row_data(i - 1)['Group name']
            self.ui.template_combo_box.addItem(str(i) + '. ' + group_name)
        self.ui.template_combo_box.currentTextChanged.connect(self.switch_to_template)

    @asyncSlot()
    async def switch_to_template(self, _: str = 'Average') -> None:
        self.update_single_deconvolution_plot(self.ui.template_combo_box.currentText(), True, True)
        self.redraw_curves_for_filename()
        self.set_rows_visibility()
        self.show_current_report_result()
        self.draw_sum_curve()
        self.draw_residual_curve()
        self.update_sigma3_curves('')

    def array_of_current_filename_in_deconvolution(self) -> np.ndarray:
        """
        @return: 2D массив спектра, который отображается на графике в данный момент
        """
        current_spectrum_name = self.current_spectrum_deconvolution_name
        arr = None
        if self.isTemplate:
            if self.ui.template_combo_box.currentText() == 'Average':
                arr = self.averaged_array
            elif self.averaged_dict and self.averaged_dict != {}:
                arr = self.averaged_dict[int(self.ui.template_combo_box.currentText().split('.')[0])]
        elif current_spectrum_name in self.baseline_corrected_dict:
            arr = self.baseline_corrected_dict[current_spectrum_name]
        else:
            return None
        return arr

    def curve_parameter_changed(self, value: float, line_index: int, param_name: str) -> None:
        self.CommandDeconvLineDraggedAllowed = True
        items_matches = self.deconvolution_data_items_by_idx(line_index)
        if items_matches is None:
            return
        curve, roi = items_matches
        line_type = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(line_index, 'Type')
        if param_name == 'x0':
            new_pos = QPointF()
            new_pos.setX(value)
            new_pos.setY(roi.pos().y())
            roi.setPos(new_pos)
        elif param_name == 'a':
            set_roi_size(roi.size().x(), value, roi, [0, 0])
        elif param_name == 'dx':
            set_roi_size(value, roi.size().y(), roi)
        params = {'a': value if param_name == 'a' else roi.size().y(),
                  'x0': roi.pos().x(),
                  'dx': roi.size().x()}
        filename = '' if self.isTemplate else self.current_spectrum_deconvolution_name
        model = self.ui.fit_params_table.model()
        if 'add_params' not in self.peak_shapes_params[line_type]:
            self.redraw_curve(params, curve, line_type)
            return
        add_params = self.peak_shapes_params[line_type]['add_params']
        for s in add_params:
            if param_name == s:
                param = value
            else:
                param = model.get_parameter_value(filename, line_index, s, 'Value')
            params[s] = param
        self.redraw_curve(params, curve, line_type)

    def clear_all_deconv_lines(self) -> None:
        if self.timer_fill is not None:
            self.timer_fill.stop()
            self.timer_fill = None
            self.updating_fill_curve_idx = None
        command = CommandClearAllDeconvLines(self, 'Remove all deconvolution lines')
        self.undoStack.push(command)

    # region draw curve

    async def draw_all_curves(self) -> None:
        """
        Отрисовка всех линий
        @return: None
        """
        line_indexes = self.ui.deconv_lines_table.model().dataframe().index
        model = self.ui.deconv_lines_table.model()
        # executor = ThreadPoolExecutor()
        current_all_lines_parameters = self.all_lines_parameters(line_indexes)
        result = []
        for i in line_indexes:
            res = self.full_amp_line(model.cell_data_by_idx_col_name(i, 'Type'),
                                     current_all_lines_parameters[i], None, i)
            result.append(res)
        # with executor:
        #     self.current_futures = [
        #         loop.run_in_executor(executor, self.full_amp_line, model.cell_data_by_idx_col_name(i, 'Type'),
        #                              self.current_line_parameters(i), None, i)
        #         for i in line_indexes]
        #     result = await gather(*self.current_futures)

        for full_amp_line, x_axis, idx in result:
            self.create_roi_curve_add_to_plot(full_amp_line, x_axis, idx, model.cell_data_by_idx_col_name(idx, 'Style'),
                                              current_all_lines_parameters[idx])
        self.draw_sum_curve()
        self.draw_residual_curve()

    def all_lines_parameters(self, line_indexes: list[int]) -> list[dict] | None:
        filename = "" if self.isTemplate or self.current_spectrum_deconvolution_name == '' \
            else self.current_spectrum_deconvolution_name
        df_params = self.ui.fit_params_table.model().get_df_by_multiindex(filename)
        parameters = {}
        for idx in line_indexes:
            line_type = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
            a = df_params.loc[(idx, 'a')].Value
            x0 = df_params.loc[(idx, 'x0')].Value
            dx = df_params.loc[(idx, 'dx')].Value
            result = {'a': a, 'x0': x0, 'dx': dx}
            if 'add_params' not in self.peak_shapes_params[line_type]:
                parameters[idx] = result
                continue
            add_params = self.peak_shapes_params[line_type]['add_params']
            for param_name in add_params:
                result[param_name] = df_params.loc[(idx, param_name)].Value
            parameters[idx] = result
        return parameters

    def draw_sum_curve(self) -> None:
        """
            Update sum curve

            Returns
            -------
            out : None
        """
        if not self.baseline_corrected_dict or self.sum_curve is None:
            return
        x_axis, y_axis = self.sum_array()
        self.sum_curve.setData(x=x_axis, y=y_axis.T)

    def sum_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
                Return x, y arrays of sum spectra of all visible fit curves

                Returns
                -------
                out : tuple[np.ndarray, np.ndarray]
                    x_axis, y_axis of sum curve
        """
        x_axis = next(iter(self.baseline_corrected_dict.values()))[:, 0]
        if self.ui.interval_checkBox.isChecked():
            x_axis, _, _ = cut_axis(x_axis, self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value())
        data_items = self.deconvolution_plotItem.listDataItems()
        y_axis = np.zeros(x_axis.shape[0])

        for i in data_items:
            if isinstance(i, PlotCurveItem) and i.isVisible():
                x, y = i.getData()
                idx = find_nearest_idx(x_axis, x[0])
                y_z = np.zeros(x_axis.shape[0])
                if x_axis.shape[0] < y.shape[0]:
                    idx_right = x_axis.shape[0] - idx - 1
                    y_z[idx: idx + idx_right] += y[:idx_right]
                else:
                    y_z[idx: idx + y.shape[0]] += y
                y_axis += y_z
        return x_axis, y_axis

    def draw_residual_curve(self) -> None:
        """
            Update residual curve after sum curve updated.
            Residual = data - sum

            Returns
            -------
            out : None
        """
        if not self.baseline_corrected_dict or self.residual_curve is None:
            return
        x_axis, y_axis = self.residual_array()
        self.residual_curve.setData(x=x_axis, y=y_axis.T)

    def residual_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Return x, y arrays of residual spectra.
            Residual = data - sum

            Returns
            -------
            out : tuple[np.ndarray, np.ndarray]
                x_axis, y_axis of residual curve
        """
        x_data, y_data = self.data_curve.getData()
        x_sum, y_sum = self.sum_curve.getData()
        c = y_data.copy()
        c[:len(y_sum)] -= y_sum
        return x_data, c

    def redraw_curve(self, params: dict | None = None, curve: PlotCurveItem = None, line_type: str | None = None,
                     idx: int | None = None) -> None:
        if params is None and idx is not None:
            params = self.current_line_parameters(idx)
        elif params is None:
            return
        if curve is None and idx is not None:
            curve = self.deconvolution_data_items_by_idx(idx)[0]
        elif curve is None:
            return
        if line_type is None and idx is not None:
            line_type = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
        elif line_type is None:
            return
        x0 = params['x0']
        full_amp_line, x_axis, _ = self.full_amp_line(line_type, params, idx=idx)
        if full_amp_line is None:
            return
        # n_array = get_deconv_curve_for_plot(full_amp_line, x_axis)
        curve.setData(x=x_axis, y=full_amp_line)
        curve.setPos(-x0, 0)

    def redraw_curves_for_filename(self) -> None:
        """
        Redraw all curves by parameters of current selected spectrum
        """
        filename = "" if self.isTemplate or self.current_spectrum_deconvolution_name == '' \
            else self.current_spectrum_deconvolution_name
        line_indexes = self.ui.deconv_lines_table.model().get_visible_line_types().index
        filename_lines_indexes = self.ui.fit_params_table.model().get_lines_indexes_by_filename(filename)
        if filename_lines_indexes is None:
            return
        line_types = self.ui.deconv_lines_table.model().column_data(1)
        if line_types.empty:
            return None
        if len(line_indexes) != len(filename_lines_indexes):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Состав линий этого спектра отличается от шаблона. Некоторые линии не будут отрисованы "
                        "правильно.")
            msg.setWindowTitle("Template error")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            line_indexes = filename_lines_indexes
        params = self.current_filename_lines_parameters(list(line_indexes), filename, line_types)
        if params is None or not params:
            return
        items = self.deconvolution_data_items_by_indexes(list(line_indexes))
        if items is None or not items:
            return
        for i in line_indexes:
            set_roi_size_pos((params[i]['a'], params[i]['x0'], params[i]['dx']), items[i][1], False)
            self.redraw_curve(params[i], items[i][0], line_types.loc[i], i)

    def redraw_curve_by_index(self, idx: int, update: bool = True) -> None:
        params = self.current_line_parameters(idx)
        items = self.deconvolution_data_items_by_idx(idx)
        if items is None:
            return
        curve, roi = items
        if curve is None or roi is None:
            return
        line_type = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
        set_roi_size_pos((params['a'], params['x0'], params['dx']), roi, update)
        self.redraw_curve(params, curve, line_type, idx)

    def update_sigma3_curves(self, filename: str | None = None) -> None:
        """
        Update self.sigma3_top and self.sigma3_bottom for current spectrum
        @param filename: optional
        @return: None
        """
        if filename is None:
            filename = "" if self.isTemplate or self.current_spectrum_deconvolution_name == '' \
                else self.current_spectrum_deconvolution_name
        if filename not in self.sigma3:
            self.fill.setVisible(False)
            return
        self.fill.setVisible(self.ui.sigma3_checkBox.isChecked())

        self.sigma3_top.setData(x=self.sigma3[filename][0], y=self.sigma3[filename][1])
        self.sigma3_bottom.setData(x=self.sigma3[filename][0], y=self.sigma3[filename][2])

    # endregion

    # region copy template
    def copy_line_parameters_from_template(self, idx: int | None = None, filename: str | None = None,
                                           redraw: bool = True) -> None:
        filename = self.current_spectrum_deconvolution_name if filename is None else filename
        model = self.ui.fit_params_table.model()
        # find current index of selected line
        if idx is None:
            row = self.ui.deconv_lines_table.selectionModel().currentIndex().row()
            if row == -1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setText("Select line")
                msg.setWindowTitle("Line isn't selected")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
                return
            idx = self.ui.deconv_lines_table.model().row_data(row).name

            # delete line params by index
            model.delete_rows_multiindex((filename, idx))
        # add line params by from template
        mi = '', idx
        df = model.get_df_by_multiindex(mi)
        for i in range(len(df)):
            row_data = df.iloc[i]
            model.append_row(idx, row_data.Parameter, row_data.Value, row_data['Min value'], row_data['Max value'],
                             filename)
        if redraw:
            self.redraw_curve_by_index(idx)

    @asyncSlot()
    async def copy_spectrum_lines_parameters_from_template(self) -> None:
        selected_rows = self.ui.dec_table.selectionModel().selectedRows()
        if len(selected_rows) == 0:
            return
        selected_filename = self.ui.dec_table.model().cell_data_by_index(selected_rows[0])
        line_indexes = self.ui.deconv_lines_table.model().dataframe().index
        model = self.ui.fit_params_table.model()
        if len(line_indexes) == 0:
            return
        query_text = 'filename == ' + '"' + str(selected_filename) + '"'
        if not model.is_query_result_empty(query_text):
            model.delete_rows_by_multiindex(selected_filename)
        executor = ThreadPoolExecutor()
        with executor:
            self.current_futures = [
                loop.run_in_executor(executor, self.copy_line_parameters_from_template, i, selected_filename)
                for i in line_indexes]
            await gather(*self.current_futures)

    @asyncSlot()
    async def copy_template_to_all_spectra(self) -> None:
        filenames = self.ui.dec_table.model().column_data(0)
        line_indexes = self.ui.deconv_lines_table.model().dataframe().index
        if len(line_indexes) == 0 or len(filenames) == 0:
            return
        self.ui.fit_params_table.model().left_only_template_data()
        executor = ThreadPoolExecutor()
        with executor:
            self.current_futures = [loop.run_in_executor(executor, self.copy_line_parameters_from_template, i, j, False)
                                    for i in line_indexes for j in filenames]
            await gather(*self.current_futures)
        for i in line_indexes:
            self.redraw_curve_by_index(idx=i)

    # endregion

    # endregion

    # region STAT ANALYSIS
    @asyncSlot()
    async def fit_classificator(self, cl_type: str):
        """
        Проверяем что датасет заполнен.
        Переходим в процедуру создания классификатора
        """
        current_dataset = self.ui.dataset_type_cb.currentText()
        if current_dataset == 'Smoothed' and self.ui.smoothed_dataset_table_view.model().rowCount() == 0 \
                or current_dataset == 'Baselined corrected' \
                and self.ui.baselined_dataset_table_view.model().rowCount() == 0 \
                or current_dataset == 'Deconvoluted' \
                and self.ui.deconvoluted_dataset_table_view.model().rowCount() == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Нет данных для обучения классификатора")
            msg.setWindowTitle("Classificator Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        try:
            await self.do_fit_classificator(cl_type)
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def do_fit_classificator(self, cl_type: str) -> None:
        """
        Обучение выбранного классификатора, обновление всех графиков.
        @param cl_type: название метода классификации
        @return: None
        """
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Fitting model...')
        self.close_progress_bar()
        self._open_progress_bar(max_value=0)
        self._open_progress_dialog("Fitting...", "Cancel", maximum=0)
        X, Y, feature_names, target_names, _ = self.dataset_for_ml()
        y_test_bin = None
        test_size = self.ui.test_data_ratio_spinBox.value() / 100.
        if len(target_names) > 2:
            Y_bin = label_binarize(Y, classes=list(np.unique(Y)))
            _, _, _, y_test_bin = train_test_split(X, Y_bin, test_size=test_size)
        # rng = np.random.RandomState(0)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        executor = ThreadPoolExecutor()
        func = self.classificator_funcs[cl_type]
        # Для БОльших датасетов возможно лучше будет ProcessPoolExecutor. Но таких пока нет
        self.current_executor = executor
        with executor:
            self.current_futures = [loop.run_in_executor(executor, func, x_train, y_train, x_test, y_test)]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Fitting cancelled.')
            return
        result = result[0]
        result['X'] = X
        result['y_train'] = y_train
        result['y_test'] = y_test
        result['x_train'] = x_train
        result['x_test'] = x_test
        result['target_names'] = target_names
        result['feature_names'] = feature_names
        result['y_test_bin'] = y_test_bin
        command = CommandAfterFittingStat(self, result, cl_type, "Fit model %s" % cl_type)
        self.undoStack.push(command)
        self.close_progress_bar()
        self.ui.statusBar.showMessage('Fitting completed', 10000)

    def dataset_for_ml(self) -> tuple[DataFrame, list[int], list[str], np.ndarray]:
        """
        Выбор данных для обучения из датасета
        @return:
        X: DataFrame. Columns - fealures, rows - samples
        Y: list[int]. True labels
        feature_names: feature names (lol)
        target_names: classes names
        """
        if self.ui.dataset_type_cb.currentText() == 'Smoothed':
            model = self.ui.smoothed_dataset_table_view.model()
        elif self.ui.dataset_type_cb.currentText() == 'Baseline corrected':
            model = self.ui.baselined_dataset_table_view.model()
        elif self.ui.dataset_type_cb.currentText() == 'Deconvoluted':
            model = self.ui.deconvoluted_dataset_table_view.model()
        else:
            return
        q_res = model.dataframe()
        if self.ui.classes_lineEdit.text() != '':
            v = list(self.ui.classes_lineEdit.text().strip().split(','))
            classes = []
            for i in v:
                classes.append(int(i))
            if len(classes) > 1:
                q_res = model.query_result_with_list('Class == @input_list', classes)
        y = list(q_res['Class'])
        classes = np.unique(q_res['Class'].values)
        if self.is_production_project:
            target_names = None
        else:
            target_names = self.ui.GroupsTable.model().dataframe().loc[classes]['Group name'].values
        return q_res.iloc[:, 2:], y, list(q_res.axes[1][2:]), target_names, q_res.iloc[:, 1]

    @asyncSlot()
    async def redraw_stat_plots(self) -> None:
        await self.loop.run_in_executor(None, self.update_lda_plots)
        await self.loop.run_in_executor(None, self.update_qda_plots)
        await self.loop.run_in_executor(None, self.update_lr_plots)
        await self.loop.run_in_executor(None, self.update_svc_plots)
        await self.loop.run_in_executor(None, self.update_nearest_plots)
        await self.loop.run_in_executor(None, self.update_gpc_plots)
        await self.loop.run_in_executor(None, self.update_dt_plots)
        await self.loop.run_in_executor(None, self.update_nb_plots)
        await self.loop.run_in_executor(None, self.update_rf_plots)
        await self.loop.run_in_executor(None, self.update_ab_plots)
        await self.loop.run_in_executor(None, self.update_mlp_plots)
        await self.loop.run_in_executor(None, self.update_pca_plots)
        await self.loop.run_in_executor(None, self.update_plsda_plots)
        self.update_stat_report_text()
        self.update_force_single_plots()
        self.update_force_full_plots()

    def update_stat_report_text(self):
        if self.ui.stat_tab_widget.currentIndex() == 0:
            classificator_type = 'LDA'
        elif self.ui.stat_tab_widget.currentIndex() == 1:
            classificator_type = 'QDA'
        elif self.ui.stat_tab_widget.currentIndex() == 2:
            classificator_type = 'Logistic regression'
        elif self.ui.stat_tab_widget.currentIndex() == 3:
            classificator_type = 'NuSVC'
        elif self.ui.stat_tab_widget.currentIndex() == 4:
            classificator_type = 'Nearest Neighbors'
        elif self.ui.stat_tab_widget.currentIndex() == 5:
            classificator_type = 'GPC'
        elif self.ui.stat_tab_widget.currentIndex() == 6:
            classificator_type = 'Decision Tree'
        elif self.ui.stat_tab_widget.currentIndex() == 7:
            classificator_type = 'Naive Bayes'
        elif self.ui.stat_tab_widget.currentIndex() == 8:
            classificator_type = 'Random Forest'
        elif self.ui.stat_tab_widget.currentIndex() == 9:
            classificator_type = 'AdaBoost'
        elif self.ui.stat_tab_widget.currentIndex() == 10:
            classificator_type = 'MLP'
        else:
            self.ui.stat_report_text_edit.setText('')
            return
        if classificator_type not in self.latest_stat_result or 'metrics_result' \
                not in self.latest_stat_result[classificator_type]:
            self.ui.stat_report_text_edit.setText('')
            return
        if classificator_type in self.top_features:
            top = self.top_features[classificator_type]
        else:
            top = None
        model_results = self.latest_stat_result[classificator_type]
        self.update_report_text(model_results['metrics_result'], model_results['cv_scores'], top,
                                model_results['model'])

    def update_force_single_plots(self, clas_type: str = '') -> None:
        cl_types = [('LDA', self.ui.lda_force_single), ('Logistic regression', self.ui.lr_force_single),
                    ('NuSVC', self.ui.svc_force_single), ('Nearest Neighbors', self.ui.nearest_force_single),
                    ('GPC', self.ui.gpc_force_single), ('Decision Tree', self.ui.dt_force_single),
                    ('Naive Bayes', self.ui.nb_force_single), ('Random Forest', self.ui.rf_force_single),
                    ('AdaBoost', self.ui.ab_force_single), ('MLP', self.ui.mlp_force_single)]
        for cl_type, plot_widget in cl_types:
            if clas_type != '' and clas_type != cl_type:
                continue
            if cl_type not in self.latest_stat_result:
                continue
            if 'shap_html' in self.latest_stat_result[cl_type]:
                shap_html = self.latest_stat_result[cl_type]['shap_html']
                if self.ui.sun_Btn.isChecked():
                    shap_html = re.sub(r'#ffe', "#000", shap_html)
                    shap_html = re.sub(r'#001', "#fff", shap_html)
                else:
                    shap_html = re.sub(r'#000', "#ffe", shap_html)
                    shap_html = re.sub(r'#fff', "#001", shap_html)
                plot_widget.setHtml(shap_html)
                plot_widget.page().setBackgroundColor(QColor(self.plot_background_color))

    def update_force_full_plots(self, clas_type: str = '') -> None:
        cl_types = [('LDA', self.ui.lda_force_full), ('Logistic regression', self.ui.lr_force_full),
                    ('NuSVC', self.ui.svc_force_full), ('Nearest Neighbors', self.ui.nearest_force_full),
                    ('GPC', self.ui.gpc_force_full), ('Decision Tree', self.ui.dt_force_full),
                    ('Naive Bayes', self.ui.nb_force_full), ('Random Forest', self.ui.rf_force_full),
                    ('AdaBoost', self.ui.ab_force_full), ('MLP', self.ui.mlp_force_full)]
        for cl_type, plot_widget in cl_types:
            if clas_type != '' and clas_type != cl_type:
                continue
            if cl_type not in self.latest_stat_result:
                continue
            if 'shap_html_full' in self.latest_stat_result[cl_type]:
                shap_html = self.latest_stat_result[cl_type]['shap_html_full']
                if self.ui.sun_Btn.isChecked():
                    shap_html = re.sub(r'#ffe', "#000", shap_html)
                    shap_html = re.sub(r'#001', "#fff", shap_html)
                else:
                    shap_html = re.sub(r'#000', "#ffe", shap_html)
                    shap_html = re.sub(r'#fff', "#001", shap_html)
                plot_widget.setHtml(shap_html)
                plot_widget.page().setBackgroundColor(QColor(self.plot_background_color_web))

    def _get_plot_colors(self, classes: list[int]) -> list[str]:
        plt_colors = []
        for cls in classes:
            clr = self.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter().name())
        return plt_colors

    def update_pr_plot_bin(self, y_score_dec_func, y_test, pos_label: int, plot_widget, name=None) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        if len(y_score_dec_func.shape) > 1 and y_score_dec_func.shape[1] > 1:
            y_score_dec_func = y_score_dec_func[:, 0]
        PrecisionRecallDisplay.from_predictions(y_test, y_score_dec_func, name=name, ax=ax, color='darkorange',
                                                pos_label=pos_label)
        # display = PrecisionRecallDisplay.from_estimator(
        #     classifier, X_test, y_test, name="LinearSVC"
        # )
        ax.set_title("2-class Precision-Recall curve")
        ax.legend(loc="best", prop={'size': 8})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_pr_plot(self, classes: list[int], Y_test, y_score, colors, plot_widget) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        # The average precision score in multi-label settings
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        n_classes = len(classes)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
        display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"],
                                         average_precision=average_precision["micro"])
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i],
                                             average_precision=average_precision[i], )
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)
        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best", prop={'size': 8})
        ax.set_title("Extension of Precision-Recall curve to multi-class")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_shap_means_plot(self, binary: bool, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        print('update_shap_means_plot ', classificator_type)
        if classificator_type == 'LDA':
            plot_widget = self.ui.lda_shap_means
        elif classificator_type == 'QDA':
            plot_widget = self.ui.qda_shap_means
        elif classificator_type == 'Logistic regression':
            plot_widget = self.ui.lr_shap_means
        elif classificator_type == 'NuSVC':
            plot_widget = self.ui.svc_shap_means
        elif classificator_type == 'Nearest Neighbors':
            plot_widget = self.ui.nearest_shap_means
        elif classificator_type == 'GPC':
            plot_widget = self.ui.gpc_shap_means
        elif classificator_type == 'Decision Tree':
            plot_widget = self.ui.dt_shap_means
        elif classificator_type == 'Naive Bayes':
            plot_widget = self.ui.nb_shap_means
        elif classificator_type == 'Random Forest':
            plot_widget = self.ui.rf_shap_means
        elif classificator_type == 'AdaBoost':
            plot_widget = self.ui.ab_shap_means
        elif classificator_type == 'MLP':
            plot_widget = self.ui.mlp_shap_means
        else:
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

        if classificator_type not in self.latest_stat_result:
            return
        result = self.latest_stat_result[classificator_type]
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
        shap.plots.bar(shap_values, show=False, max_display=20, ax=fig.gca(), fig=fig)
        self.set_canvas_colors(plot_widget.canvas)
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

    def update_shap_beeswarm_plot(self, binary: bool, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type == 'LDA':
            plot_widget = self.ui.lda_shap_beeswarm
        elif classificator_type == 'QDA':
            plot_widget = self.ui.qda_shap_beeswarm
        elif classificator_type == 'Logistic regression':
            plot_widget = self.ui.lr_shap_beeswarm
        elif classificator_type == 'NuSVC':
            plot_widget = self.ui.svc_shap_beeswarm
        elif classificator_type == 'Nearest Neighbors':
            plot_widget = self.ui.nearest_shap_beeswarm
        elif classificator_type == 'GPC':
            plot_widget = self.ui.gpc_shap_beeswarm
        elif classificator_type == 'Decision Tree':
            plot_widget = self.ui.dt_shap_beeswarm
        elif classificator_type == 'Naive Bayes':
            plot_widget = self.ui.nb_shap_beeswarm
        elif classificator_type == 'Random Forest':
            plot_widget = self.ui.rf_shap_beeswarm
        elif classificator_type == 'AdaBoost':
            plot_widget = self.ui.ab_shap_beeswarm
        elif classificator_type == 'MLP':
            plot_widget = self.ui.mlp_shap_beeswarm
        else:
            return
        if self.plt_style is None:
            plt.style.use(['dark_background'])
        else:
            plt.style.use(self.plt_style)
        if self.ui.sun_Btn.isChecked():
            color = None
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
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.beeswarm(shap_values, show=False, color=color, max_display=20, ax=fig.gca(), fig=fig)
        self.set_canvas_colors(plot_widget.canvas)
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

    def update_shap_scatter_plot(self, binary: bool, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type == 'LDA':
            plot_widget = self.ui.lda_shap_scatter
        elif classificator_type == 'QDA':
            plot_widget = self.ui.qda_shap_scatter
        elif classificator_type == 'Logistic regression':
            plot_widget = self.ui.lr_shap_scatter
        elif classificator_type == 'NuSVC':
            plot_widget = self.ui.svc_shap_scatter
        elif classificator_type == 'Nearest Neighbors':
            plot_widget = self.ui.nearest_shap_scatter
        elif classificator_type == 'GPC':
            plot_widget = self.ui.gpc_shap_scatter
        elif classificator_type == 'Decision Tree':
            plot_widget = self.ui.dt_shap_scatter
        elif classificator_type == 'Naive Bayes':
            plot_widget = self.ui.nb_shap_scatter
        elif classificator_type == 'Random Forest':
            plot_widget = self.ui.rf_shap_scatter
        elif classificator_type == 'AdaBoost':
            plot_widget = self.ui.ab_shap_scatter
        elif classificator_type == 'MLP':
            plot_widget = self.ui.mlp_shap_scatter
        else:
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
        current_feature = self.ui.current_feature_comboBox.currentText()
        cmap = None if self.ui.sun_Btn.isChecked() else plt.get_cmap("cool")
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if binary:
            class_i = 0
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        if current_feature not in shap_values.feature_names:
            return
        ct = self.ui.coloring_feature_comboBox.currentText()
        color = shap_values if ct == '' else shap_values[:, ct]
        shap.plots.scatter(shap_values[:, current_feature], color=color, show=False, cmap=cmap, ax=fig.gca(),
                           axis_color=self.plot_text_color.name())
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

    def update_shap_heatmap_plot(self, binary: bool, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type == 'LDA':
            plot_widget = self.ui.lda_shap_heatmap
        elif classificator_type == 'QDA':
            plot_widget = self.ui.qda_shap_heatmap
        elif classificator_type == 'Logistic regression':
            plot_widget = self.ui.lr_shap_heatmap
        elif classificator_type == 'NuSVC':
            plot_widget = self.ui.svc_shap_heatmap
        elif classificator_type == 'Nearest Neighbors':
            plot_widget = self.ui.nearest_shap_heatmap
        elif classificator_type == 'GPC':
            plot_widget = self.ui.gpc_shap_heatmap
        elif classificator_type == 'Decision Tree':
            plot_widget = self.ui.dt_shap_heatmap
        elif classificator_type == 'Naive Bayes':
            plot_widget = self.ui.nb_shap_heatmap
        elif classificator_type == 'Random Forest':
            plot_widget = self.ui.rf_shap_heatmap
        elif classificator_type == 'AdaBoost':
            plot_widget = self.ui.ab_shap_heatmap
        elif classificator_type == 'MLP':
            plot_widget = self.ui.mlp_shap_heatmap
        else:
            return
        if self.plt_style is None:
            plt.style.use(['dark_background'])
        else:
            plt.style.use(self.plt_style)
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
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.heatmap(shap_values, show=False, max_display=20, ax=fig.gca(), fig=fig)
        self.set_canvas_colors(plot_widget.canvas)
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

    def update_shap_force(self, class_i: int = 0, classificator_type: str = 'LDA', full: bool = False) -> None:
        if classificator_type == 'QDA':
            return
        if classificator_type not in self.latest_stat_result:
            return
        model = self.get_current_dataset_type_cb()
        if model is None:
            return

        current_instance = self.ui.current_instance_combo_box.currentText()
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
        X_display = self.ui.deconvoluted_dataset_table_view.model().dataframe()
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
        if (not full and shap_v.shape[0] != len(x_d.values)) or (full and shap_v[0].shape[0] != len(x_d.loc[0].values)):
            err = 'Force plot не смог обновиться. Количество shap_values features != количеству X features.' \
                  ' Возможно была изменена таблица с обучающими данными.' \
                  ' Нужно пересчитать %s' % classificator_type
            print(err)
            error(err)
            return
        force_plot = shap.force_plot(expected_value, shap_v, x_d, feature_names=result['feature_names'])

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        if full:
            self.latest_stat_result[classificator_type]['shap_html_full'] = shap_html
        else:
            self.latest_stat_result[classificator_type]['shap_html'] = shap_html

    def update_shap_decision_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type not in self.latest_stat_result:
            return
        if classificator_type == 'Logistic regression':
            plot_widget = self.ui.lr_shap_decision
        elif classificator_type == 'LDA':
            plot_widget = self.ui.lda_shap_decision
        elif classificator_type == 'NuSVC':
            plot_widget = self.ui.svc_shap_decision
        elif classificator_type == 'Nearest Neighbors':
            plot_widget = self.ui.nearest_shap_decision
        elif classificator_type == 'GPC':
            plot_widget = self.ui.gpc_shap_decision
        elif classificator_type == 'Decision Tree':
            plot_widget = self.ui.dt_shap_decision
        elif classificator_type == 'Naive Bayes':
            plot_widget = self.ui.nb_shap_decision
        elif classificator_type == 'Random Forest':
            plot_widget = self.ui.rf_shap_decision
        elif classificator_type == 'AdaBoost':
            plot_widget = self.ui.ab_shap_decision
        elif classificator_type == 'MLP':
            plot_widget = self.ui.mlp_shap_decision
        else:
            return
        model = self.get_current_dataset_type_cb()
        if model is None:
            return
        current_instance = self.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = None
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
        X_display = self.ui.deconvoluted_dataset_table_view.model().dataframe()
        misclassified = result['y_train_plus_test'] != result['y_pred']
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
        if (not sample_id is None and shap_v.shape[0] != len(x_d.values)) \
                or (sample_id is None and shap_v[0].shape[0] != len(x_d.loc[0].values)):
            err = 'Decision plot не смог обновиться. Количество shap_values features != количеству X features.' \
                  ' Возможно была изменена таблица с обучающими данными.' \
                  ' Нужно пересчитать %s' % classificator_type
            print(err)
            error(err)
            return
        shap.plots.decision(expected_value, shap_v, x_d, title=title,
                            feature_display_range=slice(None, None, -1), highlight=misclassified, ax=ax, fig=fig)
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

    def update_shap_waterfall_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type not in self.latest_stat_result:
            return
        if classificator_type == 'Logistic regression':
            plot_widget = self.ui.lr_shap_waterfall
        elif classificator_type == 'LDA':
            plot_widget = self.ui.lda_shap_waterfall
        elif classificator_type == 'QDA':
            plot_widget = self.ui.qda_shap_waterfall
        elif classificator_type == 'NuSVC':
            plot_widget = self.ui.svc_shap_waterfall
        elif classificator_type == 'Nearest Neighbors':
            plot_widget = self.ui.nearest_shap_waterfall
        elif classificator_type == 'GPC':
            plot_widget = self.ui.gpc_shap_waterfall
        elif classificator_type == 'Decision Tree':
            plot_widget = self.ui.dt_shap_waterfall
        elif classificator_type == 'Naive Bayes':
            plot_widget = self.ui.nb_shap_waterfall
        elif classificator_type == 'Random Forest':
            plot_widget = self.ui.rf_shap_waterfall
        elif classificator_type == 'AdaBoost':
            plot_widget = self.ui.ab_shap_waterfall
        elif classificator_type == 'MLP':
            plot_widget = self.ui.mlp_shap_waterfall
        else:
            return
        ct = self.ui.dataset_type_cb.currentText()
        if ct == 'Smoothed':
            model = self.ui.smoothed_dataset_table_view.model()
        elif ct == 'Baseline corrected':
            model = self.ui.baselined_dataset_table_view.model()
        elif ct == 'Deconvoluted':
            model = self.ui.deconvoluted_dataset_table_view.model()
        else:
            return
        if model.rowCount() == 0:
            return
        q_res = model.dataframe()
        features_names = list(q_res.columns[2:])
        n_features = len(features_names)
        model = self.get_current_dataset_type_cb()
        if model is None:
            print('model is none')
            return
        current_instance = self.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = 0
        else:
            sample_id = model.idx_by_column_value('Filename', current_instance)
        result = self.latest_stat_result[classificator_type]
        if 'model' not in result:
            print('model not in result')
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
        shap.plots.waterfall(shap_values[sample_id], n_features, ax=ax, fig=fig)
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

    def update_features_plot(self, X_train: DataFrame, model, feature_names: list[str], target_names: list[str],
                             plt_colors, plot_widget) -> str:
        """
        Update features plot

        @param plt_colors:
        @param X_train: dataframe with train data
        @param model: model
        @param feature_names: name for features
        @param target_names: classes name
        @return: top features per class
        """

        # learned coefficients weighted by frequency of appearance
        feature_names = np.array(feature_names)
        if isinstance(model, GridSearchCV):
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
        bar_size = 0.25
        padding = 0.75
        y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)
        ax = plot_widget.canvas.axes
        ax.cla()
        for i, label in enumerate(target_names):
            if i >= average_feature_effects.shape[0]:
                break
            ax.barh(y_locs + (i - 2) * bar_size, average_feature_effects[i, top_indices], height=bar_size, label=label,
                    color=plt_colors[i])
        ax.set(yticks=y_locs, yticklabels=predictive_words,
               ylim=[0 - 4 * bar_size, len(top_indices) * (4 * bar_size + padding) - 4 * bar_size])
        ax.legend(loc="best")
        ax.set_title("Average feature effect on the original data")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()
        return top

    def update_dm_plot(self, y_test, pred, target_names, model, plot_widget) -> None:
        plot_widget.canvas.axes.cla()
        ax = plot_widget.canvas.axes
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        try:
            ax.xaxis.set_ticklabels(target_names)
            ax.yaxis.set_ticklabels(target_names)
        except ValueError as err:
            error(err)
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        _ = ax.set_title(f"Confusion Matrix for {model.__class__.__name__}\non the original data")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_roc_plot(self, y_score, y_onehot_test, target_names, plt_colors, plot_widget) -> None:
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
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)", color=self.theme_colors['primaryColor'])
        for class_id, color in zip(range(n_classes), plt_colors):
            RocCurveDisplay.from_predictions(y_onehot_test[:, class_id], y_score[:, class_id],
                                             name=target_names[class_id], color=color, ax=ax, )
        ax.axis("square")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        ax.legend(loc="best", prop={'size': 8})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_roc_plot_bin(self, y_score, y_onehot_test, target_names, plot_widget) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)", color=self.theme_colors['primaryColor'])
        RocCurveDisplay.from_predictions(y_onehot_test[:, 0], y_score[:, 0],
                                         name=target_names[0], color='darkorange', ax=ax, )
        ax.axis("square")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) curve")
        ax.legend(loc="best", prop={'size': 8})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # region LDA
    def update_lda_plots(self) -> None:
        """
        Update all LDA plots and fields
        @return: None
        """
        if 'LDA' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['LDA']
        model = model_results['model']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        y_score_dec_func = model_results['y_score_dec_func']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)
        binary = len(classes) == 2
        if model_results['features_in_2d'].shape[1] == 1:
            self.update_lda_scores_plot_1d(model.classes_, y_train_plus_test, features_in_2d)
        else:
            self.update_lda_scores_plot_2d(features_in_2d, y_train_plus_test, model_results['y_pred'], model,
                                           model_results['model_2d'])
        if binary:
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.lda_roc_plot)
            self.update_pr_plot_bin(y_score_dec_func, y_test, classes[0], self.ui.lda_pr_plot, 'LDA')
        else:
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.lda_roc_plot)
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score_dec_func, plt_colors, self.ui.lda_pr_plot)
        top = self.update_features_plot(model_results['x_train'], model, model_results['feature_names'],
                                        target_names, plt_colors, self.ui.lda_features_plot_widget)
        self.top_features['LDA'] = top
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.lda_dm_plot)
        self.do_update_shap_plots('LDA')
        self.do_update_shap_plots_by_instance('LDA')

    def update_lda_scores_plot_1d(self, classes: np.ndarray, y: list[int], features_in_2d) -> None:
        self.ui.lda_scores_1d_plot_widget.setVisible(True)
        self.ui.lda_scores_2d_plot_widget.setVisible(False)

        items_matches = self.lda_scores_1d_plot_item.listDataItems()
        for i in items_matches:
            self.lda_scores_1d_plot_item.removeItem(i)
        for i in self.lda_1d_inf_lines:
            self.lda_scores_1d_plot_item.removeItem(i)
        self.lda_1d_inf_lines = []
        min_scores = int(np.round(np.min(features_in_2d))) - 1
        max_scores = int(np.round(np.max(features_in_2d))) + 1
        rng = int((max_scores - min_scores) / .1)
        bottom = np.zeros(rng - 1)
        means = []
        for i in classes:
            scores_class_i = []
            for j, score in enumerate(features_in_2d):
                if y[j] == i:
                    scores_class_i.append(score)
            hist_y, hist_x = np.histogram(scores_class_i, bins=np.linspace(min_scores, max_scores, rng))
            centroid = np.mean(scores_class_i)
            means.append(centroid)
            pen_color = self.theme_colors['plotText']
            brush = self.ui.GroupsTable.model().cell_data_by_idx_col_name(i, 'Style')['color']
            bgi = BarGraphItem(x0=hist_x[:-1], x1=hist_x[1:], y0=bottom, height=hist_y, pen=pen_color, brush=brush)
            bottom += hist_y
            inf_line = InfiniteLine(centroid, pen=QColor(brush))

            self.ui.lda_scores_1d_plot_widget.addItem(bgi)
            if not np.isnan(centroid):
                self.lda_scores_1d_plot_item.addItem(inf_line)
                self.lda_1d_inf_lines.append(inf_line)
        if len(self.lda_1d_inf_lines) > 1:
            inf_line_mean = InfiniteLine(np.mean(means), pen=QColor(self.theme_colors['inverseTextColor']))
            self.lda_scores_1d_plot_item.addItem(inf_line_mean)
            self.lda_1d_inf_lines.append(inf_line_mean)

    def update_lda_scores_plot_2d(self, features_in_2d: np.ndarray, y: list[int], y_pred: list[int], model,
                                  model_2d) -> None:
        """
        @param features_in_2d: transformed 2d
        @param y: true labels
        @param y_pred: predicted labels
        @param model: classificator
        @param model_2d: 2d model classificator
        @param explained_variance_ratio:
        @return:
        """
        self.ui.lda_scores_2d_plot_widget.canvas.axes.cla()
        self.ui.lda_scores_1d_plot_widget.setVisible(False)
        self.ui.lda_scores_2d_plot_widget.setVisible(True)
        explained_variance_ratio = model.best_estimator_.explained_variance_ratio_
        classes = model.classes_
        plt_colors = []
        for cls in classes:
            clr = self.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter().name())
        tp = y == y_pred  # True Positive
        cmap = ListedColormap(plt_colors)
        DecisionBoundaryDisplay.from_estimator(model_2d, features_in_2d, grid_resolution=1000, alpha=0.75, eps=.5,
                                               antialiased=True, cmap=cmap,
                                               ax=self.ui.lda_scores_2d_plot_widget.canvas.axes)
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*']
        for i, cls in enumerate(classes):
            true_positive_of_class = tp[y == cls]
            x_i = features_in_2d[y == cls]
            x_tp = x_i[true_positive_of_class]
            x_fp = x_i[~true_positive_of_class]
            mrkr = markers[cls]
            color = self.ui.GroupsTable.model().cell_data_by_idx_col_name(cls, 'Style')['color'].name()
            # inverted_color = invert_color(color)
            self.ui.lda_scores_2d_plot_widget.canvas.axes.scatter(x_tp[:, 0], x_tp[:, 1], marker=mrkr,
                                                                  color=color,
                                                                  edgecolor='black', s=30)
            self.ui.lda_scores_2d_plot_widget.canvas.axes.scatter(x_fp[:, 0], x_fp[:, 1], marker="x", s=30,
                                                                  color=color)
        self.ui.lda_scores_2d_plot_widget.canvas.axes.set_xlabel('LD-1 (%.2f%%)' % (explained_variance_ratio[0] * 100))
        self.ui.lda_scores_2d_plot_widget.canvas.axes.set_ylabel('LD-2 (%.2f%%)' % (explained_variance_ratio[1] * 100))
        try:
            self.ui.lda_scores_2d_plot_widget.canvas.draw()
        except ValueError:
            pass
        self.ui.lda_scores_2d_plot_widget.canvas.figure.tight_layout()

    def update_report_text(self, metrics_result: dict, cv_scores=None, top=None, model=None) -> None:
        """
        Set report text
        @param cv_scores:
        @param metrics_result:
        @param top: top 5 features for each class
        @return:
        """
        text = '\n' + 'Accuracy score (test data): {!s}%'.format(metrics_result['accuracy_score']) + '\n' \
               + 'Accuracy score (train data): {!s}%'.format(metrics_result['accuracy_score_train']) + '\n' \
               + 'Precision score: {!s}%'.format(metrics_result['precision_score']) + '\n' \
               + 'Recall score: {!s}%'.format(metrics_result['recall_score']) + '\n' \
               + 'F1 score: {!s}%'.format(metrics_result['f1_score']) + '\n' \
               + 'F_beta score: {!s}%'.format(metrics_result['fbeta_score']) + '\n' \
               + 'Hamming loss score: {!s}%'.format(metrics_result['hamming_loss']) + '\n' \
               + 'Jaccard score: {!s}%'.format(metrics_result['jaccard_score']) + '\n' + '\n'
        if top is not None:
            text += 'top 5 features per class:' + '\n' + str(top) + '\n'
        if cv_scores is not None:
            text += '\n' + "Cross validated %0.2f accuracy with a standard deviation of %0.2f" \
                    % (cv_scores.mean(), cv_scores.std()) + '\n'
        if model is not None and isinstance(model, GridSearchCV):
            text += '\n' + 'Mean Accuracy of best estimator: %.3f' % model.best_score_ + '\n'
            text += 'Config: %s' % model.best_params_ + '\n'
        self.ui.stat_report_text_edit.setText(text)
        if metrics_result['classification_report'] is None:
            return
        headers = [' ']
        rows = []
        for i in metrics_result['classification_report'].split('\n')[0].strip().split(' '):
            if i != '':
                headers.append(i)
        for i, r in enumerate(metrics_result['classification_report'].split('\n')):
            new_row = []
            if r == '' or i == 0:
                continue
            rr = r.split('  ')
            for c in rr:
                if c == '':
                    continue
                new_row.append(c)
            if new_row[0].strip() == 'accuracy':
                new_row = [new_row[0], '', '', new_row[1], new_row[2]]
            rows.append(new_row)

        insert_table_to_text_edit(self.ui.stat_report_text_edit.textCursor(), headers, rows)

    # endregion

    # region QDA
    def update_qda_plots(self) -> None:
        """
        Update all QDA plots and fields
        @return: None
        """
        if 'QDA' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['QDA']
        model = model_results['model']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        y_score_dec_func = model_results['y_score_dec_func']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)
        binary = len(classes) == 2
        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.qda_scores_plot_widget,
                                explained_variance_ratio)
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.qda_dm_plot)
        if binary:
            self.update_pr_plot_bin(y_score_dec_func, y_test, classes[0], self.ui.qda_pr_plot, 'QDA')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.qda_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score_dec_func, plt_colors, self.ui.qda_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.qda_roc_plot)
        self.do_update_shap_plots('QDA')
        self.do_update_shap_plots_by_instance('QDA')

    def update_scores_plot(self, features_in_2d: np.ndarray, y: list[int], y_pred: list[int], model,
                           model_2d, plot_widget, explained_variance_ratio) -> None:
        """
        @param plot_widget:
        @param features_in_2d: transformed 2d
        @param y: true labels
        @param y_pred: predicted labels
        @param model: classificator
        @param model_2d: 2d model classificator
        @param explained_variance_ratio:
        @return:
        """
        plot_widget.canvas.axes.cla()
        classes = model.classes_
        plt_colors = []
        for cls in classes:
            clr = self.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter().name())
        tp = y == y_pred  # True Positive
        cmap = LinearSegmentedColormap('', None).from_list('', plt_colors)
        DecisionBoundaryDisplay.from_estimator(model_2d, features_in_2d, grid_resolution=1000, eps=.5, alpha=0.75,
                                               antialiased=True, cmap=cmap, ax=plot_widget.canvas.axes)
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*']
        for i, cls in enumerate(classes):
            true_positive_of_class = tp[y == cls]
            x_i = features_in_2d[y == cls]
            x_tp = x_i[true_positive_of_class]
            x_fp = x_i[~true_positive_of_class]
            mrkr = markers[cls]
            color = self.ui.GroupsTable.model().cell_data_by_idx_col_name(cls, 'Style')['color'].name()
            # inverted_color = invert_color(color)
            plot_widget.canvas.axes.scatter(x_tp[:, 0], x_tp[:, 1], marker=mrkr, color=color,
                                            edgecolor='black', s=30)
            plot_widget.canvas.axes.scatter(x_fp[:, 0], x_fp[:, 1], marker="x", s=30, color=color)
        plot_widget.canvas.axes.set_xlabel('PC-1 (%.2f%%)' % (explained_variance_ratio[0] * 100))
        plot_widget.canvas.axes.set_ylabel('PC-2 (%.2f%%)' % (explained_variance_ratio[1] * 100))
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # endregion

    # region Logistic regression

    def update_lr_plots(self) -> None:
        """
        Update all Logistic regression plots and fields
        @return: None
        """
        if 'Logistic regression' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['Logistic regression']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        y_score_dec_func = model_results['y_score_dec_func']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.lr_scores_plot_widget,
                                explained_variance_ratio)
        top = self.update_features_plot(model_results['x_train'], model, model_results['feature_names'],
                                        target_names, plt_colors, self.ui.lr_features_plot_widget)
        self.top_features['Logistic regression'] = top
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model,
                            self.ui.lr_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score_dec_func, y_test, classes[0], self.ui.lr_pr_plot, 'Logistic regression')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.lr_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score_dec_func, plt_colors, self.ui.lr_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.lr_roc_plot)
        self.do_update_shap_plots('Logistic regression')
        self.do_update_shap_plots_by_instance('Logistic regression')

    # endregion

    # region NuSVC

    def update_svc_plots(self) -> None:
        """
        Update all svc plots and fields
        @return: None
        """
        if 'NuSVC' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['NuSVC']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        y_score_dec_func = model_results['y_score_dec_func']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.svc_scores_plot_widget,
                                explained_variance_ratio)
        top = self.update_features_plot(model_results['x_train'], model, model_results['feature_names'],
                                        target_names, plt_colors, self.ui.svc_features_plot_widget)
        self.top_features['NuSVC'] = top
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.svc_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score_dec_func, y_test, classes[0], self.ui.svc_pr_plot, 'NuSVC')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.svc_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score_dec_func, plt_colors, self.ui.svc_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.svc_roc_plot)
        self.do_update_shap_plots('NuSVC')
        self.do_update_shap_plots_by_instance('NuSVC')

    # endregion

    # region Nearest Neighbors

    def update_nearest_plots(self) -> None:
        """
        Update all Nearest Neighbors plots and fields
        @return: None
        """
        if 'Nearest Neighbors' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['Nearest Neighbors']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.nearest_scores_plot_widget,
                                explained_variance_ratio)
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.nearest_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score, y_test, classes[0], self.ui.nearest_pr_plot, 'Nearest Neighbors')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.nearest_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score, plt_colors,
                                self.ui.nearest_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.nearest_roc_plot)
        self.do_update_shap_plots('Nearest Neighbors')
        self.do_update_shap_plots_by_instance('Nearest Neighbors')

    # endregion

    # region GPC

    def update_gpc_plots(self) -> None:
        """
        Update all GPC Neighbors plots and fields
        @return: None
        """
        if 'GPC' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['GPC']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.gpc_scores_plot_widget,
                                explained_variance_ratio)
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.gpc_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score, y_test, classes[0], self.ui.gpc_pr_plot, 'GPC')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.gpc_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score, plt_colors,
                                self.ui.gpc_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.gpc_roc_plot)
        self.do_update_shap_plots('GPC')
        self.do_update_shap_plots_by_instance('GPC')

    # endregion

    # region Decision Tree

    def update_dt_plots(self) -> None:
        """
        Update all Decision Tree plots and fields
        @return: None
        """
        if 'Decision Tree' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['Decision Tree']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.dt_scores_plot_widget,
                                explained_variance_ratio)
        self.plot_tree(model.best_estimator_, self.ui.dt_features_plot_widget, model_results['feature_names'],
                       target_names)
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.dt_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score, y_test, classes[0], self.ui.dt_pr_plot, 'Decision Tree')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.dt_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score, plt_colors,
                                self.ui.dt_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.dt_roc_plot)
        self.do_update_shap_plots('Decision Tree')
        self.do_update_shap_plots_by_instance('Decision Tree')

    def plot_tree(self, clf, plot_widget, feature_names, class_names) -> None:
        print(class_names)
        ax = plot_widget.canvas.axes
        ax.cla()
        plot_tree(clf, feature_names=feature_names, class_names=class_names, ax=ax)
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()
    # endregion

    # region 'Naive Bayes'

    def update_nb_plots(self) -> None:
        """
        Update all Decision Tree plots and fields
        @return: None
        """
        if 'Naive Bayes' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['Naive Bayes']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.nb_scores_plot_widget,
                                explained_variance_ratio)
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.nb_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score, y_test, classes[0], self.ui.nb_pr_plot, 'Naive Bayes')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.nb_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score, plt_colors,
                                self.ui.nb_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.nb_roc_plot)
        self.do_update_shap_plots('Naive Bayes')
        self.do_update_shap_plots_by_instance('Naive Bayes')

    # endregion

    # region 'Random Forest'

    def update_rf_plots(self) -> None:
        """
        Update all Random Forest plots and fields
        @return: None
        """
        if 'Random Forest' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['Random Forest']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.rf_scores_plot_widget,
                                explained_variance_ratio)
        self.update_features_plot_random_forest(model.best_estimator_, self.ui.rf_features_plot_widget)

        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.rf_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score, y_test, classes[0], self.ui.rf_pr_plot, 'Random Forest')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.rf_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score, plt_colors, self.ui.rf_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.rf_roc_plot)
        self.do_update_shap_plots('Random Forest')
        self.do_update_shap_plots_by_instance('Random Forest')

    def update_features_plot_random_forest(self, model, plot_widget) -> None:

        mdi_importances = Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=True)
        ax = plot_widget.canvas.axes
        ax.cla()
        ax.barh(mdi_importances.index, mdi_importances, color=self.theme_colors['primaryColor'])
        if plot_widget == self.ui.rf_features_plot_widget:
            ax.set_title("Random Forest Feature Importances (MDI)")
        else:
            ax.set_title("AdaBoost Feature Importances (MDI)")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # endregion

    # region ''AdaBoost''

    def update_ab_plots(self) -> None:
        """
        Update all 'AdaBoost' plots and fields
        @return: None
        """
        if 'AdaBoost' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['AdaBoost']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        y_score_dec_func = model_results['y_score_dec_func']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.ab_scores_plot_widget,
                                explained_variance_ratio)
        self.update_features_plot_random_forest(model, self.ui.ab_features_plot_widget)

        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.ab_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score_dec_func, y_test, classes[0], self.ui.ab_pr_plot, 'AdaBoost')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.ab_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score_dec_func, plt_colors, self.ui.ab_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.ab_roc_plot)
        self.do_update_shap_plots('AdaBoost')
        self.do_update_shap_plots_by_instance('AdaBoost')

    # endregion

    # region MLP

    def update_mlp_plots(self) -> None:
        """
        Update all 'MLP plots and fields
        @return: None
        """
        if 'MLP' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['MLP']
        y_train_plus_test = model_results['y_train_plus_test']
        model = model_results['model']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)

        self.update_scores_plot(features_in_2d, y_train_plus_test, model_results['y_pred_2d'], model,
                                model_results['model_2d'], self.ui.mlp_scores_plot_widget,
                                explained_variance_ratio)
        self.update_dm_plot(y_test, model_results['y_pred_test'], target_names, model, self.ui.mlp_dm_plot)
        binary = len(classes) == 2
        if binary:
            self.update_pr_plot_bin(y_score, y_test, classes[0], self.ui.mlp_pr_plot, 'MLP')
            self.update_roc_plot_bin(y_score, y_onehot_test, target_names, self.ui.mlp_roc_plot)
        else:
            self.update_pr_plot(classes, model_results['y_test_bin'], y_score, plt_colors, self.ui.mlp_pr_plot)
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.ui.mlp_roc_plot)
        self.do_update_shap_plots('MLP')
        self.do_update_shap_plots_by_instance('MLP')

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

        self.update_scores_plot_pca(features_in_2d, y_train_plus_test, self.ui.pca_scores_plot_widget,
                                    explained_variance_ratio)
        if 'loadings' in model_results:
            self.ui.pca_features_table_view.model().set_dataframe(model_results['loadings'])
            self.update_pca_loadings_plot(model_results['loadings'], explained_variance_ratio)

    def update_scores_plot_pca(self, features_in_2d: np.ndarray, y: list[int], plot_widget,
                               explained_variance_ratio) -> None:
        """
        @param plot_widget:
        @param features_in_2d: transformed 2d
        @param y: fact labels
        @param explained_variance_ratio:
        @return:
        """
        plot_widget.canvas.axes.cla()
        classes = np.unique(y)
        plt_colors = []
        for cls in classes:
            clr = self.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter().name())
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*']
        for i, cls in enumerate(classes):
            x_i = features_in_2d[y == cls]
            mrkr = markers[cls]
            color = self.ui.GroupsTable.model().cell_data_by_idx_col_name(cls, 'Style')['color'].name()
            plot_widget.canvas.axes.scatter(x_i[:, 0], x_i[:, 1], marker=mrkr, color=color,
                                            edgecolor='black', s=30)
        if plot_widget == self.ui.pca_scores_plot_widget:
            plot_widget.canvas.axes.set_xlabel('PC-1 (%.2f%%)' % (explained_variance_ratio[0] * 100))
            plot_widget.canvas.axes.set_ylabel('PC-2 (%.2f%%)' % (explained_variance_ratio[1] * 100))
        else:
            plot_widget.canvas.axes.set_xlabel('PLS-DA-1 (%.2f%%)' % (explained_variance_ratio[0] * 100))
            plot_widget.canvas.axes.set_ylabel('PLS-DA-2 (%.2f%%)' % (explained_variance_ratio[1] * 100))

        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_pca_loadings_plot(self, loadings, explained_variance_ratio) -> None:
        plot_widget = self.ui.pca_loadings_plot_widget
        plot_widget.canvas.axes.cla()
        features_names = list(loadings.axes[0])
        plot_widget.canvas.axes.scatter(loadings['PC1'], loadings['PC2'], color=self.theme_colors['primaryColor'],
                                        s=30)
        for i, txt in enumerate(features_names):
            plot_widget.canvas.axes.annotate(txt, (loadings['PC1'][i], loadings['PC2'][i]))
        plot_widget.canvas.axes.set_xlabel('PC-1 (%.2f%%)' % (explained_variance_ratio[0] * 100))
        plot_widget.canvas.axes.set_ylabel('PC-2 (%.2f%%)' % (explained_variance_ratio[1] * 100))
        plot_widget.canvas.axes.set_title("PCA Loadings")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()
    # endregion

    # region PLS DA

    def update_plsda_plots(self) -> None:
        """
        Update all PLS-DA plots and fields
        @return: None
        """
        if 'PLS-DA' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['PLS-DA']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        explained_variance_ratio = model_results['explained_variance_ratio']

        self.update_scores_plot_pca(features_in_2d, y_train_plus_test, self.ui.plsda_scores_plot_widget,
                                    explained_variance_ratio)
        if 'vips' in model_results:
            vips = model_results['vips']
            df = DataFrame(list(zip(model_results['feature_names'], vips)), columns=['feature', 'VIP'])
            self.ui.plsda_vip_table_view.model().set_dataframe(df)
            self.update_vip_plot(model_results['vips'], model_results['feature_names'])

    def update_vip_plot(self, vips, features_names) -> None:
        plot_widget = self.ui.plsda_vip_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        ser = Series(vips, index=features_names).sort_values(ascending=True)
        ax.barh(ser.index, ser, color=self.theme_colors['primaryColor'])
        ax.set_title("Variable Importance in the Projection (VIP)")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # endregion

    def classificators_calibrate(self) -> None:
        ax = self.ui.calib_curves.canvas.axes
        ax.cla()

        clf_list = []
        for key, value in self.latest_stat_result.items():
            if key == 'Naive Bayes':
                clf_list.append((value['model'], key, value['x_test'], value['y_test']))
        calibration_displays = {}
        colors = plt.cm.get_cmap("Dark2")
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*']
        for i, (clf, name, X_test, y_test) in enumerate(clf_list):
            if isinstance(clf, GridSearchCV):
                clf = clf.best_estimator_
            display = CalibrationDisplay.from_estimator(
                clf,
                X_test,
                y_test,
                n_bins=10,
                name=name,
                ax=ax,
                color=colors(i),
                marker=markers[i],
            )
            calibration_displays[name] = display

        ax.grid()
        ax.set_title("Calibration plots")
        try:
            self.ui.calib_curves.canvas.draw()
        except ValueError:
            pass
        self.ui.calib_curves.canvas.figure.tight_layout()

    # endregion

    # region PREDICT

    @asyncSlot()
    async def predict(self):
        if self.is_production_project:
            try:
                await self.do_predict_production()
            except Exception as err:
                self.show_error(err)
        else:
            try:
                await self.do_predict()
            except Exception as err:
                self.show_error(err)

    @asyncSlot()
    async def do_predict(self) -> None:
        self.beforeTime = datetime.now()
        self.ui.statusBar.showMessage('Predicting...')
        self.close_progress_bar()
        if self.is_production_project:
            clfs = list(self.stat_models.keys())
        else:
            clfs = list(self.latest_stat_result.keys())
        if 'PCA' in clfs:
            clfs.remove('PCA')
        if 'PLS-DA' in clfs:
            clfs.remove('PLS-DA')
        if len(clfs) == 0:
            return
        self._open_progress_bar(max_value=len(clfs))
        self._open_progress_dialog("Predicting...", "Cancel", maximum=len(clfs))
        X, _, _, _, filenames = self.dataset_for_ml()
        executor = ThreadPoolExecutor()
        # Для БОльших датасетов возможно лучше будет ProcessPoolExecutor. Но таких пока нет
        self.current_executor = executor
        with executor:
            if self.is_production_project:
                self.current_futures = [loop.run_in_executor(executor, clf_predict, X, self.stat_models[i], i)
                                        for i in clfs]
            else:
                self.current_futures = [loop.run_in_executor(executor, clf_predict, X,
                                                             self.latest_stat_result[i]['model'], i)
                                        for i in clfs]
            for future in self.current_futures:
                future.add_done_callback(self.progress_indicator)
            result = await gather(*self.current_futures)
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Fitting cancelled.')
            return
        if result:
            self.update_predict_table(result, clfs, filenames)
        self.close_progress_bar()
        self.ui.statusBar.showMessage('Predicting completed', 10000)

    @asyncSlot()
    async def do_predict_production(self) -> None:
        self.beforeTime = datetime.now()
        filenames = list(self.ImportedArray.keys())
        interpolated = None
        try:
            interpolated = await self.get_interpolated(filenames, self.interp_ref_array)
        except Exception as err:
            self.show_error(err)
        if interpolated:
            command = CommandUpdateInterpolated(self, interpolated, "Interpolate files")
            self.undoStack.push(command)
        else:
            return
        await self.despike()
        await self.convert()
        await self.cut_first()
        await self.normalize()
        await self.smooth()
        if self.ui.dataset_type_cb.currentText() == 'Smoothed':
            await self.do_predict()
            self.set_modified(True)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Predicting completed (s)', 100000)
            winsound.MessageBeep()
            return
        await self.baseline_correction()
        if self.ui.dataset_type_cb.currentText() == 'Baseline corrected':
            await self.do_predict()
            self.set_modified(True)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Predicting completed (b.c.)', 100000)
            winsound.MessageBeep()
            return
        await self.trim()
        await self.batch_fit()
        df = self.ui.deconvoluted_dataset_table_view.model().dataframe().reset_index(drop=True)
        self.ui.deconvoluted_dataset_table_view.model().set_dataframe(df)
        await self.do_predict()
        self.set_modified(True)
        self.close_progress_bar()
        self.ui.statusBar.showMessage('Predicting completed (d)', 100000)
        winsound.MessageBeep()

    def update_predict_table(self, predict_data: list[dict], clfs: list[str], filenames: list[str]) -> None:
        df = DataFrame({'Filename': filenames})
        df2 = DataFrame(columns=clfs)
        for clf_result in predict_data:
            clf_name = clf_result['clf_name']
            predicted = clf_result['predicted']
            predicted_proba = np.round(clf_result['predicted_proba'] * 100., 0)
            str_list = []
            for i, pr in enumerate(predicted):
                class_proba = str(pr) + ' ' + str(predicted_proba[i])
                str_list.append(class_proba)
            df2[clf_name] = str_list
        df = concat([df, df2], axis=1)

        self.ui.predict_table_view.model().set_dataframe(df)
    # endregion

    # region SETTINGS WINDOW

    def theme_bckgrnd_text_changed(self, s: str) -> None:
        self.update_theme_event(self, theme_bckgrnd=s, theme_color=self.theme_color)
        self.theme_bckgrnd = s
        self.update_icons()
        self.theme_colors = get_theme((s, self.theme_color, None))
        self.ui.input_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))
        self.ui.converted_cm_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))
        self.ui.cut_cm_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))
        self.ui.normalize_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))
        self.ui.smooth_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))
        self.ui.baseline_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))
        self.ui.average_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))
        self.ui.deconv_plot_widget.setBackground(QColor(self.theme_colors['plotBackground']))

    def theme_color_setting_text_changed(self, s: str) -> None:
        self.update_theme_event(self, theme_bckgrnd=self.theme_bckgrnd, theme_color=s)
        self.theme_color = s
        self.update_icons()
        self.theme_colors = get_theme((self.theme_bckgrnd, s, None))

    def recent_limit_spin_box_changed(self, r: int) -> None:
        self.recent_limit = r

    def undo_limit_spin_box_changed(self, limit: int) -> None:
        self.undoStack.setUndoLimit(limit)

    def auto_save_spin_box_changed(self, a: int) -> None:
        self.auto_save_minutes = a
        self.auto_save_timer.stop()
        self.auto_save_timer.start(1000 * 60 * self.auto_save_minutes)

    def axis_font_size_spin_box_changed(self, f: int) -> None:
        self.plot_font_size = f
        self._initial_plots_set_fonts()

    def axis_label_font_size_changed(self, i: int) -> None:
        self.axis_label_font_size = str(i)
        self._initial_plots_set_labels_font()

    def settings_show(self) -> None:
        settings_dialog = SettingsDialog(self)
        settings_dialog.show()

    # endregion


def _excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(format_exception(exc_type, exc_value, exc_tb))
    # exceptions = ['redo(', 'undo(']
    if exc_value.args[0][0: 19] == 'invalid result from':
        # and exc_value.args[0][-6: -1] in exceptions:
        return
    show_error(exc_type, exc_value, tb)


def show_error(exc_type, exc_value, exc_tb):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText(str(exc_value))
    msg.setWindowTitle(str(exc_type))
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.setInformativeText(exc_tb)
    pyperclip.copy(exc_tb)
    msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    msg.exec()


if __name__ == "__main__":
    from modules.init import apply_stylesheet
    from multiprocessing import freeze_support
    import ctypes

    my_app_id = 'dark.rs.tool.1000'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)
    environ['PYTHONASYNCIODEBUG'] = '1'
    environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    environ['CANCEL'] = '0'
    freeze_support()
    logo = 'logo.ico'
    splash_color = Qt.GlobalColor.white
    splash_png = 'splash_sized.png'
    CANCEL = False

    extra = {
        # Button colors
        'danger': '#dc3545',
        'warning': '#ffc107',
        'success': '#17a2b8',
        # Font
        'font_family': 'AbletonSans',
        # Density
        'density_scale': '0',
        # Button Shape
        'button_shape': 'default',
    }
    check_preferences_file()
    read_prefs = read_preferences()
    theme_bckgrnd = read_prefs[0]
    theme_color = read_prefs[1]
    theme = (theme_bckgrnd, theme_color, None)
    invert = 'Light' in theme_bckgrnd and 'Dark' not in theme_bckgrnd
    if invert:
        splash_color = Qt.GlobalColor.black
        splash_png = 'splash_white_sized.png'
        logo = 'logo_white.ico'
    recent_limit = read_prefs[2]
    undo_limit = read_prefs[3]
    auto_save_minutes = read_prefs[4]
    plot_font_size = int(read_prefs[5])
    axis_label_font_size = read_prefs[6]
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    splash_img = QPixmap(splash_png)
    splash = QSplashScreen(splash_img, Qt.WindowType.WindowStaysOnTopHint)
    font = splash.font()
    font.setPixelSize(12)
    font.setWeight(QFont.Weight.Normal)
    font.setFamily('AbletonSans, Roboto')
    splash.setFont(font)
    splash.show()
    splash.showMessage('ver. 1.0.00 ' + '\n' + 'Starting QApplication.', Qt.AlignmentFlag.AlignBottom,
                       splash_color)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Round)
    app.processEvents()
    app.setQuitOnLastWindowClosed(False)
    # app.lastWindowClosed.connect(app.quit)
    win_icon = QIcon(logo)
    app.setWindowIcon(win_icon)

    frame = RuntimeStylesheets(event_loop=loop)
    # Set theme on initialization
    apply_stylesheet(app, (theme_bckgrnd, theme_color, frame.theme_colors), invert_secondary=invert, extra=extra)
    frame.showMaximized()

    with loop:
        loop.run_forever()
