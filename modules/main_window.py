import os
import sys
from asyncio import create_task, gather, sleep, wait, get_event_loop
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from gc import get_objects, collect
from logging import basicConfig, critical, error, DEBUG, info, warning
from os import environ
from pathlib import Path
from shelve import open as shelve_open
from traceback import format_exc
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import shap
from asyncqtpy import asyncSlot
from pandas import DataFrame, MultiIndex, ExcelWriter
from psutil import cpu_percent
from pyqtgraph import setConfigOption, PlotDataItem, PlotCurveItem, SignalProxy, InfiniteLine, LinearRegionItem, \
    mkPen, mkBrush, FillBetweenItem, ArrowItem
from qtpy.QtCore import Qt, QPoint, QEvent, QModelIndex, QTimer, QMarginsF, QPointF
from qtpy.QtGui import QFont, QIcon, QCloseEvent, QMouseEvent, QKeyEvent, QContextMenuEvent, QEnterEvent, \
    QMoveEvent, QColor, QPageLayout, QPageSize
from qtpy.QtWidgets import QMessageBox, QMainWindow, QColorDialog, QUndoStack, QMenu, \
    QAction, QHeaderView, QAbstractItemView, QLabel, QProgressBar, QProgressDialog, QPushButton, QFileDialog, \
    QLineEdit, QInputDialog, QTableView, QScrollArea
from qtpy.QtWinExtras import QWinTaskbarButton

import modules.setting_window
from modules.dialogs import CurvePropertiesWindow
from modules.pandas_tables import PandasModelGroupsTable, PandasModelInputTable, PandasModelDeconvTable, \
    PandasModelDeconvLinesTable, ComboDelegate, PandasModelFitParamsTable, \
    DoubleSpinBoxDelegate, PandasModelFitIntervals, IntervalsTableDelegate, \
    PandasModelSmoothedDataset, PandasModelBaselinedDataset, PandasModelDeconvolutedDataset, PandasModel, \
    PandasModelPredictTable, PandasModelPCA, PandasModelIgnoreDataset
from modules.default_values import default_values, \
    baseline_parameter_defaults, optimize_extended_range_methods
from modules.init import QtStyleTools, get_theme, opacity
from modules.mw_page1_preprocessing import PreprocessingLogic
from modules.mw_page2_fitting import FittingLogic
from modules.mw_page4_stat_analysis import StatAnalysisLogic, update_plot_tree
from modules.mw_page5_predict import PredictLogic
from modules.setting_window import SettingWindow
from modules.start_program import splash_show_message
from modules.static_functions import random_rgb, check_recent_files, \
    get_memory_used, check_rs_tool_folder, import_spectrum, curve_pen_brush_by_style, \
    action_help, set_roi_size
from modules.ui_main_window import Ui_MainWindow
from modules.undo_redo import CommandImportFiles, CommandAddGroup, CommandDeleteGroup, CommandChangeGroupCell, \
    CommandChangeGroupCellsBatch, CommandDeleteInputSpectrum, CommandChangeGroupStyle, CommandAddDeconvLine, \
    CommandDeleteDeconvLines, CommandUpdateDeconvCurveStyle, \
    CommandUpdateDataCurveStyle, \
    CommandClearAllDeconvLines, CommandStartIntervalChanged, CommandEndIntervalChanged, \
    CommandFitIntervalAdded, CommandFitIntervalDeleted
from modules.functions_guess_raman_lines import show_distribution
from modules.functions_cut_trim import cut_full_spectrum
from modules.functions_for_arrays import nearest_idx, find_nearest, normalize_between_0_1
from modules.functions_baseline_correction import ex_mod_poly, baseline_imodpoly, baseline_modpoly

plt.style.use(['dark_background'])
plt.set_loglevel("info")
shap.initjs()
environ['OPENBLAS_NUM_THREADS'] = '1'


class MainWindow(QMainWindow, QtStyleTools):

    def __init__(self, event_loop, theme, prefs, splash) -> None:
        super().__init__()

        self.plt_style = None
        self.taskbar_progress = None
        self.taskbar_button = None
        self.lda_1d_inf_lines = []
        self.current_futures = []
        self.baseline_method = None
        self.smooth_method = None
        self.normalization_method = None
        self.break_event = None
        self.auto_save_timer = None
        self.cpu_load = None
        self.timer_mem_update = None

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
        splash_show_message(splash, 'Initializing main window...')
        self.theme_bckgrnd = prefs[0]
        self.theme_color = prefs[1]
        self.plot_font_size = int(prefs[5])
        self.axis_label_font_size = prefs[6]

        self.current_executor = None

        self.time_start = None
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

        # parameters to turn on/off pushing UndoStack command during another redo/undo command executing
        self.CommandStartIntervalChanged_allowed = True
        self.CommandEndIntervalChanged_allowed = True

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowFrameSection.TopSection |
                            Qt.WindowType.WindowMinMaxButtonsHint)
        self.ImportedArray = dict()
        self.project_path = None
        self.recent_limit = prefs[2]
        self.auto_save_minutes = int(prefs[4])
        self.dragPos = None
        self.keyPressEvent = self.key_press_event
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.modified = False
        self.window_maximized = True
        self._ascending_input_table = False
        self._ascending_deconv_lines_table = False
        self.laser_wl_value_old = self.ui.laser_wl_spinbox.value()
        self.theme_colors = get_theme(theme)
        self.theme = theme
        self.taskbar_button = QWinTaskbarButton()
        self.taskbar_button.setWindow(self.windowHandle())
        self.preprocessing = PreprocessingLogic(self)
        self.fitting = FittingLogic(self)
        self._init_default_values()

        # UNDO/ REDO
        self.undoStack = QUndoStack(self)
        self.undoStack.setUndoLimit(int(prefs[3]))

        # SET UI DEFINITIONS
        self.setWindowIcon(QIcon(environ['logo']))
        self.setWindowTitle('Raman Spectroscopy Tool ')

        self.plot_text_color_value = self.theme_colors['plotText']
        self.plot_text_color = QColor(self.plot_text_color_value)
        self.plot_background_color = QColor(self.theme_colors['plotBackground'])
        self.plot_background_color_web = QColor(self.theme_colors['backgroundMainColor'])
        self.update_icons()
        path = os.getenv('APPDATA') + '/RS-tool/log.log'

        self.initial_ui_definitions()
        try:
            check_recent_files()
        except Exception as err:
            self.show_error(err)
        splash_show_message(splash, 'Initializing menu...')
        self.initial_menu()
        splash_show_message(splash, 'Initializing left side menu...')

        self._init_left_menu()
        splash_show_message(splash, 'Initializing tables...')
        self._initial_all_tables()
        splash_show_message(splash, 'Initializing scrollbar...')
        self.initial_right_scrollbar()
        splash_show_message(splash, 'Initializing figures...')
        self._initial_plots()
        splash_show_message(splash, 'Initializing plot buttons...')
        self.initial_plot_buttons()
        splash_show_message(splash, 'Initializing fitting right side frame...')
        self._initial_guess_table_frame()
        self.initial_timers()
        self.set_buttons_ability()
        self.ui.stat_tab_widget.currentChanged.connect(self.stat_tab_widget_tab_changed)
        self.ui.stat_tab_widget.setTabEnabled(12, False)
        self.ui.stat_tab_widget.setTabVisible(12, False)
        self.ui.page5_predict.clicked.connect(self.predict)
        self.ui.splitter_page1.moveSplitter(100, 1)
        self.ui.splitter_page2.moveSplitter(100, 1)
        self.memory_usage_label = QLabel(self)
        self.statusBar().addPermanentWidget(self.memory_usage_label)

        self.ui.baseline_correction_method_comboBox.setCurrentText(self.default_values['baseline_method_comboBox'])
        splash_show_message(splash, 'Setting default parameters')
        self._set_parameters_to_default()
        splash_show_message(splash, 'Checking RS-tool folder')
        try:
            check_rs_tool_folder()
        except Exception as err:
            self.show_error(err)
        self.setting_window = SettingWindow(self)
        self.stat_analysis_logic = StatAnalysisLogic(self)
        self.predict_logic = PredictLogic(self)
        basicConfig(level=DEBUG, filename=path, filemode='w', format="%(asctime)s %(levelname)s %(message)s")
        info('Logging started.')
        self.set_modified(False)
        self.ui.statusBar.showMessage('Ready', 2000)

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
        self.fitting.data_style = {'color': QColor(self.theme_colors['secondaryColor']),
                                   'style': Qt.PenStyle.SolidLine,
                                   'width': 1.0,
                                   'fill': False,
                                   'use_line_color': True,
                                   'fill_color': QColor().fromRgb(random_rgb()),
                                   'fill_opacity': 0.0}
        self.data_style_button_style_sheet(self.fitting.data_style['color'].name())
        self.fitting.sum_style = {'color': QColor(self.theme_colors['primaryColor']),
                                  'style': Qt.PenStyle.DashLine,
                                  'width': 1.0,
                                  'fill': False,
                                  'use_line_color': True,
                                  'fill_color': QColor().fromRgb(random_rgb()),
                                  'fill_opacity': 0.0}
        self.sum_style_button_style_sheet(self.fitting.sum_style['color'].name())
        self.fitting.sigma3_style = {'color': QColor(self.theme_colors['primaryDarkColor']),
                                     'style': Qt.PenStyle.SolidLine,
                                     'width': 1.0,
                                     'fill': True,
                                     'use_line_color': True,
                                     'fill_color': QColor().fromRgb(random_rgb()),
                                     'fill_opacity': 0.25}
        self.sigma3_style_button_style_sheet(self.fitting.sigma3_style['color'].name())
        self.fitting.residual_style = {'color': QColor(self.theme_colors['secondaryLightColor']),
                                       'style': Qt.PenStyle.DotLine,
                                       'width': 1.0,
                                       'fill': False,
                                       'use_line_color': True,
                                       'fill_color': QColor().fromRgb(random_rgb()),
                                       'fill_opacity': 0.0}
        self.residual_style_button_style_sheet(self.fitting.residual_style['color'].name())

        self.baseline_parameter_defaults = baseline_parameter_defaults()

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
        self.ui.random_state_sb.setValue(self.default_values['random_state_sb'])
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
        self.ui.use_pca_checkBox.setChecked(False)
        self.ui.emsc_pca_n_spinBox.setValue(self.default_values['EMSC_N_PCA'])
        self.ui.max_CCD_value_spinBox.setValue(self.default_values['max_CCD_value'])
        self.ui.interval_start_dsb.setValue(self.default_values['interval_start'])
        self.ui.interval_end_dsb.setMaximum(99_999.0)
        self.ui.interval_end_dsb.setValue(self.default_values['interval_end'])
        self.ui.max_dx_dsb.setValue(self.default_values['max_dx_guess'])
        self.ui.mlp_layer_size_spinBox.setValue(self.default_values['mlp_layer_size_spinBox'])
        self.ui.max_epoch_spinBox.setValue(self.default_values['max_epoch_spinBox'])
        self.ui.learning_rate_doubleSpinBox.setValue(self.default_values['learning_rate_doubleSpinBox'])
        self.ui.feature_display_max_spinBox.setValue(self.default_values['feature_display_max_spinBox'])
        self.ui.l_ratio_doubleSpinBox.setValue(self.default_values['l_ratio'])

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
        self.fitting.sigma3_top = PlotCurveItem(name='sigma3_top')
        self.fitting.sigma3_bottom = PlotCurveItem(name='sigma3_bottom')
        color = self.fitting.sigma3_style['color']
        color.setAlphaF(0.25)
        pen = mkPen(color=color, style=Qt.PenStyle.SolidLine)
        brush = mkBrush(color)
        self.fitting.fill = FillBetweenItem(self.fitting.sigma3_top, self.fitting.sigma3_bottom, brush, pen)
        self.deconvolution_plotItem.addItem(self.fitting.fill)
        self.fitting.fill.setVisible(False)

    def fit_plot_mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.fitting.deselect_selected_line()

    def _initial_deconv_plot_color(self) -> None:
        self.ui.deconv_plot_widget.setBackground(self.plot_background_color)
        self.deconvolution_plotItem.getAxis('bottom').setPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis('left').setPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis('bottom').setTextPen(self.plot_text_color)
        self.deconvolution_plotItem.getAxis('left').setTextPen(self.plot_text_color)

    def initial_scores_plot(self, plot_widget, cl_type=None):
        plot_widget.canvas.axes.cla()
        if cl_type == 'LDA':
            function_name = 'LD-'
            plot_widget.setVisible(True)
        elif cl_type == 'QDA':
            function_name = 'QD-'
        elif cl_type == 'PLS-DA':
            function_name = 'PLS-DA-'
        else:
            function_name = 'PC-'
        plot_widget.canvas.axes.set_xlabel(function_name + '1', fontsize=self.axis_label_font_size)
        plot_widget.canvas.axes.set_ylabel(function_name + '2', fontsize=self.axis_label_font_size)
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

    def _initial_lda_force_single_plot(self) -> None:
        self.ui.lda_force_single.page().setHtml('')
        self.ui.lda_force_single.contextMenuEvent = self.lda_force_single_context_menu_event

    def lda_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lda_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.lda_force_single, 'LDA'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_lda_force_full_plot(self) -> None:
        self.ui.lda_force_full.page().setHtml('')
        self.ui.lda_force_full.contextMenuEvent = self.lda_force_full_context_menu_event

    def lda_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lda_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.lda_force_full, 'LDA', True))
        menu.move(a0.globalPos())
        menu.show()

    def reload_force(self, plot_widget, cl_type: str, full: bool = False):
        if cl_type not in self.stat_analysis_logic.latest_stat_result:
            return
        shap_html = 'shap_html_full' if full else 'shap_html'
        if shap_html not in self.stat_analysis_logic.latest_stat_result[cl_type]:
            return
        plot_widget.setHtml(self.stat_analysis_logic.latest_stat_result[cl_type][shap_html])

    def _initial_lr_force_single_plot(self) -> None:
        self.ui.lr_force_single.page().setHtml('')
        self.ui.lr_force_single.contextMenuEvent = self.lr_force_single_context_menu_event

    def lr_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lr_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.lr_force_single, 'Logistic regression'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_lr_force_full_plot(self) -> None:
        self.ui.lr_force_full.page().setHtml('')
        self.ui.lr_force_full.contextMenuEvent = self.lr_force_full_context_menu_event

    def lr_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.lr_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.lr_force_full, 'Logistic regression', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_svc_force_single_plot(self) -> None:
        self.ui.svc_force_single.page().setHtml('')
        self.ui.svc_force_single.contextMenuEvent = self.svc_force_single_context_menu_event

    def svc_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.svc_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.svc_force_single, 'NuSVC'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_svc_force_full_plot(self) -> None:
        self.ui.svc_force_full.page().setHtml('')
        self.ui.svc_force_full.contextMenuEvent = self.svc_force_full_context_menu_event

    def svc_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.svc_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.svc_force_full, 'NuSVC', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nearest_force_single_plot(self) -> None:
        self.ui.nearest_force_single.page().setHtml('')
        self.ui.nearest_force_single.contextMenuEvent = self.nearest_force_single_context_menu_event

    def nearest_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nearest_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.nearest_force_single, 'Nearest Neighbors'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nearest_force_full_plot(self) -> None:
        self.ui.nearest_force_full.page().setHtml('')
        self.ui.nearest_force_full.contextMenuEvent = self.nearest_force_full_context_menu_event

    def nearest_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nearest_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.nearest_force_full, 'Nearest Neighbors', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_gpc_force_single_plot(self) -> None:
        self.ui.gpc_force_single.page().setHtml('')
        self.ui.gpc_force_single.contextMenuEvent = self.gpc_force_single_context_menu_event

    def gpc_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.gpc_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.gpc_force_single, 'GPC'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_gpc_force_full_plot(self) -> None:
        self.ui.gpc_force_full.page().setHtml('')
        self.ui.gpc_force_full.contextMenuEvent = self.gpc_force_full_context_menu_event

    def gpc_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.gpc_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.gpc_force_full, 'GPC', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_dt_force_single_plot(self) -> None:
        self.ui.dt_force_single.page().setHtml('')
        self.ui.dt_force_single.contextMenuEvent = self.dt_force_single_context_menu_event

    def dt_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.dt_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.dt_force_single, 'Decision Tree'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_dt_force_full_plot(self) -> None:
        self.ui.dt_force_full.page().setHtml('')
        self.ui.dt_force_full.contextMenuEvent = self.dt_force_full_context_menu_event

    def dt_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.dt_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.dt_force_full, 'Decision Tree', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nb_force_single_plot(self) -> None:
        self.ui.nb_force_single.page().setHtml('')
        self.ui.nb_force_single.contextMenuEvent = self.dt_force_single_context_menu_event

    def nb_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nb_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.nb_force_single, 'Naive Bayes'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_nb_force_full_plot(self) -> None:
        self.ui.nb_force_full.page().setHtml('')
        self.ui.nb_force_full.contextMenuEvent = self.nb_force_full_context_menu_event

    def nb_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.nb_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.nb_force_full, 'Naive Bayes', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_rf_force_single_plot(self) -> None:
        self.ui.rf_force_single.page().setHtml('')
        self.ui.rf_force_single.contextMenuEvent = self.rf_force_single_context_menu_event

    def rf_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.rf_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.rf_force_single, 'Random Forest'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_rf_force_full_plot(self) -> None:
        self.ui.rf_force_full.page().setHtml('')
        self.ui.rf_force_full.contextMenuEvent = self.rf_force_full_context_menu_event

    def rf_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.rf_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.rf_force_full, 'Random Forest', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_ab_force_single_plot(self) -> None:
        self.ui.ab_force_single.page().setHtml('')
        self.ui.ab_force_single.contextMenuEvent = self.ab_force_single_context_menu_event

    def ab_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.ab_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.ab_force_single, 'AdaBoost'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_ab_force_full_plot(self) -> None:
        self.ui.ab_force_full.page().setHtml('')
        self.ui.ab_force_full.contextMenuEvent = self.ab_force_full_context_menu_event

    def ab_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.ab_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.ab_force_full, 'AdaBoost', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_mlp_force_single_plot(self) -> None:
        self.ui.mlp_force_single.page().setHtml('')
        self.ui.mlp_force_single.contextMenuEvent = self.mlp_force_single_context_menu_event

    def mlp_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.mlp_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.mlp_force_single, 'MLP'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_mlp_force_full_plot(self) -> None:
        self.ui.mlp_force_full.page().setHtml('')
        self.ui.mlp_force_full.contextMenuEvent = self.mlp_force_full_context_menu_event

    def mlp_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.mlp_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.mlp_force_full, 'MLP', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_torch_force_single_plot(self) -> None:
        self.ui.torch_force_single.page().setHtml('')
        self.ui.torch_force_single.contextMenuEvent = self.torch_single_context_menu_event

    def torch_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.torch_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.torch_force_single, 'Torch'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_torch_force_full_plot(self) -> None:
        self.ui.torch_force_full.page().setHtml('')
        self.ui.torch_force_full.contextMenuEvent = self.torch_force_full_context_menu_event

    def torch_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.torch_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.torch_force_full, 'Torch', True))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_xgboost_force_single_plot(self) -> None:
        self.ui.xgboost_force_single.page().setHtml('')
        self.ui.xgboost_force_single.contextMenuEvent = self.xgboost_force_single_context_menu_event

    def xgboost_force_single_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.xgboost_force_single.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.xgboost_force_single, 'XGBoost'))
        menu.move(a0.globalPos())
        menu.show()

    def _initial_xgboost_force_full_plot(self) -> None:
        self.ui.xgboost_force_full.page().setHtml('')
        self.ui.xgboost_force_full.contextMenuEvent = self.xgboost_force_full_context_menu_event

    def xgboost_force_full_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Save .pdf', lambda: self.web_view_print_pdf(self.ui.xgboost_force_full.page()))
        menu.addAction('Refresh', lambda: self.reload_force(self.ui.xgboost_force_full, 'XGBoost', True))
        menu.move(a0.globalPos())
        menu.show()

    def web_view_print_pdf(self, page):
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(self, 'Print page to PDF', os.getenv('APPDATA') + '/RS-tool', "PDF (*.pdf)")
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
                        self.ui.lr_shap_waterfall, self.ui.lr_shap_decision,
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
                        self.ui.dt_shap_waterfall, self.ui.dt_tree_plot_widget,
                        self.ui.nb_dm_plot, self.ui.nb_pr_plot, self.ui.nb_roc_plot,
                        self.ui.nb_scores_plot_widget, self.ui.nb_shap_beeswarm, self.ui.nb_shap_decision,
                        self.ui.nb_shap_heatmap, self.ui.nb_shap_means, self.ui.nb_shap_scatter,
                        self.ui.nb_shap_waterfall,
                        self.ui.rf_dm_plot, self.ui.rf_features_plot_widget, self.ui.rf_pr_plot, self.ui.rf_roc_plot,
                        self.ui.rf_scores_plot_widget, self.ui.rf_shap_beeswarm, self.ui.rf_shap_decision,
                        self.ui.rf_shap_heatmap, self.ui.rf_shap_means, self.ui.rf_shap_scatter,
                        self.ui.rf_shap_waterfall, self.ui.rf_tree_plot_widget,
                        self.ui.ab_dm_plot, self.ui.ab_features_plot_widget, self.ui.ab_pr_plot, self.ui.ab_roc_plot,
                        self.ui.ab_scores_plot_widget, self.ui.ab_shap_beeswarm, self.ui.ab_shap_decision,
                        self.ui.ab_shap_heatmap, self.ui.ab_shap_means, self.ui.ab_shap_scatter,
                        self.ui.ab_shap_waterfall, self.ui.ab_tree_plot_widget,
                        self.ui.mlp_dm_plot, self.ui.mlp_pr_plot, self.ui.mlp_roc_plot,
                        self.ui.mlp_scores_plot_widget, self.ui.mlp_shap_beeswarm, self.ui.mlp_shap_decision,
                        self.ui.mlp_shap_heatmap, self.ui.mlp_shap_means, self.ui.mlp_shap_scatter,
                        self.ui.mlp_shap_waterfall,
                        self.ui.pca_scores_plot_widget, self.ui.pca_loadings_plot_widget,
                        self.ui.plsda_scores_plot_widget, self.ui.plsda_vip_plot_widget,
                        self.ui.xgboost_dm_plot, self.ui.xgboost_features_plot_widget, self.ui.xgboost_pr_plot,
                        self.ui.xgboost_roc_plot, self.ui.xgboost_scores_plot_widget, self.ui.xgboost_shap_beeswarm,
                        self.ui.xgboost_shap_decision, self.ui.xgboost_shap_heatmap, self.ui.xgboost_shap_means,
                        self.ui.xgboost_shap_scatter, self.ui.xgboost_shap_waterfall, self.ui.xgboost_tree_plot_widget,
                        self.ui.torch_scores_plot_widget, self.ui.torch_dm_plot, self.ui.torch_pr_plot,
                        self.ui.torch_roc_plot, self.ui.torch_shap_beeswarm, self.ui.torch_shap_means,
                        self.ui.torch_shap_heatmap, self.ui.torch_shap_scatter, self.ui.torch_shap_decision,
                        self.ui.torch_shap_waterfall
                        ]
        for pl in plot_widgets:
            self.set_canvas_colors(pl.canvas)
        if self.ui.current_group_shap_comboBox.currentText() == '':
            return
        await self.loop.run_in_executor(None, self._update_shap_plots)
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        self.stat_analysis_logic.update_force_single_plots()
        self.stat_analysis_logic.update_force_full_plots()

    def _update_shap_plots(self) -> None:
        self.stat_analysis_logic.do_update_shap_plots('LDA')
        self.stat_analysis_logic.do_update_shap_plots('QDA')
        self.stat_analysis_logic.do_update_shap_plots('Logistic regression')
        self.stat_analysis_logic.do_update_shap_plots('NuSVC')
        self.stat_analysis_logic.do_update_shap_plots('Nearest Neighbors')
        self.stat_analysis_logic.do_update_shap_plots('GPC')
        self.stat_analysis_logic.do_update_shap_plots('Decision Tree')
        self.stat_analysis_logic.do_update_shap_plots('Naive Bayes')
        self.stat_analysis_logic.do_update_shap_plots('Random Forest')
        self.stat_analysis_logic.do_update_shap_plots('AdaBoost')
        self.stat_analysis_logic.do_update_shap_plots('MLP')
        self.stat_analysis_logic.do_update_shap_plots('Torch')
        self.stat_analysis_logic.do_update_shap_plots('XGBoost')

    def _update_shap_plots_by_instance(self) -> None:
        self.stat_analysis_logic.do_update_shap_plots_by_instance('LDA')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('QDA')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('Logistic regression')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('NuSVC')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('Nearest Neighbors')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('GPC')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('Decision Tree')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('Naive Bayes')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('Random Forest')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('AdaBoost')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('MLP')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('Torch')
        self.stat_analysis_logic.do_update_shap_plots_by_instance('XGBoost')

    @asyncSlot()
    async def update_shap_scatters(self) -> None:
        self.loop.run_in_executor(None, self.do_update_shap_scatters)

    def do_update_shap_scatters(self):
        tab_id_and_classificator = [(0, 'LDA'), (1, 'QDA'), (2, 'Logistic regression'), (3, 'NuSVC'),
                                    (4, 'Nearest Neighbors'), (5, 'GPC'), (6, 'Decision Tree'), (7, 'Naive Bayes'),
                                    (8, 'Random Forest'), (9, 'AdaBoost'), (10, 'MLP'), (11, 'XGBoost'), (12, 'Torch')]
        for tab_id, classificator in tab_id_and_classificator:
            if self.ui.stat_tab_widget.currentIndex() == tab_id \
                    and classificator in self.stat_analysis_logic.latest_stat_result \
                    and 'shap_values' in self.stat_analysis_logic.latest_stat_result[classificator]:
                target_names = self.stat_analysis_logic.latest_stat_result[classificator]['target_names']
                if self.ui.current_group_shap_comboBox.currentText() not in target_names:
                    return
                i = int(np.where(target_names == self.ui.current_group_shap_comboBox.currentText())[0][0])
                self.stat_analysis_logic.update_shap_scatter_plot(False, i, classificator)
            else:
                continue

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
                      labelcolor=self.plot_text_color.name(), prop={'size': self.plot_font_size})
        try:
            canvas.draw()
        except ValueError | np.linalg.LinAlgError:
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
        self.initial_scores_plot(self.ui.lda_scores_2d_plot_widget, 'LDA')
        self.initial_stat_plot(self.ui.lda_features_plot_widget)
        self.initial_stat_plot(self.ui.lda_roc_plot)
        self.initial_stat_plot(self.ui.lda_dm_plot)
        self.initial_stat_plot(self.ui.lda_pr_plot)
        self.initial_shap_plot(self.ui.lda_shap_means)
        self.initial_shap_plot(self.ui.lda_shap_beeswarm)
        self.initial_shap_plot(self.ui.lda_shap_heatmap)
        self.initial_shap_plot(self.ui.lda_shap_scatter)
        self.initial_shap_plot(self.ui.lda_shap_waterfall)
        self.initial_shap_plot(self.ui.lda_shap_decision)

        self._initial_lda_force_single_plot()
        self._initial_lda_force_full_plot()

    def _initial_qda_plots(self) -> None:
        self.initial_scores_plot(self.ui.qda_scores_plot_widget, 'QDA')
        self.initial_stat_plot(self.ui.qda_dm_plot)
        self.initial_stat_plot(self.ui.qda_pr_plot)
        self.initial_stat_plot(self.ui.qda_roc_plot)
        self.initial_shap_plot(self.ui.qda_shap_means)
        self.initial_shap_plot(self.ui.qda_shap_beeswarm)
        self.initial_shap_plot(self.ui.qda_shap_heatmap)
        self.initial_shap_plot(self.ui.qda_shap_scatter)
        self.initial_shap_plot(self.ui.qda_shap_waterfall)

    def _initial_lr_plots(self) -> None:
        self.initial_scores_plot(self.ui.lr_scores_plot_widget)
        self.initial_stat_plot(self.ui.lr_dm_plot)
        self.initial_stat_plot(self.ui.lr_features_plot_widget)
        self.initial_stat_plot(self.ui.lr_pr_plot)
        self.initial_stat_plot(self.ui.lr_roc_plot)
        self.initial_shap_plot(self.ui.lr_shap_means)
        self.initial_shap_plot(self.ui.lr_shap_beeswarm)
        self.initial_shap_plot(self.ui.lr_shap_heatmap)
        self.initial_shap_plot(self.ui.lr_shap_scatter)
        self.initial_shap_plot(self.ui.lr_shap_waterfall)
        self.initial_shap_plot(self.ui.lr_shap_decision)

        self._initial_lr_force_single_plot()
        self._initial_lr_force_full_plot()

    def _initial_svc_plots(self) -> None:
        self.initial_stat_plot(self.ui.svc_dm_plot)
        self.initial_stat_plot(self.ui.svc_features_plot_widget)
        self.initial_stat_plot(self.ui.svc_pr_plot)
        self.initial_stat_plot(self.ui.svc_roc_plot)
        self.initial_scores_plot(self.ui.svc_scores_plot_widget)
        self.initial_shap_plot(self.ui.svc_shap_means)
        self.initial_shap_plot(self.ui.svc_shap_beeswarm)
        self.initial_shap_plot(self.ui.svc_shap_heatmap)
        self.initial_shap_plot(self.ui.svc_shap_scatter)
        self.initial_shap_plot(self.ui.svc_shap_waterfall)
        self.initial_shap_plot(self.ui.svc_shap_decision)

        self._initial_svc_force_single_plot()
        self._initial_svc_force_full_plot()

    def _initial_nearest_plots(self) -> None:
        self.initial_stat_plot(self.ui.nearest_dm_plot)
        self.initial_stat_plot(self.ui.nearest_pr_plot)
        self.initial_stat_plot(self.ui.nearest_roc_plot)
        self.initial_scores_plot(self.ui.nearest_scores_plot_widget)
        self.initial_shap_plot(self.ui.nearest_shap_means)
        self.initial_shap_plot(self.ui.nearest_shap_beeswarm)
        self.initial_shap_plot(self.ui.nearest_shap_heatmap)
        self.initial_shap_plot(self.ui.nearest_shap_scatter)
        self.initial_shap_plot(self.ui.nearest_shap_waterfall)
        self.initial_shap_plot(self.ui.nearest_shap_decision)
        self._initial_nearest_force_single_plot()
        self._initial_nearest_force_full_plot()

    def _initial_gpc_plots(self) -> None:
        self.initial_stat_plot(self.ui.gpc_dm_plot)
        self.initial_stat_plot(self.ui.gpc_pr_plot)
        self.initial_stat_plot(self.ui.gpc_roc_plot)
        self.initial_scores_plot(self.ui.gpc_scores_plot_widget)
        self.initial_shap_plot(self.ui.gpc_shap_means)
        self.initial_shap_plot(self.ui.gpc_shap_beeswarm)
        self.initial_shap_plot(self.ui.gpc_shap_heatmap)
        self.initial_shap_plot(self.ui.gpc_shap_scatter)
        self.initial_shap_plot(self.ui.gpc_shap_waterfall)
        self.initial_shap_plot(self.ui.gpc_shap_decision)

        self._initial_gpc_force_single_plot()
        self._initial_gpc_force_full_plot()

    def _initial_dt_plots(self) -> None:
        self.initial_stat_plot(self.ui.dt_dm_plot)
        self.initial_stat_plot(self.ui.dt_pr_plot)
        self.initial_stat_plot(self.ui.dt_roc_plot)
        self.initial_scores_plot(self.ui.dt_scores_plot_widget)
        self.initial_stat_plot(self.ui.dt_features_plot_widget)
        self.initial_stat_plot(self.ui.dt_tree_plot_widget)
        self.initial_shap_plot(self.ui.dt_shap_means)
        self.initial_shap_plot(self.ui.dt_shap_beeswarm)
        self.initial_shap_plot(self.ui.dt_shap_heatmap)
        self.initial_shap_plot(self.ui.dt_shap_scatter)
        self.initial_shap_plot(self.ui.dt_shap_waterfall)
        self.initial_shap_plot(self.ui.dt_shap_decision)

        self._initial_dt_force_single_plot()
        self._initial_dt_force_full_plot()

    def _initial_nb_plots(self) -> None:
        self.initial_stat_plot(self.ui.nb_dm_plot)
        self.initial_stat_plot(self.ui.nb_pr_plot)
        self.initial_stat_plot(self.ui.nb_roc_plot)
        self.initial_scores_plot(self.ui.nb_scores_plot_widget)
        self.initial_shap_plot(self.ui.nb_shap_means)
        self.initial_shap_plot(self.ui.nb_shap_beeswarm)
        self.initial_shap_plot(self.ui.nb_shap_heatmap)
        self.initial_shap_plot(self.ui.nb_shap_scatter)
        self.initial_shap_plot(self.ui.nb_shap_waterfall)
        self.initial_shap_plot(self.ui.nb_shap_decision)

        self._initial_nb_force_single_plot()
        self._initial_nb_force_full_plot()

    def _initial_rf_plots(self) -> None:
        self.initial_stat_plot(self.ui.rf_dm_plot)
        self.initial_stat_plot(self.ui.rf_pr_plot)
        self.initial_stat_plot(self.ui.rf_tree_plot_widget)
        self.initial_stat_plot(self.ui.rf_roc_plot)
        self.initial_scores_plot(self.ui.rf_scores_plot_widget)
        self.initial_stat_plot(self.ui.rf_features_plot_widget)
        self.initial_shap_plot(self.ui.rf_shap_means)
        self.initial_shap_plot(self.ui.rf_shap_beeswarm)
        self.initial_shap_plot(self.ui.rf_shap_heatmap)
        self.initial_shap_plot(self.ui.rf_shap_scatter)
        self.initial_shap_plot(self.ui.rf_shap_waterfall)
        self.initial_shap_plot(self.ui.rf_shap_decision)

        self._initial_rf_force_single_plot()
        self._initial_rf_force_full_plot()

    def _initial_ab_plots(self) -> None:
        self.initial_stat_plot(self.ui.ab_dm_plot)
        self.initial_stat_plot(self.ui.ab_roc_plot)
        self.initial_stat_plot(self.ui.ab_pr_plot)
        self.initial_stat_plot(self.ui.ab_tree_plot_widget)
        self.initial_scores_plot(self.ui.ab_scores_plot_widget)
        self.initial_stat_plot(self.ui.ab_features_plot_widget)
        self.initial_shap_plot(self.ui.ab_shap_means)
        self.initial_shap_plot(self.ui.ab_shap_beeswarm)
        self.initial_shap_plot(self.ui.ab_shap_heatmap)
        self.initial_shap_plot(self.ui.ab_shap_scatter)
        self.initial_shap_plot(self.ui.ab_shap_waterfall)
        self.initial_shap_plot(self.ui.ab_shap_decision)
        self._initial_ab_force_single_plot()
        self._initial_ab_force_full_plot()

    def _initial_mlp_plots(self) -> None:
        self.initial_stat_plot(self.ui.mlp_dm_plot)
        self.initial_stat_plot(self.ui.mlp_roc_plot)
        self.initial_stat_plot(self.ui.mlp_pr_plot)
        self.initial_scores_plot(self.ui.mlp_scores_plot_widget)
        self.initial_shap_plot(self.ui.mlp_shap_means)
        self.initial_shap_plot(self.ui.mlp_shap_beeswarm)
        self.initial_shap_plot(self.ui.mlp_shap_heatmap)
        self.initial_shap_plot(self.ui.mlp_shap_scatter)
        self.initial_shap_plot(self.ui.mlp_shap_waterfall)
        self.initial_shap_plot(self.ui.mlp_shap_decision)
        self._initial_mlp_force_single_plot()
        self._initial_mlp_force_full_plot()

    def _initial_torch_plots(self) -> None:
        self.initial_stat_plot(self.ui.torch_dm_plot)
        self.initial_stat_plot(self.ui.torch_roc_plot)
        self.initial_stat_plot(self.ui.torch_pr_plot)
        self.initial_scores_plot(self.ui.torch_scores_plot_widget)
        self.initial_shap_plot(self.ui.torch_shap_means)
        self.initial_shap_plot(self.ui.torch_shap_beeswarm)
        self.initial_shap_plot(self.ui.torch_shap_heatmap)
        self.initial_shap_plot(self.ui.torch_shap_scatter)
        self.initial_shap_plot(self.ui.torch_shap_waterfall)
        self.initial_shap_plot(self.ui.torch_shap_decision)
        self._initial_torch_force_single_plot()
        self._initial_torch_force_full_plot()

    def _initial_xgboost_plots(self) -> None:
        self.initial_stat_plot(self.ui.xgboost_dm_plot)
        self.initial_stat_plot(self.ui.xgboost_roc_plot)
        self.initial_stat_plot(self.ui.xgboost_pr_plot)
        self.initial_scores_plot(self.ui.xgboost_scores_plot_widget)
        self.initial_stat_plot(self.ui.xgboost_features_plot_widget)
        self.initial_stat_plot(self.ui.xgboost_tree_plot_widget)
        self.initial_shap_plot(self.ui.xgboost_shap_means)
        self.initial_shap_plot(self.ui.xgboost_shap_beeswarm)
        self.initial_shap_plot(self.ui.xgboost_shap_heatmap)
        self.initial_shap_plot(self.ui.xgboost_shap_scatter)
        self.initial_shap_plot(self.ui.xgboost_shap_waterfall)
        self.initial_shap_plot(self.ui.xgboost_shap_decision)
        self._initial_xgboost_force_single_plot()
        self._initial_xgboost_force_full_plot()

    def _initial_pca_plots(self) -> None:
        self.initial_scores_plot(self.ui.pca_scores_plot_widget)
        self.initial_scores_plot(self.ui.pca_loadings_plot_widget)

    def _initial_plsda_plots(self) -> None:
        self.initial_scores_plot(self.ui.plsda_scores_plot_widget, 'PLS-DA')
        self.initial_scores_plot(self.ui.plsda_vip_plot_widget, 'PLS-DA')

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
        self._initial_xgboost_plots()
        self._initial_torch_plots()
        self._initial_pca_plots()
        self._initial_plsda_plots()

        self.initial_plots_set_fonts()
        self.initial_plots_set_labels_font()
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
        self.ui.splitter_93.setStretchFactor(1, 10)

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
        self.ui.splitter_95.setStretchFactor(3, 2)

        self.ui.lda_scores_1d_plot_widget.setVisible(False)

    def initial_plots_set_fonts(self) -> None:
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

        plt.rcParams.update({'font.size': self.plot_font_size})

    def initial_plots_set_labels_font(self) -> None:
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
        self.file_menu_help.triggered.connect(action_help)
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
        self.clear_menu.addAction('Fitting lines', self.clear_all_deconv_lines)
        self.clear_menu.addAction('All fitting data', lambda: self.clear_selected_step('Deconvolution'))
        self.clear_menu.addSeparator()
        self.clear_menu.addAction('Smoothed dataset', self._initial_smoothed_dataset_table)
        self.clear_menu.addAction('Baseline corrected dataset', self._initial_baselined_dataset_table)
        self.clear_menu.addAction('Deconvoluted dataset', self._initial_deconvoluted_dataset_table)
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
        self.clear_menu.addAction('XGBoost', lambda: self.clear_selected_step('XGBoost'))
        # self.clear_menu.addAction('Torch', lambda: self.clear_selected_step('Torch'))
        self.clear_menu.addAction('PCA', lambda: self.clear_selected_step('PCA'))
        self.clear_menu.addAction('PLS-DA', lambda: self.clear_selected_step('PLS-DA'))
        self.clear_menu.addSeparator()
        self.clear_menu.addAction('Predicted', lambda: self.clear_selected_step('Page5'))
        self.ui.EditBtn.setMenu(self.edit_menu)

    def _init_process_menu(self) -> None:
        self.process_menu = QMenu(self)
        self.action_interpolate = QAction('Interpolate')
        self.action_interpolate.triggered.connect(lambda: self.preprocessing.interpolate())
        self.action_interpolate.setShortcut("Alt+I")
        self.action_despike = QAction('Despike')
        self.action_despike.triggered.connect(lambda: self.preprocessing.despike())
        self.action_despike.setShortcut("Alt+D")
        self.action_convert = QAction('Convert to cm\N{superscript minus}\N{superscript one}')
        self.action_convert.triggered.connect(lambda: self.preprocessing.convert())
        self.action_convert.setShortcut("Alt+C")
        self.action_cut = QAction('Cut spectrum')
        self.action_cut.triggered.connect(lambda: self.preprocessing.cut_first())
        self.action_cut.setShortcut("Alt+U")
        self.action_normalize = QAction('Normalization')
        self.action_normalize.triggered.connect(lambda: self.preprocessing.normalize())
        self.action_normalize.setShortcut("Alt+N")
        self.action_smooth = QAction('Smooth')
        self.action_smooth.triggered.connect(lambda: self.preprocessing.smooth())
        self.action_smooth.setShortcut("Alt+S")
        self.action_baseline_correction = QAction('Baseline correction')
        self.action_baseline_correction.triggered.connect(lambda: self.preprocessing.baseline_correction())
        self.action_baseline_correction.setShortcut("Alt+B")
        self.action_trim = QAction('Final trim')
        self.action_trim.triggered.connect(lambda: self.preprocessing.trim())
        self.action_trim.setShortcut("Alt+T")
        self.action_average = QAction('Average')
        self.action_average.triggered.connect(lambda: self.preprocessing.update_averaged())
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
        self.action_fit_xgboost = QAction('XGBoost')
        self.action_fit_xgboost.triggered.connect(lambda: self.fit_classificator('XGBoost'))
        self.action_fit_torch = QAction('Torch')
        self.action_fit_torch.triggered.connect(lambda: self.fit_classificator('Torch'))
        self.action_fit_pca = QAction('PCA')
        self.action_fit_pca.triggered.connect(lambda: self.fit_classificator('PCA'))
        self.action_fit_plsda = QAction('PLS-DA')
        self.action_fit_plsda.triggered.connect(lambda: self.fit_classificator('PLS-DA'))
        self.action_redraw_plots = QAction('Redraw plots')
        self.action_redraw_plots.triggered.connect(self.redraw_stat_plots)
        actions = [self.action_fit_lda, self.action_fit_qda, self.action_fit_lr, self.action_fit_svc,
                   self.action_fit_sgd, self.action_fit_gpc, self.action_fit_dt, self.action_fit_nb,
                   self.action_fit_rf, self.action_fit_ab, self.action_fit_mlp, self.action_fit_xgboost,
                   self.action_fit_pca, self.action_fit_plsda, self.action_redraw_plots]
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
        self._init_average_function_cb()
        self._init_dataset_type_cb()
        self._init_refit_score()
        self._init_solver_mlp_combo_box()
        self._init_activation_combo_box()
        self._init_current_feature_cb()
        self._init_coloring_feature_cb()
        self._init_current_tree_sb()
        self._init_use_pca_cb()
        self._init_include_x0_chb()
        self.ui.edit_template_btn.clicked.connect(lambda: self.fitting.switch_to_template())
        self.ui.template_combo_box.currentTextChanged.connect(lambda: self.fitting.switch_to_template())
        self.ui.current_group_shap_comboBox.currentTextChanged.connect(self.current_group_shap_changed)
        self.ui.intervals_gb.toggled.connect(self.intervals_gb_toggled)
        self._init_smoothing_method_combo_box()
        self.normalization_method = self.default_values['normalizing_method_comboBox']
        self.smooth_method = ''
        self.baseline_method = ''

    def _init_baseline_correction_method_combo_box(self) -> None:
        for i in self.preprocessing.baseline_methods.keys():
            self.ui.baseline_correction_method_comboBox.addItem(i)

    def _init_cost_func_combo_box(self) -> None:
        self.ui.cost_func_comboBox.addItem('asymmetric_truncated_quadratic')
        self.ui.cost_func_comboBox.addItem('symmetric_truncated_quadratic')
        self.ui.cost_func_comboBox.addItem('asymmetric_huber')
        self.ui.cost_func_comboBox.addItem('symmetric_huber')
        self.ui.cost_func_comboBox.addItem('asymmetric_indec')
        self.ui.cost_func_comboBox.addItem('symmetric_indec')

    def _init_normalizing_method_combo_box(self) -> None:
        for i in self.preprocessing.normalize_methods.keys():
            self.ui.normalizing_method_comboBox.addItem(i)

    def _init_opt_method_oer_combo_box(self) -> None:
        for i in optimize_extended_range_methods():
            self.ui.opt_method_oer_comboBox.addItem(i)

    def _init_fit_opt_method_combo_box(self) -> None:
        for key in self.fitting.fitting_methods:
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
        self.ui.max_epoch_spinBox.mouseDoubleClickEvent = self._max_epoch_spin_box_mouse_dce
        self.ui.learning_rate_doubleSpinBox.mouseDoubleClickEvent = self._learning_rate_double_spin_box_mouse_dce


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
        self.ui.spline_degree_spinBox.valueChanged.connect(self.set_modified)
        self.ui.trim_start_cm.valueChanged.connect(self._trim_start_change_event)
        self.ui.trim_end_cm.valueChanged.connect(self._trim_end_change_event)
        self.ui.updateRangebtn.clicked.connect(self.update_range_btn_clicked)
        self.ui.updateTrimRangebtn.clicked.connect(self.update_trim_range_btn_clicked)
        self.ui.whittaker_lambda_spinBox.valueChanged.connect(self.set_modified)
        self.ui.window_length_spinBox.valueChanged.connect(self.set_smooth_polyorder_bound)
        self.ui.guess_method_cb.currentTextChanged.connect(self.guess_method_cb_changed)

    def _init_smoothing_method_combo_box(self) -> None:
        for i in self.preprocessing.smoothing_methods.keys():
            self.ui.smoothing_method_comboBox.addItem(i)

    def _init_guess_method_cb(self) -> None:
        self.ui.guess_method_cb.addItem('Average')
        self.ui.guess_method_cb.addItem('Average groups')
        self.ui.guess_method_cb.addItem('All')

    def _init_average_function_cb(self) -> None:
        self.ui.average_method_cb.addItem('Mean')
        self.ui.average_method_cb.addItem('Median')

    def _init_refit_score(self) -> None:
        self.ui.refit_score.addItem('precision_score')
        self.ui.refit_score.addItem('recall_score')
        self.ui.refit_score.addItem('accuracy_score')
        self.ui.refit_score.addItem('f1_score')

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
        if model.rowCount() == 0 or self.predict_logic.is_production_project:
            self.ui.dataset_features_n.setText('')
            return
        q_res = model.dataframe()
        features_names = list(q_res.columns[2:])
        n_features = len(features_names)
        self.ui.dataset_features_n.setText('%s features' % n_features)
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
        self.set_modified()

    @asyncSlot()
    async def current_group_shap_changed(self, g: str = '') -> None:
        await self.loop.run_in_executor(None, self._update_shap_plots)
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        self.stat_analysis_logic.update_force_single_plots()
        self.stat_analysis_logic.update_force_full_plots()

    @asyncSlot()
    async def current_instance_changed(self, _: str = '') -> None:
        await self.loop.run_in_executor(None, self._update_shap_plots_by_instance)
        self.stat_analysis_logic.update_force_single_plots()

    def _init_activation_combo_box(self) -> None:
        items = ['identity', 'logistic', 'tanh', 'relu']
        for i in items:
            self.ui.activation_comboBox.addItem(i)

    def _init_solver_mlp_combo_box(self) -> None:
        items = ['lbfgs', 'sgd', 'adam']
        for i in items:
            self.ui.solver_mlp_combo_box.addItem(i)

    def _init_current_tree_sb(self) -> None:
        self.ui.current_tree_spinBox.valueChanged.connect(self.current_tree_sb_changed)

    def _init_use_pca_cb(self) -> None:
        self.ui.use_pca_checkBox.stateChanged.connect(self.use_pca_cb_changed)

    def use_pca_cb_changed(self, b: bool):
        if b:
            self.ui.use_pca_checkBox.setText('PCA dimensional reduction')
        else:
            self.ui.use_pca_checkBox.setText('PLS-DA dimensional reduction')

    def _init_include_x0_chb(self) -> None:
        self.ui.include_x0_checkBox.stateChanged.connect(self.include_x0_chb_changed)

    def include_x0_chb_changed(self, b: bool):
        self.set_deconvoluted_dataset()

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

    def _learning_rate_double_spin_box_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.learning_rate_doubleSpinBox.setValue(self.default_values['learning_rate_doubleSpinBox'])
    def _max_epoch_spin_box_mouse_dce(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.MiddleButton:
            self.ui.max_epoch_spinBox.setValue(self.default_values['max_epoch_spinBox'])

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
        self._initial_ignore_dataset_table()
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
            self.time_start = datetime.now()
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
        self.ui.dec_table.doubleClicked.connect(lambda: self.fitting.dec_table_double_clicked())

    def _reset_dec_table(self) -> None:
        df = DataFrame(columns=['Filename'])
        model = PandasModelDeconvTable(df)
        self.ui.dec_table.setModel(model)

    def _dec_table_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('To template', self.to_template_clicked)
        menu.addAction('Copy spectrum lines parameters from template',
                       self.fitting.copy_spectrum_lines_parameters_from_template)
        menu.move(a0.globalPos())
        menu.show()

    @asyncSlot()
    async def to_template_clicked(self) -> None:
        selected_rows = self.ui.dec_table.selectionModel().selectedRows()
        if len(selected_rows) == 0:
            return
        selected_filename = self.ui.dec_table.model().cell_data_by_index(selected_rows[0])
        self.fitting.update_single_deconvolution_plot(selected_filename, True)

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
        self.fitting.set_rows_visibility()
        row = selected_indexes[0].row()
        idx = self.ui.deconv_lines_table.model().index_by_row(row)
        if self.fitting.updating_fill_curve_idx is not None \
                and self.fitting.updating_fill_curve_idx in self.ui.deconv_lines_table.model().dataframe().index:
            curve_style = self.ui.deconv_lines_table.model().cell_data_by_idx_col_name(
                self.fitting.updating_fill_curve_idx,
                'Style')
            self.fitting.update_curve_style(self.fitting.updating_fill_curve_idx, curve_style)
        self.fitting.start_fill_timer(idx)

    def deconv_lines_table_key_pressed(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Delete and self.ui.deconv_lines_table.selectionModel().currentIndex().row() > -1 \
                and len(self.ui.deconv_lines_table.selectionModel().selectedIndexes()) and self.fitting.is_template:
            self.time_start = datetime.now()
            command = CommandDeleteDeconvLines(self, "Delete line")
            self.undoStack.push(command)

    def _reset_deconv_lines_table(self) -> None:
        df = DataFrame(columns=['Legend', 'Type', 'Style'])
        model = PandasModelDeconvLinesTable(self, df, [])
        self.ui.deconv_lines_table.setSortingEnabled(True)
        self.ui.deconv_lines_table.setModel(model)
        combobox_delegate = ComboDelegate(self.fitting.peak_shapes_params.keys())
        self.ui.deconv_lines_table.setItemDelegateForColumn(1, combobox_delegate)
        self.ui.deconv_lines_table.model().sigCheckedChanged.connect(self.fitting.show_hide_curve)
        combobox_delegate.sigLineTypeChanged.connect(lambda: self.fitting.curve_type_changed())
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
        self.fitting.deselect_selected_line()

    def _deconv_lines_table_context_menu_event(self, a0: QContextMenuEvent) -> None:
        line = QLineEdit(self)
        menu = QMenu(line)
        # noinspection PyTypeChecker
        menu.addAction('Delete line', self.delete_line_clicked)
        menu.addAction('Clear table', self.clear_all_deconv_lines)
        menu.move(a0.globalPos())
        menu.show()

    @asyncSlot()
    async def delete_line_clicked(self) -> None:
        if not self.fitting.is_template:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Deleting lines is only possible in template mode.")
            msg.setInformativeText('Press the Template button')
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
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
        if self.fitting.is_template:
            return
        line = QLineEdit(self)
        menu = QMenu(line)
        menu.addAction('Copy line parameters from template', lambda: self.fitting.copy_line_parameters_from_template())
        menu.move(a0.globalPos())
        menu.show()

    # endregion

    # region pca plsda  features
    def _initial_pca_features_table(self) -> None:
        self._reset_pca_features_table()

    def _reset_pca_features_table(self) -> None:
        df = DataFrame(columns=['feature', 'PC-1', 'PC-2'])
        model = PandasModelPCA(self, df)
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

    def _initial_deconvoluted_dataset_table(self) -> None:
        self._reset_deconvoluted_dataset_table()
        self.ui.deconvoluted_dataset_table_view.verticalScrollBar().valueChanged.connect(self.move_side_scrollbar)

    def _reset_deconvoluted_dataset_table(self) -> None:
        df = DataFrame(columns=['Class', 'Filename'])
        model = PandasModelDeconvolutedDataset(self, df)
        self.ui.deconvoluted_dataset_table_view.setModel(model)

    # endregion

    # region ignore features dataset

    def _initial_ignore_dataset_table(self) -> None:
        self._reset_ignore_dataset_table()
        self.ui.ignore_dataset_table_view.verticalScrollBar().valueChanged.connect(self.move_side_scrollbar)
        self.ui.ignore_dataset_table_view.verticalHeader().setVisible(False)

    def _reset_ignore_dataset_table(self) -> None:
        df = DataFrame(columns=['Feature'])
        model = PandasModelIgnoreDataset(self, df, {})
        self.ui.ignore_dataset_table_view.setModel(model)

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
        for line_type in self.fitting.peak_shapes_params.keys():
            action = guess_menu.addAction(line_type)
            action.triggered.connect(lambda checked=None, line=line_type: self.guess(line_type=line))
        self.ui.guess_button.setMenu(guess_menu)
        self.ui.guess_button.menu()  # some kind of magik

    def _initial_add_line_button(self) -> None:
        add_lines_menu = QMenu()
        line_type: str
        for line_type in self.fitting.peak_shapes_params.keys():
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
        if self.fitting.fill is None:
            return
        self.fitting.fill.setVisible(a0 == 2)

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
        if self.fitting.data_curve is None:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 999 and obj.isVisible():
                return
        data_curve_prop_window = CurvePropertiesWindow(self, self.fitting.data_style, 999, False)
        data_curve_prop_window.sigStyleChanged.connect(self._update_data_curve_style)
        data_curve_prop_window.show()

    def sum_pb_clicked(self) -> None:
        if self.fitting.sum_curve is None:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 998 and obj.isVisible():
                return
        sum_curve_prop_window = CurvePropertiesWindow(self, self.fitting.sum_style, 998, False)
        sum_curve_prop_window.sigStyleChanged.connect(self._update_sum_curve_style)
        sum_curve_prop_window.show()

    def residual_pb_clicked(self) -> None:
        if self.fitting.residual_curve is None:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 997 and obj.isVisible():
                return
        residual_curve_prop_window = CurvePropertiesWindow(self, self.fitting.residual_style, 997, False)
        residual_curve_prop_window.sigStyleChanged.connect(self._update_residual_curve_style)
        residual_curve_prop_window.show()

    def sigma3_push_button_clicked(self) -> None:
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == 996 and obj.isVisible():
                return
        prop_window = CurvePropertiesWindow(self, self.fitting.sigma3_style, 996, True)
        prop_window.sigStyleChanged.connect(self._update_sigma3_style)
        prop_window.show()

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
        if self.preprocessing.averaged_dict != {} and self.preprocessing.averaged_dict:
            x_axis = next(iter(self.preprocessing.averaged_dict.values()))[:, 0]
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
        if self.preprocessing.averaged_dict != {} and self.preprocessing.averaged_dict:
            x_axis = next(iter(self.preprocessing.averaged_dict.values()))[:, 0]
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
        self.fitting.redraw_curves_for_filename()
        self.fitting.draw_sum_curve()
        self.fitting.draw_residual_curve()

    @asyncSlot()
    async def uncut_data_sum_residual(self) -> None:
        self.uncut_data()
        self.fitting.redraw_curves_for_filename()
        self.fitting.draw_sum_curve()
        self.fitting.draw_residual_curve()

    def cut_data_interval(self) -> None:
        n_array = self.fitting.array_of_current_filename_in_deconvolution()
        if n_array is None:
            return
        n_array = cut_full_spectrum(n_array, self.ui.interval_start_dsb.value(), self.ui.interval_end_dsb.value())
        self.fitting.data_curve.setData(x=n_array[:, 0], y=n_array[:, 1])

    def uncut_data(self) -> None:
        n_array = self.fitting.array_of_current_filename_in_deconvolution()
        if n_array is None:
            return
        self.fitting.data_curve.setData(x=n_array[:, 0], y=n_array[:, 1])

    # endregion

    # region other

    def stat_tab_widget_tab_changed(self, i: int):
        self.stat_analysis_logic.update_stat_report_text()
        self.decide_vertical_scroll_bar_visible()
        self.ui.groupBox_mlp.setVisible(i == 10 or i == 12)
        # self.ui.max_epoch_spinBox.setVisible(i == 10 or i == 12)
        # self.ui.learning_rate_doubleSpinBox.setVisible(i == 10 or i == 12)
        # self.ui.label_max_epoch.setVisible(i == 10 or i == 12)
        # self.ui.label_learning_rate.setVisible(i == 10 or i == 12)
        if i == 8 and 'Random Forest' in self.stat_analysis_logic.latest_stat_result:
            model = self.stat_analysis_logic.latest_stat_result['Random Forest']['model']
            n_trees = len(model.best_estimator_.estimators_)
            self.ui.current_tree_spinBox.setVisible(True)
            self.ui.current_tree_label.setVisible(True)
            self.ui.current_tree_spinBox.setMaximum(n_trees - 1)
        elif i == 11 and 'XGBoost' in self.stat_analysis_logic.latest_stat_result:
            model = self.stat_analysis_logic.latest_stat_result['XGBoost']['model']
            n_trees = len(model.best_estimator_.get_booster().get_dump())
            self.ui.current_tree_spinBox.setVisible(True)
            self.ui.current_tree_label.setVisible(True)
            self.ui.current_tree_spinBox.setMaximum(n_trees - 1)
        else:
            self.ui.current_tree_spinBox.setVisible(False)
            self.ui.current_tree_label.setVisible(False)
            self.ui.current_tree_spinBox.setMaximum(0)

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
        self.ui.scrollArea_xgboost.verticalScrollBar().valueChanged.connect(self.scroll_area_stat_value_changed)

    def scroll_area_stat_value_changed(self, event: int):
        self.ui.verticalScrollBar.setValue(event)

    def initial_plot_buttons(self) -> None:
        self.ui.crosshairBtn.clicked.connect(self.crosshair_btn_clicked)
        self.ui.by_one_control_button.clicked.connect(self.by_one_control_button_clicked)
        self.ui.by_group_control_button.clicked.connect(self.by_group_control_button)
        self.ui.by_group_control_button.mouseDoubleClickEvent = self.by_group_control_button_double_clicked
        self.ui.all_control_button.clicked.connect(self.all_control_button)
        self.ui.despike_history_Btn.clicked.connect(self.despike_history_btn_clicked)
        self.ui.lr_movableBtn.clicked.connect(self.linear_region_movable_btn_clicked)
        self.ui.lr_showHideBtn.clicked.connect(self.linear_region_show_hide_btn_clicked)
        self.ui.sun_Btn.clicked.connect(self.change_plots_bckgrnd)

    def initial_timers(self) -> None:
        self.timer_mem_update = QTimer(self)
        self.timer_mem_update.timeout.connect(self.set_timer_memory_update)
        self.timer_mem_update.start(1000)
        self.cpu_load = QTimer(self)
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
        self.ui.titlebar.mouseMoveEvent = self.move_window
        self.ui.titlebar.mouseReleaseEvent = self.titlebar_mouse_release_event
        self.ui.minimizeAppBtn.clicked.connect(lambda: self.showMinimized())
        self.ui.maximizeRestoreAppBtn.clicked.connect(lambda: self.maximize_restore())
        self.ui.closeBtn.clicked.connect(lambda: self.close())
        self.ui.settingsBtn.clicked.connect(lambda: self.setting_window.show())

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
            if 'Light' in self.theme:
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
            if 'Light' in self.theme:
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
        self.initial_plots_set_labels_font()

    # endregion

    # endregion

    # region Plot buttons

    # region crosshair button
    def linear_region_movable_btn_clicked(self) -> None:
        b = not self.ui.lr_movableBtn.isChecked()
        self.linearRegionCmConverted.setMovable(b)
        self.linearRegionBaseline.setMovable(b)
        self.linearRegionDeconv.setMovable(b)

    def linear_region_show_hide_btn_clicked(self) -> None:
        if self.ui.lr_showHideBtn.isChecked():
            self.converted_cm_widget_plot_item.addItem(self.linearRegionCmConverted)
            self.baseline_corrected_plotItem.addItem(self.linearRegionBaseline)
            self.deconvolution_plotItem.addItem(self.linearRegionDeconv)
        else:
            self.converted_cm_widget_plot_item.removeItem(self.linearRegionCmConverted)
            self.baseline_corrected_plotItem.removeItem(self.linearRegionBaseline)
            self.deconvolution_plotItem.removeItem(self.linearRegionDeconv)

    def crosshair_btn_clicked(self) -> None:
        """Add crosshair with coordinates at title."""

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
            if len(self.preprocessing.NormalizedDict) > 0:
                text_for_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors['plotText'] \
                                 + ";font-size:14pt\">Normalized plots. Method " \
                                 + self.normalization_method + "</span>"
                self.ui.normalize_plot_widget.setTitle(text_for_title)
                info('normalize_plot_widget title is %s',
                     'Normalized plots. Method '
                     + self.normalization_method)
            else:
                if len(self.preprocessing.NormalizedDict) > 0:
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
            if len(self.preprocessing.smoothed_spectra) > 0:
                text_for_title = "<span style=\"font-family: AbletonSans; color:" + self.theme_colors['plotText'] \
                                 + ";font-size:14pt\">Smoothed plots. Method " + self.smooth_method + "</span>"
                self.ui.smooth_plot_widget.setTitle(text_for_title)

            elif len(self.preprocessing.smoothed_spectra) > 0:
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
            if len(self.preprocessing.baseline_corrected_dict) > 0:
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
                'plotText'] + ";font-size:14pt\">" + self.fitting.current_spectrum_deconvolution_name + "</span>"
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
        if self.preprocessing.curveOneInputPlot:
            self.input_plot_widget_plot_item.removeItem(self.preprocessing.curveOneInputPlot)
        self.preprocessing.curveOneInputPlot = self.get_curve_plot_data_item(arr, group_number)
        self.input_plot_widget_plot_item.addItem(self.preprocessing.curveOneInputPlot,
                                                 kargs=['ignoreBounds', 'skipAverage'])
        if self.ui.despike_history_Btn.isChecked() and self.ui.input_table.selectionModel().currentIndex().row() != -1 \
                and self.preprocessing.BeforeDespike and len(self.preprocessing.BeforeDespike) > 0 \
                and current_spectrum_name in self.preprocessing.BeforeDespike:
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
        arr = self.preprocessing.ConvertedDict[current_spectrum_name]
        if self.preprocessing.curveOneConvertPlot:
            self.converted_cm_widget_plot_item.removeItem(self.preprocessing.curveOneConvertPlot)
        self.preprocessing.curveOneConvertPlot = self.get_curve_plot_data_item(arr, group_number)
        self.converted_cm_widget_plot_item.addItem(self.preprocessing.curveOneConvertPlot,
                                                   kargs=['ignoreBounds', 'skipAverage'])
        self.converted_cm_widget_plot_item.getViewBox().updateAutoRange()

    async def update_single_cut_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_cut_plot = self.cut_cm_plotItem.listDataItems()
        if len(data_items_cut_plot) <= 0:
            return
        self.ui.cut_cm_plot_widget.setTitle(new_title)
        for i in data_items_cut_plot:
            i.setVisible(False)
        arr = self.preprocessing.CuttedFirstDict[current_spectrum_name]
        if self.preprocessing.curveOneCutPlot:
            self.cut_cm_plotItem.removeItem(self.preprocessing.curveOneCutPlot)
        self.preprocessing.curveOneCutPlot = self.get_curve_plot_data_item(arr, group_number)
        self.cut_cm_plotItem.addItem(self.preprocessing.curveOneCutPlot, kargs=['ignoreBounds', 'skipAverage'])
        self.cut_cm_plotItem.getViewBox().updateAutoRange()

    async def update_single_normal_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_normal_plot = self.normalize_plotItem.listDataItems()
        if len(data_items_normal_plot) <= 0:
            return
        self.ui.normalize_plot_widget.setTitle(new_title)
        for i in data_items_normal_plot:
            i.setVisible(False)
        arr = self.preprocessing.NormalizedDict[current_spectrum_name]
        if self.preprocessing.curveOneNormalPlot:
            self.normalize_plotItem.removeItem(self.preprocessing.curveOneNormalPlot)
        self.preprocessing.curveOneNormalPlot = self.get_curve_plot_data_item(arr, group_number)
        self.normalize_plotItem.addItem(self.preprocessing.curveOneNormalPlot, kargs=['ignoreBounds', 'skipAverage'])
        self.normalize_plotItem.getViewBox().updateAutoRange()

    async def update_single_smooth_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_smooth_plot = self.smooth_plotItem.listDataItems()
        if len(data_items_smooth_plot) <= 0:
            return
        self.ui.smooth_plot_widget.setTitle(new_title)
        for i in data_items_smooth_plot:
            i.setVisible(False)
        arr = self.preprocessing.smoothed_spectra[current_spectrum_name]
        if self.preprocessing.curveOneSmoothPlot:
            self.smooth_plotItem.removeItem(self.preprocessing.curveOneSmoothPlot)
        self.preprocessing.curveOneSmoothPlot = self.get_curve_plot_data_item(arr, group_number)
        self.smooth_plotItem.addItem(self.preprocessing.curveOneSmoothPlot, kargs=['ignoreBounds', 'skipAverage'])
        self.smooth_plotItem.getViewBox().updateAutoRange()

    async def update_single_baseline_plot(self, new_title: str, current_spectrum_name: str, group_number: str) -> None:
        data_items_baseline_plot = self.baseline_corrected_plotItem.listDataItems()
        if len(data_items_baseline_plot) <= 0:
            return
        self.ui.baseline_plot_widget.setTitle(new_title)
        for i in data_items_baseline_plot:
            i.setVisible(False)
        arr = self.preprocessing.baseline_corrected_dict[current_spectrum_name]
        if self.preprocessing.curve_one_baseline_plot:
            self.baseline_corrected_plotItem.removeItem(self.preprocessing.curve_one_baseline_plot)
        self.preprocessing.curve_one_baseline_plot = self.get_curve_plot_data_item(arr, group_number)
        self.baseline_corrected_plotItem.addItem(self.preprocessing.curve_one_baseline_plot,
                                                 kargs=['ignoreBounds', 'skipAverage'])
        if self.preprocessing.baseline_dict and len(self.preprocessing.baseline_dict) > 0 \
                and self.ui.input_table.selectionModel().currentIndex().row() != -1 \
                and current_spectrum_name in self.preprocessing.baseline_dict:
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
            await self.loop.run_in_executor(None, self.despike_history_remove_plot)
            await self.loop.run_in_executor(None, self.baseline_remove_plot)
            if self.ui.GroupsTable.selectionModel().currentIndex().column() == 0:
                await self.update_plots_for_group(None)
        else:
            self.ui.by_group_control_button.setChecked(True)

    @asyncSlot()
    async def by_group_control_button_double_clicked(self, _=None) -> None:
        if self.ui.GroupsTable.model().rowCount() < 2:
            return
        input_dialog = QInputDialog(self)
        result = input_dialog.getText(self, "Choose visible groups", 'Write groups numbers to show (example: 1, 2, 3):')
        if not result[1]:
            return
        v = list(result[0].strip().split(','))
        await self.loop.run_in_executor(None, self.despike_history_remove_plot)
        await self.loop.run_in_executor(None, self.baseline_remove_plot)
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
        if self.preprocessing.curveOneInputPlot:
            self.input_plot_widget_plot_item.removeItem(self.preprocessing.curveOneInputPlot)
        self.input_plot_widget_plot_item.getViewBox().updateAutoRange()

    def update_all_converted_plot(self) -> None:
        self.ui.converted_cm_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Converted to cm\N{superscript minus}\N{superscript one}</span>")
        data_items = self.converted_cm_widget_plot_item.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.preprocessing.curveOneConvertPlot:
            self.converted_cm_widget_plot_item.removeItem(self.preprocessing.curveOneConvertPlot)
        self.converted_cm_widget_plot_item.getViewBox().updateAutoRange()

    def update_all_cut_plot(self) -> None:
        self.ui.cut_cm_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
            'plotText'] + ";font-size:14pt\">Cutted plots</span>")
        data_items = self.cut_cm_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.preprocessing.curveOneCutPlot:
            self.cut_cm_plotItem.removeItem(self.preprocessing.curveOneCutPlot)
        self.cut_cm_plotItem.getViewBox().updateAutoRange()

    def update_all_normal_plot(self) -> None:
        if len(self.preprocessing.NormalizedDict) > 0:
            self.ui.normalize_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Normalized plots. Method " + self.normalization_method + "</span>")
        else:
            if len(self.preprocessing.NormalizedDict) > 0:
                self.ui.normalize_plot_widget.setTitle(
                    "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                        'plotText'] + ";font-size:14pt\">Normalized plots</span>")
        data_items = self.normalize_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.preprocessing.curveOneNormalPlot:
            self.normalize_plotItem.removeItem(self.preprocessing.curveOneNormalPlot)
        self.normalize_plotItem.getViewBox().updateAutoRange()

    def update_all_smooth_plot(self) -> None:
        if len(self.preprocessing.smoothed_spectra) > 0:
            self.ui.smooth_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Smoothed plots. Method " + self.smooth_method + "</span>")
        else:
            if len(self.preprocessing.smoothed_spectra) > 0:
                self.ui.smooth_plot_widget.setTitle(
                    "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                        'plotText'] + ";font-size:14pt\">Smoothed plots</span>")
        data_items = self.smooth_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.preprocessing.curveOneSmoothPlot:
            self.smooth_plotItem.removeItem(self.preprocessing.curveOneSmoothPlot)
        self.smooth_plotItem.getViewBox().updateAutoRange()

    def update_all_baseline_plot(self) -> None:
        if len(self.preprocessing.baseline_corrected_dict) > 0:
            self.ui.baseline_plot_widget.setTitle("<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                'plotText'] + ";font-size:14pt\">Baseline corrected. Method " + self.baseline_method + "</span>")
        else:
            self.ui.smooth_plot_widget.setTitle(
                "<span style=\"font-family: AbletonSans; color:" + self.theme_colors[
                    'plotText'] + ";font-size:14pt\">Smoothed </span>")
        data_items = self.baseline_corrected_plotItem.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.preprocessing.curve_one_baseline_plot:
            self.baseline_corrected_plotItem.removeItem(self.preprocessing.curve_one_baseline_plot)
        self.baseline_corrected_plotItem.getViewBox().updateAutoRange()

    def update_all_average_plot(self) -> None:
        if len(self.preprocessing.averaged_dict) > 0:
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
        if self.ui.despike_history_Btn.isChecked() \
                and self.current_spectrum_despiked_name in self.preprocessing.BeforeDespike:
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
        arr = self.preprocessing.BeforeDespike[self.current_spectrum_despiked_name]
        if self.preprocessing.curveDespikedHistory:
            self.input_plot_widget_plot_item.removeItem(self.preprocessing.curveDespikedHistory)
        self.preprocessing.curveDespikedHistory = self.get_curve_plot_data_item(arr, group_number)
        self.input_plot_widget_plot_item.addItem(self.preprocessing.curveDespikedHistory,
                                                 kargs=['ignoreBounds', 'skipAverage'])

        all_peaks = self.ui.input_table.model().cell_data(current_index.row(), 3)
        all_peaks = all_peaks.split()
        text_peaks = []
        for i in all_peaks:
            i = i.replace(',', '')
            i = i.replace(' ', '')
            text_peaks.append(i)
        list_peaks = [float(s) for s in text_peaks]
        for i in list_peaks:
            idx = nearest_idx(arr[:, 0], i)
            y_peak = arr[:, 1][idx]
            arrow = ArrowItem(pos=(i, y_peak), angle=-45)
            self.input_plot_widget_plot_item.addItem(arrow)

    async def despike_history_remove_plot(self) -> None:
        """
        remove old history _BeforeDespike plot item and arrows
        """
        if self.preprocessing.curveDespikedHistory:
            self.input_plot_widget_plot_item.removeItem(self.preprocessing.curveDespikedHistory)

        arrows = []
        for x in self.input_plot_widget_plot_item.items:
            if isinstance(x, ArrowItem):
                arrows.append(x)
        for i in reversed(arrows):
            self.input_plot_widget_plot_item.removeItem(i)

    async def baseline_remove_plot(self) -> None:
        if self.preprocessing.curveBaseline:
            self.smooth_plotItem.removeItem(self.preprocessing.curveBaseline)

    async def baseline_add_plot(self) -> None:
        # selected spectrum baseline
        current_index = self.ui.input_table.selectionModel().currentIndex()
        group_number = self.ui.input_table.model().cell_data(current_index.row(), 2)
        arr = self.preprocessing.baseline_dict[self.current_spectrum_baseline_name]
        if self.preprocessing.curveBaseline:
            self.smooth_plotItem.removeItem(self.preprocessing.curveBaseline)
        self.preprocessing.curveBaseline = self.get_curve_plot_data_item(arr, group_number,
                                                                         color=self.theme_colors['primaryColor'])
        self.smooth_plotItem.addItem(self.preprocessing.curveBaseline, kargs=['ignoreBounds', 'skipAverage'])

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
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 3:
            self.ui.ignore_dataset_table_view.verticalScrollBar().setValue(event)
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
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 11:
            self.ui.scrollArea_xgboost.verticalScrollBar().setValue(event)

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
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        file_path = fd.getOpenFileNames(parent=self, caption='Select files with Raman data',
                                        directory=path, filter="Text files (*.txt *.asc)")
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
        self.time_start = datetime.now()
        self.ui.statusBar.showMessage('Importing...')
        self.close_progress_bar()
        n_files = len(path_list)
        self.open_progress_dialog("Importing files...", "Cancel", maximum=n_files)
        self.open_progress_bar(max_value=n_files)
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
        elif self.ui.stackedWidget_mainpages.currentIndex() == 2 and self.ui.data_tables_tab_widget.currentIndex() == 3:
            tv = self.ui.ignore_dataset_table_view
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
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 and self.ui.stat_tab_widget.currentIndex() == 11:
            tv = self.ui.scrollArea_xgboost
        elif self.ui.stackedWidget_mainpages.currentIndex() == 3 \
                and (self.ui.stat_tab_widget.currentIndex() == 12 or self.ui.stat_tab_widget.currentIndex() == 13):
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
                action_help()
            case Qt.Key.Key_F7:
                self.stat_analysis_logic.dataset_for_ml()
            case Qt.Key.Key_F2:
                if self.ui.stackedWidget_mainpages.currentIndex() != 1:
                    return
                if self.fitting.intervals_data is None:
                    return
                for i, v in enumerate(self.fitting.intervals_data.items()):
                    key, item = v
                    show_distribution(item['x0'], self.fitting.averaged_array,
                                      self.fitting.all_ranges_clustered_x0_sd[i])
            case Qt.Key.Key_F3:
                from numpy.polynomial.polynomial import polyval
                from modules.functions_baseline_correction import baseline_penalized_poly, baseline_loess, baseline_quant_reg, baseline_goldindec
                from sklearn.metrics import mean_squared_error
                coefs = [1.27321927e+04, -7.62029464e+01,  2.81669762e-01, -5.66507200e-04, 6.52691243e-07,
                                  -4.38362720e-10,  1.66449516e-13, -3.21147214e-17]
                x_axis, y_raman = self.fitting.sum_array()
                baseline_test = polyval(x_axis, coefs)
                baseline_and_raman = baseline_test + y_raman
                test_arr = np.vstack((x_axis, baseline_and_raman)).T
                _, _, corrected_plus = ex_mod_poly(('1', test_arr), [7., 1e-10, 1000.])
                _, _, corrected_imod = baseline_imodpoly(('3', test_arr), [7, 1e-8, 1000])
                _, _, corrected_modp = baseline_modpoly(('3', test_arr), [7, 1e-10, 1000])
                _, _, corrected_quant = baseline_penalized_poly(('3', test_arr), [7, 1e-8, 1000, 0.999, 'asymmetric_truncated_quadratic'])
                _, _, corrected_goldin= baseline_goldindec(('3', test_arr), [7, 1e-9, 1000, 'asymmetric_truncated_quadratic', 0.5, .999])
                corr_matrix_plus = np.corrcoef(y_raman, corrected_plus[:, 1])
                corr1 = corr_matrix_plus[0, 1] ** 2
                rms1 = mean_squared_error(corrected_plus[:, 1], y_raman, squared=False)
                rms2 = mean_squared_error(corrected_imod[:, 1], y_raman, squared=False)
                rms3 = mean_squared_error(corrected_modp[:, 1], y_raman, squared=False)
                rms6 = mean_squared_error(corrected_quant[:, 1], y_raman, squared=False)
                rms7 = mean_squared_error(corrected_goldin[:, 1], y_raman, squared=False)
                corrected_plus1 = corrected_plus[corrected_plus < 0]
                corrected_imod1 = corrected_imod[corrected_imod < 0]
                corrected_modp1 = corrected_modp[corrected_modp < 0]
                corrected_quant1 = corrected_quant[corrected_quant < 0]
                corrected_goldin1 = corrected_goldin[corrected_goldin < 0]
                area0 = np.trapz(corrected_plus1)
                area1 = np.trapz(corrected_imod1)
                area2 = np.trapz(corrected_modp1)
                area3 = np.trapz(corrected_quant1)
                area4 = np.trapz(corrected_goldin1)
                print('r2 EX-Mod-Poly+ ', corr1,  'rms ', rms1, 'area ', area0)
                corr_matrix_imod = np.corrcoef(y_raman, corrected_imod[:, 1])
                corr2 = corr_matrix_imod[0, 1] ** 2
                print('r2 I-Mod-Poly ', corr2, 'area ', area1)
                corr_matrix_modp = np.corrcoef(y_raman, corrected_modp[:, 1])
                corr3 = corr_matrix_modp[0, 1] ** 2
                print('r2 Mod-Poly ', corr3, 'area ', area2)
                corr_matrix_quant = np.corrcoef(y_raman, corrected_quant[:, 1])
                corr6 = corr_matrix_quant[0, 1] ** 2
                print('r2 Penalized Poly ', corr6, 'area ', area3)
                corr_matrix_goldin= np.corrcoef(y_raman, corrected_goldin[:, 1])
                corr7 = corr_matrix_goldin[0, 1] ** 2
                print('r2 Goldindec ', corr7, 'area ', area4)
                fig, ax = plt.subplots()
                ax.plot(x_axis, y_raman, label='Synthesized spectrum', color = 'black')
                ax.plot(x_axis, corrected_plus[:, 1], label='Ex-Mod-Poly. $\mathregular{r^2}$=' + str(np.round(corr1*100, 3)) + '. RMSE=' + str(np.round(rms1, 2)), color = 'red')
                ax.plot(x_axis, corrected_imod[:, 1], label='I-Mod-Poly. $\mathregular{r^2}$=' + str(np.round(corr2*100, 3)) + '. RMSE=' + str(np.round(rms2, 2)), color = 'green')
                ax.plot(x_axis, corrected_modp[:, 1], label='Mod-Poly. $\mathregular{r^2}$=' + str(np.round(corr3*100, 3)) + '. RMSE=' + str(np.round(rms3, 2)), color = 'blue', dashes=[6, 2])
                ax.plot(x_axis, corrected_quant[:, 1], label='Penalized Poly. $\mathregular{r^2}$=' + str(np.round(corr6*100, 3)) + '. RMSE=' + str(np.round(rms6, 2)), color = 'c', dashes=[8, 4])
                ax.plot(x_axis, corrected_goldin[:, 1], label='Goldindec. $\mathregular{r^2}$=' + str(np.round(corr7*100, 3)) + '. RMSE=' + str(np.round(rms7, 2)), color = 'm', dashes=[8, 4])
                ax.legend()
                ax.grid()
                plt.show()
            case Qt.Key.Key_F4:
                arr = self.preprocessing.smoothed_spectra['[1]_Здоровый (1).asc']
                y = arr[:, 1]
                keyarr = '[1]_Здоровый (1).asc', arr
                before = datetime.now()
                devs = []
                fig, ax = plt.subplots()
                for k, v in self.preprocessing.smoothed_spectra.items():
                    _, baseline_plus_tru, corrected_tru, pitches = ex_mod_poly((k, v), [8., 1e-7, 1000.])

                    # ax.plot(corrected_tru[:, 0], corrected_tru[:, 1])
                    devs.append(pitches)
                # _, baseline_plus_new, corrected_new1, work_my, dev, sc = baseline_imodpoly_plus(keyarr, [7, 1e-20, 250], True)
                # _, baseline_plus_new, corrected_new2, work_my, dev, sc = baseline_imodpoly_plus(keyarr, [7, 1e-20, 250], False)
                # info(devs[0])
                print(datetime.now() - before)
                # key, baseline_byimodpoly, corrected = baseline_imodpoly(keyarr, [8, 1e-6, 1000])
                # key, _, corrected_md = baseline_modpoly(keyarr, [8, 1e-6, 1000])

                # for i in sc_diffs:
                #     ax.plot(i)
                # ax.plot(arr[:, 0], arr[:, 1], label='original', color='b')
                # ax.plot(corrected_tru[:, 0], corrected_tru[:, 1], label='tru', color='black')
                # ax.legend()
                # plt.show()
                # ax.plot(work[:, 0], work[:, 1], label='Спектр без пиков I-ModPoly', dashes=[6, 2], color='b')
                # ax.plot(work_my[:, 0], work_my[:, 1], label='Спектр без пиков предложенным методом', dashes=[4, 4], color='r')
                #
                # ax.plot(baseline_plus_new[:, 0], baseline_plus_new[:, 1], label='Базовая линия предложенным методом', color='r')
                # ax.plot(baseline_byimodpoly[:, 0], baseline_byimodpoly[:, 1], label='Базовая линия I-ModPoly', color='b')

                fig, ax = plt.subplots()
                for i in devs:
                    ax.plot(i)
                ax.grid()
                plt.show()
            case Qt.Key.Key_F5:
                self.arre()
            case Qt.Key.Key_F6:
                return
                from modules.functions_peak_shapes import gaussian
                from scipy.signal import deconvolve, convolve
                signal = self.preprocessing.smoothed_spectra['[1]_Здоровый (1).asc']
                x = signal[:, 0]
                fig, ax = plt.subplots()
                # ax.plot(signal[:, 0], signal[:, 1], label='original')
                xx = np.zeros_like(x)
                gauss = gaussian(x, 1, np.median(x), 60.)
                gauss = gauss[gauss > 0.1]
                conv = convolve(signal[:, 1], gauss, mode='same')
                deconv, _ = deconvolve(conv, gauss)
                # ns = len(signal[:, 0])
                #
                # n = ns - len(gauss) + 1
                # # so we need to expand it by
                # s = int((ns - n) / 2)
                # # on both sides.
                # deconv_res = np.zeros(ns)
                # deconv_res[s:ns - s - 1] = deconv
                # deconv = deconv_res
                # now deconv contains the deconvolution
                # expanded to the original shape (filled with zeros)
                # ax.plot(gauss, label='recovered')
                ax.plot(deconv, label='recovered')
                ax.grid()
                ax.legend()
                plt.show()
            case Qt.Key.Key_F9:
                areas = []
                for i in self.preprocessing.baseline_corrected_dict.values():
                    y = i[:, 1]
                    y = y[y < 0]
                    area = np.trapz(y)
                    areas.append(area)
                mean_area = np.mean(areas)
                print(mean_area)
            case Qt.Key.Key_F8:
                from modules.functions_peak_shapes import gaussian
                x_axis = next(iter(self.preprocessing.baseline_corrected_dict.values()))[:, 0]
                self.preprocessing.baseline_corrected_dict.clear()
                keys = self.preprocessing.smoothed_spectra.keys()
                for key in keys:
                    y_axis = np.zeros(x_axis.shape[0])
                    params_df_values = self.ui.fit_params_table.model().get_df_by_filename('').values
                    for i in range(0, params_df_values.shape[0], 3):
                        rnd = np.random.rand(1)[0]
                        a = params_df_values[i][1] * (1 + (0.01 * rnd - 0.005))
                        x0 = params_df_values[i+1][1]
                        dx = params_df_values[i+2][1]
                        y_shape = gaussian(x_axis, a, x0, dx)
                        y_axis += y_shape
                    std = np.std(y_axis)
                    print(std)
                    noise = np.random.normal(0, std, x_axis.shape[0])
                    self.preprocessing.baseline_corrected_dict[key] = np.vstack((x_axis, y_axis + (noise * 10.))).T
            case Qt.Key.Key_F11:
                if not self.isFullScreen():
                    self.showFullScreen()
                else:
                    self.showMaximized()

    @asyncSlot()
    async def arre(self):
        from itertools import islice
        times = []

        params = [9., 1e-7, 1000.]
        for i in range(50, 1050, 50):
            n_items = list(islice(self.preprocessing.smoothed_spectra.items(), i))
            executor = ProcessPoolExecutor()
            before = datetime.now()
            with executor:
                self.current_futures = [self.loop.run_in_executor(executor, ex_mod_poly, i, params) for i in
                                        n_items]
                for future in self.current_futures:
                    future.add_done_callback(self.progress_indicator)
                baseline_corrected = await gather(*self.current_futures)
            after = datetime.now() - before
            sec = after.total_seconds()
            print(i)
            print(sec)
            times.append(sec)
        info(times)
        print(times)

    def undo(self) -> None:
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.time_start = datetime.now()
        self.undoStack.undo()
        self.update_undo_redo_tooltips()

    def redo(self) -> None:
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.time_start = datetime.now()
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
            items_matches = (x for x in list_data_items if not isinstance(x, PlotDataItem)
                             and x.get_group_number() == group_number)
            for i in items_matches:
                color = style['color']
                color.setAlphaF(1.0)
                pen = mkPen(color=color, style=style['style'], width=style['width'])
                i.setPen(pen)

    def get_color_by_group_number(self, group_number: str) -> QColor:
        if group_number != 'nan' and group_number != '' and self.ui.GroupsTable.model().rowCount() > 0 \
                and int(group_number) <= self.ui.GroupsTable.model().rowCount():
            color = self.ui.GroupsTable.model().cell_data(int(group_number) - 1, 1)['color']
            return color
        else:
            return QColor(self.theme_colors['secondaryColor'])

    # endregion

    # region ACTIONS FILE MENU

    def action_new_project(self) -> None:
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(self, 'Create Project File', path, "ZIP (*.zip)")
        if file_path[0] != '':
            try:
                f = shelve_open(file_path[0], 'n')
                f.close()
                self.open_project(file_path[0], new=True)
            except BaseException:
                raise

    def action_open_project(self) -> None:
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        file_path = fd.getOpenFileName(self, 'Select RS-tool project file to open', path, "(*.zip)")
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
            path = os.getenv('APPDATA') + '/RS-tool'
            fd = QFileDialog(self)
            file_path = fd.getSaveFileName(self, 'Create Project File', path, "ZIP (*.zip)")
            if file_path[0] != '':
                self.save_with_shelve(file_path[0])
                self.ui.projectLabel.setText(file_path[0])
                self.setWindowTitle(file_path[0])
                self._add_path_to_recent(file_path[0])
                self.update_recent_list()
                self.project_path = file_path[0]
        else:
            self.save_with_shelve(self.project_path)

    @asyncSlot()
    async def action_save_production_project(self) -> None:
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(self, 'Create Production Project File', path, "ZIP (*.zip)")
        if file_path[0] != '':
            self.save_with_shelve(file_path[0], True)

    @asyncSlot()
    async def action_save_as(self) -> None:
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(self, 'Create Project File', path, "ZIP (*.zip)")
        if file_path[0] != '':
            self.save_with_shelve(file_path[0])
            self.project_path = file_path[0]

    def save_with_shelve(self, path: str, production_export: bool = False) -> None:
        self.ui.statusBar.showMessage('Saving file...')
        self.close_progress_bar()
        self.open_progress_bar()
        filename = str(Path(path).parent) + '/' + str(Path(path).stem)
        with shelve_open(filename, 'n') as db:
            db["GroupsTable"] = self.ui.GroupsTable.model().dataframe()
            db["DeconvLinesTableDF"] = self.ui.deconv_lines_table.model().dataframe()
            db["DeconvParamsTableDF"] = self.ui.fit_params_table.model().dataframe()
            db["intervals_table_df"] = self.ui.fit_intervals_table_view.model().dataframe()
            db["DeconvLinesTableChecked"] = self.ui.deconv_lines_table.model().checked()
            db["IgnoreTableChecked"] = self.ui.ignore_dataset_table_view.model().checked
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
            db["data_style"] = self.fitting.data_style.copy()
            db["data_curve_checked"] = self.ui.data_checkBox.isChecked()
            db["sum_style"] = self.fitting.sum_style.copy()
            db["sigma3_style"] = self.fitting.sigma3_style.copy()
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
            db['baseline_corrected_dict'] = self.preprocessing.baseline_corrected_dict
            db['groupBox_mlp_checked'] = self.ui.groupBox_mlp.isChecked()
            db['activation_comboBox'] = self.ui.activation_comboBox.currentText()
            db['refit_score'] = self.ui.refit_score.currentText()
            db['solver_mlp_combo_box'] = self.ui.solver_mlp_combo_box.currentText()
            db['mlp_layer_size_spinBox'] = self.ui.mlp_layer_size_spinBox.value()
            db['max_epoch_spinBox'] = self.ui.max_epoch_spinBox.value()
            db['learning_rate_doubleSpinBox'] = self.ui.learning_rate_doubleSpinBox.value()
            db['feature_display_max_checkBox'] = self.ui.feature_display_max_checkBox.isChecked()
            db['include_x0_checkBox'] = self.ui.include_x0_checkBox.isChecked()
            db['feature_display_max_spinBox'] = self.ui.feature_display_max_spinBox.value()
            db['old_Y'] = self.stat_analysis_logic.old_labels
            db['new_Y'] = self.stat_analysis_logic.new_labels
            db['use_pca_checkBox'] = self.ui.use_pca_checkBox.isChecked()
            db['intervals_data'] = self.fitting.intervals_data
            db['all_ranges_clustered_lines_x0'] = self.fitting.all_ranges_clustered_x0_sd
            if not self.predict_logic.is_production_project:
                self.predict_logic.stat_models = {}
                for key, v in self.stat_analysis_logic.latest_stat_result.items():
                    self.predict_logic.stat_models[key] = v['model']
                db["stat_models"] = self.predict_logic.stat_models
                if self.ImportedArray:
                    db['interp_ref_array'] = next(iter(self.ImportedArray.values()))
            db["averaged_dict"] = self.preprocessing.averaged_dict.copy()
            if not production_export:
                db["InputTable"] = self.ui.input_table.model().dataframe()
                db["smoothed_dataset_df"] = self.ui.smoothed_dataset_table_view.model().dataframe()
                db["baselined_dataset_df"] = self.ui.baselined_dataset_table_view.model().dataframe()
                db["deconvoluted_dataset_df"] = self.ui.deconvoluted_dataset_table_view.model().dataframe()
                db["ignore_dataset_df"] = self.ui.ignore_dataset_table_view.model().dataframe()
                db["predict_df"] = self.ui.predict_table_view.model().dataframe()
                db["ImportedArray"] = self.ImportedArray
                db["ConvertedDict"] = self.preprocessing.ConvertedDict
                db["BeforeDespike"] = self.preprocessing.BeforeDespike
                db["NormalizedDict"] = self.preprocessing.NormalizedDict
                db["CuttedFirstDict"] = self.preprocessing.CuttedFirstDict
                db["SmoothedDict"] = self.preprocessing.smoothed_spectra
                db['baseline_corrected_not_trimmed_dict'] = self.preprocessing.baseline_corrected_not_trimmed_dict
                db["baseline_dict"] = self.preprocessing.baseline_dict

                db["report_result"] = self.fitting.report_result.copy()
                db["sigma3"] = self.fitting.sigma3.copy()
                db["latest_stat_result"] = self.stat_analysis_logic.latest_stat_result
                db["is_production_project"] = False
            else:
                db["is_production_project"] = True
            if self.predict_logic.is_production_project:
                db["is_production_project"] = True
                db['interp_ref_array'] = self.predict_logic.interp_ref_array
                db['stat_models'] = self.predict_logic.stat_models
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
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        folder_path = fd.getExistingDirectory(self, 'Choose folder to export files in nm', path)
        if folder_path:
            self.ui.statusBar.showMessage('Exporting files...')
            self.close_progress_bar()
            self.progressBar = QProgressBar(self)
            self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.export_folder_path = folder_path + '/nm'
            if not Path(self.export_folder_path).exists():
                Path(self.export_folder_path).mkdir(parents=True)
            with ThreadPoolExecutor() as executor:
                self.current_futures = [self.loop.run_in_executor(executor, self._export_files, i)
                                        for i in self.ImportedArray.items()]
                await gather(*self.current_futures)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Export completed. {} new files created.'.format(len(self.ImportedArray)),
                                          50_000)

    @asyncSlot()
    async def action_export_files_cm(self) -> None:

        if not self.preprocessing.baseline_corrected_dict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No files to save")
            msg.setWindowTitle("Export failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        folder_path = fd.getExistingDirectory(self, 'Choose folder to export files in cm-1', path)
        if folder_path:
            self.ui.statusBar.showMessage('Exporting files...')
            self.close_progress_bar()
            self.progressBar = QProgressBar(self)
            self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.export_folder_path = folder_path + '/cm-1'
            if not Path(self.export_folder_path).exists():
                Path(self.export_folder_path).mkdir(parents=True)
            with ThreadPoolExecutor() as executor:
                self.current_futures = [self.loop.run_in_executor(executor, self._export_files, i)
                                        for i in self.preprocessing.baseline_corrected_dict.items()]
                await gather(*self.current_futures)
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Export completed. {} new files '
                                          'created.'.format(len(self.preprocessing.baseline_corrected_dict)), 50_000)

    @asyncSlot()
    async def action_export_average(self) -> None:

        if not self.preprocessing.averaged_dict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("No files to save")
            msg.setWindowTitle("Export failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        folder_path = fd.getExistingDirectory(self, 'Choose folder to export files in cm-1', path)
        if folder_path:
            self.ui.statusBar.showMessage('Exporting files...')
            self.close_progress_bar()
            self.progressBar = QProgressBar(self)
            self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.export_folder_path = folder_path + '/average'
            if not Path(self.export_folder_path).exists():
                Path(self.export_folder_path).mkdir(parents=True)
            with ThreadPoolExecutor() as executor:
                self.current_futures = [self.loop.run_in_executor(executor, self._export_files_av, i)
                                        for i in self.preprocessing.averaged_dict.items()]
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
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        folder_path = fd.getExistingDirectory(self, 'Choose folder to save excel file', path)
        if not folder_path:
            return
        self.ui.statusBar.showMessage('Saving file...')
        self.close_progress_bar()
        self.open_progress_bar(max_value=0)
        self.open_progress_dialog("Exporting Excel...", "Cancel", maximum=0)
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
            if self.ui.ignore_dataset_table_view.model().rowCount() > 0:
                self.ui.ignore_dataset_table_view.model().dataframe().to_excel(writer,
                                                                                     sheet_name='Ignored features')
            if self.ui.pca_features_table_view.model().rowCount() > 0:
                self.ui.pca_features_table_view.model().dataframe().to_excel(writer, sheet_name='PCA loadings')
            if self.ui.plsda_vip_table_view.model().rowCount() > 0:
                self.ui.plsda_vip_table_view.model().dataframe().to_excel(writer, sheet_name='PLS-DA VIP')
            if self.ui.predict_table_view.model().rowCount() > 0:
                self.ui.predict_table_view.model().dataframe().to_excel(writer, sheet_name='Predicted')

    def clear_selected_step(self, step: str) -> None:
        if step in self.stat_analysis_logic.latest_stat_result:
            del self.stat_analysis_logic.latest_stat_result[step]
        match step:
            case 'Converted':
                self.preprocessing.ConvertedDict.clear()
                self.preprocessing.curveOneConvertPlot = None
                self.converted_cm_widget_plot_item.clear()
                self._initial_converted_cm_plot()
            case 'Cutted':
                self.preprocessing.CuttedFirstDict.clear()
                self.preprocessing.curveOneCutPlot = None
                self.cut_cm_plotItem.clear()
                self._initial_cut_cm_plot()
            case 'Normalized':
                self.preprocessing.NormalizedDict.clear()
                self.preprocessing.curveOneNormalPlot = None
                self.normalize_plotItem.clear()
                self._initial_normalize_plot()
            case 'Smoothed':
                self.preprocessing.smoothed_spectra.clear()
                self.preprocessing.curveOneSmoothPlot = None
                self.smooth_plotItem.clear()
                self._initial_smooth_plot()
                self.ui.smoothed_dataset_table_view.model().clear_dataframe()
            case 'Baseline':
                self.preprocessing.baseline_corrected_dict.clear()
                self.preprocessing.baseline_corrected_not_trimmed_dict.clear()
                self.preprocessing.baseline_dict.clear()
                self.preprocessing.curve_one_baseline_plot = None
                self.baseline_corrected_plotItem.clear()
                self.ui.baselined_dataset_table_view.model().clear_dataframe()
                self._initial_baseline_plot()
            case 'Averaged':
                self.preprocessing.averaged_dict.clear()
                self.averaged_plotItem.clear()
                self._initial_averaged_plot()
            case 'Deconvolution':
                self.deconvolution_plotItem.clear()
                self.fitting.report_result.clear()
                self.fitting.sigma3.clear()
                del self.fitting.fill
                self._initial_deconvolution_plot()
                self.fitting.data_curve = None
                self.fitting.current_spectrum_deconvolution_name = ''
                try:
                    self.ui.template_combo_box.currentTextChanged.disconnect(self.fitting.switch_to_template)
                except:
                    error('failed to disconnect currentTextChanged self.switch_to_template)')
                self.ui.template_combo_box.clear()
                self.fitting.is_template = False
                self.fitting.all_ranges_clustered_x0_sd = None
                self.ui.data_checkBox.setChecked(True)
                self.ui.sum_checkBox.setChecked(False)
                self.ui.residual_checkBox.setChecked(False)
                self.ui.include_x0_checkBox.setChecked(False)
                self.ui.sigma3_checkBox.setChecked(False)
                self.data_style_button_style_sheet(self.fitting.data_style['color'].name())
                self.sum_style_button_style_sheet(self.fitting.sum_style['color'].name())
                self.sigma3_style_button_style_sheet(self.fitting.sigma3_style['color'].name())
                self.residual_style_button_style_sheet(self.fitting.residual_style['color'].name())
                self.ui.interval_checkBox.setChecked(False)
                self.linearRegionDeconv.setVisible(False)
                self.ui.deconvoluted_dataset_table_view.model().clear_dataframe()
                self.ui.ignore_dataset_table_view.model().clear_dataframe()
                self.fitting.update_template_combo_box()
            case 'Stat':
                self._initial_all_stat_plots()
                self.ui.current_group_shap_comboBox.clear()
                self.ui.current_feature_comboBox.clear()
                self.ui.coloring_feature_comboBox.clear()
                self.ui.current_instance_combo_box.clear()
                self.ui.feature_display_max_checkBox.setChecked(False)
                self.ui.random_state_cb.setChecked(False)
                self.ui.use_shapley_cb.setChecked(True)
                self.stat_analysis_logic.latest_stat_result = {}
                self._initial_pca_features_table()
                self._initial_plsda_vip_table()
                self.ui.groupBox_mlp.setChecked(True)
                self.ui.refit_score.setCurrentText('recall_score')
            case 'LDA':
                self._initial_lda_plots()
            case 'QDA':
                self._initial_qda_plots()
            case 'Logistic regression':
                self._initial_lr_plots()
            case 'NuSVC':
                self._initial_svc_plots()
            case 'Nearest Neighbors':
                self._initial_nearest_plots()
            case 'GPC':
                self._initial_gpc_plots()
            case 'Decision Tree':
                self._initial_dt_plots()
            case 'Naive Bayes':
                self._initial_nb_plots()
            case 'Random Forest':
                self._initial_rf_plots()
            case 'AdaBoost':
                self._initial_ab_plots()
            case 'MLP':
                self._initial_mlp_plots()
            case 'XGBoost':
                self._initial_xgboost_plots()
            case 'Torch':
                self._initial_torch_plots()
            case 'PCA':
                self._initial_pca_plots()
                self._initial_pca_features_table()
            case 'PLS-DA':
                self._initial_plsda_plots()
                self._initial_plsda_vip_table()
            case 'Page5':
                self._initial_predict_dataset_table()

    def _clear_all_parameters(self) -> None:
        before = datetime.now()
        self.close_progress_bar()
        self.open_progress_bar()
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
        self.ui.ignore_dataset_table_view.model().clear_dataframe()
        self._reset_deconvoluted_dataset_table()
        self.predict_logic.is_production_project = False
        self.predict_logic.stat_models = {}
        self.predict_logic.interp_ref_array = None
        self.stat_analysis_logic.top_features = {}
        self.stat_analysis_logic.old_labels = None
        self.stat_analysis_logic.new_labels = None
        self.stat_analysis_logic.latest_stat_result = {}
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
        self.preprocessing.BeforeDespike.clear()
        self.preprocessing.baseline_dict.clear()
        self.preprocessing.baseline_corrected_dict.clear()
        self.preprocessing.baseline_corrected_not_trimmed_dict.clear()
        self.preprocessing.averaged_dict.clear()
        self.preprocessing.curveOneInputPlot = None
        self.preprocessing.curveDespikedHistory = None
        self.preprocessing.curveBaseline = None
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
        b = not self.predict_logic.is_production_project
        self.ui.left_side_head_stackedWidget.setEnabled(b)
        self.ui.stackedWidget_left.setEnabled(b)
        self.ui.deconv_lines_table.setEnabled(b)
        self.ui.deconv_buttons_frame_top.setEnabled(b)
        self.ui.fit_params_table.setEnabled(b)

    @asyncSlot()
    async def load_params(self, path: str) -> None:
        self.ui.statusBar.showMessage('Reading data file...')
        self.fitting.deselect_selected_line()
        self.close_progress_bar()
        self.open_progress_bar()
        self.setEnabled(False)
        self.open_progress_dialog("Opening project...", "Cancel")
        self.time_start = datetime.now()
        self.unzip_project_file(path)
        await self.update_all_plots()
        self.fitting.update_template_combo_box()
        await self.fitting.switch_to_template()
        self.fitting.update_deconv_intervals_limits()
        self.dataset_type_cb_current_text_changed(self.ui.dataset_type_cb.currentText())
        if self.ui.fit_params_table.model().rowCount() != 0 \
                and self.ui.deconv_lines_table.model().rowCount() != 0:
            await self.fitting.draw_all_curves()
        self.currentProgress.setMaximum(1)
        self.currentProgress.setValue(1)
        self.close_progress_bar()
        self.stat_analysis_logic.update_force_single_plots()
        self.stat_analysis_logic.update_force_full_plots()
        self.set_buttons_ability()
        self.set_forms_ability()
        seconds = round((datetime.now() - self.time_start).total_seconds())
        self.set_modified(False)
        self.decide_vertical_scroll_bar_visible()
        self.setEnabled(True)
        await self.redraw_stat_plots()
        self.stat_analysis_logic.update_force_single_plots()
        self.stat_analysis_logic.update_force_full_plots()
        self.ui.statusBar.showMessage('Project opened for ' + str(seconds) + ' sec.', 15000)

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
            if "IgnoreTableChecked" in db:
                checked = db["IgnoreTableChecked"]
                self.ui.ignore_dataset_table_view.model().set_checked(checked)
            if "DeconvParamsTableDF" in db:
                df = db["DeconvParamsTableDF"]
                self.ui.fit_params_table.model().set_dataframe(df)
            if "intervals_table_df" in db:
                df = db["intervals_table_df"]
                self.ui.fit_intervals_table_view.model().set_dataframe(df)
            if "smoothed_dataset_df" in db:
                self.ui.smoothed_dataset_table_view.model().set_dataframe(db["smoothed_dataset_df"])
            if "stat_models" in db:
                self.predict_logic.stat_models = db["stat_models"]
            if "baselined_dataset_df" in db:
                self.ui.baselined_dataset_table_view.model().set_dataframe(db["baselined_dataset_df"])
            if "deconvoluted_dataset_df" in db:
                self.ui.deconvoluted_dataset_table_view.model().set_dataframe(db["deconvoluted_dataset_df"])
            if "ignore_dataset_df" in db:
                self.ui.ignore_dataset_table_view.model().set_dataframe(db["ignore_dataset_df"])
            if "interp_ref_array" in db:
                self.predict_logic.interp_ref_array = db["interp_ref_array"]
            if "predict_df" in db:
                self.ui.predict_table_view.model().set_dataframe(db["predict_df"])
            if "is_production_project" in db:
                self.predict_logic.is_production_project = db["is_production_project"]
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
                self.predict_logic.y_axis_ref_EMSC = db["_y_axis_ref_EMSC"]
            if "interval_start_cm" in db:
                self.ui.interval_start_dsb.setValue(db["interval_start_cm"])
            if "interval_end_cm" in db:
                self.ui.interval_end_dsb.setValue(db["interval_end_cm"])
            if "ConvertedDict" in db:
                self.preprocessing.ConvertedDict = db["ConvertedDict"]
                self.update_cm_min_max_range()
            if "LaserWL" in db:
                self.ui.laser_wl_spinbox.setValue(db["LaserWL"])
            if "BeforeDespike" in db:
                self.preprocessing.BeforeDespike = db["BeforeDespike"]
            if "Maxima_count_despike" in db:
                self.ui.maxima_count_despike_spin_box.setValue(db["Maxima_count_despike"])
            if "Despike_fwhm_width" in db:
                self.ui.despike_fwhm_width_doubleSpinBox.setValue(db["Despike_fwhm_width"])
            if "CuttedFirstDict" in db:
                self.preprocessing.CuttedFirstDict = db["CuttedFirstDict"]
            if "averaged_dict" in db:
                self.preprocessing.averaged_dict = db["averaged_dict"]
            if "report_result" in db:
                self.fitting.report_result = db["report_result"]
            if "sigma3" in db:
                self.fitting.sigma3 = db["sigma3"]
            if "neg_grad_factor_spinBox" in db:
                self.ui.neg_grad_factor_spinBox.setValue(db["neg_grad_factor_spinBox"])
            if "NormalizedDict" in db:
                self.preprocessing.NormalizedDict = db["NormalizedDict"]
            if "normalizing_method_comboBox" in db:
                self.ui.normalizing_method_comboBox.setCurrentText(db["normalizing_method_comboBox"])
            if "normalizing_method_used" in db:
                self.read_normalizing_method_used(db["normalizing_method_used"])
            if "SmoothedDict" in db:
                self.preprocessing.smoothed_spectra = db["SmoothedDict"]
            if "baseline_corrected_dict" in db:
                self.preprocessing.baseline_corrected_dict = db["baseline_corrected_dict"]
            if "baseline_corrected_not_trimmed_dict" in db:
                self.preprocessing.baseline_corrected_not_trimmed_dict = db["baseline_corrected_not_trimmed_dict"]
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
                self.preprocessing.baseline_dict = db["baseline_dict"]
            if "baseline_method" in db:
                self.read_baseline_method_used(db["baseline_method"])
            if "data_style" in db:
                self.fitting.data_style.clear()
                self.fitting.data_style = db["data_style"].copy()
                self.data_style_button_style_sheet(self.fitting.data_style['color'].name())
            if "data_curve_checked" in db:
                self.ui.data_checkBox.setChecked(db["data_curve_checked"])
            if "sum_style" in db:
                self.fitting.sum_style.clear()
                self.fitting.sum_style = db["sum_style"].copy()
                self.sum_style_button_style_sheet(self.fitting.sum_style['color'].name())
            if "sigma3_style" in db:
                self.fitting.sigma3_style.clear()
                self.fitting.sigma3_style = db["sigma3_style"].copy()
                self.sigma3_style_button_style_sheet(self.fitting.sigma3_style['color'].name())
                pen, brush = curve_pen_brush_by_style(self.fitting.sigma3_style)
                self.fitting.fill.setPen(pen)
                self.fitting.fill.setBrush(brush)
            if "sum_curve_checked" in db:
                self.ui.sum_checkBox.setChecked(db["sum_curve_checked"])
            if "sigma3_checked" in db:
                self.ui.sigma3_checkBox.setChecked(db["sigma3_checked"])
            if "residual_style" in db:
                self.fitting.residual_style.clear()
                self.fitting.residual_style = db["residual_style"].copy()
                self.residual_style_button_style_sheet(self.fitting.residual_style['color'].name())
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
            if 'max_dx_guess' in db:
                self.ui.max_dx_dsb.setValue(db['max_dx_guess'])
            if 'latest_stat_result' in db:
                self.stat_analysis_logic.latest_stat_result = db['latest_stat_result']
            if 'groupBox_mlp_checked' in db:
                self.ui.groupBox_mlp.setChecked(db['groupBox_mlp_checked'])
            if 'activation_comboBox' in db:
                self.ui.activation_comboBox.setCurrentText(db['activation_comboBox'])
            if 'solver_mlp_combo_box' in db:
                self.ui.solver_mlp_combo_box.setCurrentText(db['solver_mlp_combo_box'])
            if 'mlp_layer_size_spinBox' in db:
                self.ui.mlp_layer_size_spinBox.setValue(db['mlp_layer_size_spinBox'])
            if 'max_epoch_spinBox' in db:
                self.ui.max_epoch_spinBox.setValue(db['max_epoch_spinBox'])
            if 'learning_rate_doubleSpinBox' in db:
                self.ui.learning_rate_doubleSpinBox.setValue(db['learning_rate_doubleSpinBox'])
            if 'refit_score' in db:
                self.ui.refit_score.setCurrentText(db['refit_score'])
            if 'feature_display_max_checkBox' in db:
                self.ui.feature_display_max_checkBox.setChecked(db['feature_display_max_checkBox'])
            if 'include_x0_checkBox' in db:
                self.ui.include_x0_checkBox.setChecked(db['include_x0_checkBox'])
            if 'feature_display_max_spinBox' in db:
                self.ui.feature_display_max_spinBox.setValue(db['feature_display_max_spinBox'])
            if 'use_pca_checkBox' in db:
                self.ui.use_pca_checkBox.setChecked(db['use_pca_checkBox'])
            if 'intervals_data' in db:
                self.fitting.intervals_data = db['intervals_data']
            if 'all_ranges_clustered_lines_x0' in db:
                self.fitting.all_ranges_clustered_x0_sd = db['all_ranges_clustered_lines_x0']

            if 'old_Y' in db:
                self.stat_analysis_logic.old_labels = db['old_Y']
            if 'new_Y' in db:
                self.stat_analysis_logic.new_labels = db['new_Y']

            if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
                self.ui.statusBar.showMessage('Import canceled by user.')
                self._clear_all_parameters()
                return

    def read_normalizing_method_used(self, method: str) -> None:
        self.normalization_method = method
        if len(self.preprocessing.NormalizedDict) > 0:
            text_for_title = "<span style=\"font-family: AbletonSans; color:" \
                             + self.theme_colors['plotText'] \
                             + ";font-size:14pt\">Normalized plots. Method " \
                             + method + "</span>"
            self.ui.normalize_plot_widget.setTitle(text_for_title)

    def read_smoothing_method_used(self, method: str) -> None:
        self.smooth_method = method
        if len(self.preprocessing.smoothed_spectra) > 0:
            text_for_title = "<span style=\"font-family: AbletonSans; color:" \
                             + self.theme_colors['plotText'] \
                             + ";font-size:14pt\">Smoothed plots. Method " \
                             + method + "</span>"
            self.ui.smooth_plot_widget.setTitle(text_for_title)

    def read_baseline_method_used(self, method: str) -> None:
        self.baseline_method = method
        if len(self.preprocessing.baseline_dict) > 0:
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
            case 'ModPoly' | 'iModPoly' | 'ExModPoly':
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
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        file_path = fd.getOpenFileName(self, 'Open fit template file', path, "ZIP (*.zip)")
        if not file_path[0]:
            return
        path = file_path[0]
        self.ui.statusBar.showMessage('Reading data file...')
        self.close_progress_bar()
        self.open_progress_bar()
        self.open_progress_dialog("Opening template...", "Cancel")
        self.time_start = datetime.now()
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
            if "IgnoreTableChecked" in db:
                checked = db["IgnoreTableChecked"]
                self.ui.ignore_dataset_table_view.model().set_checked(checked)
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
        seconds = round((datetime.now() - self.time_start).total_seconds())
        self.set_modified(False)
        self.ui.statusBar.showMessage('Fit tenplate imported for ' + str(seconds) + ' sec.', 5000)
        if self.ui.fit_params_table.model().rowCount() != 0 \
                and self.ui.deconv_lines_table.model().rowCount() != 0:
            await self.fitting.draw_all_curves()

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
        path = os.getenv('APPDATA') + '/RS-tool'
        fd = QFileDialog(self)
        file_path = fd.getSaveFileName(self, 'Save fit template file', path, "ZIP (*.zip)")
        if not file_path[0]:
            return
        self.ui.statusBar.showMessage('Saving file...')
        self.close_progress_bar()
        self.open_progress_bar()
        filename = file_path[0]
        with shelve_open(filename, 'n') as db:
            db["DeconvLinesTableDF"] = self.ui.deconv_lines_table.model().dataframe()
            db["DeconvParamsTableDF"] = self.ui.fit_params_table.model().dataframe()
            db["intervals_table_df"] = self.ui.fit_intervals_table_view.model().dataframe()
            db["DeconvLinesTableChecked"] = self.ui.deconv_lines_table.model().checked()
            db["IgnoreTableChecked"] = self.ui.ignore_dataset_table_view.model().checked
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

    # region Main window functions

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
        if self.currentProgress:
            self.currentProgress.setMaximum(1)
            self.currentProgress.setValue(1)
        self.close_progress_bar()
        modules.start_program.show_error(type(err), err, str(tb))
        self.executor_stop()

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
        await self.preprocessing.update_plot_item(self.ImportedArray.items())
        await self.preprocessing.update_plot_item(self.preprocessing.ConvertedDict.items(), 1)
        await self.preprocessing.update_plot_item(self.preprocessing.CuttedFirstDict.items(), 2)
        await self.preprocessing.update_plot_item(self.preprocessing.NormalizedDict.items(), 3)
        await self.preprocessing.update_plot_item(self.preprocessing.smoothed_spectra.items(), 4)
        await self.preprocessing.update_plot_item(self.preprocessing.baseline_corrected_dict.items(), 5)
        await self.preprocessing.update_plot_item(self.preprocessing.averaged_dict.items(), 6)

    def open_progress_dialog(self, text: str, buttons: str, maximum: int = 0) -> None:
        self.currentProgress = QProgressDialog(text, buttons, 0, maximum)
        self.currentProgress.setWindowFlags(Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowTitleHint)
        self.currentProgress.setWindowModality(Qt.WindowModal)
        self.currentProgress.setWindowTitle(' ')
        self.currentProgress.open()
        cancel_button = self.currentProgress.findChild(QPushButton)
        environ['CANCEL'] = '0'
        cancel_button.clicked.connect(self.executor_stop)

    def executor_stop(self) -> None:
        if not self.current_executor or self.break_event is None:
            return
        for f in self.current_futures:
            if not f.done():
                f.cancel()
        try:
            self.break_event.set()
        except FileNotFoundError:
            warning('FileNotFoundError self.break_event.set()')
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

    def open_progress_bar(self, min_value: int = 0, max_value: int = 0) -> None:
        self.progressBar = QProgressBar(self)
        self.statusBar().insertPermanentWidget(0, self.progressBar, 1)
        self.progressBar.setRange(min_value, max_value)
        self.taskbar_button = QWinTaskbarButton()
        self.taskbar_progress = self.taskbar_button.progress()
        self.taskbar_progress.setRange(min_value, max_value)
        self.taskbar_button.setWindow(self.windowHandle())
        self.taskbar_progress.show()

    def cancelled_by_user(self) -> bool:
        """
        Cancel button was pressed by user?

        Returns
        -------
        out: bool
            True if Cancel button pressed
        """
        if self.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            self.close_progress_bar()
            self.ui.statusBar.showMessage('Cancelled by user.')
            info('Cancelled by user')
            return True
        else:
            return False

    def close_progress_bar(self) -> None:
        if self.currentProgress is not None:
            self.currentProgress.close()
        if self.progressBar is not None:
            self.statusBar().removeWidget(self.progressBar)
        if self.taskbar_progress is not None:
            self.taskbar_progress.hide()
            self.taskbar_progress.stop()

    def set_buttons_ability(self) -> None:
        self.action_despike.setDisabled(len(self.ImportedArray) == 0)
        self.action_interpolate.setDisabled(len(self.ImportedArray) < 2)
        self.action_convert.setDisabled(len(self.ImportedArray) == 0)
        self.action_cut.setDisabled(len(self.preprocessing.ConvertedDict) == 0)
        self.action_normalize.setDisabled(len(self.preprocessing.CuttedFirstDict) == 0)
        self.action_smooth.setDisabled(len(self.preprocessing.NormalizedDict) == 0)
        self.action_baseline_correction.setDisabled(len(self.preprocessing.smoothed_spectra) == 0)
        self.action_trim.setDisabled(len(self.preprocessing.baseline_corrected_not_trimmed_dict) == 0)
        self.action_average.setDisabled(len(self.preprocessing.baseline_corrected_not_trimmed_dict) == 0)

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
            self.save_with_shelve(self.project_path)

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
        if self.preprocessing.ConvertedDict:
            x_axis = next(iter(self.preprocessing.ConvertedDict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.cm_range_start.setValue(new_value)
        if new_value >= self.ui.cm_range_end.value():
            self.ui.cm_range_start.setValue(self.ui.cm_range_start.minimum())
        self.linearRegionCmConverted.setRegion((self.ui.cm_range_start.value(), self.ui.cm_range_end.value()))

    def cm_range_end_change_event(self, new_value: float) -> None:
        self.set_modified()
        if self.preprocessing.ConvertedDict:
            x_axis = next(iter(self.preprocessing.ConvertedDict.values()))[:, 0]
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
        if self.preprocessing.ConvertedDict:
            self.preprocessing.update_range_cm()
        else:
            self.ui.statusBar.showMessage('Range update failed because there are no any converted plot ', 15000)

    # endregion

    # region baseline_correction
    def lr_baseline_region_changed(self) -> None:
        current_region = self.linearRegionBaseline.getRegion()
        self.ui.trim_start_cm.setValue(current_region[0])
        self.ui.trim_end_cm.setValue(current_region[1])

    def _trim_start_change_event(self, new_value: float) -> None:
        self.set_modified()
        if self.preprocessing.baseline_corrected_not_trimmed_dict:
            x_axis = next(iter(self.preprocessing.baseline_corrected_not_trimmed_dict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.trim_start_cm.setValue(new_value)
        if new_value >= self.ui.trim_end_cm.value():
            self.ui.trim_start_cm.setValue(self.ui.trim_start_cm.minimum())
        self.linearRegionBaseline.setRegion((self.ui.trim_start_cm.value(), self.ui.trim_end_cm.value()))

    def _trim_end_change_event(self, new_value: float) -> None:
        self.set_modified()
        if self.preprocessing.baseline_corrected_not_trimmed_dict:
            x_axis = next(iter(self.preprocessing.baseline_corrected_not_trimmed_dict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.trim_end_cm.setValue(new_value)
        if new_value <= self.ui.trim_start_cm.value():
            self.ui.trim_end_cm.setValue(self.ui.trim_end_cm.maximum())
        self.linearRegionBaseline.setRegion((self.ui.trim_start_cm.value(), self.ui.trim_end_cm.value()))

    def update_trim_range_btn_clicked(self) -> None:
        if self.preprocessing.baseline_corrected_not_trimmed_dict:
            self.preprocessing.update_range_baseline_corrected()
        else:
            self.ui.statusBar.showMessage('Range update failed because there are no any baseline corrected plot ',
                                          15_000)

    # endregion

    # region Fitting page 2

    # region Add line
    @asyncSlot()
    async def add_deconv_line(self, line_type: str):
        if not self.fitting.is_template:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Switch to Template mode to add new line")
            msg.setWindowTitle("Add line failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        elif not self.preprocessing.baseline_corrected_dict:
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
        elif not self.preprocessing.baseline_corrected_dict or len(self.preprocessing.baseline_corrected_dict) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("There is No any data to fit")
            msg.setWindowTitle("Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        try:
            await self.fitting.do_batch_fit()
        except Exception as err:
            self.show_error(err)

    def set_deconvoluted_dataset(self) -> None:
        if self.ui.input_table.model().rowCount() == 0:
            return
        df = self.fitting.create_deconvoluted_dataset_new()
        self.ui.deconvoluted_dataset_table_view.model().set_dataframe(df)
        self.fitting.update_ignore_features_table()

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
        elif self.fitting.array_of_current_filename_in_deconvolution is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("There is No any data to fit")
            msg.setWindowTitle("Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
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
        if not self.preprocessing.baseline_corrected_dict:
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
                (not self.preprocessing.averaged_dict or len(self.preprocessing.averaged_dict) < 2):
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
            await self.fitting.do_auto_guess(line_type)
        except Exception as err:
            self.show_error(err)

    def _update_deconv_curve_style(self, style: dict, old_style: dict, index: int) -> None:
        command = CommandUpdateDeconvCurveStyle(self, index, style, old_style, "Update style for curve idx %s" % index)
        self.undoStack.push(command)

    def clear_all_deconv_lines(self) -> None:
        if self.fitting.timer_fill is not None:
            self.fitting.timer_fill.stop()
            self.fitting.timer_fill = None
            self.fitting.updating_fill_curve_idx = None
        command = CommandClearAllDeconvLines(self, 'Remove all deconvolution lines')
        self.undoStack.push(command)

    def curve_parameter_changed(self, value: float, line_index: int, param_name: str) -> None:
        self.fitting.CommandDeconvLineDraggedAllowed = True
        items_matches = self.fitting.deconvolution_data_items_by_idx(line_index)
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
        filename = '' if self.fitting.is_template else self.fitting.current_spectrum_deconvolution_name
        model = self.ui.fit_params_table.model()
        if 'add_params' not in self.fitting.peak_shapes_params[line_type]:
            self.fitting.redraw_curve(params, curve, line_type)
            return
        add_params = self.fitting.peak_shapes_params[line_type]['add_params']
        for s in add_params:
            if param_name == s:
                param = value
            else:
                param = model.get_parameter_value(filename, line_index, s, 'Value')
            params[s] = param
        self.fitting.redraw_curve(params, curve, line_type)

    # endregion

    # region Stat analysis (machine learning) page4
    @asyncSlot()
    async def fit_classificator(self, cl_type: str):
        """
        Проверяем что датасет заполнен.
        Переходим в процедуру создания классификатора
        """
        current_dataset = self.ui.dataset_type_cb.currentText()
        if current_dataset == 'Smoothed' and self.ui.smoothed_dataset_table_view.model().rowCount() == 0 \
                or current_dataset == 'Baseline corrected' \
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
        if cl_type == 'Torch':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("На этой версии убрана возможность создания PyTorch сети.")
            msg.setWindowTitle("Classificator Fitting failed")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        try:
            await self.stat_analysis_logic.do_fit_classificator(cl_type)
        except Exception as err:
            self.show_error(err)

    @asyncSlot()
    async def redraw_stat_plots(self) -> None:
        clfs = list(self.stat_analysis_logic.classificator_funcs.keys())
        if 'PCA' in clfs:
            clfs.remove('PCA')
        if 'PLS-DA' in clfs:
            clfs.remove('PLS-DA')
        self.stat_analysis_logic.update_stat_report_text()
        self.stat_analysis_logic.update_force_single_plots()
        self.stat_analysis_logic.update_force_full_plots()
        for clf in clfs:
            await self.loop.run_in_executor(None, self.stat_analysis_logic.update_plots, clf)
        await self.loop.run_in_executor(None, self.stat_analysis_logic.update_pca_plots)
        await self.loop.run_in_executor(None, self.stat_analysis_logic.update_plsda_plots)

    def current_tree_sb_changed(self, idx: int) -> None:
        if self.ui.stat_tab_widget.currentIndex() == 8 \
                and 'Random Forest' in self.stat_analysis_logic.latest_stat_result:
            model_results = self.stat_analysis_logic.latest_stat_result['Random Forest']
            model = model_results['model']
            update_plot_tree(model.best_estimator_.estimators_[idx], self.ui.rf_tree_plot_widget,
                             model_results['feature_names'], model_results['target_names'])
        elif self.ui.stat_tab_widget.currentIndex() == 11 and \
                'XGBoost' in self.stat_analysis_logic.latest_stat_result:
            model_results = self.stat_analysis_logic.latest_stat_result['XGBoost']
            model = model_results['model']
            self.stat_analysis_logic.update_xgboost_tree_plot(model.best_estimator_, idx)

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
