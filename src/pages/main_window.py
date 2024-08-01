# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin

"""
Module for handling main window events and operations in the application.

This module contains the MainWindow class which manages the main window
operations, key press events, undo and redo functionality, and other
helper functions related to the main window.

Classes
-------
Attrs
    Dataclass for storing various attributes related to the main window.

MainWindow
    Class to manage the main window operations and events.
"""

import dataclasses
from asyncio import create_task, sleep, wait
from os import environ, getenv, getpid
from pathlib import Path
from sys import exit

from asyncqtpy import asyncSlot
from matplotlib import pyplot as plt
from psutil import cpu_percent, Process
from pyqtgraph import setConfigOption
from qtpy.QtCore import Qt, QModelIndex, QTimer
from qtpy.QtGui import QFont, QIcon, QCloseEvent, QColor
from qtpy.QtWidgets import (QMenu, QMainWindow, QAction, QInputDialog, QTableView,
                            QScrollArea, QLabel)

from src.backend.progress import Progress
from src.data.config import get_config
from src.files.help import action_help
from src.stages.fitting.functions.guess_raman_lines import show_distribution
from src.ui.MultiLine import MultiLine
from src.ui.ui_average_widget import Ui_AverageForm
from src.ui.ui_bl_widget import Ui_BaselineForm
from src.ui.ui_cut_widget import Ui_CutForm
from src.ui.ui_main_window import Ui_MainWindow
from src.ui.ui_normalize_widget import Ui_NormalizeForm
from src.ui.ui_smooth_widget import Ui_SmoothForm
from src.widgets.setting_window import SettingWindow
from ..backend.context import Context
from ..backend.project import Project
from ..data.plotting import get_curve_plot_data_item
from ..ui.ui_convert_widget import Ui_ConvertForm
from ..ui.ui_import_widget import Ui_ImportForm
from ..widgets.drag_items import DragWidget, DragItem
from qfluentwidgets import TableView


@dataclasses.dataclass
class Attrs:
    """
    A dataclass to store various attributes related to the main window.

    Attributes
    ----------
    plot_background_color : QColor
        Background color for the plot.
    plot_text_color : QColor
        Text color for the plot.
    window_maximized : bool
        Flag indicating if the window is maximized.
    latest_file_path : str
        Path to the latest file used.
    action_undo : QAction
        Undo action.
    action_redo : QAction
        Redo action.
    """
    plot_background_color: QColor
    plot_text_color: QColor
    window_maximized: bool
    latest_file_path: str
    action_undo: QAction
    action_redo: QAction


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Main window widget with all user interface elements.

    This class initializes and manages the main window, handles events,
    and provides various functionalities for the application.
    """

    def __init__(self) -> None:
        """
        Initialize the main window.

        This constructor sets up the user interface, initializes attributes,
        and configures various components of the main window.

        Returns
        -------
        None
        """
        super().__init__(None)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.attrs = Attrs(plot_background_color=QColor(environ["plotBackground"]),
                           plot_text_color=QColor(environ["plotText"]),
                           window_maximized=True,
                           latest_file_path=getenv("APPDATA") + "/RS-tool",
                           action_undo=QAction(), action_redo=QAction())
        self.context = Context(self)
        self.progress = Progress(self)
        self.project = Project(self)
        self.ui.memory_usage_label = QLabel(self)
        self.statusBar().addPermanentWidget(self.ui.memory_usage_label)
        self.drag_pos = None
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowFrameSection.TopSection
                            | Qt.WindowType.WindowMinMaxButtonsHint)
        self.keyPressEvent = self._key_press_event
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowIcon(QIcon(get_config()["logo"]["path"]))
        self.setWindowTitle("Raman Spectroscopy Tool ")
        self.update_icons()
        self._initial_ui_definitions()
        self._initial_menu()
        self._init_left_menu()
        self._initial_right_scrollbar()
        self._initial_plots()
        self._initial_plot_buttons()
        self._initial_timers()
        self.setAcceptDrops(True)
        self._init_drag_widget()
        self.context.set_modified(False)

    def _init_drag_widget(self):
        """
        Initialize the drag widget.

        This method sets up the drag widget with various UI forms and connects
        the necessary signals.

        Returns
        -------
        None
        """
        self.ui.chain_layout.setAlignment(Qt.AlignLeft)
        self.ui.drag_widget = DragWidget(orientation=Qt.Orientation.Horizontal)
        for ui, dr, b in zip((Ui_ImportForm, Ui_ConvertForm, Ui_CutForm, Ui_BaselineForm,
                              Ui_SmoothForm, Ui_NormalizeForm, Ui_CutForm, Ui_AverageForm),
                             (0, 0, 0, 1, 1, 1, 0, 0),
                             (self.context.preprocessing.stages.input_data,
                              self.context.preprocessing.stages.convert_data,
                              self.context.preprocessing.stages.cut_data,
                              self.context.preprocessing.stages.bl_data,
                              self.context.preprocessing.stages.smoothed_data,
                              self.context.preprocessing.stages.normalized_data,
                              self.context.preprocessing.stages.trim_data,
                              self.context.preprocessing.stages.av_data)):
            form = DragItem(draggable=dr)
            w = ui()
            w.setupUi(form)
            form.set_backend_instance(b)
            b.set_ui(w)
            self.ui.drag_widget.add_item(form)
        self.ui.drag_widget.doubleClickedWidget.connect(self._widget_selected)
        self.ui.chain_layout.addWidget(self.ui.drag_widget)

    def _widget_selected(self, w: DragWidget) -> None:
        """
        Handle selection of a widget.

        Parameters
        ----------
        w : DragWidget
            The selected drag widget.

        Returns
        -------
        None
        """
        self.context.preprocessing.active_stage = w.backend_instance
        self.context.preprocessing.update_plot_item(w.backend_instance.name)

    def dragEnterEvent(self, e) -> None:
        """
        Handle drag enter event.

        Parameters
        ----------
        e : QDragEnterEvent
            The drag enter event.

        Returns
        -------
        None
        """
        e.accept()

    def dropEvent(self, e) -> None:
        """
        Handle drop event.

        Parameters
        ----------
        e : QDropEvent
            The drop event.

        Returns
        -------
        None
        """
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

    def mousePressEvent(self, event) -> None:
        """
        Handle mouse press event.

        Parameters
        ----------
        event : QMouseEvent
            The mouse press event.

        Returns
        -------
        None
        """
        # SET DRAG POS WINDOW
        self.drag_pos = event.globalPos()

    def closeEvent(self, a0: QCloseEvent) -> None:
        """
        Handle close event.

        Parameters
        ----------
        a0 : QCloseEvent
            The close event.

        Returns
        -------
        None
        """
        if not self.project.can_close_project():
            a0.ignore()
            return
        self.ui.preproc_plot_widget.getPlotItem().close()
        del self.ui.preproc_plot_widget.vertical_line
        del self.ui.preproc_plot_widget.horizontal_line
        self.project.delete_db_files()
        exit()

    # region init

    # region plots
    def _initial_preproc_plot_color(self) -> None:
        """
        Initialize pre-processing plot color.

        This method sets the background and text color for the pre-processing plot.

        Returns
        -------
        None
        """
        self.ui.preproc_plot_widget.setBackground(self.attrs.plot_background_color)
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item.getAxis("bottom").setPen(self.attrs.plot_text_color)
        plot_item.getAxis("left").setPen(self.attrs.plot_text_color)
        plot_item.getAxis("bottom").setTextPen(self.attrs.plot_text_color)
        plot_item.getAxis("left").setTextPen(self.attrs.plot_text_color)

    def _initial_plots(self) -> None:
        """
        Initialize plots.

        This method sets up the initial configuration for various plots.

        Returns
        -------
        None
        """
        setConfigOption("antialias", True)
        self.ui.stackedWidget_mainpages.currentChanged.connect(self._stacked_widget_changed)
        self.context.ml.initial_all_stat_plots()
        self.initial_plots_set_fonts()
        self.initial_plots_labels()

    def initial_plots_set_fonts(self) -> None:
        """
        Set fonts for plots.

        This method sets the font styles for the plot elements.

        Returns
        -------
        None
        """
        plot_font = QFont("AbletonSans", int(environ["plot_font_size"]), QFont.Weight.Normal)
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item.getAxis("bottom").setStyle(tickFont=plot_font)
        plot_item.getAxis("left").setStyle(tickFont=plot_font)
        self.ui.deconv_plot_widget.getPlotItem().getAxis("bottom").setStyle(tickFont=plot_font)
        self.ui.deconv_plot_widget.getPlotItem().getAxis("left").setStyle(tickFont=plot_font)
        plt.rcParams.update({"font.size": int(environ["plot_font_size"])})

    def get_plot_label_style(self) -> dict:
        """
        Get the plot label style.

        Returns
        -------
        dict
            A dictionary containing the plot label style attributes.
        """
        return {
            "color": self.attrs.plot_text_color.name(),
            "font-size": str(environ["axis_label_font_size"]) + "pt",
            "font-family": "AbletonSans",
        }

    def initial_plots_labels(self) -> None:
        """
        Initialize plot labels.

        This method sets the labels for the plots with the appropriate styles.

        Returns
        -------
        None
        """
        label_style = self.get_plot_label_style()
        self.ui.preproc_plot_widget.setLabel("left", "Intensity, rel. un.", units="", **label_style)
        self.ui.deconv_plot_widget.setLabel("left", "Intensity, rel. un.", units="", **label_style)
        self.ui.deconv_plot_widget.setLabel(
            "bottom", "Raman shift, cm\N{superscript minus}\N{superscript one}", units="",
            **label_style)

    # endregion

    # region MenuBar

    def _initial_menu(self) -> None:
        """
        Initialize the menu bar.

        This method sets up the menu bar with various menus and actions.

        Returns
        -------
        None
        """
        self._init_file_menu()
        self._init_edit_menu()
        self._init_stat_analysis_menu()

    def _init_file_menu(self) -> None:
        """
        Initialize the file menu.

        This method sets up the file menu with various actions.

        Returns
        -------
        None
        """
        file_menu = QMenu(self)
        file_menu.addAction("New Project", self.project.action_new_project)
        file_menu.addAction("Open Project", self.project.action_open_project)
        recent_menu = file_menu.addMenu("Open Recent")
        self.project.set_recent_menu(recent_menu)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu("Export")
        export_menu.addAction("Tables to excel", self.context.datasets.action_export_table_excel)
        export_menu.addAction("Production project", self.project.action_save_production_project)
        export_menu.addAction("Decomposed lines to .csv",
                              self.context.datasets.action_save_decomposed_to_csv)
        file_menu.addSeparator()
        file_menu_save_all_action = QAction("Save all", file_menu)
        file_menu_save_all_action.triggered.connect(self.project.action_save_project)
        file_menu_save_all_action.setShortcut("Ctrl+S")
        file_menu_save_as_action = QAction("Save as", file_menu)
        file_menu_save_as_action.triggered.connect(self.project.action_save_as)
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
        """
        Initialize the edit menu.

        This method sets up the edit menu with various actions.

        Returns
        -------
        None
        """
        edit_menu = QMenu(self)
        self.attrs.action_undo = QAction("Undo")
        self.attrs.action_undo.triggered.connect(self._undo)
        self.attrs.action_undo.setShortcut("Ctrl+Z")
        self.attrs.action_redo = QAction("Redo")
        self.attrs.action_redo.triggered.connect(self._redo)
        self.attrs.action_redo.setShortcut("Ctrl+Y")
        actions = [self.attrs.action_undo, self.attrs.action_redo]
        edit_menu.addActions(actions)
        edit_menu.setToolTipsVisible(True)
        self.attrs.action_undo.setToolTip("")
        edit_menu.addSeparator()
        clear_menu = edit_menu.addMenu("Clear")
        clear_menu.addAction("Fitting lines",
                             self.context.decomposition.tables.decomp_lines.clear_all_deconv_lines)
        clear_menu.addAction("All fitting data", self.context.decomposition.reset)
        clear_menu.addSeparator()
        clear_menu.addAction("Smoothed dataset", self.context.datasets.reset_smoothed_dataset_table)
        clear_menu.addAction("Baseline dataset",
                             self.context.datasets.reset_baselined_dataset_table)
        clear_menu.addAction("Decomposed dataset",
                             self.context.datasets.reset_deconvoluted_dataset_table)
        clear_menu.addAction("Ignore features", self.context.datasets.reset_ignore_dataset_table)
        clear_menu.addSeparator()
        clear_menu.addAction("LDA", lambda: self.context.ml.clear_model("LDA"))
        clear_menu.addAction("Logistic regression",
                             lambda: self.context.ml.clear_model("Logistic regression"), )
        clear_menu.addAction("SVC", lambda: self.context.ml.clear_model("SVC"))
        clear_menu.addAction("Decision Tree", lambda: self.context.ml.clear_model("Decision Tree"))
        clear_menu.addAction("Random Forest", lambda: self.context.ml.clear_model("Random Forest"))
        clear_menu.addAction("XGBoost", lambda: self.context.ml.clear_model("XGBoost"))
        clear_menu.addSeparator()
        clear_menu.addAction("Predicted", self.context.predict.reset)
        self.ui.EditBtn.setMenu(edit_menu)

    def _init_stat_analysis_menu(self) -> None:
        """
        Initialize the statistical analysis menu.

        This method sets up the statistical analysis menu with various actions.

        Returns
        -------
        None
        """
        menu = QMenu(self)
        menu.addAction("Fit", self.context.ml.fit_classificator)
        menu.addAction("PCA", lambda: self.context.ml.fit_classificator("PCA"))
        menu.addAction("Refresh plots", self.context.ml.redraw_stat_plots)
        menu.addAction("Refresh SHAP", self.context.ml.refresh_shap_push_button_clicked)
        menu.addAction("Refresh learning curve", self.context.ml.refresh_learning_curve)
        menu.addAction("Refresh Optuna plots", self.context.ml.plots.update_optuna_plots)
        self.ui.stat_analysis_btn.setMenu(menu)

    # endregion

    # region left_side_menu

    def _init_left_menu(self) -> None:
        """
        Initialize the left menu.

        This method sets up the left side menu with various buttons and actions.

        Returns
        -------
        None
        """
        self.ui.left_side_frame.setFixedWidth(350)
        self.ui.left_hide_frame.hide()
        self.ui.dec_list_btn.setVisible(False)
        self.ui.gt_add_Btn.setToolTip("Add new group")
        self.ui.gt_add_Btn.clicked.connect(self.context.group_table.add_new_group)
        self.ui.gt_dlt_Btn.setToolTip("Delete selected group")
        self.ui.gt_dlt_Btn.clicked.connect(self.context.group_table.dlt_selected_group)
        self.ui.leftsideBtn.clicked.connect(self._left_side_btn_clicked)
        self.ui.dec_list_btn.clicked.connect(self._dec_list_btn_clicked)
        self.ui.stat_param_btn.clicked.connect(self._stat_param_btn_clicked)
        self.ui.intervals_gb.toggled.connect(self._intervals_gb_toggled)

    def _intervals_gb_toggled(self, b: bool) -> None:
        self.context.set_modified()
        self.ui.fit_borders_TableView.setVisible(b)
        if b:
            self.ui.intervals_gb.setMaximumHeight(200)
        else:
            self.ui.intervals_gb.setMaximumHeight(0)

    # endregion

    # region other

    def _initial_right_scrollbar(self) -> None:
        """
        Initialize the right scrollbar.

        This method sets up the right scrollbar with initial configurations.

        Returns
        -------
        None
        """
        self.ui.verticalScrollBar.setVisible(False)
        self.ui.verticalScrollBar.setMinimum(1)
        self.ui.verticalScrollBar.enterEvent = self._vertical_scroll_bar_enter_event
        self.ui.verticalScrollBar.leaveEvent = self._vertical_scroll_bar_leave_event
        self.ui.verticalScrollBar.valueChanged.connect(self._vertical_scroll_bar_value_changed)
        self.ui.data_tables_tab_widget.currentChanged.connect(
            self.decide_vertical_scroll_bar_visible)
        self.ui.page1Btn.clicked.connect(self._page1_btn_clicked)
        self.ui.page2Btn.clicked.connect(self._page2_btn_clicked)
        self.ui.page3Btn.clicked.connect(self._page3_btn_clicked)
        self.ui.page4Btn.clicked.connect(self._page4_btn_clicked)
        self.ui.page5Btn.clicked.connect(self._page5_btn_clicked)

    def _initial_plot_buttons(self) -> None:
        """
        Initialize plot buttons.

        This method sets up the buttons related to the plots.

        Returns
        -------
        None
        """
        self.ui.crosshairBtn.clicked.connect(self.crosshair_btn_clicked)
        self.ui.by_one_control_button.clicked.connect(self.by_one_control_button_clicked)
        self.ui.by_group_control_button.clicked.connect(self.by_group_control_button)
        self.ui.by_group_control_button.mouseDoubleClickEvent = \
            self._by_group_control_button_double_clicked
        self.ui.all_control_button.clicked.connect(self.all_control_button)
        self.ui.lr_movableBtn.clicked.connect(self._linear_region_movable_btn_clicked)
        self.ui.lr_showHideBtn.clicked.connect(self._linear_region_show_hide_btn_clicked)
        self.ui.sun_Btn.clicked.connect(self._change_plots_bckgrnd)

    def _initial_timers(self) -> None:
        """
        Initialize timers.

        This method sets up various timers used in the application.

        Returns
        -------
        None
        """
        timer_mem_update = QTimer(self)
        timer_mem_update.timeout.connect(self._set_timer_memory_update)
        timer_mem_update.start(1000)
        cpu_load = QTimer(self)
        cpu_load.timeout.connect(self._set_cpu_load)
        cpu_load.start(300)

    def _initial_ui_definitions(self) -> None:
        """
        Initialize UI definitions.

        This method sets up the initial UI elements and configurations.

        Returns
        -------
        None
        """
        self.ui.maximizeRestoreAppBtn.setToolTip("Maximize")
        self.ui.unsavedBtn.hide()
        self.ui.titlebar.mouseDoubleClickEvent = self._double_click_maximize_restore
        self.ui.titlebar.mouseMoveEvent = self._move_window
        self.ui.titlebar.mouseReleaseEvent = self._titlebar_mouse_release_event
        self.ui.right_buttons_frame.mouseMoveEvent = self._move_window
        self.ui.right_buttons_frame.mouseReleaseEvent = self._titlebar_mouse_release_event
        self.ui.minimizeAppBtn.clicked.connect(lambda: self.showMinimized())
        self.ui.maximizeRestoreAppBtn.clicked.connect(lambda: self._maximize_restore())
        self.ui.closeBtn.clicked.connect(lambda: self.close())
        self.ui.settingsBtn.clicked.connect(lambda: SettingWindow(self).show())

    def _titlebar_mouse_release_event(self, _) -> None:
        """
        Handle title bar mouse release event.

        Parameters
        ----------
        _ : QMouseEvent
            The mouse release event.

        Returns
        -------
        None
        """
        self.setWindowOpacity(1)

    def _move_window(self, mouse_event) -> None:
        """
        Handle window move event.

        Parameters
        ----------
        mouse_event : QMouseEvent
            The mouse event.

        Returns
        -------
        None
        """
        # IF MAXIMIZED CHANGE TO NORMAL
        if self.attrs.window_maximized:
            self._maximize_restore()
        # MOVE WINDOW
        if mouse_event.buttons() == Qt.MouseButton.LeftButton:
            new_pos = self.pos() + mouse_event.globalPos() - self.drag_pos
            self.setWindowOpacity(0.9)
            self.move(new_pos)
            self.drag_pos = mouse_event.globalPos()
            mouse_event.accept()

    def _double_click_maximize_restore(self, mouse_event) -> None:
        """
        Handle double click on window event.

        Parameters
        ----------
        mouse_event : QMouseEvent
            The mouse event.

        Returns
        -------
        None
        """
        # IF DOUBLE CLICK CHANGE STATUS
        if mouse_event.type() == 4:
            timer = QTimer(self)
            timer.singleShot(250, self._maximize_restore)

    def _maximize_restore(self) -> None:
        """
        Handle maximize and restore events.
        """
        if not self.attrs.window_maximized:
            self.showMaximized()
            self.attrs.window_maximized = True
            self.ui.maximizeRestoreAppBtn.setToolTip("Restore")
            self._set_icon_for_restore_button()
        else:
            self.attrs.window_maximized = False
            self.showNormal()
            self.ui.maximizeRestoreAppBtn.setToolTip("Maximize")
            self._set_icon_for_restore_button()

    def _set_icon_for_restore_button(self) -> None:
        """
        Change icon for restore button after clicked event.
        """
        if "Light" in environ["theme"] and self.attrs.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down_black.svg"))
        elif "Light" in environ["theme"] and not self.attrs.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-up_black.svg"))
        elif self.attrs.window_maximized:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down.svg"))
        else:
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon("material/resources/source/chevron-up.svg"))

    def _show_hide_left_menu(self) -> None:
        """
        Change left sidebar size and icons.
        """
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
                    QIcon("material/resources/source/chevron-left_black.svg"))
            else:
                self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/chevron-left.svg"))
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
                self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/sliders_black.svg"))
                self.ui.dec_list_btn.setIcon(
                    QIcon("material/resources/source/align-justify_black.svg"))
                self.ui.stat_param_btn.setIcon(QIcon("material/resources/source/percent_black.svg"))
            else:
                self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/sliders.svg"))
                self.ui.dec_list_btn.setIcon(QIcon("material/resources/source/align-justify.svg"))
                self.ui.stat_param_btn.setIcon(QIcon("material/resources/source/percent.svg"))

    def _left_side_btn_clicked(self) -> None:
        """
        Handle button clicked event for show / hide left sidebar.
        """
        self._show_hide_left_menu()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_1)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_1)

    def _dec_list_btn_clicked(self) -> None:
        """
        Handle button clicked event for show / hide left sidebar in case decomposition bar clicked.
        """
        self._show_hide_left_menu()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_2)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_2)

    def _stat_param_btn_clicked(self) -> None:
        """
        Handle button clicked event for show / hide left sidebar in case ML bar clicked.
        """
        self._show_hide_left_menu()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_3)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_3)

    def _change_plots_bckgrnd(self) -> None:
        """
        Change color of plots background when sunBtn clicked.
        """
        if self.ui.sun_Btn.isChecked():
            self.attrs.plot_text_color = QColor(environ["inversePlotText"])
            self.attrs.plot_background_color = QColor(environ["inversePlotBackground"])
            plt.style.use(["default"])
        else:
            self.attrs.plot_text_color = QColor(environ["plotText"])
            self.attrs.plot_background_color = QColor(environ["plotBackground"])
            plt.style.use(["dark_background"])
        self._initial_preproc_plot_color()
        self.context.decomposition.graph_drawing.initial_deconv_plot_color()
        self.context.ml.initial_stat_plots_color()
        self.initial_plots_labels()

    # endregion

    # endregion

    # region Plot buttons

    # region crosshair button
    def _linear_region_movable_btn_clicked(self) -> None:
        """
        Set linear region movable or not.
        """
        b = not self.ui.lr_movableBtn.isChecked()
        self.context.preprocessing.stages.cut_data.linear_region.setMovable(b)
        self.context.decomposition.plotting.linear_region.setMovable(b)

    def _linear_region_show_hide_btn_clicked(self) -> None:
        """
        Show / hide linear regions.
        """
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item_dec = self.ui.deconv_plot_widget.getPlotItem()
        if self.ui.lr_showHideBtn.isChecked():
            plot_item.addItem(self.context.preprocessing.stages.cut_data.linear_region)
            plot_item_dec.addItem(self.context.decomposition.plotting.linear_region)
        else:
            plot_item.removeItem(self.context.preprocessing.stages.cut_data.linear_region)
            plot_item_dec.removeItem(self.context.decomposition.plotting.linear_region)

    def crosshair_btn_clicked(self) -> None:
        """
        Add crosshair with coordinates at title.
        """
        if self.ui.stackedWidget_mainpages.currentIndex() == 0:
            self._crosshair_preproc_plot()
        elif self.ui.stackedWidget_mainpages.currentIndex() == 1:
            self._crosshair_btn_clicked_for_deconv_plot()

    def _crosshair_preproc_plot(self) -> None:
        """
        Handle crosshair for preprocessing plot.
        """
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        plot_item.removeItem(self.ui.preproc_plot_widget.vertical_line)
        plot_item.removeItem(self.ui.preproc_plot_widget.horizontal_line)
        if self.ui.crosshairBtn.isChecked():
            plot_item.addItem(self.ui.preproc_plot_widget.vertical_line, ignoreBounds=True)
            plot_item.addItem(self.ui.preproc_plot_widget.horizontal_line, ignoreBounds=True)
        elif not self.ui.by_one_control_button.isChecked():
            self.context.preprocessing.set_preproc_title(self.ui.drag_widget.get_current_widget())

    def _crosshair_btn_clicked_for_deconv_plot(self) -> None:
        """
        Handle crosshair for fitting plot.
        """
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

    # endregion

    # region '1' button

    @asyncSlot()
    async def by_one_control_button_clicked(self) -> None:
        """
        Using with button self.ui.by_one_control_button for hide all plot item besides selected
        in input table
        """
        if self.ui.by_one_control_button.isChecked():
            self.ui.by_group_control_button.setChecked(False)
            self.ui.all_control_button.setChecked(False)
            tasks = [create_task(self.update_plots_for_single())]
            await wait(tasks)
        else:
            self.ui.by_one_control_button.setChecked(True)

    async def update_plots_for_single(self) -> None:
        """
        Loop to set visible for all plot items
        """
        self.ui.statusBar.showMessage("Updating plot...")
        current_index = self.ui.input_table.selectionModel().currentIndex()
        if current_index.row() == -1:
            return
        current_spectrum_name = self.ui.input_table.model().get_filename_by_row(current_index.row())
        group_number = self.ui.input_table.model().cell_data(current_index.row(), 2)
        new_title = (
            '<span style="font-family: AbletonSans; color:'
            f' {environ["plotText"]};font-size:14pt"> {current_spectrum_name}</span>'
        )
        tasks = [create_task(self._update_single_plot(new_title, current_spectrum_name,
                                                      group_number))]
        await wait(tasks)

        self.ui.statusBar.showMessage("Plot updated", 5000)

    async def _update_single_plot(self, new_title: str, current_spectrum_name: str,
                                  group_number: str) -> None:
        """
        Show only one spectrum.
        """
        data_items = self.ui.preproc_plot_widget.getPlotItem().listDataItems()
        if len(data_items) <= 0:
            return
        self.ui.preproc_plot_widget.setTitle(new_title)
        for i in data_items:
            i.setVisible(False)
        current_widget = self.ui.drag_widget.get_current_widget_name()
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
        color = self.context.group_table.get_color_by_group_number(group_number)
        self.context.preprocessing.one_curve = get_curve_plot_data_item(arr, color)
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
              in self.context.preprocessing.stages.bl_data.baseline_data):
            tasks = [create_task(
                self.context.preprocessing.stages.bl_data.baseline_add_plot(current_spectrum_name))]
            await wait(tasks)
        self.ui.preproc_plot_widget.getPlotItem().getViewBox().updateAutoRange()

    # endregion

    # region 'G' button
    @asyncSlot()
    async def by_group_control_button(self) -> None:
        """
        Using with button self.ui.by_group_control_button for hide all plot items besides
        selected in group table
        """
        if self.ui.by_group_control_button.isChecked():
            self.ui.by_one_control_button.setChecked(False)
            self.ui.all_control_button.setChecked(False)
            tasks = [create_task(
                self.context.preprocessing.stages.input_data.despike_history_remove_plot())]
            await wait(tasks)
            await self.update_plots_for_group(None)
        else:
            self.ui.by_group_control_button.setChecked(True)

    def _by_group_control_button_double_clicked(self, _) -> None:
        """
        Show input dialog to select groups id to render.
        """
        if self.context.group_table.table_widget.model().rowCount() < 2:
            return
        input_dialog = QInputDialog(self)
        result = input_dialog.getText(self, "Choose visible groups",
                                      "Write groups numbers to show (example: 1, 2, 3):")
        if not result[1]:
            return
        v = list(result[0].strip().split(","))
        self.context.preprocessing.stages.input_data.despike_history_remove_plot()
        groups = [int(x) for x in v]
        self.update_plots_for_group(groups)

    @asyncSlot()
    async def update_plots_for_group(self, current_group: list[int] | None) -> None:
        """
        loop to set visible for all plot items in group
        """
        if not current_group:
            current_row \
                = self.context.group_table.table_widget.selectionModel().currentIndex().row()
            current_group_name = self.context.group_table.table_widget.model().cell_data(
                current_row, 0)
            current_group = [current_row + 1]
            new_title = ('<span style="font-family: AbletonSans; color:' + environ["plotText"]
                         + ';font-size:14pt">' + current_group_name + "</span>")
        else:
            new_title = ('<span style="font-family: AbletonSans; color:' + environ["plotText"]
                         + ';font-size:14pt">' + str(current_group) + "</span>")
        self.ui.preproc_plot_widget.setTitle(new_title)

        self.ui.statusBar.showMessage("Updating plot...")
        tasks = [
            create_task(self.context.preprocessing.stages.input_data.despike_history_remove_plot()),
        ]
        await wait(tasks)
        tasks = [create_task(self._update_group_plot(current_group)), ]
        await wait(tasks)
        self.ui.statusBar.showMessage("Plot updated", 5000)

    async def _update_group_plot(self, current_group: list[int]) -> None:
        """
        Show spectra of selected group.
        """
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
        """
        loop to set visible True for all plot items
        """
        if self.ui.all_control_button.isChecked():
            self.ui.by_one_control_button.setChecked(False)
            self.ui.by_group_control_button.setChecked(False)
            tasks = [
                create_task(
                    self.context.preprocessing.stages.input_data.despike_history_remove_plot()),
                create_task(self.context.preprocessing.stages.bl_data.baseline_remove_plot()),
                create_task(self._update_plot_all()),
            ]
            await wait(tasks)
        else:
            self.ui.all_control_button.setChecked(True)

    async def _update_plot_all(self) -> None:
        """
        Show all spectra.
        """
        self.ui.statusBar.showMessage("Updating plot...")
        self.context.preprocessing.set_preproc_title(self.ui.drag_widget.get_current_widget())
        plot_item = self.ui.preproc_plot_widget.getPlotItem()
        data_items = plot_item.listDataItems()
        for i in data_items:
            i.setVisible(True)
        if self.context.preprocessing.one_curve:
            plot_item.removeItem(self.context.preprocessing.one_curve)
        plot_item.getViewBox().updateAutoRange()
        self.ui.statusBar.showMessage("Plot updated", 5000)

    # endregion

    def _stacked_widget_changed(self) -> None:
        """
        Page change event.
        """
        self.crosshair_btn_clicked()
        self.ui.deconv_plot_widget.getPlotItem().getViewBox().updateAutoRange()

    # endregion

    # region VERTICAL SCROLL BAR

    def _vertical_scroll_bar_value_changed(self, event: int) -> None:
        """
        Scroll connected table view to main scroll bar.
        """
        match self.ui.stackedWidget_mainpages.currentIndex():
            case 0:
                self.ui.input_table.verticalScrollBar().setValue(event)
            case 2:
                idx = self.ui.stackedWidget_mainpages.currentIndex()
                if idx == 0:
                    self.ui.smoothed_dataset_table_view.verticalScrollBar().setValue(event)
                elif idx == 1:
                    self.ui.baselined_dataset_table_view.verticalScrollBar().setValue(event)
                elif idx == 2:
                    self.ui.deconvoluted_dataset_table_view.verticalScrollBar().setValue(event)
                else:
                    self.ui.ignore_dataset_table_view.verticalScrollBar().setValue(event)
            case 4:
                self.ui.predict_table_view.verticalScrollBar().setValue(event)

    def _vertical_scroll_bar_enter_event(self, _) -> None:
        """
        CSS render option.
        """
        self.ui.verticalScrollBar.setStyleSheet(
            "#verticalScrollBar {background: {{scrollLineHovered}};}")

    def _vertical_scroll_bar_leave_event(self, _) -> None:
        """
        CSS render option.
        """
        self.ui.verticalScrollBar.setStyleSheet("#verticalScrollBar {background: transparent;}")

    def move_side_scrollbar(self, idx: int) -> None:
        """
        Scroll connected table view to main scroll bar.
        """
        self.ui.verticalScrollBar.setValue(idx)

    def _page1_btn_clicked(self) -> None:
        """
        Handle event when changed to page 1.
        """
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page1)
        self._update_pages_icons()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_1)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_1)
        self.decide_vertical_scroll_bar_visible()

    def _page2_btn_clicked(self) -> None:
        """
        Handle event when changed to page 2.
        """
        self.ui.verticalScrollBar.setVisible(False)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page2)
        self._update_pages_icons()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_2)
        self.context.decomposition.switch_template('Average')
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_2)

    def _page3_btn_clicked(self) -> None:
        """
        Handle event when changed to page 3.
        """
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page3)
        self._update_pages_icons()
        self.decide_vertical_scroll_bar_visible()

    def _page4_btn_clicked(self) -> None:
        """
        Handle event when changed to page 4.
        """
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page4)
        self._update_pages_icons()
        self.ui.stackedWidget_left.setCurrentWidget(self.ui.left_page_3)
        self.ui.left_side_head_stackedWidget.setCurrentWidget(self.ui.lsh_sw_3)
        self.decide_vertical_scroll_bar_visible()

    def _page5_btn_clicked(self) -> None:
        """
        Handle event when changed to page 5.
        """
        self.ui.verticalScrollBar.setVisible(True)
        self.ui.stackedWidget_mainpages.setCurrentWidget(self.ui.page5)
        self._update_pages_icons()
        self.decide_vertical_scroll_bar_visible()

    def _update_pages_icons(self):
        """
        Handle event when changed page.
        """
        self.ui.page1Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page1)
        self.ui.page2Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page2)
        self.ui.page3Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page3)
        self.ui.page4Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page4)
        self.ui.page5Btn.setChecked(
            self.ui.stackedWidget_mainpages.currentWidget() == self.ui.page5)

    # endregion

    def decide_vertical_scroll_bar_visible(
            self, _model_index: QModelIndex = None, _start: int = 0, _end: int = 0
    ) -> None:
        """
        Show or hide scroll bar.
        """
        self.ui.verticalScrollBar.setVisible(False)
        match self.ui.stackedWidget_mainpages.currentIndex():
            case 0:
                tv = self.ui.input_table
            case 2:
                idx = self.ui.data_tables_tab_widget.currentIndex()
                if idx == 0:
                    tv = self.ui.smoothed_dataset_table_view
                elif idx == 1:
                    tv = self.ui.baselined_dataset_table_view
                elif idx == 2:
                    tv = self.ui.deconvoluted_dataset_table_view
                else:
                    tv = self.ui.ignore_dataset_table_view
            case 4:
                tv = self.ui.predict_table_view
            case 1 | 3 | _:
                return
        if isinstance(tv, QTableView):
            row_count = tv.model().rowCount()
            row_height = tv.rowHeight(0)
            if row_count > 0:
                page_step = tv.verticalScrollBar().pageStep()
                self.ui.verticalScrollBar.setMinimum(tv.verticalScrollBar().minimum())
                self.ui.verticalScrollBar.setVisible(page_step <= row_height * row_count)
                self.ui.verticalScrollBar.setPageStep(page_step)
                if tv.verticalScrollBar().maximum() > 0:
                    max_v = tv.verticalScrollBar().maximum()
                else:
                    max_v = row_count * row_height - page_step
                self.ui.verticalScrollBar.setMaximum(max_v)
            else:
                self.ui.verticalScrollBar.setVisible(False)
            self.ui.verticalScrollBar.setValue(tv.verticalScrollBar().value())
        elif isinstance(tv, QScrollArea):
            self.ui.verticalScrollBar.setValue(tv.verticalScrollBar().value())
            self.ui.verticalScrollBar.setMinimum(0)
            self.ui.verticalScrollBar.setVisible(True)
            self.ui.verticalScrollBar.setMaximum(tv.verticalScrollBar().maximum())

    # region EVENTS

    def _key_press_event(self, key_event) -> None:
        """
        Handle key press events for the main window.

        Parameters
        ----------
        key_event : QKeyEvent
            The key event triggered by the user.

        Returns
        -------
        None
        """
        match key_event.key():
            case (Qt.Key.Key_Control, Qt.Key.Key_Z):
                self._undo()
            case (Qt.Key.Key_Control, Qt.Key.Key_Y):
                self._redo()
            case (Qt.Key.Key_Control, Qt.Key.Key_S):
                self.project.action_save_project()
            case (Qt.Key.Key_Shift, Qt.Key.Key_S):
                self.project.action_save_as()
            case Qt.Key.Key_End:
                self.progress.executor_stop()
            case Qt.Key.Key_F1:
                action_help()
            case Qt.Key.Key_F2:
                if self.ui.stackedWidget_mainpages.currentIndex() != 1:
                    return
                if self.context.decomposition.plotting.intervals_data is None:
                    return
                for i, v in enumerate(self.context.decomposition.plotting.intervals_data.items()):
                    _, item = v
                    show_distribution(
                        item["x0"],
                        self.context.preprocessing.stages.av_data.data,
                        self.context.decomposition.data.all_ranges_clustered_x0_sd[i],
                        self.context.group_table.table_widget.model().groups_colors)
            case Qt.Key.Key_F11:
                if not self.isFullScreen():
                    self.showFullScreen()
                else:
                    self.showMaximized()

    def _undo(self) -> None:
        """
        Perform undo operation.

        Clears the selection and performs the undo action.

        Returns
        -------
        None
        """
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.context.undo_stack.undo()

    def _redo(self) -> None:
        """
        Perform redo operation.

        Clears the selection and performs the redo action.

        Returns
        -------
        None
        """
        self.ui.input_table.setCurrentIndex(QModelIndex())
        self.ui.input_table.clearSelection()
        self.context.undo_stack.redo()

    # endregion

    # region Main window functions

    def _set_timer_memory_update(self) -> None:
        """
        Update the status bar with the current RAM memory usage.

        This function updates the status bar to show how much RAM memory is
        used at the moment.

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
            mb = Process(getpid()).memory_info().rss / 1024 ** 2
            string_mem = f"{mb:.1f} Mb RAM used"
            usage_string = string_selected_files + string_n + string_mem
            self.ui.memory_usage_label.setText(usage_string)
        except KeyboardInterrupt:
            pass

    def _set_cpu_load(self) -> None:
        """
        Update the CPU load bar with the current CPU usage.

        This function updates the CPU load bar to show the current CPU
        usage percentage.

        Returns
        -------
        None
        """
        cpu_perc = int(cpu_percent())
        self.ui.cpuLoadBar.setValue(cpu_perc)

    def update_icons(self) -> None:
        """
        Update the icons based on the current theme.

        This function updates the main window icons depending on whether the
        light or dark theme is active.

        Returns
        -------
        None
        """
        if "Light" in environ["theme"]:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down_black.svg"))
            self.ui.minimizeAppBtn.setIcon(QIcon("material/resources/source/minus_black.svg"))
            self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/chevron-left_black.svg"))
            self.ui.gt_add_Btn.setIcon(QIcon("material/resources/source/plus_black.svg"))
            self.ui.gt_dlt_Btn.setIcon(QIcon("material/resources/source/minus_black.svg"))
        else:
            self.ui.maximizeRestoreAppBtn.setIcon(
                QIcon("material/resources/source/chevron-down.svg"))
            self.ui.minimizeAppBtn.setIcon(QIcon("material/resources/source/minus.svg"))
            self.ui.leftsideBtn.setIcon(QIcon("material/resources/source/chevron-left.svg"))
            self.ui.gt_add_Btn.setIcon(QIcon("material/resources/source/plus.svg"))
            self.ui.gt_dlt_Btn.setIcon(QIcon("material/resources/source/minus.svg"))
    # endregion
