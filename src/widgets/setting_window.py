# pylint: disable=no-name-in-module, import-error, relative-beyond-top-level, invalid-name

"""
Module for handling settings events and operations in the application.

Classes
-------
SettingWindow
    Class to manage the settings window.
"""

import os
from pathlib import Path
from typing import Generator

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget, QComboBox

from qfluentwidgets import toggle_theme
from src.ui.style import apply_stylesheet
from src.ui.ui_settings import Ui_Form
from ..data.config import get_config
from ..files import save_preferences
from ..ui.style import get_theme_colors


def add_menu_combobox(combobox_ref: QComboBox, background: bool = True) -> None:
    """
    Initial combobox.

    Parameters
    -------
    combobox_ref: QComboBox

    background: bool, default = True
        invert secondary colors
    """
    themes = list_themes(background)
    for i in themes:
        combobox_ref.addItem(i.stem)


def list_themes(background: bool = True) -> Generator[Path, None, None]:
    """
    Returns theme names from folder.

    Parameters
    -------
    background: bool, default = True
        invert secondary colors

    Returns
    -------
    out: Generator[Path, None, None]
        with theme names
    """
    theme_path = get_config()['theme']['path']
    sub_dir = 'background' if background else 'colors'
    themes_dir = theme_path[sub_dir]
    return Path(themes_dir).iterdir()


def update_theme_event(parent, theme_bckgrnd: str = 'Dark', theme_color: str = 'Amber',
                       theme_colors: dict = None) -> None:
    """
    Update theme.

    Parameters
    -------
    parent: MainWindow

    theme_bckgrnd: str, default='Dark'
        theme name for background
    theme_color: str, default='Amber'
        theme name for colors
    theme_colors: dict, default=None
    """
    invert = 'Light' in theme_bckgrnd and 'Dark' not in theme_bckgrnd
    apply_stylesheet(parent, (theme_bckgrnd, theme_color, theme_colors), invert)
    toggle_theme()


class SettingWindow(QWidget):
    """
    Form with program global settings. Open with settingsBtn parent is a MainWindow class
    """

    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Dialog)
        self.ui_form = Ui_Form()
        self.parent = parent
        self.ui_form.setupUi(self)
        self.ui_form.tabWidget.setTabEnabled(1, False)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowTitle('Preferences')
        add_menu_combobox(self.ui_form.ThemeComboBox_Bckgrnd)
        add_menu_combobox(self.ui_form.ThemeComboBox_Color, False)
        self.ui_form.ThemeComboBox_Bckgrnd.setCurrentText(os.environ['theme_bckgrnd'])
        self.ui_form.ThemeComboBox_Color.setCurrentText(os.environ['theme_color'])
        self.ui_form.recent_limit_spinBox.setValue(int(os.environ['recent_limit']))
        self.ui_form.undo_limit_spinBox.setValue(parent.context.undo_stack.undoLimit())
        self.ui_form.undo_limit_spinBox.valueChanged.connect(self.undo_limit_spin_box_changed)
        self.ui_form.axis_font_size_spinBox.setValue(int(os.environ['plot_font_size']))
        self.ui_form.axis_font_size_spinBox.valueChanged.connect(
            self.axis_font_size_spin_box_changed)
        self.ui_form.axis_label_font_size_spinBox.setValue(int(os.environ['axis_label_font_size']))
        self.ui_form.axis_label_font_size_spinBox.valueChanged.connect(
            self.axis_label_font_size_changed)
        self.closeEvent = self.settings_form_close_event
        self.ui_form.ThemeComboBox_Bckgrnd.currentTextChanged.connect(
            self.theme_bckgrnd_text_changed)
        self.ui_form.ThemeComboBox_Color.currentTextChanged.connect(
            self.theme_color_setting_text_changed)

    def settings_form_close_event(self, _) -> None:
        """
        Close event.
        """
        self.write_pref_file()
        self.close()

    def write_pref_file(self) -> None:
        """
        Save parameter's values to environ.
        """
        settings = {'theme_bckgrnd': self.ui_form.ThemeComboBox_Bckgrnd.currentText(),
                    'theme_color': self.ui_form.ThemeComboBox_Color.currentText(),
                    'recent_limit': str(self.ui_form.recent_limit_spinBox.value()),
                    'undo_limit': str(self.parent.context.undo_stack.undoLimit()),
                    'plot_font_size': str(self.ui_form.axis_font_size_spinBox.value()),
                    'axis_label_font_size': str(self.ui_form.axis_label_font_size_spinBox.value())}
        for param, value in settings.items():
            os.environ[param] = value
        save_preferences()

    def theme_bckgrnd_text_changed(self, theme_name: str) -> None:
        """
        Handle theme background color changed event.
        """
        mw = self.parent
        os.environ['theme_bckgrnd'] = theme_name
        update_theme_event(mw, theme_bckgrnd=theme_name, theme_color=os.environ['theme_color'])
        mw.update_icons()
        bckgrnd_color = QColor(os.environ['plotBackground'])
        mw.ui.preproc_plot_widget.setBackground(bckgrnd_color)
        mw.ui.deconv_plot_widget.setBackground(bckgrnd_color)

    def theme_color_setting_text_changed(self, theme_name: str) -> None:
        """
        Handle theme color changed event.
        """
        parent = self.parent
        os.environ['theme_color'] = theme_name
        update_theme_event(parent, theme_bckgrnd=os.environ['bckgrnd_theme'],
                           theme_color=theme_name)
        parent.update_icons()

    def undo_limit_spin_box_changed(self, limit: int) -> None:
        """
        Handle undo limit changed event.
        """
        self.parent.context.undo_stack.setUndoLimit(limit)

    def axis_font_size_spin_box_changed(self, font_size: int) -> None:
        """
        Handle plot_font_size changed event.
        """
        os.environ['plot_font_size'] = str(font_size)
        self.parent.initial_plots_set_fonts()

    def axis_label_font_size_changed(self, font_size: int) -> None:
        """
        Handle axis_label_font_size changed event.
        """
        os.environ['axis_label_font_size'] = str(font_size)
        self.parent.initial_plots_labels()
