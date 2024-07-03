import os
from qfluentwidgets import toggle_theme
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget
from src.ui.style import apply_stylesheet
from ..ui.style import get_theme_colors
from src.ui.ui_settings import Ui_Form
from ..data.config import get_config


def add_menu_combobox(combobox_ref, background: bool = True):
    themes = list_themes(background)
    for i in themes:
        combobox_ref.addItem(i.stem)


def list_themes(background: bool = True):
    """"""
    theme_path = get_config()['theme']['path']
    sub_dir = 'background' if background else 'colors'
    themes_dir = theme_path[sub_dir]
    return Path(themes_dir).iterdir()


def update_theme_event(parent, theme_bckgrnd: str = 'Dark', theme_color: str = 'Amber',
                       theme_colors: dict = None) -> None:
    invert = 'Light' in theme_bckgrnd and 'Dark' not in theme_bckgrnd
    apply_stylesheet(parent, (theme_bckgrnd, theme_color, theme_colors), invert)
    toggle_theme()


class SettingWindow(QWidget):
    """form with program global settings. Open with settingsBtn
    parent is a MainWindow class
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
        self.ui_form.ThemeComboBox_Bckgrnd.setCurrentText(os.environ['theme'])
        self.ui_form.ThemeComboBox_Color.setCurrentText(os.environ['theme_color'])
        self.ui_form.recent_limit_spinBox.setValue(int(os.environ['recent_limit']))
        self.ui_form.recent_limit_spinBox.valueChanged.connect(self.recent_limit_spin_box_changed)
        self.ui_form.undo_limit_spinBox.setValue(parent.undoStack.undoLimit())
        self.ui_form.undo_limit_spinBox.valueChanged.connect(self.undo_limit_spin_box_changed)
        self.ui_form.axis_font_size_spinBox.setValue(int(os.environ['plot_font_size']))
        self.ui_form.axis_font_size_spinBox.valueChanged.connect(self.axis_font_size_spin_box_changed)
        self.ui_form.axis_label_font_size_spinBox.setValue(int(os.environ['axis_label_font_size']))
        self.ui_form.axis_label_font_size_spinBox.valueChanged.connect(self.axis_label_font_size_changed)

        self.ui_form.target_peak_parameter_combo_box.addItems(['Amplitude', 'Area'])
        # self.ui_form.target_peak_parameter_combo_box.setCurrentText(parent.prefs['target_peak_parameter'])

        self.closeEvent = self.settings_form_close_event
        self.ui_form.ThemeComboBox_Bckgrnd.currentTextChanged.connect(self.theme_bckgrnd_text_changed)
        self.ui_form.ThemeComboBox_Color.currentTextChanged.connect(self.theme_color_setting_text_changed)
        self.ui_form.target_peak_parameter_combo_box.currentTextChanged.connect(self.target_peak_parameter_changed)

    def settings_form_close_event(self, _) -> None:
        self.write_pref_file()
        self.close()

    def write_pref_file(self) -> None:
        settings = {'theme': self.ui_form.ThemeComboBox_Bckgrnd.currentText(),
                    'theme_color': self.ui_form.ThemeComboBox_Color.currentText(),
                    'recent_limit': str(self.ui_form.recent_limit_spinBox.value()),
                    'undo_limit': str(self.parent.undoStack.undoLimit()),
                    'plot_font_size': str(self.ui_form.axis_font_size_spinBox.value()),
                    'axis_label_font_size': str(self.ui_form.axis_label_font_size_spinBox.value()),
                    'target_peak_parameter': self.ui_form.target_peak_parameter_combo_box.currentText()}
        for param, value in settings.items():
            os.environ[param] = value

    def theme_bckgrnd_text_changed(self, theme_name: str) -> None:
        parent = self.parent
        os.environ['bckgrnd_theme'] = theme_name
        update_theme_event(parent, theme_bckgrnd=theme_name, theme_color=os.environ['theme_color'])
        os.environ['theme'] = theme_name
        parent.update_icons()
        parent.theme_colors = get_theme_colors(theme_name, os.environ['theme_color'])
        bckgrnd_color = QColor(os.environ['plotBackground'])
        parent.ui.input_plot_widget.setBackground(bckgrnd_color)
        parent.ui.converted_cm_plot_widget.setBackground(bckgrnd_color)
        parent.ui.cut_cm_plot_widget.setBackground(bckgrnd_color)
        parent.ui.normalize_plot_widget.setBackground(bckgrnd_color)
        parent.ui.smooth_plot_widget.setBackground(bckgrnd_color)
        parent.ui.baseline_plot_widget.setBackground(bckgrnd_color)
        parent.ui.average_plot_widget.setBackground(bckgrnd_color)
        parent.ui.deconv_plot_widget.setBackground(bckgrnd_color)

    def theme_color_setting_text_changed(self, theme_name: str) -> None:
        parent = self.parent
        os.environ['theme'] = theme_name
        update_theme_event(parent, theme_bckgrnd=os.environ['theme'], theme_color=theme_name)
        os.environ['theme_color'] = theme_name
        parent.update_icons()
        parent.theme[2] = get_theme_colors(os.environ['theme'], theme_name)

    def target_peak_parameter_changed(self, target_parameter: str) -> None:
        os.environ['target_peak_parameter'] = target_parameter

    def recent_limit_spin_box_changed(self, recent_limit: int) -> None:
        os.environ['recent_limit'] = recent_limit

    def undo_limit_spin_box_changed(self, limit: int) -> None:
        self.parent.undoStack.setUndoLimit(limit)

    def axis_font_size_spin_box_changed(self, font_size: int) -> None:
        parent = self.parent
        os.environ['plot_font_size'] = font_size
        parent.initial_plots_set_fonts()

    def axis_label_font_size_changed(self, font_size: int) -> None:
        parent = self.parent
        os.environ['axis_label_font_size'] = font_size
        parent.initial_plots_set_labels_font()
