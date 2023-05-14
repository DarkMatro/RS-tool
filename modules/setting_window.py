from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget

from modules.init import get_theme
from modules.ui_settings import Ui_Form


class SettingWindow(QWidget):
    """form with program global settings. Open with settingsBtn
    parent is a MainWindow class
    """

    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Dialog)
        self.parent = parent
        self.ui_form = Ui_Form()
        self.ui_form.setupUi(self)
        self.ui_form.tabWidget.setTabEnabled(1, False)
        self.parent.add_menu_combobox(self.ui_form.ThemeComboBox_Bckgrnd)
        self.parent.add_menu_combobox(self.ui_form.ThemeComboBox_Color, False)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowTitle('Preferences')
        self.ui_form.ThemeComboBox_Bckgrnd.currentTextChanged.connect(self.theme_bckgrnd_text_changed)
        self.ui_form.ThemeComboBox_Color.currentTextChanged.connect(self.theme_color_setting_text_changed)
        self.ui_form.ThemeComboBox_Bckgrnd.setCurrentText(parent.theme_bckgrnd)
        self.ui_form.ThemeComboBox_Color.setCurrentText(parent.theme_color)
        self.ui_form.recent_limit_spinBox.setValue(int(parent.recent_limit))
        self.ui_form.recent_limit_spinBox.valueChanged.connect(self.recent_limit_spin_box_changed)
        self.ui_form.undo_limit_spinBox.setValue(int(parent.undoStack.undoLimit()))
        self.ui_form.undo_limit_spinBox.valueChanged.connect(self.undo_limit_spin_box_changed)
        self.ui_form.auto_save_spinBox.setValue(parent.auto_save_minutes)
        self.ui_form.auto_save_spinBox.valueChanged.connect(self.auto_save_spin_box_changed)
        self.ui_form.axis_font_size_spinBox.setValue(parent.plot_font_size)
        self.ui_form.axis_font_size_spinBox.valueChanged.connect(self.axis_font_size_spin_box_changed)
        self.ui_form.axis_label_font_size_spinBox.setValue(int(parent.axis_label_font_size))
        self.ui_form.axis_label_font_size_spinBox.valueChanged.connect(self.axis_label_font_size_changed)
        self.ui_form.auto_save_spinBox.setVisible(False)
        self.ui_form.auto_save_label.setVisible(False)
        self.closeEvent = self.settings_form_close_event

    def settings_form_close_event(self, _) -> None:
        theme_bckgrnd_ch = str(self.ui_form.ThemeComboBox_Bckgrnd.currentText())
        theme_color_ch = str(self.ui_form.ThemeComboBox_Color.currentText())
        recent_limit_ch = str(self.ui_form.recent_limit_spinBox.value())
        undo_stack_limit_ch = str(self.parent.undoStack.undoLimit())
        auto_save_minutes_ch = str(self.parent.auto_save_minutes)
        plot_font_size_ch = str(self.ui_form.axis_font_size_spinBox.value())
        axis_label_font_size_ch = str(self.ui_form.axis_label_font_size_spinBox.value())
        f = Path('preferences.txt').open('w+')
        f.write(
            theme_bckgrnd_ch + '\n' + theme_color_ch + '\n' + recent_limit_ch + '\n'
            + undo_stack_limit_ch + '\n' + auto_save_minutes_ch + '\n' + plot_font_size_ch + '\n'
            + axis_label_font_size_ch + '\n')
        f.close()
        self.close()

    def theme_bckgrnd_text_changed(self, theme_name: str) -> None:
        parent = self.parent
        parent.update_theme_event(parent, theme_bckgrnd=theme_name, theme_color=parent.theme_color)
        parent.theme_bckgrnd = theme_name
        parent.update_icons()
        parent.theme_colors = get_theme((theme_name, parent.theme_color, None))
        bckgrnd_color = QColor(parent.theme_colors['plotBackground'])
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
        parent.update_theme_event(parent, theme_bckgrnd=parent.theme_bckgrnd, theme_color=theme_name)
        parent.theme_color = theme_name
        parent.update_icons()
        parent.theme_colors = get_theme((parent.theme_bckgrnd, theme_name, None))

    def recent_limit_spin_box_changed(self, recent_limit: int) -> None:
        self.parent.recent_limit = recent_limit

    def undo_limit_spin_box_changed(self, limit: int) -> None:
        self.parent.undoStack.setUndoLimit(limit)

    def auto_save_spin_box_changed(self, auto_save_minutes: int) -> None:
        parent = self.parent
        parent.auto_save_minutes = auto_save_minutes
        parent.auto_save_timer.stop()
        parent.auto_save_timer.start(1000 * 60 * auto_save_minutes)

    def axis_font_size_spin_box_changed(self, font_size: int) -> None:
        parent = self.parent
        parent.plot_font_size = font_size
        parent.initial_plots_set_fonts()

    def axis_label_font_size_changed(self, font_size: int) -> None:
        parent = self.parent
        parent.axis_label_font_size = str(font_size)
        parent.initial_plots_set_labels_font()
