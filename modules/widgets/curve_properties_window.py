import os
from BlurWindow.blurWindow import blur
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QMainWindow

from modules.ui import ui_curve_properties


class CurvePropertiesWindow(QWidget):
    sigStyleChanged = pyqtSignal(dict, dict, int)

    def __init__(self, parent: QMainWindow, style=None, idx: int = 0, fill_enabled: bool = True):
        super().__init__(parent, Qt.WindowType.Dialog)
        if style is None:
            style = {}

        self.setWindowOpacity(0.99)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        blur(self.winId(), Dark='Dark' in os.environ['bckgrnd_theme'])
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0)")
        self._style = style
        self.parent = parent
        self._idx = idx
        self.setFixedSize(240, 300)
        self._ui_form = ui_curve_properties.Ui_Dialog()
        self._ui_form.setupUi(self)
        self._init_style_cb()
        self._set_initial_values()
        self._ui_form.line_color_button.clicked.connect(self._set_new_line_color)
        self._ui_form.style_comboBox.currentTextChanged.connect(self._set_new_line_type)
        self._ui_form.width_doubleSpinBox.valueChanged.connect(self._set_new_width)
        self._ui_form.fill_group_box.toggled.connect(self._set_new_fill)
        self._ui_form.use_line_color_checkBox.toggled.connect(self._use_line_color_cb_toggled)
        self._ui_form.fill_color_button.clicked.connect(self._set_new_fill_color)
        self._ui_form.opacity_spinBox.valueChanged.connect(self._set_new_fill_opacity)
        self._ui_form.fill_group_box.setVisible(fill_enabled)
        self.closeEvent = self.at_close

    def _init_style_cb(self) -> None:
        self._ui_form.style_comboBox.addItem('SolidLine')
        self._ui_form.style_comboBox.addItem('DotLine')
        self._ui_form.style_comboBox.addItem('DashLine')
        self._ui_form.style_comboBox.addItem('DashDotLine')
        self._ui_form.style_comboBox.addItem('DashDotDotLine')

    def _set_initial_values(self) -> None:
        if isinstance(self._style, dict):
            self._line_color_button_new_style_sheet(self._style['color'].name())
            self._select_style_cb_item(self._style['style'])
            self._ui_form.width_doubleSpinBox.setValue(self._style['width'])
            self._ui_form.fill_group_box.setChecked(self._style['fill'])
            self._ui_form.use_line_color_checkBox.setChecked(self._style['use_line_color'])
            self._fill_color_button_new_style_sheet(self._style['fill_color'].name())
            self._ui_form.opacity_spinBox.setValue(int(self._style['fill_opacity'] * 100))

    def _select_style_cb_item(self, pen_style: int) -> None:
        match pen_style:
            case Qt.PenStyle.SolidLine:
                s = 'SolidLine'
            case Qt.PenStyle.DashLine:
                s = 'DashLine'
            case Qt.PenStyle.DotLine:
                s = 'DotLine'
            case Qt.PenStyle.DashDotLine:
                s = 'DashDotLine'
            case Qt.PenStyle.DashDotDotLine:
                s = 'DashDotDotLine'
            case _:
                s = 'SolidLine'
        self._ui_form.style_comboBox.setCurrentText(s)

    def _set_new_line_color(self):
        init_color = self._style['color']
        color_dialog = self.parent.color_dialog(init_color)
        color = color_dialog.getColor(init_color)
        if color.isValid():
            self._line_color_button_new_style_sheet(color.name())
            old_style = self._style.copy()
            self._style['color'] = color
            self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _line_color_button_new_style_sheet(self, hex_color: str) -> None:
        self._ui_form.line_color_button.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def _fill_color_button_new_style_sheet(self, hex_color: str) -> None:
        self._ui_form.fill_color_button.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def _set_new_line_type(self, current_text: str) -> None:
        line_type = Qt.PenStyle.SolidLine
        match current_text:
            case 'DotLine':
                line_type = Qt.PenStyle.DotLine
            case 'DashLine':
                line_type = Qt.PenStyle.DashLine
            case 'DashDotLine':
                line_type = Qt.PenStyle.DashDotLine
            case 'DashDotDotLine':
                line_type = Qt.PenStyle.DashDotDotLine
        old_style = self._style.copy()
        self._style['style'] = line_type
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_width(self, f: float) -> None:
        old_style = self._style.copy()
        self._style['width'] = f
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill(self, b: bool) -> None:
        old_style = self._style.copy()
        self._style['fill'] = b
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _use_line_color_cb_toggled(self, b: bool) -> None:
        old_style = self._style.copy()
        self._style['use_line_color'] = b
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill_color(self):
        init_color = self._style['fill_color']
        color_dialog = self.parent.color_dialog(init_color)
        color = color_dialog.getColor(init_color)
        if color.isValid():
            self._fill_color_button_new_style_sheet(color.name())
            old_style = self._style.copy()
            self._style['fill_color'] = color
            self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill_opacity(self, i: int) -> None:
        old_style = self._style.copy()
        self._style['fill_opacity'] = float(i / 100)
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def accept(self) -> None:
        self.close()

    def reject(self) -> None:
        self.close()

    def idx(self) -> int:
        return self._idx

    def at_close(self, _):
        self.parent.update_all_plots()
