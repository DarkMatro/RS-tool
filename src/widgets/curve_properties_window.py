# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Curve Properties Window Module
==============================

This module provides a class for creating a window to modify the properties of a curve
in a plot. The window allows users to set various attributes of the curve, such as
line color, line style, line width, fill color, and fill opacity. Changes to these
properties are signaled to the parent window, which can update the plot accordingly.

Classes
-------
CurvePropertiesWindow(QWidget)
    A window to modify the properties of a curve in a plot.

"""

from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QMainWindow

from src.ui import ui_curve_properties
from src.ui.style import color_dialog


class CurvePropertiesWindow(QWidget):
    """
    A window to modify the properties of a curve in a plot.

    Attributes
    ----------
    sigStyleChanged : pyqtSignal
        Signal emitted when the style of the curve changes. It passes the new
        style, old style, and the index of the curve.
    """
    sigStyleChanged = pyqtSignal(dict, dict, int)

    def __init__(self, parent: QMainWindow, style=None, idx: int = 0, fill_enabled: bool = True):
        """
        Initializes the CurvePropertiesWindow with the given parent, style, index,
        and fill enabled flag.

        Parameters
        ----------
        parent : QMainWindow
            The parent main window.
        style : dict, optional
            Initial style settings for the curve (default is None).
        idx : int, optional
            Index of the curve (default is 0).
        fill_enabled : bool, optional
            Flag to enable or disable fill settings (default is True).
        """
        super().__init__(parent, Qt.WindowType.Dialog)
        if style is None:
            style = {}
        self._style = style
        self.parent = parent
        self._idx = idx
        self.setFixedSize(240, 340)
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

    def _init_style_cb(self) -> None:
        """
        Initializes the combo box with line style options.
        """
        self._ui_form.style_comboBox.addItem('SolidLine')
        self._ui_form.style_comboBox.addItem('DotLine')
        self._ui_form.style_comboBox.addItem('DashLine')
        self._ui_form.style_comboBox.addItem('DashDotLine')
        self._ui_form.style_comboBox.addItem('DashDotDotLine')

    def _set_initial_values(self) -> None:
        """
        Sets the initial values of the curve properties based on the provided style.
        """
        if isinstance(self._style, dict):
            self._line_color_button_new_style_sheet(self._style['color'].name())
            self._select_style_cb_item(self._style['style'])
            self._ui_form.width_doubleSpinBox.setValue(self._style['width'])
            self._ui_form.fill_group_box.setChecked(self._style['fill'])
            self._ui_form.use_line_color_checkBox.setChecked(self._style['use_line_color'])
            self._fill_color_button_new_style_sheet(self._style['fill_color'].name())
            self._ui_form.opacity_spinBox.setValue(int(self._style['fill_opacity'] * 100))

    def _select_style_cb_item(self, pen_style: int) -> None:
        """
        Selects the appropriate combo box item based on the given pen style.

        Parameters
        ----------
        pen_style : int
            The pen style to select in the combo box.
        """
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
        """
        Opens a color dialog to select a new line color and updates the style.
        """
        init_color = self._style['color']
        dialog = color_dialog(init_color)
        color = dialog.getColor(init_color)
        if color.isValid():
            self._line_color_button_new_style_sheet(color.name())
            old_style = self._style.copy()
            self._style['color'] = color
            self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _line_color_button_new_style_sheet(self, hex_color: str) -> None:
        """
        Updates the stylesheet of the line color button with the given hex color.

        Parameters
        ----------
        hex_color : str
            The hex color code to set as the button's background color.
        """
        self._ui_form.line_color_button.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def _fill_color_button_new_style_sheet(self, hex_color: str) -> None:
        """
        Updates the stylesheet of the fill color button with the given hex color.

        Parameters
        ----------
        hex_color : str
            The hex color code to set as the button's background color.
        """
        self._ui_form.fill_color_button.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def _set_new_line_type(self, current_text: str) -> None:
        """
        Updates the line type based on the selected combo box item.

        Parameters
        ----------
        current_text : str
            The text of the selected combo box item representing the line type.
        """
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
        """
        Updates the width of the line.

        Parameters
        ----------
        f : float
            The new width of the line.
        """
        old_style = self._style.copy()
        self._style['width'] = f
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill(self, b: bool) -> None:
        """
        Updates the fill property of the curve.

        Parameters
        ----------
        b : bool
            The new fill property of the curve.
        """
        old_style = self._style.copy()
        self._style['fill'] = b
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _use_line_color_cb_toggled(self, b: bool) -> None:
        """
        Toggles the use of the line color for filling the curve.

        Parameters
        ----------
        b : bool
            The new state of the use line color checkbox.
        """
        old_style = self._style.copy()
        self._style['use_line_color'] = b
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill_color(self):
        """
        Opens a color dialog to select a new fill color and updates the style.
        """
        init_color = self._style['fill_color']
        dialog = color_dialog(init_color)
        color = dialog.getColor(init_color)
        if color.isValid():
            self._fill_color_button_new_style_sheet(color.name())
            old_style = self._style.copy()
            self._style['fill_color'] = color
            self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill_opacity(self, i: int) -> None:
        """
        Updates the fill opacity of the curve.

        Parameters
        ----------
        i : int
            The new fill opacity value as an integer percentage.
        """
        old_style = self._style.copy()
        self._style['fill_opacity'] = float(i / 100)
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def accept(self) -> None:
        """
        Closes the window and accepts the changes.
        """
        self.close()

    def reject(self) -> None:
        """
        Closes the window and rejects the changes.
        """
        self.close()

    def idx(self) -> int:
        """
        Returns the index of the curve.

        Returns
        -------
        int
            The index of the curve.
        """
        return self._idx
