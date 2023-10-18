# coding:utf-8
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QWidget

from ...common.style_sheet import is_dark_theme


class HorizontalSeparator(QWidget):
    """ Horizontal separator """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(3)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        if is_dark_theme():
            painter.setPen(QColor(255, 255, 255, 51))
        else:
            painter.setPen(QColor(0, 0, 0, 22))

        painter.drawLine(0, 1, self.width(), 1)


class VerticalSeparator(QWidget):
    """ Vertical separator """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedWidth(3)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        if is_dark_theme():
            painter.setPen(QColor(255, 255, 255, 51))
        else:
            painter.setPen(QColor(0, 0, 0, 22))

        painter.drawLine(1, 0, 1, self.height())