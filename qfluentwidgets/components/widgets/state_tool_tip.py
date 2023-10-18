# coding:utf-8
from PyQt5.QtCore import QPropertyAnimation, Qt, QTimer, pyqtSignal, QPoint, QRectF
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QLabel, QWidget, QToolButton, QGraphicsOpacityEffect

from ...common import FluentStyleSheet, is_dark_theme, Theme
from ...common.icon import FluentIcon as FIF
from ..dialog_box.dialog import MessageBox
from ..widgets.progress_ring import ProgressRing


class StateCloseButton(QToolButton):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.isPressed = False
        self.isEnter = False

    def enterEvent(self, e):
        self.isEnter = True
        self.update()

    def leaveEvent(self, e):
        self.isEnter = False
        self.isPressed = False
        self.update()

    def mousePressEvent(self, e):
        self.isPressed = True
        self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self.isPressed = False
        self.update()
        super().mouseReleaseEvent(e)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if self.isPressed:
            painter.setOpacity(0.6)
        elif self.isEnter:
            painter.setOpacity(0.8)

        theme = Theme.DARK if not is_dark_theme() else Theme.LIGHT
        FIF.CLOSE.render(painter, self.rect(), theme)


class StateToolTip(QWidget):
    """ State tooltip """

    closedSignal = pyqtSignal()

    def __init__(self, title, content, parent=None, maximum: int = 0):
        """
        Parameters
        ----------
        title: str
            title of tooltip

        content: str
            content of tooltip

        parant:
            parent window
        """
        super().__init__(parent)

        self.title = title
        self.content = content
        self.maximum = maximum

        self.titleLabel = QLabel(self.title, self)
        self.contentLabel = QLabel(self.content, self)
        self.rotateTimer = QTimer(self)

        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.animation = QPropertyAnimation(self.opacityEffect, b"opacity")
        self.closeButton = StateCloseButton(self)

        self.isDone = False
        self.rotateAngle = 0
        self.deltaAngle = 20
        self.cancelled = False
        self.value = 0
        self.progress_ring = None
        if self.maximum > 0:
            self.progress_ring = ProgressRing(self)
            self.progress_ring.setMaximum(self.maximum)
            self.progress_ring.setStrokeWidth(4)
            self.progress_ring.setTextVisible(True)
            self.progress_ring.setStyleSheet('background-color: black; color: black;')
            self.progress_ring.setColor('#000000')
            self.progress_ring.set_text_theme_invert(True)
        self.__initWidget()


    def __initWidget(self):
        """ initialize widgets """
        self.setAttribute(Qt.WA_StyledBackground)
        self.setGraphicsEffect(self.opacityEffect)
        self.opacityEffect.setOpacity(1)
        self.rotateTimer.setInterval(50)
        self.contentLabel.setMinimumWidth(200)

        # connect signal to slot
        self.closeButton.clicked.connect(self.__onCloseButtonClicked)
        self.rotateTimer.timeout.connect(self.__rotateTimerFlowSlot)

        self.__setQss()
        self.__initLayout()
        if self.progress_ring is None:
            self.rotateTimer.start()

    def __initLayout(self):
        """ initialize layout """
        self.setFixedSize(max(self.titleLabel.width(),
                          self.contentLabel.width()) + 56, 51)
        self.titleLabel.move(32, 9)
        self.contentLabel.move(12, 27)
        self.closeButton.move(self.width() - 24, 19)
        if self.progress_ring is not None:
            self.progress_ring.setFixedWidth(40)
            self.progress_ring.setFixedHeight(40)
            self.progress_ring.move(self.width() - 80, 6)
            self.progress_ring.adjustSize()

    def __setQss(self):
        """ set style sheet """
        self.titleLabel.setObjectName("titleLabel")
        self.contentLabel.setObjectName("contentLabel")

        FluentStyleSheet.STATE_TOOL_TIP.apply(self)

        self.titleLabel.adjustSize()
        self.contentLabel.adjustSize()

    def setValue(self, v: int) -> None:
        if self.progress_ring:
            self.progress_ring.setValue(v)

    def setTitle(self, title: str):
        """ set the title of tooltip """
        self.title = title
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def setContent(self, content: str):
        """ set the content of tooltip """
        self.content = content
        self.contentLabel.setText(content)

        # adjustSize() will mask spinner get stuck
        self.contentLabel.adjustSize()

    def setState(self, isDone=False):
        """ set the state of tooltip """
        self.isDone = isDone
        self.update()
        if isDone:
            QTimer.singleShot(1000, self.__fadeOut)

    def __onCloseButtonClicked(self):
        """ close button clicked slot """
        msg = MessageBox('Stop process?', 'Note: ' + '\n' + 'All results will be lost. 🫣', self.parent(),
                         {'Yes', 'Cancel'})
        result = msg.exec()
        if not result == 1:
            return
        try:
            self.cancelled = True
            self.closedSignal.emit()
            QTimer.singleShot(1000, self.__fadeOut)
        except RuntimeError:
            pass

    def wasCanceled(self) -> bool:
        return self.cancelled

    def __fadeOut(self):
        """ fade out """
        self.rotateTimer.stop()
        self.animation.setDuration(200)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(self.deleteLater)
        self.animation.start()

    def __rotateTimerFlowSlot(self):
        """ rotate timer time out slot """
        self.rotateAngle = (self.rotateAngle + self.deltaAngle) % 360
        self.update()

    def getSuitablePos(self):
        """ get suitable position in main window """
        for i in range(10):
            dy = i*(self.height() + 16)
            pos = QPoint(self.parent().width() - self.width() - 24, 50+dy)
            widget = self.parent().childAt(pos + QPoint(2, 2))
            if isinstance(widget, StateToolTip):
                pos += QPoint(0, self.height() + 16)
            else:
                break

        return pos

    def paintEvent(self, e):
        """ paint state tooltip """
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        theme = Theme.DARK if not is_dark_theme() else Theme.LIGHT

        if not self.isDone and self.maximum == 0:
            painter.translate(19, 18)
            painter.rotate(self.rotateAngle)
            FIF.SYNC.render(painter, QRectF(-8, -8, 16, 16), theme)
        elif self.cancelled:
            FIF.REMOVE_FROM.render(painter, QRectF(11, 10, 16, 16), theme)
        elif self.isDone:
            FIF.COMPLETED.render(painter, QRectF(11, 10, 16, 16), theme)

