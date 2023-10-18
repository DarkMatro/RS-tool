from qtpy.QtWidgets import QSplashScreen, QApplication
from qtpy.QtCore import Qt
from BlurWindow.blurWindow import blur


class SplashScreen(QSplashScreen):
    """
    QSplashScreen with blur and transparency
    """
    def __init__(self, *__args):
        super(SplashScreen, self).__init__(*__args)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._move_to_center()
        hWnd = self.winId()
        blur(hWnd)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0)")

    def _move_to_center(self):
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
