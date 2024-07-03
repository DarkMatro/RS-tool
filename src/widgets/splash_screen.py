"""
Custom SplashScreen widget

classes:
    * SplashScreen
"""

from qtpy.QtWidgets import QSplashScreen, QDesktopWidget
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from BlurWindow.blurWindow import blur


class SplashScreen(QSplashScreen):
    """
    QSplashScreen with blur and transparency
    """

    def __init__(self, desktop: QDesktopWidget, config: dict, *__args):
        super().__init__(*__args)
        self.desktop = desktop
        self.setPixmap(QPixmap(config['path']))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._move_to_center()
        blur(self.winId())
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0)")
        if 'font_size' in config and 'font_family' in config:
            self._set_font(config['font_size'], config['font_family'])

    def _move_to_center(self):
        """
        Move splash screen to center of current desktop where cursor positioned.
        """
        frame_gm = self.frameGeometry()
        center_point = self.desktop.screenGeometry().center()
        frame_gm.moveCenter(center_point)
        self.move(frame_gm.topLeft())

    def _set_font(self, font_size: int, font_family: str) -> None:
        """
        Setup font for messages inside splash screen.

        Parameters
        -------
        font_size: int
        font_family: str
        """
        font = self.font()
        font.setPixelSize(font_size)
        font.setFamily(font_family)
        self.setFont(font)

    def show_message(self, text: str) -> None:
        """
        Show white text at bottom of splash screen.

        Parameters
        -------
        text: str
            Message to show
        """
        self.showMessage(text,
                         alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                         color=Qt.GlobalColor.white)
