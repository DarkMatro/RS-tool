import ctypes
import sys
from asyncio import set_event_loop
from multiprocessing import freeze_support
from os import environ
from traceback import format_exception

import pyperclip
from asyncqtpy import QEventLoop
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap, QFont, QIcon
from qtpy.QtWidgets import QMessageBox, QApplication, QSplashScreen

from modules.init import apply_stylesheet, splash_show_message
from modules.main_window import MainWindow
from modules.static_functions import check_preferences_file, read_preferences


def start_program() -> None:
    my_app_id = 'rs.tool'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)
    environ['PYTHONASYNCIODEBUG'] = '1'
    environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    environ['CANCEL'] = '0'
    environ['splash_color'] = 'white'
    environ['logo'] = 'images/logo.ico'
    freeze_support()
    splash_png = 'images/splash_sized.png'
    check_preferences_file()
    read_prefs = read_preferences()
    theme_bckgrnd = read_prefs[0]
    theme_color = read_prefs[1]
    invert = 'Light' in theme_bckgrnd and 'Dark' not in theme_bckgrnd
    if invert:
        environ['splash_color'] = 'black'
        splash_png = 'images/splash_white_sized.png'
        environ['logo'] = 'images/logo_white.ico'

    sys.excepthook = _except_hook
    app = QApplication(sys.argv)
    splash_img = QPixmap(splash_png)
    splash = QSplashScreen(splash_img, Qt.WindowType.WindowStaysOnTopHint)
    font = splash.font()
    font.setPixelSize(12)
    font.setWeight(QFont.Weight.Normal)
    font.setFamily('AbletonSans, Roboto')
    splash.setFont(font)
    splash.show()
    splash_show_message(splash, 'Starting QApplication.')
    loop = QEventLoop(app)
    set_event_loop(loop)
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Round)
    app.processEvents()
    app.setQuitOnLastWindowClosed(False)
    win_icon = QIcon(environ['logo'])
    app.setWindowIcon(win_icon)

    frame = MainWindow(loop, (theme_bckgrnd, theme_color, None), read_prefs, splash)
    splash_show_message(splash, 'Initializing CSS')
    # Set theme on initialization
    apply_stylesheet(app, (theme_bckgrnd, theme_color, frame.theme_colors), invert)
    frame.showMaximized()
    splash_show_message(splash, 'Initializing finished')
    splash.finish(frame)
    with loop:
        loop.run_forever()


def _except_hook(exc_type, exc_value, exc_tb):
    tb = "".join(format_exception(exc_type, exc_value, exc_tb))
    if exc_value.args[0][0: 19] == 'invalid result from':
        return
    show_error(exc_type, exc_value, tb)


def show_error(exc_type, exc_value, exc_tb):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText(str(exc_value))
    msg.setWindowTitle(str(exc_type))
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.setInformativeText(exc_tb)
    pyperclip.copy(exc_tb)
    msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    msg.exec()
