import sys
import warnings
from pathlib import Path
from ctypes import windll
from asyncio import set_event_loop
from os import environ
from traceback import format_exception
from asyncqtpy import QEventLoop
from qtpy.QtGui import QPixmap, QIcon
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt
from shap import initjs
from matplotlib import pyplot as plt
from seaborn import set_style as sns_set_style
from modules import show_error_msg
from qfluentwidgets import MessageBox

from modules.widgets import SplashScreen
from modules.work_with_files.recent_files import recent_files_exists_and_empty
from modules.mutual_functions.static_functions import action_help
from modules.init import apply_stylesheet, splash_show_message
from modules.main_window import MainWindow
from modules.work_with_files.preferences_file import read_preferences, check_preferences_file


def start_program() -> None:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings('once')
    my_app_id = 'rs.tool'  # arbitrary string
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)  # set windows taskbar icon
    environ['PYTHONASYNCIODEBUG'] = '1'
    environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    environ['CANCEL'] = '0'
    environ['logo'] = 'images/logo.ico'
    environ['OPENBLAS_NUM_THREADS'] = '1'

    check_preferences_file()
    read_prefs = read_preferences()
    theme_bckgrnd = read_prefs[0]
    theme_color = read_prefs[1]
    environ['bckgrnd_theme'] = theme_bckgrnd
    environ['theme'] = theme_color
    invert = 'Light' in theme_bckgrnd and 'Dark' not in theme_bckgrnd
    sys.excepthook = _except_hook
    app = QApplication(sys.argv)
    splash_img = QPixmap('images/splash.tif')
    splash = SplashScreen(splash_img)
    font = splash.font()
    font.setPixelSize(14)
    font.setFamily('Nasalization')
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
    initjs()
    plt.style.use(['dark_background'])
    plt.set_loglevel("info")
    sns_set_style(rc={"font.family": 'DejaVu Sans'})
    frame = MainWindow(loop, (theme_bckgrnd, theme_color, None), read_prefs, splash)
    splash_show_message(splash, 'Apply stylesheets..')
    # Set theme on initialization
    apply_stylesheet(app, (theme_bckgrnd, theme_color, frame.theme_colors), invert)
    frame.showMaximized()
    splash_show_message(splash, 'Initializing finished.')
    splash.finish(frame)
    if Path('examples/demo_project.zip').exists() and recent_files_exists_and_empty() \
            and MessageBox('Open demo project?', '', frame, {'Yes', 'Cancel'}).exec() == 1:
        frame.open_demo_project()
        action_help()
    with loop:
        loop.run_forever()


def _except_hook(exc_type, exc_value, exc_tb, **kwargs):
    tb = "".join(format_exception(exc_type, exc_value, exc_tb))
    if exc_value.args[0][0: 19] == 'invalid result from':
        return
    show_error_msg(exc_type, exc_value, tb, **kwargs)
