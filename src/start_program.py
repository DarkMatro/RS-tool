"""
Start program.

functions:
    * start_program
"""

import sys
import warnings
from asyncio import set_event_loop
from ctypes import windll
from logging import basicConfig, info
from os import environ, getenv
from pathlib import Path
from traceback import format_exception
from types import TracebackType

from asyncqtpy import QEventLoop
from matplotlib import pyplot as plt
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication
from seaborn import set_style as sns_set_style

from qfluentwidgets import MessageBox
from src.mutual_functions.static_functions import show_error_msg
from src.data.config import get_config
from src.files.help import action_help
from src.files.preferences_file import read_preferences
from src.pages.main_window import MainWindow
from src.ui.style import apply_stylesheet, get_theme_colors
from src.widgets import SplashScreen
import dill, multiprocessing


def start_program() -> None:
    """
    Setup environ and many others, create app and main window widget.
    """
    # dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
    # multiprocessing.reduction.ForkingPickler = dill.Pickler
    # multiprocessing.reduction.dump = dill.dump
    # multiprocessing.queues._ForkingPickler = dill.Pickler
    cfg = get_config()
    cfg_env = cfg["environ"]
    read_preferences()
    bckgrnd_theme = environ["theme"]
    theme_color = environ["theme_color"]
    invert = "Light" in bckgrnd_theme and "Dark" not in bckgrnd_theme
    theme_colors = get_theme_colors(bckgrnd_theme, theme_color, invert)

    _setup()
    _set_environ(bckgrnd_theme, theme_color, cfg_env)
    app = _get_app(cfg["logo"]["path"])
    # Splash start.
    splash_cfg = cfg["splash"]
    splash = SplashScreen(app.desktop(), splash_cfg)
    splash.show()
    splash.show_message("Starting QApplication.")

    loop = QEventLoop(app)
    set_event_loop(loop)

    # Create main window.
    splash.show_message("Initializing main window..")
    frame = MainWindow(loop)

    splash.show_message("Start logging...")
    _check_rs_tool_folder()
    _start_logging()
    splash.show_message("Apply stylesheets..")

    # Set theme on initialization
    apply_stylesheet(app, (bckgrnd_theme, theme_color, theme_colors), invert)

    frame.showMaximized()
    splash.show_message("Initializing finished.")
    splash.finish(frame)
    # Open help manual for first launch.
    if (
        Path("examples/demo_project.zip").exists()
        and len(frame.project.recent_list) == 0
        and MessageBox("Open demo project?", "", frame, {"Yes", "Cancel"}).exec() == 1
    ):
        frame.project.open_demo_project()
        action_help()
    with loop:
        loop.run_forever()


def _get_app(logo_path: str) -> QApplication:
    """
    Catch exception and show error message.

    Parameters
    -------
    logo_path: str

    Returns
    -------
    app: QApplication
    """
    app = QApplication(sys.argv)
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Round)
    app.processEvents()
    app.setQuitOnLastWindowClosed(False)
    win_icon = QIcon(logo_path)
    app.setWindowIcon(win_icon)
    return app


def _set_environ(bckgrnd_theme: str, theme_color: str, cfg_env: dict) -> None:
    """
    Setup environ for further usage.

    Parameters
    -------
    bckgrnd_theme: str
        from preferences
    theme_color: str
        from preferences
    cfg_env: dict
        params for environ
    """
    for k, v in cfg_env.items():
        environ[k] = v
    environ["bckgrnd_theme"] = bckgrnd_theme
    environ["theme"] = theme_color


def _setup() -> None:
    """
    Code for setup before app start.
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings("ignore")
    my_app_id = "rs.tool"  # arbitrary string
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        my_app_id
    )  # set windows taskbar icon
    plt.style.use(["dark_background"])
    plt.set_loglevel("info")
    sns_set_style(rc={"font.family": "DejaVu Sans"})
    sys.excepthook = _except_hook
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)


def _start_logging() -> None:
    """
    Start log file in appdata.
    """
    cfg = get_config()["logging"]
    path = getenv("APPDATA") + cfg["folder"] + cfg["filename"]
    basicConfig(
        filename=path, filemode="w", format="%(asctime)s %(levelname)s %(message)s"
    )
    info("Logging started.")


def _check_rs_tool_folder() -> None:
    """
    Before start logging check folder exists.
    """
    path = getenv("APPDATA") + get_config()["logging"]["folder"]
    if not Path(path).exists():
        Path(path).mkdir()


def _except_hook(
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    exc_tb: TracebackType | None,
    **kwargs
) -> None:
    """
    Catch exception and show error message.

    Parameters
    -------
    exc_type: type[BaseException] | None
    exc_value: BaseException | None
    exc_tb: TracebackType | None
    """
    tb = "".join(format_exception(exc_type, exc_value, exc_tb))
    show_error_msg(exc_type, exc_value, tb, **kwargs)
