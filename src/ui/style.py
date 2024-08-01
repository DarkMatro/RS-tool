"""
Set stylesheet, update svg icons color, themes control

This file contains the following functions:
    * apply_stylesheet
    * get_theme_colors
    * opacity
"""

import os
import gc
from os import environ
from pathlib import Path
from platform import system
from stat import S_IWUSR, S_IREAD, S_IWRITE
from sys import modules as sys_modules
from traceback import format_exc

import yaml
from jinja2 import FileSystemLoader, Environment
from qtpy.QtCore import QDir
from qtpy.QtGui import QFontDatabase, QColor
from qtpy.QtWidgets import QApplication, QColorDialog

from qfluentwidgets import toggle_theme
from src.ui.resourse_generator import ResourceGenerator
from ..data.config import get_config
from ..data.get_data import show_error_msg
from ..widgets.drag_items import DragItem


def apply_stylesheet(app: QApplication, theme: tuple[str, str, dict] = ('Mid-Dark', 'Default', {}),
                     invert_secondary: bool = False) -> None:
    """
    Build and apply stylesheet to app.

    Parameters
    -------
    app: QApplication
        target app
    theme: tuple[str, str, dict], default = ('Mid-Dark', 'Default', {})
        Background color: 'Dark', 'Mid-Dark', 'Mid-Light', or 'Light'
        theme color: see material/themes
        theme_colors: dict
    invert_secondary: bool
        invert secondary colors
    """
    stylesheet = _build_stylesheet(theme, invert_secondary)
    if stylesheet is None:
        return
    app.setStyleSheet(stylesheet)
    widgets = [w for w in gc.get_objects() if isinstance(w, DragItem)]
    for w in widgets:
        w.setStyleSheet(stylesheet)
    toggle_theme()


def get_theme_colors(theme_bckgrnd: str, theme_color: str, invert_secondary: bool = False) \
        -> dict[str]:
    """
    Read theme colors from files at material/themes by selected theme_bckgrnd and theme_color.

    Parameters
    -------
    theme_bckgrnd: str
        name of background theme
    theme_color: str
        name of colors theme
    invert_secondary: bool, default=False
        invert secondary colors

    Returns
    -------
    out: dict[str]
        with colors
    """
    theme_path = get_config()['theme']['path']
    theme_color_path = theme_path['colors'] + theme_color + '.yml'
    theme_bckgrnd_path = theme_path['background'] + theme_bckgrnd + '.yml'

    with open(theme_color_path, encoding="utf-8") as file:
        theme_color_dict = yaml.load(file, Loader=yaml.FullLoader)['colors']
    with open(theme_bckgrnd_path, encoding="utf-8") as file:
        theme_dict = yaml.load(file, Loader=yaml.FullLoader)['colors']

    theme_dict.update(theme_color_dict)

    # Write colors to environ.
    for k, v in theme_dict.items():
        environ[str(k)] = v

    environ["bckgrnd_theme"] = theme_bckgrnd
    environ["theme"] = theme_color

    if invert_secondary:
        (theme_dict['secondaryLightColor'],
         theme_dict['secondaryDarkColor']) = (theme_dict['secondaryDarkColor'],
                                              theme_dict['secondaryLightColor'])
    return theme_dict


def _build_stylesheet(theme_in: tuple[str, str, dict] = ('Mid-Dark', 'Default', None),
                      invert_secondary: bool = False) -> str:
    """
    Build stylesheet .css
    """
    bckgrnd_theme, theme_color, theme = theme_in
    if not theme:
        theme = get_theme_colors(bckgrnd_theme, theme_color, invert_secondary)
    # Fonts.
    _add_fonts()
    # Built-in icons color.
    _color_icons(theme)
    # Render custom template
    cfg = get_config()
    parent = cfg['material']['path']
    template = cfg['material']['styles_name']
    loader = FileSystemLoader(parent)
    env = Environment(loader=loader)

    env.filters['opacity'] = opacity
    env.filters['density'] = _density
    stylesheet = env.get_template(template)

    theme.setdefault('icon', None)
    theme.setdefault('font_family', 'AbletonSans, Roboto')
    theme.setdefault('danger', '#dc3545')
    theme.setdefault('warning', '#ffc107')
    theme.setdefault('success', '#17a2b8')
    theme.setdefault('density_scale', '0')
    theme.setdefault('button_shape', 'default')

    try:
        _update_svg_colors(theme)
    except PermissionError as err:
        show_error_msg(err, 'Run program as administrator', str(format_exc()))

    env = {
        'linux': system() == 'Linux',
        'windows': system() == 'Windows',
        'darwin': system() == 'Darwin',
        'isthemelight': invert_secondary,
        'pyqt5': 'PyQt5' in sys_modules,
        'pyqt6': 'PyQt6' in sys_modules,
        'pyside2': 'PySide2' in sys_modules,
        'pyside6': 'PySide6' in sys_modules,
    }

    env.update(theme)
    return stylesheet.render(env)


def _update_svg_colors(theme: dict) -> None:
    """
    Update custom svg files according to theme.

    Parameters
    -------
    theme: dict)
        with colors
    """
    colors_files_dict = {
        'backgroundInsideColor': ['activity', 'activity_hover', 'menu', 'menu_hover', 'table',
                                  'table_hover', 'bar-chart-2', 'bar-chart-2_hover', 'cpu',
                                  'cpu_hover'],
        'primaryColor': ['activity_checked', 'activity_checked_hover', 'menu_checked',
                         'menu_checked_hover', 'refresh-cm-hover', 'table_checked',
                         'table_checked_hover', 'bar-chart-2_checked', 'bar-chart-2_checked_hover',
                         'cpu_checked', 'cpu_checked_hover', 'reset-pressed', 'download-pressed',
                         'file-text-pressed'],
        'primaryTextColor': ['file-plus', 'refresh-cm', 'save'],
        'inverseTextColor': ['file-plus_checked'],
        'primaryDarker': ['crosshair','anchor', 'code', 'sun'],
        'secondaryColor': ['crosshair-off', 'anchor-off', 'code-off', 'sun-off'],
        'inversePlotBackground': ['reset', 'reset-hover', 'download', 'download-hover',
                                  'file-text-hover', 'file-text']}
    os.chmod('material/resources/source/', S_IWUSR | S_IREAD | S_IWRITE)
    for color, filenames in colors_files_dict.items():
        color_data = 'stroke="' + theme[color] + '" \n'
        for filename in filenames:
            filepath = 'material/resources/source/' + filename + '.svg'
            os.chmod(filepath, S_IWUSR | S_IREAD | S_IWRITE)
            with open(filepath, encoding='utf=8') as file:
                data = file.readlines()
                data[1] = color_data
            with open(filepath, 'w', encoding='utf=8') as file:
                file.writelines(data)

    colors_files_dict = {'selectedBtn': ['branch-open', 'branch-closed']}
    for color, filenames in colors_files_dict.items():
        for filename in filenames:
            filepath = 'material/resources/source/' + filename + '.svg'
            os.chmod(filepath, S_IWUSR | S_IREAD | S_IWRITE)
            with open(filepath, encoding='utf=8') as file:
                data = file.readlines()
                data[1] = 'fill="' + theme[color] + '" \n'
                data[2] = 'stroke="' + theme[color] + '" \n'
            with open(filepath, 'w', encoding='utf=8') as file:
                file.writelines(data)


def _add_fonts() -> None:
    """
    Read fonts from folder and add to QFontDatabase.
    """
    fonts_path = get_config()['fonts']['path']
    fonts_path = Path(fonts_path)
    folders = [f for f in fonts_path.iterdir() if f.is_dir()]
    for font_dir in folders:
        for font in font_dir.iterdir():
            if str(font).endswith('.ttf') or str(font).endswith('.otf'):
                QFontDatabase.addApplicationFont(str(font))


def _color_icons(theme: dict) -> None:
    """
    Paint icons in accordance with theme

    Parameters
    -------
    theme: dict
        with color scheme
    """
    cfg = get_config()
    resources_path = Path(cfg['resources']['path'])
    source = resources_path / 'source'
    resources = ResourceGenerator(theme['primaryColor'], secondary=theme['secondaryColor'],
                                  disabled=theme['secondaryLightColor'], source=source)
    resources.generate()

    QDir.addSearchPath('icon', resources.index)
    QDir.addSearchPath('material', str(resources_path))


def opacity(value: str, alpha: float = 0.5) -> str:
    """
    Convert RGB value to RGBA

    Parameters
    -------
    value: str
        name of color like #dc3545
    alpha: float, default = 0.5

    Returns
    -------
    out: dict[str]
        with colors
    """
    r, g, b = value[1:][0:2], value[1:][2:4], value[1:][4:]
    r, g, b = int(r, 16), int(g, 16), int(b, 16)

    return f'rgba({r}, {g}, {b}, {alpha})'


def _density(value: int | str, density_scale: int, border: int = 0, scale: int = 1,
             density_interval: int = 4) -> str | int:
    """
    Overwrite density

    Parameters
    -------
    value: int | str
        name of color like #dc3545
    density_scale: int
    border: int, default = 0
    scale: int, default = 1
    density_interval: int, default = 4

    Returns
    -------
    out: str | int
        new density
    """
    if isinstance(value, str) and value.startswith('@'):
        return value[1:] * scale

    if value == 'unset':
        return 'unset'

    if isinstance(value, str):
        value = float(value.replace('px', ''))

    density = (value + (density_interval * int(density_scale)) - (border * 2)) * scale
    density = max(0, density)

    return density

def color_dialog(initial: QColor) -> QColorDialog:
    """
    Create and configure a QColorDialog.

    Parameters
    ----------
    initial : QColor
        The initial color for the dialog.

    Returns
    -------
    QColorDialog
        The configured color dialog.
    """
    dialog = QColorDialog(initial)
    for i, color_name in zip(range(6), ['primaryColor', 'primaryDarker', 'primaryDarkColor',
                                        'secondaryColor', 'secondaryLightColor',
                                        'secondaryDarkColor']):
        dialog.setCustomColor(i, QColor(environ[color_name]))
    return dialog
