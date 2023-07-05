import platform
import sys
from logging import warning
from os import environ
from pathlib import Path
from xml.dom.minidom import parse
import jinja2
from qtpy.QtCore import QDir, Qt
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import QApplication, QComboBox
from modules.default_values import program_version
from modules.resourse_generator import ResourseGenerator


def update_svg_colors(theme: dict) -> None:
    colors_files_dict = dict()
    colors_files_dict['backgroundInsideColor'] = ['activity', 'activity_hover', 'menu', 'menu_hover', 'table',
                                                  'table_hover', 'bar-chart-2', 'bar-chart-2_hover', 'cpu', 'cpu_hover']
    colors_files_dict['primaryColor'] = ['activity_checked', 'activity_checked_hover', 'menu_checked',
                                         'menu_checked_hover', 'refresh-cm-hover', 'table_checked',
                                         'table_checked_hover', 'bar-chart-2_checked', 'bar-chart-2_checked_hover',
                                         'cpu_checked', 'cpu_checked_hover']
    colors_files_dict['primaryTextColor'] = ['file-plus', 'refresh-cm', 'save']
    colors_files_dict['inverseTextColor'] = ['file-plus_checked']
    colors_files_dict['primaryDarker'] = ['crosshair', 'archive', 'anchor', 'code', 'sun']
    colors_files_dict['secondaryColor'] = ['crosshair-off', 'archive-off', 'anchor-off', 'code-off', 'sun-off']
    for color in colors_files_dict:
        color_data = 'stroke="' + theme[color] + '" \n'
        for filename in colors_files_dict[color]:
            filepath = 'material/resources/source/' + filename + '.svg'
            with open(filepath, 'r') as file:
                data = file.readlines()
                data[1] = color_data
            with open(filepath, 'w') as file:
                file.writelines(data)

    colors_files_dict = dict()
    colors_files_dict['selectedBtn'] = ['branch-open', 'branch-closed']
    for color in colors_files_dict:
        for filename in colors_files_dict[color]:
            filepath = 'material/resources/source/' + filename + '.svg'
            with open(filepath, 'r') as file:
                data = file.readlines()
                data[1] = 'fill="' + theme[color] + '" \n'
                data[2] = 'stroke="' + theme[color] + '" \n'
            with open(filepath, 'w') as file:
                file.writelines(data)


def build_stylesheet(theme_in: tuple[str, str, dict] = ('Mid Dark', 'Default', None), invert_secondary: bool = False,
                     parent: str = 'theme') -> str:

    try:
        add_fonts()
    except Exception as e:
        warning(e)

    theme = theme_in[2]
    if not theme:
        theme = get_theme(theme_in, invert_secondary)
    if theme is None:
        return None

    set_icons_theme(theme, parent=parent)
    # Render custom template
    parent = Path(__file__).parent.parent / 'material'
    template = parent / 'material.css.template'
    template = template.name
    loader = jinja2.FileSystemLoader(parent)
    env = jinja2.Environment(loader=loader)
    env.filters['opacity'] = opacity
    env.filters['density'] = density
    stylesheet = env.get_template(template)

    theme.setdefault('icon', None)
    theme.setdefault('font_family', 'AbletonSans, Roboto')
    theme.setdefault('danger', '#dc3545')
    theme.setdefault('warning', '#ffc107')
    theme.setdefault('success', '#17a2b8')
    theme.setdefault('density_scale', '0')
    theme.setdefault('button_shape', 'default')

    update_svg_colors(theme)

    environ = {
        'linux': platform.system() == 'Linux',
        'windows': platform.system() == 'Windows',
        'darwin': platform.system() == 'Darwin',
        'isthemelight': invert_secondary,
        'pyqt5': 'PyQt5' in sys.modules,
        'pyqt6': 'PyQt6' in sys.modules,
        'pyside2': 'PySide2' in sys.modules,
        'pyside6': 'PySide6' in sys.modules,
    }

    environ.update(theme)
    return stylesheet.render(environ)


def get_theme(theme: tuple[str, str, dict | None] = ('Mid Dark', 'Default', None), invert_secondary: bool = False) \
        -> dict[str]:
    theme_bckgrnd, theme_color, _ = theme

    theme_bckgrnd_path = Path(__file__).parent.parent / 'material/themes' / (theme_bckgrnd + '.xml')
    theme_color_path = Path(__file__).parent.parent / 'material/themes' / (theme_color + '.xml')

    if not Path(theme_bckgrnd_path).exists:
        warning(f"{theme_bckgrnd_path} not exist!")
        return None
    if not Path(theme_color_path).exists:
        warning(f"{theme_color_path} not exist!")
        return None

    doc_bckgrnd = parse(str(theme_bckgrnd_path))
    doc_color = parse(str(theme_color_path))
    theme_dict = {child.getAttribute('name'): child.firstChild.nodeValue
                  for child in doc_bckgrnd.getElementsByTagName('color')}
    theme_color_dict = {child.getAttribute('name'): child.firstChild.nodeValue
                        for child in doc_color.getElementsByTagName('color')}
    theme_dict.update(theme_color_dict)

    for k in theme_dict:
        environ[str(k)] = theme_dict[k]

    if invert_secondary:
        theme_dict['secondaryColor'], theme_dict['secondaryLightColor'], theme_dict['secondaryDarkColor'] = \
            theme_dict['secondaryColor'], theme_dict['secondaryDarkColor'], theme_dict['secondaryLightColor']

    for color in ['primaryColor',
                  'secondaryColor',
                  'secondaryLightColor',
                  'secondaryDarkColor',
                  'primaryTextColor',
                  'secondaryTextColor']:
        environ[f'QTMATERIAL_{color.upper()}'] = theme_dict[color]
    environ["QTMATERIAL_THEME"] = theme_bckgrnd + ' ' + theme_color

    return theme_dict


def add_fonts() -> None:
    fonts_path = Path(__file__).parent.parent / 'material/fonts'
    folders = [f for f in fonts_path.iterdir() if f.is_dir()]
    for font_dir in folders:
        for font in font_dir.iterdir():
            if str(font).endswith('.ttf'):
                QFontDatabase.addApplicationFont(str(font))


def apply_stylesheet(app: QApplication, theme: tuple[str, str, dict] = ('Mid Dark', 'Default', {}),
                     invert_secondary: bool = False, parent: str = 'theme') -> None:
    stylesheet = build_stylesheet(theme, invert_secondary, parent)
    if stylesheet is None:
        return
    app.setStyleSheet(stylesheet)


def opacity(theme: dict, value: float = 0.5) -> str:
    """"""
    r, g, b = theme[1:][0:2], theme[1:][2:4], theme[1:][4:]
    r, g, b = int(r, 16), int(g, 16), int(b, 16)

    return f'rgba({r}, {g}, {b}, {value})'


def density(value, density_scale, border=0, scale=1, density_interval=4):
    """"""
    # https://material.io/develop/web/supporting/density
    if isinstance(value, str) and value.startswith('@'):
        return value[1:] * scale

    if value == 'unset':
        return 'unset'

    if isinstance(value, str):
        value = float(value.replace('px', ''))

    density = (value + (density_interval * int(density_scale)) -
               (border * 2)) * scale

    if density < 0:
        density = 0
    return density


def set_icons_theme(theme: dict, parent: str = 'theme') -> None:
    resources_Path = Path(__file__).parent.parent / 'material/resources'
    source = resources_Path / 'source'
    resources = ResourseGenerator(primary=theme['primaryColor'], secondary=theme['secondaryColor'],
                                  disabled=theme['secondaryLightColor'], source=source, parent=parent)
    resources.generate()

    QDir.addSearchPath('icon', resources.index)
    QDir.addSearchPath('material', str(resources_Path))


def list_themes(background: bool = True):
    """"""
    themes_dir = Path(__file__).parent.parent / 'material/themes'
    themes = themes_dir.iterdir()
    bckgrnds = ['Light', 'Mid Light', 'Mid Dark', 'Dark']
    if background:
        result = bckgrnds
    else:
        themes = filter(lambda a: a.stem not in bckgrnds, themes)
        result = [i.stem for i in themes]
        result = sorted(list(result))
    return result


def splash_show_message(splash, text) -> None:
    splash_color = Qt.GlobalColor.black if environ['splash_color'] == 'black' else Qt.GlobalColor.white
    splash.showMessage(program_version() + '\n' + text, Qt.AlignmentFlag.AlignBottom, splash_color)


class QtStyleTools:
    """"""

    @staticmethod
    def add_menu_combobox(combobox_ref: QComboBox, background: bool = True):
        """"""
        themes = list_themes(background)
        for i in themes:
            combobox_ref.addItem(i)

    @staticmethod
    def apply_stylesheet(parent, theme, invert_secondary=False, callable_=None):
        """"""
        apply_stylesheet(parent, theme=theme, invert_secondary=invert_secondary)

        if callable_:
            callable_()

    def update_theme_event(self, parent, theme_bckgrnd: str = 'Dark', theme_color: str = 'Amber',
                           theme_colors: dict = None) -> None:
        invert = 'Light' in theme_bckgrnd and 'Dark' not in theme_bckgrnd
        self.apply_stylesheet(parent, theme=(theme_bckgrnd, theme_color, theme_colors), invert_secondary=invert)
