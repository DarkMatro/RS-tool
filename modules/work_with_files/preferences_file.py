from logging import warning
from os import environ
from pathlib import Path
from xml.dom.minidom import parse


def read_preferences() -> list[str, str, str, str, str, str, str]:
    theme_d = 'Dark'
    theme_color_d = 'Amber'
    recent_limit_d = '10'
    undo_limit_d = '20'
    auto_save_minutes_d = '5'
    plot_font_size_d = '10'
    axis_label_font_size_d = '14'
    with open('preferences.txt', 'r') as f:
        theme_from_file_f = f.readline().strip()
        theme_color_from_file_f = f.readline().strip()
        recent_limit_from_file_f = f.readline().strip()
        undo_limit_f = f.readline().strip()
        auto_save_minutes_f = f.readline().strip()
        plot_font_size_f = f.readline().strip()
        axis_label_font_size_f = f.readline().strip()
    theme_from_file = theme_from_file_f if theme_from_file_f else theme_d
    theme_color_from_file_f = theme_color_from_file_f if theme_color_from_file_f else theme_color_d
    recent_limit_from_file = recent_limit_from_file_f if recent_limit_from_file_f else recent_limit_d
    undo_limit_from_file = undo_limit_f if undo_limit_f else undo_limit_d
    auto_save_minutes_from_file = auto_save_minutes_f if auto_save_minutes_f else auto_save_minutes_d
    plot_font_size_from_file = plot_font_size_f if plot_font_size_f else plot_font_size_d
    axis_label_font_size_from_file = axis_label_font_size_f if axis_label_font_size_f else axis_label_font_size_d
    return [theme_from_file,
            theme_color_from_file_f,
            recent_limit_from_file,
            undo_limit_from_file,
            auto_save_minutes_from_file,
            plot_font_size_from_file,
            axis_label_font_size_from_file]


def check_preferences_file() -> None:
    path = 'preferences.txt'
    if not Path(path).exists():
        with open(path, 'w') as text_file:
            text_file.write('Dark ' + '\n' + 'Default' + '\n' + '10' + '\n' + '20' + '\n' + '16' + '\n' + '12' + '\n' +
                            '13' + '\n')


def get_theme(theme: tuple[str, str, dict | None] = ('Mid Dark', 'Default', None), invert_secondary: bool = False) \
        -> dict[str]:
    theme_bckgrnd, theme_color, _ = theme
    theme_bckgrnd_path = Path(__file__).parent.parent.parent / './material/themes' / (theme_bckgrnd + '.xml')
    theme_color_path = Path(__file__).parent.parent.parent / './material/themes' / (theme_color + '.xml')

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
