"""
Read global settings

This file contains the following functions:
    * read_preferences
"""

import json
from pathlib import Path
from os import environ

from ..data.config import get_config


def read_preferences() -> None:
    """
    Read setting from preferences.json into environ
    """
    _check_preferences_file()
    path = get_config()['preferences']['path']
    with open(path, encoding="utf-8") as f:
        settings = json.load(f)
    for k, v in settings.items():
        environ[k] = str(v)


def save_preferences() -> None:
    """
    Read setting from environ and write into preferences.json
    """
    _check_preferences_file()
    keys = _standard_settings().keys()
    path = get_config()['preferences']['path']
    with open(path, 'r+', encoding="utf-8") as f:
        data = json.load(f)
        for k in keys:
            data[k] = environ[k]
        f.seek(0)
        json.dump(data, f)
        f.truncate()


def _check_preferences_file() -> None:
    """
    Check that preferences exists. If it is not - create new.
    """
    path = get_config()['preferences']['path']
    if Path(path).exists():
        return
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(_standard_settings(), f)


def _standard_settings() -> dict:
    """
    Setting to create new preferences file if it's missing.
    """
    return {'theme': 'Dark', 'theme_color': 'Amber', 'recent_limit': '10', 'undo_limit': '20',
            'plot_font_size': '10', 'axis_label_font_size': '14'}
