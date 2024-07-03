"""
F1 instructions

This file contains the following functions:
    * read_preferences
"""
from os import startfile
from ..data.config import get_config


def action_help() -> None:
    """
    Open help manual in browser.
    """
    path = get_config()['help']['path']
    startfile(path)
