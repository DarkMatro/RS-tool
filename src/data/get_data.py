# pylint: disable=no-name-in-module, too-many-lines, invalid-name, import-error
"""
This module provides utility functions for handling errors and displaying error messages in a GUI
application.

Functions:
    get_parent(parent, class_name: str):
        Recursively searches for and returns the parent widget of a given class name.

    show_error_msg(exc_type, exc_value, exc_tb, parent=None):
        Displays an error message box, logs the error, and copies the error traceback to the
        clipboard.
"""
from logging import error

from pyperclip import copy as pyperclip_copy

from qfluentwidgets import MessageBox


def get_parent(parent, class_name: str):
    """
    Recursively searches for and returns the parent widget of a given class name.

    Parameters:
    parent : QWidget
        The starting widget to search from.
    class_name : str
        The class name of the desired parent widget.

    Returns:
    QWidget
        The parent widget of the specified class name.
    """
    if parent.__class__.__name__ == class_name:
        return parent
    return get_parent(parent.parent, class_name)


def show_error_msg(exc_type, exc_value, exc_tb, parent=None):
    """
    Displays an error message box, logs the error, and copies the error traceback to the clipboard.

    Parameters:
    exc_type : str
        The type of the exception.
    exc_value : str
        The value or message of the exception.
    exc_tb : str
        The traceback of the exception.
    parent : Optional[object]
        The parent widget for the message box. Default is None.
    """
    msg = MessageBox(str(exc_type), str(exc_value), parent, {'Ok'})
    msg.setInformativeText('For full text of error go to %appdata%/RS-Tool/log.log' + '\n' + exc_tb)
    print(exc_tb)
    error(exc_tb)
    pyperclip_copy(exc_tb)
    msg.exec()
