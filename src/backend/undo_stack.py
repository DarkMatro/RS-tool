"""
UndoStack master class.

classes:
    * UndoCommand
"""
from datetime import datetime
from typing import Any

from qtpy.QtCore import QObject
from qtpy.QtWidgets import QUndoCommand

from ..data.get_data import get_parent


class UndoCommand(QUndoCommand, QObject):
    """
    UndoStack master class. Parent of all undoCommand classes.

    Parameters
    -------
    data: Any
    parent: Context class
    text: str
        description

    Attributes
    -------
    mw: MainWindow class

    time_start: datetime
        Using for print operation duration in status bar.
    """

    def __init__(self, data: Any, parent, text: str, *args, **kwargs) -> None:
        super().__init__(text, *args, **kwargs)
        self.data = data
        self.parent = parent

        self.mw = get_parent(self.parent, 'MainWindow')
        self.time_start = None

    def redo(self) -> None:
        """
        Overrides standard redo in QUndoCommand.
        To override redo for current operation use redo_special below.
        """
        self.time_start = datetime.now() if self.mw.progress.time_start is None \
            else self.mw.progress.time_start
        self.mw.ui.statusBar.showMessage('Redo...' + self.text())
        self.redo_special()
        self.stop()

    def redo_special(self):
        """
        Override this function
        """

    def undo(self):
        """
        Overrides standard undo in QUndoCommand.
        To override undo for current operation use undo_special below.
        """
        self.time_start = datetime.now() if self.mw.progress.time_start is None \
            else self.mw.progress.time_start
        self.mw.ui.statusBar.showMessage('Redo...' + self.text())
        self.undo_special()
        self.stop()

    def undo_special(self):
        """
        Override this function
        """

    def stop(self) -> None:
        """
        Show operation time and set modified.
        """
        self.stop_special()
        seconds = round((datetime.now() - self.time_start).total_seconds())
        self.mw.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.parent.set_modified()
        self.mw.progress.time_start = None

    def stop_special(self) -> None:
        """
        Override this function
        """
