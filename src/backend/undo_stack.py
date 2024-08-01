"""
UndoStack master class.

classes:
    * UndoCommand
"""
from datetime import datetime
from typing import Any

from qtpy.QtCore import Qt
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


class CommandUpdateTableCell(UndoCommand):
    """
    Using in PandasModelGroupsTable and PandasModelDeconvLinesTable

    Parameters
    ----------
    data : tuple[str, str]
        value, old_value
    parent : Context
        The parent object.
    text : str
        Description of the command.
    """

    def __init__(self, data: tuple[str, str], parent, text: str, *args,
                 **kwargs) -> None:
        """
        Initialize the command.

        Parameters
        ----------
        data : tuple
            The new style, old style, and group index.
        parent : Context
            The parent object.
        text : str
            Description of the command.
        """
        self._index = kwargs.pop('index')
        self._obj = kwargs.pop('obj')
        super().__init__(data, parent, text, *args, **kwargs)
        self._value, self._old_value = data

    def redo_special(self):
        """
        Redo the command, applying the new style.
        """
        self._obj.set_cell_data_by_index(self._index, self._value)
        self._obj.dataChanged.emit(self._index, self._index)

    def undo_special(self):
        """
        UUndo the command, reverting to the old style.
        """
        self._obj.set_cell_data_by_index(self._index, self._old_value)
        self._obj.dataChanged.emit(self._index, self._index)

    def stop_special(self) -> None:
        self.parent.set_modified()


class CommandFitIntervalChanged(UndoCommand):
    """
    undo / redo change value of fit_borders_TableView row

    Parameters
    ----------
    data : float
        new_value
    parent : Context
        The parent object.
    text : str
        Description of the command.
    """

    def __init__(self, data: tuple[str, str], parent, text: str, *args,
                 **kwargs) -> None:
        """
        Initialize the command.

        Parameters
        ----------
        data : tuple
            The new style, old style, and group index.
        parent : Context
            The parent object.
        text : str
            Description of the command.
        """
        self.index = kwargs.pop('index')
        self.model = kwargs.pop('model')
        super().__init__(data, parent, text, *args, **kwargs)
        self.new_value = data
        self.df = self.mw.ui.fit_borders_TableView.model().dataframe.copy()

    def redo_special(self):
        """
        Redo the command, applying the new style.
        """
        self.model.setData(self.index, self.new_value, Qt.EditRole)

    def undo_special(self):
        """
        UUndo the command, reverting to the old style.
        """
        self.mw.ui.fit_borders_TableView.model().set_dataframe(self.df)

    def stop_special(self) -> None:
        self.model.sort_by_border()
        self.parent.set_modified()
