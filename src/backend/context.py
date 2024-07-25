"""
Master class. Control all backend variables and modules.

classes:
    * Context
"""

from qtpy.QtWidgets import QMainWindow, QUndoStack

from .input_table import InputTable
from ..backend.group_table import GroupTable
from ..stages.datasets.classes.datasets import Datasets
from ..stages.preprocessing.classes.preprocessing import Preprocessing
from src.stages.fitting.classes.decomposition import DecompositionStage
from ..stages.ml.classes.ml import ML


class Context:
    """
    Master backend class.
    """

    def __init__(self, parent: QMainWindow) -> None:
        self.parent = parent
        self.modified = False
        self.preprocessing = Preprocessing(self)
        self.decomposition = DecompositionStage(self)
        self.ml = ML(self)
        self.datasets = Datasets(self)

        self.undo_stack = QUndoStack(self.parent)
        self.undo_stack.canUndoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.canRedoChanged.connect(self._update_undo_redo_tooltips)
        self.group_table = GroupTable(self)
        self.input_table = InputTable(self)

    def set_modified(self, b: bool = True) -> None:
        """
        Set 'modified' attribute when data was changed.

        self.modified is using before close app for check that all changed was saved before quit.

        Parameters
        -------
        b: bool, default = True
            set modified or not
        """
        self.modified = b
        if b:
            self.parent.ui.unsavedBtn.show()
        else:
            self.parent.ui.unsavedBtn.hide()

    def _update_undo_redo_tooltips(self) -> None:
        if self.undo_stack.canUndo():
            self.parent.action_undo.setToolTip(self.undo_stack.undoText())
        else:
            self.parent.action_undo.setToolTip("")

        if self.undo_stack.canRedo():
            self.parent.action_redo.setToolTip(self.undo_stack.redoText())
        else:
            self.parent.action_redo.setToolTip("")

    def reset(self):
        self.preprocessing.reset()
        self.decomposition.reset()
        self.datasets.reset()
        self.ml.reset()
        self.input_table.reset()
        self.group_table.reset()
        self.set_modified(False)


