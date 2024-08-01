# pylint: disable=no-name-in-module, relative-beyond-top-level, import-error
# pylint: disable=too-many-instance-attributes
"""
Master class. Control all backend variables and modules.

classes:
    * Context
"""
from os import environ

from qtpy.QtCore import QObject
from qtpy.QtWidgets import QMainWindow, QUndoStack

from src.stages.fitting.classes.decomposition import DecompositionStage
from .input_table import InputTable
from ..backend.group_table import GroupTable
from ..stages.datasets.classes.datasets import Datasets
from ..stages.ml.classes.ml import ML
from ..stages.predict.predict import Predict
from ..stages.preprocessing.classes.preprocessing import Preprocessing


class Context(QObject):
    """
    Master backend class managing the application's core functionality.

    This class is responsible for managing the state of the application, including handling
    different stages such as preprocessing, decomposition, machine learning, datasets,
    and prediction. It also manages undo/redo operations and tracks modifications.

    Attributes
    ----------
    parent : QMainWindow
        The parent window of the context.
    modified : bool
        Indicates whether the data has been modified.
    preprocessing : Preprocessing
        An instance of the Preprocessing stage.
    decomposition : DecompositionStage
        An instance of the Decomposition stage.
    ml : ML
        An instance of the Machine Learning stage.
    datasets : Datasets
        An instance of the Datasets stage.
    predict : Predict
        An instance of the Predict stage.
    undo_stack : QUndoStack
        The stack for undo and redo operations.
    group_table : GroupTable
        An instance of the GroupTable.
    input_table : InputTable
        An instance of the InputTable.

    Methods
    -------
    set_modified(b=True)
        Sets the 'modified' attribute when data is changed.
    _update_undo_redo_tooltips()
        Updates the tooltips for the undo and redo actions.
    reset()
        Resets all stages and tables to their initial states and clears the modified flag.
    """

    def __init__(self, parent: QMainWindow, *args, **kwargs) -> None:
        """
        Initializes the Context with the given parent and other arguments.

        Parameters
        ----------
        parent : QMainWindow
            The parent window of the context.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.modified = False
        self.preprocessing = Preprocessing(self)
        self.decomposition = DecompositionStage(self)
        self.ml = ML(self)
        self.datasets = Datasets(self)
        self.predict = Predict(self)

        self.undo_stack = QUndoStack(self.parent)
        self.undo_stack.setUndoLimit(int(environ["undo_limit"]))
        self.undo_stack.canUndoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.canRedoChanged.connect(self._update_undo_redo_tooltips)
        self.group_table = GroupTable(self)
        self.input_table = InputTable(self)

    def set_modified(self, b: bool = True) -> None:
        """
        Sets the 'modified' attribute when data is changed.

        This attribute is used before closing the app to check if all changes have been saved.

        Parameters
        ----------
        b : bool, optional, default=True
            Indicates whether the data has been modified.
        """
        self.modified = b
        if b:
            self.parent.ui.unsavedBtn.show()
        else:
            self.parent.ui.unsavedBtn.hide()

    def _update_undo_redo_tooltips(self) -> None:
        """
        Updates the tooltips for the undo and redo actions.

        The tooltips display the action text for the next undo or redo operation.
        """
        if self.undo_stack.canUndo():
            self.parent.attrs.action_undo.setToolTip(self.undo_stack.undoText())
        else:
            self.parent.attrs.action_undo.setToolTip("")

        if self.undo_stack.canRedo():
            self.parent.attrs.action_redo.setToolTip(self.undo_stack.redoText())
        else:
            self.parent.attrs.action_redo.setToolTip("")

    def reset(self):
        """
        Resets all stages and tables to their initial states and clears the modified flag.

        This method resets the preprocessing, decomposition, datasets, machine learning,
        and prediction stages, as well as the input and group tables. It also clears the
        modified flag, indicating that there are no unsaved changes.
        """
        self.preprocessing.reset()
        self.decomposition.reset()
        self.datasets.reset()
        self.ml.reset()
        self.predict.reset()
        self.input_table.reset()
        self.group_table.reset()
        self.set_modified(False)
