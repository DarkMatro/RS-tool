# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module contains the `CommandAfterFittingStat` class, which represents
an undo/redo command for fitting a classifier model within a larger system.
The command allows for reverting and reapplying changes related to the model's
fitting process and updates the associated UI elements accordingly.

Dependencies
------------
- `deepcopy` from `copy`: To create deep copies of objects for undo operations.
- `UndoCommand`: A base class likely providing undo/redo functionality.

Notes
-----
The `CommandAfterFittingStat` class encapsulates the logic needed to manage the
state of a fitted classifier model, including storing previous states, applying
new states, and updating the user interface.
"""
from copy import deepcopy

from src import UndoCommand


class CommandAfterFittingStat(UndoCommand):
    """
    A command for undoing or redoing the fitting of a classifier model.

    Parameters
    ----------
    data : fit_data
        The new state data of the classifier model.
    parent : context class
        The parent context that holds UI or application-related state.
    text : str
        Description or text to be used in the command's action (e.g., for labeling).
    idx_type_param_count_legend_func : list of tuple[int, str, int, str, callable]
        A list of tuples containing information about the classifier's parameters, legends, and
        functions.
    """
    def __init__(self, data, parent, text: str, *args, **kwargs) \
            -> None:
        """
        Initializes the `CommandAfterFittingStat` with the given data and context.

        Parameters
        ----------
        data : fit_data
            The new state data for the classifier model.
        parent : context class
            The parent context that manages UI or application state.
        text : str
            Description or label for the command.
        *args : tuple
            Additional positional arguments passed to the superclass.
        **kwargs : dict
            Additional keyword arguments passed to the superclass. Should include:
                - stage : The current stage of the fitting process.
                - cl_type : The type of classifier being fitted.
        """
        self.stage = kwargs.pop('stage')
        self.cl_type = kwargs.pop('cl_type')
        super().__init__(data, parent, text, *args, **kwargs)
        if self.cl_type not in self.stage.data:
            self.stat_result_old = None
        else:
            self.stat_result_old = deepcopy(self.stage.data[self.cl_type])

    def redo_special(self):
        """
        Applies the new state data for the classifier model.

        This method updates the model's state to the new data provided during
        the redo operation and updates the UI to reflect these changes.
        """
        if self.data is not None:
            self.stage.data[self.cl_type] = self.data

    def undo_special(self):
        """
        Reverts the classifier model to the previous state.

        This method restores the model's state to the previous state (before the
        redo operation) and updates the UI to reflect the reverted changes.
        """
        self.stage.data[self.cl_type] = self.stat_result_old

    def stop_special(self) -> None:
        """
        Updates UI elements to reflect the current state.

        This method refreshes the statistics report and plots for the classifier
        model and marks the parent context as modified to indicate changes have
        occurred.
        """
        self.stage.update_stat_report_text(self.cl_type)
        self.stage.plots.update_plots(self.cl_type)
        self.parent.set_modified()
