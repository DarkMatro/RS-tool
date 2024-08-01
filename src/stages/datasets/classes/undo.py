"""
undo.py

This module defines an undo command class for deleting rows in a deconvoluted dataset table view.
The class includes methods for performing the deletion, undoing the deletion, and updating
UI elements after the operation. It leverages deep copying to maintain the integrity of the
dataset during undo and redo operations.
"""

from copy import deepcopy

from src import UndoCommand


class CommandDeleteDatasetRow(UndoCommand):
    """
    Command to delete a row in the deconvoluted_dataset_table_view.

    Parameters
    ----------
    data : None
        Placeholder for command data.
    idx_type_param_count_legend_func : list of tuple
        A list of tuples containing index, type, parameter count, legend, and function.
    description : str
        Description to set in tooltip.

    Attributes
    ----------
    stage :
        The current stage of the process.
    dataset_table :
        The table from which rows will be deleted.
    _df_backup :
        Backup of the dataframe before deletion.
    _selected_filenames :
        Filenames of the selected rows to be deleted.
    """

    def __init__(self, data: None, parent, text: str, *args, **kwargs) -> None:
        """
        Initialize the command with data, parent, text, and additional arguments.

        Parameters
        ----------
        data : None
            Placeholder for command data.
        parent :
            The parent object.
        text : str
            Text description for the command.
        *args :
            Additional arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        None
        """
        self.stage = kwargs.pop('stage')
        self.dataset_table = kwargs.pop('dataset_table')
        super().__init__(data, parent, text, *args, **kwargs)
        self._df_backup = deepcopy(self.dataset_table.model().dataframe)
        selected_rows = [x.row() for x in self.dataset_table.selectionModel().selectedRows()]
        self._selected_filenames = self.dataset_table.model().column_data_by_indexes(selected_rows,
                                                                                     'Filename')

    def redo_special(self):
        """
        Redo the deletion of rows.

        Returns
        -------
        None
        """
        self.dataset_table.model().delete_rows_by_filenames(self._selected_filenames)


    def undo_special(self):
        """
        Undo the deletion of rows by restoring the dataframe.

        Returns
        -------
        None
        """
        self.dataset_table.model().set_dataframe(self._df_backup)


    def stop_special(self) -> None:
        """
        Update UI elements to reflect changes.

        Returns
        -------
        None
        """
        self.parent.set_modified()