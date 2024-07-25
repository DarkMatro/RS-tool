from copy import deepcopy

from src import UndoCommand


class CommandDeleteDatasetRow(UndoCommand):
    """
    delete row in deconvoluted_dataset_table_view

    Parameters
    ----------
    data : None
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
    description : str
        Description to set in tooltip

    """

    def __init__(self, data: None, parent, text: str, *args, **kwargs) -> None:
        self.stage = kwargs.pop('stage')
        self.dataset_table = kwargs.pop('dataset_table')
        super().__init__(data, parent, text, *args, **kwargs)
        self._df_backup = deepcopy(self.dataset_table.model().dataframe())
        selected_rows = [x.row() for x in self.dataset_table.selectionModel().selectedRows()]
        self._selected_filenames = self.dataset_table.model().column_data_by_indexes(selected_rows,
                                                                                     'Filename')

    def redo_special(self):
        """
        f
        """
        self.dataset_table.model().delete_rows_by_filenames(self._selected_filenames)


    def undo_special(self):
        """
        Undo data
        """
        self.dataset_table.model().set_dataframe(self._df_backup)


    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.set_modified()