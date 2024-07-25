from src import UndoCommand


class CommandAfterFittingStat(UndoCommand):
    """
    undo / redo fit classificator model

    Parameters
    ----------
    data : model
        new_style, old_style
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
    description : str
        Description to set in tooltip
    parent: context class
    """

    def __init__(self, data, parent, text: str, *args, **kwargs) \
            -> None:
        self.stage = kwargs.pop('stage')
        self.cl_type = kwargs.pop('cl_type')
        super().__init__(data, parent, text, *args, **kwargs)

    def redo_special(self):
        """
        f
        """



    def undo_special(self):
        """
        Undo data
        """



    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.stage.update_stat_report_text(self.cl_type)