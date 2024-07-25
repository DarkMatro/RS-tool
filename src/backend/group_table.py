"""
Module for managing the groups table in a GUI application.

This module provides classes and functions to interact with and control the groups table,
which is part of the application's user interface. It allows for adding, deleting, and
modifying group entries, as well as handling the associated styles and interactions.

Classes:
    GroupTable: Main class for managing the groups table interface.
    CommandChangeGroupStyle: Command for changing the style of a group.
    CommandChangeGroupCellsBatch: Command for changing group IDs of multiple rows.
    CommandAddGroup: Command for adding a new group to the table.
    CommandDeleteGroup: Command for deleting a group from the table.
"""

from gc import get_objects
from os import environ

import pandas as pd
from asyncqtpy import asyncSlot
from pyqtgraph import mkPen
from qtpy.QtCore import QObject, QModelIndex, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QHeaderView, QAbstractItemView

from src.backend.undo_stack import UndoCommand
from ..data.get_data import get_parent
from ..pandas_tables import GroupsTableModel
from ..ui.MultiLine import MultiLine
from ..widgets.curve_properties_window import CurvePropertiesWindow
from ..data.plotting import random_rgb
from src.ui.style import color_dialog


class GroupTable(QObject):
    """
    Class for Groups table control.

    This class manages the groups table widget, including initialization, data loading, and
    handling user interactions such as clicks and selections.

    Parameters
    ----------
    parent: Context
        The parent context in which this table operates.

    Attributes
    ----------
    parent: Context
        The parent context.
    mw: MainWindow
        Reference to the main window.
    table_widget: QTableWidget
        The table widget for displaying groups.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the GroupTable instance.

        Parameters
        ----------
        parent : Context
            The parent context in which this table operates.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(parent.parent, *args, **kwargs)
        self.parent = parent
        self.mw = get_parent(self.parent, "MainWindow")
        self.table_widget = self.mw.ui.GroupsTable
        self._init_table()

    def _init_table(self) -> None:
        """
        Initialize pandas model for table widget.
        """
        self.reset()
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_widget.setColumnWidth(1, 160)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_widget.clicked.connect(self._group_table_cell_clicked)

    def reset(self):
        """
        Reset the table by initializing an empty dataframe model.
        """
        df = pd.DataFrame(columns=["Group name", "Style"])
        model = GroupsTableModel(df, self.parent)
        self.table_widget.setModel(model)

    def read(self) -> pd.DataFrame:
        """
        Read the data from the table.

        Returns
        -------
        pd.DataFrame
            The dataframe representing the current state of the table.
        """
        return self.table_widget.model().dataframe()

    def load(self, df: pd.DataFrame) -> None:
        """
        Load data into the table from a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing data to load into the table.
        """
        self.table_widget.model().set_dataframe(df)

    @property
    def rowCount(self):
        """
        Get the number of rows in the table.

        Returns
        -------
        int
            The number of rows in the table.
        """
        return self.table_widget.model().rowCount()

    def groups_list(self) -> list:
        """
        Get the list of groups from the table.

        Returns
        -------
        list
            A list of groups.
        """
        return self.table_widget.model().groups_list()

    def target_names(self, classes: list[str]) -> list[str]:
        """
        Get the target names for specified classes.

        Parameters
        ----------
        classes : list of str
            The classes for which to get target names.

        Returns
        -------
        list of str
            The target names corresponding to the specified classes.
        """
        return self.table_widget.model().dataframe().loc[classes]["Group name"].values

    @asyncSlot()
    async def _group_table_cell_clicked(self) -> None:
        """
        Handle click events on the table cells.
        Opens a style window for style changes, or changes group numbers if applicable.
        """
        main_window = get_parent(self.parent, "MainWindow")
        if main_window.progress.time_start is not None:
            return
        current_column = self.table_widget.selectionModel().currentIndex().column()
        selected_input_table_idx = self.mw.ui.input_table.selectionModel().selectedIndexes()
        if current_column == 1:
            # change color
            await self._change_style_of_group()
        elif (
                current_column == 0
                and not self.mw.ui.by_group_control_button.isChecked()
                and len(selected_input_table_idx) > 1
        ):
            await self._change_group_for_many_rows(selected_input_table_idx)
        elif current_column == 0 and self.mw.ui.by_group_control_button.isChecked():
            # show group's spectrum only in plot
            await self.mw.loop.run_in_executor(
                None, self.parent.preprocessing.stages.input_data.despike_history_remove_plot
            )
            await self.mw.update_plots_for_group(None)

    async def _change_style_of_group(self) -> None:
        """
        Open the CurvePropertiesWindow to change the style of a group.
        """
        current_row = self.table_widget.selectionModel().currentIndex().row()
        row_data = self.table_widget.model().row_data(current_row)
        idx = row_data.name
        style = row_data["Style"]
        for obj in get_objects():
            if (
                    isinstance(obj, CurvePropertiesWindow)
                    and obj.idx() == idx
                    and obj.isVisible()
            ):
                return
        window_cp = CurvePropertiesWindow(self.mw, style, idx, fill_enabled=False)
        window_cp.sigStyleChanged.connect(self._group_style_changed)
        window_cp.show()

    def _group_style_changed(self, style: dict, old_style: dict, idx: int) -> None:
        """
        Handle changes to the group's style.

        Parameters
        ----------
        style : dict
            The new style settings.
        old_style : dict
            The old style settings.
        idx : int
            The index of the group in the table.
        """
        data = style, old_style, idx
        command = CommandChangeGroupStyle(data, self.parent, f"Change group ({idx}) style")
        self.parent.undo_stack.push(command)

    async def _change_group_for_many_rows(self, selected_indexes: list[QModelIndex]) -> None:
        """
        Change the group number for multiple selected rows.

        Parameters
        ----------
        selected_indexes : list of QModelIndex
            The selected indexes in the input table.
        """
        undo_dict = {}
        for i in selected_indexes:
            idx = i.row()
            current_row_group = (self.table_widget.selectionModel().currentIndex().row())
            new_value = current_row_group + 1
            old_value = self.mw.ui.input_table.model().cell_data(i.row(), 2)
            if new_value != old_value:
                undo_dict[idx] = (new_value, old_value)
        if not undo_dict:
            return
        command = CommandChangeGroupCellsBatch(undo_dict, self.parent,
                                               "Change group numbers for cells")
        self.parent.undo_stack.push(command)

    def add_new_group(self) -> None:
        """
        Add a new group to the table.
        """
        main_window = get_parent(self.parent, "MainWindow")
        if main_window.progress.time_start is not None:
            return
        this_row = self.table_widget.model().rowCount()
        init_color = QColor(environ["secondaryColor"])
        dialog = color_dialog(init_color)
        color = dialog.getColor(init_color)
        if not color.isValid():
            return
        command = CommandAddGroup((this_row, color), self.parent, f"Add group {this_row + 1}")
        self.parent.undo_stack.push(command)

    def dlt_selected_group(self) -> None:
        """
        Delete the selected group from the table.
        """
        main_window = get_parent(self.parent, "MainWindow")
        if main_window.progress.time_start is not None:
            return
        selection = self.table_widget.selectionModel()
        row = selection.currentIndex().row()
        if row == -1:
            return
        group_number = self.table_widget.model().row_data(row).name - 1
        if row <= -1:
            return
        name, style = self.table_widget.model().row_data(row)
        command = CommandDeleteGroup((group_number, name, style), self.parent,
                                     f"Delete group {group_number}")
        self.parent.undo_stack.push(command)

    def get_color_by_group_number(self, group_number: str) -> QColor:
        """
        Get the color associated with a group number.

        Parameters
        ----------
        group_number : str
            The group number as a string.

        Returns
        -------
        QColor
            The color associated with the group number.
        """
        if (group_number != "nan" and group_number != ""
                and self.table_widget.model().rowCount() > 0
                and int(group_number) <= self.table_widget.model().rowCount()):
            color = self.table_widget.model().cell_data(int(group_number) - 1, 1)["color"]
            return color

        return QColor(environ["secondaryColor"])


class CommandChangeGroupStyle(UndoCommand):
    """
    Command to change the style of a group.

    This command encapsulates the logic for changing a group's style and provides undo/redo
    functionality.

    Parameters
    ----------
    data : tuple
        A tuple containing the new style, old style, and the group index.
    parent : QObject
        The parent object.
    text : str
        Description of the command.
    """

    def __init__(self, data: tuple[dict, dict, int], parent, text: str, *args,
                 **kwargs) -> None:
        """
        Initialize the command.

        Parameters
        ----------
        data : tuple
            The new style, old style, and group index.
        parent : QObject
            The parent object.
        text : str
            Description of the command.
        """
        super().__init__(data, parent, text, *args, **kwargs)
        self.new_style, self.old_style, self.idx = data
        self.table_widget = self.parent.group_table.table_widget

    def redo_special(self):
        """
        Redo the command, applying the new style.
        """
        self.table_widget.model().change_cell_data(self.idx - 1, 1, self.new_style)
        self.change_plot_color_for_group(self.idx, self.new_style)

    def undo_special(self):
        """
        UUndo the command, reverting to the old style.
        """
        self.table_widget.model().change_cell_data(self.idx - 1, 1, self.old_style)
        self.change_plot_color_for_group(self.idx, self.old_style)

    def change_plot_color_for_group(self, group_number: int, style: dict) -> None:
        """
        Change style for all curves in preproc_plot_widget of group_number.

        Parameters
        -------
        group_number: int

        style: dict
        """
        list_data_items = self.mw.ui.preproc_plot_widget.getPlotItem().listDataItems()
        items_matches = (x for x in list_data_items if isinstance(x, MultiLine)
                         and x.get_group_number() == group_number)
        for i in items_matches:
            color = style["color"]
            color.setAlphaF(1.0)
            pen = mkPen(color=color, style=style["style"], width=style["width"])
            i.setPen(pen)


class CommandChangeGroupCellsBatch(UndoCommand):
    """
    Change group id for selected rows in input table and change plot color for these files.

    Parameters
    -------
    data: dict
        key: int
            idx
        value: tuple of
            new_value: int
                group id new
            old_value: int
                group id old
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: dict, parent, text: str, *args, **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self.data = data
        self.input_table = self.mw.ui.input_table

    def redo_special(self):
        """
        Update cell and plot colors for this group.
        """
        for i in self.data.items():
            self._do(i)

    def undo_special(self):
        """
        Undo update cell and plot colors for this group.
        """
        for i in self.data.items():
            self._do(i, True)

    def _do(self, item: tuple[int, tuple[int, int]], undo: bool = False) -> None:
        row = item[0]
        new_value, old_value = item[1]
        group_number = int(old_value) if undo else int(new_value)

        self.input_table.model().change_cell_data(row, 2, group_number)
        filename = self.input_table.model().index_by_row(row)
        for ds in [self.mw.ui.smoothed_dataset_table_view, self.mw.ui.baselined_dataset_table_view,
                   self.mw.ui.deconvoluted_dataset_table_view]:
            model = ds.model()
            if model.rowCount() > 0 and filename in model.filenames:
                idx_sm = model.idx_by_column_value('Filename', filename)
                model.set_cell_data_by_idx_col_name(idx_sm, 'Class', group_number)

    def stop_special(self) -> None:
        """
        Override this function
        """
        # TODO  self.rs.preprocessing.update_averaged()
        self.parent.preprocessing.update_plot_item()


class CommandAddGroup(UndoCommand):
    """
    Add new row into Groups table, update plot curves color.

    Parameters
    -------
    data: tuple[int, QColor]
        row: int
            id of new row in table
        color: QColor
            of this new group
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: tuple[int, QColor], parent, text: str, *args, **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self.row, self.color = data
        self._style = {'color': self.color,
                       'style': Qt.PenStyle.SolidLine,
                       'width': 1.0,
                       'fill': False,
                       'use_line_color': True,
                       'fill_color': QColor().fromRgb(random_rgb()),
                       'fill_opacity': 0.0}
        self.group_table_view = parent.group_table.table_widget

    def redo_special(self):
        """
        Add new row
        """
        self.group_table_view.model().append_group(group=f"Group {self.row + 1}",
                                                   style=self._style, index=self.row + 1)

    def undo_special(self):
        """
        Undo add new row
        """
        self.group_table_view.model().remove_group(self.row + 1)

    def stop_special(self) -> None:
        """
        Override this function
        """
        # TODO  self.rs.preprocessing.update_averaged()
        self.parent.preprocessing.update_plot_item()


class CommandDeleteGroup(UndoCommand):
    """
    Delete selected row from Groups table, update plot curves color.

    Parameters
    -------
    data: tuple[int, QColor]
        row: int
            id of new row in table
        name: str
            group name
        style: dict
            group style
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: tuple[int, str, dict], parent, text: str, *args, **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self.row, self.name, self.style = data
        self.group_table_view = parent.group_table.table_widget

    def redo_special(self):
        """
        Remove row
        """
        self.group_table_view.model().remove_group(self.row + 1)

    def undo_special(self):
        """
        Undo remove row
        """
        self.group_table_view.model().append_group(group=f"Group {self.row + 1}",
                                                   style=self.style, index=self.row + 1)

    def stop_special(self) -> None:
        """
        Override this function
        """
        # TODO  self.rs.preprocessing.update_averaged()
        self.parent.preprocessing.update_plot_item()
