# pylint: disable=no-name-in-module, too-many-lines, invalid-name, import-error,
# relative-beyond-top-level
"""
Module for managing the input table, handling initialization, data loading,
user interactions, and undo functionality.

This module integrates with a Qt-based GUI to provide functionalities for
managing the input table in the application. It includes classes and methods
to reset the table, handle item changes, manage context menus, and support
undo operations.
"""
import asyncio
from asyncio import create_task, wait

import pandas as pd
from qtpy.QtGui import QMouseEvent
from asyncqtpy import asyncSlot
from qtpy.QtCore import QObject, QModelIndex, Qt
from qtpy.QtWidgets import QHeaderView, QLineEdit, QMenu

from src.backend.undo_stack import UndoCommand
from ..data.get_data import get_parent
from ..pandas_tables import InputPandasTable


class InputTable(QObject):
    """
    Manages the input table, including initialization, data loading,
    and handling user interactions.

    Parameters
    ----------
    parent : Context
        The parent context in which this table operates.
    *args : tuple
        Additional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the InputTable instance.

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
        self._ascending_input_table = False
        self.parent = parent
        self.mw = get_parent(self.parent, "MainWindow")
        self.table_widget = self.mw.ui.input_table
        self.previous_group_of_item = None
        self._init_table()
        self.table_widget.model().dataChanged.connect(self.mw.decide_vertical_scroll_bar_visible)

    def _init_table(self) -> None:
        """
        Initialize pandas model for table widget.
        """
        self.reset()
        self.table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().resizeSection(0, 80)
        self.table_widget.horizontalHeader().setMinimumWidth(10)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().resizeSection(1, 80)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().resizeSection(2, 50)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().resizeSection(4, 130)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().resizeSection(5, 90)
        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table_widget.mouseDoubleClickEvent = self.sync_input_table_selection_changed
        self.table_widget.verticalScrollBar().valueChanged.connect(self.mw.move_side_scrollbar)
        self.table_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table_widget.moveEvent = self.mw.decide_vertical_scroll_bar_visible
        self.table_widget.model().dataChanged.connect(self.input_table_item_changed)
        self.table_widget.rowCountChanged = self.input_table_rows_changed
        self.table_widget.rowsInserted = self.input_table_rows_changed
        self.table_widget.rowsRemoved = self.input_table_rows_changed
        self.table_widget.horizontalHeader().sectionClicked.connect(
            self._input_table_header_clicked
        )
        self.table_widget.contextMenuEvent = self._input_table_context_menu_event
        self.table_widget.keyPressEvent = self._input_table_key_pressed

    def reset(self):
        """
        Reset the table by initializing an empty dataframe model.
        """
        df = pd.DataFrame(
            columns=["Min, nm", "Max, nm", "Group", "Despiked, nm", "Rayleigh line, nm",
                     "FWHM, nm", "FWHM, cm\N{superscript minus}\N{superscript one}", "SNR",
                     ]
        )
        model = InputPandasTable(df)
        self.table_widget.setSortingEnabled(True)
        self.table_widget.setModel(model)

    def sync_input_table_selection_changed(self, event: QMouseEvent):
        # Call the async function using ensure_future
        asyncio.ensure_future(self.input_table_selection_changed(event))

    @asyncSlot()
    async def input_table_selection_changed(self, _: QMouseEvent):
        """
        Handle changes in table selection, updating plots or saving previous
        group information as needed.
        """
        current_index = self.table_widget.selectionModel().currentIndex()
        if current_index:
            self.table_widget.scrollTo(current_index)
        if current_index.column() == 2:  # groups
            self.previous_group_of_item = int(
                self.table_widget.model().cell_data_by_index(current_index)
            )
        elif self.mw.ui.by_one_control_button.isChecked():  # names
            await self.parent.preprocessing.stages.input_data.despike_history_remove_plot()
            await self.mw.update_plots_for_single()

    def input_table_rows_changed(self):
        """
        Update dec_table.
        """
        print('input_table_rows_changed')
        self.mw.decide_vertical_scroll_bar_visible()
        self.mw.ui.dec_table.model().clear_dataframe()
        self.mw.ui.dec_table.model().concat_deconv_table(
            filename=self.table_widget.model().dataframe.index)

    def input_table_item_changed(self, top_left: QModelIndex = None, _: QModelIndex = None) -> None:
        """
        Handle changes to items in the input table, specifically allowing
        changes to the group column.

        Parameters
        ----------
        top_left : QModelIndex, optional
            The top-left index of the changed item (default is None).
        _ : QModelIndex, optional
            Unused parameter (default is None).
        """
        print('input_table_item_changed')
        if self.table_widget.selectionModel().currentIndex().column() != 2:
            return
        try:
            new_value = int(self.table_widget.model().cell_data_by_index(top_left))
        except ValueError:
            self.table_widget.model().setData(top_left, self.previous_group_of_item, Qt.EditRole)
            return
        if self.parent.group_table.table_widget.model().rowCount() >= new_value >= 0:
            filename = self.table_widget.model().cell_data(top_left.row(), 0)
            command = CommandChangeGroupCell((top_left, new_value), self.parent,
                                             f"Change group number for ({filename})",
                                             **{'table': self})
            self.parent.undo_stack.push(command)
        else:
            self.table_widget.model().setData(top_left, self.previous_group_of_item, Qt.EditRole)

    def _input_table_header_clicked(self, idx: int):
        """
        Handle clicks on the table header to sort the table.

        Parameters
        ----------
        idx : int
            The index of the clicked header.
        """
        print('_input_table_header_clicked')
        df = self.mw.ui.input_table.model().dataframe
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending_input_table = not self._ascending_input_table
        self.mw.ui.input_table.model().sort_values(current_name, self._ascending_input_table)

    def _input_table_context_menu_event(self, a0) -> None:
        """
        Show context menu on right-click event in the input table.

        Parameters
        ----------
        a0 : QContextMenuEvent
            The context menu event triggering the function.
        """
        line = QLineEdit(self.mw)
        menu = QMenu(line)
        menu.addAction("Sort by index ascending",
                       lambda: self.table_widget.model().sort_index())
        menu.addAction("Sort by index descending",
                       lambda: self.table_widget.model().sort_index(ascending=False))
        menu.move(a0.globalPos())
        menu.show()

    def _input_table_key_pressed(self, key_event) -> None:
        """
        Handle key press events in the input table, such as deleting rows.

        Parameters
        ----------
        key_event : QKeyEvent
            The key event triggering the function.
        """
        if (key_event.key() == Qt.Key.Key_Delete
                and self.mw.ui.input_table.selectionModel().currentIndex().row() > -1
                and len(self.mw.ui.input_table.selectionModel().selectedIndexes())):
            command = CommandDeleteInputSpectrum(None, self.parent,
                                                 "Delete files",
                                                 **{'table': self})
            self.parent.undo_stack.push(command)


class CommandChangeGroupCell(UndoCommand):
    """
    Command for changing the group number of a row in the input table.

    Parameters
    ----------
    data : tuple[QModelIndex, int]
        A tuple containing the index and new_value.
    parent : Context
        The parent object.
    text : str
        Description of the command.
    """

    def __init__(self, data: tuple[QModelIndex, int], parent, text: str, *args,
                 **kwargs) -> None:
        """
        Initialize the command.

        Parameters
        ----------
        data : tuple
            The index and new group number.
        parent : Context
            The parent object.
        text : str
            Description of the command.
        """
        self.table = kwargs.pop('table')
        super().__init__(data, parent, text, *args, **kwargs)
        self.index, self.new_value = data
        self.table_widget = self.table.table_widget
        self.previous_value = int(self.table.previous_group_of_item)

    def redo_special(self):
        """
        Redo the command, applying the new group number.
        """
        self.mw.ui.input_table.model().change_cell_data(self.index.row(), self.index.column(),
                                                        self.new_value)
        filename = self.mw.ui.input_table.model().index_by_row(self.index.row())
        self.update_datasets(filename, self.new_value)

    def undo_special(self):
        """
        Undo the command, reverting to the previous group number.
        """
        self.mw.ui.input_table.model().setData(self.index.row(), self.index.column(),
                                               self.previous_value)
        filename = self.mw.ui.input_table.model().index_by_row(self.index.row())
        self.update_datasets(filename, self.previous_value)

    def stop_special(self) -> None:
        """
        Handle any special cleanup after stopping the command.
        """
        prev_stage = self.mw.ui.drag_widget.get_previous_stage(self)
        if prev_stage is not None and prev_stage.data:
            self.parent.preprocessing.stages.av_data.update_averaged(self.mw, prev_stage.data)
        self.parent.preprocessing.update_plot_item(self.ui.mw.drag_widget.get_current_widget_name())
        self.parent.set_modified()

    def update_datasets(self, filename, value):
        """
        Update datasets with the new group number.

        Parameters
        ----------
        filename : str
            The filename of the dataset to update.
        value : int
            The new group number.
        """
        for model in (self.mw.ui.smoothed_dataset_table_view.model(),
                      self.mw.ui.baselined_dataset_table_view.model(),
                      self.mw.ui.deconvoluted_dataset_table_view.model()):
            if model.rowCount() > 0:
                idx_sm = model.idx_by_column_value('Filename', filename)
                model.set_cell_data_by_idx_col_name(idx_sm, 'Class', value)


class CommandDeleteInputSpectrum(UndoCommand):
    """
    Command for deleting rows in the input table, including managing
    associated data and plots.

    Parameters
    ----------
    data : None
    parent : Context
        The parent object.
    text : str
        Description of the command.
    """

    def __init__(self, data: None, parent, text: str, *args,
                 **kwargs) -> None:
        """
        Initialize the command.

        Parameters
        ----------
        data : None
        parent : Context
            The parent object.
        text : str
            Description of the command.
        """
        self.table = kwargs.pop('table').table_widget.model()
        super().__init__(data, parent, text, *args, **kwargs)
        self.selected_indexes = self.mw.ui.input_table.selectionModel().selectedIndexes()
        self.old_data_stores = None
        stages = self.parent.preprocessing.stages
        decomp_data = self.parent.decomposition.data
        self.data_stores = (stages.input_data.data, stages.convert_data.data, stages.cut_data.data,
                            stages.normalized_data.data, stages.smoothed_data.data,
                            stages.bl_data.data,
                            stages.trim_data.data, decomp_data.report_result,
                            decomp_data.sigma3, decomp_data.params_stderr)
        self.prepare()
        self.d = self.old_data_stores[0]
        values_list = list(self.d.keys())
        self.df = self.mw.ui.fit_params_table.model().dataframe.query('filename in @values_list')

        self.dfs = {'smoothed': self.mw.ui.smoothed_dataset_table_view.model().dataframe,
                    'baselined': self.mw.ui.baselined_dataset_table_view.model().dataframe,
                    'deconvoluted': self.mw.ui.deconvoluted_dataset_table_view.model().dataframe,
                    'input': self.table.dataframe}

    def prepare(self) -> None:
        """
        Prepare data for deletion.
        """
        self.old_data_stores = []
        for d in self.data_stores:
            old_data = {}
            for i in self.selected_indexes:
                row_data = self.table.row_data(i.row())
                filename = row_data.name
                if not d or filename not in d:
                    continue
                old_data[filename] = d[filename]
            self.old_data_stores.append(old_data)

    def redo_special(self):
        """
        Redo the command, applying the deletion.
        """
        self.table.delete_rows(self.d.keys())
        self.mw.ui.dec_table.model().delete_rows(self.d.keys())
        self.mw.ui.fit_params_table.model().delete_rows_by_filenames(self.d.keys())
        if self.d:
            for k in self.d:
                self._del_row(k)
        self.update_datasets()

    def undo_special(self):
        """
        Undo the command, restoring the deleted rows.
        """
        self.table.set_dataframe(self.dfs['input'])
        self.mw.ui.dec_table.model().concat_deconv_table(self.d.keys())
        self.mw.ui.fit_params_table.model().concat_df(self.df)
        if self.d:
            for k in self.d:
                self._add_row(k)
        for model, df in zip((self.mw.ui.smoothed_dataset_table_view.model(),
                              self.mw.ui.baselined_dataset_table_view.model(),
                              self.mw.ui.deconvoluted_dataset_table_view.model()),
                             (self.dfs['smoothed'], self.dfs['baselined'],
                              self.dfs['deconvoluted'])):
            if model.rowCount() > 0:
                model.set_dataframe(df)
                model.reset_index()
                model.sort_index()

    def stop_special(self) -> None:
        """
        Handle any special cleanup after stopping the command.
        """
        self.mw.decide_vertical_scroll_bar_visible()
        prev_stage = self.mw.ui.drag_widget.get_previous_stage(self)
        if prev_stage is not None and prev_stage.data:
            self.parent.preprocessing.stages.av_data.update_averaged(self.mw, prev_stage.data)
        self.parent.preprocessing.update_plot_item(self.mw.ui.drag_widget.get_current_widget_name())
        self.parent.set_modified()

    def update_datasets(self):
        """
        Update datasets after deletion or restoration.
        """
        for model in (self.mw.ui.smoothed_dataset_table_view.model(),
                      self.mw.ui.baselined_dataset_table_view.model(),
                      self.mw.ui.deconvoluted_dataset_table_view.model()):
            if model.rowCount() > 0:
                model.delete_rows_by_filenames(self.d.keys())
                model.reset_index()
                model.sort_index()

    def _del_row(self, key: str) -> None:
        """
        Delete a row from the datasets.

        Parameters
        ----------
        key : str
            The key identifying the row to delete.
        """
        for d in self.data_stores:
            if key in d:
                del d[key]

    def _add_row(self, key: str) -> None:
        """
        Add a row back to the datasets.

        Parameters
        ----------
        key : str
            The key identifying the row to add.
        """
        for data, old_data in zip(self.data_stores, self.old_data_stores):
            if key in old_data:
                data[key] = old_data[key]
