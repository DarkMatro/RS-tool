# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module manages the decomposition lines table within the user interface, providing
functionality for adding, removing, updating, and interacting with decomposition curves and their
properties.

Classes
-------
TableDecompLines
CommandDeconvLineTypeChanged
CommandUpdateDeconvCurveStyle
CommandDeleteDeconvLines
CommandClearAllDeconvLines

Functions
---------
None

"""

from copy import deepcopy
from gc import get_objects

import numpy as np
from pandas import DataFrame
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHeaderView, QAbstractItemView, QLineEdit, QMenu

from qfluentwidgets import MessageBox
from src import get_parent, get_config, UndoCommand
from src.data.default_values import peak_shapes_params
from src.pandas_tables import PandasModelDeconvLinesTable, ComboDelegate
from src.stages.fitting.functions.plotting import deconvolution_data_items_by_idx, \
    update_curve_style
from src.widgets import CurvePropertiesWindow


class TableDecompLines:
    """
    Manages the decomposition lines table UI, including setting up the UI, handling user
    interactions, and updating the table and associated plots.

    Parameters
    ----------
    parent : object
        The parent object, typically the main application window.
    """
    def __init__(self, parent):
        """
        Initializes the TableDecompLines instance with the provided parent.

        Parameters
        ----------
        parent : object
            The parent object, typically the main application window.
        """
        self.parent = parent  # decomposition stage
        self.reset()
        self.set_ui()
        self._ascending = False

    def reset(self):
        """
        Resets the decomposition lines table, including clearing existing data and setting up the
        model.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        df = DataFrame(columns=["Legend", "Type", "Style"])
        model = PandasModelDeconvLinesTable(context, df, [])
        mw.ui.deconv_lines_table.setSortingEnabled(True)
        mw.ui.deconv_lines_table.setModel(model)
        peak_shape_names = get_config('fitting')['peak_shape_names']
        combobox_delegate = ComboDelegate(peak_shape_names)
        mw.ui.deconv_lines_table.setItemDelegateForColumn(1, combobox_delegate)
        mw.ui.deconv_lines_table.model().sigCheckedChanged.connect(self._show_hide_curve)
        combobox_delegate.sigLineTypeChanged.connect(self._curve_type_changed)
        mw.ui.deconv_lines_table.clicked.connect(self.deconv_lines_table_clicked)

    def set_ui(self):
        """
        Configures the UI for the decomposition lines table, including setting headers,
        selection behaviors, and connecting signals.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.deconv_lines_table.verticalHeader().setSectionsMovable(True)
        mw.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        mw.ui.deconv_lines_table.horizontalHeader().resizeSection(0, 110)
        mw.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        mw.ui.deconv_lines_table.horizontalHeader().resizeSection(1, 150)
        mw.ui.deconv_lines_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        mw.ui.deconv_lines_table.horizontalHeader().resizeSection(2, 150)
        mw.ui.deconv_lines_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        mw.ui.deconv_lines_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        mw.ui.deconv_lines_table.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        mw.ui.deconv_lines_table.setDragDropOverwriteMode(False)
        mw.ui.deconv_lines_table.horizontalHeader().sectionClicked.connect(
            self._deconv_lines_table_header_clicked
        )
        mw.ui.deconv_lines_table.keyPressEvent = self.deconv_lines_table_key_pressed
        mw.ui.deconv_lines_table.contextMenuEvent = self._deconv_lines_table_context_menu_event
        mw.ui.deconv_lines_table.clicked.connect(self._deconv_lines_table_clicked)

    def _show_hide_curve(self, idx: int, b: bool) -> None:
        """
        Shows or hides the curve corresponding to the specified index.

        Parameters
        ----------
        idx : int
            The index of the curve.
        b : bool
            If True, shows the curve; if False, hides the curve.
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        data_items = plot_item.listDataItems()
        items_matches = deconvolution_data_items_by_idx(idx, data_items)
        if items_matches is None:
            return
        curve, roi = items_matches
        if curve is None or roi is None:
            return
        self.parent.graph_drawing.redraw_curve_by_index(idx)
        curve.setVisible(b)
        roi.setVisible(b)
        self.parent.graph_drawing.draw_sum_curve()
        self.parent.graph_drawing.draw_residual_curve()
        plot_item.getViewBox().updateAutoRange()

    def _curve_type_changed(self, line_type_new: str, line_type_old: str, row: int) -> None:
        """
        Handles the event when the curve type is changed, updating the table and redrawing the
        curve.

        Parameters
        ----------
        line_type_new : str
            The new line type.
        line_type_old : str
            The old line type.
        row : int
            The row index in the table.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        idx = mw.ui.deconv_lines_table.model().row_data(row).name
        self.parent.switch_template()
        command = CommandDeconvLineTypeChanged((line_type_new, line_type_old, idx), context,
                                               text=f"Change line type for curve idx {idx}")
        context.undo_stack.push(command)

    def deconv_lines_table_clicked(self) -> None:
        """
        Handles the event when a row in the decomposition lines table is clicked, showing the curve
        properties window.
        """
        mw = get_parent(self.parent, "MainWindow")
        current_index = mw.ui.deconv_lines_table.selectionModel().currentIndex()
        current_column, current_row = current_index.column(), current_index.row()
        row_data = mw.ui.deconv_lines_table.model().row_data(current_row)
        idx, style = row_data.name, row_data["Style"]
        if current_column != 2:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == idx and obj.isVisible():
                return
        window_cp = CurvePropertiesWindow(mw, style, idx)
        window_cp.sigStyleChanged.connect(self._update_deconv_curve_style)
        window_cp.show()

    def _update_deconv_curve_style(self, style: dict, old_style: dict, index: int) -> None:
        """
        Updates the style of the specified curve.

        Parameters
        ----------
        style : dict
            The new style for the curve.
        old_style : dict
            The old style for the curve.
        index : int
            The index of the curve.
        """
        context = get_parent(self.parent, "Context")
        command = CommandUpdateDeconvCurveStyle((style, old_style, index), context,
                                                text=f"Update style for curve idx {index}")
        context.undo_stack.push(command)

    def _deconv_lines_table_header_clicked(self, idx: int):
        """
        Handles the event when a header in the decomposition lines table is clicked, sorting the
        table.

        Parameters
        ----------
        idx : int
            The index of the clicked header.
        """
        mw = get_parent(self.parent, "MainWindow")
        df = mw.ui.deconv_lines_table.model().dataframe
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending = not self._ascending
        if current_name != "Style":
            mw.ui.deconv_lines_table.model().sort_values(current_name, self._ascending)
        self.parent.graph_drawing.deselect_selected_line()

    def deconv_lines_table_key_pressed(self, key_event) -> None:
        """
        Handles key press events for the decomposition lines table, enabling deletion of selected
        lines.

        Parameters
        ----------
        key_event : QKeyEvent
            The key event object.
        """
        mw = get_parent(self.parent, "MainWindow")
        if (key_event.key() == Qt.Key.Key_Delete
                and mw.ui.deconv_lines_table.selectionModel().currentIndex().row() > -1
                and len(mw.ui.deconv_lines_table.selectionModel().selectedIndexes())
                and self.parent.data.is_template
        ):
            context = get_parent(self.parent, "Context")
            command = CommandDeleteDeconvLines(None, context, text="Delete line", **{'table': self})
            context.undo_stack.push(command)

    def _deconv_lines_table_context_menu_event(self, a0) -> None:
        """
        Shows a context menu for the decomposition lines table, providing options for deleting
        lines or clearing the table.

        Parameters
        ----------
        a0 : QContextMenuEvent
            The context menu event object.
        """
        mw = get_parent(self.parent, "MainWindow")
        line = QLineEdit(mw)
        menu = QMenu(line)
        menu.addAction("Delete line", self.delete_line_clicked)
        menu.addAction("Clear table", self.clear_all_deconv_lines)
        menu.move(a0.globalPos())
        menu.show()

    def delete_line_clicked(self) -> None:
        """
        Deletes the selected line from the decomposition lines table.
        """
        mw = get_parent(self.parent, "MainWindow")
        if not self.parent.data.is_template:
            msg = MessageBox("Warning.", "Deleting lines is only possible in template mode.",
                             mw, {"Ok"})
            msg.setInformativeText("Press the Template button")
            msg.exec()
            return
        selected_indexes = mw.ui.deconv_lines_table.selectionModel().selectedIndexes()
        if len(selected_indexes) == 0:
            return
        context = get_parent(self.parent, "Context")
        command = CommandDeleteDeconvLines(None, context, text="Delete line", **{'table': self})
        context.undo_stack.push(command)

    def delete_deconv_curve(self, idx: int) -> None:
        """
        Deletes the specified decomposition curve from the plot.

        Parameters
        ----------
        idx : int
            The index of the curve to delete.
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        data_items = plot_item.listDataItems()
        items_matches = deconvolution_data_items_by_idx(idx, data_items)
        if items_matches is None:
            return
        curve, roi = items_matches
        plot_item.removeItem(roi)
        plot_item.removeItem(curve)
        plot_item.getViewBox().updateAutoRange()

    def clear_all_deconv_lines(self) -> None:
        """
        Clears all lines from the decomposition lines table and the associated plot.
        """
        if self.parent.graph_drawing.curve_select.timer_fill is None:
            return
        self.parent.graph_drawing.curve_select.timer_fill.stop()
        self.parent.graph_drawing.curve_select.timer_fill = None
        self.parent.graph_drawing.curve_select.curve_idx = None
        context = get_parent(self.parent, "Context")
        command = CommandClearAllDeconvLines(None, context, text="Remove all lines")
        context.undo_stack.push(command)

    def _deconv_lines_table_clicked(self) -> None:
        """
        Handles the event when the decomposition lines table is clicked, updating the visibility
        of rows and curves.
        """
        mw = get_parent(self.parent, "MainWindow")
        selected_indexes = mw.ui.deconv_lines_table.selectionModel().selectedIndexes()
        if len(selected_indexes) == 0:
            return
        self.parent.set_rows_visibility()
        row = selected_indexes[0].row()
        model = mw.ui.deconv_lines_table.model()
        idx = model.index_by_row(row)
        if (self.parent.graph_drawing.curve_select.curve_idx is not None
                and self.parent.graph_drawing.curve_select.curve_idx
                in model.dataframe.index):
            curve_style = model.cell_data_by_idx_col_name(
                self.parent.graph_drawing.curve_select.curve_idx, "Style"
            )
            plot_item = mw.ui.deconv_plot_widget.getPlotItem()
            data_items = plot_item.listDataItems()
            update_curve_style(self.parent.graph_drawing.curve_select.curve_idx, curve_style,
                               data_items)
        self.parent.graph_drawing.start_fill_timer(idx)


class CommandDeconvLineTypeChanged(UndoCommand):
    """
    Command to change the type of decomposition line, supporting undo and redo operations.

    Parameters
    ----------
    data : tuple[str, str, int]
        A tuple containing the new line type, old line type, and the index of the curve.
    parent : Context
        The parent context object.
    text : str
        The description of the command.
    """

    def __init__(self, data: tuple[str, str, int], parent, text: str, *args, **kwargs) -> None:
        """
        Initializes the CommandDeconvLineTypeChanged instance.

        Parameters
        ----------
        data : tuple[str, str, int]
            A tuple containing the new line type, old line type, and the index of the curve.
        parent : Context
            The parent context object.
        text : str
            The description of the command.
        """
        super().__init__(data, parent, text, *args, **kwargs)
        self._line_type_new, self._line_type_old, self._idx = data
        self.peak_shapes_params = peak_shapes_params()

    def redo_special(self):
        """
        Executes the redo operation for changing the line type.
        """
        self.mw.ui.deconv_lines_table.model().set_cell_data_by_idx_col_name(self._idx, 'Type',
                                                                            self._line_type_new)
        self.update_params_table(self._line_type_new, self._line_type_old)
        self.parent.decomposition.graph_drawing.redraw_curve(line_type=self._line_type_new,
                                                             idx=self._idx)

    def undo_special(self):
        """
        Executes the undo operation for reverting the line type change.
        """
        self.mw.ui.deconv_lines_table.model().set_cell_data_by_idx_col_name(self._idx, 'Type',
                                                                            self._line_type_old)
        self.update_params_table(self._line_type_old, self._line_type_new)
        self.parent.decomposition.graph_drawing.redraw_curve(line_type=self._line_type_old,
                                                             idx=self._idx)

    def stop_special(self) -> None:
        """
        Finalizes the command, updating the UI elements.
        """
        self.parent.decomposition.graph_drawing.draw_sum_curve()
        self.parent.decomposition.graph_drawing.draw_residual_curve()
        self.parent.set_modified()

    def update_params_table(self, line_type_new: str, line_type_old: str) -> None:
        """
        Updates the parameters table based on the new and old line types.

        Parameters
        ----------
        line_type_new : str
            The new line type.
        line_type_old : str
            The old line type.
        """
        params_old = self.peak_shapes_params[line_type_old]['add_params'] \
            if 'add_params' in self.peak_shapes_params[line_type_old] else None
        params_new = self.peak_shapes_params[line_type_new]['add_params'] \
            if 'add_params' in self.peak_shapes_params[line_type_new] else None
        if params_old == params_new or (params_old is None and params_new is None):
            return
        params_to_add = params_new if params_old is None \
            else [i for i in params_new if i not in params_old]
        params_to_dlt = params_old if params_new is None \
            else [i for i in params_old if i not in params_new]
        line_params = self.parent.decomposition.initial_peak_parameters(line_type_new)
        for i in params_to_add:
            self.mw.ui.fit_params_table.model().append_row(self._idx, i, line_params[i])
        for i in params_to_dlt:
            self.mw.ui.fit_params_table.model().delete_rows_multiindex(('', self._idx, i))


class CommandUpdateDeconvCurveStyle(UndoCommand):
    """
    Command to update the style of a decomposition curve, supporting undo and redo operations.

    Parameters
    ----------
    data : tuple[dict, dict, int]
        A tuple containing the new style, old style, and the index of the curve.
    parent : Context
        The parent context object.
    text : str
        The description of the command.
    """
    def __init__(self, data: tuple[dict, dict, int], parent, text: str, *args, **kwargs) -> None:
        """
        Initializes the CommandUpdateDeconvCurveStyle instance.

        Parameters
        ----------
        data : tuple[dict, dict, int]
            A tuple containing the new style, old style, and the index of the curve.
        parent : Context
            The parent context object.
        text : str
            The description of the command.
        """
        super().__init__(data, parent, text, *args, **kwargs)
        self._style, self._old_style, self._idx = data

    def redo_special(self):
        """
        Executes the redo operation for updating the curve style.
        """
        self.parent.decomposition.update_curve_style(self._idx, self._style)

    def undo_special(self):
        """
        Executes the undo operation for reverting the curve style update.
        """
        self.parent.decomposition.update_curve_style(self._idx, self._old_style)

    def stop_special(self) -> None:
        """
        Finalizes the command, updating the UI elements.
        """
        self.parent.set_modified()


class CommandDeleteDeconvLines(UndoCommand):
    """
    Command to delete decomposition lines from the table and plot, supporting undo and redo
    operations.

    Parameters
    ----------
    data : None
    parent : Context
        The parent context object.
    text : str
        The description of the command.
    """
    def __init__(self, data: None, parent, text: str, *args, **kwargs) -> None:
        """
        Initializes the CommandDeleteDeconvLines instance.

        Parameters
        ----------
        data : None
        parent : Context
            The parent context object.
        text : str
            The description of the command.
        """
        self.table = kwargs.pop('table')
        super().__init__(data, parent, text, *args, **kwargs)
        selected_row = self.mw.ui.deconv_lines_table.selectionModel().selectedIndexes()[0].row()
        self._selected_row = int(self.mw.ui.deconv_lines_table.model().row_data(selected_row).name)
        fit_lines_df = self.mw.ui.deconv_lines_table.model().query_result(
            f'index == {self._selected_row}')
        self._legend = fit_lines_df['Legend'][self._selected_row]
        self._line_type = fit_lines_df['Type'][self._selected_row]
        self._style = fit_lines_df['Style'][self._selected_row]
        self._line_params = self.parent.decomposition.current_line_parameters(self._selected_row)
        self._line_params['x_axis'] = self.parent.decomposition.graph_drawing.x_axis_for_line(
            self._line_params['x0'],
            self._line_params['dx']
        )
        self._df = self.mw.ui.fit_params_table.model().dataframe.query(
            f'line_index == {self._selected_row}')

    def redo_special(self):
        """
        Executes the redo operation for deleting the decomposition lines.
        """
        self.mw.ui.deconv_lines_table.model().delete_row(self._selected_row)
        self.table.delete_deconv_curve(self._selected_row)
        self.mw.ui.fit_params_table.model().delete_rows(self._selected_row)

    def undo_special(self):
        """
        Executes the undo operation for restoring the deleted decomposition lines.
        """
        self.mw.ui.deconv_lines_table.model().append_row(self._legend, self._line_type, self._style,
                                                         self._selected_row)
        self.parent.decomposition.graph_drawing.add_deconv_curve_to_plot(self._line_params,
                                                                         self._selected_row,
                                                                         self._style,
                                                                         self._line_type)
        self.mw.ui.fit_params_table.model().concat_df(self._df)

    def stop_special(self) -> None:
        """
        Finalizes the command, updating the UI elements.
        """
        self.parent.decomposition.graph_drawing.draw_sum_curve()
        self.parent.decomposition.graph_drawing.draw_residual_curve()
        self.parent.decomposition.graph_drawing.deselect_selected_line()
        self.parent.set_modified()


class CommandClearAllDeconvLines(UndoCommand):
    """
    Command to clear all decomposition lines from the table and plot, supporting undo and redo
    operations.

    Parameters
    ----------
    data : None
    parent : Context
        The parent context object.
    text : str
        The description of the command.
    """
    def __init__(self, data: None, parent, text: str, *args, **kwargs) -> None:
        """
        Initializes the CommandClearAllDeconvLines instance.

        Parameters
        ----------
        data : None
        parent : Context
            The parent context object.
        text : str
            The description of the command.
        """
        super().__init__(data, parent, text, *args, **kwargs)
        self._deconv_lines_table_df = self.mw.ui.deconv_lines_table.model().dataframe
        self._checked = self.mw.ui.deconv_lines_table.model().checked()
        self._deconv_params_table_df = self.mw.ui.fit_params_table.model().dataframe
        self.report_result_old = self.parent.decomposition.data.report_result.copy()
        self.sigma3 = deepcopy(self.parent.decomposition.data.sigma3)

    def redo_special(self):
        """
        Executes the redo operation for clearing all decomposition lines.
        """
        self.mw.ui.deconv_lines_table.model().clear_dataframe()
        self.mw.ui.fit_params_table.model().clear_dataframe()
        d = self.parent.decomposition
        d.remove_all_lines_from_plot()
        d.data.report_result.clear()
        self.mw.ui.report_text_edit.setText('')
        d.data.sigma3.clear()
        d.curves.sigma3_top.setData(x=np.array([0]), y=np.array([0]))
        d.curves.sigma3_bottom.setData(x=np.array([0]), y=np.array([0]))
        d.graph_drawing.update_sigma3_curves('')

    def undo_special(self):
        """
        Executes the undo operation for restoring the cleared decomposition lines.
        """
        self.mw.ui.deconv_lines_table.model().append_dataframe(self._deconv_lines_table_df)
        self.mw.ui.deconv_lines_table.model().set_checked(self._checked)
        self.mw.ui.fit_params_table.model().append_dataframe(self._deconv_params_table_df)
        d = self.parent.decomposition
        d.data.report_result = self.report_result_old
        d.show_current_report_result()
        d.data.sigma3 = self.sigma3
        filename = '' if d.data.is_template else d.data.current_spectrum_name
        d.graph_drawing.update_sigma3_curves(filename)
        d.graph_drawing.draw_all_curves()

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.decomposition.graph_drawing.draw_sum_curve()
        self.parent.decomposition.graph_drawing.draw_residual_curve()
        self.parent.set_modified()
