# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functionality for managing fit borders in a graphical user interface for
spectral analysis.

The main class, TableFitBorders, is responsible for initializing, resetting, and updating the fit
borders table. It handles user interactions, including adding and deleting borders, auto-searching
borders based on spectral data, and managing context menu events for the table.
"""

import numpy as np
from pandas import DataFrame
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHeaderView, QAbstractItemView, QLineEdit, QMenu
from scipy.signal import argrelmin

from src import get_parent
from src.pandas_tables import PandasModelFitIntervals, IntervalsTableDelegate


class TableFitBorders:
    """
    A class to manage the fit borders table for spectral analysis in a GUI.

    This class handles the initialization, resetting, and updating of the fit borders table.
    It manages user interactions, including adding and deleting borders, auto-searching borders
    based on spectral data, and managing context menu events for the table.

    Parameters
    ----------
    parent : object
        The parent object, typically the main window of the application.
    """
    def __init__(self, parent):
        """
        Initialize the TableFitBorders instance.

        This method sets up the initial state of the fit borders table and the user interface.

        Parameters
        ----------
        parent : object
            The parent object, typically the main window of the application.
        """
        self.parent = parent
        self.reset()
        self.set_ui()

    def reset(self):
        """
        Reset the fit borders table.

        This method clears the fit borders table and sets up a new empty model.
        """
        mw = get_parent(self.parent, "MainWindow")
        df = DataFrame(columns=["Border"])
        model = PandasModelFitIntervals(df)
        mw.ui.fit_borders_TableView.setModel(model)

    def set_ui(self):
        """
        Set up the user interface for the fit borders table.

        This method configures the table delegates, headers, selection behavior, and context menu.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        mw.ui.fit_borders_TableView.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        mw.ui.fit_borders_TableView.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        mw.ui.fit_borders_TableView.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        mw.ui.fit_borders_TableView.contextMenuEvent = (
            self._fit_intervals_table_context_menu_event
        )
        mw.ui.fit_borders_TableView.keyPressEvent = self._fit_intervals_table_key_pressed
        dsb_delegate = IntervalsTableDelegate(mw.ui.fit_borders_TableView, context)
        mw.ui.fit_borders_TableView.setItemDelegateForColumn(0, dsb_delegate)
        mw.ui.fit_borders_TableView.verticalHeader().setVisible(False)
        mw.ui.fit_borders_TableView.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

    def _fit_intervals_table_context_menu_event(self, a0) -> None:
        """
        Handle context menu event for the fit borders table.

        This method displays a context menu for adding and deleting borders, and for auto-searching
        borders.

        Parameters
        ----------
        a0 : QContextMenuEvent
            The context menu event.
        """
        mw = get_parent(self.parent, "MainWindow")
        line = QLineEdit(mw)
        menu = QMenu(line)
        menu.addAction("Add border", self._fit_intervals_table_add)
        menu.addAction("Delete selected", self._fit_intervals_table_delete)
        menu.addAction("Auto-search borders", self.auto_search_borders)
        menu.move(a0.globalPos())
        menu.show()

    def _fit_intervals_table_add(self) -> None:
        """
        Add a new border to the fit borders table.

        This method appends a new row to the fit borders table model and marks the context as
        modified.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        mw.ui.fit_borders_TableView.model().append_row()
        context.set_modified()

    def _fit_intervals_table_delete(self) -> None:
        """
        Delete the selected border from the fit borders table.

        This method deletes the selected row from the fit borders table model and marks the context
        as modified.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        selection = mw.ui.fit_borders_TableView.selectionModel()
        row = selection.currentIndex().row()
        interval_number = mw.ui.fit_borders_TableView.model().row_data(row).name
        mw.ui.fit_borders_TableView.model().delete_row(interval_number)
        context.set_modified()

    def _fit_intervals_table_key_pressed(self, key_event) -> None:
        """
        Handle key press event for the fit borders table.

        This method deletes the selected border if the Delete key is pressed.

        Parameters
        ----------
        key_event : QKeyEvent
            The key press event.
        """
        mw = get_parent(self.parent, "MainWindow")
        if (key_event.key() == Qt.Key.Key_Delete
                and mw.ui.fit_borders_TableView.selectionModel().currentIndex().row() > -1
                and len(mw.ui.fit_borders_TableView.selectionModel().selectedIndexes())
        ):
            self._fit_intervals_table_delete()

    def auto_search_borders(self) -> None:
        """
        Automatically search and add borders to the fit borders table.

        This method identifies local minima in the averaged spectrum data and adds them as borders
        to the fit borders table.
        """
        mw = get_parent(self.parent, "MainWindow")
        if self.parent.data.averaged_spectrum is None and self.parent.data.averaged_spectrum != []:
            return
        if len(self.parent.data.averaged_spectrum.shape) != 2:
            return
        x = self.parent.data.averaged_spectrum[:, 0]
        y = self.parent.data.averaged_spectrum[:, 1]
        minima_idx = argrelmin(y, order=int(y.shape[0] / 10))
        x_minima = np.round(x[minima_idx], 3)
        mw.ui.fit_borders_TableView.model().add_auto_found_borders(x_minima)
