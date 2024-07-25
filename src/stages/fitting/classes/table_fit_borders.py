import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHeaderView, QAbstractItemView, QLineEdit, QMenu
from pandas import DataFrame
from scipy.signal import argrelmin

from src import get_parent
from src.pandas_tables import PandasModelFitIntervals, IntervalsTableDelegate


class TableFitBorders:
    def __init__(self, parent):
        self.parent = parent
        self.reset()
        self.set_ui()

    def reset(self):
        mw = get_parent(self.parent, "MainWindow")
        df = DataFrame(columns=["Border"])
        model = PandasModelFitIntervals(df)
        mw.ui.fit_borders_TableView.setModel(model)

    def set_ui(self):
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
        mw = get_parent(self.parent, "MainWindow")
        line = QLineEdit(mw)
        menu = QMenu(line)
        menu.addAction("Add border", self._fit_intervals_table_add)
        menu.addAction("Delete selected", self._fit_intervals_table_delete)
        menu.addAction("Auto-search borders", self.auto_search_borders)
        menu.move(a0.globalPos())
        menu.show()

    def _fit_intervals_table_add(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        mw.ui.fit_borders_TableView.model().append_row()
        context.set_modified()

    def _fit_intervals_table_delete(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        selection = mw.ui.fit_borders_TableView.selectionModel()
        row = selection.currentIndex().row()
        interval_number = mw.ui.fit_borders_TableView.model().row_data(row).name
        mw.ui.fit_borders_TableView.model().delete_row(interval_number)
        context.set_modified()

    def _fit_intervals_table_key_pressed(self, key_event) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if (key_event.key() == Qt.Key.Key_Delete
                and mw.ui.fit_borders_TableView.selectionModel().currentIndex().row() > -1
                and len(mw.ui.fit_borders_TableView.selectionModel().selectedIndexes())
        ):
            self._fit_intervals_table_delete()

    def auto_search_borders(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if self.parent.data.averaged_spectrum is None and self.parent.data.averaged_spectrum != []:
            return
        x = self.parent.data.averaged_spectrum[:, 0]
        y = self.parent.data.averaged_spectrum[:, 1]
        minima_idx = argrelmin(y, order=int(y.shape[0] / 10))
        x_minima = np.round(x[minima_idx], 3)
        mw.ui.fit_borders_TableView.model().add_auto_found_borders(x_minima)
