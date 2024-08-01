# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functionalities for cutting and trimming data as part of the preprocessing
stages.
It includes the CutData class for handling the cut/trim operations and CommandCut for managing the
undo stack.

Classes
-------
CutData
    Manages the cut and trim operations for data preprocessing.
CommandCut
    Handles the undo and redo operations for the cut stage.

Functions
---------
data_changed(data)
    Updates the range for the trim stage.
set_ui(ui)
    Sets the user interface object.
init_linear_region()
    Initializes the linear region for selection.
lr_cm_region_changed()
    Updates the linear region based on cm range changes.
set_ranges_default()
    Sets default ranges for the trim stage.
reset()
    Resets the class data.
read()
    Reads and returns the attributes data.
load(db)
    Loads attributes data from a dictionary.
reset_field(event, field_id)
    Resets the field value to default on middle mouse button double click.
plot_items()
    Returns data for plotting.
get_range_info()
    Returns range information for the linear region.
cm_range_start_change_event(new_value)
    Handles the event for changing the start of the cm range.
cm_range_end_change_event(new_value)
    Handles the event for changing the end of the cm range.
_update_range_cm()
    Sets the left bound for the cut range.
_update_trim_range()
    Sets the left and right bounds for the trim range.
_cut_clicked()
    Handles the cut button click event.
_cut(mw, data)
    Performs the cut operation.
redo_special()
    Updates data and input table columns.
undo_special()
    Undoes the data and input table columns.
stop_special()
    Updates UI elements.
"""

from copy import deepcopy
from os import environ
from typing import ItemsView

import numpy as np
from asyncqtpy import asyncSlot
from pyqtgraph import LinearRegionItem
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QMouseEvent
from qtpy.QtWidgets import QMainWindow

from src.backend.undo_stack import UndoCommand
from src.data.collections import ObservableDict
from src.data.config import get_config
from src.data.get_data import get_parent
from src.data.work_with_arrays import find_nearest, find_nearest_by_idx
from src.stages.preprocessing.classes.stages import PreprocessingStage
from src.stages.preprocessing.functions.cut_trim import (find_fluorescence_beginning, cut_spectrum,
                                                         find_first_right_local_minimum,
                                                         find_first_left_local_minimum)
from src.ui.ui_cut_widget import Ui_CutForm


class CutData(PreprocessingStage):
    """
    Cut data from the previous stage.

    Parameters
    ----------
    parent : Preprocessing
        Instance of the Preprocessing class.
    is_trim : bool, optional
        Indicates if the operation is trim (default is False).

    Attributes
    ----------
    ui : object
        User interface form.
    name : str
        Name of the stage (TrimData or CutData).
    data : ObservableDict
        Data to be processed.
    """

    def __init__(self, parent, is_trim: bool = False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.linear_region = None
        self.ui = None
        self.name = 'TrimData' if is_trim else 'CutData'
        self.data.on_change(self.data_changed)

    def data_changed(self, data: ObservableDict):
        """
        Change range for trim stage.

        Parameters
        ----------
        data : ObservableDict
            Data to be processed.
        """
        if not data:
            return
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        if self.name == 'CutData':
            range_start_spin_box = self.parent.stages.trim_data.ui.cm_range_start
            range_end_spin_box = self.parent.stages.trim_data.ui.cm_range_end
            lr = self.parent.stages.trim_data.linear_region
        else:
            range_start_spin_box = mw.ui.interval_start_dsb
            range_end_spin_box = mw.ui.interval_end_dsb
            lr = context.decomposition.plotting.linear_region
        min_cm = next(iter(data.values()))[:, 0][0]
        max_cm = next(iter(data.values()))[:, 0][-1]
        for v in data.values():
            min_cm = max(min_cm, v[:, 0][0])
            max_cm = min(max_cm, v[:, 0][-1])
        range_start_spin_box.setMinimum(min_cm)
        range_start_spin_box.setMaximum(max_cm)
        range_end_spin_box.setMinimum(min_cm)
        range_end_spin_box.setMaximum(max_cm)

        current_value_start = range_start_spin_box.value()
        current_value_end = range_end_spin_box.value()
        if current_value_start < min_cm or current_value_start > max_cm:
            range_start_spin_box.setValue(min_cm)
        if current_value_end < min_cm or current_value_end > max_cm:
            range_start_spin_box.setValue(max_cm)
        lr.setBounds((min_cm, max_cm))

    def set_ui(self, ui: Ui_CutForm) -> None:
        """
        Set user interface object.

        Parameters
        ----------
        ui : Ui_CutForm
            User interface form.
        """
        context = get_parent(self.parent, "Context")
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.cut_btn.clicked.connect(self.cut_clicked)

        self.ui.neg_grad_factor_spin_box.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'neg_grad_factor_spin_box')
        self.ui.cm_range_start.valueChanged.connect(self.cm_range_start_change_event)
        self.ui.cm_range_end.valueChanged.connect(self.cm_range_end_change_event)
        self.ui.neg_grad_factor_spin_box.valueChanged.connect(context.set_modified)
        self.init_linear_region()
        if self.name == 'TrimData':
            self.ui.neg_grad_factor_spin_box.setVisible(False)
            self.ui.cm_range_start.mouseDoubleClickEvent = lambda event: \
                self.reset_field(event, 'trim_range_start')
            self.ui.cm_range_end.mouseDoubleClickEvent = lambda event: \
                self.reset_field(event, 'trim_range_end')
            self.ui.updateRangebtn.clicked.connect(self._update_trim_range)
        else:
            self.ui.cm_range_start.mouseDoubleClickEvent = lambda event: \
                self.reset_field(event, 'cm_range_start')
            self.ui.cm_range_end.mouseDoubleClickEvent = lambda event: \
                self.reset_field(event, 'cm_range_end')
            self.ui.updateRangebtn.clicked.connect(self._update_range_cm)
        self.set_ranges_default()

    def init_linear_region(self):
        """
        Initialize the linear region for selection.
        """
        cfg = get_config('plots')['preproc']
        mw = get_parent(self.parent, "MainWindow")
        color_for_lr = QColor(environ["secondaryDarkColor"])
        color_for_lr.setAlpha(cfg['linear_region_alpha'])
        color_for_lr_hover = QColor(environ["secondaryDarkColor"])
        color_for_lr_hover.setAlpha(cfg['linear_region_hover_alpha'])
        start, end, start_min, end_max = self.get_range_info()
        self.linear_region = LinearRegionItem(
            [start, end], bounds=[start_min, end_max], swapMode="push",
        )
        self.linear_region.setBrush(color_for_lr)
        self.linear_region.setHoverBrush(color_for_lr_hover)
        self.linear_region.sigRegionChangeFinished.connect(
            self.lr_cm_region_changed
        )
        self.linear_region.setMovable(not mw.ui.lr_movableBtn.isChecked())

    def lr_cm_region_changed(self) -> None:
        """
        Update the linear region based on cm range changes.
        """
        start, end = self.linear_region.getRegion()
        self.ui.cm_range_start.setValue(start)
        self.ui.cm_range_end.setValue(end)

    def set_ranges_default(self) -> None:
        """
        Set default ranges for the trim stage.
        """
        defaults = get_config('defaults')
        if self.name == 'TrimData':
            self.ui.cm_range_start.setValue(defaults['trim_range_start'])
            self.ui.cm_range_end.setValue(defaults['trim_range_end'])
        else:
            self.ui.cm_range_start.setValue(defaults['cm_range_start'])
            self.ui.cm_range_end.setValue(defaults['cm_range_end'])

    def reset(self) -> None:
        """
        Reset the class data.
        """
        self.data.clear()
        defaults = get_config('defaults')
        self.ui.neg_grad_factor_spin_box.setValue(defaults['neg_grad_factor_spin_box'])
        self.set_ranges_default()
        if self.parent.active_stage == self:
            self.parent.update_plot_item(self.name)

    def read(self, production_export: bool=False) -> dict:
        """
        Read and return the attributes' data.

        Returns
        -------
        dict
            Dictionary containing the class attributes' data.
        """
        dt = {'neg_grad_factor_spin_box': self.ui.neg_grad_factor_spin_box.value(),
              'cm_range_start': self.ui.cm_range_start.value(),
              'cm_range_end': self.ui.cm_range_end.value()}
        if not production_export:
            dt['data'] = self.data.get_data()
        return dt

    def load(self, db: dict) -> None:
        """
        Load attributes data from a dictionary.

        Parameters
        ----------
        db : dict
            Dictionary containing the class attributes' data.
        """
        if 'data' in db:
            self.data.update(db['data'])
        self.ui.neg_grad_factor_spin_box.setValue(db['neg_grad_factor_spin_box'])
        self.ui.cm_range_start.setValue(db['cm_range_start'])
        self.ui.cm_range_end.setValue(db['cm_range_end'])

    def reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Change field value to default on double click by MiddleButton.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event.
        field_id : str
            Name of the field.
        """
        if event.buttons() != Qt.MouseButton.MiddleButton:
            return
        value = get_config('defaults')[field_id]
        match field_id:
            case 'neg_grad_factor_spin_box':
                self.ui.neg_grad_factor_spin_box.setValue(value)
            case 'cm_range_start':
                self.ui.cm_range_start.setValue(value)
            case 'cm_range_end':
                self.ui.cm_range_end.setValue(value)
            case 'trim_range_start':
                self.ui.cm_range_start.setValue(value)
            case 'trim_range_end':
                self.ui.cm_range_end.setValue(value)
            case _:
                return

    def plot_items(self) -> ItemsView:
        """
        Return data for plotting.

        Returns
        -------
        ItemsView
            Data for plotting.
        """
        if len(self.data) > 0:
            return self.data.items()

        mw = get_parent(self.parent, "MainWindow")
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        return prev_stage.data.items()

    def get_range_info(self) -> tuple:
        """
        Return range information for the linear region.

        Returns
        -------
        tuple
            Information about the range.
        """
        return (self.ui.cm_range_start.value(), self.ui.cm_range_end.value(),
                self.ui.cm_range_start.minimum(), self.ui.cm_range_end.maximum())

    def cm_range_start_change_event(self, new_value: float) -> None:
        """
        Handle the event for changing the start of the cm range.

        Parameters
        ----------
        new_value : float
            The new start value for the cm range.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        context.set_modified()
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if prev_stage is None:
            return
        if prev_stage.data:
            x_axis = next(iter(prev_stage.data.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.cm_range_start.setValue(new_value)
        if new_value >= self.ui.cm_range_end.value():
            self.ui.cm_range_start.setValue(self.ui.cm_range_start.minimum())
        self.linear_region.setRegion((self.ui.cm_range_start.value(), self.ui.cm_range_end.value()))

    def cm_range_end_change_event(self, new_value: float) -> None:
        """
        Handle the event for changing the end of the cm range.

        Parameters
        ----------
        new_value : float
            The new end value for the cm range.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        context.set_modified()
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if prev_stage is None:
            return
        if prev_stage.data:
            x_axis = next(iter(prev_stage.data.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            self.ui.cm_range_end.setValue(new_value)
        if new_value <= self.ui.cm_range_start.value():
            self.ui.cm_range_end.setValue(self.ui.cm_range_end.maximum())
        self.linear_region.setRegion((self.ui.cm_range_start.value(), self.ui.cm_range_end.value()))

    @asyncSlot()
    async def _update_range_cm(self) -> None:
        """
        Set the left bound for the cut range.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if not prev_stage.data:
            mw.ui.statusBar.showMessage(
                "Range update failed because there are no any converted plot ", 15000
            )
            return
        n_files = len(prev_stage.data)
        cfg = get_config("texty")["update_range"]
        mw.progress.open_progress(cfg, n_files)

        x_axis = next(iter(prev_stage.data.values()))[:, 0]
        value_right = find_nearest(x_axis, self.ui.cm_range_end.value())
        self.ui.cm_range_end.setValue(value_right)
        factor = self.ui.neg_grad_factor_spin_box.value()
        args = [factor]
        kwargs = {'n_files': n_files}
        result = await mw.progress.run_in_executor(
            "update_range", find_fluorescence_beginning, prev_stage.data.items(), *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        idx = max(result)
        value_left = find_nearest_by_idx(x_axis, idx)
        self.ui.cm_range_start.setValue(value_left)
        mw.progress.time_start = None

    @asyncSlot()
    async def _update_trim_range(self) -> None:
        """
        Set the left and right bounds for the trim range.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if not prev_stage.data:
            mw.ui.statusBar.showMessage('Range update failed because there are no any data', 15000)
            return
        n_files = len(prev_stage.data)
        cfg = get_config("texty")["update_range"]
        mw.progress.open_progress(cfg, n_files)

        x_axis = next(iter(prev_stage.data.values()))[:, 0]
        result = await mw.progress.run_in_executor(
            "update_range", find_first_right_local_minimum, prev_stage.data.items()
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        idx = int(np.percentile(result, 0.95))
        value_right = x_axis[idx]
        self.ui.cm_range_end.setValue(value_right)

        result = await mw.progress.run_in_executor(
            "update_range", find_first_left_local_minimum, prev_stage.data.items()
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        idx = np.max(result)
        value_left = x_axis[idx]
        self.ui.cm_range_start.setValue(value_left)
        mw.progress.time_start = None

    @asyncSlot()
    async def cut_clicked(self) -> None:
        """
        Handle the cut button click event.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if not prev_stage.data:
            mw.ui.statusBar.showMessage("No data to cut")
            return
        x_axis = next(iter(prev_stage.data.values()))[:, 0]
        value_start = self.ui.cm_range_start.value()
        value_end = self.ui.cm_range_end.value()
        if round(value_start, 5) == round(x_axis[0], 5) \
                and round(value_end, 5) == round(x_axis[-1], 5):
            mw.ui.statusBar.showMessage("Cut range is equal to actual spectrum range. No need to "
                                        "cut.")
            return
        await self._cut(mw, prev_stage.data)

    @asyncSlot()
    async def _cut(self, mw: QMainWindow, data: ObservableDict) -> None:
        """
        Perform the cut operation.

        Parameters
        ----------
        mw : QMainWindow
            The main window instance.
        data : ObservableDict
            Data to be processed.
        """
        n_files = len(data)
        cfg = get_config("texty")["cut"]

        mw.progress.open_progress(cfg, n_files)
        args = [self.ui.cm_range_start.value(), self.ui.cm_range_end.value()]
        kwargs = {'n_files': n_files}
        result = await mw.progress.run_in_executor(
            "cut", cut_spectrum, data.items(), *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandCut(result, context, text="Cut range", **{'cut_stage': self})
        context.undo_stack.push(command)


class CommandCut(UndoCommand):
    """
    Change data for the cut stage.

    Parameters
    ----------
    data : list of tuple
        List of tuples containing filenames and processed data arrays.
    parent : Context
        Backend context class.
    text : str
        Description of the command.
    """

    def __init__(self, data: list[tuple[str, np.ndarray]],
                 parent, text: str, *args, **kwargs) -> None:
        self.cut_stage = kwargs.pop('cut_stage')
        super().__init__(data, parent, text, *args, **kwargs)
        self.data = data
        self.old_data = deepcopy(self.cut_stage.data.get_data())

    def redo_special(self):
        """
        Update data and input table columns.
        """
        self.cut_stage.data.clear()
        new_data = dict(self.data)
        self.cut_stage.data.update(new_data)

    def undo_special(self):
        """
        Undo data and input table columns.
        """
        self.cut_stage.data.clear()
        self.cut_stage.data.update(self.old_data)

    def stop_special(self) -> None:
        """
        Update UI elements.
        """
        self.parent.preprocessing.update_plot_item(self.cut_stage.name)
