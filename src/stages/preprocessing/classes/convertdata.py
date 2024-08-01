# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functionalities for converting data from input units (e.g., nm) to other units
(e.g., cm-1).
It includes the ConvertData class for managing the conversion operations and CommandConvert for
handling the undo stack.

Classes
-------
ConvertData
    Manages the conversion operations for data preprocessing.
CommandConvert
    Handles the undo and redo operations for the conversion stage.

Functions
---------
data_changed(data)
    Updates the range for the next stage after conversion.
set_ui(ui)
    Sets the user interface object.
reset()
    Resets the class data to default values.
read()
    Reads and returns the attributes data.
load(db)
    Loads attributes data from a dictionary.
reset_field(event, field_id)
    Resets the field value to default on middle mouse button double click.
plot_items()
    Returns data for plotting.
_convert_clicked()
    Handles the convert button click event.
_convert(mw, data)
    Performs the conversion operation.
redo_special()
    Updates data and input table columns.
undo_special()
    Undoes the data and input table columns.
stop_special()
    Updates UI elements.
change_column_rayleigh(i, undo)
    Updates the Rayleigh line and related columns in the input table.
"""

from copy import deepcopy

import numpy as np
from asyncqtpy import asyncSlot
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QMainWindow

from src import get_config
from src.backend.undo_stack import UndoCommand
from src.data.collections import ObservableDict
from src.data.get_data import get_parent
from src.data.work_with_arrays import nearest_idx
from src.stages.preprocessing.classes.stages import PreprocessingStage
from src.stages.preprocessing.functions.converting import convert
from src.ui.ui_convert_widget import Ui_ConvertForm


class ConvertData(PreprocessingStage):
    """
    Convert data from input data as nm. to cm-1 units.

    Parameters
    ----------
    parent : Preprocessing
        Instance of the Preprocessing class.

    Attributes
    ----------
    ui : object
        User interface form.
    name : str
        Name of the stage (ConvertData).
    data : ObservableDict
        Data to be processed.
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.ui = None
        self.data.on_change(self.data_changed)
        self.name = 'ConvertData'

    def data_changed(self, data: ObservableDict):
        """
        Change range for the next stage after conversion.

        Parameters
        ----------
        data : ObservableDict
            Data to be processed.
        """
        if not data:
            return
        mw = get_parent(self.parent, "MainWindow")
        next_stage = mw.ui.drag_widget.get_next_stage(self)
        min_cm = next(iter(data.values()))[:, 0][0]
        max_cm = next(iter(data.values()))[:, 0][-1]
        for v in data.values():
            min_cm = max(min_cm, v[:, 0][0])
            max_cm = min(max_cm, v[:, 0][-1])
        next_stage.ui.cm_range_start.setMinimum(min_cm)
        next_stage.ui.cm_range_start.setMaximum(max_cm)
        next_stage.ui.cm_range_end.setMinimum(min_cm)
        next_stage.ui.cm_range_end.setMaximum(max_cm)

        current_value_start = next_stage.ui.cm_range_start.value()
        current_value_end = next_stage.ui.cm_range_end.value()
        if current_value_start < min_cm or current_value_start > max_cm:
            next_stage.ui.cm_range_start.setValue(min_cm)
        if current_value_end < min_cm or current_value_end > max_cm:
            next_stage.ui.cm_range_start.setValue(max_cm)
        next_stage.linear_region.setBounds((min_cm, max_cm))

    def set_ui(self, ui: Ui_ConvertForm) -> None:
        """
        Set user interface object.

        Parameters
        ----------
        ui : Ui_ConvertForm
            User interface form.
        """
        context = get_parent(self.parent, "Context")
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.convert_btn.clicked.connect(self.convert_clicked)

        self.ui.laser_wl_spinbox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'laser_wl_spinbox')
        self.ui.max_ccd_value_spin_box.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'max_ccd_value_spin_box')
        self.ui.laser_wl_spinbox.valueChanged.connect(context.set_modified)
        self.ui.max_ccd_value_spin_box.valueChanged.connect(context.set_modified)

    def reset(self) -> None:
        """
        Reset class data to default values.
        """
        self.data.clear()
        defaults = get_config('defaults')
        self.ui.laser_wl_spinbox.setValue(defaults['laser_wl_spinbox'])
        self.ui.max_ccd_value_spin_box.setValue(defaults['max_ccd_value_spin_box'])
        if self.parent.active_stage == self:
            self.parent.update_plot_item('ConvertData')

    def read(self, production_export: bool=False) -> dict:
        """
        Read and return the attributes' data.

        Returns
        -------
        dict
            Dictionary containing the class attributes' data.
        """
        dt = {'laser_wl_spinbox': self.ui.laser_wl_spinbox.value(),
              'max_ccd_value_spin_box': self.ui.max_ccd_value_spin_box.value()}
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
        self.ui.laser_wl_spinbox.setValue(db['laser_wl_spinbox'])
        self.ui.max_ccd_value_spin_box.setValue(db['max_ccd_value_spin_box'])

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
            case 'laser_wl_spinbox':
                self.ui.laser_wl_spinbox.setValue(value)
            case 'max_ccd_value_spin_box':
                self.ui.max_ccd_value_spin_box.setValue(value)
            case _:
                return

    def plot_items(self) -> dict:
        """
        Returns data for plotting.

        Returns
        -------
        dict
            Data for plotting.
        """
        return self.data.items()

    @asyncSlot()
    async def convert_clicked(self) -> None:
        """
        Handle the convert button click event.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        input_data = mw.ui.drag_widget.get_previous_stage(self)
        if len(input_data.data) == 0:
            mw.ui.statusBar.showMessage("No data to convert")
            return
        if len(input_data.ranges) > 1:
            mw.ui.statusBar.showMessage("Spectra must be interpolated before convert")
            return
        await self._convert(mw, input_data.data)

    @asyncSlot()
    async def _convert(self, mw: QMainWindow, data: ObservableDict) -> None:
        """
        Perform the conversion operation.

        Parameters
        ----------
        mw : QMainWindow
            The main window instance.
        data : ObservableDict
            Data to be processed.
        """
        n_files = len(data)
        cfg = get_config("texty")["convert"]

        mw.progress.open_progress(cfg, n_files)
        x_axis = next(iter(data.values()))[:, 0]
        laser_nm = self.ui.laser_wl_spinbox.value()
        near_idx = nearest_idx(x_axis, laser_nm + 5.)
        max_ccd_value = self.ui.max_ccd_value_spin_box.value()
        args = [near_idx, laser_nm, max_ccd_value]
        kwargs = {'n_files': n_files}
        result = await mw.progress.run_in_executor(
            "convert", convert, data.items(), *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandConvert(result, context, text="Convert")
        context.undo_stack.push(command)


class CommandConvert(UndoCommand):
    """
    Change data for Convert stage and update related columns in the input plot.

    Parameters
    ----------
    data : list of tuple
        List of tuples containing filenames and processed data arrays.
        Each tuple contains:
            filename : str
                Name of the file.
            array : np.ndarray
                Processed 2D array with converted wavelengths and intensities.
            Rayleigh line : float
                The base wavelength.
            fwhm_cm : float
                FWHM of the laser peak.
            SNR : float
                Signal-to-noise ratio.
    parent : Context
        Backend context class.
    text : str
        Description of the command.
    """

    def __init__(self, data: list[tuple[str, np.ndarray, float, float, float]],
                 parent, text: str, *args, **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self.data = data
        self.convert_stage = self.parent.preprocessing.stages.convert_data
        self.old_data = deepcopy(self.convert_stage.data.get_data())
        self.input_table_model = self.mw.ui.input_table.model()
        self.change_list = []
        col_name = 'FWHM, cm\N{superscript minus}\N{superscript one}'
        for name, _, rayleigh_line, fwhm_cm, snr in data:
            row_data = self.input_table_model.row_data_by_filename(name)
            prev_rl = row_data['Rayleigh line, nm']
            old_fwhm = row_data[col_name]
            old_snr = row_data['SNR']
            self.change_list.append((name, rayleigh_line, prev_rl, fwhm_cm, old_fwhm, snr, old_snr))

    def redo_special(self):
        """
        Update data and input table columns.
        """
        self.convert_stage.data.clear()
        new_data = {k: v for k, v, _, _, _ in self.data}
        self.convert_stage.data.update(new_data)
        for i in self.change_list:
            self.change_column_rayleigh(i)

    def undo_special(self):
        """
        Undo data and input table columns.
        """
        self.convert_stage.data.clear()
        self.convert_stage.data.update(self.old_data)

        for i in self.change_list:
            self.change_column_rayleigh(i, True)

    def stop_special(self) -> None:
        """
        Update UI elements.
        """
        self.parent.preprocessing.update_plot_item("ConvertData")

    def change_column_rayleigh(self, i: tuple[str, float, float, float, float, float, float],
                               undo: bool = False) -> None:
        """
        Update the Rayleigh line and related columns in the input table.

        Parameters
        ----------
        i : tuple
            containing the name, new Rayleigh line, old Rayleigh line,
            FWHM in cm-1, old FWHM, SNR, and old SNR.
        undo : bool, optional
            Indicates if the operation is an undo (default is False).
        """
        name, new_rl, old_rl, fwhm_cm, old_fwhm, snr, old_snr = i
        col_name = 'FWHM, cm\N{superscript minus}\N{superscript one}'
        values = (old_rl, old_fwhm, old_snr) if undo else (new_rl, fwhm_cm, snr)
        self.input_table_model.change_cell_data(name, 'Rayleigh line, nm', values[0])
        self.input_table_model.change_cell_data(name, col_name, values[1])
        self.input_table_model.change_cell_data(name, 'SNR', values[2])
