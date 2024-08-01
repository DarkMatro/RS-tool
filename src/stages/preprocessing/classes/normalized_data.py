# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for normalizing spectral data in the preprocessing workflow.

This module defines the `NormalizedData` class, which manages the normalization of
spectral data using various methods such as EMSC. It includes functionality to set
up the user interface, reset data, load and save data, and handle normalization
processes asynchronously.

Classes
-------
NormalizedData
    Manages the normalization stage of the preprocessing workflow.
CommandNormalize
    Handles undo and redo operations for the normalization process.
"""

from copy import deepcopy, copy
from typing import ItemsView

import numpy as np
from asyncqtpy import asyncSlot
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QMainWindow

from src.backend.undo_stack import UndoCommand
from src.data.collections import ObservableDict
from src.data.config import get_config
from src.data.default_values import normalize_methods
from src.data.get_data import get_parent
from src.stages.preprocessing.classes.stages import PreprocessingStage
from src.stages.preprocessing.functions.normalization import get_emsc_average_spectrum
from src.ui.ui_normalize_widget import Ui_NormalizeForm


class NormalizedData(PreprocessingStage):
    """
    Manages the normalization stage of the preprocessing workflow.

    Parameters
    ----------
    parent : Preprocessing
        The parent preprocessing object.

    Attributes
    ----------
    ui : Ui_NormalizeForm
        The user interface form.
    normalize_methods : dict
        Dictionary of available normalization methods.
    current_method : str
        The current normalization method being used.
    name : str
        The name of the stage.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the NormalizedData object.

        Parameters
        ----------
        parent : Preprocessing
            The parent preprocessing object.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(parent, *args, **kwargs)
        self.ui = None
        self.normalize_methods = normalize_methods()
        self.current_method = ''
        self.name = 'NormalizedData'

    def set_ui(self, ui: Ui_NormalizeForm) -> None:
        """
        Set the user interface object.

        Parameters
        ----------
        ui : Ui_NormalizeForm
            The user interface form.
        """
        context = get_parent(self.parent, "Context")
        defaults = get_config('defaults')
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.activate_btn.clicked.connect(self.activate)
        self.ui.normalize_btn.clicked.connect(self.process_clicked)
        self.ui.emsc_pca_n_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'emsc_pca_n_spinBox')
        self.ui.normalizing_method_comboBox.currentTextChanged.connect(self.method_changed)
        self._init_normalizing_method_combo_box()
        self.ui.emsc_pca_n_spinBox.valueChanged.connect(context.set_modified)
        self.ui.normalizing_method_comboBox.setCurrentText(defaults['normalizing_method_comboBox'])
        self.ui.emsc_pca_n_spinBox.setVisible(False)

    def reset(self) -> None:
        """
        Reset class data to default values.
        """
        self.data.clear()
        defaults = get_config('defaults')
        self.ui.normalizing_method_comboBox.setCurrentText(defaults['normalizing_method_comboBox'])
        self.ui.emsc_pca_n_spinBox.setValue(defaults['emsc_pca_n_spinBox'])
        self.ui.emsc_pca_n_spinBox.setVisible(False)
        self.current_method = ''
        if self.parent.active_stage == self:
            self.parent.update_plot_item('NormalizedData')
        self.activate(True)

    def read(self, production_export: bool=False) -> dict:
        """
        Read and return the attributes' data.

        Returns
        -------
        dict
            Dictionary containing all class attributes data.
        """
        dt = {'emsc_pca_n_spinBox': self.ui.emsc_pca_n_spinBox.value(),
              'normalizing_method_comboBox': self.ui.normalizing_method_comboBox.currentText(),
              'current_method': self.current_method,
              'active': self.active}
        if not production_export:
            dt['data'] = self.data.get_data()
        return dt

    def load(self, db: dict) -> None:
        """
        Load attributes data from a dictionary.

        Parameters
        ----------
        db : dict
            Dictionary containing all class attributes data.
        """
        if 'data' in db:
            self.data.update(db['data'])
        self.ui.emsc_pca_n_spinBox.setValue(db['emsc_pca_n_spinBox'])
        self.ui.normalizing_method_comboBox.setCurrentText(db['normalizing_method_comboBox'])
        self.current_method = db['current_method']
        if 'active' in db:
            self.activate(db['active'])

    def reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Reset a field value to its default on a double-click event.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event object.
        field_id : str
            The ID of the field to reset.
        """
        if event.buttons() != Qt.MouseButton.MiddleButton:
            return
        value = get_config('defaults')[field_id]
        match field_id:
            case 'emsc_pca_n_spinBox':
                self.ui.emsc_pca_n_spinBox.setValue(value)
            case _:
                return

    def plot_items(self) -> ItemsView:
        """
        Get the data for plotting.

        Returns
        -------
        ItemsView
            View of the data items.
        """
        return self.data.items()

    def _init_normalizing_method_combo_box(self) -> None:
        """
        Initialize the combo box with available normalization methods.
        """
        self.ui.normalizing_method_comboBox.addItems(self.normalize_methods.keys())

    def method_changed(self, current_text: str):
        """
        Handle the event when the normalization method is changed.

        Parameters
        ----------
        current_text : str
            The current text of the combo box.
        """
        context = get_parent(self.parent, "Context")
        context.set_modified()
        self.ui.emsc_pca_n_spinBox.setVisible(current_text == 'EMSC')

    @asyncSlot()
    async def process_clicked(self) -> None:
        """
        Handle the event when the normalize button is clicked.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if not prev_stage.data:
            mw.ui.statusBar.showMessage("No data to normalization")
            return
        await self._normalize(mw, prev_stage.data)

    @asyncSlot()
    async def _normalize(self, mw: QMainWindow, data: ObservableDict) -> None:
        """
        Perform the normalization process asynchronously.

        Parameters
        ----------
        mw : QMainWindow
            The main window object.
        data : ObservableDict
            The data to be normalized.
        """
        n_files = len(data)
        cfg = get_config("texty")["normalization"]
        context = get_parent(self.parent, "Context")
        mw.progress.open_progress(cfg, n_files)
        method = self.ui.normalizing_method_comboBox.currentText()
        func, n_limit = self.normalize_methods[method]
        args = []
        kwargs = {'n_files': n_files, 'n_limit': n_limit}
        if method == 'EMSC':
            if context.predict.is_production_project:
                np_y_axis = context.predict.y_axis_ref_emsc
            else:
                np_y_axis = get_emsc_average_spectrum(data.values())
                context.predict.y_axis_ref_emsc = np_y_axis
            args.extend([np_y_axis, self.ui.emsc_pca_n_spinBox.value()])
        result: list[tuple[str, np.ndarray]] = await mw.progress.run_in_executor(
            "normalization", func, data.items(), *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandNormalize(result, context, text="Normalize", **{'stage': self,
                                                                         'method': method})
        context.undo_stack.push(command)


class CommandNormalize(UndoCommand):
    """
    Handles undo and redo operations for the normalization process.

    Parameters
    ----------
    data : list of tuple
        List of tuples containing filenames and processed 2D arrays with normalized data.
    parent : Context
        The backend context object.
    text : str
        Description of the command.

    Attributes
    ----------
    data : list of tuple
        The new data to be applied.
    old_data : dict
        The previous data before normalization.
    method : str
        The current normalization method.
    method_old : str
        The previous normalization method.
    """

    def __init__(self, data: list[tuple[str, np.ndarray]],
                 parent, text: str, *args, **kwargs) -> None:
        """
        Initialize the CommandNormalize object.

        Parameters
        ----------
        data : list of tuple
            List of tuples containing filenames and processed 2D arrays with normalized data.
        parent : Context
            The backend context object.
        text : str
            Description of the command.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.stage = kwargs.pop('stage')
        self.method = kwargs.pop('method')
        super().__init__(data, parent, text, *args, **kwargs)
        self.data = data
        self.old_data = deepcopy(self.stage.data.get_data())
        self.method_old = copy(self.stage.current_method)

    def redo_special(self):
        """
        Apply the normalization data.
        """
        self.stage.data.clear()
        new_data = dict(self.data)
        self.stage.data.update(new_data)
        self.stage.current_method = self.method

    def undo_special(self):
        """
        Revert to the previous normalization data.
        """
        self.stage.data.clear()
        self.stage.data.update(self.old_data)
        self.stage.current_method = self.method_old

    def stop_special(self) -> None:
        """
        Update UI elements after stopping the command.
        """
        self.parent.preprocessing.update_plot_item("NormalizedData")
