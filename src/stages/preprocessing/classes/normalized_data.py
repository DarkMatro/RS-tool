from copy import deepcopy, copy

import numpy as np
from qtpy.QtWidgets import QMainWindow
from asyncqtpy import asyncSlot

from src.stages.preprocessing.functions.normalization import get_emsc_average_spectrum
from src.backend.undo_stack import UndoCommand
from src.data.collections import ObservableDict
from src.stages.preprocessing.classes.stages import PreprocessingStage
from src.ui.ui_normalize_widget import Ui_NormalizeForm
from src.data.get_data import get_parent
from typing import ItemsView
from qtpy.QtGui import QMouseEvent
from qtpy.QtCore import Qt
from src.data.config import get_config
from src.data.default_values import normalize_methods


class NormalizedData(PreprocessingStage):
    """
    Normalize spectrum from previous stage

    Parameters
    -------
    parent: Preprocessing
        class Preprocessing

    Attributes
    -------
    ui: object
        user interface form
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.ui = None
        self.normalize_methods = normalize_methods()
        self.current_method = ''
        self.name = 'NormalizedData'

    def set_ui(self, ui: Ui_NormalizeForm) -> None:
        """
        Set user interface object

        Parameters
        -------
        ui: Ui_NormalizeForm
            widget
        """
        context = get_parent(self.parent, "Context")
        defaults = get_config('defaults')
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.activate_btn.clicked.connect(self.activate)
        self.ui.normalize_btn.clicked.connect(self._normalize_clicked)
        self.ui.emsc_pca_n_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'emsc_pca_n_spinBox')
        self.ui.normalizing_method_comboBox.currentTextChanged.connect(self.method_changed)
        self._init_normalizing_method_combo_box()
        self.ui.emsc_pca_n_spinBox.valueChanged.connect(context.set_modified)
        self.ui.normalizing_method_comboBox.setCurrentText(defaults['normalizing_method_comboBox'])
        self.ui.emsc_pca_n_spinBox.setVisible(False)

    def reset(self) -> None:
        """
        Reset class data.
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

    def read(self) -> dict:
        """
        Read attributes data.

        Returns
        -------
        dt: dict
            all class attributes data
        """
        dt = {"data": self.data.get_data(),
              'emsc_pca_n_spinBox': self.ui.emsc_pca_n_spinBox.value(),
              'normalizing_method_comboBox': self.ui.normalizing_method_comboBox.currentText(),
              'current_method': self.current_method,
              'active': self.active}
        return dt

    def load(self, db: dict) -> None:
        """
        Load attributes data from file.

        Parameters
        -------
        db: dict
            all class attributes data
        """
        self.data.update(db['data'])
        self.ui.emsc_pca_n_spinBox.setValue(db['emsc_pca_n_spinBox'])
        self.ui.normalizing_method_comboBox.setCurrentText(db['normalizing_method_comboBox'])
        self.current_method = db['current_method']
        if 'active' in db:
            self.activate(db['active'])

    def reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Change field value to default on double click by MiddleButton.

        Parameters
        -------
        event: QMouseEvent

        field_id: str
            name of field
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
        Returns data for plotting
        """
        return self.data.items()

    def _init_normalizing_method_combo_box(self) -> None:
        self.ui.normalizing_method_comboBox.addItems(self.normalize_methods.keys())

    def method_changed(self, current_text: str):
        context = get_parent(self.parent, "Context")
        context.set_modified()
        self.ui.emsc_pca_n_spinBox.setVisible(current_text == 'EMSC')

    @asyncSlot()
    async def _normalize_clicked(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.drag_widget.get_previous_stage(self)
        if not prev_stage.data:
            mw.ui.statusBar.showMessage("No data to normalization")
            return
        await self._normalize(mw, prev_stage.data)

    @asyncSlot()
    async def _normalize(self, mw: QMainWindow, data: ObservableDict) -> None:
        n_files = len(data)
        cfg = get_config("texty")["normalization"]

        mw.progress.open_progress(cfg, n_files)
        method = self.ui.normalizing_method_comboBox.currentText()
        func, n_limit = self.normalize_methods[method]
        args = []
        kwargs = {'n_files': n_files, 'n_limit': n_limit}
        if method == 'EMSC':
            if mw.predict_logic.is_production_project:
                np_y_axis = mw.predict_logic.y_axis_ref_EMSC
            else:
                np_y_axis = get_emsc_average_spectrum(data.values())
                mw.predict_logic.y_axis_ref_EMSC = np_y_axis
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
    Change data for normalization stage.

    Parameters
    -------
    data: list[tuple[str, ndarray]]
        filename: str
            as input
        array: np.ndarray
            processed 2D array with normalized wavelengths and intensities
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: list[tuple[str, np.ndarray]],
                 parent, text: str, *args, **kwargs) -> None:
        self.stage = kwargs.pop('stage')
        self.method = kwargs.pop('method')
        super().__init__(data, parent, text, *args, **kwargs)
        self.data = data
        self.old_data = deepcopy(self.stage.data.get_data())
        self.method_old = copy(self.stage.current_method)

    def redo_special(self):
        """
        Update data
        """
        self.stage.data.clear()
        new_data = dict(self.data)
        self.stage.data.update(new_data)
        self.stage.current_method = self.method

    def undo_special(self):
        """
        Undo data
        """
        self.stage.data.clear()
        self.stage.data.update(self.old_data)
        self.stage.current_method = self.method_old

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.preprocessing.update_plot_item("NormalizedData")
