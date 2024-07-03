from copy import deepcopy, copy

import numpy as np
from qtpy.QtWidgets import QMainWindow
from asyncqtpy import asyncSlot
import pandas as pd
from src.backend.undo_stack import UndoCommand
from src.data.collections import ObservableDict
from src.stages.preprocessing.classes.stages import PreprocessingStage
from src.ui.ui_smooth_widget import Ui_SmoothForm
from src.data.get_data import get_parent
from typing import ItemsView
from qtpy.QtGui import QMouseEvent
from qtpy.QtCore import Qt
from src.data.config import get_config
from src.data.default_values import smoothing_methods


class SmoothedData(PreprocessingStage):
    """
    Smoothed spectrum from previous stage

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
        self.smoothing_methods = smoothing_methods()
        self.current_method = ''
        self.name = 'SmoothedData'

    def set_ui(self, ui: Ui_SmoothForm) -> None:
        """
        Set user interface object

        Parameters
        -------
        ui: Ui_SmoothForm
            widget
        """
        context = get_parent(self.parent, "Context")
        defaults = get_config('defaults')
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.activate_btn.clicked.connect(self.activate)
        self.ui.smooth_btn.clicked.connect(self._smooth_clicked)

        self.ui.smoothing_method_comboBox.currentTextChanged.connect(self.method_changed)
        self.ui.smoothing_method_comboBox.setCurrentText(defaults['smoothing_method_comboBox'])

        self._init_smoothing_method_combo_box()
        self.ui.window_length_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'window_length_spinBox')
        self.ui.window_length_spinBox.valueChanged.connect(context.set_modified)
        self.ui.window_length_spinBox.setVisible(False)
        self.ui.whittaker_lambda_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'whittaker_lambda_spinBox')
        self.ui.whittaker_lambda_spinBox.valueChanged.connect(context.set_modified)
        self.ui.whittaker_lambda_spinBox.setVisible(False)
        self.ui.smooth_polyorder_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'smooth_polyorder_spinBox')
        self.ui.smooth_polyorder_spinBox.valueChanged.connect(context.set_modified)
        self.ui.smooth_polyorder_spinBox.setVisible(False)
        self.ui.sigma_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'sigma_spinBox')
        self.ui.sigma_spinBox.valueChanged.connect(context.set_modified)
        self.ui.sigma_spinBox.setVisible(True)
        self.ui.kaiser_beta_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'kaiser_beta_doubleSpinBox')
        self.ui.kaiser_beta_doubleSpinBox.valueChanged.connect(context.set_modified)
        self.ui.kaiser_beta_doubleSpinBox.setVisible(False)
        self.ui.emd_noise_modes_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'emd_noise_modes_spinBox')
        self.ui.emd_noise_modes_spinBox.valueChanged.connect(context.set_modified)
        self.ui.emd_noise_modes_spinBox.setVisible(False)
        self.ui.eemd_trials_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'eemd_trials_spinBox')
        self.ui.eemd_trials_spinBox.valueChanged.connect(context.set_modified)
        self.ui.eemd_trials_spinBox.setVisible(False)

    def reset(self) -> None:
        """
        Reset class data.
        """
        self.data.clear()
        defaults = get_config('defaults')
        self.ui.smoothing_method_comboBox.setCurrentText(defaults['smoothing_method_comboBox'])
        self.ui.window_length_spinBox.setValue(defaults['window_length_spinBox'])
        self.ui.window_length_spinBox.setVisible(False)
        self.ui.whittaker_lambda_spinBox.setValue(defaults['window_length_spinBox'])
        self.ui.whittaker_lambda_spinBox.setVisible(False)
        self.ui.smooth_polyorder_spinBox.setValue(defaults['window_length_spinBox'])
        self.ui.smooth_polyorder_spinBox.setVisible(False)
        self.ui.sigma_spinBox.setValue(defaults['window_length_spinBox'])
        self.ui.sigma_spinBox.setVisible(True)
        self.ui.kaiser_beta_doubleSpinBox.setValue(defaults['window_length_spinBox'])
        self.ui.kaiser_beta_doubleSpinBox.setVisible(False)
        self.ui.emd_noise_modes_spinBox.setValue(defaults['window_length_spinBox'])
        self.ui.emd_noise_modes_spinBox.setVisible(False)
        self.ui.eemd_trials_spinBox.setValue(defaults['window_length_spinBox'])
        self.ui.eemd_trials_spinBox.setVisible(False)
        self.current_method = ''
        if self.parent.active_stage == self:
            self.parent.update_plot_item('SmoothedData')
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
              'window_length_spinBox': self.ui.window_length_spinBox.value(),
              'whittaker_lambda_spinBox': self.ui.whittaker_lambda_spinBox.value(),
              'smooth_polyorder_spinBox': self.ui.smooth_polyorder_spinBox.value(),
              'sigma_spinBox': self.ui.sigma_spinBox.value(),
              'kaiser_beta_doubleSpinBox': self.ui.kaiser_beta_doubleSpinBox.value(),
              'emd_noise_modes_spinBox': self.ui.emd_noise_modes_spinBox.value(),
              'eemd_trials_spinBox': self.ui.eemd_trials_spinBox.value(),
              'smoothing_method_comboBox': self.ui.smoothing_method_comboBox.currentText(),
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
        self.ui.window_length_spinBox.setValue(db['window_length_spinBox'])
        self.ui.whittaker_lambda_spinBox.setValue(db['whittaker_lambda_spinBox'])
        self.ui.smooth_polyorder_spinBox.setValue(db['smooth_polyorder_spinBox'])
        self.ui.sigma_spinBox.setValue(db['sigma_spinBox'])
        self.ui.kaiser_beta_doubleSpinBox.setValue(db['kaiser_beta_doubleSpinBox'])
        self.ui.emd_noise_modes_spinBox.setValue(db['emd_noise_modes_spinBox'])
        self.ui.eemd_trials_spinBox.setValue(db['eemd_trials_spinBox'])
        self.ui.smoothing_method_comboBox.setCurrentText(db['smoothing_method_comboBox'])
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
            case 'eemd_trials_spinBox':
                self.ui.eemd_trials_spinBox.setValue(value)
            case 'emd_noise_modes_spinBox':
                self.ui.emd_noise_modes_spinBox.setValue(value)
            case 'kaiser_beta_doubleSpinBox':
                self.ui.kaiser_beta_doubleSpinBox.setValue(value)
            case 'sigma_spinBox':
                self.ui.sigma_spinBox.setValue(value)
            case 'smooth_polyorder_spinBox':
                self.ui.smooth_polyorder_spinBox.setValue(value)
            case 'whittaker_lambda_spinBox':
                self.ui.whittaker_lambda_spinBox.setValue(value)
            case 'window_length_spinBox':
                self.ui.window_length_spinBox.setValue(value)
            case _:
                return

    def plot_items(self) -> ItemsView:
        """
        Returns data for plotting
        """
        return self.data.items()

    def _init_smoothing_method_combo_box(self) -> None:
        self.ui.smoothing_method_comboBox.addItems(self.smoothing_methods.keys())

    def method_changed(self, current_text: str):
        context = get_parent(self.parent, "Context")
        context.set_modified()
        self.ui.window_length_spinBox.setVisible(False)
        self.ui.smooth_polyorder_spinBox.setVisible(False)
        self.ui.whittaker_lambda_spinBox.setVisible(False)
        self.ui.kaiser_beta_doubleSpinBox.setVisible(False)
        self.ui.emd_noise_modes_spinBox.setVisible(False)
        self.ui.eemd_trials_spinBox.setVisible(False)
        self.ui.sigma_spinBox.setVisible(False)
        match current_text:
            case "Savitsky-Golay filter":
                self.ui.window_length_spinBox.setVisible(True)
                self.ui.smooth_polyorder_spinBox.setVisible(True)
            case "MLESG":
                self.ui.sigma_spinBox.setVisible(True)
            case "Whittaker smoother":
                self.ui.whittaker_lambda_spinBox.setVisible(True)
            case ("Flat window" | "hanning" | "hamming" | "bartlett" | "blackman" | "Median filter"
                  | "Wiener filter"):
                self.ui.window_length_spinBox.setVisible(True)
            case "kaiser":
                self.ui.window_length_spinBox.setVisible(True)
                self.ui.kaiser_beta_doubleSpinBox.setVisible(True)
            case "EMD":
                self.ui.emd_noise_modes_spinBox.setVisible(True)
            case "EEMD" | "CEEMDAN":
                self.ui.emd_noise_modes_spinBox.setVisible(True)
                self.ui.eemd_trials_spinBox.setVisible(True)

    @asyncSlot()
    async def _smooth_clicked(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.drag_widget.get_previous_stage(self)
        if prev_stage is None or not prev_stage.data:
            mw.ui.statusBar.showMessage("No data for smoothing")
            return
        await self._smooth(mw, prev_stage.data)

    @asyncSlot()
    async def _smooth(self, mw: QMainWindow, data: ObservableDict) -> None:
        n_files = len(data)
        cfg = get_config("texty")["smooth"]
        method = self.ui.smoothing_method_comboBox.currentText()
        args = self.smoothing_params(method)
        if method == 'Savitsky-Golay filter' and args[1] >= args[0]:
            mw.show_error(ValueError("polyorder must be less than window_length."))
            return
        mw.progress.open_progress(cfg, n_files)
        func, n_limit = self.smoothing_methods[method]
        snr_df = mw.ui.input_table.model().get_column('SNR')

        kwargs = {'n_files': n_files, 'n_limit': n_limit}
        iter_by = {k: (arr, snr_df[k]) for k, arr in data.items()} if method == 'MLESG' \
            else data
        result: list[tuple[str, np.ndarray]] = await mw.progress.run_in_executor(
            "smooth", func, iter_by.items(), *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandSmooth(result, context, text="Smooth", **{'stage': self,
                                                                   'method': method,
                                                                   'params': args})
        context.undo_stack.push(command)

    def smoothing_params(self, method: str) -> int | tuple[int, int | str] | tuple[float, int, int]:
        mw = get_parent(self.parent, "MainWindow")
        params = None
        match method:
            case 'EMD':
                params = [self.ui.emd_noise_modes_spinBox.value()]
            case 'EEMD' | 'CEEMDAN':
                params = (self.ui.emd_noise_modes_spinBox.value(),
                          self.ui.eemd_trials_spinBox.value())
            case 'MLESG':
                fwhm_cm_df = mw.ui.input_table.model().get_column(
                    'FWHM, cm\N{superscript minus}\N{superscript one}')
                distance = np.max(fwhm_cm_df.values)
                sigma = self.ui.sigma_spinBox.value()
                params = (distance, sigma)
            case 'Savitsky-Golay filter':
                params = (self.ui.window_length_spinBox.value(),
                          self.ui.smooth_polyorder_spinBox.value())
            case 'Whittaker smoother':
                params = [self.ui.whittaker_lambda_spinBox.value()]
            case 'Flat window':
                params = [self.ui.window_length_spinBox.value()]
            case 'hanning' | 'hamming' | 'bartlett' | 'blackman':
                params = self.ui.window_length_spinBox.value(), method
            case 'kaiser':
                params = (self.ui.window_length_spinBox.value(),
                          self.ui.kaiser_beta_doubleSpinBox.value())
            case 'Median filter' | 'Wiener filter':
                params = [self.ui.window_length_spinBox.value()]
        return params


class CommandSmooth(UndoCommand):
    """
    Change data for smoothing stage.

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
        method = kwargs.pop('method')
        self.params = kwargs.pop('params')
        super().__init__(data, parent, text, *args, **kwargs)
        self.data = data
        self.old_data = deepcopy(self.stage.data.get_data())
        self.method_new = self.generate_title_text(method)
        self.method_old = copy(self.stage.current_method)
        self.smoothed_df_new = deepcopy(self.create_smoothed_dataset_new())
        self.smoothed_df_old = deepcopy(self.mw.ui.smoothed_dataset_table_view.model().dataframe())

    def redo_special(self):
        """
        Update data
        """
        self.stage.data.clear()
        self.stage.data.update(dict(self.data))
        self.stage.current_method = self.method_new
        self.mw.ui.smoothed_dataset_table_view.model().set_dataframe(self.smoothed_df_new)

    def undo_special(self):
        """
        Undo data
        """
        self.stage.data.clear()
        self.stage.data.update(self.old_data)
        self.stage.current_method = self.method_old
        self.mw.ui.smoothed_dataset_table_view.model().set_dataframe(self.smoothed_df_old)

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.preprocessing.update_plot_item("SmoothedData")

    def generate_title_text(self, method: str) -> str:
        text = method + '. '
        match method:
            case 'EMD':
                text += 'IMFs: ' + str(self.params[0])
            case 'MLESG':
                text += 'sigma: ' + str(self.params[1])
            case 'EEMD' | 'CEEMDAN':
                imfs, trials = self.params
                text += 'IMFs: ' + str(imfs) + ', trials: ' + str(trials)
            case 'Savitsky-Golay filter':
                window_len, polyorder = self.params
                text += 'Window length: ' + str(window_len) + ', polynome order: ' + str(polyorder)
            case 'Whittaker smoother':
                text += 'λ: ' + str(self.params[0])
            case 'Flat window':
                text += 'Window length: ' + str(self.params[0])
            case 'hanning' | 'hamming' | 'bartlett' | 'blackman':
                text += 'Window length: ' + str(self.params[0])
            case 'kaiser':
                window_len, kaiser_beta = self.params
                text += 'Window length: ' + str(window_len) + ', β: ' + str(kaiser_beta)
            case 'Median filter':
                text += 'Window length: ' + str(self.params[0])
            case 'Wiener filter':
                text += 'Window length: ' + str(self.params[0])
        return text

    def create_smoothed_dataset_new(self) -> pd.DataFrame:
        filename_group = self.mw.ui.input_table.model().column_data(2)
        x_axis = self.data[0][1][:, 0]
        columns_params = [f'k{np.round(i, 2)}' for i in x_axis]
        df = pd.DataFrame(columns=columns_params)
        class_ids = []
        for filename, n_array in self.data:
            class_ids.append(filename_group.loc[filename])
            df2 = pd.DataFrame(n_array[:, 1].reshape(1, -1), columns=columns_params)
            df = pd.concat([df, df2], ignore_index=True)
        df2 = pd.DataFrame({'Class': class_ids, 'Filename': list(filename_group.index)})
        df = pd.concat([df2, df], axis=1)
        return df
