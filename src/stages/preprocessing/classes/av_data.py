import numpy as np
from matplotlib import pyplot as plt
from qtpy.QtWidgets import QMainWindow
from asyncqtpy import asyncSlot
import pandas as pd
import seaborn as sns
from src.data.collections import ObservableDict
from src.stages.preprocessing.classes.stages import PreprocessingStage
from src.stages.preprocessing.functions.averaging import get_average_spectrum
from src.ui.ui_average_widget import Ui_AverageForm
from src.data.get_data import get_parent
from typing import ItemsView
from qtpy.QtGui import QMouseEvent
from qtpy.QtCore import Qt
from src.data.config import get_config


class AvData(PreprocessingStage):
    """
    Average spectrum of previous stage data

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
        self.name = 'AvData'

    def set_ui(self, ui: Ui_AverageForm) -> None:
        """
        Set user interface object

        Parameters
        -------
        ui: Ui_AverageForm
            widget
        """
        context = get_parent(self.parent, "Context")
        defaults = get_config('defaults')
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.average_btn.clicked.connect(self._average_clicked)
        self.ui.sns_plot_btn.clicked.connect(self._sns_plot_clicked)

        self.ui.average_method_cb.addItems(["Mean", "Median"])
        self.ui.average_errorbar_method_combo_box.addItems(["ci", "pi", "se", "sd"])
        self.ui.average_method_cb.currentTextChanged.connect(context.set_modified)
        self.ui.average_method_cb.setCurrentText(defaults['average_function'])
        self.ui.average_errorbar_method_combo_box.currentTextChanged.connect(context.set_modified)
        self.ui.average_errorbar_method_combo_box.setCurrentText(defaults['average_errorbar'])

        self.ui.average_n_boot_spin_box.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'average_n_boot_spin_box')
        self.ui.average_n_boot_spin_box.valueChanged.connect(context.set_modified)

        self.ui.average_level_spin_box.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'average_level_spin_box')
        self.ui.average_level_spin_box.valueChanged.connect(context.set_modified)

    def reset(self) -> None:
        """
        Reset class data.
        """
        self.data.clear()
        defaults = get_config('defaults')
        self.ui.average_method_cb.setCurrentText(defaults['average_function'])
        self.ui.average_errorbar_method_combo_box.setCurrentText(defaults['average_errorbar'])
        self.ui.average_n_boot_spin_box.setValue(defaults['average_n_boot_spin_box'])
        self.ui.average_level_spin_box.setValue(defaults['average_level_spin_box'])
        if self.parent.active_stage == self:
            self.parent.update_plot_item('AvData')

    def read(self) -> dict:
        """
        Read attributes data.

        Returns
        -------
        dt: dict
            all class attributes data
        """
        dt = {"data": self.data.get_data(),
              'average_errorbar_method_combo_box': self.ui.average_errorbar_method_combo_box.currentText(),
              'average_level_spin_box': self.ui.average_level_spin_box.value(),
              'average_method_cb': self.ui.average_method_cb.currentText(),
              'average_n_boot_spin_box': self.ui.average_n_boot_spin_box.value()}
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
        self.ui.average_level_spin_box.setValue(db['average_level_spin_box'])
        self.ui.average_n_boot_spin_box.setValue(db['average_n_boot_spin_box'])
        self.ui.average_errorbar_method_combo_box.setCurrentText(db['average_errorbar_method_combo_box'])
        self.ui.average_method_cb.setCurrentText(db['average_method_cb'])

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
            case 'average_n_boot_spin_box':
                self.ui.average_n_boot_spin_box.setValue(value)
            case 'average_level_spin_box':
                self.ui.average_level_spin_box.setValue(value)
            case _:
                return

    def plot_items(self) -> ItemsView:
        """
        Returns data for plotting
        """
        return self.data.items()

    @asyncSlot()
    async def _average_clicked(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.drag_widget.get_previous_stage(self)
        if prev_stage is None or not prev_stage.data:
            mw.ui.statusBar.showMessage("No data for averaging")
            return
        await self._update_averaged(mw, prev_stage.data)
        self.parent.update_plot_item("AvData")
        mw.fitting.update_template_combo_box()
        if not mw.predict_logic.is_production_project:
            mw.fitting.update_deconv_intervals_limits()

    @asyncSlot()
    async def _sns_plot_clicked(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.drag_widget.get_previous_stage(self)
        if prev_stage is None or not prev_stage.data:
            mw.ui.statusBar.showMessage("No data for averaging")
            return
        error_method = self.ui.average_errorbar_method_combo_box.currentText()
        error_level = self.ui.average_level_spin_box.value()
        n_boot = self.ui.average_n_boot_spin_box.value()
        fig, ax = plt.subplots()
        av_df = self.create_averaged_df(prev_stage.data)
        colors = context.group_table.table_widget.model().groups_colors
        palette = sns.color_palette(colors)
        sns.lineplot(data=av_df, x='Raman shift, cm\N{superscript minus}\N{superscript one}',
                     y='Intensity, rel. un.', hue='Label', size='Label', style='Label',
                     palette=palette,
                     sizes=context.group_table.table_widget.model().groups_width,
                     errorbar=(error_method, error_level),
                     dashes=context.group_table.table_widget.model().groups_dashes,
                     legend='full', ax=ax,
                     n_boot=n_boot)
        fig.tight_layout()
        plt.show()

    @asyncSlot()
    async def _update_averaged(self, mw: QMainWindow, data: ObservableDict) -> None:
        context = get_parent(self.parent, "Context")
        n_files = len(data)
        cfg = get_config("texty")["average"]
        mw.progress.open_progress(cfg, n_files)
        n_groups = context.group_table.table_widget.model().rowCount()
        self.data.clear()
        averaging_method = self.ui.average_method_cb.currentText()
        for i in range(n_groups):
            group_id = i + 1
            filenames = mw.ui.input_table.model().names_of_group(group_id)
            if len(filenames) == 0:
                continue
            arrays_list = [data[x] for x in filenames if x in data]
            if not arrays_list:
                continue
            arrays_list_av = get_average_spectrum(arrays_list, averaging_method)
            self.data[group_id] = arrays_list_av
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return

    def create_averaged_df(self, data: ObservableDict) -> pd.DataFrame:
        """
        Function creates DataFrame for seaborn line plot.

        Returns
        -------
        av_df: pd.DataFrame
            with 3 columns:
                Label: group_id
                Raman shift: cm-1 for x-axis
                Intensity, rel. un.: y-axis value
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        av_df = pd.DataFrame(
            columns=['Label', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                     'Intensity, rel. un.'])
        n_groups = context.group_table.table_widget.model().rowCount()
        for i in range(n_groups):
            group_id = i + 1
            filenames = mw.ui.input_table.model().names_of_group(group_id)
            n_spectrum = len(filenames)
            if n_spectrum == 0:
                continue
            arrays_y = [data[x][:, 1] for x in filenames]
            arrays_y = np.array(arrays_y).flatten()
            x_axis = next(iter(data.values()))[:, 0]
            x_axis = np.array(x_axis)
            x_axis = np.tile(x_axis, n_spectrum)
            label = context.group_table.table_widget.model().get_group_name_by_int(group_id)
            labels = [label] * arrays_y.size
            df = pd.DataFrame(
                {'Label': labels, 'Raman shift, cm\N{superscript minus}\N{superscript one}': x_axis,
                 'Intensity, rel. un.': arrays_y})
            av_df = pd.concat([av_df, df])
        return av_df
