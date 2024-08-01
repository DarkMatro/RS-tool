# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module contains the `AvData` class, which is used for managing the averaging
of spectra in a preprocessing stage of data analysis. It includes methods for
setting up the user interface, handling data changes, resetting class data,
reading and loading data, plotting, and updating averaged data.

Classes
-------
AvData
    Manages the averaging of spectra in a preprocessing stage, handles data
    changes, sets up the user interface, and updates averaged data.

Functions
---------
__init__(self, parent, *args, **kwargs)
    Initialize the AvData class with the given parent and optional arguments.
data_changed(self, _)
    Update the template combo box when data changes.
set_ui(self, ui: Ui_AverageForm)
    Set the user interface object for the class.
reset(self)
    Reset class data to default values.
read(self) -> dict
    Read and return the attributes data of the class.
load(self, db: dict)
    Load attributes data from a given dictionary.
reset_field(self, event: QMouseEvent, field_id: str)
    Reset the value of a specified field to its default on double-click by MiddleButton.
plot_items(self) -> ItemsView
    Return data for plotting.
_average_clicked(self)
    Handle the averaging button click event asynchronously.
_sns_plot_clicked(self)
    Handle the seaborn plot button click event asynchronously.
update_averaged(self, mw: QMainWindow, data: ObservableDict)
    Update the averaged data based on the given data.
create_averaged_df(self, data: ObservableDict) -> pd.DataFrame
    Create a DataFrame for seaborn line plot from the given data.
"""

from typing import ItemsView

import numpy as np
import pandas as pd
import seaborn as sns
from asyncqtpy import asyncSlot
from matplotlib import pyplot as plt
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QMainWindow

from src.data.collections import ObservableDict
from src.data.config import get_config
from src.data.get_data import get_parent
from src.stages.preprocessing.classes.stages import PreprocessingStage
from src.stages.preprocessing.functions.averaging import get_average_spectrum
from src.ui.ui_average_widget import Ui_AverageForm


class AvData(PreprocessingStage):
    """
    Average spectrum of previous stage data.

    Parameters
    ----------
    parent : Preprocessing
        The parent Preprocessing instance.

    Attributes
    ----------
    ui : object
        The user interface form.
    name : str
        Name of the stage, set to 'AvData'.
    data : ObservableDict
        Observable dictionary to store the data.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the AvData class with the given parent and optional arguments.

        Parameters
        ----------
        parent : Preprocessing
            The parent Preprocessing instance.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(parent, *args, **kwargs)
        self.ui = None
        self.name = 'AvData'
        self.data.on_change(self.data_changed)

    def data_changed(self, _):
        """
        Update the template combo box when data changes.
        """
        context = get_parent(self.parent, "Context")
        context.decomposition.update_template_combo_box()

    def set_ui(self, ui: Ui_AverageForm) -> None:
        """
        Set the user interface object for the class.

        Parameters
        ----------
        ui : Ui_AverageForm
            The user interface form to be set.
        """
        context = get_parent(self.parent, "Context")
        defaults = get_config('defaults')
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.average_btn.clicked.connect(self.average_clicked)
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
        Reset class data to default values.
        """
        self.data.clear()
        defaults = get_config('defaults')
        self.ui.average_method_cb.setCurrentText(defaults['average_function'])
        self.ui.average_errorbar_method_combo_box.setCurrentText(defaults['average_errorbar'])
        self.ui.average_n_boot_spin_box.setValue(defaults['average_n_boot_spin_box'])
        self.ui.average_level_spin_box.setValue(defaults['average_level_spin_box'])
        if self.parent.active_stage == self:
            self.parent.update_plot_item('AvData')

    def read(self, production_export: bool=False) -> dict:
        """
        Read and return the attributes data of the class.

        Returns
        -------
        dict
            A dictionary containing the class attributes' data.
        """
        dt = {'average_errorbar_method_combo_box':
                  self.ui.average_errorbar_method_combo_box.currentText(),
              'average_level_spin_box': self.ui.average_level_spin_box.value(),
              'average_method_cb': self.ui.average_method_cb.currentText(),
              'average_n_boot_spin_box': self.ui.average_n_boot_spin_box.value()}
        if not production_export:
            dt['data'] = self.data.get_data()
        return dt

    def load(self, db: dict) -> None:
        """
        Load attributes data from a given dictionary.

        Parameters
        ----------
        db : dict
            A dictionary containing the attributes data to be loaded.
        """
        if 'data' in db:
            self.data.update(db['data'])
        self.ui.average_level_spin_box.setValue(db['average_level_spin_box'])
        self.ui.average_n_boot_spin_box.setValue(db['average_n_boot_spin_box'])
        self.ui.average_errorbar_method_combo_box.setCurrentText(
            db['average_errorbar_method_combo_box'])
        self.ui.average_method_cb.setCurrentText(db['average_method_cb'])

    def reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Reset the value of a specified field to its default on double-click by MiddleButton.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event triggering the reset.
        field_id : str
            The identifier of the field to be reset.
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
        Return data for plotting.

        Returns
        -------
        ItemsView
            The data items for plotting.
        """
        return self.data.items()

    @asyncSlot()
    async def average_clicked(self) -> None:
        """
        Handle the averaging button click event asynchronously.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if prev_stage is None or not prev_stage.data:
            mw.ui.statusBar.showMessage("No data for averaging")
            return
        await self.update_averaged(mw, prev_stage.data)
        self.parent.update_plot_item("AvData")

    @asyncSlot()
    async def _sns_plot_clicked(self) -> None:
        """
        Handle the seaborn plot button click event asynchronously.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
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
    async def update_averaged(self, mw: QMainWindow, data: ObservableDict) -> None:
        """
        Update the averaged data based on the given data.

        Parameters
        ----------
        mw : QMainWindow
            The main window instance.
        data : ObservableDict
            The observable dictionary containing the data to be averaged.
        """
        context = get_parent(self.parent, "Context")
        n_files = len(data)
        cfg = get_config("texty")["average"]
        mw.progress.open_progress(cfg, n_files)
        n_groups = context.group_table.table_widget.model().rowCount()
        self.data.clear()
        averaging_method = self.ui.average_method_cb.currentText()
        new_dict = {}
        for i in range(n_groups):
            group_id = i + 1
            filenames = mw.ui.input_table.model().names_of_group(group_id)
            if len(filenames) == 0:
                continue
            arrays_list = [data[x] for x in filenames if x in data]
            if not arrays_list:
                continue
            arrays_list_av = get_average_spectrum(arrays_list, averaging_method)
            new_dict[group_id] = arrays_list_av
        self.data.update(new_dict)
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return

    def create_averaged_df(self, data: ObservableDict) -> pd.DataFrame:
        """
        Create a DataFrame for seaborn line plot from the given data.

        Parameters
        ----------
        data : ObservableDict
            The observable dictionary containing the data.

        Returns
        -------
        av_df: pd.DataFrame
            A DataFrame containing the averaged data for plotting.
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
