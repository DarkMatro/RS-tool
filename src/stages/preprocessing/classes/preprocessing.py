"""
Module for managing preprocessing stages in spectral data analysis.

This module defines the `Preprocessing` class, which handles various preprocessing stages
such as input data loading, conversion, cutting, normalization, smoothing, baseline correction,
trimming, and averaging. It also includes functions for updating and plotting data at different
preprocessing stages.

Classes
-------
PreprocessingStages
    Container for all preprocessing stage instances.
Preprocessing
    Manages the entire preprocessing workflow and updates the UI accordingly.
"""

import dataclasses
from os import environ
from typing import ItemsView

import numpy as np
from qtpy.QtCore import QObject, Qt
from qtpy.QtGui import QColor
from pyqtgraph import SignalProxy, InfiniteLine

from src import get_config
from src.data.get_data import get_parent
from src.data.plotting import random_rgb
from .convertdata import ConvertData
from .cut_data import CutData
from .input_data import InputData
from .normalized_data import NormalizedData
from .smoothed_data import SmoothedData
from .baseline_data import BaselineData
from .av_data import AvData
from src.ui.MultiLine import MultiLine


@dataclasses.dataclass
class PreprocessingStages:
    """
    Container for all preprocessing stage instances.

    Attributes
    ----------
    input_data : InputData
        Instance managing the input data stage.
    convert_data : ConvertData
        Instance managing the data conversion stage.
    cut_data : CutData
        Instance managing the data cutting stage.
    normalized_data : NormalizedData
        Instance managing the data normalization stage.
    smoothed_data : SmoothedData
        Instance managing the data smoothing stage.
    bl_data : BaselineData
        Instance managing the baseline correction stage.
    trim_data : CutData
        Instance managing the data trimming stage.
    av_data : AvData
        Instance managing the data averaging stage.
    """
    input_data: InputData
    convert_data: ConvertData
    cut_data: CutData
    normalized_data: NormalizedData
    smoothed_data: SmoothedData
    bl_data: BaselineData
    trim_data: CutData
    av_data: AvData


class Preprocessing(QObject):
    """
    Manages the entire preprocessing workflow and updates the UI accordingly.

    Parameters
    ----------
    parent : Context
        The parent context.

    Attributes
    ----------
    stage_at_plot : str or None
        The name of the current stage being plotted.
    parent : Context
        The parent context.
    stages : PreprocessingStages
        Container for all preprocessing stage instances.
    one_curve : MultiLine or None
        The current curve being plotted.
    active_stage : PreprocessingStage
        The currently active preprocessing stage.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the Preprocessing object with parent context.

        Parameters
        ----------
        parent : Context
            The parent context.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.stage_at_plot = None
        self.parent = parent
        self.stages = PreprocessingStages(
            input_data=InputData(parent=self), convert_data=ConvertData(parent=self),
            cut_data=CutData(self), normalized_data=NormalizedData(self),
            smoothed_data=SmoothedData(self), bl_data=BaselineData(self),
            trim_data=CutData(self, is_trim=True), av_data=AvData(self)
        )
        self.one_curve = None
        self.active_stage = self.stages.input_data
        self._initial_preproc_plot()

    def reset(self):
        """
        Reset all preprocessing stages to their default state.
        """
        self.stages.input_data.reset()
        self.stages.convert_data.reset()
        self.stages.cut_data.reset()
        self.stages.normalized_data.reset()
        self.stages.smoothed_data.reset()
        self.stages.bl_data.reset()
        self.stages.trim_data.reset()
        self.stages.av_data.reset()

    def update_plot_item(self, stage_name: str | None = None) -> None:
        """
        Update the plot item for the specified preprocessing stage.

        Parameters
        ----------
        stage_name : str or None, optional
            The name of the stage to update. If None, updates the current stage.
        """
        mw = get_parent(self.parent, "MainWindow")
        if stage_name is None:
            stage_name = mw.drag_widget.get_current_widget_name()
        match stage_name:
            case "InputData" | '':
                stage = self.stages.input_data
                plot_item_id = 0
            case 'ConvertData':
                stage = self.stages.convert_data
                plot_item_id = 1
            case 'CutData':
                stage = self.stages.cut_data
                plot_item_id = 2
            case 'NormalizedData':
                stage = self.stages.normalized_data
                plot_item_id = 3
            case 'SmoothedData':
                stage = self.stages.smoothed_data
                plot_item_id = 4
            case 'BaselineData':
                stage = self.stages.bl_data
                plot_item_id = 5
            case 'TrimData':
                stage = self.stages.trim_data
                plot_item_id = 6
            case 'AvData':
                stage = self.stages.av_data
                plot_item_id = 7
            case _:
                return
        label_style = mw.get_plot_label_style()
        label = 'Wavelength, nm.' if plot_item_id == 0 \
            else 'Raman shift, cm\N{superscript minus}\N{superscript one}'
        mw.ui.preproc_plot_widget.setLabel("bottom", label, units='', **label_style)

        self._clear_plots_before_update(mw)
        self.set_preproc_title(stage)
        self.stage_at_plot = stage_name

        if plot_item_id == 7:
            if self.parent.group_table.rowCount == 0:
                return
            groups_styles = self.parent.group_table.table_widget.model().groups_styles
            for i, (key, arr) in enumerate(stage.plot_items()):
                if len(arr) == 0:
                    continue
                try:
                    self.add_lines(np.array([arr[:, 0]]), np.array([arr[:, 1]]),
                                   groups_styles[i], key, plot_item_id)
                except IndexError:
                    pass
        else:
            arrays = self._get_arrays_by_group(stage.plot_items(), mw)
            self._combine_arrays_by_groups(arrays, plot_item_id)

        self._after_updating_data(plot_item_id)

    def set_preproc_title(self, stage: PreprocessingStages):
        """
        Set the title for the preprocessing plot based on the current stage.

        Parameters
        ----------
        stage : PreprocessingStages
            The current preprocessing stage.
        """
        mw = get_parent(self.parent, "MainWindow")
        cfg = get_config('plots')['preproc']
        if stage is None:
            return
        current_stage = stage.name
        label = '' if not current_stage else cfg[current_stage] + '. '
        method = stage.current_method if hasattr(stage, 'current_method') else ''

        title = ('<span style="font-family: AbletonSans; color:' + environ["plotText"]
                 + ';font-size: ' + str(environ['plot_font_size'])
                 + '">' + label + method + '</span>')
        mw.ui.preproc_plot_widget.setTitle(title)

    def _after_updating_data(self, plot_item_id: int):
        """
        Perform actions after updating the data in the plot.

        Parameters
        ----------
        plot_item_id : int
            The ID of the plot item being updated.
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.preproc_plot_widget.getPlotItem()
        if plot_item_id == 2:
            plot_item.addItem(self.stages.cut_data.linear_region)
        elif plot_item_id == 6:
            plot_item.addItem(self.stages.trim_data.linear_region)
        else:
            plot_item.removeItem(self.stages.cut_data.linear_region)
        if mw.ui.by_one_control_button.isChecked():
            mw.by_one_control_button_clicked()
        elif mw.ui.by_group_control_button.isChecked():
            mw.by_group_control_button()
        plot_item.getViewBox().updateAutoRange()
        plot_item.updateParamList()
        plot_item.recomputeAverages()

    def _get_arrays_by_group(self, items: ItemsView, mw) -> list[tuple[dict, list]]:
        """
        Get arrays of data grouped by their respective styles.

        Parameters
        ----------
        items : ItemsView
            The items view of the data to be grouped.
        mw : MainWindow
            The main window object.

        Returns
        -------
        list of tuple
            A list of tuples containing style dictionaries and grouped data arrays.
        """
        styles = self.parent.group_table.table_widget.model().column_data(1)
        std_style = {
            "color": QColor(environ["secondaryColor"]),
            "style": Qt.PenStyle.SolidLine,
            "width": 1.0,
            "fill": False,
            "use_line_color": True,
            "fill_color": QColor().fromRgb(random_rgb()),
            "fill_opacity": 0.0,
        }
        arrays = [(std_style, [], 1)]
        idx = 0
        for style in styles:
            arrays.append((style, []))
            idx += 1

        for i in items:
            name = i[0]
            arr = i[1]
            group_number = int(mw.ui.input_table.model().get_group_by_name(name))

            if group_number > len(styles):
                # in case when have group number, but there is no corresponding group actually
                group_number = 0
            arrays[group_number][1].append(arr)
        return arrays

    def _combine_arrays_by_groups(
            self, arrays: list[tuple[dict, list]], plot_item_id: int
    ) -> None:
        """
        Combine data arrays by their groups and plot them.

        Parameters
        ----------
        arrays : list of tuple
            A list of tuples containing style dictionaries and grouped data arrays.
        plot_item_id : int
            The ID of the plot item being updated.
        """
        for idx, item in enumerate(arrays):
            style = item[0]
            xy_arrays = item[1]
            if len(xy_arrays) == 0:
                continue
            if plot_item_id == 0:
                self.process_input_plots_by_different_ranges(
                    style, xy_arrays, idx, plot_item_id
                )
            else:
                self.process_plots_by_different_ranges(style, xy_arrays, idx, plot_item_id)

    def process_input_plots_by_different_ranges(
            self, style: dict, xy_arrays: list, group_idx: int, plot_item_id: int
    ) -> None:
        """
        Process input plots by different ranges.

        Parameters
        ----------
        style : dict
            The style dictionary for plotting.
        xy_arrays : list
            The list of x and y data arrays.
        group_idx : int
            The index of the group being processed.
        plot_item_id : int
            The ID of the plot item being updated.
        """
        arrays_by_ranges = dict()
        for j in xy_arrays:
            array_len = j.shape[0]
            x_axis = j[:, 0]
            y_axis = j[:, 1]
            if array_len not in arrays_by_ranges:
                arrays_by_ranges[array_len] = ([x_axis], [y_axis])
            elif array_len in arrays_by_ranges:
                xy_axes_tuple = arrays_by_ranges[array_len]
                xy_axes_tuple[0].append(x_axis)
                xy_axes_tuple[1].append(y_axis)
        for _, xy_axes_list in arrays_by_ranges.items():
            x_arrays = np.array(xy_axes_list[0])
            y_arrays = np.array(xy_axes_list[1])
            self.add_lines(x_arrays, y_arrays, style, group_idx, plot_item_id)

    def process_plots_by_different_ranges(self, style: dict, xy_arrays: list, group_idx: int,
                                          plot_item_id: int) -> None:
        """
        Process plots by different ranges.

        Parameters
        ----------
        style : dict
            The style dictionary for plotting.
        xy_arrays : list
            The list of x and y data arrays.
        group_idx : int
            The index of the group being processed.
        plot_item_id : int
            The ID of the plot item being updated.
        """
        x_axes = []
        y_axes = []

        for j in xy_arrays:
            x_axes.append(j[:, 0])
            y_axes.append(j[:, 1])
        x_arrays = np.array(x_axes)
        y_arrays = np.array(y_axes)
        self.add_lines(x_arrays, y_arrays, style, group_idx, plot_item_id)

    def add_lines(
            self, x: np.ndarray, y: np.ndarray, style: dict, _group: int, plot_item_id: int
    ) -> None:
        """
        Add lines to the plot.

        Parameters
        ----------
        x : np.ndarray
            The x data array.
        y : np.ndarray
            The y data array.
        style : dict
            The style dictionary for plotting.
        _group : int
            The group index.
        plot_item_id : int
            The ID of the plot item being updated.
        """
        curve = MultiLine(x, y, style, _group)
        mw = get_parent(self.parent, "MainWindow")
        if plot_item_id == 0:
            mw.ui.preproc_plot_widget.getPlotItem().addItem(
                curve, kargs={"ignoreBounds": False}
            )
        else:
            mw.ui.preproc_plot_widget.getPlotItem().addItem(curve)

    def _clear_plots_before_update(self, mw):
        """
        Clear existing plots before updating with new data.

        Parameters
        ----------
        mw : MainWindow
            The main window object.
        """
        plot_item = mw.ui.preproc_plot_widget.getPlotItem()
        plot_item.clear()
        if self.one_curve:
            plot_item.removeItem(self.one_curve)
        if self.stages.input_data.despiked_one_curve:
            plot_item.removeItem(self.stages.input_data.despiked_one_curve)
        if self.stages.bl_data.baseline_one_curve:
            plot_item.removeItem(self.stages.bl_data.baseline_one_curve)
        if mw.ui.crosshairBtn.isChecked():
            plot_item.addItem(mw.ui.preproc_plot_widget.vertical_line, ignoreBounds=True)
            plot_item.addItem(mw.ui.preproc_plot_widget.horizontal_line, ignoreBounds=True)

    def _initial_preproc_plot(self) -> None:
        """
        Perform initial setup for the preprocessing plot.
        """
        mw = get_parent(self.parent, "MainWindow")
        cfg = get_config('plots')['preproc']
        mw.ui.crosshair_update_preproc = SignalProxy(
            mw.ui.preproc_plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=mw.update_crosshair,
        )
        plot_item = mw.ui.preproc_plot_widget.getPlotItem()
        mw.ui.preproc_plot_widget.setAntialiasing(1)
        plot_item.enableAutoRange()
        plot_item.showGrid(True, True, cfg['alpha'])
        mw.ui.preproc_plot_widget.vertical_line = InfiniteLine()
        mw.ui.preproc_plot_widget.horizontal_line = InfiniteLine(angle=0)
        mw.ui.preproc_plot_widget.vertical_line.setPen(QColor(environ["secondaryColor"]))
        mw.ui.preproc_plot_widget.horizontal_line.setPen(QColor(environ["secondaryColor"]))
        current_stage = self.active_stage.name
        title = ('<span style="font-family: AbletonSans; color:' + environ["plotText"]
                 + ';font-size: ' + str(environ['plot_font_size'])
                 + '">' + cfg[current_stage] + '</span>')
        mw.ui.preproc_plot_widget.setTitle(title)
        items_matches = (
            x
            for x in plot_item.listDataItems()
            if not x.name()
        )
        for i in items_matches:
            plot_item.removeItem(i)
        self._initial_plot_color()

    def _initial_plot_color(self) -> None:
        """
        Set the initial colors for the plot elements.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.preproc_plot_widget.setBackground(mw.plot_background_color)
        mw.ui.preproc_plot_widget.getPlotItem().getAxis("bottom").setPen(mw.plot_text_color)
        mw.ui.preproc_plot_widget.getPlotItem().getAxis("left").setPen(mw.plot_text_color)
        mw.ui.preproc_plot_widget.getPlotItem().getAxis("bottom").setTextPen(mw.plot_text_color)
        mw.ui.preproc_plot_widget.getPlotItem().getAxis("left").setTextPen(mw.plot_text_color)
