import pandas as pd
import seaborn as sns
from asyncqtpy import asyncSlot
from pyqtgraph import ArrowItem
from datetime import datetime
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from modules import get_average_spectrum, random_rgb, find_first_right_local_minimum, \
    find_first_left_local_minimum, convert, find_fluorescence_beginning, \
    find_nearest_by_idx, subtract_cosmic_spikes_moll, interpolate
from os import environ
from asyncio import gather
from modules.undo_redo import CommandTrim, CommandBaselineCorrection, CommandSmooth, CommandNormalize, \
    CommandCutFirst, CommandConvert, CommandUpdateDespike, CommandUpdateInterpolated
import numpy as np
from modules.stages.preprocessing.MultiLine import MultiLine
from modules import DialogListBox
from qtpy.QtGui import QColor
from qtpy.QtCore import Qt
from qfluentwidgets import MessageBox
from modules.default_values import baseline_methods, normalize_methods, smoothing_methods
from modules.stages.preprocessing.functions.normalization.normalization import get_emsc_average_spectrum
from modules.stages.preprocessing.functions.cut_trim.cut_trim import cut_spectrum
from modules.mutual_functions.work_with_arrays import nearest_idx, find_nearest


class PreprocessingLogic:

    def __init__(self, parent):
        self.av_df = None  # DataFrame of baseline_corrected spectra with class labels
        self.parent = parent
        self.BeforeDespike = dict()
        self.ConvertedDict = dict()
        self.CuttedFirstDict = dict()
        self.NormalizedDict = dict()
        self.smoothed_spectra = dict()
        self.baseline_dict = dict()
        self.baseline_corrected_dict = dict()
        self.baseline_corrected_not_trimmed_dict = dict()
        self.averaged_dict = dict()

        self.curveOneInputPlot = None
        self.curveOneCutPlot = None
        self.curveDespikedHistory = None
        self.curveOneConvertPlot = None
        self.curveOneNormalPlot = None
        self.curveOneSmoothPlot = None
        self.curve_one_baseline_plot = None
        self.curveBaseline = None
        self.normalize_methods = normalize_methods()
        self.smoothing_methods = smoothing_methods()
        self.baseline_methods = baseline_methods()

    # region Plots
    @asyncSlot()
    async def update_plot_item(self, items, plot_item_id: int = 0) -> None:
        self.clear_plots_before_update(plot_item_id)
        if plot_item_id == 6:
            if self.parent.ui.GroupsTable.model().rowCount() == 0:
                return
            groups_styles = self.parent.ui.GroupsTable.model().groups_styles
            i = 0
            for key, arr in items:
                if len(arr) != 0:
                    self.add_lines(np.array([arr[:, 0]]), np.array([arr[:, 1]]), groups_styles[i], key, plot_item_id)
                i += 1
        else:
            arrays = self.get_arrays_by_group(items)
            self.combine_arrays_by_groups(arrays, plot_item_id)
        self.after_updating_data(plot_item_id)

    def clear_plots_before_update(self, plot_item_id: int) -> None:
        mw = self.parent
        match plot_item_id:
            case 0:
                mw.input_plot_widget_plot_item.clear()
                if self.curveOneInputPlot:
                    mw.input_plot_widget_plot_item.removeItem(self.curveOneInputPlot)
                if self.curveDespikedHistory:
                    mw.input_plot_widget_plot_item.removeItem(self.curveDespikedHistory)
            case 1:
                mw.converted_cm_widget_plot_item.clear()
                if self.curveOneConvertPlot:
                    mw.converted_cm_widget_plot_item.removeItem(self.curveOneConvertPlot)
            case 2:
                mw.cut_cm_plotItem.clear()
                if self.curveOneCutPlot:
                    mw.cut_cm_plotItem.removeItem(self.curveOneCutPlot)
            case 3:
                mw.normalize_plotItem.clear()
                if self.curveOneNormalPlot:
                    mw.normalize_plotItem.removeItem(self.curveOneNormalPlot)
            case 4:
                mw.smooth_plotItem.clear()
                if self.curveOneSmoothPlot:
                    mw.smooth_plotItem.removeItem(self.curveOneSmoothPlot)
            case 5:
                mw.baseline_corrected_plotItem.clear()
                if self.curve_one_baseline_plot:
                    mw.baseline_corrected_plotItem.removeItem(self.curve_one_baseline_plot)
                if self.curveBaseline:
                    mw.baseline_corrected_plotItem.removeItem(self.curveBaseline)
            case 6:
                mw.averaged_plotItem.clear()

    def add_lines(self, x: np.ndarray, y: np.ndarray, style: dict, _group: int, plot_item_id: int) -> None:
        curve = MultiLine(x, y, style, _group)
        mw = self.parent
        if plot_item_id == 0:
            mw.input_plot_widget_plot_item.addItem(curve, kargs={'ignoreBounds': False})
        elif plot_item_id == 1:
            mw.converted_cm_widget_plot_item.addItem(curve)
        elif plot_item_id == 2:
            mw.cut_cm_plotItem.addItem(curve)
        elif plot_item_id == 3:
            mw.normalize_plotItem.addItem(curve)
        elif plot_item_id == 4:
            mw.smooth_plotItem.addItem(curve)
        elif plot_item_id == 5:
            mw.baseline_corrected_plotItem.addItem(curve)
        elif plot_item_id == 6:
            mw.averaged_plotItem.addItem(curve)

    def get_arrays_by_group(self, items: dict[str, np.ndarray]) -> list[tuple[dict, list]]:
        styles = self.parent.ui.GroupsTable.model().column_data(1)
        std_style = {'color': QColor(self.parent.theme_colors['secondaryColor']),
                     'style': Qt.PenStyle.SolidLine,
                     'width': 1.0,
                     'fill': False,
                     'use_line_color': True,
                     'fill_color': QColor().fromRgb(random_rgb()),
                     'fill_opacity': 0.0}
        arrays = [(std_style, [], 1)]
        idx = 0
        for style in styles:
            arrays.append((style, []))
            idx += 1

        for i in items:
            name = i[0]
            arr = i[1]
            group_number = int(self.parent.ui.input_table.model().get_group_by_name(name))
            if group_number > len(styles):
                group_number = 0  # in case when have group number, but there is no corresponding group actually
            arrays[group_number][1].append(arr)
        return arrays

    def combine_arrays_by_groups(self, arrays: list[tuple[dict, list]], plot_item_id: int) -> None:
        for idx, item in enumerate(arrays):
            style = item[0]
            xy_arrays = item[1]
            if len(xy_arrays) == 0:
                continue
            if plot_item_id == 0:
                self.process_input_plots_by_different_ranges(style, xy_arrays, idx, plot_item_id)
            else:
                self.process_plots_by_different_ranges(style, xy_arrays, idx, plot_item_id)

    def process_input_plots_by_different_ranges(self, style: dict, xy_arrays: list, group_idx: int,
                                                plot_item_id: int) -> None:
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
        x_axes = []
        y_axes = []

        for j in xy_arrays:
            x_axes.append(j[:, 0])
            y_axes.append(j[:, 1])
        x_arrays = np.array(x_axes)
        y_arrays = np.array(y_axes)
        self.add_lines(x_arrays, y_arrays, style, group_idx, plot_item_id)

    def after_updating_data(self, plot_item_id: int) -> None:
        mw = self.parent
        match plot_item_id:
            case 0:
                mw.input_plot_widget_plot_item.getViewBox().updateAutoRange()
                mw.input_plot_widget_plot_item.updateParamList()
                mw.input_plot_widget_plot_item.recomputeAverages()
            case 1:
                mw.converted_cm_widget_plot_item.addItem(mw.linearRegionCmConverted)
                mw.converted_cm_widget_plot_item.getViewBox().updateAutoRange()
                mw.converted_cm_widget_plot_item.updateParamList()
                mw.converted_cm_widget_plot_item.recomputeAverages()
            case 2:
                mw.cut_cm_plotItem.getViewBox().updateAutoRange()
                mw.cut_cm_plotItem.updateParamList()
                mw.cut_cm_plotItem.recomputeAverages()
            case 3:
                mw.normalize_plotItem.getViewBox().updateAutoRange()
                mw.normalize_plotItem.updateParamList()
                mw.normalize_plotItem.recomputeAverages()
            case 4:
                mw.smooth_plotItem.getViewBox().updateAutoRange()
                mw.smooth_plotItem.updateParamList()
                mw.smooth_plotItem.recomputeAverages()
            case 5:
                mw.baseline_corrected_plotItem.addItem(mw.linearRegionBaseline)
                mw.baseline_corrected_plotItem.getViewBox().updateAutoRange()
                mw.baseline_corrected_plotItem.updateParamList()
                mw.baseline_corrected_plotItem.recomputeAverages()
            case 6:
                mw.linearRegionDeconv.setVisible(mw.ui.interval_checkBox.isChecked())
                mw.averaged_plotItem.getViewBox().updateAutoRange()
                mw.averaged_plotItem.updateParamList()
                mw.averaged_plotItem.recomputeAverages()

    # endregion

    # region INTERPOLATION
    @asyncSlot()
    async def interpolate(self) -> None:
        mw = self.parent
        mw.time_start = datetime.now()
        if len(mw.ImportedArray) < 2:
            MessageBox('Interpolation error.', 'Нужно хотя бы 2 спектра с разными диапазонами для интерполяции.',
                       self.parent, {'Ok'}).exec()
            return
        result = self.check_ranges()
        different_shapes, _, _ = self.check_arrays_shape()
        if len(result) <= 1 and not different_shapes and not mw.predict_logic.is_production_project:
            MessageBox("Interpolation didn't started.", 'Все спектры в одинаковых диапазонах длин волн.', self.parent,
                       {'Ok'}).exec()
            return
        elif len(result) <= 1 and different_shapes and not mw.predict_logic.is_production_project:
            await self.interpolate_shapes()
            return

        dialog = DialogListBox(title='RS-tool', checked_ranges=result)
        dialog_code = dialog.exec()

        if dialog_code == 0:
            return
        range_nm = dialog.get_result()
        filenames = self.get_filenames_of_this_range(range_nm)
        ref_file = self.get_ref_file(range_nm)
        interpolated = None
        try:
            interpolated = await self.get_interpolated(filenames, ref_file)
        except Exception as err:
            mw.show_error(err)

        if interpolated:
            command = CommandUpdateInterpolated(self.parent, interpolated, "Interpolate files")
            mw.undoStack.push(command)
        else:
            await self.interpolate_shapes()

    async def interpolate_shapes(self) -> None:
        mw = self.parent
        different_shapes, ref_array, filenames = self.check_arrays_shape()
        if not different_shapes:
            return
        interpolated = None
        try:
            interpolated = await self.get_interpolated(filenames, ref_array)
        except Exception as err:
            mw.show_error(err)
        if interpolated:
            command = CommandUpdateInterpolated(mw, interpolated, "Interpolate files")
            mw.undoStack.push(command)

    async def get_interpolated(self, filenames: list[str], ref_file: np.ndarray) -> list[tuple[str, np.ndarray]]:
        mw = self.parent
        mw.ui.statusBar.showMessage('Interpolating...')
        mw.close_progress_bar()
        n_files = len(filenames)
        mw.open_progress_bar(max_value=n_files)
        mw.open_progress_dialog("Interpolating...", "Cancel", maximum=n_files)
        executor = ThreadPoolExecutor()
        if n_files >= 10_000:
            executor = ProcessPoolExecutor()
        mw.current_executor = executor
        with Manager() as manager:
            mw.break_event = manager.Event()
            with executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, interpolate, mw.ImportedArray[i], i, ref_file)
                                      for i in filenames]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                interpolated = await gather(*mw.current_futures)

        if mw.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            mw.close_progress_bar()
            mw.ui.statusBar.showMessage('Cancelled.')
            return
        mw.close_progress_bar()
        return interpolated

    def get_filenames_of_this_range(self, target_range_nm: tuple[int, int]) -> list[str]:
        filenames = []
        for i in self.parent.ImportedArray:
            spectrum = self.parent.ImportedArray[i]
            max_nm = spectrum.max(axis=0)[0]
            min_nm = spectrum.min(axis=0)[0]
            file_range = (min_nm, max_nm)
            if file_range != target_range_nm:
                filenames.append(i)
        return filenames

    def get_ref_file(self, target_range_nm: tuple[int, int]) -> np.ndarray:
        for i in self.parent.ImportedArray:
            spectrum = self.parent.ImportedArray[i]
            max_nm = spectrum.max(axis=0)[0]
            min_nm = spectrum.min(axis=0)[0]
            new_range = (min_nm, max_nm)
            if new_range == target_range_nm:
                return spectrum

    def check_ranges(self) -> list[int, int]:
        ranges = []
        for i in self.parent.ImportedArray.items():
            arr = i[1]
            max_nm = arr.max(axis=0)[0]
            min_nm = arr.min(axis=0)[0]
            new_range = (min_nm, max_nm)
            if new_range not in ranges:
                ranges.append(new_range)

        return ranges

    def check_arrays_shape(self) -> tuple[bool, np.ndarray, list[str]]:
        shapes = dict()
        for key, arr in self.parent.ImportedArray.items():
            shapes[key] = arr.shape[0]
        list_shapes = list(shapes.values())
        min_shape = np.min(list_shapes)
        max_shape = np.max(list_shapes)
        different_shapes = min_shape != max_shape
        ref_array = None
        filenames = []
        if different_shapes:
            counts = np.bincount(list_shapes)
            most_shape = np.argmax(counts)
            keys = [k for k, v in shapes.items() if v == most_shape]
            key = keys[0]
            ref_array = self.parent.ImportedArray[key]
            filenames = [k for k, v in shapes.items() if v != most_shape]
        return different_shapes, ref_array, filenames

    # endregion

    # region DESPIKE

    @asyncSlot()
    async def despike(self):
        try:
            await self._do_despike()
        except Exception as err:
            self.parent.show_error(err)

    @asyncSlot()
    async def _do_despike(self) -> None:
        if len(self.parent.ImportedArray) <= 0:
            MessageBox("Despike error.", 'Import spectra before despike.', self.parent, {'Ok'}).exec()
            return
        laser_wavelength = self.parent.ui.laser_wl_spinbox.value()
        maxima_count = self.parent.ui.maxima_count_despike_spin_box.value()
        fwhm_width = self.parent.ui.despike_fwhm_width_doubleSpinBox.value()
        self.parent.ui.statusBar.showMessage('Despiking...')
        self.parent.disable_buttons(True)
        self.parent.close_progress_bar()
        items_to_despike = self.get_items_to_despike()
        n_files = len(items_to_despike)
        self.parent.open_progress_bar(max_value=n_files)
        self.parent.open_progress_dialog("Despiking...", "Cancel", maximum=n_files)
        if self.parent.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            return
        self.parent.time_start = datetime.now()
        self.parent.current_executor = ThreadPoolExecutor()
        if n_files > 1000:
            self.parent.current_executor = ProcessPoolExecutor()
        fwhm_nm_df = self.parent.ui.input_table.model().get_column('FWHM, nm')
        with Manager() as manager:
            self.parent.break_event = manager.Event()
            with self.parent.current_executor as executor:
                self.parent.current_futures = [
                    self.parent.loop.run_in_executor(executor, subtract_cosmic_spikes_moll, i, fwhm_nm_df[i[0]],
                                                     laser_wavelength, maxima_count, fwhm_width)
                    for i in items_to_despike]
                for future in self.parent.current_futures:
                    future.add_done_callback(self.parent.progress_indicator)
                result_of_despike = await gather(*self.parent.current_futures)

        if self.parent.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            self.parent.close_progress_bar()
            self.parent.ui.statusBar.showMessage('Cancelled.')
            return
        despiked_list = [i for i in result_of_despike if i]
        self.parent.disable_buttons(False)
        self.parent.close_progress_bar()
        if despiked_list:
            command = CommandUpdateDespike(self.parent, despiked_list, "Despike")
            self.parent.undoStack.push(command)
        elif not self.parent.predict_logic.is_production_project:
            seconds = round((datetime.now() - self.parent.time_start).total_seconds())
            self.parent.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
            MessageBox("Despike finished.", 'No peaks found.', self.parent, {'Ok'}).exec()

    def get_items_to_despike(self) -> list[tuple[str, np.ndarray]]:
        current_row = self.parent.ui.input_table.selectionModel().currentIndex().row()
        if self.parent.ui.by_one_control_button.isChecked() and current_row != -1:
            current_spectrum_name = self.parent.ui.input_table.model().get_filename_by_row(current_row)
            items_to_despike = [(current_spectrum_name, self.parent.ImportedArray[current_spectrum_name])]
        elif self.parent.ui.by_group_control_button.isChecked() \
                and self.parent.ui.GroupsTable.selectionModel().currentIndex().row() != -1:
            current_group = self.parent.ui.GroupsTable.selectionModel().currentIndex().row() + 1
            filenames = self.get_names_of_group(current_group)
            items_to_despike = []
            for i in self.parent.ImportedArray.items():
                if i[0] in filenames:
                    items_to_despike.append(i)
        else:
            items_to_despike = self.parent.ImportedArray.items()
        return items_to_despike

    def get_names_of_group(self, group_number: int) -> list[str]:
        df = self.parent.ui.input_table.model().dataframe()
        rows = df.loc[df['Group'] == group_number]
        filenames = rows.index
        return filenames

    async def despike_history_add_plot(self) -> None:
        """
        Add arrows and BeforeDespike plot item to imported_plot for compare
        """
        # selected spectrum despiked
        mw = self.parent
        current_index = mw.ui.input_table.selectionModel().currentIndex()
        group_number = mw.ui.input_table.model().cell_data(current_index.row(), 2)
        arr = self.BeforeDespike[mw.current_spectrum_despiked_name]
        if self.curveDespikedHistory:
            mw.input_plot_widget_plot_item.removeItem(self.curveDespikedHistory)
        self.curveDespikedHistory = mw.get_curve_plot_data_item(arr, group_number)
        mw.input_plot_widget_plot_item.addItem(self.curveDespikedHistory, kargs=['ignoreBounds', 'skipAverage'])

        all_peaks = mw.ui.input_table.model().cell_data(current_index.row(), 3)
        all_peaks = all_peaks.split()
        text_peaks = []
        for i in all_peaks:
            i = i.replace(',', '')
            i = i.replace(' ', '')
            text_peaks.append(i)
        list_peaks = [float(s) for s in text_peaks]
        for i in list_peaks:
            idx = nearest_idx(arr[:, 0], i)
            y_peak = arr[:, 1][idx]
            arrow = ArrowItem(pos=(i, y_peak), angle=-45)
            mw.input_plot_widget_plot_item.addItem(arrow)

    # endregion

    # region CONVERT
    @asyncSlot()
    async def convert(self):
        try:
            await self._do_convert()
        except Exception as err:
            self.parent.show_error(err)

    @asyncSlot()
    async def _do_convert(self) -> None:
        mw = self.parent
        if len(mw.ImportedArray) > 1 and len(self.check_ranges()) > 1:
            MessageBox("Convert stopped.", 'Spectra must be interpolated before convert.', self.parent, {'Ok'}).exec()
            return
        different_shapes, _, _ = self.check_arrays_shape()
        if different_shapes:
            MessageBox("Convert stopped.", 'Files have different shapes, interpolation required.',
                       self.parent, {'Ok'}).exec()
            return

        mw.time_start = datetime.now()
        mw.ui.statusBar.showMessage('Converting...')
        mw.close_progress_bar()
        n_files = len(mw.ImportedArray)
        mw.open_progress_dialog("Converting nm to cm\N{superscript minus}\N{superscript one}...", "Cancel",
                                maximum=n_files)
        mw.open_progress_bar(max_value=n_files)

        x_axis = np.zeros(1)
        for _, arr in mw.ImportedArray.items():
            x_axis = arr[:, 0]
            break
        laser_nm = mw.ui.laser_wl_spinbox.value()
        near_idx = nearest_idx(x_axis, laser_nm + 5)
        max_ccd_value = mw.ui.max_CCD_value_spinBox.value()
        executor = ThreadPoolExecutor()
        if n_files >= 12_000:
            executor = ProcessPoolExecutor()
        mw.current_executor = executor
        with Manager() as manager:
            mw.break_event = manager.Event()
            with executor:
                mw.current_futures = [
                    mw.loop.run_in_executor(executor, convert, i, near_idx, laser_nm, max_ccd_value)
                    for i in mw.ImportedArray.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                converted = await gather(*mw.current_futures)
        if mw.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            mw.close_progress_bar()
            mw.ui.statusBar.showMessage('Cancelled.')
            return
        command = CommandConvert(mw, converted, "Convert to cm\N{superscript minus}\N{superscript one}")
        mw.undoStack.push(command)
        mw.close_progress_bar()
        if not mw.predict_logic.is_production_project:
            await self.update_range_cm()

    @asyncSlot()
    async def update_range_cm(self) -> None:
        if not self.ConvertedDict:
            return
        mw = self.parent
        mw.ui.statusBar.showMessage('Updating range...')
        mw.close_progress_bar()
        mw.open_progress_bar(max_value=len(self.ConvertedDict))
        time_before = datetime.now()
        x_axis = next(iter(self.ConvertedDict.values()))[:, 0]  # any of dict
        value_right = find_nearest(x_axis, mw.ui.cm_range_end.value())
        mw.ui.cm_range_end.setValue(value_right)
        factor = mw.ui.neg_grad_factor_spinBox.value()
        with Manager() as manager:
            mw.break_event = manager.Event()
            with ThreadPoolExecutor() as executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, find_fluorescence_beginning, i, factor)
                                      for i in self.ConvertedDict.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                result = await gather(*mw.current_futures)
        idx = max(result)
        value_left = find_nearest_by_idx(x_axis, idx)
        mw.ui.cm_range_start.setValue(value_left)
        seconds = round((datetime.now() - time_before).total_seconds())
        mw.ui.statusBar.showMessage('Range updated for ' + str(seconds) + ' sec.', 5000)
        mw.close_progress_bar()

    # endregion

    # region cut_first
    @asyncSlot()
    async def cut_first(self):
        try:
            await self._do_cut_first()
        except Exception as err:
            self.parent.show_error(err)

    @asyncSlot()
    async def _do_cut_first(self) -> None:
        if not self.ConvertedDict:
            MessageBox("Cut failed.", 'Files have different shapes, interpolation required.', self.parent,
                       {'Ok'}).exec()
            return
        x_axis = next(iter(self.ConvertedDict.values()))[:, 0]
        mw = self.parent
        value_start = mw.ui.cm_range_start.value()
        value_end = mw.ui.cm_range_end.value()
        if round(value_start, 5) == round(x_axis[0], 5) \
                and round(value_end, 5) == round(x_axis[-1], 5):
            MessageBox("Cut failed.", 'Cut range is equal to actual spectrum range. No need to cut.', self.parent,
                       {'Ok'}).exec()
            return
        mw.ui.statusBar.showMessage('Cut in progress...')
        mw.close_progress_bar()
        n_files = len(self.ConvertedDict)
        mw.open_progress_dialog("Cut in progress...", "Cancel",
                                maximum=n_files)
        mw.open_progress_bar(max_value=n_files)
        mw.time_start = datetime.now()
        executor = ThreadPoolExecutor()
        if n_files >= 16_000:
            executor = ProcessPoolExecutor()
        mw.current_executor = executor
        with Manager() as manager:
            mw.break_event = manager.Event()
            with executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, cut_spectrum, i, value_start, value_end)
                                      for i in self.ConvertedDict.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                cutted_arrays = await gather(*mw.current_futures)
        if mw.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            mw.close_progress_bar()
            mw.ui.statusBar.showMessage('Cancelled.')
            return
        if cutted_arrays:
            command = CommandCutFirst(mw, cutted_arrays, "Cut spectrum")
            mw.undoStack.push(command)
        mw.close_progress_bar()

    def cm_range_start_change_event(self, new_value: float) -> None:
        mw = self.parent
        mw.set_modified()
        if self.ConvertedDict:
            x_axis = next(iter(self.ConvertedDict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            mw.ui.cm_range_start.setValue(new_value)
        if new_value >= mw.ui.cm_range_end.value():
            mw.ui.cm_range_start.setValue(mw.ui.cm_range_start.minimum())
        mw.linearRegionCmConverted.setRegion((mw.ui.cm_range_start.value(), mw.ui.cm_range_end.value()))

    def cm_range_end_change_event(self, new_value: float) -> None:
        mw = self.parent
        mw.set_modified()
        if self.ConvertedDict:
            x_axis = next(iter(self.ConvertedDict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            mw.ui.cm_range_end.setValue(new_value)
        if new_value <= mw.ui.cm_range_start.value():
            mw.ui.cm_range_end.setValue(mw.ui.cm_range_end.maximum())
        mw.linearRegionCmConverted.setRegion((mw.ui.cm_range_start.value(), mw.ui.cm_range_end.value()))
    # endregion

    # region Normalizing
    @asyncSlot()
    async def normalize(self):
        try:
            await self._do_normalize()
        except Exception as err:
            self.parent.show_error(err)

    @asyncSlot()
    async def _do_normalize(self) -> None:
        if not self.CuttedFirstDict or len(self.CuttedFirstDict) == 0:
            MessageBox("Normalization stopped.", 'No cutted spectra for normalization.', self.parent, {'Ok'}).exec()
            return
        mw = self.parent
        mw.time_start = datetime.now()
        mw.ui.statusBar.showMessage('Normalization...')
        mw.close_progress_bar()
        n_files = len(self.CuttedFirstDict)
        mw.open_progress_bar(max_value=n_files)
        mw.open_progress_dialog("Normalization...", "Cancel", maximum=n_files)
        method = mw.ui.normalizing_method_comboBox.currentText()
        func = self.normalize_methods[method][0]
        params = None
        n_limit = self.normalize_methods[method][1]
        if method == 'EMSC':
            if mw.predict_logic.is_production_project:
                params = mw.predict_logic.y_axis_ref_EMSC, mw.ui.emsc_pca_n_spinBox.value()
            else:
                np_y_axis = get_emsc_average_spectrum(self.CuttedFirstDict.values())
                params = np_y_axis, mw.ui.emsc_pca_n_spinBox.value()
                mw.predict_logic.y_axis_ref_EMSC = np_y_axis
        executor = ThreadPoolExecutor()
        if n_files >= n_limit:
            executor = ProcessPoolExecutor()
        mw.current_executor = executor
        with Manager() as manager:
            mw.break_event = manager.Event()
            with mw.current_executor as executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, func, i, params)
                                      for i in self.CuttedFirstDict.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                normalized = await gather(*mw.current_futures)
        if mw.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            mw.close_progress_bar()
            mw.ui.statusBar.showMessage('Cancelled.')
            return
        if normalized:
            command = CommandNormalize(mw, normalized, method, "Normalize")
            mw.undoStack.push(command)
        mw.close_progress_bar()

    # endregion

    # region Smoothing
    @asyncSlot()
    async def smooth(self):
        try:
            await self.do_smooth()
        except Exception as err:
            self.parent.show_error(err)

    @asyncSlot()
    async def do_smooth(self) -> None:
        mw = self.parent
        if not self.NormalizedDict or len(self.NormalizedDict) == 0:
            MessageBox("Smoothing stopped.", 'No normalized spectra for smoothing.', self.parent, {'Ok'}).exec()
            return
        mw.time_start = datetime.now()
        mw.ui.statusBar.showMessage('Smoothing...')
        mw.close_progress_bar()
        n_files = len(self.NormalizedDict)
        mw.open_progress_bar(max_value=n_files)
        mw.open_progress_dialog("Smoothing...", "Cancel", maximum=n_files)
        snr_df = mw.ui.input_table.model().get_column('SNR')
        method = mw.ui.smoothing_method_comboBox.currentText()
        executor = ThreadPoolExecutor()
        func = self.smoothing_methods[method][0]
        n_samples_limit = self.smoothing_methods[method][1]
        params = self.smoothing_params(method)
        if n_files >= n_samples_limit:
            executor = ProcessPoolExecutor()
        mw.current_executor = executor
        with Manager() as manager:
            mw.break_event = manager.Event()
            with executor:
                if method == 'MLESG':
                    mw.current_futures = [mw.loop.run_in_executor(executor, func, i, (params[0], params[1],
                                                                                      snr_df.at[i[0]]))
                                          for i in self.NormalizedDict.items()]
                else:
                    mw.current_futures = [mw.loop.run_in_executor(executor, func, i, params)
                                          for i in self.NormalizedDict.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                smoothed = await gather(*mw.current_futures)
        if mw.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            mw.close_progress_bar()
            mw.ui.statusBar.showMessage('Cancelled.')
            return
        if smoothed:
            command = CommandSmooth(mw, smoothed, method, params, "Smooth")
            mw.undoStack.push(command)
        mw.close_progress_bar()

    def smoothing_params(self, method: str) -> int | tuple[int, int | str] | tuple[float, int, int]:
        mw = self.parent
        params = None
        match method:
            case 'EMD':
                params = mw.ui.emd_noise_modes_spinBox.value()
            case 'EEMD' | 'CEEMDAN':
                params = mw.ui.emd_noise_modes_spinBox.value(), mw.ui.eemd_trials_spinBox.value()
            case 'MLESG':
                fwhm_cm_df = mw.ui.input_table.model().get_column('FWHM, cm\N{superscript minus}\N{superscript one}')
                distance = np.max(fwhm_cm_df.values)
                sigma = mw.ui.sigma_spinBox.value()
                params = (distance, sigma, 0)
            case 'Savitsky-Golay filter':
                params = mw.ui.window_length_spinBox.value(), mw.ui.smooth_polyorder_spinBox.value()
            case 'Whittaker smoother':
                params = mw.ui.whittaker_lambda_spinBox.value()
            case 'Flat window':
                params = mw.ui.window_length_spinBox.value()
            case 'hanning' | 'hamming' | 'bartlett' | 'blackman':
                params = mw.ui.window_length_spinBox.value(), method
            case 'kaiser':
                params = mw.ui.window_length_spinBox.value(), mw.ui.kaiser_beta_doubleSpinBox.value()
            case 'Median filter' | 'Wiener filter':
                params = mw.ui.window_length_spinBox.value()
        return params

    # endregion

    # region baseline_correction
    @asyncSlot()
    async def baseline_correction(self):
        if not self.smoothed_spectra or len(self.smoothed_spectra) == 0:
            MessageBox("Baseline correction stopped.", 'No spectra for baseline correction.', self.parent,
                       {'Ok'}).exec()
            return
        try:
            await self.do_baseline_correction()
        except Exception as err:
            self.parent.show_error(err)

    @asyncSlot()
    async def do_baseline_correction(self) -> None:
        mw = self.parent
        mw.time_start = datetime.now()
        mw.ui.statusBar.showMessage('Baseline correction...')
        mw.close_progress_bar()
        n_files = len(self.smoothed_spectra)
        mw.open_progress_bar(max_value=n_files)
        mw.open_progress_dialog("Baseline correction...", "Cancel", maximum=n_files)
        method = mw.ui.baseline_correction_method_comboBox.currentText()
        executor = ThreadPoolExecutor()
        func = self.baseline_methods[method][0]
        n_samples_limit = self.baseline_methods[method][1]
        params = self.baseline_correction_params(method)
        if n_files >= n_samples_limit:
            executor = ProcessPoolExecutor()
        mw.current_executor = executor
        with Manager() as manager:
            mw.break_event = manager.Event()
            with executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, func, i, params)
                                      for i in self.smoothed_spectra.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                baseline_corrected = await gather(*mw.current_futures)
        if mw.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            mw.close_progress_bar()
            mw.ui.statusBar.showMessage('Cancelled.')
            return
        if baseline_corrected:
            command = CommandBaselineCorrection(mw, baseline_corrected, method, params, "Baseline correction")
            mw.undoStack.push(command)
        mw.close_progress_bar()
        if not mw.predict_logic.is_production_project:
            await self.update_range_baseline_corrected()

    def baseline_correction_params(self, method: str) -> float | list:
        params = None
        mw = self.parent
        match method:
            case 'Poly':
                params = mw.ui.polynome_degree_spinBox.value()
            case 'ModPoly' | 'iModPoly':
                params = [mw.ui.polynome_degree_spinBox.value(), mw.ui.grad_doubleSpinBox.value(),
                          mw.ui.n_iterations_spinBox.value()]
            case 'ExModPoly':
                # params = List()
                params = [float(mw.ui.polynome_degree_spinBox.value()), mw.ui.grad_doubleSpinBox.value(),
                           float(mw.ui.n_iterations_spinBox.value())]
                # [params.append(x) for x in params_]
            case 'Penalized poly':
                params = [mw.ui.polynome_degree_spinBox.value(), mw.ui.grad_doubleSpinBox.value(),
                          mw.ui.n_iterations_spinBox.value(), mw.ui.alpha_factor_doubleSpinBox.value(),
                          mw.ui.cost_func_comboBox.currentText()]
            case 'LOESS':
                poly_order = min(6, mw.ui.polynome_degree_spinBox.value())
                params = [poly_order, mw.ui.grad_doubleSpinBox.value(), mw.ui.n_iterations_spinBox.value(),
                          mw.ui.fraction_doubleSpinBox.value(), mw.ui.scale_doubleSpinBox.value()]
            case 'Goldindec':
                params = [mw.ui.polynome_degree_spinBox.value(), mw.ui.grad_doubleSpinBox.value(),
                          mw.ui.n_iterations_spinBox.value(), mw.ui.cost_func_comboBox.currentText(),
                          mw.ui.peak_ratio_doubleSpinBox.value(), mw.ui.alpha_factor_doubleSpinBox.value()]
            case 'Quantile regression':
                params = [mw.ui.polynome_degree_spinBox.value(), mw.ui.grad_doubleSpinBox.value(),
                          mw.ui.n_iterations_spinBox.value(), mw.ui.quantile_doubleSpinBox.value()]
            case 'AsLS' | 'iAsLS' | 'arPLS' | 'airPLS' | 'iarPLS' | 'asPLS' | 'psaLSA' | 'DerPSALSA' | 'MPLS':
                params = [mw.ui.lambda_spinBox.value(), mw.ui.p_doubleSpinBox.value(),
                          mw.ui.n_iterations_spinBox.value()]
            case 'drPLS':
                params = [mw.ui.lambda_spinBox.value(), mw.ui.p_doubleSpinBox.value(),
                          mw.ui.n_iterations_spinBox.value(), mw.ui.eta_doubleSpinBox.value()]
            case 'iMor' | 'MorMol' | 'AMorMol' | 'JBCD':
                params = [mw.ui.n_iterations_spinBox.value(), mw.ui.grad_doubleSpinBox.value()]
            case 'MPSpline':
                params = [mw.ui.lambda_spinBox.value(), mw.ui.p_doubleSpinBox.value(),
                          mw.ui.spline_degree_spinBox.value()]
            case 'Mixture Model':
                params = [mw.ui.lambda_spinBox.value(), mw.ui.p_doubleSpinBox.value(),
                          mw.ui.spline_degree_spinBox.value(), mw.ui.n_iterations_spinBox.value(),
                          mw.ui.grad_doubleSpinBox.value()]
            case 'IRSQR':
                params = [mw.ui.lambda_spinBox.value(), mw.ui.quantile_doubleSpinBox.value(),
                          mw.ui.spline_degree_spinBox.value(), mw.ui.n_iterations_spinBox.value()]
            case 'Corner-Cutting':
                params = mw.ui.n_iterations_spinBox.value()
            case 'RIA':
                params = mw.ui.grad_doubleSpinBox.value()
            case 'Dietrich':
                params = [mw.ui.num_std_doubleSpinBox.value(), mw.ui.polynome_degree_spinBox.value(),
                          mw.ui.grad_doubleSpinBox.value(), mw.ui.n_iterations_spinBox.value(),
                          mw.ui.interp_half_window_spinBox.value(), mw.ui.min_length_spinBox.value()]
            case 'Golotvin':
                params = [mw.ui.num_std_doubleSpinBox.value(), mw.ui.interp_half_window_spinBox.value(),
                          mw.ui.min_length_spinBox.value(), mw.ui.sections_spinBox.value()]
            case 'Std Distribution':
                params = [mw.ui.num_std_doubleSpinBox.value(), mw.ui.interp_half_window_spinBox.value(),
                          mw.ui.fill_half_window_spinBox.value()]
            case 'FastChrom':
                params = [mw.ui.interp_half_window_spinBox.value(), mw.ui.n_iterations_spinBox.value(),
                          mw.ui.min_length_spinBox.value()]
            case 'FABC':
                params = [mw.ui.lambda_spinBox.value(), mw.ui.num_std_doubleSpinBox.value(),
                          mw.ui.min_length_spinBox.value()]
            case 'OER' | 'Adaptive MinMax':
                params = mw.ui.opt_method_oer_comboBox.currentText()
        return params

    @asyncSlot()
    async def update_range_baseline_corrected(self) -> None:
        mw = self.parent
        if not self.baseline_corrected_not_trimmed_dict:
            return
        mw.ui.statusBar.showMessage('Updating range...')
        mw.close_progress_bar()
        mw.open_progress_bar(max_value=len(self.baseline_corrected_not_trimmed_dict))
        x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]  # any of dict
        with Manager() as manager:
            mw.break_event = manager.Event()
            with ThreadPoolExecutor() as executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, find_first_right_local_minimum, i)
                                      for i in self.baseline_corrected_not_trimmed_dict.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                result = await gather(*mw.current_futures)
        idx = int(np.percentile(result, 0.95))
        value_right = x_axis[idx]
        mw.ui.trim_end_cm.setValue(value_right)
        with Manager() as manager:
            mw.break_event = manager.Event()
            with ThreadPoolExecutor() as executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, find_first_left_local_minimum, i)
                                      for i in self.baseline_corrected_not_trimmed_dict.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                result = await gather(*mw.current_futures)
        idx = np.max(result)
        value_left = x_axis[idx]
        mw.ui.trim_start_cm.setValue(value_left)
        mw.close_progress_bar()

    # endregion

    # region final trim
    @asyncSlot()
    async def trim(self):
        try:
            await self._do_trim()
        except Exception as err:
            self.parent.show_error(err)

    @asyncSlot()
    async def _do_trim(self) -> None:
        mw = self.parent
        if not self.baseline_corrected_not_trimmed_dict:
            MessageBox("Trimming failed.", 'No baseline corrected plots.', self.parent, {'Ok'}).exec()
            return
        x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]
        value_start = mw.ui.trim_start_cm.value()
        value_end = mw.ui.trim_end_cm.value()
        if round(value_start, 5) == round(x_axis[0], 5) \
                and round(value_end, 5) == round(x_axis[-1], 5):
            MessageBox("Trim failed.", 'Trim range is equal to actual spectrum range. No need to cut.', self.parent,
                       {'Ok'}).exec()
            return
        mw.ui.statusBar.showMessage('Trimming in progress...')
        mw.close_progress_bar()
        n_files = len(self.baseline_corrected_not_trimmed_dict)
        mw.open_progress_dialog("Trimming in progress...", "Cancel",
                                maximum=n_files)
        mw.open_progress_bar(max_value=n_files)
        mw.time_start = datetime.now()
        executor = ThreadPoolExecutor()
        if n_files >= 16_000:
            executor = ProcessPoolExecutor()
        mw.current_executor = executor
        with Manager() as manager:
            mw.break_event = manager.Event()
            with executor:
                mw.current_futures = [mw.loop.run_in_executor(executor, cut_spectrum, i, value_start, value_end)
                                      for i in self.baseline_corrected_not_trimmed_dict.items()]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                cutted_arrays = await gather(*mw.current_futures)
        if mw.stateTooltip.wasCanceled() or environ['CANCEL'] == '1':
            mw.close_progress_bar()
            mw.ui.statusBar.showMessage('Cancelled.')
            return
        if cutted_arrays:
            command = CommandTrim(mw, cutted_arrays, "Trim spectrum")
            mw.undoStack.push(command)
        mw.close_progress_bar()

    def trim_start_change_event(self, new_value: float) -> None:
        mw = self.parent
        mw.set_modified()
        if self.baseline_corrected_not_trimmed_dict:
            x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            mw.ui.trim_start_cm.setValue(new_value)
        if new_value >= mw.ui.trim_end_cm.value():
            mw.ui.trim_start_cm.setValue(mw.ui.trim_start_cm.minimum())
        mw.linearRegionBaseline.setRegion((mw.ui.trim_start_cm.value(), mw.ui.trim_end_cm.value()))

    def trim_end_change_event(self, new_value: float) -> None:
        mw = self.parent
        mw.set_modified()
        if self.baseline_corrected_not_trimmed_dict:
            x_axis = next(iter(self.baseline_corrected_not_trimmed_dict.values()))[:, 0]
            new_value = find_nearest(x_axis, new_value)
            mw.ui.trim_end_cm.setValue(new_value)
        if new_value <= mw.ui.trim_start_cm.value():
            mw.ui.trim_end_cm.setValue(mw.ui.trim_end_cm.maximum())
        mw.linearRegionBaseline.setRegion((mw.ui.trim_start_cm.value(), mw.ui.trim_end_cm.value()))
    # endregion

    # region Average spectrum
    async def update_averaged(self) -> None:
        """
        Refresh averaged spectrum plot and self.averaged_dict.
        Returns
        -------
            None
        """
        mw = self.parent
        if not self.baseline_corrected_dict:
            return
        self.update_averaged_dict()
        await self.update_plot_item(self.averaged_dict.items(), 6)
        mw.fitting.update_template_combo_box()
        if not mw.predict_logic.is_production_project:
            mw.fitting.update_deconv_intervals_limits()

    def update_averaged_dict(self) -> None:
        """
        self.averaged_dict with key = group_id
        and values = averages spectrum of this group
        Returns
        -------
            None
        """
        mw = self.parent
        n_groups = mw.ui.GroupsTable.model().rowCount()
        self.averaged_dict.clear()
        averaging_method = mw.ui.average_method_cb.currentText()
        for i in range(n_groups):
            group_id = i + 1
            filenames = mw.ui.input_table.model().names_of_group(group_id)
            if len(filenames) == 0:
                continue
            arrays_list = [self.baseline_corrected_dict[x] for x in filenames]
            arrays_list_av = get_average_spectrum(arrays_list, averaging_method)
            self.averaged_dict[group_id] = arrays_list_av

    async def refresh_averaged_spectrum_plot(self) -> None:
        """
        Function updates averaged seaborn line plot.

        Returns
        -------
        None
        """
        if not self.baseline_corrected_dict:
            return
        error_method = self.parent.ui.average_errorbar_method_combo_box.currentText()
        error_level = self.parent.ui.average_level_spin_box.value()
        n_boot = self.parent.ui.average_n_boot_spin_box.value()
        ax = self.parent.ui.average_sns_plot_widget.canvas.axes
        ax.cla()
        self.av_df = self.create_averaged_df()
        colors = self.parent.ui.GroupsTable.model().groups_colors
        palette = sns.color_palette(colors)
        sns.lineplot(data=self.av_df, x='Raman shift, cm\N{superscript minus}\N{superscript one}',
                     y='Intensity, rel. un.', hue='Label', size='Label', style='Label', palette=palette,
                     sizes=self.parent.ui.GroupsTable.model().groups_width, errorbar=(error_method, error_level),
                     dashes=self.parent.ui.GroupsTable.model().groups_dashes, legend='full', ax=ax, n_boot=n_boot)
        self.parent.ui.average_sns_plot_widget.canvas.figure.tight_layout()
        self.parent.ui.average_sns_plot_widget.canvas.draw()

    def create_averaged_df(self) -> pd.DataFrame:
        """
        Function creates DataFrame for seaborn line plot.

        Returns
        -------
        av_df: pd.DataFrame
            DataFrame with 3 columns:
                Label: group_id
                Raman shift: cm-1 for x-axis
                Intensity, rel. un.: y-axis value
        """
        mw = self.parent
        av_df = pd.DataFrame(columns=['Label', 'Raman shift, cm\N{superscript minus}\N{superscript one}',
                                      'Intensity, rel. un.'])
        n_groups = mw.ui.GroupsTable.model().rowCount()
        for i in range(n_groups):
            group_id = i + 1
            filenames = mw.ui.input_table.model().names_of_group(group_id)
            n_spectrum = len(filenames)
            if n_spectrum == 0:
                continue
            arrays_y = [self.baseline_corrected_dict[x][:, 1] for x in filenames]
            arrays_y = np.array(arrays_y).flatten()
            x_axis = next(iter(self.baseline_corrected_dict.values()))[:, 0]
            x_axis = np.array(x_axis)
            x_axis = np.tile(x_axis, n_spectrum)
            label = mw.ui.GroupsTable.model().get_group_name_by_int(group_id)
            labels = [label] * arrays_y.size
            df = pd.DataFrame({'Label': labels, 'Raman shift, cm\N{superscript minus}\N{superscript one}': x_axis,
                               'Intensity, rel. un.': arrays_y})
            av_df = pd.concat([av_df, df])
        return av_df

    # endregion

    def compare_baseline_methods(self):
        """
        Delete later после диссертации
        Returns
        -------

        """
        mw = self.parent
        from numpy.polynomial.polynomial import polyval
        from modules.stages.preprocessing.functions.baseline_correction import baseline_penalized_poly, baseline_goldindec, ex_mod_poly, \
            baseline_imodpoly, baseline_modpoly
        from sklearn.metrics import mean_squared_error
        from matplotlib import pyplot as plt
        coefs = [1.27321927e+04, -7.62029464e+01, 2.81669762e-01, -5.66507200e-04, 6.52691243e-07,
                 -4.38362720e-10, 1.66449516e-13, -3.21147214e-17]
        x_axis, y_raman = mw.fitting.sum_array()
        baseline_test = polyval(x_axis, coefs)
        baseline_and_raman = baseline_test + y_raman
        test_arr = np.vstack((x_axis, baseline_and_raman)).T
        _, _, corrected_plus = ex_mod_poly(('1', test_arr), [7., 1e-10, 1000.])
        _, _, corrected_imod = baseline_imodpoly(('3', test_arr), [7, 1e-8, 1000])
        _, _, corrected_modp = baseline_modpoly(('3', test_arr), [7, 1e-10, 1000])
        _, _, corrected_quant = baseline_penalized_poly(('3', test_arr), [7, 1e-8, 1000, 0.999,
                                                                          'asymmetric_truncated_quadratic'])
        _, _, corrected_goldin = baseline_goldindec(('3', test_arr),
                                                    [7, 1e-9, 1000, 'asymmetric_truncated_quadratic', 0.5,
                                                     .999])
        corr_matrix_plus = np.corrcoef(y_raman, corrected_plus[:, 1])
        corr1 = corr_matrix_plus[0, 1] ** 2
        rms1 = mean_squared_error(corrected_plus[:, 1], y_raman, squared=False)
        rms2 = mean_squared_error(corrected_imod[:, 1], y_raman, squared=False)
        rms3 = mean_squared_error(corrected_modp[:, 1], y_raman, squared=False)
        rms6 = mean_squared_error(corrected_quant[:, 1], y_raman, squared=False)
        rms7 = mean_squared_error(corrected_goldin[:, 1], y_raman, squared=False)
        corrected_plus1 = corrected_plus[corrected_plus < 0]
        corrected_imod1 = corrected_imod[corrected_imod < 0]
        corrected_modp1 = corrected_modp[corrected_modp < 0]
        corrected_quant1 = corrected_quant[corrected_quant < 0]
        corrected_goldin1 = corrected_goldin[corrected_goldin < 0]
        area0 = np.trapz(corrected_plus1)
        area1 = np.trapz(corrected_imod1)
        area2 = np.trapz(corrected_modp1)
        area3 = np.trapz(corrected_quant1)
        area4 = np.trapz(corrected_goldin1)
        print('r2 EX-Mod-Poly+ ', corr1, 'rms ', rms1, 'area ', area0)
        corr_matrix_imod = np.corrcoef(y_raman, corrected_imod[:, 1])
        corr2 = corr_matrix_imod[0, 1] ** 2
        print('r2 I-Mod-Poly ', corr2, 'area ', area1)
        corr_matrix_modp = np.corrcoef(y_raman, corrected_modp[:, 1])
        corr3 = corr_matrix_modp[0, 1] ** 2
        print('r2 Mod-Poly ', corr3, 'area ', area2)
        corr_matrix_quant = np.corrcoef(y_raman, corrected_quant[:, 1])
        corr6 = corr_matrix_quant[0, 1] ** 2
        print('r2 Penalized Poly ', corr6, 'area ', area3)
        corr_matrix_goldin = np.corrcoef(y_raman, corrected_goldin[:, 1])
        corr7 = corr_matrix_goldin[0, 1] ** 2
        print('r2 Goldindec ', corr7, 'area ', area4)
        fig, ax = plt.subplots()
        ax.plot(x_axis, y_raman, label='Synthesized spectrum', color='black')
        ax.plot(x_axis, corrected_plus[:, 1],
                label='Ex-Mod-Poly. $\mathregular{r^2}$=' + str(np.round(corr1 * 100, 3)) + '. RMSE=' + str(
                    np.round(rms1, 2)), color='red')
        ax.plot(x_axis, corrected_imod[:, 1],
                label='I-Mod-Poly. $\mathregular{r^2}$=' + str(np.round(corr2 * 100, 3)) + '. RMSE=' + str(
                    np.round(rms2, 2)), color='green')
        ax.plot(x_axis, corrected_modp[:, 1],
                label='Mod-Poly. $\mathregular{r^2}$=' + str(np.round(corr3 * 100, 3)) + '. RMSE=' + str(
                    np.round(rms3, 2)), color='blue', dashes=[6, 2])
        ax.plot(x_axis, corrected_quant[:, 1],
                label='Penalized Poly. $\mathregular{r^2}$=' + str(np.round(corr6 * 100, 3)) + '. RMSE=' + str(
                    np.round(rms6, 2)), color='c', dashes=[8, 4])
        ax.plot(x_axis, corrected_goldin[:, 1],
                label='Goldindec. $\mathregular{r^2}$=' + str(np.round(corr7 * 100, 3)) + '. RMSE=' + str(
                    np.round(rms7, 2)), color='m', dashes=[8, 4])
        ax.legend()
        ax.grid()
        plt.show()

    def ex_mod_poly_build_dev(self):
        """
        Delete later после диссертации
        Returns
        -------

        """
        from matplotlib import pyplot as plt
        from modules.stages.preprocessing.functions.baseline_correction import ex_mod_poly
        before = datetime.now()
        devs = []
        for k, v in self.smoothed_spectra.items():
            _, baseline_plus_tru, corrected_tru, pitches = ex_mod_poly((k, v), [12., 1e-12, 1000.])
            devs.append(pitches)
        print(datetime.now() - before)
        fig, ax = plt.subplots()
        for i in devs:
            ax.plot(i)
        ax.grid()
        plt.show()
