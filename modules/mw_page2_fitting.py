from copy import deepcopy
from asyncio import gather
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager
from datetime import datetime
from logging import info
from typing import Any

import numpy as np
from asyncqtpy import asyncSlot
from lmfit import Parameters, Model
from lmfit.model import ModelResult
from numpy import ndarray
from pandas import DataFrame, Series, MultiIndex, concat
from pyqtgraph import PlotCurveItem, ROI, mkBrush, PlotDataItem
from qtpy.QtCore import QTimer
from qtpy.QtGui import QMouseEvent, QColor
from scipy.signal import argrelmin
from qfluentwidgets import MessageBox

from modules.default_values import peak_shape_params_limits, fitting_methods, peak_shapes_params
from modules import packed_current_line_parameters, models_params_splitted_batch, fit_model_batch, \
    eval_uncert, models_params_splitted, get_curve_for_deconvolution, curve_pen_brush_by_style, set_roi_size, \
    set_roi_size_pos, get_average_spectrum
from modules.undo_redo import CommandAfterBatchFitting, CommandAfterFitting, CommandAfterGuess, \
    CommandDeconvLineDragged, CommandDeconvLineTypeChanged, CommandEndIntervalChanged, CommandStartIntervalChanged
from modules import nearest_idx, find_nearest
from modules.stages.preprocessing import cut_full_spectrum, cut_axis
from modules import split_by_borders, fitting_model, fit_model, clustering_lines_intervals, intervals_by_borders, \
    create_intervals_data, params_func_legend, legend_from_float, update_fit_parameters, guess_peaks


class FittingLogic:

    def __init__(self, parent):
        self.all_ranges_clustered_x0_sd = None
        self.intervals_data = None
        self.timer_fill = None
        self.rad = None
        self.parent = parent
        self.is_template = False
        self.averaged_spectrum = None
        self.peak_shapes_params = peak_shapes_params()
        self.peak_shape_params_limits = peak_shape_params_limits()
        self.fitting_methods = fitting_methods()
        self.current_spectrum_deconvolution_name = ''
        self.updating_fill_curve_idx = None
        self.dragged_line_parameters = None
        self.prev_dragged_line_parameters = None
        self.data_curve = None
        self.sum_curve = None
        self.residual_curve = None
        self.data_style = None
        self.sum_style = None
        self.residual_style = None
        self.sigma3_style = None
        self.sigma3_fill = None
        self.report_result = dict()
        self.sigma3 = dict()
        self.sigma3_top = PlotCurveItem(name='sigma3_top')
        self.sigma3_bottom = PlotCurveItem(name='sigma3_bottom')
        # parameters to turn on/off pushing UndoStack command during another redo/undo command executing
        self.CommandDeconvLineDraggedAllowed = True

    @asyncSlot()
    async def do_batch_fit(self) -> None:
        """
        Fitting line's parameters to all spectrum files
        1. Get x, y axes of current spectrum
        2. Prepare data before creating model and parameters
        3. Create params
        4. Create model
        5. Fit model to y_data
        """
        main_window = self.parent
        main_window.time_start = datetime.now()
        main_window.ui.statusBar.showMessage('Batch fitting...')
        main_window.close_progress_bar()
        baseline_corrected_spectra = {}
        filenames = main_window.preprocessing.baseline_corrected_dict.keys() \
                   & main_window.ui.input_table.model().filenames()
        # fitted_keys = main_window.ui.fit_params_table.model().fitted_filenames()
        # filenames = all_keys if samples == 'All' else all_keys ^ fitted_keys
        if main_window.predict_logic.is_production_project \
                and main_window.ui.deconvoluted_dataset_table_view.model().rowCount() != 0:
            filenames_in_dataset = list(main_window.ui.deconvoluted_dataset_table_view.model().column_data(1).values)
            for key, arr in main_window.preprocessing.baseline_corrected_dict.items():
                if key not in filenames_in_dataset:
                    baseline_corrected_spectra[key] = [arr]
        else:
            for key in filenames:
                try:
                    baseline_corrected_spectra[key] = [main_window.preprocessing.baseline_corrected_dict[key]]
                except KeyError:
                    pass
        splitted_arrays = self._split_array_for_fitting(baseline_corrected_spectra)
        idx_type_param_count_legend_func = self.prepare_data_fitting()
        list_params_full = self._fitting_params_batch(idx_type_param_count_legend_func, baseline_corrected_spectra)
        x_y_models_params = models_params_splitted_batch(splitted_arrays, list_params_full,
                                                         idx_type_param_count_legend_func)
        method_full_name = main_window.ui.fit_opt_method_comboBox.currentText()
        method = self.fitting_methods[method_full_name]
        executor = ProcessPoolExecutor()
        main_window.current_executor = executor
        key_x_y_models_params = []
        for key, item in x_y_models_params.items():
            for x_axis, y_axis, model, interval_params in item:
                key_x_y_models_params.append((key, x_axis, y_axis, model, interval_params))
        intervals = len(key_x_y_models_params)
        main_window.open_progress_bar(max_value=intervals if intervals > 1 else 0)
        main_window.open_progress_dialog("Batch Fitting...", "Cancel", maximum=intervals if intervals > 1 else 0)
        with Manager() as manager:
            main_window.break_event = manager.Event()
            with main_window.current_executor as ex:
                main_window.current_futures = [main_window.loop.run_in_executor(ex, fit_model_batch, model, y, params,
                                                                                x, method, key)
                                               for key, x, y, model, params in key_x_y_models_params]
                for future in main_window.current_futures:
                    future.add_done_callback(main_window.progress_indicator)
                result = await gather(*main_window.current_futures)
        if main_window.cancelled_by_user():
            return
        main_window.close_progress_bar()
        main_window.ui.statusBar.showMessage('Calculating uncertainties...')
        n_files = len(result)
        main_window.open_progress_bar(max_value=n_files if n_files > 1 else 0)
        main_window.open_progress_dialog("Calculating uncertainties...", "Cancel",
                                         maximum=n_files if n_files > 1 else 0)
        # if not self.parent.predict_logic.is_production_project
        executor = ProcessPoolExecutor()
        main_window.current_executor = executor
        with Manager() as manager:
            main_window.break_event = manager.Event()
            with executor:
                main_window.current_futures = [
                    main_window.loop.run_in_executor(main_window.current_executor, eval_uncert, i)
                    for i in result]
                for future in main_window.current_futures:
                    future.add_done_callback(main_window.progress_indicator)
                dely = await gather(*main_window.current_futures)
        main_window.close_progress_bar()
        main_window.open_progress_bar(max_value=0)
        main_window.open_progress_dialog("updating data...", "Cancel", maximum=0)
        command = CommandAfterBatchFitting(main_window, result, idx_type_param_count_legend_func, dely,
                                           "Fitted spectrum in batch ")
        main_window.undoStack.push(command)
        main_window.close_progress_bar()

    @asyncSlot()
    async def do_fit(self) -> None:
        """
        Fitting line's parameters to current spectrum data
        1. Get x, y axes of current spectrum
        2. Prepare data before creating model and parameters
        3. Create params
        4. Create model
        5. Fit model to y_data
        """
        main_window = self.parent
        main_window.time_start = datetime.now()
        main_window.ui.statusBar.showMessage('Fitting...')
        main_window.close_progress_bar()

        spec_name = self.current_spectrum_deconvolution_name
        arr = self.array_of_current_filename_in_deconvolution()
        splitted_arrays = self._split_array_for_fitting({spec_name: [arr]})[spec_name]
        intervals = len(splitted_arrays)
        main_window.open_progress_bar(max_value=intervals if intervals > 1 else 0)
        main_window.open_progress_dialog("Fitting...", "Cancel", maximum=intervals if intervals > 1 else 0)
        idx_type_param_count_legend_func = self.prepare_data_fitting()
        params = self.fitting_params(idx_type_param_count_legend_func, np.max(arr[:, 1]), arr[:, 0][0],
                                     arr[:, 0][-1])
        x_y_models_params, _ = models_params_splitted(splitted_arrays, params, idx_type_param_count_legend_func)
        method_full_name = main_window.ui.fit_opt_method_comboBox.currentText()
        method = self.fitting_methods[method_full_name]
        executor = ProcessPoolExecutor()
        main_window.current_executor = executor
        with executor:
            main_window.current_futures = [
                main_window.loop.run_in_executor(executor, fit_model, i[2], i[1], i[3], i[0], method)
                for i in x_y_models_params]
            for future in main_window.current_futures:
                future.add_done_callback(main_window.progress_indicator)
            result = await gather(*main_window.current_futures)
        if main_window.cancelled_by_user():
            return
        command = CommandAfterFitting(main_window, result, idx_type_param_count_legend_func, spec_name,
                                      "Fitted spectrum %s" % spec_name)
        main_window.undoStack.push(command)
        main_window.close_progress_bar()
        main_window.ui.statusBar.showMessage('Fitting completed', 10000)

    @asyncSlot()
    async def do_auto_guess(self, line_type: str) -> None:
        """
        Автоматический подбор линий к модели. Результат отображается только для шаблона.
        Peaks are added at position of global extremum of data-baseline with previous peaks subtracted.
        Таблица параметров для всех спектров очищается. Результаты прошлых fit очищаются.
        Если на момент начала анализа уже есть линии в таблице линий, то поиск линий начнется не с нуля,
         а с этого состава.
        Спектр делится на интервалы.
        Guess method:
            Если выбран 'Average', то анализируется только усредненный спектр по всем группам.
            Если выбран 'Average groups', то анализируются усредненные по каждой отдельной группе
            Если выбран 'All', то анализируются усредненные по всем спектрам из левой таблицы.
        Для 'Average groups' и 'All' собираются линии x0. По каждому интервалу анализируется количество линий.
        Итоговое количество линий для каждого интервала определяется методом кластеризации k-means.
        На вход в алгоритм k-means подается количество линий

        После k-means полученный состав линий опять прогоняется через Fit и мы получаем итоговую модель.

        В процессе анализа на каждый параметр линий накладывается ряд ограничений.
         a - амплитуда от 0 до максимального значения в интервале. Интервал зависит от x0 и dx
         x0 - положение максимума линии. Изначально определяется по положению максимума residual +- 1 см-1. Или после
            k-means границы задаются мин-макс значением в кластере
         dx, dx_left - полуширина линии левая/правая, максимальное значение задается в Max peak HWHM:, минимальное
            определяется из наименьшего FWHM CM-1 / 2 в таблице
        Остальные параметры имеют границы указанные в peak_shape_params_limits

        @param line_type: str
        @return: None
        """
        mw = self.parent
        mw.time_start = datetime.now()
        mw.ui.statusBar.showMessage('Guessing peaks...')
        mw.close_progress_bar()
        parameters_to_guess = self._parameters_to_guess(line_type)
        sliced_arrays = self._arrays_for_peak_guess()
        n_files = len(sliced_arrays)
        n_files = 0 if n_files == 1 else n_files
        mw.open_progress_bar(max_value=n_files)
        mw.open_progress_dialog("Fitting each spectrum", "Cancel", maximum=n_files)
        mw.current_executor = ProcessPoolExecutor()
        with Manager() as manager:
            mw.break_event = manager.Event()
            with mw.current_executor:
                mw.current_futures = [mw.loop.run_in_executor(mw.current_executor, guess_peaks, arr,
                                                              parameters_to_guess, mw.break_event)
                                      for arr in sliced_arrays]
                for future in mw.current_futures:
                    future.add_done_callback(mw.progress_indicator)
                latest_guess_result = await gather(*mw.current_futures)
        print('guess_peaks done.')
        info('guess_peaks done.')
        if mw.cancelled_by_user():
            return
        if mw.ui.guess_method_cb.currentText() != 'Average':
            mw.close_progress_bar()
            mw.open_progress_bar(max_value=0)
            mw.open_progress_dialog("Creating mutual model...", "Cancel")
            x_y_models_params = await self._analyze_guess_results(latest_guess_result,
                                                                  parameters_to_guess['param_names'], line_type)
            mw.close_progress_bar()
            mw.progressBar.setValue(0)
            intervals = len(x_y_models_params)
            mw.open_progress_bar(max_value=intervals if intervals > 1 else 0)
            mw.open_progress_dialog("Fitting...", "Cancel", maximum=intervals if intervals > 1 else 0)
            mw.current_executor = ProcessPoolExecutor()
            with Manager() as manager:
                mw.break_event = manager.Event()
                with mw.current_executor:
                    mw.current_futures = [
                        mw.loop.run_in_executor(mw.current_executor, fit_model, i[2], i[1], i[3], i[0],
                                                parameters_to_guess['method'])
                        for i in x_y_models_params]
                    for future in mw.current_futures:
                        future.add_done_callback(mw.progress_indicator)
                    result = await gather(*mw.current_futures)
        else:
            result = latest_guess_result
        if mw.cancelled_by_user():
            return
        mw.progressBar.setMinimum(0)
        mw.progressBar.setMaximum(0)
        command = CommandAfterGuess(mw, result, line_type, len(parameters_to_guess['param_names']))
        mw.undoStack.push(command)
        mw.close_progress_bar()

    @asyncSlot()
    async def curve_type_changed(self, line_type_new: str, line_type_old: str, row: int) -> None:
        """
        Find curves by name = idx, and redraw it
        add/delete parameters
        """
        idx = self.parent.ui.deconv_lines_table.model().row_data(row).name
        await self.switch_to_template()
        command = CommandDeconvLineTypeChanged(self.parent, line_type_new, line_type_old, idx,
                                               "Change line type for curve idx %s" % idx)
        self.parent.undoStack.push(command)

    @asyncSlot()
    async def switch_to_template(self, _: str = 'Average') -> None:
        self.update_single_deconvolution_plot(self.parent.ui.template_combo_box.currentText(), True, True)
        self.redraw_curves_for_filename()
        self.show_all_roi()
        self.set_rows_visibility()
        self.show_current_report_result()
        self.draw_sum_curve()
        self.draw_residual_curve()
        self.update_sigma3_curves('')

    def _split_array_for_fitting(self, arrays: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
        """
        The function cuts the spectra at intervals specified by the user.
        If checked 'interval' cut all spectra by this range
        elif 'Split by intervals': use intervals from table
        else: not selected range use input arrays as out without changes.

        Parameters
        ----------
        arrays: dict[str, list[np.ndarray]]
            filename, spectrum array with x_axis and y_axis

        Returns
        -------
        out: dict[str, list[np.ndarray]]
            same arrays but sliced
        """
        split_arrays = {}
        if self.parent.ui.interval_checkBox.isChecked():
            for key, arr in arrays.items():
                split_arrays[key] = [cut_full_spectrum(arr[0], self.parent.ui.interval_start_dsb.value(),
                                                       self.parent.ui.interval_end_dsb.value())]
        elif self.parent.ui.intervals_gb.isChecked():
            borders = list(self.parent.ui.fit_borders_TableView.model().column_data(0))
            x_axis = next(iter(self.parent.preprocessing.baseline_corrected_dict.values()))[:, 0]
            intervals_idx = intervals_by_borders(borders, x_axis, idx=True)
            for key, arr in arrays.items():
                split_arrays[key] = split_by_borders(arr[0], intervals_idx)
        else:
            split_arrays = arrays
        return split_arrays

    def prepare_data_fitting(self) -> list[tuple[int, str, int, str, callable]]:
        """
        Get prepared data to create model and parameters

        Returns
        -------
        func_legend : list[tuple[func, str]]
            using to create fit Model. func is a line function from spec_functions/ peak_shapes, str is curve legend
        idx_type_paramcount_legend : list[tuple[int, str, int, str]]
            using to create Parameters of model. idx - curve index, type - curve type, param_count - number of
             parameters of line_type, legend - curve legend by index
        """
        # get dataframe with active lines - idx | Legend, Type, Style
        line_types = self.parent.ui.deconv_lines_table.model().get_visible_line_types()
        idx_type_param_count_legend = []
        for i in line_types.index:
            ser = line_types.loc[i]
            if 'add_params' in self.peak_shapes_params[ser.Type]:
                param_count = 3 + len(self.peak_shapes_params[ser.Type]['add_params'])
            else:
                param_count = 3
            legend = 'Curve_%s_' % i
            idx_type_param_count_legend.append((i, ser.Type, param_count, legend,
                                                self.peak_shapes_params[ser.Type]['func']))
        return idx_type_param_count_legend

    def _fitting_params_batch(self, idx_type_param_count_legend_func: list[tuple[int, str, int, str, callable]],
                              arrays: dict[str, list[np.ndarray]]) -> dict:
        try:
            x_axis = next(iter(arrays.values()))[0][:, 0]
        except StopIteration:
            x_axis = next(iter(self.parent.preprocessing.baseline_corrected_dict.values()))[:, 0]
        params_mutual = self.fitting_params(idx_type_param_count_legend_func, 1000., x_axis[0], x_axis[-1])
        list_params_full = {}
        for key, item in arrays.items():
            this_key_params = deepcopy(params_mutual)
            y_axis = item[0][:, 1]
            for par in this_key_params:
                par_splitted = par.split('_', 2)
                if par_splitted[-1] != 'a':
                    continue
                dx_right_param_name = par_splitted[0] + '_' + par_splitted[1] + '_dx'
                dx_right_max = this_key_params[dx_right_param_name].max
                dx_left_param_name = par_splitted[0] + '_' + par_splitted[1] + '_dx_left'
                if dx_left_param_name in this_key_params:
                    dx_left_max = this_key_params[dx_left_param_name].max
                else:
                    dx_left_max = this_key_params[dx_right_param_name].max
                x0_min = this_key_params[par_splitted[0] + '_' + par_splitted[1] + '_x0'].min
                x0_max = this_key_params[par_splitted[0] + '_' + par_splitted[1] + '_x0'].max
                x0_left = x0_min - dx_left_max
                x0_right = x0_max + dx_right_max
                arg_x0_left = nearest_idx(x_axis, x0_left)
                arg_x0_right = nearest_idx(x_axis, x0_right)
                y_max_in_range = np.amax(y_axis[arg_x0_left:arg_x0_right])
                this_key_params[par].max = y_max_in_range
                this_key_params[par].value = this_key_params[par].init_value = y_max_in_range / 2.
            list_params_full[key] = this_key_params
        return list_params_full

    def fitting_params(self, list_idx_type: list[tuple[int, str, int, str, callable]], bound_max_a: float,
                       bound_min_x0: float, bound_max_x0: float) -> Parameters:
        """
        Set parameters for fit model

        Parameters
        ---------
        list_idx_type : list[tuple[int, str, int, str, callable]
            idx - line index
            line_type - 'Gaussian' for example
            param_count - number of parameters for line type. example for Gaussian = 3, pearson4 = 5
            legend - for parameter name
            _callable - not used here

        bound_max_a : float
            maximal Intensity of y_axis spectrum

        bound_min_x0 : float
            minimal x of x_axis (first value of x_axis)

        bound_max_x0 : float
            maximal x of x_axis (last value of x_axis)

        Returns
        -------
        params : Parameters()
            initial values of fit parameters
        """
        bound_min_a = 0.
        if self.current_spectrum_deconvolution_name == '':
            bound_min_dx = np.max(self.parent.ui.input_table.model().column_data(6)) / 2
        else:
            row_data = self.parent.ui.input_table.model().row_data_by_index(self.current_spectrum_deconvolution_name)
            bound_min_dx = row_data['FWHM, cm\N{superscript minus}\N{superscript one}'] / 2
        params = Parameters()
        i = 0
        for idx, line_type, param_count, legend, _ in list_idx_type:
            add_params_len = param_count - 3
            params_from_table = self.current_line_parameters(idx, '')
            param_names = ['a', 'x0', 'dx']
            for j in range(add_params_len):
                param_names.append(self.peak_shapes_params[line_type]['add_params'][j])
            # value must be between bounds
            for param_name in param_names:
                v = params_from_table[param_name]
                # bounds correction
                bound_min_v = None
                bound_max_v = None
                if param_name == 'a':
                    bound_min_v = bound_min_a
                    bound_max_v = bound_max_a
                elif param_name == 'x0':
                    bound_min_v = bound_min_x0
                    bound_max_v = bound_max_x0
                elif param_name == 'dx' or param_name == 'dx_left':
                    bound_min_v = bound_min_dx
                    bound_max_v = (bound_max_x0 - bound_min_x0) / 2
                elif param_name in self.peak_shape_params_limits:
                    bound_min_v = self.peak_shape_params_limits[param_name][0]
                    bound_max_v = self.peak_shape_params_limits[param_name][1]
                min_v = params_from_table['min_' + param_name]
                max_v = params_from_table['max_' + param_name]
                if bound_min_v is not None:
                    min_v = bound_min_v if min_v < bound_min_v else min_v
                if bound_max_v is not None:
                    max_v = bound_max_v if max_v > bound_max_v else max_v
                v = min_v + 1e-4 if v <= min_v else v
                v = max_v - 1e-4 if v >= max_v else v
                if min_v == max_v:
                    max_v += 1e-4
                if param_name == 'a':
                    max_v = bound_max_a
                params.add(legend + param_name, v, min=min_v, max=max_v)
            i += param_count
        return params

    def current_line_parameters(self, index: int, filename: str | None = None) -> dict | None:
        if filename is None:
            filename = "" if self.is_template or self.current_spectrum_deconvolution_name == '' \
                else self.current_spectrum_deconvolution_name
        if filename not in self.parent.ui.fit_params_table.model().filenames and not filename == "":
            return None
        df_params = self.parent.ui.fit_params_table.model().get_df_by_multiindex((filename, index))
        line_type = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(index, 'Type')
        if df_params.empty:
            return None
        return packed_current_line_parameters(df_params, line_type, self.peak_shapes_params)

    def array_of_current_filename_in_deconvolution(self) -> np.ndarray | None:
        """
        @return: 2D массив спектра, который отображается на графике в данный момент
        """
        current_spectrum_name = self.current_spectrum_deconvolution_name
        arr = None
        if self.is_template:
            if self.parent.ui.template_combo_box.currentText() == 'Average':
                arr = self.averaged_spectrum
            elif self.parent.preprocessing.averaged_dict and self.parent.preprocessing.averaged_dict != {}:
                arr = self.parent.preprocessing.averaged_dict[
                    int(self.parent.ui.template_combo_box.currentText().split('.')[0])]
        elif current_spectrum_name in self.parent.preprocessing.baseline_corrected_dict:
            arr = self.parent.preprocessing.baseline_corrected_dict[current_spectrum_name]
        else:
            return None
        return arr

    def set_parameters_after_fit_for_spectrum(self, fit_result: ModelResult, filename: str) -> None:
        """
        Set fitted parameters value after fitting

        Parameters
        ---------
        fit_result : lmfit.model.ModelResult

        filename : str
            filename of spectrum model was fitted to
        """
        for key, param in fit_result.params.items():
            idx_param_name = key.split('_', 2)
            idx = int(idx_param_name[1])
            param_name = idx_param_name[2]
            table_model = self.parent.ui.fit_params_table.model()
            table_model.set_parameter_value(filename, idx, param_name, 'Value', param.value, False)
            table_model.set_parameter_value(filename, idx, param_name, 'Max value', param.max, False)
            table_model.set_parameter_value(filename, idx, param_name, 'Min value', param.min, False)

    def set_rows_visibility(self) -> None:
        """
        Show only rows of selected curve in fit_params_table. Other rows hiding.
        If not selected - show first curve's params
        """
        row_count = self.parent.ui.fit_params_table.model().rowCount()
        if row_count == 0:
            return
        filename = '' if self.is_template else self.current_spectrum_deconvolution_name
        row_line = self.parent.ui.deconv_lines_table.selectionModel().currentIndex().row()
        row = row_line if row_line != -1 else 0
        idx = self.parent.ui.deconv_lines_table.model().row_data(row).name
        row_id_to_show = self.parent.ui.fit_params_table.model().row_number_for_filtering((filename, idx))
        if row_id_to_show is None:
            return
        for i in range(row_count):
            self.parent.ui.fit_params_table.setRowHidden(i, True)
        for i in row_id_to_show:
            self.parent.ui.fit_params_table.setRowHidden(i, False)

    def show_current_report_result(self) -> None:
        """
        Show report result of currently showing spectrum name
        """
        filename = '' if self.is_template else self.current_spectrum_deconvolution_name
        report = self.report_result[filename] if filename in self.report_result else ''
        self.parent.ui.report_text_edit.setText(report)

    def update_template_combo_box(self) -> None:
        """
        template_combo_box always has 0-element 'Average'
        next elements are groups names
        Returns
        -------
        None
        """
        self.parent.ui.template_combo_box.clear()
        self.parent.ui.template_combo_box.addItem('Average')
        if not self.parent.preprocessing.averaged_dict or self.parent.ui.GroupsTable.model().rowCount() == 0:
            return
        for i in self.parent.preprocessing.averaged_dict:
            group_name = self.parent.ui.GroupsTable.model().row_data(i - 1)['Group name']
            self.parent.ui.template_combo_box.addItem(str(i) + '. ' + group_name)
        self.parent.ui.template_combo_box.currentTextChanged.connect(lambda: self.switch_to_template())

    def update_deconv_intervals_limits(self) -> None:
        mw = self.parent
        if not mw.preprocessing.averaged_dict or 1 not in mw.preprocessing.averaged_dict:
            return
        x_axis = mw.preprocessing.averaged_dict[1][:, 0]
        if x_axis.size == 0:
            return
        min_cm = np.min(x_axis)
        max_cm = np.max(x_axis)
        if mw.ui.interval_start_dsb.value() < min_cm or mw.ui.interval_start_dsb.value() > max_cm:
            mw.ui.interval_start_dsb.setValue(min_cm)
        if mw.ui.interval_end_dsb.value() < min_cm or mw.ui.interval_end_dsb.value() > max_cm:
            mw.ui.interval_end_dsb.setValue(max_cm)
        mw.ui.interval_start_dsb.setMinimum(min_cm)
        mw.ui.interval_start_dsb.setMaximum(max_cm)
        mw.ui.interval_end_dsb.setMinimum(min_cm)
        mw.ui.interval_end_dsb.setMaximum(max_cm)

    @asyncSlot()
    async def dec_table_double_clicked(self):
        """
        When selected item in list.
        Change current spectrum in deconv_plot_widget
        """
        current_index = self.parent.ui.dec_table.selectionModel().currentIndex()
        current_spectrum_name = self.parent.ui.dec_table.model().cell_data(current_index.row())
        if current_spectrum_name not in self.parent.ui.fit_params_table.model().filenames:
            self.hide_all_roi()
        else:
            self.show_all_roi()
        self.current_spectrum_deconvolution_name = current_spectrum_name
        self.update_single_deconvolution_plot(current_spectrum_name)
        self.redraw_curves_for_filename()
        self.set_rows_visibility()
        self.draw_sum_curve()
        self.draw_residual_curve()
        self.show_current_report_result()
        self.update_sigma3_curves(current_spectrum_name)

    def add_line_params_from_template_batch(self, keys: list[str]) -> None:
        key = keys[0]
        self.add_line_params_from_template(key)
        df_a = self.parent.ui.fit_params_table.model().dataframe()
        tuples = []
        for i in df_a.index:
            if i[0] == '':
                tuples.append(i)
        df_a = df_a.loc['']
        mi = MultiIndex.from_tuples(tuples, names=('filename', 'line_index', 'param_name'))
        df_a.index = mi
        df_c = df_a.copy(deep=True)
        for key in keys:
            tuples_b = []
            for filename, line_idx, param_name in tuples:
                tuples_b.append((key, line_idx, param_name))
            df_b = df_a.copy(deep=True)
            mi = MultiIndex.from_tuples(tuples_b, names=('filename', 'line_index', 'param_name'))
            df_b.index = mi
            df_c = concat([df_c, df_b])
        self.parent.ui.fit_params_table.model().set_dataframe(df_c)
        self.parent.ui.fit_params_table.model().sort_index()

    def add_line_params_from_template(self, filename: str | None = None) -> None:
        """
        When selected item in list.
        update parameters for lines of current spectrum filename
        if no parameters for filename - copy from template and update limits of amplitude parameter
        """
        model = self.parent.ui.fit_params_table.model()
        model.delete_rows_by_filenames(filename)
        if filename is None:
            filename = self.current_spectrum_deconvolution_name
        df_a = self.parent.ui.fit_params_table.model().dataframe()
        tuples = []
        for i in df_a.index:
            if i[0] == '':
                tuples.append(i)
        df_a = df_a.loc['']
        tuples_b = []
        for _, line_idx, param_name in tuples:
            tuples_b.append((filename, line_idx, param_name))
        df_b = df_a.copy(deep=True)
        mi = MultiIndex.from_tuples(tuples_b, names=('filename', 'line_index', 'param_name'))
        df_b.index = mi
        model.concat_df(df_b)

    def create_deconvoluted_dataset_new(self) -> Any | None:
        """
        Создает датасет на основе данных разделенных линий.
        Колонки - амплитуда и x0 (опционально)
        Строки - значения для каждого спектра
        Returns
        -------

        """
        filenames = self.parent.ui.fit_params_table.model().filenames
        line_legends = self.parent.ui.deconv_lines_table.model().column_data(0).sort_values()
        line_indexes = list(line_legends.index)
        line_legends = list(line_legends)
        line_legends_a = []
        line_legends_x0 = []
        for i in line_legends:
            line_legends_a.append(i + '_a')
            if self.parent.ui.include_x0_checkBox.isChecked():
                line_legends_x0.append(i + '_x0')
        if self.parent.ui.include_x0_checkBox.isChecked():
            line_legends = np.concatenate((line_legends_a, line_legends_x0))
        else:
            line_legends = line_legends_a
        params = self.parent.ui.fit_params_table.model().column_data(1)
        df = DataFrame(columns=line_legends)
        class_ids = []
        if self.parent.predict_logic.is_production_project:
            for i in filenames:
                class_ids.append(0)
        filename_group = self.parent.ui.input_table.model().column_data(2)
        for filename in filenames:
            if not self.parent.predict_logic.is_production_project:
                try:
                    group_id = filename_group.loc[filename]
                    class_ids.append(group_id)
                except KeyError:
                    class_ids.append(1)
            values_a = []
            for i in line_indexes:
                v = params.loc[(filename, i, 'a')]
                values_a.append(v)
            values_x0 = []
            if self.parent.ui.include_x0_checkBox.isChecked():
                for i in line_indexes:
                    v = params.loc[(filename, i, 'x0')]
                    values_x0.append(v)
                values = np.concatenate((values_a, values_x0))
            else:
                values = values_a
            df2 = DataFrame(np.array(values).reshape(1, -1), columns=line_legends)
            df = concat([df, df2], ignore_index=True)
        df2 = DataFrame({'Class': class_ids, 'Filename': filenames})
        df = concat([df2, df], axis=1)
        if self.parent.predict_logic.is_production_project \
                and self.parent.ui.deconvoluted_dataset_table_view.model().rowCount() != 0:
            df_old = self.parent.ui.deconvoluted_dataset_table_view.model().dataframe()
            df = concat([df_old, df])
        df.reset_index(drop=True)
        self.update_ignore_features_table()
        return df

    def update_ignore_features_table(self):
        features = self.parent.ui.deconvoluted_dataset_table_view.model().features
        df = DataFrame({'Feature': features}, columns=['Feature', 'Score', 'P value'])
        self.parent.ui.ignore_dataset_table_view.model().set_dataframe(df)
        self.parent.ui.ignore_dataset_table_view.model().set_checked({})

    # region GUESS
    def _parameters_to_guess(self, line_type: str) -> dict:
        """
        Prepare parameters for send to guess_peaks()

        Parameters
        ---------
        line_type : str
            {'Gaussian', 'Split Gaussian', ... etc}. Line type chosen by user in Guess button.
             see peak_shapes_params() in default_values.py

        Returns
        -------
        out :
            dict with keys:
            'func': callable; Function for peak shape calculation. Look peak_shapes_params() in default_values.py.
            'param_names': list[str]; List of parameter names. Example: ['a', 'x0', 'dx'].
            'init_model_params': list[float]; Initial values of parameters for a given spectrum and line type.
            'min_fwhm': float; the minimum value FWHM, determined from the input table (the minimum of all).
            'method': str; Optimization method, see fitting_methods() in default_values.py.
            'params_limits': dict[str, tuple[float, float]]; see peak_shape_params_limits() in default_values.py.
            'noise_level': float; limit for peak detection.
                Peaks with amplitude less than noise_level will not be detected.
            'max_dx': float; Maximal possible value for dx. For all peaks.
            'gamma_factor: float;
                 from 0. to 1. limit for max gamma value set by: max_v = min(dx_left, dx_right) * gamma_factor

        # The following parameters are empty if there are no lines. If at the beginning of the analysis there are
        # lines already created by the user, then the parameters will be filled.

            'func_legend': list[tuple]; - (callable func, legend),
                func - callable; Function for peak shape calculation. Look peak_shapes_params() in default_values.py.
                legend - prefix for the line in the model. All lines in a heap. As a result,
                    we select only those that belong to the current interval.
            'params': Parameters(); parameters of existing lines.
            'used_legends': list[str]; already used legends (used wave-numbers)
                ['k977dot15_', 'k959dot68_', 'k917dot49_']. We control it because model cant have lines with duplicate
                 legends (same x0 position lines)
            'used_legends_dead_zone': dict; keys - used_legends, values - tuple (idx - left idx, right idx - idx).
                dead zone size to set y_residual to 0.
                {'k977dot15_': (1, 1), 'k959dot68_': (2, 1), 'k917dot49_': (3, 4)}
        """
        func = self.peak_shapes_params[line_type]['func']
        param_names = self._param_names(line_type)
        init_model_params = self._init_model_params(line_type)
        min_fwhm = self.parent.ui.input_table.model().min_fwhm()
        mean_snr = np.mean(self.parent.ui.input_table.model().get_column('SNR').values)
        noise_level = np.max(self.averaged_spectrum[:, 1]) / mean_snr
        noise_level = max(noise_level, self.parent.ui.max_noise_level_dsb.value())
        method = self.fitting_methods[self.parent.ui.fit_opt_method_comboBox.currentText()]
        max_dx = self.parent.ui.max_dx_dsb.value()
        params_limits = self.peak_shape_params_limits
        params_limits['l_ratio'] = (0., self.parent.ui.l_ratio_doubleSpinBox.value())
        # The following parameters are empty if there are no lines. If at the beginning of the analysis there are
        # lines already created by the user, then the parameters will be filled.
        visible_lines = self.parent.ui.deconv_lines_table.model().get_visible_line_types()
        func_legend = []
        params = Parameters()
        used_legends = []
        used_legends_dead_zone = {}
        if len(visible_lines) > 0 and not self.parent.ui.interval_checkBox.isChecked():
            static_parameters = param_names, min_fwhm, self.peak_shape_params_limits
            func_legend, params, used_legends, used_legends_dead_zone = self._initial_template(visible_lines, func,
                                                                                               static_parameters,
                                                                                               init_model_params)

        return {'func': func, 'param_names': param_names, 'init_model_params': init_model_params, 'min_fwhm': min_fwhm,
                'method': method, 'params_limits': params_limits, 'noise_level': noise_level, 'max_dx': max_dx,
                'func_legend': func_legend, 'params': params, 'used_legends': used_legends,
                'used_legends_dead_zone': used_legends_dead_zone}

    def _param_names(self, line_type: str) -> list[str]:
        """
        Function returns list of peak shape parameters names

        Parameters
        ---------
        line_type : str
            {'Gaussian', 'Split Gaussian', ... etc}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        out : list[str]
            ['a', 'x0', 'dx' + 'add_params']
        """
        param_names = ['a', 'x0', 'dx']
        if line_type in self.peak_shapes_params and 'add_params' in self.peak_shapes_params[line_type]:
            for i in self.peak_shapes_params[line_type]['add_params']:
                param_names.append(i)
        return param_names

    def _init_model_params(self, line_type: str) -> list[float]:
        """
        Function returns list of initial parameter values.
        For param names ['a', 'x0', 'dx'] will return [float, float, float]

        Parameters
        ---------
        line_type : str
            {'Gaussian', 'Split Gaussian', ... etc}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        out : list[float]
        """
        init_params = self.initial_peak_parameters(line_type)
        init_model_params = []
        for i, j in init_params.items():
            if i != 'x_axis':
                init_model_params.append(j)
        return init_model_params

    def initial_peak_parameters(self, line_type: str) -> dict:
        """
        The function returns dict with initial parameters and x_axis.
            'x_axis': np.ndarray, 'a': float, 'x0': float, 'dx': float, + additional parameters

        Parameters
        ----------
        line_type: str
            input array to slice

        line_type: str
            {'Gaussian', 'Split Gaussian', ... etc}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        out : dict
            'x_axis': np.ndarray, 'a': float, 'x0': float, 'dx': float, + additional parameters
        """
        x_axis = np.array(range(920, 1080))
        a = 100.0
        x0 = 1000.0
        dx = 10.0
        arr = None
        if self.parent.ui.template_combo_box.currentText() == 'Average':
            arr = self.averaged_spectrum
            dx = np.max(self.parent.ui.input_table.model().column_data(6)) * np.pi / 2
        elif self.current_spectrum_deconvolution_name != '':
            arr = self.parent.preprocessing.baseline_corrected_dict[self.current_spectrum_deconvolution_name]
            row_data = self.parent.ui.input_table.model().row_data_by_index(self.current_spectrum_deconvolution_name)
            dx = row_data['FWHM, cm\N{superscript minus}\N{superscript one}'] * np.pi / 2
        elif self.current_spectrum_deconvolution_name == '' \
                and self.parent.ui.template_combo_box.currentText() != 'Average':
            array_id = int(self.parent.ui.template_combo_box.currentText().split('.')[0])
            arr = self.parent.preprocessing.averaged_dict[array_id]
            dx = np.max(self.parent.ui.input_table.model().column_data(6)) * np.pi / 2

        if arr is not None:
            x_axis = arr[:, 0]
            a = np.max(arr[:, 1]) / 2
            x0 = np.mean(x_axis)

        result = {'x_axis': x_axis,
                  'a': np.round(a, 5),
                  'x0': np.round(x0, 5),
                  'dx': np.round(dx, 5)}
        if 'add_params' not in self.peak_shapes_params[line_type]:
            return result
        # add additional parameters into result dict
        add_params = self.peak_shapes_params[line_type]['add_params']
        for param_name in add_params:
            match param_name:
                case 'dx_left':
                    result[param_name] = np.round(dx - dx * 0.1, 5)
                case 'gamma':
                    result[param_name] = 0.5
                case 'skew':
                    result[param_name] = 0.0
                case 'l_ratio':
                    result[param_name] = 0.6
                case 'expon' | 'beta':
                    result[param_name] = 1.0
                case 'alpha' | 'q':
                    result[param_name] = 0.1
        return result

    def _initial_template(self, visible_lines: DataFrame, func, static_parameters: tuple[list[str], float, dict],
                          init_model_params: list[float]) -> tuple[list[tuple], Parameters, list[str], dict]:
        """
        Prepare parameters of existing template lines. New lines will be guessed adding to this template.

        Parameters
        ---------
        visible_lines : DataFrame
            Raman lines table with columns: Legend, Type, Style.
        func : callable
            Function for peak shape calculation. Look peak_shapes_params() in default_values.py.
        static_parameters : tuple[list[str], float, dict]
            param_names : list[str]; List of parameter names. Example: ['a', 'x0', 'dx'].
            min_fwhm, : float; the minimum value FWHM, determined from the input table (the minimum of all).
            peak_shape_params_limits: dict[str, tuple[float, float]]; see peak_shape_params_limits()
                in default_values.py.
            gamma_factor: float; from 0. to 1. limit for max gamma value set by:
                max_v = min(dx_left, dx_right) * gamma_factor
        init_model_params: list[float]; Initial values of parameters for a given spectrum and line type.

        Returns
        -------
        func_legend: list[tuple]; - (callable func, legend),
            func - callable; Function for peak shape calculation. Look peak_shapes_params() in default_values.py.
            legend - prefix for the line in the model. All lines in a heap.
            As a result, we select only those that belong to the current interval;

        params: Parameters();
            parameters of existing lines.

        used_legends: list[str];
            already used legends (used wave-numbers)
            ['k977dot15_', 'k959dot68_', 'k917dot49_']. We control it because model cant have lines with duplicate
             legends (same x0 position lines)

        used_legends_dead_zone: dict;
            keys - used_legends, values - tuple (idx - left idx, right idx - idx).
            dead zone size to set y_residual to 0.
            {'k977dot15_': (1, 1), 'k959dot68_': (2, 1), 'k917dot49_': (3, 4)}
        """
        func_legend = []
        params = Parameters()
        used_legends = []
        x_axis = next(iter(self.parent.preprocessing.baseline_corrected_dict.values()))[:, 0]
        max_dx = self.parent.ui.max_dx_dsb.value()
        used_legends_dead_zone = {}
        for i in visible_lines.index:
            x0_series = self.parent.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'x0'))
            a_series = self.parent.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'a'))
            dx_series = self.parent.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'dx'))
            legend = legend_from_float(x0_series.Value)
            func_legend.append((func, legend))
            init_params = init_model_params.copy()
            init_params[0] = a_series.Value
            init_params[1] = x0_series.Value
            dx_right = min(float(dx_series.Value), max_dx)
            if 'dx_left' in static_parameters[0]:
                dx_left_series = self.parent.ui.fit_params_table.model().get_df_by_multiindex(('', i, 'dx_left'))
                dx_left = min(dx_left_series.Value, max_dx)
            else:
                dx_left = dx_right
            used_legends.append(legend)
            y_max_in_range = a_series['Max value']
            x0_arg = nearest_idx(x_axis, x0_series.Value)
            x0_arg_dx_l = nearest_idx(x_axis, x0_series.Value - dx_left / 2.)
            x0_arg_dx_r = nearest_idx(x_axis, x0_series.Value + dx_right / 2.)
            used_legends_dead_zone[legend] = (x0_arg - x0_arg_dx_l, x0_arg_dx_r - x0_arg)
            dynamic_parameters = legend, init_params, y_max_in_range, dx_left, dx_right
            params = update_fit_parameters(params, static_parameters, dynamic_parameters)
        return func_legend, params, used_legends, used_legends_dead_zone

    def _arrays_for_peak_guess(self) -> list[ndarray]:
        """
        The function return sliced spectra for guess peaks.
        If Guess method == 'Average' return only Averaged spectrum
        elif 'Average groups' - return spectra of average groups
        elif 'All' - return all spectra

        Returns
        -------
        out: list[ndarray]
            sliced spectra for guess peaks
        """
        guess_method = self.parent.ui.guess_method_cb.currentText()
        arrays = {'Average': [self.averaged_spectrum]}

        if guess_method == 'Average groups':
            arrays = {}
            for key, arr in self.parent.preprocessing.averaged_dict.items():
                arrays[key] = [arr]
        elif guess_method == 'All':
            arrays = {}
            for key, arr in self.parent.preprocessing.baseline_corrected_dict.items():
                arrays[key] = [arr]
        arrays_for_guess = self._split_array_for_fitting(arrays).values()
        result = []
        for i in arrays_for_guess:
            for j in i:
                result.append(j)
        return result

    @asyncSlot()
    async def _analyze_guess_results(self, result: list[ModelResult], param_names: list[str], line_type: str) \
            -> ndarray | Model:
        """
        The function creates structure for final fitting. After final fitting we have final template for deconvolution.

        Parameters
        ----------
        result: list[ModelResult]
            results of fitting for every spectrum and every interval range.

        param_names: list[str]
            Names of peak_shape parameters. Standard params are 'a', 'x0' and 'dx_right'. Other param names given from
             peak_shapes_params() in default_values.py

        line_type: str
            {'Gaussian', 'Split Gaussian', ... etc}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        x_y_models_params: list[tuple[np.ndarray, np.ndarray, Model, Parameters]]
            result of self.models_params_after_guess()
        """
        self.intervals_data = self._create_intervals_data(result, len(param_names))
        self.all_ranges_clustered_x0_sd = await clustering_lines_intervals(self.intervals_data,
                                                                           self.parent.ui.max_dx_dsb.value())
        x_y_models_params = self._models_params_after_guess(self.all_ranges_clustered_x0_sd, param_names, line_type)
        return x_y_models_params

    def _create_intervals_data(self, result: list[ModelResult], peak_n_params: int) \
            -> dict[str, dict[str, tuple[float, float] | list]]:
        """
        The function creates intervals range and create_intervals_data.

        Parameters
        ----------
        result: list[ModelResult]
            results of fitting for every spectrum and every interval range.

        peak_n_params: int
            number of parameters of peak shape. Look peak_shapes_params() in default_values.py

        Returns
        -------
        data_by_intervals: dict[dict]
            with keys 'interval': (start, end), 'x0': list, lines_count': []
        """
        # create intervals for create_intervals_data()
        if self.parent.ui.interval_checkBox.isChecked():
            intervals = [(self.parent.ui.interval_start_dsb.value(), self.parent.ui.interval_end_dsb.value())]
        elif not self.parent.ui.intervals_gb.isChecked():
            x_axis = next(iter(self.parent.preprocessing.baseline_corrected_dict.values()))[:, 0]
            intervals = [(x_axis[0], x_axis[-1])]
        else:
            borders = list(self.parent.ui.fit_borders_TableView.model().column_data(0))
            x_axis = next(iter(self.parent.preprocessing.baseline_corrected_dict.values()))[:, 0]
            intervals = intervals_by_borders(borders, x_axis)

        data_by_intervals = create_intervals_data(result, peak_n_params, intervals)
        return data_by_intervals

    def _models_params_after_guess(self, all_ranges_clustered_x0_sd: list[np.ndarray], param_names: list[str],
                                   line_type: str) -> list[tuple[np.ndarray, np.ndarray, Model, Parameters]]:
        """
        Preparation of sliced x_axis, y_axis of averaged spectrum, fit model and parameters from a list of wave numbers,
         divided into ranges.

        Parameters
        ----------
        all_ranges_clustered_x0_sd: list[ndarray]
            result of estimate_n_lines_in_range(x0, hwhm) for each range
            2D array with 2 columns: center of cluster x0 and standard deviation of each cluster

        param_names: list[str]
            Names of peak_shape parameters. Standard params are 'a', 'x0' and 'dx_right'. Other param names given from
             peak_shapes_params() in default_values.py

        line_type: str
            {'Gaussian', 'Split Gaussian', ... etc}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        x_y_models_params: list[tuple[np.ndarray, np.ndarray, Model, Parameters]]
            tuples with x_axis, y_axis, fitting model, params for fitting model. For each cm-1 range.
        """
        sliced_average_array_by_ranges = self._split_array_for_fitting({'Average': [self.averaged_spectrum]})['Average']

        # form static_params for all ranges
        init_params = self.initial_peak_parameters(line_type)
        max_dx = self.parent.ui.max_dx_dsb.value()
        min_hwhm = self.parent.ui.input_table.model().min_fwhm() / 2.
        func = self.peak_shapes_params[line_type]['func']
        static_params = init_params, max_dx, min_hwhm, func, self.peak_shape_params_limits, param_names, \
            self.parent.ui.l_ratio_doubleSpinBox.value()
        x_y_model_params = []
        for i, cur_range_clustered_x0_sd in enumerate(all_ranges_clustered_x0_sd):
            n_array = sliced_average_array_by_ranges[i]
            x_axis = n_array[:, 0]
            y_axis = n_array[:, 1]
            params, func_legend = params_func_legend(cur_range_clustered_x0_sd, n_array, static_params)
            model = fitting_model(func_legend)
            x_y_model_params.append((x_axis, y_axis, model, params))
        return x_y_model_params

    def auto_search_borders(self) -> None:
        if self.averaged_spectrum is None:
            return
        x = self.averaged_spectrum[:, 0]
        y = self.averaged_spectrum[:, 1]
        minima_idx = argrelmin(y, order=int(y.shape[0] / 10))
        x_minima = np.round(x[minima_idx], 3)
        self.parent.ui.fit_borders_TableView.model().add_auto_found_borders(x_minima)

    # endregion

    # region Curves
    def add_deconv_curve_to_plot(self, params: dict, idx: int, style: dict, line_type: str) \
            -> None:
        x0 = params['x0']
        dx = params['dx']
        if 'x_axis' not in params:
            params['x_axis'] = self.x_axis_for_line(x0, dx)
        x_axis = params['x_axis']
        full_amp_line, x_axis, _ = self.curve_y_x_idx(line_type, params, x_axis, idx)
        self.create_roi_curve_add_to_plot(full_amp_line, x_axis, idx, style, params)

    def x_axis_for_line(self, x0: float, dx: float) -> np.ndarray:
        if self.parent.preprocessing.baseline_corrected_dict:
            return next(iter(self.parent.preprocessing.baseline_corrected_dict.values()))[:, 0]
        else:
            return np.array(range(int(x0 - dx * 20), int(x0 + dx * 20)))

    def curve_y_x_idx(self, line_type: str, params: dict | None, x_axis: np.ndarray = None, idx: int = 0) \
            -> tuple[np.ndarray, np.ndarray, int]:
        """Returns y-axis and x_axis of deconvolution line

        Inputs
        ------
        line_type : str
            name of line from self.deconv_line_params.keys()
        params : dict
            with parameters value - a, x0, dx and etc.
        x_axis : np.ndarray
            corresponding x_axis of y_axis line

        Returns
        -------
        out : tuple[np.ndarray, np.ndarray]
            y, x axis signal
        """
        a = params['a']
        dx = params['dx']
        x0 = params['x0']
        if x_axis is None:
            x_axis = self.x_axis_for_line(x0, dx)

        y = None
        func = self.peak_shapes_params[line_type]['func']
        func_param = []
        if 'add_params' in self.peak_shapes_params[line_type]:
            add_params = self.peak_shapes_params[line_type]['add_params']
            for i in add_params:
                if i in params:
                    func_param.append(params[i])
        if not func_param:
            y = func(x_axis, a, x0, dx)
        elif len(func_param) == 1:
            y = func(x_axis, a, x0, dx, func_param[0])
        elif len(func_param) == 2:
            y = func(x_axis, a, x0, dx, func_param[0], func_param[1])
        elif len(func_param) == 3:
            y = func(x_axis, a, x0, dx, func_param[0], func_param[1], func_param[2])
        if self.parent.ui.interval_checkBox.isChecked() and y is not None:
            x_axis, idx_start, idx_end = cut_axis(x_axis, self.parent.ui.interval_start_dsb.value(),
                                                  self.parent.ui.interval_end_dsb.value())
            y = y[idx_start: idx_end + 1]
        return y, x_axis, idx

    def create_roi_curve_add_to_plot(self, full_amp_line: np.ndarray | None, x_axis: np.ndarray | None, idx: int,
                                     style: dict, params: dict) -> None:
        a = params['a']
        x0 = params['x0']
        dx = params['dx']
        if full_amp_line is None:
            return
        n_array = np.vstack((x_axis, full_amp_line)).T
        curve = get_curve_for_deconvolution(n_array, idx, style)
        curve.sigClicked.connect(self.curve_clicked)
        self.parent.deconvolution_plotItem.addItem(curve)
        roi = ROI([x0, 0], [dx, a], resizable=False, removable=True, rotatable=False, movable=False, pen='transparent')

        roi.addTranslateHandle([0, 1])
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        self.parent.deconvolution_plotItem.addItem(roi)
        curve.setParentItem(roi)
        curve.setPos(-x0, 0)
        if not self.parent.ui.deconv_lines_table.model().checked()[idx] and roi is not None:
            roi.setVisible(False)
        roi.sigRegionChangeStarted.connect(lambda checked=None, index=idx: self.curve_roi_pos_change_started(index,
                                                                                                             roi))
        roi.sigRegionChangeFinished.connect(
            lambda checked=None, index=idx: self.curve_roi_pos_change_finished(index, roi))
        roi.sigRegionChanged.connect(lambda checked=None, index=idx: self.curve_roi_pos_changed(index, roi, curve))

    def curve_clicked(self, curve: PlotCurveItem, _event: QMouseEvent) -> None:
        if self.updating_fill_curve_idx is not None \
                and self.updating_fill_curve_idx in self.parent.ui.deconv_lines_table.model().dataframe().index:
            curve_style = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(
                self.updating_fill_curve_idx,
                'Style')
            self.update_curve_style(self.updating_fill_curve_idx, curve_style)
        idx = int(curve.name())
        self.select_curve(idx)

    def update_curve_style(self, idx: int, style: dict) -> None:
        pen, brush = curve_pen_brush_by_style(style)
        items_matches = self.deconvolution_data_items_by_idx(idx)
        if items_matches is None:
            return
        curve, _ = items_matches
        curve.setPen(pen)
        curve.setBrush(brush)

    def deconvolution_data_items_by_idx(self, idx: int) -> tuple[PlotCurveItem, ROI] | None:
        data_items = self.parent.deconvolution_plotItem.listDataItems()
        if len(data_items) == 0:
            return None
        curve = None
        roi = None
        for i in data_items:
            if i.name() == idx:
                curve = i
                roi = i.parentItem()
                break
        return curve, roi

    def select_curve(self, idx: int) -> None:
        row = self.parent.ui.deconv_lines_table.model().row_by_index(idx)
        self.parent.ui.deconv_lines_table.selectRow(row)
        self.set_rows_visibility()
        self.start_fill_timer(idx)

    def start_fill_timer(self, idx: int) -> None:
        self.rad = 0.
        self.updating_fill_curve_idx = idx
        self.timer_fill = QTimer()
        self.timer_fill.timeout.connect(self.update_curve_fill_realtime)
        self.timer_fill.start(10)

    def update_curve_fill_realtime(self):
        self.rad += 0.02
        idx = self.updating_fill_curve_idx
        if idx not in list(self.parent.ui.deconv_lines_table.model().indexes):
            self.deselect_selected_line()
            return
        sin_v = np.abs(np.sin(self.rad))
        if self.parent.ui.deconv_lines_table.model().rowCount() == 0:
            return
        curve_style = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Style')
        fill_color = curve_style['color'] if curve_style['use_line_color'] else curve_style['fill_color']
        fill_color.setAlphaF(sin_v)
        brush = mkBrush(fill_color)
        curve, _ = self.deconvolution_data_items_by_idx(idx)
        if not curve:
            self.deselect_selected_line()
            return
        curve.setBrush(brush)

    def deselect_selected_line(self) -> None:
        if self.timer_fill is not None:
            self.timer_fill.stop()
            if self.updating_fill_curve_idx is not None \
                    and self.updating_fill_curve_idx in list(self.parent.ui.deconv_lines_table.model().indexes):
                curve_style = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(
                    self.updating_fill_curve_idx,
                    'Style')
                self.update_curve_style(self.updating_fill_curve_idx, curve_style)
                self.updating_fill_curve_idx = None

    def curve_roi_pos_change_started(self, index: int, roi: ROI) -> None:
        params = self.current_line_parameters(index)
        if not params:
            return
        a = params['a']
        x0 = params['x0']
        dx = params['dx']
        color = QColor(self.parent.theme_colors['secondaryColor'])
        color.setAlphaF(0.5)
        roi.setPen(color)
        self.dragged_line_parameters = a, x0, dx

    def curve_roi_pos_change_finished(self, index: int, roi: ROI) -> None:
        params = self.current_line_parameters(index)
        if not params:
            return
        a = params['a']
        x0 = params['x0']
        dx = params['dx']
        roi_a = roi.size().y()
        roi_x0 = roi.pos().x()
        roi_dx = roi.size().x()

        if (a, x0, dx) != self.dragged_line_parameters and (a, x0, dx) != self.prev_dragged_line_parameters and \
                (a, x0, dx) != (roi_a, roi_x0, roi_dx) and self.CommandDeconvLineDraggedAllowed:
            command = CommandDeconvLineDragged(self.parent, (a, x0, dx), self.dragged_line_parameters,
                                               roi, "Edit line %s" % index)
            roi.setPen('transparent')
            self.parent.undoStack.push(command)
            self.prev_dragged_line_parameters = self.dragged_line_parameters
            self.dragged_line_parameters = a, x0, dx

    def curve_roi_pos_changed(self, index: int, roi: ROI, curve: PlotCurveItem) -> None:
        dx = roi.size().x()
        x0 = roi.pos().x()
        new_height = roi.pos().y() + roi.size().y()
        params = self.current_line_parameters(index)

        if not params:
            return
        model = self.parent.ui.fit_params_table.model()
        filename = '' if self.is_template else self.current_spectrum_deconvolution_name
        line_type = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(index, 'Type')

        if x0 < params['min_x0']:
            model.set_parameter_value(filename, index, 'x0', 'Min value', x0 - 1)
        if x0 > params['max_x0']:
            model.set_parameter_value(filename, index, 'x0', 'Max value', x0 + 1)

        if new_height < params['min_a']:
            model.set_parameter_value(filename, index, 'a', 'Min value', new_height)
        if new_height > params['max_a']:
            model.set_parameter_value(filename, index, 'a', 'Max value', new_height)

        if dx < params['min_dx']:
            model.set_parameter_value(filename, index, 'dx', 'Min value', dx - dx / 2)
        if dx > params['max_dx']:
            model.set_parameter_value(filename, index, 'dx', 'Max value', dx + dx / 2)

        if np.round(new_height, 5) == np.round(params['a'], 5) and np.round(dx, 5) == np.round(params['dx'], 5) \
                and np.round(params['x0'], 5) == np.round(x0, 5):
            return
        model.set_parameter_value(filename, index, 'dx', 'Value', dx)
        model.set_parameter_value(filename, index, 'a', 'Value', new_height)
        model.set_parameter_value(filename, index, 'x0', 'Value', x0)
        set_roi_size(roi.size().x(), new_height, roi)
        params = {'a': new_height, 'x0': x0, 'dx': dx}
        if 'add_params' not in self.peak_shapes_params[line_type]:
            self.redraw_curve(params, curve, line_type)
            return
        add_params = self.peak_shapes_params[line_type]['add_params']
        for param_name in add_params:
            params[param_name] = model.get_parameter_value(filename, index, param_name, 'Value')
        self.redraw_curve(params, curve, line_type)

    def redraw_curve(self, params: dict | None = None, curve: PlotCurveItem = None, line_type: str | None = None,
                     idx: int | None = None) -> None:
        if params is None and idx is not None:
            params = self.current_line_parameters(idx)
        elif params is None:
            return
        if curve is None and idx is not None:
            curve = self.deconvolution_data_items_by_idx(idx)[0]
        elif curve is None:
            return
        if line_type is None and idx is not None:
            line_type = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
        elif line_type is None:
            return
        x0 = params['x0']
        full_amp_line, x_axis, _ = self.curve_y_x_idx(line_type, params, idx=idx)
        if full_amp_line is None:
            return
        curve.setData(x=x_axis, y=full_amp_line)
        curve.setPos(-x0, 0)

    def delete_deconv_curve(self, idx: int) -> None:
        items_matches = self.deconvolution_data_items_by_idx(idx)
        if items_matches is None:
            return
        curve, roi = items_matches
        self.parent.deconvolution_plotItem.removeItem(roi)
        self.parent.deconvolution_plotItem.removeItem(curve)
        self.parent.deconvolution_plotItem.getViewBox().updateAutoRange()

    def current_filename_lines_parameters(self, indexes: list[int], filename, line_types: Series) -> dict | None:
        df_params = self.parent.ui.fit_params_table.model().get_df_by_multiindex(filename)
        if df_params.empty:
            return None
        params = {}
        for idx in indexes:
            params[idx] = packed_current_line_parameters(df_params.loc[idx], line_types.loc[idx],
                                                         self.peak_shapes_params)
        return params

    def show_hide_curve(self, idx: int, b: bool) -> None:
        """
        Find curves by name = idx, and setVisible for it by bool param
        """
        items_matches = self.deconvolution_data_items_by_idx(idx)
        if items_matches is None:
            return
        curve, roi = items_matches
        if curve is None or roi is None:
            return
        self.redraw_curve_by_index(idx)
        curve.setVisible(b)
        roi.setVisible(b)
        self.draw_sum_curve()
        self.draw_residual_curve()
        self.parent.deconvolution_plotItem.getViewBox().updateAutoRange()

    def redraw_curve_by_index(self, idx: int, update: bool = True) -> None:
        params = self.current_line_parameters(idx)
        items = self.deconvolution_data_items_by_idx(idx)
        if items is None:
            return
        curve, roi = items
        if curve is None or roi is None:
            return
        line_type = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
        set_roi_size_pos((params['a'], params['x0'], params['dx']), roi, update)
        self.redraw_curve(params, curve, line_type, idx)

    def draw_sum_curve(self) -> None:
        """
            Update sum curve

            Returns
            -------
            out : None
        """
        if not self.parent.preprocessing.baseline_corrected_dict or self.sum_curve is None:
            return
        x_axis, y_axis = self.sum_array()
        self.sum_curve.setData(x=x_axis, y=y_axis.T)

    def sum_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return x, y arrays of sum spectra of all visible fit curves

        Returns
        -------
        out : tuple[np.ndarray, np.ndarray]
            x_axis, y_axis of sum curve
        """
        mw = self.parent
        x_axis = next(iter(mw.preprocessing.baseline_corrected_dict.values()))[:, 0]
        if mw.ui.interval_checkBox.isChecked():
            x_axis, _, _ = cut_axis(x_axis, mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value())
        data_items = mw.deconvolution_plotItem.listDataItems()
        y_axis = np.zeros(x_axis.shape[0])

        for i in data_items:
            if not (isinstance(i, PlotCurveItem) and i.isVisible()):
                continue
            x, y = i.getData()
            idx = nearest_idx(x_axis, x[0])
            y_z = np.zeros(x_axis.shape[0])
            if x_axis.shape[0] < y.shape[0]:
                idx_right = x_axis.shape[0] - idx - 1
                y_z[idx: idx + idx_right] += y[:idx_right]
            else:
                y_z[idx: idx + y.shape[0]] += y
            y_axis += y_z
        return x_axis, y_axis

    def overlapping_coefficients_for_each_line(self) -> dict:
        """
        Returns list of overlapping coefficients (OVC) of lines sorted by cm-1 position.
        OVC = overlapping_area * 100 / current_line_area
        overlapping_area is meant overlapping between current line and all rest

        Returns
        -------
        out: dict
            key is line index
            value - OVC
        """
        ovcs = {}
        x_axis = next(iter(self.parent.preprocessing.baseline_corrected_dict.values()))[:, 0]
        # get IDs of lines sorted by their positions cm-1
        idxs = self.parent.ui.fit_params_table.model().lines_idx_by_x0_sorted()
        # get curves with x, y data
        data_items = self.parent.deconvolution_plotItem.listDataItems()
        for idx in idxs:
            y_z = np.zeros(x_axis.shape[0])
            area_of_current_curve = .0
            for i in data_items:
                if not (isinstance(i, PlotCurveItem) and i.isVisible()):
                    continue
                if i.name() == idx:
                    x_cur, y_cur = i.getData()
                    area_of_current_curve = np.trapz(y_cur, x_cur)
                else:
                    _, y = i.getData()
                    y_z += y
            area_overlapping = np.trapz(np.minimum(y_z, y_cur), x_axis)
            area_overlapping = area_overlapping * 100 / area_of_current_curve
            ovcs[idx] = area_overlapping

        return ovcs

    def draw_residual_curve(self) -> None:
        """
            Update residual curve after sum curve updated.
            Residual = data - sum

            Returns
            -------
            out : None
        """
        if not self.parent.preprocessing.baseline_corrected_dict or self.residual_curve is None:
            return
        x_axis, y_axis = self.residual_array()
        self.residual_curve.setData(x=x_axis, y=y_axis.T)

    def residual_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Return x, y arrays of residual spectra.
            Residual = data - sum

            Returns
            -------
            out : tuple[np.ndarray, np.ndarray]
                x_axis, y_axis of residual curve
        """
        x_data, y_data = self.data_curve.getData()
        x_sum, y_sum = self.sum_curve.getData()
        c = y_data.copy()
        c[:len(y_sum)] -= y_sum
        return x_data, c

    def update_sigma3_curves(self, filename: str | None = None) -> None:
        """
        Update self.sigma3_top and self.sigma3_bottom for current spectrum
        @param filename: optional
        @return: None
        """
        if filename is None:
            filename = "" if self.is_template or self.current_spectrum_deconvolution_name == '' \
                else self.current_spectrum_deconvolution_name
        if filename not in self.sigma3:
            self.sigma3_fill.setVisible(False)
            return
        self.sigma3_fill.setVisible(self.parent.ui.sigma3_checkBox.isChecked())
        self.sigma3_top.setData(x=self.sigma3[filename][0], y=self.sigma3[filename][1])
        self.sigma3_bottom.setData(x=self.sigma3[filename][0], y=self.sigma3[filename][2])

    def update_single_deconvolution_plot(self, current_spectrum_name: str, is_template: bool = False,
                                         is_averaged_or_group: bool = False) -> None:
        """
        Change current data spectrum in deconv_plot_widget
        set self.isTemplate

        isTemplate - True if current spectrum is averaged or group's averaged
        isAveraged_or_Group
        """
        mw = self.parent
        if not mw.preprocessing.baseline_corrected_dict:
            return
        self.is_template = is_template
        data_items = mw.deconvolution_plotItem.listDataItems()
        arr = None
        if is_template and is_averaged_or_group:
            self.current_spectrum_deconvolution_name = ''
            if current_spectrum_name == 'Average':
                arrays_list = [mw.preprocessing.baseline_corrected_dict[x] for x in
                               mw.preprocessing.baseline_corrected_dict]
                arr = get_average_spectrum(arrays_list, mw.ui.average_method_cb.currentText())
                self.averaged_spectrum = arr
                mw.ui.max_noise_level_dsb.setValue(np.max(arr[:, 1]) / 99.)
            elif mw.preprocessing.averaged_dict:
                arr = mw.preprocessing.averaged_dict[int(current_spectrum_name.split('.')[0])]
        else:
            arr = mw.preprocessing.baseline_corrected_dict[current_spectrum_name]
            self.current_spectrum_deconvolution_name = current_spectrum_name

        if arr is None:
            return
        if mw.ui.interval_checkBox.isChecked():
            arr = cut_full_spectrum(arr, mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value())
        title_text = current_spectrum_name
        if is_template:
            title_text = 'Template. ' + current_spectrum_name

        new_title = "<span style=\"font-family: AbletonSans; color:" + mw.theme_colors[
            'plotText'] + ";font-size:14pt\">" + title_text + "</span>"
        mw.ui.deconv_plot_widget.setTitle(new_title)

        for i in data_items:
            if isinstance(i, PlotDataItem):
                i.setVisible(False)

        if self.data_curve:
            mw.deconvolution_plotItem.removeItem(self.data_curve)
        if self.sum_curve:
            mw.deconvolution_plotItem.removeItem(self.sum_curve)
        if self.residual_curve:
            mw.deconvolution_plotItem.removeItem(self.residual_curve)

        self.data_curve = mw.get_curve_plot_data_item(arr, color=self.data_style['color'], name='data')
        x_axis, y_axis = self.sum_array()
        self.sum_curve = mw.get_curve_plot_data_item(np.vstack((x_axis, y_axis)).T, color=self.sum_style['color'],
                                                     name='sum')
        x_res, y_res = self.residual_array()
        self.residual_curve = mw.get_curve_plot_data_item(np.vstack((x_res, y_res)).T,
                                                          color=self.residual_style['color'], name='sum')

        self.data_curve.setVisible(mw.ui.data_checkBox.isChecked())
        self.sum_curve.setVisible(mw.ui.sum_checkBox.isChecked())
        self.residual_curve.setVisible(mw.ui.residual_checkBox.isChecked())
        mw.deconvolution_plotItem.addItem(self.data_curve, kargs=['ignoreBounds', 'skipAverage'])
        mw.deconvolution_plotItem.addItem(self.sum_curve, kargs=['ignoreBounds', 'skipAverage'])
        mw.deconvolution_plotItem.addItem(self.residual_curve, kargs=['ignoreBounds', 'skipAverage'])
        mw.deconvolution_plotItem.getViewBox().updateAutoRange()

    def redraw_curves_for_filename(self) -> None:
        """
        Redraw all curves by parameters of current selected spectrum.
        """
        filename = "" if self.is_template or self.current_spectrum_deconvolution_name == '' \
            else self.current_spectrum_deconvolution_name
        line_indexes = self.parent.ui.deconv_lines_table.model().get_visible_line_types().index
        filename_lines_indexes = self.parent.ui.fit_params_table.model().get_lines_indexes_by_filename(filename)
        if filename_lines_indexes is None:
            return
        line_types = self.parent.ui.deconv_lines_table.model().column_data(1)
        if line_types.empty:
            return None
        if len(line_indexes) != len(filename_lines_indexes):
            MessageBox("Template error", "Состав линий этого спектра отличается от шаблона. Некоторые линии не"
                                         " будут отрисованы правильно.", self.parent, {'Ok'}).exec()
            line_indexes = filename_lines_indexes
        params = self.current_filename_lines_parameters(list(line_indexes), filename, line_types)
        if params is None or not params:
            return
        items = self.deconvolution_data_items_by_indexes(list(line_indexes))
        if items is None or not items:
            return
        for i in line_indexes:
            if i not in params or i not in items:
                continue
            set_roi_size_pos((params[i]['a'], params[i]['x0'], params[i]['dx']), items[i][1], False)
            self.redraw_curve(params[i], items[i][0], line_types.loc[i], i)

    def deconvolution_data_items_by_indexes(self, indexes: list[int]) -> dict | None:
        result = {}
        data_items = self.parent.deconvolution_plotItem.listDataItems()
        if len(data_items) == 0:
            return None
        for i in data_items:
            if i.name() in indexes:
                curve = i
                roi = i.parentItem()
                result[i.name()] = curve, roi
        return result

    def remove_all_lines_from_plot(self) -> None:
        data_items = self.parent.deconvolution_plotItem.listDataItems()
        if len(data_items) == 0:
            return
        items_matches = (x for x in data_items if isinstance(x.name(), int))
        for i in items_matches:
            self.parent.deconvolution_plotItem.removeItem(i.parentItem())
            self.parent.deconvolution_plotItem.removeItem(i)
        self.parent.deconvolution_plotItem.addItem(self.parent.linearRegionDeconv)
        self.parent.deconvolution_plotItem.addItem(self.sigma3_fill)
        self.parent.deconvolution_plotItem.getViewBox().updateAutoRange()

    async def draw_all_curves(self) -> None:
        """
        Отрисовка всех линий
        @return: None
        """
        line_indexes = self.parent.ui.deconv_lines_table.model().dataframe().index
        model = self.parent.ui.deconv_lines_table.model()
        current_all_lines_parameters = self.all_lines_parameters(line_indexes)
        result = []
        for i in line_indexes:
            res = self.curve_y_x_idx(model.cell_data_by_idx_col_name(i, 'Type'),
                                     current_all_lines_parameters[i], None, i)
            result.append(res)

        for full_amp_line, x_axis, idx in result:
            self.create_roi_curve_add_to_plot(full_amp_line, x_axis, idx, model.cell_data_by_idx_col_name(idx, 'Style'),
                                              current_all_lines_parameters[idx])
        self.draw_sum_curve()
        self.draw_residual_curve()

    def all_lines_parameters(self, line_indexes: list[int]) -> dict | None:
        filename = "" if self.is_template or self.current_spectrum_deconvolution_name == '' \
            else self.current_spectrum_deconvolution_name
        df_params = self.parent.ui.fit_params_table.model().get_df_by_multiindex(filename)
        parameters = {}
        for idx in line_indexes:
            line_type = self.parent.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
            a = df_params.loc[(idx, 'a')].Value
            x0 = df_params.loc[(idx, 'x0')].Value
            dx = df_params.loc[(idx, 'dx')].Value
            result = {'a': a, 'x0': x0, 'dx': dx}
            if 'add_params' not in self.peak_shapes_params[line_type]:
                parameters[idx] = result
                continue
            add_params = self.peak_shapes_params[line_type]['add_params']
            for param_name in add_params:
                result[param_name] = df_params.loc[(idx, param_name)].Value
            parameters[idx] = result
        return parameters

    def hide_all_roi(self):
        for i in self.parent.deconvolution_plotItem.listDataItems():
            if isinstance(i, PlotCurveItem):
                i.parentItem().hide()

    def show_all_roi(self):
        for i in self.parent.deconvolution_plotItem.listDataItems():
            if isinstance(i, PlotCurveItem):
                i.parentItem().show()

    def cut_data_interval(self) -> None:
        mw = self.parent
        n_array = self.array_of_current_filename_in_deconvolution()
        if n_array is None:
            return
        n_array = cut_full_spectrum(n_array, mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value())
        self.data_curve.setData(x=n_array[:, 0], y=n_array[:, 1])

    # endregion

    # region copy template
    def copy_line_parameters_from_template(self, idx: int | None = None, filename: str | None = None,
                                           redraw: bool = True) -> None:
        filename = self.current_spectrum_deconvolution_name if filename is None else filename
        model = self.parent.ui.fit_params_table.model()
        # find current index of selected line
        if idx is None:
            row = self.parent.ui.deconv_lines_table.selectionModel().currentIndex().row()
            if row == -1:
                MessageBox("Line isn't selected", 'Select line', self.parent, {'Ok'}).exec()
                return
            idx = self.parent.ui.deconv_lines_table.model().row_data(row).name

            # delete line params by index
            model.delete_rows_multiindex((filename, idx))
        # add line params by from template
        mi = '', idx
        df = model.get_df_by_multiindex(mi)
        for i in range(len(df)):
            row_data = df.iloc[i]
            model.append_row(idx, row_data.Parameter, row_data.Value, row_data['Min value'], row_data['Max value'],
                             filename)
        if redraw:
            self.redraw_curve_by_index(idx)

    @asyncSlot()
    async def copy_spectrum_lines_parameters_from_template(self) -> None:
        selected_rows = self.parent.ui.dec_table.selectionModel().selectedRows()
        if len(selected_rows) == 0:
            return
        selected_filename = self.parent.ui.dec_table.model().cell_data_by_index(selected_rows[0])
        line_indexes = self.parent.ui.deconv_lines_table.model().dataframe().index
        model = self.parent.ui.fit_params_table.model()
        if len(line_indexes) == 0:
            return
        query_text = 'filename == ' + '"' + str(selected_filename) + '"'
        if not model.is_query_result_empty(query_text):
            model.delete_rows_by_multiindex(selected_filename)
        executor = ThreadPoolExecutor()
        with Manager() as manager:
            self.parent.break_event = manager.Event()
            with executor:
                self.parent.current_futures = [
                    self.parent.loop.run_in_executor(executor, self.copy_line_parameters_from_template, i,
                                                     selected_filename)
                    for i in line_indexes]
                await gather(*self.parent.current_futures)

    # endregion

    # region GUI
    def interval_start_dsb_change_event(self, new_value: float) -> None:
        """
        executing when value of self.ui.interval_start_dsb was changed by user or by code
        self.old_start_interval_value contains previous value

        Parameters
        ----------
        new_value : float
            New value of self.ui.interval_start_dsb
        """
        mw = self.parent
        mw.set_modified()
        corrected_value = None
        # correct value - take nearest from x_axis
        if mw.preprocessing.averaged_dict != {} and mw.preprocessing.averaged_dict:
            x_axis = next(iter(mw.preprocessing.averaged_dict.values()))[:, 0]
            corrected_value = find_nearest(x_axis, new_value)
        if corrected_value is not None and round(corrected_value, 5) != new_value:
            mw.ui.interval_start_dsb.setValue(corrected_value)
            return
        if new_value >= mw.ui.interval_end_dsb.value():
            mw.ui.interval_start_dsb.setValue(mw.ui.interval_start_dsb.minimum())
            return
        if mw.CommandStartIntervalChanged_allowed:
            command = CommandStartIntervalChanged(mw, new_value, mw.old_start_interval_value,
                                                  'Change start-interval value')
            mw.undoStack.push(command)
        mw.linearRegionDeconv.setRegion((mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value()))
        # if interval checked - change cut interval
        if mw.ui.interval_checkBox.isChecked():
            mw.cut_data_sum_residual_interval()

    def interval_end_dsb_change_event(self, new_value: float) -> None:
        """
        executing when value of self.ui.interval_end_dsb was changed by user or by code
        self.old_end_interval_value contains previous value

        Parameters
        ----------
        new_value : float
            New value of self.ui.interval_end_dsb
        """
        mw = self.parent
        mw.set_modified()
        corrected_value = None
        # correct value - take nearest from x_axis
        if mw.preprocessing.averaged_dict != {} and mw.preprocessing.averaged_dict:
            x_axis = next(iter(mw.preprocessing.averaged_dict.values()))[:, 0]
            corrected_value = find_nearest(x_axis, new_value)
        if corrected_value is not None and round(corrected_value, 5) != new_value:
            mw.ui.interval_end_dsb.setValue(corrected_value)
            return
        if new_value <= mw.ui.interval_start_dsb.value():
            mw.ui.interval_end_dsb.setValue(mw.ui.interval_end_dsb.minimum())
            return
        if mw.CommandEndIntervalChanged_allowed:
            command = CommandEndIntervalChanged(mw, new_value, mw.old_end_interval_value,
                                                'Change end-interval value')
            mw.undoStack.push(command)
        mw.linearRegionDeconv.setRegion((mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value()))
        # if interval checked - change cut interval
        if mw.ui.interval_checkBox.isChecked():
            mw.cut_data_sum_residual_interval()
    # endregion
