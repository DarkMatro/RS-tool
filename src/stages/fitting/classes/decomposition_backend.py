from copy import deepcopy
from typing import KeysView

import numpy as np
import pandas as pd
from asyncqtpy import asyncSlot
from lmfit import Parameters
from lmfit.model import ModelResult, Model
from pandas import MultiIndex

from qfluentwidgets import MessageBox
from src import get_parent, get_config, ObservableDict
from src.data.default_values import peak_shapes_params
from src.data.work_with_arrays import nearest_idx
from src.stages import guess_peaks, fit_model, update_fit_parameters, legend_from_float, \
    clustering_lines_intervals, intervals_by_borders, create_intervals_data, params_func_legend, \
    fitting_model, fit_model_batch, eval_uncert, models_params_splitted, \
    models_params_splitted_batch, cut_full_spectrum, split_by_borders
from src.stages.fitting.classes.undo import CommandAfterGuess, CommandAfterBatchFitting, \
    CommandAfterFitting


class DecompositionBackend:
    """
    parent - DecompositionStage
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent

    # region Fit
    @asyncSlot()
    async def fit(self):
        """
        Check conditions when Fit button pressed, if all ok - go do_fit
        For fitting must be more than 0 lines to fit and current spectrum in plot must also be
        Должна быть хотя бы 1 линия и выделен текущий спектр
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.ui.deconv_lines_table.model().rowCount() == 0:
            MessageBox("Fitting failed.", "Add some new lines before fitting", mw,
                       {"Ok"}).exec()
            return
        if self.parent.graph_drawing.array_of_current_filename_in_deconvolution() is None:
            MessageBox("Fitting failed.", "There is No any data to fit", mw, {"Ok"}).exec()
            return
        if mw.ui.intervals_gb.isChecked() \
                and len(mw.ui.fit_borders_TableView.model().column_data(0)) == 0:
            MessageBox("Fitting failed.", "Turn off SPLIT SPECTRUM or fill table", mw,
                       {"Ok"}).exec()
            return

        await self._do_fit()

    @asyncSlot()
    async def _do_fit(self) -> None:
        """
        Fitting line's parameters to current spectrum data
        1. Get x, y axes of current spectrum
        2. Prepare data before creating model and parameters
        3. Create params
        4. Create model
        5. Fit model to y_data
        """
        mw = get_parent(self.parent, "MainWindow")
        cfg = get_config("texty")["fit"]
        fitting_methods = get_config("fitting")['fitting_methods']
        arr = self.parent.graph_drawing.array_of_current_filename_in_deconvolution()
        filename = self.parent.data.current_spectrum_name
        spl_arr = self.split_array_for_fitting({filename: [arr]})[filename]
        mw.progress.open_progress(cfg, len(spl_arr))
        static_params = self.prepare_data_fitting()
        params = self.fitting_params(static_params, {'max_a': np.max(arr[:, 1]),
                                                     'min_x0': arr[:, 0][0],
                                                     'max_x0': arr[:, 0][-1]})
        x_y_models_params, _ = models_params_splitted(spl_arr, params, static_params)
        method = fitting_methods[mw.ui.fit_opt_method_comboBox.currentText()]
        kwargs = {'n_files': len(spl_arr), 'method': method}
        result = await mw.progress.run_in_executor(
            "fit", fit_model, x_y_models_params, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandAfterFitting(result, context, text=f"Fitted spectrum {filename}",
                                      **{'stage': self.parent, 'static_params': static_params,
                                         'filename': filename})
        context.undo_stack.push(command)

    def prepare_data_fitting(self) -> list[tuple[int, str, int, str, callable]]:
        """
        Get prepared data to create model and parameters

        Returns
        -------
        idx_type_paramcount_legend : list[tuple[int, str, int, str]]
            using to create Parameters of model. idx - curve index, type - curve type,
             param_count - number of
            parameters of line_type, legend - curve legend by index
        """
        # get dataframe with active lines - idx | Legend, Type, Style
        mw = get_parent(self.parent, "MainWindow")
        line_types = mw.ui.deconv_lines_table.model().get_visible_line_types()
        ans = []
        psp = peak_shapes_params()
        for i in line_types.index:
            ser = line_types.loc[i]
            if 'add_params' in psp[ser.Type]:
                param_count = 3 + len(psp[ser.Type]['add_params'])
            else:
                param_count = 3
            legend = f"Curve_{i}_"
            ans.append((i, ser.Type, param_count, legend,
                        psp[ser.Type]['func']))
        return ans

    def fitting_params(self, list_idx_type: list[tuple[int, str, int, str, callable]],
                       bounds: dict) -> Parameters:
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
        bounds:
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
        bounds['min_a'] = 0.
        if not self.parent.data.current_spectrum_name:
            bounds['min_dx'] = np.max(
                get_parent(self.parent, "MainWindow").ui.input_table.model().column_data(6)) / 2
        else:
            bounds['min_dx'] = get_parent(self.parent,
                                          "MainWindow").ui.input_table.model().row_data_by_index(
                self.parent.data.current_spectrum_name)[
                                   'FWHM, cm\N{superscript minus}\N{superscript one}'] / 2.
        params = Parameters()
        i = 0
        pspl = get_config('fitting')['peak_shape_params_limits']
        for idx, line_type, param_count, legend, _ in list_idx_type:
            params_from_table = self.parent.current_line_parameters(idx, '')
            param_names = ['a', 'x0', 'dx']
            for j in range(param_count - 3):
                param_names.append(peak_shapes_params()[line_type]['add_params'][j])
            # value must be between bounds
            for pn in param_names:
                v = [params_from_table[pn], params_from_table['min_' + pn],
                     params_from_table['max_' + pn]]
                # bounds correction
                bounds['min_v'], bounds['max_v'] = None, None
                if pn == 'a':
                    bounds['min_v'] = bounds['min_a']
                    bounds['max_v'] = bounds['max_a']
                elif pn == 'x0':
                    bounds['min_v'] = bounds['min_x0']
                    bounds['max_v'] = bounds['max_x0']
                elif pn in ('dx', 'dx_left'):
                    bounds['min_v'] = bounds['min_dx']
                    bounds['max_v'] = (bounds['max_x0'] - bounds['min_x0']) / 2
                elif pn in pspl:
                    bounds['min_v'] = pspl[pn][0]
                    bounds['max_v'] = pspl[pn][1]
                if bounds['min_v'] is not None:
                    v[1] = bounds['min_v'] if v[1] < bounds['min_v'] else v[1]
                if bounds['max_v'] is not None:
                    v[2] = bounds['max_v'] if v[2] > bounds['max_v'] else v[2]
                v[0] = v[1] + 1e-3 if v[0] <= v[1] else v[0]
                v[0] = v[2] - 1e-3 if v[0] >= v[2] else v[0]
                v[2] = v[2] + 1e-3 if v[1] == v[2] else v[2]
                v[2] = bounds['max_a'] if pn == 'a' else v[2]
                params.add(legend + pn, v[0], min=v[1], max=v[2])
            i += param_count
        return params

    # endregion

    # region Batch
    @asyncSlot()
    async def batch_fit(self) -> None:
        """
        Check conditions when Fit button pressed, if all ok - go do_batch_fit
        For fitting must be more than 0 lines to fit
        """
        mw = get_parent(self.parent, "MainWindow")
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if mw.ui.deconv_lines_table.model().rowCount() == 0:
            msg = MessageBox("Fitting failed.", "Add some new lines before fitting", mw, {"Ok"})
            msg.exec()
            return
        if not data or len(data) == 0:
            MessageBox("Fitting failed.", "There is No any data to fit", mw, {"Ok"}).exec()
            return
        try:
            await self._do_batch_fit()
        except Exception as err:
            mw.show_error(err)

    @asyncSlot()
    async def _do_batch_fit(self) -> None:
        """
        Fitting line's parameters to all spectrum files
        1. Get x, y axes of current spectrum
        2. Prepare data before creating model and parameters
        3. Create params
        4. Create model
        5. Fit model to y_data
        """
        mw = get_parent(self.parent, "MainWindow")
        cfg = get_config("texty")["batch_fit"]
        mw.progress.open_progress(cfg, 0)
        assert mw.drag_widget.get_latest_active_stage() is not None, \
            'Cant find latest active stage.'
        static_params = self.prepare_data_fitting()
        method = get_config("fitting")['fitting_methods'][
            mw.ui.fit_opt_method_comboBox.currentText()]
        key_x_y_models_params = self._prepare_batch_fit_data(static_params)
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        mw.progress.open_progress(cfg, len(key_x_y_models_params))
        kwargs = {'n_files': len(key_x_y_models_params), 'method': method}
        result = await mw.progress.run_in_executor(
            "batch_fit", fit_model_batch, key_x_y_models_params, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        cfg = get_config("texty")["uncertain"]
        mw.progress.open_progress(cfg, len(result))
        dely = await mw.progress.run_in_executor(
            "uncertain", eval_uncert, result, **{'n_files': len(result)}
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        cfg = get_config("texty")["batch_fit_undo"]
        mw.progress.open_progress(cfg, 0)
        context = get_parent(self.parent, "Context")
        command = CommandAfterBatchFitting(result, context, text="Batch fitting",
                                           **{'stage': self.parent, 'static_params': static_params,
                                              'dely': dely})
        context.undo_stack.push(command)
        mw.progress.close_progress(cfg)

    def prepare_baseline_corrected_spectra(self, data: ObservableDict,
                                           filenames: KeysView) -> dict:
        mw = get_parent(self.parent, "MainWindow")
        if mw.predict_logic.is_production_project \
                and mw.ui.deconvoluted_dataset_table_view.model().rowCount() != 0:
            filenames_in_dataset = list(
                mw.ui.deconvoluted_dataset_table_view.model().column_data(1).values)
            baseline_corrected_spectra = {key: arr for key, arr in data.items()
                                          if key not in filenames_in_dataset}
        else:
            baseline_corrected_spectra = {}
            for key in filenames:
                try:
                    baseline_corrected_spectra[key] = [data[key]]
                except KeyError:
                    pass
        return baseline_corrected_spectra

    def _fitting_params_batch(self, static_params: list[tuple[int, str, int, str, callable]],
                              arrays: dict[str, list[np.ndarray]]) -> dict:
        try:
            x_axis = next(iter(arrays.values()))[0][:, 0]
        except StopIteration:
            x_axis = self.parent.get_x_axis()
        params_mutual = self.fitting_params(static_params, {'max_a': 1000.,
                                                            'min_x0': x_axis[0],
                                                            'max_x0': x_axis[-1]})
        list_params_full = {}
        for key, item in arrays.items():
            params = deepcopy(params_mutual)
            for p in params:
                psplt = p.split('_', 2)
                if psplt[-1] != 'a':
                    continue
                dx_left_pname = psplt[0] + '_' + psplt[1] + '_dx_left'
                arg_x0_left = nearest_idx(x_axis,
                                          params[psplt[0] + '_' + psplt[1] + '_x0'].min -
                                          (params[dx_left_pname].max if dx_left_pname in params
                                           else params[f"{psplt[0]}_{psplt[1]}_dx"].max))
                arg_x0_right = nearest_idx(x_axis,
                                           params[psplt[0] + '_' + psplt[1] + '_x0'].max
                                           + params[f"{psplt[0]}_{psplt[1]}_dx"].max)
                y_max_in_range = np.amax(item[0][:, 1][arg_x0_left:arg_x0_right])
                params[p].max = y_max_in_range
                params[p].value = params[p].init_value = y_max_in_range / 2.
            list_params_full[key] = params
        return list_params_full

    def _prepare_batch_fit_data(self, static_params: list[tuple[int, str, int, str, callable]]) \
            -> list[tuple[str, np.ndarray, np.ndarray, Model, dict]]:
        mw = get_parent(self.parent, "MainWindow")
        filenames = (mw.drag_widget.get_latest_active_stage().data.keys()
                     & mw.ui.input_table.model().filenames())
        baseline_corrected_spectra = self.prepare_baseline_corrected_spectra(
            mw.drag_widget.get_latest_active_stage().data, filenames)
        splitted_arrays = self.split_array_for_fitting(baseline_corrected_spectra)
        list_params_full = self._fitting_params_batch(static_params, baseline_corrected_spectra)
        x_y_models_params = models_params_splitted_batch(splitted_arrays, list_params_full,
                                                         static_params)
        key_x_y_models_params = []
        for key, item in x_y_models_params.items():
            for x_axis, y_axis, model, interval_params in item:
                key_x_y_models_params.append((key, x_axis, y_axis, model, interval_params))
        return key_x_y_models_params

    def add_line_params_from_template_batch(self, keys: set[str]) -> None:
        mw = get_parent(self.parent, "MainWindow")
        key = next(iter(keys))
        self.parent.add_line_params_from_template(key)
        df_a = mw.ui.fit_params_table.model().dataframe()
        tuples = [i for i in df_a.index if i[0] == '']
        df_a = df_a.loc['']
        mi = MultiIndex.from_tuples(tuples, names=('filename', 'line_index', 'param_name'))
        df_a.index = mi
        df_c = df_a.copy(deep=True)
        for key in keys:
            tuples_b = []
            for _, line_idx, param_name in tuples:
                tuples_b.append((key, line_idx, param_name))
            df_b = df_a.copy(deep=True)
            mi = MultiIndex.from_tuples(tuples_b,
                                        names=('filename', 'line_index', 'param_name'))
            df_b.index = mi
            df_c = pd.concat([df_c, df_b])
        mw.ui.fit_params_table.model().set_dataframe(df_c)
        mw.ui.fit_params_table.model().sort_index()

    # endregion

    # region Guess
    @asyncSlot()
    async def guess(self, line_type: str) -> None:
        """
        Auto guess lines, finds number of lines and positions x0
        """
        mw = get_parent(self.parent, "MainWindow")
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if not data:
            MessageBox("Guess failed.", "Do baseline correction before guessing peaks", mw, {"Ok"}
                       ).exec()
            return
        if mw.ui.interval_checkBox.isChecked() and mw.ui.intervals_gb.isChecked():
            MessageBox("Guess failed.",
                       "'Split spectrum' and 'Interval' both active. Leave only one of them "
                       "turned on",
                       mw, {"Ok"}).exec()
            return
        if mw.ui.intervals_gb.isChecked() and mw.ui.fit_borders_TableView.model().rowCount() == 0:
            MessageBox("Guess failed.", "If 'Split spectrum' table is active, you must fill the"
                                        " table", mw, {"Ok"}).exec()
            return
        av_data = get_parent(self.parent, "Context").preprocessing.stages.av_data
        if mw.ui.guess_method_cb.currentText() == "Average groups" and (not av_data.data
                                                                        or len(av_data.data) < 2):
            MessageBox("Guess failed.", "If the 'Average groups' method is selected, then there"
                                        " must be more than 1 group", mw, {"Ok"}).exec()
            return
        if not mw.ui.interval_checkBox.isChecked() and not mw.ui.intervals_gb.isChecked():
            msg = MessageBox("Warning!", "Automatic detection of lines in the full range can take a"
                                         " very long time" + "\n" + "Are you sure to continue?", mw,
                             {"Yes", "No", "Cancel"})
            msg.setInformativeText(
                "Dividing the spectrum into ranges can reduce analysis time by 2-3 orders of"
                " magnitude. And without division, the calculation can take hours."
            )
            result = msg.exec()
            if result == 1:
                msg = MessageBox("Warning!",
                                 "Last chance to change your mind" + "\n" +
                                 "Are you sure to continue?",
                                 mw, {"Yes", "No", "Cancel"})
                if not msg.exec() == 1:
                    return
            else:
                return
        await self.do_auto_guess(line_type)

    @asyncSlot()
    async def do_auto_guess(self, line_type: str) -> None:
        """
        Автоматический подбор линий к модели. Результат отображается только для шаблона.
        Peaks are added at position of global extremum of data-baseline with previous peaks
        subtracted.
        Таблица параметров для всех спектров очищается. Результаты прошлых fit очищаются.
        Если на момент начала анализа уже есть линии в таблице линий, то поиск линий начнется не с
        нуля,
         а с этого состава.
        Спектр делится на интервалы.
        Guess method:
            Если выбран 'Average', то анализируется только усредненный спектр по всем группам.
            Если выбран 'Average groups', то анализируются усредненные по каждой отдельной группе
            Если выбран 'All', то анализируются усредненные по всем спектрам из левой таблицы.
        Для 'Average groups' и 'All' собираются линии x0. По каждому интервалу анализируется
        количество линий.
        Итоговое количество линий для каждого интервала определяется методом кластеризации k-means.
        На вход в алгоритм k-means подается количество линий

        После k-means полученный состав линий опять прогоняется через Fit и мы получаем итоговую
        модель.

        В процессе анализа на каждый параметр линий накладывается ряд ограничений.
         a - амплитуда от 0 до максимального значения в интервале. Интервал зависит от x0 и dx
         x0 - положение максимума линии. Изначально определяется по положению максимума residual
         +- 1 см-1. Или после
            k-means границы задаются мин-макс значением в кластере
         dx, dx_left - полуширина линии левая/правая, максимальное значение задается в Max peak
         HWHM:, минимальное
            определяется из наименьшего FWHM CM-1 / 2 в таблице
        Остальные параметры имеют границы указанные в peak_shape_params_limits

        @param line_type: str
        @return: None
        """
        mw = get_parent(self.parent, "MainWindow")
        cfg = get_config("texty")["guess"]
        parameters_to_guess = self._parameters_to_guess(line_type)
        sliced_arrays = self._arrays_for_peak_guess()
        n_files = len(sliced_arrays)
        n_files = 0 if n_files == 1 else n_files
        mw.progress.open_progress(cfg, n_files)
        kwargs = {'n_files': n_files, 'input_parameters': parameters_to_guess,
                  'break_event_by_user': mw.progress.break_event}
        result = await mw.progress.run_in_executor("guess", guess_peaks, sliced_arrays, **kwargs)
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        if mw.ui.guess_method_cb.currentText() != 'Average':
            cfg = get_config("texty")["guess_fit"]
            mw.progress.open_progress(cfg, 0)
            x_y_models_params = await self._analyze_guess_results(result,
                                                                  parameters_to_guess[
                                                                      'param_names'], line_type)
            cancel = mw.progress.close_progress(cfg)
            if cancel:
                return
            intervals = len(x_y_models_params)
            intervals = intervals if intervals > 1 else 0
            mw.progress.open_progress(cfg, intervals)
            kwargs = {'n_files': n_files, 'method': parameters_to_guess['method']}
            result = await mw.progress.run_in_executor("fit", fit_model, x_y_models_params,
                                                       **kwargs)
            cancel = mw.progress.close_progress(cfg)
            if cancel:
                return
            if not result:
                mw.ui.statusBar.showMessage(cfg["no_result_msg"])

        if mw.progress.cancelled_by_user():
            return

        cfg = get_config("texty")["guess_undo"]
        mw.progress.open_progress(cfg, 0)
        context = get_parent(self.parent, "Context")
        command = CommandAfterGuess(result, context, text="Guess model",
                                    **{'stage': self.parent, 'line_type': line_type,
                                       'n_params': len(parameters_to_guess['param_names'])})
        context.undo_stack.push(command)
        mw.progress.close_progress(cfg)

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
        mw = get_parent(self.parent, "MainWindow")
        func = peak_shapes_params()[line_type]['func']
        param_names = self._param_names(line_type)
        init_model_params = self._init_model_params(line_type)
        min_fwhm = mw.ui.input_table.model().min_fwhm()
        mean_snr = np.mean(mw.ui.input_table.model().get_column('SNR').values)
        method = get_config("fitting")['fitting_methods'][
            mw.ui.fit_opt_method_comboBox.currentText()]
        params_limits = get_config("fitting")['peak_shape_params_limits']
        params_limits['l_ratio'] = (0., mw.ui.l_ratio_doubleSpinBox.value())
        # The following parameters are empty if there are no lines. If at the beginning of the
        # analysis there are lines already created by the user, then the parameters will be filled.
        visible_lines = mw.ui.deconv_lines_table.model().get_visible_line_types()
        func_legend, used_legends, used_legends_dead_zone, params = [], [], [], Parameters()
        if len(visible_lines) > 0 and not mw.ui.interval_checkBox.isChecked():
            func_legend, params, used_legends, used_legends_dead_zone = self._initial_template(
                visible_lines, func, (param_names, min_fwhm, params_limits), init_model_params)

        return {'func': func, 'param_names': param_names, 'init_model_params': init_model_params,
                'min_fwhm': min_fwhm,
                'method': method, 'params_limits': params_limits,
                'noise_level': max(np.max(self.parent.data.averaged_spectrum[:, 1]) / mean_snr,
                                   mw.ui.max_noise_level_dsb.value()),
                'max_dx': mw.ui.max_dx_dsb.value(),
                'func_legend': func_legend, 'params': params, 'used_legends': used_legends,
                'mean_snr': mean_snr,
                'used_legends_dead_zone': used_legends_dead_zone}

    def _initial_template(self, visible_lines: pd.DataFrame, func,
                          static_parameters: tuple[list[str], float, dict],
                          init_model_params: list[float]) -> tuple[
        list[tuple], Parameters, list[str], dict]:
        """
        Prepare parameters of existing template lines. New lines will be guessed adding to this
        template.

        Parameters
        ---------
        visible_lines : DataFrame
            Raman lines table with columns: Legend, Type, Style.
        func : callable
            Function for peak shape calculation. Look peak_shapes_params() in default_values.py.
        static_parameters : tuple[list[str], float, dict]
            param_names : list[str]; List of parameter names. Example: ['a', 'x0', 'dx'].
            min_fwhm, : float; the minimum value FWHM, determined from the input table
            (the minimum of all).
            peak_shape_params_limits: dict[str, tuple[float, float]]; see peak_shape_params_limits()
                in default_values.py.
            gamma_factor: float; from 0. to 1. limit for max gamma value set by:
                max_v = min(dx_left, dx_right) * gamma_factor
        init_model_params: list[float]; Initial values of parameters for a given spectrum and line
        type.

        Returns
        -------
        func_legend: list[tuple]; - (callable func, legend),
            func - callable; Function for peak shape calculation. Look peak_shapes_params() in
             default_values.py.
            legend - prefix for the line in the model. All lines in a heap.
            As a result, we select only those that belong to the current interval;

        params: Parameters();
            parameters of existing lines.

        used_legends: list[str];
            already used legends (used wave-numbers)
            ['k977dot15_', 'k959dot68_', 'k917dot49_']. We control it because model cant have
            lines with duplicate
             legends (same x0 position lines)

        used_legends_dead_zone: dict;
            keys - used_legends, values - tuple (idx - left idx, right idx - idx).
            dead zone size to set y_residual to 0.
            {'k977dot15_': (1, 1), 'k959dot68_': (2, 1), 'k917dot49_': (3, 4)}
        """
        func_legend, used_legends, params, used_legends_dead_zone = [], [], Parameters(), {}
        for i in visible_lines.index:
            vals = self._initial_params_for_template(i, 'dx_left' in static_parameters[0])
            func_legend.append((func, vals['legend']))
            init_params = init_model_params.copy()
            init_params[0], init_params[1] = vals['a_series'].Value, vals['x0']
            used_legends.append(vals['legend'])
            used_legends_dead_zone[vals['legend']] = (vals['x0_arg_dx_l'], vals['x0_arg_dx_r'])
            dynamic_parameters = (vals['legend'], init_params, vals['a_series']['Max value'],
                                  vals['dx_left'], vals['dx_right'])
            params = update_fit_parameters(params, static_parameters, dynamic_parameters)
        return func_legend, params, used_legends, used_legends_dead_zone

    def _initial_params_for_template(self, i: int, is_dx_left_in_static_params: bool) -> dict:
        mw = get_parent(self.parent, "MainWindow")
        model = mw.ui.fit_params_table.model()
        x_axis = self.parent.get_x_axis()
        x0 = model.get_df_by_multiindex(('', i, 'x0')).Value
        dx = model.get_df_by_multiindex(('', i, 'dx')).Value
        max_dx = mw.ui.max_dx_dsb.value()
        dx_right = min(float(dx), max_dx)
        dx_left = min(mw.ui.fit_params_table.model().get_df_by_multiindex(
            ('', i, 'dx_left')).Value, max_dx) if is_dx_left_in_static_params else dx_right
        x0_arg = nearest_idx(x_axis, x0)
        x0_arg_dx_l = nearest_idx(x_axis, x0 - dx_left / 2.)
        x0_arg_dx_r = nearest_idx(x_axis, x0 + dx_right / 2.)
        return {'x0': x0, 'dx': dx, 'dx_left': dx_left, 'dx_right': dx_right,
                'x0_arg_dx_r': x0_arg_dx_r - x0_arg, 'x0_arg_dx_l': x0_arg - x0_arg_dx_l,
                'a_series': model.get_df_by_multiindex(('', i, 'a')),
                'legend': legend_from_float(x0)}

    def _param_names(self, line_type: str) -> list[str]:
        """
        Function returns list of peak shape parameters names

        Parameters
        ---------
        line_type : str
            {'Gaussian', 'Split Gaussian', ... etc.}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        out : list[str]
            ['a', 'x0', 'dx' + 'add_params']
        """
        param_names = ['a', 'x0', 'dx']
        psp = peak_shapes_params()
        if line_type in psp and 'add_params' in psp[line_type]:
            for i in psp[line_type]['add_params']:
                param_names.append(i)
        return param_names

    def _init_model_params(self, line_type: str) -> list[float]:
        """
        Function returns list of initial parameter values.
        For param names ['a', 'x0', 'dx'] will return [float, float, float]

        Parameters
        ---------
        line_type : str
            {'Gaussian', 'Split Gaussian', ... etc.}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        out : list[float]
        """
        init_params = self.parent.initial_peak_parameters(line_type)
        init_model_params = [j for i, j in init_params.items() if i != 'x_axis']
        return init_model_params

    def _arrays_for_peak_guess(self) -> list[np.ndarray]:
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
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        guess_method = mw.ui.guess_method_cb.currentText()
        arrays = {'Average': [self.parent.data.averaged_spectrum]}
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if guess_method == 'Average groups':
            arrays = {key: [arr] for key, arr in context.preprocessing.stages.av_data.data.items()}
        elif guess_method == 'All':
            arrays = {key: [arr] for key, arr in data.items()}
        arrays_for_guess = self.split_array_for_fitting(arrays).values()
        result = []
        for i in arrays_for_guess:
            for j in i:
                result.append(j)
        return result

    @asyncSlot()
    async def _analyze_guess_results(self, result: list[ModelResult], param_names: list[str],
                                     line_type: str) \
            -> np.ndarray | Model:
        """
        The function creates structure for final fitting. After final fitting we have final template
        for deconvolution.

        Parameters
        ----------
        result: list[ModelResult]
            results of fitting for every spectrum and every interval range.

        param_names: list[str]
            Names of peak_shape parameters. Standard params are 'a', 'x0' and 'dx_right'. Other
            param names given from
             peak_shapes_params() in default_values.py

        line_type: str
            {'Gaussian', 'Split Gaussian', ... etc.}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        x_y_models_params: list[tuple[np.ndarray, np.ndarray, Model, Parameters]]
            result of self.models_params_after_guess()
        """
        mw = get_parent(self.parent, "MainWindow")
        self.parent.plotting.intervals_data = self._create_intervals_data(result, len(param_names))
        self.parent.data.all_ranges_clustered_x0_sd = await clustering_lines_intervals(
            self.parent.plotting.intervals_data,
            mw.ui.max_dx_dsb.value())
        x_y_models_params = self._models_params_after_guess(
            self.parent.data.all_ranges_clustered_x0_sd, param_names, line_type)
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
        mw = get_parent(self.parent, "MainWindow")
        if mw.ui.interval_checkBox.isChecked():
            intervals = [(mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value())]
        elif not mw.ui.intervals_gb.isChecked():
            x_axis = self.parent.get_x_axis()
            intervals = [(x_axis[0], x_axis[-1])]
        else:
            borders = list(mw.ui.fit_borders_TableView.model().column_data(0))
            x_axis = self.parent.get_x_axis()
            intervals = intervals_by_borders(borders, x_axis)

        data_by_intervals = create_intervals_data(result, peak_n_params, intervals)
        return data_by_intervals

    def _models_params_after_guess(self, all_ranges_clustered_x0_sd: list[np.ndarray],
                                   param_names: list[str],
                                   line_type: str) \
            -> list[tuple[np.ndarray, np.ndarray, Model, Parameters]]:
        """
        Preparation of sliced x_axis, y_axis of averaged spectrum, fit model and parameters from
        a list of wave numbers, divided into ranges.

        Parameters
        ----------
        all_ranges_clustered_x0_sd: list[ndarray]
            result of estimate_n_lines_in_range(x0, hwhm) for each range
            2D array with 2 columns: center of cluster x0 and standard deviation of each cluster

        param_names: list[str]
            Names of peak_shape parameters. Standard params are 'a', 'x0' and 'dx_right'. Other param names given from
             peak_shapes_params() in default_values.py

        line_type: str
            {'Gaussian', 'Split Gaussian', ... etc.}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        x_y_models_params: list[tuple[np.ndarray, np.ndarray, Model, Parameters]]
            tuples with x_axis, y_axis, fitting model, params for fitting model. For each cm-1 range.
        """
        mw = get_parent(self.parent, "MainWindow")
        sliced_average_array_by_ranges = self.split_array_for_fitting(
            {'Average': [self.parent.data.averaged_spectrum]})['Average']
        # form static_params for all ranges
        init_params = self.parent.initial_peak_parameters(line_type)
        static_params = (init_params, mw.ui.max_dx_dsb.value(),
                         mw.ui.input_table.model().min_fwhm() / 2.,
                         peak_shapes_params()[line_type]['func'],
                         get_config('fitting')['peak_shape_params_limits'], param_names,
                         mw.ui.l_ratio_doubleSpinBox.value())
        x_y_model_params = []
        for i, cur_range_clustered_x0_sd in enumerate(all_ranges_clustered_x0_sd):
            n_array = sliced_average_array_by_ranges[i]
            params, func_legend = params_func_legend(cur_range_clustered_x0_sd, n_array,
                                                     static_params)
            model = fitting_model(func_legend)
            x_y_model_params.append((n_array[:, 0], n_array[:, 1], model, params))
        return x_y_model_params
    # endregion

    def split_array_for_fitting(self, arrays: dict[str, list[np.ndarray]]) \
            -> dict[str, list[np.ndarray]]:
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
        mw = get_parent(self.parent, "MainWindow")
        split_arrays = {}
        if mw.ui.interval_checkBox.isChecked():
            for key, arr in arrays.items():
                split_arrays[key] = [cut_full_spectrum(arr[0], mw.ui.interval_start_dsb.value(),
                                                       mw.ui.interval_end_dsb.value())]
        elif mw.ui.intervals_gb.isChecked():
            borders = list(mw.ui.fit_borders_TableView.model().column_data(0))
            x_axis = self.parent.get_x_axis()
            intervals_idx = intervals_by_borders(borders, x_axis, idx=True)
            for key, arr in arrays.items():
                split_arrays[key] = split_by_borders(arr[0], intervals_idx)
        else:
            split_arrays = arrays
        return split_arrays