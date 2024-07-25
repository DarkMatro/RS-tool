import dataclasses
from collections import defaultdict
from copy import deepcopy
from gc import collect
from logging import debug

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from lmfit.model import ModelResult

from src import UndoCommand
from src.data.collections import NestedDefaultDict
from src.stages import curve_idx_from_par_name
from src.stages.fitting.functions.plotting import random_line_style


def fitting_metrics(fit_results: list[ModelResult]) -> tuple[str, np.ndarray, np.ndarray]:
    ranges, chisqr_av, redchi_av, aic_av, bic_av, rsquared_av, av_text = 0, [], [], [], [], [], ''
    sigma3_top, sigma3_bottom = np.array([]), np.array([])
    for z in fit_results:
        if not z:
            continue
        ranges += 1
        chisqr_av.append(z.chisqr)
        redchi_av.append(z.redchi)
        aic_av.append(z.aic)
        bic_av.append(z.bic)
        try:
            rsquared_av.append(z.rsquared)
        except Exception:
            debug("fit_result.rsquared error")
        dely = z.eval_uncertainty(sigma=3)
        sigma3_top = np.concatenate((sigma3_top, z.best_fit + dely))
        sigma3_bottom = np.concatenate((sigma3_bottom, z.best_fit - dely))
    if ranges != 0:
        av_text = "[[Average Fit Statistics]]" + '\n' \
                  + f"    chi-square         = {np.round(np.mean(chisqr_av), 6)}" + '\n' \
                  + f"    reduced chi-square = {np.round(np.mean(redchi_av), 6)}" + '\n' \
                  + f"    Akaike info crit   = {np.round(np.mean(aic_av), 6)}" + '\n' \
                  + f"    Bayesian info crit = {np.round(np.mean(bic_av), 6)}" + '\n' \
                  + f"    R-squared          = {np.round(np.mean(rsquared_av), 8)}" + '\n' + '\n'
    return av_text, sigma3_top, sigma3_bottom


class CommandAddDeconvLine(UndoCommand):
    """
    add row to self.ui.deconv_lines_table
    add curve to deconvolution_plotItem

    Parameters
    -------
    data: tuple[int, str]
        idx, line_type
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: tuple[int, str], parent, text: str, *args, **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self._idx, self._line_type = data
        self._legend = f'Curve {self._idx + 1}'
        self._style = random_line_style()
        self._line_params = self.parent.decomposition.initial_peak_parameters(self._line_type)
        self._line_param_names = self._line_params.keys()

    def redo_special(self):
        """
        Update data and input table columns
        """
        self.mw.ui.deconv_lines_table.model().append_row(self._legend, self._line_type, self._style,
                                                         self._idx)
        for param_name in self._line_param_names:
            if param_name != 'x_axis':
                self.mw.ui.fit_params_table.model().append_row(self._idx, param_name,
                                                               self._line_params[param_name])
        self.parent.decomposition.graph_drawing.add_deconv_curve_to_plot(self._line_params,
                                                                         self._idx, self._style,
                                                                         self._line_type)

    def undo_special(self):
        """
        Undo data and input table columns
        """
        self.mw.ui.deconv_lines_table.model().delete_row(self._idx)
        self.parent.decomposition.tables.decomp_lines.delete_deconv_curve(self._idx)
        self.mw.ui.fit_params_table.model().delete_rows(self._idx)

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        d = self.parent.decomposition
        d.graph_drawing.draw_sum_curve()
        d.graph_drawing.draw_residual_curve()
        self.parent.set_modified()


@dataclasses.dataclass
class FitCache:
    sum_ar: np.ndarray
    sigma3_top: np.ndarray
    sigma3_bottom: np.ndarray
    report_text: str
    fit_report: str
    params_stderr_for_filename_new: NestedDefaultDict
    params_stderr_for_filename_old: NestedDefaultDict
    df: pd.DataFrame


class CommandAfterFitting(UndoCommand):
    """
    1. Set parameters value
    2. Update graph
    3. Show report

    Parameters
    ----------
    rs
        Main window class
    results : list[ModelResult]
    static_params : list[tuple[int, str, int, str, callable]]
    filename : str
        self.current_spectrum_deconvolution_name - current spectrum
    description : str
        Description to set in tooltip
    """

    def __init__(self, data: list[ModelResult], parent, text: str, *args, **kwargs) -> None:
        self.stage = kwargs.pop('stage')
        self.static_params = kwargs.pop('static_params')
        self.filename = kwargs.pop('filename')
        super().__init__(data, parent, text, *args, **kwargs)
        df = self.mw.ui.fit_params_table.model().query_result(f"filename == {self.filename!r}")
        self.attrs = FitCache(sum_ar=self.stage.sum_array(),
                              sigma3_top=np.array([]),
                              sigma3_bottom=np.array([]), report_text='',
                              fit_report='',
                              params_stderr_for_filename_new=NestedDefaultDict(),
                              params_stderr_for_filename_old=self.stage.data.params_stderr[
                                  self.filename],
                              df=df)
        self.report_result_old = self.stage.data.report_result[self.filename] \
            if self.filename in self.stage.data.report_result else ''
        self.sigma3 = self.stage.data.sigma3[self.filename] \
            if self.filename in self.stage.data.sigma3 else None
        self.prepare_data()

    def prepare_data(self) -> None:
        av_text, self.attrs.sigma3_top, self.attrs.sigma3_bottom = fitting_metrics(self.data)
        for r in self.data:
            self.attrs.fit_report += self.edited_fit_report(r) + '\n' + '\n'
            # Find stderr for all parameters.
            if not r.errorbars:
                continue
            for k, v in r.params.items():
                idx, param_name = curve_idx_from_par_name(k)
                self.attrs.params_stderr_for_filename_new[idx][param_name] = v.stderr
        x_axis, _ = self.attrs.sum_ar
        if self.attrs.sigma3_top.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - self.attrs.sigma3_top.shape[0]
            zer = np.zeros(d)
            self.attrs.sigma3_top = np.concatenate((self.attrs.sigma3_top, zer))
        if self.attrs.sigma3_bottom.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - self.attrs.sigma3_bottom.shape[0]
            zer = np.zeros(d)
            self.attrs.sigma3_bottom = np.concatenate((self.attrs.sigma3_bottom, zer))
        self.attrs.report_text = av_text + self.attrs.fit_report + '\n' + '\n' if av_text \
            else self.attrs.fit_report

    def redo_special(self):
        """
        Update data and input table columns
        """
        if self.filename != '':
            self.mw.ui.fit_params_table.model().delete_rows_by_filenames([self.filename])
            self.stage.add_line_params_from_template(self.filename)
        for fit_result in self.data:
            self.stage.set_parameters_after_fit_for_spectrum(fit_result, self.filename)
        x_axis, _ = self.attrs.sum_ar
        self.stage.data.sigma3[self.filename] = (x_axis, self.attrs.sigma3_top,
                                                 self.attrs.sigma3_bottom)
        self.stage.graph_drawing.update_sigma3_curves(self.filename)
        self.stage.data.report_result[self.filename] = self.attrs.report_text
        self.mw.ui.report_text_edit.setText(self.attrs.report_text)
        self.stage.data.params_stderr[self.filename] = self.attrs.params_stderr_for_filename_new

    def undo_special(self):
        """
        Undo data and input table columns
        """
        self.mw.ui.fit_params_table.model().delete_rows_by_filenames([self.filename])
        self.mw.ui.fit_params_table.model().concat_df(self.attrs.df)
        self.stage.set_rows_visibility()
        x_axis, y_axis = self.attrs.sum_ar
        self.stage.curves.sum.setData(x=x_axis, y=y_axis)
        self.stage.data.report_result[self.filename] = self.report_result_old
        self.stage.show_current_report_result()
        if self.sigma3 is not None:
            self.stage.data.sigma3[self.filename] = self.sigma3[0], self.sigma3[1], self.sigma3[2]
        else:
            del self.stage.data.sigma3[self.filename]
            if self.stage.curves.sigma3_fill is not None:
                self.stage.curves.sigma3_fill.setVisible(False)
        self.stage.graph_drawing.update_sigma3_curves(self.filename)
        self.stage.data.params_stderr[self.filename] = self.attrs.params_stderr_for_filename_old

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.stage.show_all_roi()
        self.mw.ui.fit_params_table.model().sort_index()
        self.mw.ui.fit_params_table.model().model_reset_emit()
        self.stage.graph_drawing.redraw_curves_for_filename()
        self.stage.graph_drawing.draw_sum_curve()
        self.stage.graph_drawing.draw_residual_curve()
        self.stage.show_current_report_result()
        self.stage.graph_drawing.update_sigma3_curves()
        self.stage.set_rows_visibility()
        self.parent.set_modified()
        collect(2)

    def edited_fit_report(self, res: ModelResult) -> str:
        fit_report = res.fit_report(show_correl=False)
        param_legend = []
        line_types = self.mw.ui.deconv_lines_table.model().get_visible_line_types()
        for key in res.best_values.keys():
            idx, param_name = curve_idx_from_par_name(key)
            legend = line_types.loc[idx].Legend
            param_legend.append((key, legend + ' ' + param_name))
        for old, new in param_legend:
            fit_report = fit_report.replace(old, new)
        if '[[Fit Statistics]]' in fit_report:
            idx = fit_report.find('[[Fit Statistics]]')
            fit_report = fit_report[idx:]
        return fit_report


class CommandAfterGuess(UndoCommand):
    """
    1. deconv_lines_table clear and add new from guess result
    2. fit_params_table clear and add params for ''
    3. update fit_report
    4. delete all lines from plot and create new
    5. update sum, residual, sigma3 data

    Parameters
    ----------
    mw : MainWindow
        Main window class
    result : list[ModelResult]
    line_type : str
        Gaussian, Lorentzian... etc.
    n_params : int
        count of line parameters
    description : str
        Description to set in tooltip
    """

    def __init__(self, data: list[ModelResult], parent, text: str, *args, **kwargs) -> None:
        self.stage = kwargs.pop('stage')
        self.line_type = kwargs.pop('line_type')
        self.n_params: int = kwargs.pop('n_params')
        super().__init__(data, parent, text, *args, **kwargs)
        self._fit_report = ''
        self._df = {'lines_old': self.mw.ui.deconv_lines_table.model().dataframe().copy(),
                    'params_old': self.mw.ui.fit_params_table.model().dataframe().copy()}
        self.report_result_old = self.stage.data.report_result[''] \
            if '' in self.stage.data.report_result else ''
        self.sigma3_old = self.stage.data.sigma3[''] if '' in self.stage.data.sigma3 else None

    def redo_special(self):
        """
        f
        """
        self.mw.ui.deconv_lines_table.model().clear_dataframe()
        self.mw.ui.fit_params_table.model().clear_dataframe()
        self.mw.ui.report_text_edit.setText('')
        self.stage.remove_all_lines_from_plot()
        av_text, sigma3_top, sigma3_bottom = fitting_metrics(self.data)

        for r in self.data:
            self.process_result(r)
            self._fit_report += self.edited_fit_report(r.fit_report(show_correl=False))
        report_text = av_text + self._fit_report if av_text else self._fit_report
        self.stage.data.report_result.clear()
        self.stage.data.report_result[''] = report_text
        self.mw.ui.report_text_edit.setText(report_text)
        self.stage.graph_drawing.draw_sum_curve()
        self.stage.graph_drawing.draw_residual_curve()
        x_axis, _ = self.stage.sum_array()
        if sigma3_top.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - sigma3_top.shape[0]
            sigma3_top = np.concatenate((sigma3_top, np.zeros(d)))
        if sigma3_bottom.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - sigma3_bottom.shape[0]
            sigma3_bottom = np.concatenate((sigma3_bottom, np.zeros(d)))
        self.stage.data.sigma3[''] = x_axis, sigma3_top, sigma3_bottom
        self.stage.graph_drawing.update_sigma3_curves('')

    def undo_special(self):
        """
        Undo data
        """
        self.mw.ui.deconv_lines_table.model().set_dataframe(self._df['lines_old'])
        self.mw.ui.fit_params_table.model().set_dataframe(self._df['params_old'])
        self.stage.set_rows_visibility()
        self.stage.remove_all_lines_from_plot()
        self.stage.data.report_result[''] = self.report_result_old
        self.stage.data.show_current_report_result()
        if self.sigma3_old is not None:
            self.stage.data.sigma3[''] = self.sigma3_old[0], self.sigma3_old[1], self.sigma3_old[2]
        else:
            del self.stage.data.sigma3['']
            if self.stage.curves.sigma3_fill is not None:
                self.stage.curves.sigma3_fill.setVisible(False)
        self.stage.graph_drawing.update_sigma3_curves('')
        self.stage.data.graph_drawing.draw_all_curves()

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.stage.set_rows_visibility()
        self.parent.set_modified()

    def process_result(self, fit_result: ModelResult) -> None:
        params = fit_result.params
        idx, line_params = 0, {}
        rnd_style = random_line_style()
        # add fit lines and fit parameters table rows
        for i, j in enumerate(fit_result.best_values.items()):
            legend_param = j[0].replace('dot', '.').split('_', 1)
            if i % self.n_params == 0:
                line_params = {}
                rnd_style = random_line_style()
                idx = self.mw.ui.deconv_lines_table.model().append_row(legend_param[0],
                                                                       self.line_type, rnd_style)
            line_params[legend_param[1]] = j[1]
            v = np.round(j[1], 5)
            min_v = np.round(params[j[0]].min, 5)
            max_v = np.round(params[j[0]].max, 5)
            self.mw.ui.fit_params_table.model().append_row(idx, legend_param[1], v, min_v, max_v)
            if i % self.n_params == self.n_params - 1:
                self.stage.graph_drawing.add_deconv_curve_to_plot(line_params, idx, rnd_style,
                                                                  self.line_type)

    @staticmethod
    def edited_fit_report(fit_report: str) -> str:
        if '[[Fit Statistics]]' in fit_report:
            idx = fit_report.find('[[Fit Statistics]]')
            fit_report = fit_report[idx:]
        fit_report += '\n' + '\n'
        fit_report = fit_report.replace('dot', '.')
        return fit_report


@dataclasses.dataclass
class BatchCache:
    keys: set
    sum_ar: np.ndarray
    sigma3_conc_up: dict
    sigma3_conc_bottom: dict
    fit_reports: dict
    chisqr_av: dict
    redchi_av: dict
    aic_av: dict
    bic_av: dict
    rsquared_av: dict
    params_stderr_new: NestedDefaultDict
    params_stderr_old: NestedDefaultDict
    report_text: dict
    av_text: str
    dataset_new: pd.DataFrame | None
    df_fit_params: pd.DataFrame
    report_result_old: dict
    sigma3: dict
    dataset_old: pd.DataFrame


class CommandAfterBatchFitting(UndoCommand):
    """
    1. Set parameters value
    2. Update graph
    3. Update / Show report

    Parameters
    ----------
    data : list[tuple[str, ModelResult]]
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
    description : str
        Description to set in tooltip

    """

    def __init__(self, data: list[tuple[str, ModelResult]], parent, text: str, *args, **kwargs) \
            -> None:
        self.stage = kwargs.pop('stage')
        self.static_params: list[tuple[int, str, int, str, callable]] = kwargs.pop('static_params')
        self.dely: list[tuple[str, np.ndarray]] = kwargs.pop('dely')
        super().__init__(data, parent, text, *args, **kwargs)
        self.cache = BatchCache(keys=set(), sum_ar=self.stage.sum_array(), sigma3_conc_up={},
                                sigma3_conc_bottom={}, fit_reports={}, chisqr_av={}, redchi_av={},
                                aic_av={}, bic_av={}, rsquared_av={}, report_text={},
                                params_stderr_new=NestedDefaultDict(),
                                av_text='', dataset_new=None,
                                df_fit_params=deepcopy(
                                    self.mw.ui.fit_params_table.model().dataframe()),
                                report_result_old=deepcopy(self.stage.data.report_result),
                                sigma3=deepcopy(self.stage.data.sigma3),
                                dataset_old=deepcopy(
                                    self.mw.ui.deconvoluted_dataset_table_view.model().dataframe()),
                                params_stderr_old=self.stage.data.params_stderr)
        self.prepare_data()

    def prepare_data(self) -> None:
        self.cache.keys = {x for x, _ in self.data}
        for key in self.cache.keys:
            self.cache.fit_reports[key], self.cache.rsquared_av[key] = '', []
            self.cache.chisqr_av[key], self.cache.redchi_av[key] = [], []
            self.cache.aic_av[key], self.cache.bic_av[key] = [], []
            self.cache.sigma3_conc_up[key], self.cache.sigma3_conc_bottom[key] = (np.array([]),
                                                                                  np.array([]))
        line_types = self.mw.ui.deconv_lines_table.model().get_visible_line_types()
        x_axis, _ = self.cache.sum_ar
        for key, res in self.data:
            if not res:
                continue
            self.cache.fit_reports[key] += self.edited_fit_report(res, line_types) + '\n'
            if res.chisqr != 1e-250:
                self.cache.chisqr_av[key].append(res.chisqr)
            self.cache.redchi_av[key].append(res.redchi)
            self.cache.aic_av[key].append(res.aic)
            if res.bic != -np.inf:
                self.cache.bic_av[key].append(res.bic)
            # try:
            self.cache.rsquared_av[key].append(res.rsquared)
            # except Exception:
            #     debug("fit_result.rsquared error")
            # Find stderr for all parameters.
            if not res.errorbars:
                continue
            for k, v in res.params.items():
                idx, param_name = curve_idx_from_par_name(k)
                self.cache.params_stderr_new[key][idx][param_name] = v.stderr

        for i, item in enumerate(self.data):
            key, res = item
            if not self.dely[i] or not res:
                continue
            self.cache.sigma3_conc_up[key] = np.concatenate(
                (self.cache.sigma3_conc_up[key], res.best_fit + self.dely[i][1]))
            self.cache.sigma3_conc_bottom[key] = np.concatenate((self.cache.sigma3_conc_bottom[key],
                                                                 res.best_fit - self.dely[i][1]))
        # check that shape of sigma curves = shape of mutual x_axis
        for key in self.cache.keys:
            if self.cache.sigma3_conc_up[key].shape[0] < x_axis.shape[0]:
                d = x_axis.shape[0] - self.cache.sigma3_conc_up[key].shape[0]
                self.cache.sigma3_conc_up[key] = np.concatenate((self.cache.sigma3_conc_up[key],
                                                                 np.zeros(d)))
            if self.cache.sigma3_conc_bottom[key].shape[0] < x_axis.shape[0]:
                d = x_axis.shape[0] - self.cache.sigma3_conc_bottom[key].shape[0]
                self.cache.sigma3_conc_bottom[key] = np.concatenate((
                    self.cache.sigma3_conc_bottom[key], np.zeros(d)))
        self.create_report_text()
        self.create_av_text()

    def create_report_text(self):
        ranges = int(len(self.data) / len(self.cache.bic_av))
        for key in self.cache.keys:
            if ranges > 1:
                av_text = "[[Average For Spectrum Fit Statistics]]" + '\n' \
                          + (f"    chi-square         = "
                             f"{np.round(np.mean(self.cache.chisqr_av[key]), 6)}") + '\n' \
                          + (f"    reduced chi-square = "
                             f"{np.round(np.mean(self.cache.redchi_av[key]), 6)}") + '\n' \
                          + (f"    Akaike info crit   = "
                             f"{np.round(np.mean(self.cache.aic_av[key]), 6)}") + '\n' \
                          + (f"    Bayesian info crit = "
                             f"{np.round(np.mean(self.cache.bic_av[key]), 6)}") + '\n' \
                          + (f"    R-squared          = "
                             f"{np.round(np.mean(self.cache.rsquared_av[key]), 8)}") + '\n' + '\n'
                self.cache.report_text[key] = av_text + self.cache.fit_reports[key]
            else:
                self.cache.report_text[key] = self.cache.fit_reports[key]

    def create_av_text(self):
        flat_list_c = [item for sublist in list(self.cache.chisqr_av.values()) for item in sublist]
        flat_list_r = [item for sublist in list(self.cache.redchi_av.values()) for item in sublist]
        flat_list_a = [item for sublist in list(self.cache.aic_av.values()) for item in sublist]
        flat_list_b = [item for sublist in list(self.cache.bic_av.values()) for item in sublist]
        flat_list = [item for sublist in list(self.cache.rsquared_av.values()) for item in sublist]
        self.cache.av_text = ("[[Усредненная статистика по всем спектрам]]" + '\n'
                              + f"    chi-square         = {np.round(np.mean(flat_list_c), 6)}"
                              + '\n'
                              + f"    reduced chi-square = {np.round(np.mean(flat_list_r), 6)}"
                              + '\n'
                              + f"    Akaike info crit   = {np.round(np.mean(flat_list_a), 6)}"
                              + '\n'
                              + f"    Bayesian info crit = {np.round(np.mean(flat_list_b), 6)}"
                              + '\n'
                              + f"    R-squared          = {np.round(np.mean(flat_list), 8)}"
                              + '\n' + '\n')

    def redo_special(self):
        """
        f
        """
        self.stage.data.report_result.clear()
        self.mw.ui.fit_params_table.model().delete_rows_by_filenames(self.cache.keys)
        self.stage.b.add_line_params_from_template_batch(self.cache.keys)
        for key, fit_result in self.data:
            if key in self.stage.data.params_stderr:
                del self.stage.data.params_stderr[key]
            if not fit_result:
                continue
            self.stage.set_parameters_after_fit_for_spectrum(fit_result, key)
        x_axis, _ = self.cache.sum_ar
        for key in self.cache.keys:
            self.stage.data.sigma3[key] = (x_axis, self.cache.sigma3_conc_up[key],
                                           self.cache.sigma3_conc_bottom[key])
        for key in self.cache.keys:
            self.stage.data.report_result[key] = self.cache.report_text[key]
        self.cache.dataset_new = deepcopy(self.stage.create_deconvoluted_dataset_new())
        self.mw.ui.deconvoluted_dataset_table_view.model().set_dataframe(self.cache.dataset_new)
        for i in self.stage.data.report_result:
            self.stage.data.report_result[i] += '\n' + '\n' + self.cache.av_text
        for k, v in self.cache.params_stderr_new.items():
            self.stage.data.params_stderr[k] = v

    def undo_special(self):
        """
        Undo data
        """
        self.mw.ui.fit_params_table.model().set_dataframe(self.cache.df_fit_params)
        self.stage.data.report_result = deepcopy(self.cache.report_result_old)
        if self.cache.sigma3 is not None:
            self.stage.data.sigma3 = deepcopy(self.cache.sigma3)
        else:
            self.stage.data.sigma3.clear()
            self.stage.curves.sigma3_fill.setVisible(False)
        self.mw.ui.deconvoluted_dataset_table_view.model().set_dataframe(self.cache.dataset_old)
        self.stage.data.params_stderr = self.cache.params_stderr_old

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.stage.graph_drawing.redraw_curves_for_filename()
        self.stage.graph_drawing.draw_sum_curve()
        self.stage.graph_drawing.draw_residual_curve()
        self.stage.show_current_report_result()
        self.stage.graph_drawing.update_sigma3_curves()
        self.stage.set_rows_visibility()
        self.stage.update_ignore_features_table()
        self.mw.ui.fit_params_table.model().sort_index()
        self.mw.ui.fit_params_table.model().model_reset_emit()
        self.parent.set_modified()

    @staticmethod
    def edited_fit_report(fit_result: ModelResult, line_types: pd.DataFrame) -> str:
        fit_report = fit_result.fit_report(show_correl=False)
        param_legend = []
        for key in fit_result.best_values.keys():
            idx, param_name = curve_idx_from_par_name(key)
            legend = line_types.loc[idx].Legend
            param_legend.append((key, legend + ' ' + param_name))
        for old, new in param_legend:
            fit_report = fit_report.replace(old, new)
        if '[[Fit Statistics]]' in fit_report:
            idx = fit_report.find('[[Fit Statistics]]')
            fit_report = fit_report[idx:]
        return fit_report


class CommandUpdateDataCurveStyle(UndoCommand):
    """


    Parameters
    ----------
    data : tuple[dict, dict]
        new_style, old_style
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
    description : str
        Description to set in tooltip

    """

    def __init__(self, data: tuple[dict, dict], parent, text: str, *args, **kwargs) \
            -> None:
        self.stage = kwargs.pop('stage')
        self.curve_type = kwargs.pop('curve_type')
        super().__init__(data, parent, text, *args, **kwargs)

    def redo_special(self):
        """
        f
        """
        self.stage.set_pen(self.data[0], self.curve_type)


    def undo_special(self):
        """
        Undo data
        """
        self.stage.set_pen(self.data[1], self.curve_type)


    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.set_modified()


class CommandDeconvLineParameterChanged(UndoCommand):
    """


    Parameters
    ----------
    stage - delegate
    data : tuple[int, float, float]
        index, new_float, current_float
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
    description : str
        Description to set in tooltip

    """

    def __init__(self, data: tuple[int, float, float], parent, text: str, *args, **kwargs) \
            -> None:
        self.stage = kwargs.pop('stage')
        self.model = kwargs.pop('model')
        self.line_index = kwargs.pop('line_index')
        self.param_name = kwargs.pop('param_name')
        super().__init__(data, parent, text, *args, **kwargs)

    def redo_special(self):
        """
        f
        """
        self.model.setData(self.data[0], self.data[1], Qt.EditRole)
        self.stage.sigLineParamChanged.emit(self.data[1], self.line_index, self.param_name)

    def undo_special(self):
        """
        Undo data
        """
        self.model.setData(self.data[0], self.data[2], Qt.EditRole)
        self.stage.sigLineParamChanged.emit(self.data[2], self.line_index, self.param_name)

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.decomposition.graph_drawing.draw_sum_curve()
        self.parent.decomposition.graph_drawing.draw_residual_curve()
        self.parent.set_modified()
