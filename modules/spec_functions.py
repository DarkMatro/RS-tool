import copy
from logging import info, debug
from math import ceil as ceil
from os import environ
from os import getpid
from pathlib import Path
from re import findall
from typing import Any

import lmfit.model
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from numba import njit, float64, int64
from numpy import ndarray
from numpy.random import default_rng
from psutil import Process
from pyqtgraph import mkPen, PlotCurveItem, ROI, mkBrush
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QColor, QTextCursor
from scipy.cluster.vq import kmeans
from scipy.signal import peak_widths, peak_prominences, savgol_filter, argrelmin
from scipy.stats import mode

from modules.fit_single_peak import ExpSpec, CalcPeak, fit_single_peak, moving_average_molification


# region RS

def invert_color(color: str) -> str:
    rgb = QColor(color).getRgb()
    new_r = 255 - rgb[0]
    new_g = 255 - rgb[1]
    new_b = 255 - rgb[2]
    return QColor(new_r, new_g, new_b, rgb[3]).name()


def read_preferences() -> list[str, str, str, str, str, str, str]:
    theme_d = 'Dark'
    theme_color_d = 'Amber'
    recent_limit_d = '10'
    undo_limit_d = '20'
    auto_save_minutes_d = '5'
    plot_font_size_d = '10'
    axis_label_font_size_d = '14'
    with open('preferences.txt', 'r') as f:
        theme_from_file_f = f.readline().strip()
        theme_color_from_file_f = f.readline().strip()
        recent_limit_from_file_f = f.readline().strip()
        undo_limit_f = f.readline().strip()
        auto_save_minutes_f = f.readline().strip()
        plot_font_size_f = f.readline().strip()
        axis_label_font_size_f = f.readline().strip()
    theme_from_file = theme_from_file_f if theme_from_file_f else theme_d
    theme_color_from_file_f = theme_color_from_file_f if theme_color_from_file_f else theme_color_d
    recent_limit_from_file = recent_limit_from_file_f if recent_limit_from_file_f else recent_limit_d
    undo_limit_from_file = undo_limit_f if undo_limit_f else undo_limit_d
    auto_save_minutes_from_file = auto_save_minutes_f if auto_save_minutes_f else auto_save_minutes_d
    plot_font_size_from_file = plot_font_size_f if plot_font_size_f else plot_font_size_d
    axis_label_font_size_from_file = axis_label_font_size_f if axis_label_font_size_f else axis_label_font_size_d
    return [theme_from_file,
            theme_color_from_file_f,
            recent_limit_from_file,
            undo_limit_from_file,
            auto_save_minutes_from_file,
            plot_font_size_from_file,
            axis_label_font_size_from_file]


def get_memory_used() -> float:
    return Process(getpid()).memory_info().rss / 1024 ** 2


def random_rgb() -> int:
    rnd_gen = default_rng()
    rnd_rgb = rnd_gen.random() * 1e9
    return int(rnd_rgb)


def curve_pen_brush_by_style(style: dict) -> tuple[mkPen, mkBrush]:
    color = style['color']
    color.setAlphaF(1.0)
    pen = mkPen(color=color, style=style['style'], width=style['width'])
    fill_color = style['color'] if style['use_line_color'] else style['fill_color']
    fill_color.setAlphaF(style['fill_opacity'])
    brush = mkBrush(fill_color) if style['fill'] else None
    return pen, brush


def random_line_style() -> dict:
    return {'color': QColor().fromRgb(random_rgb()),
            'style': Qt.PenStyle.SolidLine,
            'width': 1.0,
            'fill': False,
            'use_line_color': True,
            'fill_color': QColor().fromRgb(random_rgb()),
            'fill_opacity': 0.5}


def check_rs_tool_folder() -> None:
    username = environ.get('USERNAME')
    path = '/users/' + str(username) + '/Documents/RS-tool'
    if not Path(path).exists():
        Path(path).mkdir()


def check_recent_files() -> None:
    path = 'recent-files.txt'
    if not Path(path).exists():
        with open(path, 'w') as text_file:
            text_file.write('')


def check_preferences_file() -> None:
    path = 'preferences.txt'
    if not Path(path).exists():
        with open(path, 'w') as text_file:
            text_file.write('Dark ' + '\n' + 'Default' + '\n' + '10' + '\n' + '20' + '\n' + '16' + '\n' + '12' + '\n' +
                            '13' + '\n')


# endregion


# region Import

def import_spectrum(file: str, laser_wl: float) \
        -> tuple[str, ndarray, int, ndarray | int | float | complex, ndarray | int | float | complex, float]:
    """
    INPUT - filename
    OUTPUT - numpy 2D array, basename, group_number, min_nm, max_nm, fwhm
    """
    n_array = np.loadtxt(file)
    basename_of_file = Path(file).name
    group_number = get_group_number_from_filename(basename_of_file)
    x_axis = n_array[:, 0]
    min_nm = np.min(x_axis)

    max_nm = np.max(x_axis)
    if min_nm < laser_wl:
        fwhm = get_laser_peak_fwhm(n_array, laser_wl, min_nm, max_nm)
    else:
        fwhm = 0.
    return basename_of_file, n_array, group_number, min_nm, max_nm, fwhm


def get_group_number_from_filename(basename_with_group_number: str) -> int:
    """
    Using in import_spectrum only.
    Supported patterns: '1 filename.txt', '1_filename.asc', '[2]_filename.asc', '[2] filename.asc'
    @param basename_with_group_number: str - filename
    @return: int - group number
    """
    result_just_number = findall(r'^\d+', basename_with_group_number)
    result_square_brackets = get_result_square_brackets(basename_with_group_number)
    result_round_brackets = findall(r'^\(\d+\)', basename_with_group_number)
    if result_round_brackets:
        result_round_brackets = findall(r'\d', result_round_brackets[0])

    if result_just_number and result_just_number[0] != '':
        group_number_str = result_just_number[0]
    elif result_square_brackets and result_square_brackets[0] != '':
        group_number_str = result_square_brackets[0]
    elif result_round_brackets and result_round_brackets[0] != '':
        group_number_str = result_round_brackets[0]
    else:
        group_number_str = '0'

    return int(group_number_str)


def get_result_square_brackets(basename_with_group_number: str) -> list[Any]:
    result_square_brackets = findall(r'^\[\d]', basename_with_group_number)
    if result_square_brackets:
        result_square_brackets = findall(r'\d', result_square_brackets[0])
    return result_square_brackets


# endregion

# region Convert

def convert(i: tuple[str, np.ndarray], nearest_idx: int, laser_nm: float = 784.5, max_ccd_value: float = 65536.0) \
        -> tuple[str, Any, Any, float, Any]:
    filename = i[0]
    array = i[1]
    x_axis = array[:, 0]
    y_axis = array[:, 1][:nearest_idx]
    max_value = np.amax(y_axis)
    if max_value >= max_ccd_value:
        args = get_args_by_value(y_axis, max_ccd_value)
        if args:
            idx = np.mean(args)
            local_maxima = int(idx)
        else:
            local_maxima = np.argmax(y_axis)
    else:
        local_maxima = np.argmax(y_axis)
    base_wave_length = array[local_maxima][0]
    new_x_axis = convert_nm_to_cm(x_axis, base_wave_length, laser_nm)
    n_array = np.vstack((new_x_axis, array[:, 1])).T
    fwhm_cm = get_laser_peak_fwhm(n_array, 0, np.min(new_x_axis), np.max(new_x_axis))
    y_axis = n_array[:, 1]
    filtered = savgol_filter(y_axis, 9, 3)
    noise = y_axis - filtered
    snr = y_axis.max() / noise.std()
    return filename, n_array, base_wave_length, fwhm_cm, round(snr, 2)


def get_laser_peak_fwhm(array: np.ndarray, laser_wl: float, min_nm: float, max_nm: float) -> float:
    # get +-5 nm range of array from laser peak
    diff = 40 if laser_wl == 0 else 5
    left_nm = laser_wl - diff
    right_nm = laser_wl + diff
    x_axis = array[:, 0]
    array_len = array.shape[0]
    dx = (max_nm - min_nm) / array_len
    left_idx = find_nearest_idx(x_axis, left_nm)
    right_idx = find_nearest_idx(x_axis, right_nm)
    laser_peak_array = array[left_idx:right_idx]
    # fit peak and find fwhm
    y_axis = laser_peak_array[:, 1]
    max_idx = find_nearest_idx(y_axis, np.max(y_axis))
    peaks = [max_idx]
    prominences = peak_prominences(y_axis, peaks)
    results_half = peak_widths(y_axis, peaks, prominence_data=prominences)
    fwhm = results_half[0][0] * dx
    return np.round(fwhm, 5)


@njit(float64[:](float64[:], float64, float64), fastmath=True)
def convert_nm_to_cm(x_axis: np.ndarray, base_wave_length: float, laser_nm: float) -> np.ndarray:
    return (1e7 / laser_nm) - (1e7 / (x_axis + laser_nm - base_wave_length))


def find_fluorescence_beginning(i: tuple[str, np.ndarray], factor: int = 1) -> int:
    idx_100cm = find_nearest_idx(i[1][:, 0], 100)
    part_of_y_axis = i[1][:, 1][idx_100cm:]
    grad = np.gradient(part_of_y_axis)
    grad_std = np.std(grad)
    start_of_grow = np.argmax(grad > grad_std)
    negative_grads = np.argwhere(grad[start_of_grow:] < 0)
    f = 0
    end_of_grow_idx = None
    for i in range(len(negative_grads)):
        current_idx = negative_grads[i][0]
        prev_idx = negative_grads[i - 1][0]
        if current_idx - prev_idx == 1:
            f += 1
        else:
            f = 0
        if f == factor - 1:
            end_of_grow_idx = current_idx
            break
    return idx_100cm + start_of_grow + end_of_grow_idx


# endregion

# region Interpolate

def interpolate(array: np.ndarray, filename: str, ref_file: np.ndarray) -> tuple[str, np.ndarray]:
    x_axis_old = array[:, 0]
    y_axis_old = array[:, 1]
    x_axis_new = ref_file[:, 0]
    y_axis_new = np.interp(x_axis_new, x_axis_old, y_axis_old)
    return_array = np.vstack((x_axis_new, y_axis_new)).T
    return filename, return_array


# endregion

# region Arrays routines
@njit(fastmath=True, cache=True)
def get_args_by_value(array: np.ndarray, value: int) -> list[int]:
    return_list = []
    prev_value = float(0)
    for i in range(len(array)):
        if array[i] == value:
            return_list.append(i)
            prev_value = array[i]
        elif prev_value == value:
            break
    return return_list


@njit(int64(float64[:], float64), fastmath=True)
def find_nearest_idx(array: np.ndarray, value: float) -> int:
    """
    find_nearest_idx(array: np.ndarray, value: float)

    Return an index of value nearest to input value in array

    Parameters
    ---------
    array : 1D ndarray
        Search the nearest value in this array
    value : float
        Input value to compare with values in array

    Returns
    -------
    out : int
        Index of nearest value

    Examples
    --------
    >>> ar = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> find_nearest_idx(ar, 2.9)
    2
    """
    idx = np.abs(array - value).argmin()
    array_v = np.round(array[idx], 5)
    value_r = np.round(value, 5)
    if np.abs(array_v - value_r) > 1 and idx != 0:
        return idx - 1
    else:
        return idx


def find_nearest(array: np.ndarray, value: float, take_left_value: bool = False) -> float:
    idx = np.abs(array - value).argmin()
    if take_left_value and array[idx] > value and idx != 0:
        return array[idx - 1]
    else:
        return array[idx]


def find_nearest_by_idx(array: np.ndarray, idx: int, take_left_value: bool = False) -> float:
    value = array[idx]
    idx = np.abs(array - value).argmin()
    if take_left_value and array[idx] > value and idx != 0:
        return array[idx - 1]
    else:
        return array[idx]


# endregion

# region Despike

def subtract_cosmic_spikes_moll(input_tuple: tuple[str, np.ndarray], laser_fwhm: float, laser_wl: float = 785,
                                maxima_count: int = 2, width: float = 0.0, _print_messages: bool = False) \
        -> tuple[str, Any, list | list[Any]]:
    """
    width : width of cosmic spikes, in units of x-axis,
                                    not in pixels!
        By default, width = distance be two pixels of CCD;
            if the data were automatically processed (like, say, in Bruker Senterra),
            then the width should be set to ~2
    The script assumes that x are sorted in the ascending order.
    """

    filename = input_tuple[0]
    input_spectrum = input_tuple[1]
    x = input_spectrum[:, 0]
    y = input_spectrum[:, 1]
    # dx = (np.max(x) - np.min(x)) / x.shape[0]
    spectrum_nm_range = np.abs(x[-1] - x[0])
    if width == 0 or width < 0.05:
        width = spectrum_nm_range / (len(x) - 1) * 1.6
        # info(f"[auto]width = {width}")
    # else:
    #     info(f"input width = {width}")
    # info('laser peak fwhm  = %s', laser_fwhm)
    width = min([width, laser_fwhm])
    width_in_pixels = width * len(x) / spectrum_nm_range

    # calculate the mollification_width:
    mollification_width = int(2 * ceil(width_in_pixels) + 1)
    # info(f"width in pixels = {width_in_pixels}")
    # info(f"mollification_width, pixels = {mollification_width}")
    # область возле лазерного пика не исследуем
    idx = find_nearest_idx(x, laser_wl + 5)
    x_part1 = x[0:idx]
    x = x[idx:]
    y_part1 = y[0:idx]
    y = y[idx:]
    spectrum = ExpSpec(x, y)
    iteration_number = 0
    negative_signs_list = []
    for i in range(maxima_count):
        negative_signs_list.append(-1)
    subtracted_peaks = []
    subtracted_peaks_idx = []
    peaks_subtracted = False
    while iteration_number < 5:
        if environ['CANCEL'] == '1':
            print('break CANCEL')
            break
        # info(f"iteration number = {iteration_number}")
        y_moll = moving_average_molification(spectrum.y, struct_el=mollification_width)
        y_mod_score = np.abs(spectrum.y - y_moll)
        # if print_messages:
        #     matplotlib.pyplot.plt.plot(x, y_mod_score / np.max(y_mod_score), 'r', linewidth=0.5)
        #     matplotlib.pyplot.plt.plot(x, (spectrum.y - np.min(spectrum.y)) / (np.max(spectrum.y)
        #     - np.min(spectrum.y)), 'k', linewidth=1)
        #     matplotlib.pyplot.plt.title('iteration ' + str(iteration_number))
        #     matplotlib.pyplot.plt.show()
        iteration_number += 1
        # find n largest peak:
        top_peak_indexes = []
        stop_loop = False
        for i in range(maxima_count):
            top_peak_index = (np.abs(y_mod_score)).argmax()
            stop_loop = top_peak_index in subtracted_peaks_idx or stop_loop
            top_peak_indexes.append(top_peak_index)
            subtracted_peaks_idx.append(top_peak_index)
            y_mod_score[top_peak_index] = 0
            for k in range(mollification_width):
                y_mod_score[top_peak_index + k] = 0
            for k in range(mollification_width):
                y_mod_score[top_peak_index - k] = 0
        if stop_loop:
            break
        cosmic_positions = []
        for i in range(maxima_count):
            cosmic_positions.append(x[top_peak_indexes[i]])
        peak_signs = []
        for i in range(maxima_count):
            y_i = spectrum.y[top_peak_indexes[i]] - y_moll[top_peak_indexes[i]]
            peak_signs.append(np.sign(y_i))
        # fit:
        # if peak_signs == negative_signs_list:
        #     debug(f"all peaks are negative")
        cosmic_spikes = get_cosmic_spikes(maxima_count, peak_signs, spectrum, cosmic_positions, width)
        # subtract:
        result_of_subtract = subtract_spectrum_and_peak(x, maxima_count, cosmic_spikes, width, spectrum,
                                                        subtracted_peaks, cosmic_positions)
        all_peaks_not_fitted = result_of_subtract[0]
        if environ['CANCEL'] == '1':
            print('break CANCEL')
            break
        if all_peaks_not_fitted:
            # debug(f"all_peaks_not_fitted - BREAK")
            break
        all_peaks_are_too_broad = result_of_subtract[1]
        if np.max(all_peaks_are_too_broad) == np.min(all_peaks_are_too_broad) and all_peaks_are_too_broad[0] == 1:
            # debug(f"all_peaks_are_too_broad - BREAK")
            break
        subtracted_peaks = result_of_subtract[2]
        peaks_subtracted = result_of_subtract[3]
        spectrum = result_of_subtract[4]

    x = np.concatenate((x_part1, x))

    spectrum_y = np.concatenate((y_part1, spectrum.y))
    # if print_messages:
    #     y = np.concatenate((y_part1, y))
    #     spectrum = ExpSpec(x, spectrum_y)
    #     matplotlib.pyplot.plt.plot(x, y, 'k', linewidth=1)
    #     matplotlib.pyplot.plt.plot(x, spectrum.y, 'r', linewidth=1)
    #     matplotlib.pyplot.plt.title('final')
    #     matplotlib.pyplot.plt.show()
    return_array = np.vstack((x, spectrum_y)).T
    if peaks_subtracted:
        return filename, return_array, subtracted_peaks


def subtract_spectrum_and_peak(x: list[np.ndarray | ExpSpec], maxima_count: int, cosmic_spikes: list[int | CalcPeak],
                               width: float, spectrum: ExpSpec, subtracted_peaks: list, cosmic_positions: list) \
        -> tuple[bool, np.ndarray, list, bool, ExpSpec]:
    all_peaks_not_fitted = True
    all_peaks_are_too_broad = np.zeros(maxima_count)
    peaks_subtracted = False
    for i in range(maxima_count):
        # if cosmic_spikes[i] == -1:
        #     debug(f"negative sign of peak at {cosmic_positions[i]}")
        # continue
        peak2subtract = CalcPeak(x)
        peak2subtract.specs_array = cosmic_spikes[i].specs_array
        if peak2subtract.fwhm > width:
            all_peaks_are_too_broad[i] = 1
            # debug(f"the current peak is broad, not subtracted")
            continue
        # debug(f"fwhm of the fitted peak = {peak2subtract.fwhm}")
        spectrum = ExpSpec(x, spectrum.y - peak2subtract.curve)
        subtracted_peaks.append(cosmic_positions[i])
        peaks_subtracted = True
        all_peaks_not_fitted = False
    return all_peaks_not_fitted, all_peaks_are_too_broad, subtracted_peaks, peaks_subtracted, spectrum


def get_cosmic_spikes(maxima_count: int, peak_signs: list, spectrum: ExpSpec, cosmic_positions: list,
                      width: float) -> list[int | CalcPeak]:
    cosmic_spikes = []
    for i in range(maxima_count):
        cosmic_spike = fit_single_peak(spectrum, peak_position=cosmic_positions[i], fit_range=(
            cosmic_positions[i] - 8 * width,
            cosmic_positions[i] + 8 * width), peak_sign=peak_signs[i], fwhm=width)
        cosmic_spikes.append(cosmic_spike)
    return cosmic_spikes


# endregion

# region Baseline

def find_first_right_local_minimum(i: tuple[str, np.ndarray]) -> int:
    y = i[1][:, 1]
    y_min = argrelmin(y)[0][-1]
    return y_min


def find_first_left_local_minimum(i: tuple[str, np.ndarray]) -> int:
    y = i[1][:, 1]
    y_min = argrelmin(y)[0][0]
    return y_min


# endregion

# region Cut / Trim
@njit(cache=True, fastmath=True)
def cut_spectrum(i: tuple[str, np.ndarray], value_start: int, value_end: int) -> tuple[str, np.ndarray]:
    array = i[1]
    name = i[0]
    x_axis = array[:, 0]
    idx_start = find_nearest_idx(x_axis, value_start)
    idx_end = find_nearest_idx(x_axis, value_end)
    return name, array[idx_start: idx_end + 1]


def cut_full_spectrum(array: np.ndarray, value_start: float, value_end: float) -> np.ndarray:
    x_axis = array[:, 0]
    idx_start = find_nearest_idx(x_axis, value_start)
    idx_end = find_nearest_idx(x_axis, value_end)
    result = array[idx_start: idx_end + 1]
    return result


def cut_axis(axis: np.ndarray, value_start: float, value_end: float) -> tuple[Any, int, int]:
    idx_start = find_nearest_idx(axis, value_start)
    idx_end = find_nearest_idx(axis, value_end)
    return axis[idx_start: idx_end + 1], idx_start, idx_end


# endregion

# region Average
def get_average_spectrum(item: list[np.ndarray], method: str = 'Mean') -> np.ndarray:
    x_axis = item[0][:, 0]
    y_axes = []
    for array in item:
        y_axes.append(array[:, 1])
    np_y_axes = np.array(y_axes)
    if method == 'Mean':
        np_y_axis = np.mean(np_y_axes, axis=0)
    else:
        np_y_axis = np.median(np_y_axes, axis=0)
    return np.vstack((x_axis, np_y_axis)).T


# endregion

# region Fitting plot

def set_roi_size_pos(params: tuple[float, float, float], roi: ROI, update: bool = True) -> None:
    a, x0, dx = params
    set_roi_size(dx, a, roi, update=update)
    set_roi_pos(x0, roi)


def set_roi_size(x: float, y: float, roi: ROI, center=None, finish: bool = False, update: bool = True) -> None:
    if center is None:
        center = [0, 1]
    new_size = QPointF()
    new_size.setX(x)
    new_size.setY(y)
    roi.setSize(new_size, center, finish=finish, update=update)


def set_roi_pos(x0: float, roi: ROI) -> None:
    new_pos = QPointF()
    new_pos.setX(x0)
    new_pos.setY(0.0)
    roi.setPos(new_pos, finish=True, update=True)


def get_curve_for_deconvolution(n_array: np.ndarray, idx: int, style: dict) -> PlotCurveItem:
    curve = PlotCurveItem(skipFiniteCheck=True, useCache=True, clickable=True, name=idx)
    curve.setSkipFiniteCheck(True)
    curve.setClickable(True)
    pen, brush = curve_pen_brush_by_style(style)
    curve.setPen(pen)
    curve.setData(x=n_array[:, 0], y=n_array[:, 1], fillLevel=0.0, brush=brush)
    return curve


def packed_current_line_parameters(df_params: pd.DataFrame, line_type: str, peak_shapes_params: dict) -> dict:
    query_result_dx = df_params.loc['dx']
    query_result_a = df_params.loc['a']
    query_result_x0 = df_params.loc['x0']
    a = query_result_a['Value']
    min_a = query_result_a['Min value']
    max_a = query_result_a['Max value']
    x0 = query_result_x0['Value']
    min_x0 = query_result_x0['Min value']
    max_x0 = query_result_x0['Max value']
    dx = query_result_dx['Value']
    min_dx = query_result_dx['Min value']
    max_dx = query_result_dx['Max value']
    result = {'a': a, 'min_a': min_a, 'max_a': max_a,
              'x0': x0, 'min_x0': min_x0, 'max_x0': max_x0,
              'dx': dx, 'min_dx': min_dx, 'max_dx': max_dx}
    if 'add_params' not in peak_shapes_params[line_type]:
        return result
    add_params = peak_shapes_params[line_type]['add_params']
    for param_name in add_params:
        query_result_add_param = df_params.loc[param_name]
        if not query_result_add_param.empty:
            result[param_name] = query_result_add_param['Value']
            result['min_%s' % param_name] = query_result_add_param['Min value']
            result['max_%s' % param_name] = query_result_add_param['Max value']
    return result

# endregion

# region Fitting


def fitting_model(func_legend: list[tuple]) -> Model:
    """
    Prepare fitting model. Final model is a sum of different line models.

    Parameters
    ---------
    func_legend : list[tuple[func, str]]
        using to create fit Model. func is a line function from spec_functions/ peak_shapes, str is curve legend

    Returns
    -------
    out : Model()
    """
    model = None
    for i, f_l in enumerate(func_legend):
        if i == 0:
            model = Model(f_l[0], prefix=f_l[1])
            continue
        model += Model(f_l[0], prefix=f_l[1])
    return model


def fit_model(model, y, params, x, method: str) -> ModelResult:
    result = model.fit(y, params, x=x, method=method)
    return result


def fit_model_batch(model, y, params, x, method: str, key: str) -> tuple[str, ModelResult]:
    result = model.fit(y, params, x=x, method=method)
    return key, result


def guess_peaks(n_array: np.ndarray, parameters_to_guess: dict) \
        -> lmfit.model.ModelResult:
    """
    Итеративно подбирается состав линий, аппроксимирующий спектр n_array.
    Если изначально есть линий, то начинается процесс не с 0, а с имеющихся уже линий.
    Цикл заканчивается когда новая добавленная линий имеет амплитуду меньше noise_level.
    noise_level определяется из отношения сигнал/шум SNR.
    noise_level = y_max / SNR

    @param n_array: 2D array x|y
    @param parameters_to_guess: dict with keys:
    'func': callable function for peak shape calculaction,
    'param_names': список наименований параметров (например ['a', 'x0', 'dx']),
    'init_model_params': list[float] начальные значения параметров для данного спектра и типа линии,
    'min_fwhm': float - минимальное значение ширины линии, определяется из таблицы (берется минимальное из всех),
    'method': str - optimization method,
    'params_limits': dict[str, tuple[float, float]] - see self.peak_shape_params_limits,
    'mean_snr': float - mean_snr,
    'max_dx': float - max_dx,
    Следующие параметры пустые если нет линий. Если на начало анализа есть уже созданные пользователем линии,
     то параметры будут заполнены:
    'func_legend': list[tuple] - callable func, legend, func - функция для вычисления линии,
        legend - префикс для линии в модели
    'params': Parameters() - параметры имеющихся линий,
    'prev_k': list[int],
    'zero_under_05_hwhm_curve_k': dict - (x0_arg - x0_arg_dx_l, x0_arg_dx_r - x0_arg)
    @return: tuple[lmfit.model.ModelResult, Parameters, np.ndarray]
    """
    x_input = n_array[:, 0]
    y_input = n_array[:, 1]
    y_residual = n_array[:, 1].copy()
    params_to_guess = copy.deepcopy(parameters_to_guess)
    noise_level = params_to_guess['noise_level']
    max_dx = params_to_guess['max_dx']
    func = params_to_guess['func']
    param_names = params_to_guess['param_names']
    init_model_params = params_to_guess['init_model_params']
    min_fwhm = params_to_guess['min_fwhm']
    params_limits = params_to_guess['params_limits']
    method = params_to_guess['method']
    func_legend_from_params = params_to_guess['func_legend']
    params_from_params = params_to_guess['params']
    prev_k = params_to_guess['prev_k']
    zero_under_05_hwhm_curve_k = params_to_guess['zero_under_05_hwhm_curve_k']
    x_min = x_input[0]
    x_max = x_input[-1]
    func_legend = []
    params = Parameters()
    for func, legend in func_legend_from_params:
        k = float(legend.replace('k', '').replace('_', '').replace('dot', '.'))
        if x_min < k < x_max:
            func_legend.append((func, legend))
    for i in params_from_params:
        k = float(i.split('_')[0].replace('k', '').replace('dot', '.'))
        if x_min < k < x_max:
            params.add(params_from_params[i])
    fit_result = None
    prev_arg = []
    zero_under_05_hwhm_curve = {}
    n_peaks = len(func_legend)
    static_parameters = param_names, min_fwhm, params_limits
    while y_residual.max() > noise_level:
        n_peaks += 1
        # select max x, y in residual
        arg_max = np.argmax(y_residual)
        x_arg_max = x_input[arg_max]
        # print(arg_max, x_arg_max)
        if arg_max in prev_arg:
            # print('if arg_max in prev_arg', arg_max, prev_arg)
            arg_l, arg_r = zero_under_05_hwhm_curve[arg_max]
            y_residual[arg_l: arg_r] = 0.
            continue
        if legend_by_float(x_arg_max) in prev_k and legend_by_float(x_arg_max) in zero_under_05_hwhm_curve_k:
            arg_l, arg_r = zero_under_05_hwhm_curve_k[legend_by_float(x_arg_max)]
            arg_l = arg_max - arg_l
            arg_r = arg_max + arg_r
            y_residual[arg_l: arg_r] = 0.
            continue
        prev_arg.append(arg_max)
        y_at_arg_max = y_input[arg_max]
        # prepare model for n_peaks
        legend = legend_by_float(x_arg_max)
        func_legend.append((func, legend))
        init_params = init_model_params.copy()
        init_params[0] = y_at_arg_max
        init_params[1] = x_arg_max
        dx_left, dx_right, arg_left, arg_right, arg_left_4, arg_right_4 = find_dx_of_peak(n_array, arg_max)
        zero_under_05_hwhm_curve[arg_max] = (arg_left_4, arg_right_4)
        dx_left = min(dx_left, max_dx)
        dx_right = min(dx_right, max_dx)

        y_max_in_range = np.amax(y_input[arg_left:arg_right])
        dynamic_parameters = legend, init_params, y_max_in_range, dx_left, dx_right
        # fit
        params = update_fit_parameters(params, static_parameters, dynamic_parameters)
        model = fitting_model(func_legend)
        fit_result = fit_model(model, y_input, params, x_input, method)
        y_residual = y_input - fit_result.best_fit
        y_residual[y_residual < 0] = 0
        print('n_peaks ', n_peaks, ' np.max(y_residual) ', np.max(y_residual), ' x_arg_max ', x_arg_max)
        info('interval {!s}, n_peaks: {!s},'
             ' noise_level {!s}, max residual {!s}'.format((np.round(x_min, 1), np.round(x_max, 1)), n_peaks,
                                                           np.round(noise_level, 1), np.round(np.max(y_residual), 1)))
        if environ['CANCEL'] == '1':
            print('break CANCEL')
            break
    return fit_result


def update_fit_parameters(params: Parameters, static_parameters: tuple[list[str], float, dict],
                          dynamic_parameters: tuple[str, list, float, float, float]) -> Parameters:
    """
    update parameters for fit. Set inital value and bounds
    @param params: lmfit.Parameters() to update
    @param static_parameters: param_names, min_fwhm, params_limits
    @param dynamic_parameters: legend, init_params, y_max_in_range, dx_left, dx_right
    @return: updated Parameters
    """
    param_names, min_fwhm, params_limits = static_parameters
    legend, init_params, y_max_in_range, dx_left, dx_right = dynamic_parameters
    for i, param_name in enumerate(param_names):
        v = init_params[i]
        min_v = None
        max_v = None
        if param_name == 'a':
            min_v = 0
            max_v = y_max_in_range
        elif param_name == 'x0':
            dx0 = (min_fwhm / 4.0) + 1
            min_v = v - dx0
            max_v = v + dx0
        elif param_name == 'dx':
            min_v = min_fwhm / 2.0
            max_v = dx_right
        elif param_name == 'dx_left':
            min_v = min_fwhm / 2.0
            max_v = dx_left
        elif param_name in params_limits:
            min_v = params_limits[param_name][0]
            max_v = params_limits[param_name][1]
        if min_v is None or max_v is None:
            params.add(legend + param_name, v)
        else:
            if min_v == max_v:
                max_v += 0.001
            params.add(legend + param_name, v, min=min_v, max=max_v)

    return params


def find_dx_of_peak(n_array: np.ndarray, arg_max) -> tuple[float, float, int, int, int, int]:
    """
    @param n_array: 2D array x|y
    @param arg_max: index of peak
    @return: dx_left, dx_right, arg_left (HWHM), arg_right (HWHM), arg_left_4 (HWHM/2), arg_right_4 (HWHM/2)
    """
    x_input = n_array[:, 0]
    y_input = n_array[:, 1]
    x_arg_max = x_input[arg_max]
    y_arg_max = y_input[arg_max]
    arg_left_4 = arg_max
    y = y_input[arg_left_4]
    while y > y_arg_max * 0.75:
        arg_left_4 -= 1
        y = y_input[arg_left_4]
        if arg_left_4 <= 0:
            break
    arg_left = arg_left_4
    y = y_input[arg_left]
    while y > y_arg_max / 2:
        arg_left -= 1
        y = y_input[arg_left]
        if arg_left <= 0:
            arg_left = 0
            break

    arg_right_4 = arg_max
    y = y_input[arg_right_4]
    while y > y_arg_max * 0.75:
        arg_right_4 += 1
        if arg_right_4 >= y_input.shape[0]:
            arg_right_4 = y_input.shape[0] - 1
            break
        y = y_input[arg_right_4]

    arg_right = arg_right_4
    y = y_input[arg_right]
    while y > y_arg_max / 2:
        arg_right += 1
        if arg_right >= y_input.shape[0]:
            arg_right = y_input.shape[0] - 1
            break
        y = y_input[arg_right]

    if arg_max == 0:
        arg_left = 0
        arg_left_4 = 0
    if arg_max == y_input.shape[0]:
        arg_right = y_input.shape[0]
        arg_right_4 = y_input.shape[0]
    dx_left = x_arg_max - x_input[arg_left]
    dx_right = x_input[arg_right] - x_arg_max
    if arg_max == 0:
        dx_left = dx_right
    if arg_max == y_input.shape[0]:
        dx_right = dx_left
    return dx_left, dx_right, arg_left, arg_right, arg_left_4, arg_right_4


def split_by_borders(arr: np.ndarray, intervals: list[tuple[int, int]]) -> list[np.ndarray]:
    x_axis = arr[:, 0]
    y_axis = arr[:, 1]
    arrays_splitted = []
    for idx_start, idx_end in intervals:
        arr = np.vstack((x_axis[idx_start:idx_end], y_axis[idx_start:idx_end])).T
        arrays_splitted.append(arr)
    return arrays_splitted


def models_params_splitted(splitted_array: list[np.ndarray], params: Parameters,
                           idx_type_param_count_legend_func: list[tuple[int, str, int, str, callable]],
                           models: dict = None) \
        -> tuple[list[tuple[Any, Any, Model, Parameters]], dict[Any, Any] | None]:
    """
    Подготавливаем модель и параметры для fit_model. Делим все параметры на диапазоны, по splitted_arrays
    @param splitted_array: list[np.ndarray] 2D array. Список массивов, поделенных на интервалы
    @param params: Parameters. Все параметры
    @param idx_type_param_count_legend_func: list[tuple[int, str, int, str, callable]]. idx - индекс линии,
        type - тип линии, param_count - количество параметров, описывающих контур линии,
        legend - имя - префикс параметра, func - функция по которой считается контур линии
    @param models: при batch модели одинаковые и чтобы каждый раз не уходить в fitting_model испльзуются уже
        просчитанные модели
    @return: list[tuple[np.ndarray, np.ndarray, Model, Parameters]].
        x, y, Model, Parameters - для каждого интервала
    """
    result = []
    calculated_models = None
    if models is None:
        calculated_models = {}
    for arr2d in splitted_array:
        x_axis = arr2d[:, 0]
        y_axis = arr2d[:, 1]
        x_min = x_axis[0]
        x_max = x_axis[-1]
        func_legend = []
        indexes_of_this_interval = []
        interval_params = Parameters()
        for par in params:
            par_legend_splitted = par.split('_', 2)
            idx = int(par_legend_splitted[1])
            peak_param_name = par_legend_splitted[2]
            if peak_param_name == 'x0' and x_min < params[par].value < x_max:
                indexes_of_this_interval.append(idx)

        for par in params:
            par_legend_splitted = par.split('_', 2)
            idx = int(par_legend_splitted[1])
            if idx in indexes_of_this_interval:
                interval_params.add(params[par])
        if models is None:
            for idx, line_type, param_count, legend, func in idx_type_param_count_legend_func:
                if idx in indexes_of_this_interval:
                    func_legend.append((func, legend))
            model = fitting_model(func_legend)
            calculated_models[int(x_axis[0])] = model
        else:
            model = models[int(x_axis[0])]
        result.append((x_axis, y_axis, model, interval_params))
    return result, calculated_models


def models_params_splitted_batch(splitted_arrays: dict, list_params_full: dict,
                                 idx_type_param_count_legend_func: list[tuple[int, str, int, str, callable]]) -> dict:
    x_y_models_params = {}
    calculated_models = None
    for key, arrays in splitted_arrays.items():
        res, calculated_models = models_params_splitted(arrays, list_params_full[key], idx_type_param_count_legend_func,
                                                        calculated_models)
        x_y_models_params[key] = res
    return x_y_models_params


def find_interval_key(x0: float, data_by_intervals: dict) -> str | None:
    result_key = None
    for key, item in data_by_intervals.items():
        start, end = item['interval']
        if start <= x0 < end:
            result_key = key
            break
    return result_key


def process_data_by_intervals(data_by_intervals: dict, method: str) -> list[tuple]:
    """
    Определение итогового состава линий для каждого интервала. Результат функции дальше нужен для определения
     параметров и модели
    @param data_by_intervals: dict[] параметр 'x0' содержит список волновых чисел интервала
    @param method: метод определения итогового числа линий интервала
    @return: list[tuple], итоговый список x0 и sd - отклонение от центроида
    """
    key_clustered_x0 = []
    for key, item in data_by_intervals.items():
        x0_lines = item['x0']
        lines_count = item['lines_count']
        if method == 'Min':
            k = np.min(lines_count)
        elif method == 'Max':
            k = np.max(lines_count)
        elif method == 'Mean':
            k = np.mean(lines_count)
        elif method == 'Median':
            k = np.median(lines_count)
        else:
            k = mode(lines_count).mode[0]
        k = np.round(k)
        clustered = kmeans(x0_lines, int(k), iter=100)
        key_clustered_x0.append(clustered)
    return key_clustered_x0


def legend_by_float(x0: float) -> str:
    """
    @param x0: float. Ex: 575.31
    @return: str. Ex: k575dot31_x0
    """
    return 'k%s_' % str(np.round(x0, 2)).replace('.', 'dot')


def process_wavenumbers_interval(wavenumbers: list[float], n_array: np.ndarray, param_names: list[str], sd: float,
                                 static_params: tuple[list[float], float, float, callable, dict]) \
        -> tuple[Parameters, list[tuple[callable, str]]]:
    """
    Используется для создания параметров и модели после анализа линий найденных автоматически из разных спектров
    @param wavenumbers: list[float] - массив волновых чисел после k-means
    @param n_array: 2D ndarray x|y определенного интервала
    @param param_names: ['a', 'x0', 'dx', ...]
    @param sd: float - разброс от центра x0, используется для определения левой и правой границы x0
    @param static_params: init_params, max_dx, min_hwhm, func, self.peak_shape_params_limits
    @return: tuple[Parameters, list[tuple[callable, str]]] - Параметры и (функция, префикс) для модели
    """
    func_legend = []
    params = Parameters()
    x_axis = n_array[:, 0]
    y_axis = n_array[:, 1]
    init_params, max_dx, min_hwhm, func, peak_shape_params_limits = static_params
    wavenumbers = np.unique(wavenumbers)
    debug('process_wavenumbers_interval wavenumbers: {!s}'.format(wavenumbers))
    for x0 in wavenumbers:
        legend = legend_by_float(x0)
        func_legend.append((func, legend))
        arg_max = find_nearest_idx(x_axis, x0)
        dx_left, dx_right, arg_left, arg_right, _, _ = find_dx_of_peak(n_array, arg_max)
        a = np.amax(y_axis[arg_left:arg_right])
        for j, param_name in enumerate(param_names):
            if param_name == 'a':
                v = max_v = a
                min_v = 0.
            elif param_name == 'x0':
                v = x0
                min_v = x0 - sd
                max_v = x0 + sd
            elif param_name == 'dx':
                v = min(dx_right, max_dx)
                min_v = min_hwhm
                max_v = max_dx
            elif param_name == 'dx_left':
                v = min(dx_left, max_dx)
                min_v = min_hwhm
                max_v = max_dx
            else:
                v = init_params[param_names[j]]
                min_v = peak_shape_params_limits[param_name][0]
                max_v = peak_shape_params_limits[param_name][1]
            params.add(legend + param_name, v, min=min_v, max=max_v)
    return params, func_legend


def eval_uncert(item: tuple[str, lmfit.model.ModelResult]) -> tuple[str, np.ndarray]:
    key, fit_result = item
    return key, fit_result.eval_uncertainty(fit_result.params, 3)
# endregion


# region Fitting

def insert_table_to_text_edit(cursor, headers, rows) -> None:

    cursor.insertTable(len(rows) + 1, len(headers))
    for header in headers:
        cursor.insertText(header)
        cursor.movePosition(QTextCursor.NextCell)
    for row in rows:
        for value in row:
            cursor.insertText(str(value))
            cursor.movePosition(QTextCursor.NextCell)

# endregion


def calculate_vips(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)
    return vips

