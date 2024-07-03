from logging import error
from os import getpid
from typing import Any

import lmfit.model
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from psutil import Process
from pyperclip import copy as pyperclip_copy
from pyqtgraph import mkPen, PlotCurveItem, ROI, mkBrush
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QColor, QTextCursor

from sklearn.metrics import mean_squared_log_error

from qfluentwidgets import MessageBox
from src.stages.fitting.functions.fitting import fitting_model
from ..data.plotting import random_rgb


# region RS


def invert_color(color: str) -> str:
    rgb = QColor(color).getRgb()
    new_r = 255 - rgb[0]
    new_g = 255 - rgb[1]
    new_b = 255 - rgb[2]
    return QColor(new_r, new_g, new_b, rgb[3]).name()


def get_memory_used() -> float:
    return Process(getpid()).memory_info().rss / 1024 ** 2


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


# endregion

# region Fitting plot

def set_roi_size_pos(params: tuple[float, float, float], roi: ROI, update: bool = True) -> None:
    a, x0, dx = params
    set_roi_size(dx, a, roi, update=update)
    set_roi_pos(x0, roi)


def set_roi_size(x: float, y: float, roi: ROI, center=None, finish: bool = False,
                 update: bool = True) -> None:
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


def packed_current_line_parameters(df_params: pd.DataFrame, line_type: str,
                                   peak_shapes_params: dict) -> dict:
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

def models_params_splitted(splitted_array: list[np.ndarray], params: Parameters,
                           idx_type_param_count_legend_func: list[
                               tuple[int, str, int, str, callable]],
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
            if peak_param_name == 'x0' and x_min <= params[par].value <= x_max:
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
                                 idx_type_param_count_legend_func: list[
                                     tuple[int, str, int, str, callable]]) -> dict:
    x_y_models_params = {}
    calculated_models = None
    for key, arrays in splitted_arrays.items():
        res, calculated_models = models_params_splitted(arrays, list_params_full[key],
                                                        idx_type_param_count_legend_func,
                                                        calculated_models)
        x_y_models_params[key] = res
    return x_y_models_params


def eval_uncert(item: tuple[str, lmfit.model.ModelResult]) -> tuple[str, np.ndarray] | None:
    key, fit_result = item
    if not fit_result:
        return
    return key, fit_result.eval_uncertainty(fit_result.params, sigma=3)


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


def show_error_msg(exc_type, exc_value, exc_tb, parent=None):
    msg = MessageBox(str(exc_type), str(exc_value), parent, {'Ok'})
    msg.setInformativeText('For full text of error go to %appdata%/RS-Tool/log.log' + '\n' + exc_tb)
    print(exc_tb)
    error(exc_tb)
    pyperclip_copy(exc_tb)
    msg.exec()


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    The Root Mean Squared Log Error (RMSLE) metric
    Логаритмическая ошибка средней квадратичной ошибки
    """
    try:
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    except:
        return None
