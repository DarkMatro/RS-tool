from typing import Any

import lmfit.model
import numpy as np
import pandas as pd
from lmfit import Model, Parameters, CompositeModel
from lmfit.model import ModelResult
from logging import error, critical


def fitting_model(func_legend: list[tuple]) -> Model:
    """
    Prepare fitting model. Final model is a sum of different line models.

    Parameters
    ---------
    func_legend : list[tuple[func, str]]
        using to create fit Model. func is a line function from spec_functions/ peak_shapes, str is curve legend

    Returns
    -------
    out :
        Model()
    """
    model = None
    for i, f_l in enumerate(func_legend):
        if i == 0:
            model = Model(f_l[0], prefix=f_l[1])
            continue
        model += Model(f_l[0], prefix=f_l[1])
    return model


def split_by_borders(arr: np.ndarray, intervals: list[tuple[int, int]]) -> list[np.ndarray]:
    """
    The function returns sliced array by given interval range.

    Parameters
    ----------
    arr: np.ndarray
        input array to slice

    intervals: list[tuple[int, int]]
        ranges (start, end) indexes

    Returns
    -------
    out : list[np.ndarray]
        arr sliced

    Examples
    --------
    >>> data = [[0., 1.], [1., 1.1], [2., 1.2], [3., 1.3], [4., 1.4], [5., 1.5], [6., 1.4], [7., 1.3], [8., 1.2]]
    >>> arr = np.array(data)
    >>> intervals = [(0, 2), (2, 5), (5, 8)]
    >>> split_by_borders(arr, intervals)
    [array([[0. , 1. ],
           [1. , 1.1]]), array([[2. , 1.2],
           [3. , 1.3],
           [4. , 1.4]]), array([[5. , 1.5],
           [6. , 1.4],
           [7. , 1.3]])]
    """
    x_axis = arr[:, 0]
    y_axis = arr[:, 1]
    arrays_sliced = []
    for idx_start, idx_end in intervals:
        arr = np.vstack((x_axis[idx_start:idx_end], y_axis[idx_start:idx_end])).T
        arrays_sliced.append(arr)
    return arrays_sliced


def fit_model(item: tuple[np.ndarray, np.ndarray, Model | str, Parameters], method: str) \
        -> ModelResult | None:
    """
    Fit model using lmfit

    Parameters
    ---------
    item
        model : Model
            lmfit Model
        y : np.ndarray
             Array of data to be fit.
        params : Parameters,
            Parameters to use in fit (default is None).
        x: np.ndarray
             Array of x_axis.
    method : str
        Name of fitting method to use (default is `'leastsq'`).

    Returns
    -------
    ModelResult
    """
    x, y, model, params = item
    if model is None:
        return
    result = model.fit(y, params, x=x, method=method, n_jobs=-1, nan_policy='omit')
    return result


def load_model_result(res_json: str) -> ModelResult:
    params = Parameters()
    mod_res = ModelResult(Model(lambda x: x, None), params)
    mod_res.loads(res_json)
    return mod_res


def load_model(mod_json: str) -> Model:
    mod = Model(lambda x: x, None)
    mod.loads(mod_json)
    return mod


def fit_model_batch(item: tuple[str, np.ndarray, np.ndarray, Model, dict], method: str) \
        -> tuple[str, ModelResult]:
    key, x, y, model, params = item
    try:
        result = model.fit(y, params, x=x, method=method, n_jobs=-1)
    except ValueError as ex:
        error(f'Empty parameter for filename:{key}', exc_info=ex)
        result = None
    return key, result


def curve_idx_from_par_name(curve_par_name: str) -> tuple[int, str]:
    """
    Decompose param_name str like 'Curve_10_x0' into (10, 'x0').

    Parameters
    ---------
    curve_par_name : str
        string with curve idx and param_name

    Returns
    -------
    tuple[int, str]

    Examples
    --------
    >>> curve_idx_from_par_name('Curve_10_x0')
    (10, 'x0')
    """
    idx_param_name = curve_par_name.split('_', 2)
    idx = int(idx_param_name[1])
    param_name = idx_param_name[2].strip()
    return idx, param_name


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
            result[f'min_{param_name}'] = query_result_add_param['Min value']
            result[f'max_{param_name}'] = query_result_add_param['Max value']
    return result


def models_params_splitted(splitted_array: list[np.ndarray], params: Parameters,
                           static_params: list[
                               tuple[int, str, int, str, callable]],
                           models: dict = None) \
        -> tuple[list[tuple[Any, Any, Model, Parameters]], dict[Any, Any] | None]:
    """
    Подготавливаем модель и параметры для fit_model. Делим все параметры на диапазоны,
    по splitted_arrays
    @param splitted_array: list[np.ndarray] 2D array. Список массивов, поделенных на интервалы
    @param params: Parameters. Все параметры
    @param static_params: list[tuple[int, str, int, str, callable]]. idx - индекс линии,
        type - тип линии, param_count - количество параметров, описывающих контур линии,
        legend - имя - префикс параметра, func - функция по которой считается контур линии
    @param models: при batch модели одинаковые и чтобы каждый раз не уходить в fitting_model
    испльзуются уже
        просчитанные модели
    @return: list[tuple[np.ndarray, np.ndarray, Model, Parameters]].
        x, y, Model, Parameters - для каждого интервала
    """
    result = []
    calculated_models = None
    if models is None:
        calculated_models = {}
    for arr in splitted_array:
        idx_of_the_interval = []
        interval_params = Parameters()
        for p in params:
            par_legend_splt = p.split('_', 2)
            if par_legend_splt[2] == 'x0' and arr[:, 0][0] <= params[p].value <= arr[:, 0][-1]:
                idx_of_the_interval.append(int(par_legend_splt[1]))

        for p in params:
            par_legend_splt = p.split('_', 2)
            if int(par_legend_splt[1]) in idx_of_the_interval:
                interval_params.add(params[p])
        if models is None:
            func_legend = [(func, legend) for idx, _, _, legend, func in static_params
                           if idx in idx_of_the_interval]
            model = fitting_model(func_legend)
            calculated_models[int(arr[:, 0][0])] = fitting_model(func_legend)
        else:
            model = models[int(arr[:, 0][0])]
        result.append((arr[:, 0], arr[:, 1], model, interval_params))
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
