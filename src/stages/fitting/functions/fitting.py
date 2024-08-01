# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for fitting spectral data using various models and parameters.

This module provides functions to prepare fitting models, fit the models to spectral data,
and handle parameters for the fitting process. The main purpose is to iteratively select
and fit Raman lines in spectral data using lmfit.

Functions:
- fitting_model: Prepare a fitting model by summing different line models.
- split_by_borders: Slice an array by given interval ranges.
- fit_model: Fit a model to the data using lmfit.
- load_model_result: Load a model result from a JSON string.
- load_model: Load a model from a JSON string.
- fit_model_batch: Fit models to batches of data.
- curve_idx_from_par_name: Decompose parameter name to extract curve index and parameter name.
- packed_current_line_parameters: Extract and pack current line parameters from a DataFrame.
- models_params_splitted: Prepare models and parameters for fitting by intervals.
- models_params_splitted_batch: Prepare models and parameters for fitting in batch mode.
- eval_uncert: Evaluate the uncertainty of the fitted model.
"""

from logging import error
from typing import Any

import lmfit.model
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from lmfit.model import ModelResult


def fitting_model(func_legend: list[tuple]) -> Model:
    """
    Prepare fitting model. Final model is a sum of different line models.

    Parameters
    ----------
    func_legend : list[tuple[callable, str]]
        List of tuples containing functions and their legends to create the fit model.

    Returns
    -------
    Model
        The combined fitting model.
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
    Slice array by given interval range.

    Parameters
    ----------
    arr : np.ndarray
        Input array to slice.
    intervals : list[tuple[int, int]]
        List of ranges (start, end) indexes.

    Returns
    -------
    list[np.ndarray]
        List of sliced arrays.

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
    Fit model using lmfit.

    Parameters
    ----------
    item : tuple
        containing x array, y array, model, and parameters.
    method : str
        Name of the fitting method to use (default is `'leastsq'`).

    Returns
    -------
    ModelResult or None
        The result of the fit.
    """
    x, y, model, params = item
    if model is None:
        return None
    result = model.fit(y, params, x=x, method=method, nan_policy='omit')
    return result


def load_model_result(res_json: str) -> ModelResult:
    """
    Load a model result from a JSON string.

    Parameters
    ----------
    res_json : str
        JSON string representing the model result.

    Returns
    -------
    ModelResult
        The loaded model result.
    """
    params = Parameters()
    mod_res = ModelResult(Model(lambda x: x, None), params)
    mod_res.loads(res_json)
    return mod_res


def load_model(mod_json: str) -> Model:
    """
    Load a model from a JSON string.

    Parameters
    ----------
    mod_json : str
        JSON string representing the model.

    Returns
    -------
    Model
        The loaded model.
    """
    mod = Model(lambda x: x, None)
    mod.loads(mod_json)
    return mod


def fit_model_batch(item: tuple[str, np.ndarray, np.ndarray, Model, dict], method: str) \
        -> tuple[str, ModelResult]:
    """
    Fit models to batches of data.

    Parameters
    ----------
    item : tuple
        containing key, x array, y array, model, and parameters.
    method : str
        Name of the fitting method to use.

    Returns
    -------
    tuple[str, ModelResult]
        The key and the result of the fit.
    """
    key, x, y, model, params = item
    try:
        result = model.fit(y, params, x=x, method=method)
    except ValueError as ex:
        error(f'Empty parameter for filename:{key}', exc_info=ex)
        print(f'Empty parameter for filename:{key}')
        result = None
    return key, result


def curve_idx_from_par_name(curve_par_name: str) -> tuple[int, str]:
    """
    Decompose parameter name to extract curve index and parameter name.

    Parameters
    ----------
    curve_par_name : str
        String with curve index and parameter name.

    Returns
    -------
    tuple[int, str]
        Tuple containing curve index and parameter name.

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
    """
    Extract and pack current line parameters from a DataFrame.

    Parameters
    ----------
    df_params : pd.DataFrame
        containing parameters.
    line_type : str
        Type of the line.
    peak_shapes_params : dict
        Dictionary containing additional peak shape parameters.

    Returns
    -------
    dict
        Dictionary of packed parameters.
    """
    query_result_dx = df_params.loc['dx']
    query_result_a = df_params.loc['a']
    query_result_x0 = df_params.loc['x0']
    result = {'a': query_result_a['Value'], 'min_a': query_result_a['Min value'],
              'max_a': query_result_a['Max value'],
              'x0': query_result_x0['Value'], 'min_x0': query_result_x0['Min value'],
              'max_x0': query_result_x0['Max value'],
              'dx': query_result_dx['Value'], 'min_dx': query_result_dx['Min value'],
              'max_dx': query_result_dx['Max value']}
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
    Prepare models and parameters for fitting by intervals.

    Parameters
    ----------
    splitted_array : list[np.ndarray]
        List of arrays, each representing a different interval.
    params : Parameters
        All parameters for fitting.
    static_params : list[tuple[int, str, int, str, callable]]
        List of tuples containing index, type, parameter count, legend, and function for each line.
    models : dict, optional
        Dictionary of pre-calculated models to avoid recalculating.

    Returns
    -------
    tuple
        List of tuples containing x, y, model, and parameters for each interval, and
        dictionary of calculated models.
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
    """
    Prepare models and parameters for fitting in batch mode.

    Parameters
    ----------
    splitted_arrays : dict
        Dictionary of arrays, each representing a different interval.
    list_params_full : dict
        Dictionary of all parameters for fitting.
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
        List of tuples containing index, type, parameter count, legend, and function for each line.

    Returns
    -------
    dict
        Dictionary containing tuples of x, y, model, and parameters for each interval.
    """
    x_y_models_params = {}
    calculated_models = None
    for key, arrays in splitted_arrays.items():
        res, calculated_models = models_params_splitted(arrays, list_params_full[key],
                                                        idx_type_param_count_legend_func,
                                                        calculated_models)
        x_y_models_params[key] = res
    return x_y_models_params


def eval_uncert(item: tuple[str, lmfit.model.ModelResult]) -> tuple[str, np.ndarray] | None:
    """
    Evaluate the uncertainty of the fitted model.

    Parameters
    ----------
    item : tuple
        containing the key and the fitted model result.

    Returns
    -------
    tuple[str, np.ndarray] or None
        The key and the evaluated uncertainty, or None if the result is not available.
    """
    key, fit_result = item
    if not fit_result:
        return
    return key, fit_result.eval_uncertainty(fit_result.params, sigma=3)
