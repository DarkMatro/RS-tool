import copy
from asyncqtpy import asyncSlot
from asyncio import gather, get_event_loop
from datetime import datetime
from logging import info, debug, warning
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import mpl_axes_aligner
import numpy as np
from lmfit import Parameters
from lmfit.model import ModelResult
from numpy import ndarray
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

from modules.stages.preprocessing.functions.cut_trim.cut_trim import cut_full_spectrum
from modules.stages.fitting.functions.fitting import fitting_model, fit_model
from modules.mutual_functions.work_with_arrays import find_nearest, nearest_idx


def guess_peaks(n_array: np.ndarray, input_parameters: dict, break_event_by_user, verbose: int = 1) \
        -> ModelResult:
    """
    Part of the algorithm for iterative selection of Raman lines in the n_array spectral contour.
    Here we collect possible variations in the number of lines and their positions by fitting them across all spectra.
    In advance, the number of lines, and the parameters of the lines are unknown.
    First you need to decide on the number of lines and their maximum positions on the X axis.
    This function is run for M wavenumber intervals for N spectra.

    Lines are added one at a time.
    The position of the new line corresponds to the position of the maximum in y_residual.
                                    y_residual = y_input - fit_result.best_fit
    A number of restrictions are imposed on the line parameters.
    Amplitude 'a' is not greater than the maximum value in the interval x0±HWHM and not less than 0.
    'dx' aka HWHM is limited from above by the user's choice 'max_dx',
        from below by the minimum possible HWHM of the Rayleigh line by 0 cm-1
    The fluctuation of 'x0' values is calculated by the formula: Δx0 = (min_fwhm / 4.) + 1.
    These restrictions do not allow the line to get out of the boundaries of the original spectrum n_array.
    After each iteration, the value of the maximum residual is checked.
    If it is less than the noise_level threshold value, then we end the search for lines, because the following
        lines will be considered noise. If the threshold has not yet been reached, then add another line, and so on.

    If initially there are lines set by the user, then the process starts not from scratch, but from the already
     existing lines. The loop ends when the newly added line has an amplitude less than noise_level.

     At the output, we have a complex model consisting of the sum of models describing each line.
     Accordingly, this model has information about the position of each line and their number.

    Parameters
    ---------
    n_array : np.ndarray
        2D array x|y of current spectral shape.
    input_parameters : dict
        with keys:
            'func': callable; Function for peak shape calculation. Look peak_shapes_params() in default_values.py.
            'param_names': list[str]; List of parameter names. Example: ['a', 'x0', 'dx'].
            'init_model_params': list[float]; Initial values of parameters for a given spectrum and line type.
            'min_fwhm': float; the minimum value FWHM, determined from the input table (the minimum of all).
            'method': str; Optimization method, {'Levenberg-Marquardt', 'Least-Squares, Trust Region Reflective method',
                'Differential evolution', 'Basin-hopping', 'Adaptive Memory Programming for Global Optimization',
                'Nelder-Mead', 'L-BFGS-B', 'Powell', 'Conjugate-Gradient', 'BFGS', 'Truncated Newton',
                'trust-region for constrained optimization', 'Sequential Linear Squares Programming',
                 'Maximum likelihood via Monte-Carlo Markov Chain', 'Dual Annealing optimization'}
            'params_limits': dict[str, tuple[float, float]]; see peak_shape_params_limits() in default_values.py.
            'noise_level': float; limit for peak detection.
                Peaks with amplitude less than noise_level will not be detected. noise_level = y_max / SNR
            'max_dx': float; Maximal possible value for dx. For all peaks.
        # The following parameters are empty if there are no existing lines.
        # If at the beginning of the analysis there are lines already created by the user,
        # then the parameters will be filled.

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
    break_event_by_user : threading.Event.
        breaks all executing tasks on all threads
    verbose: int
        1 - show messages.

    Returns
    -------
    ModelResult
        fitted peaks to this spectral shape n_array

    Examples
    -------
    >>> x_input = [900.79160552, 902.55742054, 904.3227092, 906.08747174, 907.85184857, 909.61555954, 911.3787451, 913.14140548, 914.90354092, 916.6652916, 918.42637781, 920.18693979, 921.94697775, 923.70649194, 925.46562235, 927.22408965, 928.98203389, 930.73945529, 932.49649366, 934.25287004, 936.00872428, 937.76405661, 939.51886727, 941.27329587, 943.02706383, 944.78031083, 946.53303708, 948.28524281, 950.03706743, 951.7882328, 953.53887835, 955.28900431, 957.03861092, 958.78783736, 960.53640592, 962.28445581, 964.03198726, 965.77913932, 967.52563457, 969.27161208, 971.01707208, 972.7620148, 974.50657906, 976.25048787, 977.99388009, 979.73675595, 981.47911568, 983.2210979, 984.96242601]
    >>> y_input = [19.03885234, 20.05219638, 22.18841171, 25.73975861, 31.05152628, 37.85842055, 45.76006094, 53.61874806, 60.43457353, 66.01767101, 70.57862791, 74.44021118, 78.63591054, 84.45305653, 93.33817007, 105.87340249, 122.35738568, 143.0156349, 168.05370637, 198.07624588, 233.70881506, 276.74761159, 327.61511136, 387.54749906, 460.03738257, 545.0511741, 643.26105361, 751.67484931, 869.18685947, 992.00284964, 1112.0830707, 1213.13555555, 1264.87864212, 1230.02046493, 1115.65062841, 952.07515744, 775.35377045, 613.33409198, 478.24536214, 366.94443183, 274.32571713, 198.4521729, 139.85282567, 94.1191349, 62.4133825, 39.73972405, 24.93694253, 16.17885449, 11.73133146]
    >>> n_array = np.vstack((x_input, y_input)).T
    >>> from modules import gaussian
    >>> params_limits = {'gamma': (-100.0, 100.0), 'skew': (-100.0, 100.0), 'l_ratio': (0.0, 0.25), 'expon': (0.1, 100.0), 'beta': (0.1, 100.0), 'alpha': (-1.0, 1.0), 'q': (-0.5, 0.5)}
    >>> param_names = ['a', 'x0', 'dx']
    >>> init_model_params = [632.43932, 1177.92227, 10.6696]
    >>> input_parameters = {'func': gaussian, 'param_names': param_names, 'init_model_params': init_model_params, 'min_fwhm': 6.34427, 'method': 'least_squares', 'params_limits': params_limits, 'noise_level': 12.78, 'max_dx': 12.0, 'func_legend': [], 'params': Parameters(), 'used_legends': [], 'used_legends_dead_zone': {}}
    >>> from multiprocessing import Manager
    >>> break_event_by_user = Manager().Event()
    >>> guess_peaks(n_array, input_parameters, break_event_by_user, 0)
    """
    x_input = n_array[:, 0]
    y_input = n_array[:, 1]
    y_input[y_input < 0.] = 0.
    y_residual = y_input.copy()
    input_parameters = copy.deepcopy(input_parameters)
    func = input_parameters['func']
    init_model_params = input_parameters['init_model_params']
    method = input_parameters['method']
    noise_level = input_parameters['noise_level']
    max_dx = input_parameters['max_dx']
    static_parameters = input_parameters['param_names'], input_parameters['min_fwhm'],\
        input_parameters['params_limits']

    func_legend, params = func_legend_params(input_parameters['func_legend'], input_parameters['params'], x_input[0],
                                             x_input[-1])
    used_legends = input_parameters['used_legends']
    used_legends_dead_zone = input_parameters['used_legends_dead_zone']  # dict with key as str like 'k977dot15_'

    fit_result = None
    used_idx_x = []  # Keep indexes of already fitted peaks to prevent infinite loop.
    used_idx_dead_zone = {}  # Same as used_legends_dead_zone, but key is float like 977.15
    n_peaks = len(func_legend)

    time_start = datetime.now()                                                 # Executing timelimit is 5 min here. lol
    while y_residual.max() > noise_level and not break_event_by_user.is_set() \
            and (datetime.now() - time_start).total_seconds() / 60. < 3.:
        n_peaks += 1
        x_max_idx = np.argmax(y_residual)                                          # idx of maximal value in y_residual
        x_max = x_input[x_max_idx]                                                 # x value of maximal y_residual
        if this_idx_already_was(x_max_idx, used_idx_x, used_idx_dead_zone, y_residual):
            continue
        legend = legend_from_float(x_max)
        if this_legend_already_was(legend, x_max_idx, y_residual, used_legends, used_legends_dead_zone):
            continue
        used_idx_x.append(x_max_idx)                    # Save idx for duplicates control.
        # Prepare model and parameters including new peak for new fitting iteration.
        func_legend.append((func, legend))
        init_params = init_model_params.copy()
        init_params[0] = y_input[x_max_idx]             # Max y of new peak as starting parameter value for 'a'.
        init_params[1] = x_max                          # x position of new peak as starting parameter value for 'x0'.
        dx_left, dx_right, arg_left, arg_right, arg_left_qr, arg_right_qr = possible_peak_dx(n_array, x_max_idx)
        used_idx_dead_zone[x_max_idx] = (arg_left_qr, arg_right_qr)        # Update dead zone for used peaks idx.
        dx_left = min(dx_left, max_dx)                                     # Starting dx_left and dx_right for new peak.
        dx_right = min(dx_right, max_dx)
        y_max_in_range = np.amax(y_input[arg_left:arg_right])                   # Possible max limit for amplitude
        dynamic_parameters = legend, init_params, y_max_in_range, dx_left, dx_right
        params = update_fit_parameters(params, static_parameters, dynamic_parameters)
        model = fitting_model(func_legend)
        # fit model and estimate y_residual.
        fit_result = fit_model(model, y_input, params, x_input, method)
        y_residual = y_input - fit_result.best_fit
        y_residual[y_residual < 0.] = 0.                                        # we don't want see negative amplitude.
        if verbose:
            debug_msg = 'n_peaks: {!s}, max_amp {!s},' \
                        ' max x_arg_max {!s}, calc time {!s}'.format(n_peaks, np.round(np.max(y_residual), 2),
                                                                     np.round(x_max, 2), (datetime.now() - time_start))
            print(debug_msg)
            info(debug_msg)
    return fit_result


def func_legend_params(func_legend_from_initial_params, params_from_initial_params, x_min, x_max) \
        -> tuple[list[tuple[callable, str]], Parameters]:
    """
    func_legend_from_initial_params and params_from_initial_params includes data for all cm-1 ranges.
    we take only values for current interval.

    Parameters
    ---------
    func_legend_from_initial_params : list[tuple]; - (callable func, legend),
                func - callable; Function for peak shape calculation. Look peak_shapes_params() in default_values.py.
                legend - prefix for the line in the model. All lines in a heap. As a result,
                    we select only those that belong to the current interval.
    params_from_initial_params : Parameters();
        parameters of existing lines.
    x_min : float;
        x_axis[0]
    x_max : float;
        x_axis[-1]

    Returns
    -------
    func_legend : list[tuple[callable, str]]
    params : Parameters
    """
    func_legend = []
    params = Parameters()

    if len(func_legend_from_initial_params) == 0:
        return func_legend, params

    for func, legend in func_legend_from_initial_params:
        k = float(legend.replace('k', '').replace('_', '').replace('dot', '.'))
        if x_min < k < x_max:
            func_legend.append((func, legend))
    for i in params_from_initial_params:
        k = float(i.split('_')[0].replace('k', '').replace('dot', '.'))
        if x_min < k < x_max:
            params.add(params_from_initial_params[i])

    return func_legend, params


def this_idx_already_was(x_max_idx: np.ndarray[int], used_idx_x: list[np.ndarray[int]], used_idx_dead_zone: dict,
                         y_residual: np.ndarray) -> bool:
    """
    Model cant have lines with duplicate legends (same x0 position lines). So because we have to control peaks indexes
    for duplicates. If new maximum index already fitted we set y_residual at this idx to 0. with near elements dead zone
    Size of dead zone given by used_idx_dead_zone.

    Parameters
    ---------
    x_max_idx : int,
        index of y_residual maximum.
    used_idx_x : list[int];
        indexes of peaks already fitted
    used_idx_dead_zone : dict
        keys - used_legends, values - tuple (idx - left idx, right idx - idx).
        dead zone size to set y_residual to 0.
        {'k977dot15_': (1, 1), 'k959dot68_': (2, 1), 'k917dot49_': (3, 4)}
    y_residual : np.ndarray;
        1D. residual = y_input - the latest fitted sum of peaks

    Returns
    -------
    bool
    """
    if x_max_idx in used_idx_x:
        arg_l, arg_r = used_idx_dead_zone[x_max_idx]
        y_residual[arg_l: arg_r] = 0.
        return True
    else:
        return False


def this_legend_already_was(legend, x_max_idx, y_residual, used_legends, used_legends_dead_zone) -> bool:
    """
    Model cant have lines with duplicate legends (same x0 position lines). So because we have to control peaks legends
    for duplicates. If new legend already was used we set y_residual at this peak idx to 0. with near elements dead zone
    Size of dead zone given by used_legends_dead_zone.

    Parameters
    ---------
    legend : str,
        prefix for fitting model of peak like 'k977dot15_'
    x_max_idx : int,
        index of y_residual maximum.
    y_residual : np.ndarray;
        1D. residual = y_input - the latest fitted sum of peaks
    used_legends: list[str]; already used legends (used wave-numbers)
        ['k977dot15_', 'k959dot68_', 'k917dot49_']. We control it because model cant have lines with duplicate
         legends (same x0 position lines)
    used_legends_dead_zone: dict; keys - used_legends, values - tuple (idx - left idx, right idx - idx).
        dead zone size to set y_residual to 0.
        {'k977dot15_': (1, 1), 'k959dot68_': (2, 1), 'k917dot49_': (3, 4)}


    Returns
    -------
    bool
    """
    if legend in used_legends and legend in used_legends_dead_zone:
        arg_l, arg_r = used_legends_dead_zone[legend]
        arg_l = x_max_idx - arg_l
        arg_r = x_max_idx + arg_r
        y_residual[arg_l: arg_r] = 0.
        return True
    else:
        return False


def intervals_by_borders(borders: list[float], x_axis: np.ndarray, idx: bool = False) \
        -> list[tuple[float | int, float | int]]:
    """
    Values of intervals in x_axis. Borders given from BORDERS table set by user.

    Parameters
    ----------
    borders: list[float]
        borders of cm-1 ranges
    x_axis: np.ndarray
        common x_axis for all spectra given from any of baseline_corrected_dict
    idx: bool
        If true: return indexes, False - return values.

    Returns
    -------
    out: list[tuple[float, float]]

    Examples of output
    -------
    >>> x_axis = np.arange(11., 23.)
    >>> x_axis
    array([11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.])
    >>> borders = [14., 18.]
    >>> intervals_by_borders(borders, x_axis)
    [(0, 14.0), (14.0, 18.0), (18.0, 22.0)]
    >>> intervals_by_borders(borders, x_axis, idx=True)
    [(0, 3), (3, 7), (7, 11)]
    """
    v_in_range = []
    for i in borders:
        if x_axis[0] < i < x_axis[-1]:
            v = nearest_idx(x_axis, i) if idx else find_nearest(x_axis, i)
            v_in_range.append(v)
    res = [(0, v_in_range[0])]
    for i in range(len(v_in_range) - 1):
        res.append((v_in_range[i], v_in_range[i + 1]))
    last_el = x_axis.shape[0] - 1 if idx else x_axis[-1]
    res.append((v_in_range[-1], last_el))
    return res


def create_intervals_data(model_result: list[ModelResult], peak_n_params: int, intervals: list[tuple[float, float]]) \
        -> dict[str, dict[str, tuple[float, float] | list]]:
    """
    The function creates a structure with 'interval': (start cm-1, end cm-1), x0': list of x0 lines, lines_count': [].
    The result of the function is then passed to clustering_lines_intervals()

    Parameters
    ----------
    model_result: list[ModelResult]
        results of fitting for every spectrum and every interval range.

    peak_n_params: int
        number of parameters of peak shape. Look peak_shapes_params() in default_values.py

    intervals: list[tuple[float, float]]
        Values of intervals in x_axis. Borders given from BORDERS table set by user.
        Result of intervals_by_borders_values()

    Returns
    -------
    data_by_intervals: dict[dict]
        with keys 'interval': (start, end), 'x0': list, lines_count': []
    """
    data_by_intervals = {}
    for start, end in intervals:
        key = str(round(start)) + '_' + str(round(end))
        data_by_intervals[key] = {'interval': (start, end), 'x0': [], 'lines_count': []}
    for fit_result in model_result:
        if not fit_result:
            warning('None fit_result')
            continue
        parameters = fit_result.params
        lines_count = int(len(parameters) / peak_n_params)
        interval_key = None
        for j, par in enumerate(parameters):
            str_split = par.split('_', 1)
            param_name = str_split[1]
            if j == len(parameters) - 1 and interval_key is not None:
                data_by_intervals[interval_key]['lines_count'].append(lines_count)
            if param_name != 'x0':
                continue
            x0 = parameters[par].value
            interval_key = find_interval_key(x0, data_by_intervals)
            if interval_key is None:
                continue
            data_by_intervals[interval_key]['x0'].append(x0)
    return data_by_intervals


def find_interval_key(x0: float, data_by_intervals: dict) -> str | None:
    """
    The function creates a structure with 'interval': (start cm-1, end cm-1), x0': list of x0 lines, lines_count': [].
    The result of the function is then passed to clustering_lines_intervals()

    Parameters
    ----------
    x0: float
        x0 value to find range

    data_by_intervals: dict
        here we need for key 'interval' including start and end cm-1 of range

    Returns
    -------
    out: str | None
        found key
    """
    result_key = None
    for key, item in data_by_intervals.items():
        start, end = item['interval']
        if start <= x0 < end:
            result_key = key
            break
    return result_key


@asyncSlot()
async def clustering_lines_intervals(data_by_intervals: dict, hwhm: float) -> list[ndarray]:
    """
    Determining the final composition of lines for the transferred set of lines. The result of the function is further
     needed to determine final parameters and line deconvolution model template.

    Parameters
    ----------
    data_by_intervals: {'interval': (start, end), 'x0': list, 'lines_count': list}
        parameter 'x0' contains a list of wave-numbers of the interval of all spectra in one list. To return one again
        list to list of lists is used by 'lines_count' which stores the number of lines for each spectrum.

    hwhm: float
        Half Width at Half Maximum - set by the user, affects the size of the clusters.
        The maximum distance between two samples (or sample and center of cluster) for one to be considered
        as in the neighborhood of the other.

    Returns
    -------
    all_ranges_clustered_x0_sd: list[ndarray]
        result of estimate_n_lines_in_range(x0, hwhm) for each range
        2D array with 2 columns: center of cluster x0 and standard deviation of each cluster
    """
    all_ranges_clustered_x0_sd = []
    for item in data_by_intervals.values():
        x0 = split_list(item['x0'], item['lines_count'])
        # info('x0 {!s}'.format(x0))
        x_merged = await centers_of_clusters(x0, hwhm)
        clustered_lines_of_current_range = estimate_n_lines_in_range(x0, hwhm, x_merged)
        all_ranges_clustered_x0_sd.append(clustered_lines_of_current_range)
    # info('clustered_lines_x0 {!s}'.format(all_ranges_clustered_x0_sd))
    return all_ranges_clustered_x0_sd


@asyncSlot()
async def centers_of_clusters(x0: list[list], hwhm: float) -> list[list]:
    """
    A function finds most frequent estimation of raman lines centers list.
    Parameters
    ----------
    x0 : list[list]
        list of estimated centers of raman lines for each variant of estimation
        [[431.48339021272716, 455.09678879007845, 407.35186507907326, 481.3566921374394],
         [430.04300739751113, 458.0407297631946, 405.1497056267086, 475.6762495875454, 445.13839466191627,
          498.92140788914725, 375.36714911050314, 389.2553236563882], ....]
    hwhm : list

    Returns
    -------
    x_merged : list[list]
        for final estimate_n_lines_in_range
    """
    executor = ProcessPoolExecutor()
    with executor:
        current_futures = [get_event_loop().run_in_executor(executor, estimate_n_lines_in_range, x0, hwhm, None) for _
                           in x0]
        various_estimations = await gather(*current_futures)
    various_centers_of_clusters = [np.sort(i[:, 0]) for i in various_estimations]
    shapes = [i.size for i in various_centers_of_clusters]
    most_frequent_shape = np.argmax(np.bincount(shapes))
    various_centers_of_clusters_right_shape = [i for i in various_centers_of_clusters if i.size == most_frequent_shape]
    centers = np.stack(various_centers_of_clusters_right_shape, axis=0)
    centers = np.median(centers, axis=0)
    x_merged = [[x] for x in centers]
    return x_merged


def estimate_n_lines_in_range(x0_lines_spl: list[list[float]], hwhm: float = 12., x_merged=None) \
        -> np.ndarray:
    """
    1. Create first clusters using first 2 (randomly selected) spectrum data
    2. Add to those clusters' data (to the nearest cluster) if distance <= hwhm
    3. If distance > hwhm it's a new cluster
    Parameters
    ----------
    x0_lines_spl: list[list[float]]
        Contains lists of centers x0 of fitted lines for every spectrum
    hwhm: float, optional
        The maximum distance between two samples (or sample and center of cluster) for one to be considered
        as in the neighborhood of the other.
    x_merged: list[list] | None, optional
        centers of clusters. If is None it will be calculated by DBSCAN

    Returns
    -------
    np.ndarray
        2D np.array with 2 columns: center of cluster x0 and standard deviation of each cluster
    """
    if x_merged is None:
        idx_1 = np.random.randint(0, len(x0_lines_spl))
        idx_2 = np.random.randint(0, len(x0_lines_spl))
        x_merged = create_first_clusters(x0_lines_spl[idx_1], x0_lines_spl[idx_2], hwhm)
        if len(x0_lines_spl) == 2:
            return clusters_means(x_merged, std=True).T
    for i in range(0, len(x0_lines_spl)):
        x_means = clusters_means(x_merged)
        for x in x0_lines_spl[i]:
            distances = np.abs(x_means - x)
            min_dist = distances.min()
            if min_dist <= hwhm:
                idx = np.argwhere(distances == min_dist)[0][0]
                x_merged[idx].append(x)
                x_means[idx] = np.mean(x_merged[idx])
            else:
                x_merged.append([x])
                x_means = clusters_means(x_merged)

    return clusters_means(x_merged, std=True).T


def split_list(orig: list, dividers: list) -> list:
    """
    A function to split a general list into a list of lists.
    Parameters
    ----------
    orig : list
        Input list.
    dividers : list
        sizes of portions to divide orig list.

    Returns
    -------
    separated : list
        Contain separated list of lists

    Examples
    --------
    >>> orig = [11, 23, 331, 431, 554, 61]
    >>> dividers = [4, 2]
    >>> split_list(orig, dividers)
    [[11, 23, 331, 431], [554, 61]]
    """
    separated = []
    k = 0
    for i in dividers:
        cur_spec_x0 = orig[k: k + i]
        if not cur_spec_x0 or len(cur_spec_x0) < i:
            continue
        separated.append(cur_spec_x0)
        k += i
    return separated


def create_first_clusters(x1: list[float], x2: list[float], hwhm: float = 12.) -> list[list]:
    """
    Using DBSCAN create first clusters by 1st and 2nd x0_lines
    It is not right to create the first clusters based on a random spectrum just like that, because there may be lines
     too close together.
    Lists are compared element by elements. Those. one element is taken from x1, added to x2. If this new element from
    x1 forms a cluster with other elements, then this cluster is stored in x0_merged.
    Elements are removed from x1 and x2.
    If the cluster is not formed, then these elements remain and are considered emissions (outliers).
    x1 and x2 are references to lists at x0 in estimate_n_lines_in_range, so changes to this function affect x0
    Parameters
    ----------
    x1 : list
        lines x0 parameters of 1st spectrum
    x2 : list
        lines x0 parameters of 2nd spectrum
    hwhm : float, optional
        Half Width at Half Maximum value, selected by user

    Returns
    -------
    x0_merged : list[list[float]]
        list of lists with first clusters

    Examples
    --------
    >>> x1 = [430.38, 452.91, 475.71, 409.45, 388.46, 533.08, 498.37]
    >>> x2 = [430.21, 452.91, 475.68, 410.36, 386.74, 528.14, 504.03, 491.37]
    >>> create_first_clusters(x1, x2)
    [[430.21, 430.38], [452.91, 452.91], [475.68, 475.71], [410.36, 409.45], [386.74, 388.46], [528.14, 533.08], [504.03, 491.37, 498.37]]

    >>> x1 = [585.24, 851.48, 610.96, 814.45, 876.00, 561.77, 776.64, 687.58, 658.56, 757.85]
    >>> x2 = [586.75, 852.08, 612.68, 813.96, 875.17, 567.38, 788.97, 679.67, 645.24, 761.47, 835.62, 721.89, 696.17]
    >>> create_first_clusters(x1, x2)
    [[586.75, 585.24], [852.08, 851.48], [612.68, 610.96], [813.96, 814.45], [875.17, 876.0], [567.38, 561.77], [679.67, 696.17, 687.58], [761.47, 757.85], [788.97], [645.24], [835.62], [721.89], [776.64], [658.56]]
    """

    x0_merged = []
    z = np.append(x2, x1)
    clustered = DBSCAN(eps=hwhm, min_samples=2).fit(z.reshape(-1, 1))
    labels = clustered.labels_
    for i in range(0, labels.max() + 1):
        indexes = np.argwhere(labels == i)
        x0_merged.append(list(z[indexes][:, 0]))
    indexes_outliers = np.argwhere(labels == -1)
    for i in z[indexes_outliers][:, 0]:
        x0_merged.append([i])
    return x0_merged


def clusters_means(x0_merged: list[list], std: bool = False) -> np.ndarray:
    """
    The function returns the centers of the clusters (means)

    Parameters
    ----------
    x0_merged : list[list]
        contains clusters

    std: bool
        True - return 2D array with Standart deviations of clusters. False - 1D array with centers only
    Returns
    -------
    x0_means : np.ndarray
        centers of clusters

    Examples
    --------
    >>> x0_merged = [[586.75, 585.24], [852.08, 851.48], [813.96, 814.45], [875.17, 876.], [612.68, 610.96], [567.38, 561.77], [679.67, 696.17, 687.58], [761.47, 757.85], [788.97], [835.62], [721.89], [645.24], [776.64], [658.56]]
    >>> clusters_means(x0_merged)
    array([585.995, 851.78, 814.205, 875.585, 611.82, 564.575, 687.80666667, 759.66, 788.97, 835.62, 721.89, 645.24,
       776.64, 658.56])
    """
    x0_means = [np.mean(x) for x in x0_merged]
    if std:
        x0_std = [np.std(x) for x in x0_merged]
        return np.stack((x0_means, x0_std))
    else:
        return np.array(x0_means)


def params_func_legend(cur_range_clustered_x0_sd: np.ndarray, n_array: np.ndarray,
                       static_params: tuple[dict, float, float, callable, dict, list[str], float]) \
        -> tuple[Parameters, list[tuple[callable, str]]]:
    """
    Returns parameters and (function, legend) for fitting model preparation .
    Used to create parameters and models after analyzing the lines found automatically from different spectra.
    'a' - peak amplitude value range from 0. to max_y in range x0 ± hwhm
    'x0' - position of peak maximum. x0 ± sd
    'dx' - possible values set by possible_peak_dx() function

    Parameters
    ----------
    cur_range_clustered_x0_sd: list[ndarray]
        result of estimate_n_lines_in_range(x0, hwhm) for current 1 range
        2D array with 2 columns: center of cluster x0 and standard deviation of each cluster

    n_array: np.ndarray
        sliced array of averaged spectrum of current range cm-1.

    static_params: tuple[dict, float, float, callable, dict, list[str]]
         init_params, max_dx, min_hwhm, func, peak_shape_params_limits, param_names
         init_params: dict with 'x_axis': np.ndarray, 'a': float, 'x0': float, 'dx': float, + additional parameters
         max_dx: float -  max limit for dx
         min_hwhm: float - min limit for dx
         func: callable - look peak_shapes_params() in default_values.py
         peak_shape_params_limits: dict - look peak_shape_params_limits() in default_values.py
         param_names: list[str] - Names of peak_shape parameters. Standard params are 'a', 'x0' and 'dx_right'.
          Other param names given from peak_shapes_params() in default_values.py


    Returns
    -------
    out : tuple
         params: Parameters and func_legend: list[tuple[callable, str]]
    """
    func_legend = []
    params = Parameters()
    x_axis = n_array[:, 0]
    y_axis = n_array[:, 1]
    init_params, max_dx, min_hwhm, func, peak_shape_params_limits, param_names, gamma_factor = static_params
    debug('process params_func_legend(): {!s}'.format(cur_range_clustered_x0_sd))
    for x0, sd in cur_range_clustered_x0_sd:
        sd = max(.01, sd)
        legend = legend_from_float(x0)
        func_legend.append((func, legend))
        idx_x0 = nearest_idx(x_axis, x0)
        dx_left, dx_right, arg_left, arg_right, _, _ = possible_peak_dx(n_array, idx_x0)
        a = np.amax(y_axis)
        try:
            y = y_axis[arg_left:arg_right]
            if y.size == 0 or y.shape[0] == 0:
                y = y_axis
            a = np.amax(y)
        except ValueError as msg:
            debug(msg)
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
            if param_name == 'gamma':
                max_v = max_dx * gamma_factor
            if min_v == max_v:
                max_v += .1
                min_v -= .1
            params.add(legend + param_name, v, min=min_v, max=max_v)
    return params, func_legend


def legend_from_float(x0: float) -> str:
    """
    Returns legend (str) like 'k956dot76_' from float 956.76

    Parameters
    ----------
    x0 :
        float - usually x0 value

    Returns
    -------
    out :
        str - legend

    Examples
    --------
    >>> legend_from_float(575.31)
    'k575dot31_'
    >>> legend_from_float(956.76)
    'k956dot76_'
    """
    return 'k%s_' % str(np.round(x0, 2)).replace('.', 'dot')


def possible_peak_dx(n_array: np.ndarray, idx_x0: int | np.ndarray[int]) -> tuple[float, float, int, int, int, int]:
    """
    Finds the maximum possible peak width on the left and right along the spectral contour.

    Parameters
    ----------
    n_array : np.ndarray
        2D array x|y
    idx_x0: int
        index of peak

    Returns
    -------
    out :
        dx_left, dx_right, arg_left (HWHM), arg_right (HWHM), arg_left_4 (HWHM/2), arg_right_4 (HWHM/2)

    Examples
    --------
    >>> x_axis = np.arange(900., 1000.,)
    >>> from modules import gaussian
    >>> y_axis = gaussian(x_axis, 100., 950., 25.)
    >>> n_array = np.vstack((x_axis, y_axis)).T
    >>> possible_peak_dx(n_array, 50)
    (25.0, 25.0, 25, 75, 33, 67)
    >>> y_axis = gaussian(x_axis, 100., 910., 25.)
    >>> n_array = np.vstack((x_axis, y_axis)).T
    >>> possible_peak_dx(n_array, 10)
    (10.0, 25.0, 0, 35, 0, 27)
    """
    x_input = n_array[:, 0]
    y_input = n_array[:, 1]
    x0 = x_input[idx_x0]
    y_x0 = y_input[idx_x0]
    idx_left_qr = idx_x0
    y = y_input[idx_left_qr]
    # find position of left quarter of HWHM
    while y > y_x0 * .75:
        idx_left_qr -= 1
        y = y_input[idx_left_qr]
        if idx_left_qr <= 0:
            break
    # find position of left HWHM
    arg_left = idx_left_qr
    y = y_input[arg_left]
    while y > y_x0 / 2:
        arg_left -= 1
        try:
            y = y_input[arg_left]
        except IndexError:
            break
        if arg_left <= 0:
            arg_left = 0
            break
    # find position of right quarter of HWHM
    idx_right_qr = idx_x0
    y = y_input[idx_right_qr]
    while y > y_x0 * 0.75:
        idx_right_qr += 1
        if idx_right_qr >= y_input.shape[0]:
            idx_right_qr = y_input.shape[0] - 1
            break
        y = y_input[idx_right_qr]
    # find position of right HWHM
    arg_right = idx_right_qr
    y = y_input[arg_right]
    while y > y_x0 / 2:
        arg_right += 1
        if arg_right >= y_input.shape[0]:
            arg_right = y_input.shape[0] - 1
            break
        y = y_input[arg_right]

    if idx_x0 == 0:
        arg_left = 0
        idx_left_qr = 0
    if idx_x0 == y_input.shape[0]:
        arg_right = y_input.shape[0]
        idx_right_qr = y_input.shape[0]
    dx_left = x0 - x_input[arg_left]
    dx_right = x_input[arg_right] - x0
    if idx_x0 == 0:
        dx_left = dx_right
    if idx_x0 == y_input.shape[0]:
        dx_right = dx_left
    if arg_left == arg_right:
        arg_right += 1
    return dx_left, dx_right, arg_left, arg_right, idx_left_qr, idx_right_qr


def update_fit_parameters(params: Parameters, static_parameters: tuple[list[str], float, dict],
                          dynamic_parameters: tuple[str, list, float, float, float]) -> Parameters:
    """
    Prepare parameters for send to guess_peaks()

    Parameters
    ---------
    params : lmfit Parameters
    static_parameters : tuple[list[str], float, dict]
        param_names : list[str]; List of parameter names. Example: ['a', 'x0', 'dx'].
        min_fwhm, : float; the minimum value FWHM, determined from the input table (the minimum of all).
        peak_shape_params_limits: dict[str, tuple[float, float]]; see peak_shape_params_limits()
            in default_values.py.

    dynamic_parameters : tuple[str, list, float, float, float]
        legend : str - prefix for the line in the model. like 'k977dot15_'
        init_params : list[float]; Initial values of parameters for a given spectrum and line type.
        y_max_in_range : float; amplitude value from table.
        dx_left : float; left HWHM
        dx_right : float; right HWHM

    Returns
    -------
    out : Parameters
        updated Parameters
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
            dx0 = (min_fwhm / 4.) + 1.
            min_v = v - dx0
            max_v = v + dx0
        elif param_name == 'dx':
            min_v = min_fwhm / 2.
            max_v = dx_right
        elif param_name == 'dx_left':
            min_v = min_fwhm / 2.
            max_v = dx_left
        elif param_name in params_limits:
            min_v = params_limits[param_name][0]
            max_v = params_limits[param_name][1]
        if param_name == 'gamma':
            max_v = min(dx_left, dx_right) * params_limits['l_ratio'][1]
        if min_v is None or max_v is None:
            params.add(legend + param_name, v)
        else:
            if min_v == max_v:
                max_v += .001
            params.add(legend + param_name, v, min=min_v, max=max_v)

    return params


def show_distribution(x0_lines: list[float], averaged_spec: np.ndarray, clustered_x0: np.ndarray):
    """
    Not used in code. Available only by F2 on page2 fitting after guess.
    """
    X = np.array(x0_lines).reshape(-1, 1)
    averaged_spec = cut_full_spectrum(averaged_spec, X.min(), X.max())
    X_plot = np.linspace(X.min(), X.max(), 1000)[:, np.newaxis]
    fig, ax = plt.subplots()
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    Y = np.exp(log_dens)
    ax.plot(
        X_plot[:, 0],
        Y,
        color="black",
        lw=2,
        linestyle="-",
        label='Gaussian kernel density'
    )

    ax.plot(X[:, 0], -0.01 * np.random.random(X.shape[0]), ".", color='black')
    color = 'tab:red'
    ax2 = ax.twinx()
    ax.set_ylabel('Distribution density', color='black')
    ax2.set_ylabel('Intensity, rel. un.', color=color)
    ax2.plot(averaged_spec[:, 0], averaged_spec[:, 1], color=color, label='Averaged Raman spectrum')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, averaged_spec[:, 1].max()])
    fig.tight_layout()
    ax.set_xlabel('Raman shift, cm\N{superscript minus}\N{superscript one}')
    mpl_axes_aligner.align.yaxes(ax, 0, ax2, 0)

    for i in clustered_x0:
        x0 = i[0]
        sd = i[1]
        ax.axvspan(x0 - sd, x0 + sd, alpha=0.1, color='blue', label=str(np.round(x0, 1)))
        print(x0)
    ax.vlines(list(clustered_x0[:, 0]), 0, 1, colors='blue', linestyles='dashed')
    # ax.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    plt.show()
