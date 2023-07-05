import numpy as np
from numpy.linalg import norm
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyval as np_polyval
from pybaselines import Baseline
from scipy.sparse import csc_matrix, spdiags, diags, eye as sparse_eye
from scipy.sparse.linalg import spsolve
from modules.functions_for_arrays import normalize_between_0_1, diff
from numba import njit, float64, int64, char
from modules.numba_polyfit import polyfit, polyval, fit_poly
from logging import info
from scipy.signal import deconvolve


def baseline_poly(item: tuple[str, np.ndarray], params: int) -> tuple[str, np.ndarray, np.ndarray]:
    """Computes a polynomial that fits the baseline of the data. """
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    poly_order = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.poly(y_input, poly_order)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_modpoly(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    """Implementation of Modified polyfit method from paper: Automated Method for Subtraction of Fluorescence from
    Biological Raman Spectra, by Lieber & Mahadevan-Jansen (2003)

    degree: Polynomial degree, default is 6

    max_iter: How many iterations to run. Default is 100

    gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration.
    If gain in any iteration is less than this, further improvement will stop
    """
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]

    # # initial improvement criteria is set as positive infinity, to be replaced later on with actual value
    # criteria = np.inf
    # baseline_fitted = y_input
    # y_old = y_input.copy()
    # n_iter = 0
    #
    # while (criteria >= gradient) and (n_iter < max_iter):
    #     poly_coef = polynomial.polyfit(x_input, y_old, degree)
    #     baseline_fitted = polyval(x_input, poly_coef)
    #     y_work = np.minimum(y_input, baseline_fitted)
    #     criteria = sum(abs((y_work - y_old) / y_old))
    #     y_old = y_work
    #     n_iter += 1
    # # end of fitting procedure
    # y_new = y_input - baseline_fitted
    # print(n_iter)
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.modpoly(y_input, degree, gradient, max_iter)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_imodpoly(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    """IModPoly from paper: Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman
     Spectroscopy, by Zhao, Jianhua, Lui, Harvey, McLean, David I., Zeng, Haishan (2007)

    degree: Polynomial degree, default is 7

    max_iter: How many iterations to run. Default is 100

    gradient: Gradient for polynomial loss, default is 0.002. It measures incremental gain over each iteration. If gain
     in any iteration is less than this, further improvement will stop
    """
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    # y = y_input
    # x = x_input
    # n_iter = 0
    # dev = 0
    # while n_iter < max_iter:
    #     n_iter += 1
    #     previous_dev = dev
    #     poly_coef = polynomial.polyfit(x, y, degree)
    #     baseline_fitted = polyval(x, poly_coef)
    #     di = y - baseline_fitted
    #     dev = np.std(di)
    #     if n_iter == 1:
    #         y = y[y_input <= (baseline_fitted + dev)]
    #         x = x[y_input <= (baseline_fitted + dev)]
    #         baseline_fitted = baseline_fitted[y_input <= (baseline_fitted + dev)]
    #     stop_criteria = np.abs((dev - previous_dev) / dev)
    #     # print('imodpoly stop_criteria: ', stop_criteria, ' dev ', dev, ' n_iter ', n_iter)
    #     if stop_criteria < gradient:
    #         break
    #     y = np.minimum(y, baseline_fitted + dev)
    # poly_coef = polynomial.polyfit(x, y, degree)
    # baseline_fitted = polyval(x_input, poly_coef)

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.imodpoly(y_input, degree, gradient, max_iter)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


# def baseline_imodpoly_plus(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
#     """IModPoly from paper: Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman
#     Spectroscopy, by Zhao, Jianhua, Lui, Harvey, McLean, David I., Zeng, Haishan (2007)
#
#     degree: Polynomial degree, default is 2
#
#     max_iter: How many iterations to run. Default is 100
#
#     gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration.
#     If gain in any iteration is less than this, further improvement will stop
#     """
#
#     key = item[0]
#     input_array = item[1]
#     x_input = input_array[:, 0]
#     y_input = input_array[:, 1]
#     # optional parameters
#     degree = params[0]
#     gradient = params[1]
#     max_iter = params[2]
#     y = y_input
#     x = x_input
#     n_iter = 0
#     dev = 0
#     while n_iter <= max_iter:
#         n_iter += 1
#         previous_dev = dev
#         poly_coef = polynomial.polyfit(x, y, degree)
#         baseline_fitted = polyval(x, poly_coef)
#         di = y - baseline_fitted
#         residual = di * np.power(np.std(di) / 1000, (1 - (abs(di) / di)))
#         dev = np.std(residual)
#         if n_iter == 1:
#             y = y[y_input <= (baseline_fitted + dev)]
#             x = x[y_input <= (baseline_fitted + dev)]
#             baseline_fitted = baseline_fitted[y_input <= (baseline_fitted + dev)]
#         stop_criteria = abs((dev - previous_dev) / dev)
#         if stop_criteria < gradient or dev < 1e-06:
#             break
#         y = np.minimum(y, baseline_fitted + dev)
#     poly_coef = polynomial.polyfit(x, y, degree)
#     baseline_fitted = polyval(x_input, poly_coef)
#     y_new = y_input - baseline_fitted
#     print(n_iter)
#     return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T

@njit(cache=True, fastmath=True)
def ex_mod_poly(item: tuple[str, np.ndarray], params: list[float, float, float]) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Ex-ModPoly is extension of ModPoly method from paper: 'Automated Method for Subtraction of Fluorescence from
    Biological Raman Spectra, by Lieber & Mahadevan-Jansen (2003)'

    Returns baseline and raman spectrum as 2D ndarray.

    Main differences from ModPoly and I-ModPoly are:
        1) Each iteration, the vector of weights is calculated for polyfit. Weights are inverted derivative of given y.
            Weight value for lows is higher than for peaks.
        2) At iteration = 0 removing strong raman peaks at intensity higher than fitted baseline.
            Not higher than (baseline + DEV) like in I-ModPoly.
        3) Residual deviation now calculating by formula:
            r = y - baseline
            coef = min(np.std(r) / 1000., .1)
            dev = r * coef ^ (1 - np.abs(r) / r)
            dev = np.std(dev)
        4) Stop criteria now is tilt angle of line approximated to latest N (N = window_size) stop_criteria.
            Pitch < than threshold (usually 1e-7) means that for latest N iteration dev wasn't significantly changed and
            further iterations cannot significantly improve the fitting.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        key: str
            filename
        input_array: ndarray
            2D array of smoothed spectrum
    params : list[float, float, float]
        degree: float
            polynome degree as float (will be converted to int). Recommended value from 5 to 11.
        threshold: float
            pitch corresponding to angle of tilt for line of stop_criteria with some window_size. Default is 1e-7.
             Recommended (1e-7 ; 1e-10)
            Less values means more iterations.
        max_iter: float
            iterations limit (will be converted to int). Recommended value from 250 to 1000.

    Returns
    -------
    out : tuple[str, np.ndarray, np.ndarray]
        key: str
            filename as given without changes.
        baseline: ndarray
            2D array of baseline fitted by polynome of some degree.
        y_raman: ndarray
            2D array of Raman spectrum = input_array - baseline.
    """
    key, input_array = item
    x_input, y_input = input_array[:, 0], input_array[:, 1]
    degree, threshold, max_iter = params
    degree, max_iter = int(degree), int(max_iter)
    x, y = x_input.copy(), y_input.copy()
    stop_criteria = [0.]
    dev = 0.
    window_size = min(int(max_iter / 10), 100)
    for i in range(max_iter):
        dev_prev = dev
        poly_coef = fit_poly(x, y, degree, _weights(y))
        baseline = polyval(x, poly_coef)
        if i == 0:                                                          # Remove strong raman peaks.
            idx = np.argwhere(y <= baseline).T[0]
            x, y, baseline = x[idx], y[idx], baseline[idx]
        dev = max(_residual_deviation(y - baseline), 1e-20)
        y = np.minimum(y, baseline + dev)                                   # Reconstruct model input y data.
        sc = np.abs((dev - dev_prev) / dev)
        stop_criteria.append(sc)
        if i != 0 and _pitch(i, stop_criteria, window_size) < threshold:    # Check that further iterations cannot ...
            break                                                           # ... significantly improve the fitting.
    poly_coef = fit_poly(x, y, degree, _weights(y))              # Final fitting.
    baseline = polyval(x_input, poly_coef)                       # Make baseline same shape like x_input.
    y_raman = y_input - baseline                                 # Baseline corrected spectrum
    return key, np.vstack((x_input, baseline)).T, np.vstack((x_input, y_raman)).T


@njit(cache=True, fastmath=True)
def _pitch(i: int, stop_criteria: list, window_size: int) -> float:
    """
    Returns pitch of line.
    Pitch close to 0 means no significant changes in deviation.
    Pitch is 'b' in polynome of 1st degree y = a + bx.

    Parameters
    ----------
    i : int
        current iteration.
    stop_criteria : list
        contains all deviations calculated.
    window_size : int
        size of window to approximate polynome.

    Returns
    -------
    pitch : float
        angle of tilt for line
    """
    window = stop_criteria[-window_size: -1] if i >= window_size else stop_criteria[: i + 1]
    x = np.linspace(0, i, len(window))
    y = np.array(window)
    pitch = polyfit(x, y, 1)[1]
    return abs(pitch)


@njit(fastmath=True)
def _residual_deviation(r: np.ndarray) -> float:
    """
    Returns deviation of residual by formula:
    dev = r * coef ^ (1 - abs(r) / r)
    dev = std(dev)

    Parameters
    ----------
    r : ndarray
        Input array

    Returns
    -------
    dev : ndarray
        deviation
    """
    coef = min(np.std(r) / 1000., .1)
    m = 1 - (np.abs(r) / r)
    r *= np.power(coef, m)
    return np.std(r)


@njit(float64[:](float64[:]), fastmath=True)
def _weights(y: np.ndarray) -> np.ndarray:
    """
    Returns weights for polyfit.
    Weights is inverted diff of given y.

    Parameters
    ----------
    y : ndarray
        Input array

    Returns
    -------
    w : ndarray
        Weights

    Examples
    --------
    >>> x = np.array([1., 2., 4., 7., 0.])
    >>> _weights(x)
    array([1.        , 0.85714286, 0.71428571, 0.57142857, 0.        ])
    """
    w = diff(y)
    w = np.abs(w)
    w = normalize_between_0_1(w)
    return np.abs(w - 1.)


def baseline_penalized_poly(item: tuple[str, np.ndarray], params: list[int, float, int, float, str]) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    alpha_factor = params[3]
    cost_function = params[4]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.penalized_poly(y_input, degree, gradient, max_iter, cost_function=cost_function,
                                                     alpha_factor=alpha_factor)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_loess(item: tuple[str, np.ndarray], params: list[int, float, int, float, int]) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    fraction = params[3]
    scale = params[4]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.loess(y_input, fraction, poly_order=degree, tol=gradient, max_iter=max_iter,
                                            scale=scale)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_quant_reg(item: tuple[str, np.ndarray], params: list[int, float, int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    quantile = params[3]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.quant_reg(y_input, degree, quantile, gradient, max_iter)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_goldindec(item: tuple[str, np.ndarray], params: list[int, float, int, str, float, float]) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    cost_func = params[3]
    peak_ratio = params[4]
    alpha_factor = params[5]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.goldindec(y_input, degree, gradient, max_iter, cost_function=cost_func,
                                                peak_ratio=peak_ratio, alpha_factor=alpha_factor)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_asls(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    # Matlab code in Eilers et Boelens 2005
    # Python adaptation found on stackoverflow:
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    niter = params[2]
    z = np.zeros_like(y_input)
    # starting the algorithm
    y_shape = y_input.shape[0]
    d = csc_matrix(np.diff(np.eye(y_shape), 2))
    w = np.ones(y_shape)
    for i in range(niter):
        w_w = spdiags(w, 0, y_shape, y_shape)
        z_z = w_w + lam * d.dot(d.transpose())
        z = spsolve(z_z, w * y_input)
        w = p * (y_input > z) + (1 - p) * (y_input < z)

    baseline_fitted = z
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_iasls(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.iasls(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_arpls(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    # Adaptation of the Matlab code in Baek et al. 2015 DOI: 10.1039/c4an01061b

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    # optional parameters
    lam = params[0]
    ratio = params[1]
    n_iter = params[2]

    n = y_input.shape[0]
    d_d = csc_matrix(np.diff(np.eye(n), 2))
    w = np.ones(n)
    i = 0
    while True:
        i += 1
        w_w = spdiags(w, 0, n, n)
        z_z = w_w + lam * d_d.dot(d_d.transpose())
        z = spsolve(z_z, w * y_input)
        d = y_input - z
        # make d- and get w^t with m and s
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        # check exit condition and backup
        if norm(w - wt) / norm(w) < ratio:
            break
        if i >= n_iter:
            break
        w = wt
    baseline_fitted = z
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def _whittaker_smooth(x, w, lambda_):
    """
        Penalized the least squares algorithm for background fitting

        input
            x: input data (i.e. chromatogram of spectrum)
            w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
            lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting
             background

        output
            the fitted background vector
    """
    x = np.matrix(x)
    m = x.size
    e = sparse_eye(m, format='csc')
    d = e[1:] - e[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    w = diags(w, shape=(m, m))
    a = csc_matrix(w + (lambda_ * d.T * d))
    b = csc_matrix(w * x.T)
    background = spsolve(a, b)
    return np.array(background)


def baseline_airpls(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Implementation of Zhang fit for Adaptive iteratively reweighted penalized the least squares for baseline fitting.
    Modified from Original implementation by Professor Zhimin Zhang at https://github.com/zmzhang/airPLS/
    https://doi.org/10.1039/B922045C

    lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z

    max_iter: how many iterations to run, and default value is 15.
    """

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitted = []
    y_len = y_input.shape[0]
    w = np.ones(y_len)
    for i in range(1, max_iter + 1):
        baseline_fitted = _whittaker_smooth(y_input, w, lam)
        d = y_input - baseline_fitted
        dssn = abs(d[d < 0].sum())
        if dssn < p * (np.sum(abs(y_input))):
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]

    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_drpls(item: tuple[str, np.ndarray], params: list[int, float, int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    # according to Applied Optics, 2019, 58, 3913-3920. https://doi.org/10.1364/AO.58.003913
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    ratio = params[1]
    niter = params[2]
    eta = params[3]
    niter = 100 if niter > 100 else niter
    # optional smoothing in the next line, currently commented out
    # y = np.around(savgol_filter(raw_data,19,2,deriv=0,axis=1),decimals=6)

    y_shape = y_input.shape[0]

    d_d = diags([1, -2, 1], [0, -1, -2], shape=(y_shape, y_shape - 2), format='csr')
    d_d = d_d.dot(d_d.transpose())
    d_1 = diags([-1, 1], [0, -1], shape=(y_shape, y_shape - 1), format='csr')
    d_1 = d_1.dot(d_1.transpose())

    w_0 = np.ones(y_shape)
    i_n = diags(w_0, format='csr')

    # this is the code for the fitting procedure
    w = w_0
    w_w = diags(w, format='csr')
    z_z = w_0

    for jj in range(int(niter)):
        w_w.setdiag(w)
        z_prev = z_z
        z_z = spsolve(w_w + d_1 + lam * (i_n - eta * w_w) * d_d, w_w * y_input, permc_spec='NATURAL')
        if np.linalg.norm(z_z - z_prev) > ratio:
            d = y_input - z_z
            d_negative = d[d < 0]
            sigma_negative = np.std(d_negative)
            mean_negative = np.mean(d_negative)
            w = 0.5 * (1 - np.exp(jj) * (d - (-mean_negative + 2 * sigma_negative)) / sigma_negative / (
                    1 + abs(np.exp(jj) * (d - (-mean_negative + 2 * sigma_negative)) / sigma_negative)))
        else:
            break
    # end of fitting procedure

    baseline_fitted = z_z
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_iarpls(item: tuple[str, np.ndarray], params: list[int, int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    tol = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.iarpls(y_input, lam, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_aspls(item: tuple[str, np.ndarray], params: list[int, int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    tol = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.aspls(y_input, lam, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_psalsa(item: tuple[str, np.ndarray], params: list[int, int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.psalsa(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_derpsalsa(item: tuple[str, np.ndarray], params: list[int, int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.derpsalsa(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mpls(item: tuple[str, np.ndarray], params: tuple[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mpls(y_input, lam=lam, p=p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mor(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mor(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_imor(item: tuple[str, np.ndarray], params: list[int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    max_iter = params[0]
    tol = params[1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.imor(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mormol(item: tuple[str, np.ndarray], params: list[int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    max_iter = params[0]
    tol = params[1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mormol(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_amormol(item: tuple[str, np.ndarray], params: list[int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    max_iter = params[0]
    tol = params[1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.amormol(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_rolling_ball(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.rolling_ball(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mwmv(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mwmv(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_tophat(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.tophat(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mpspline(item: tuple[str, np.ndarray], params: tuple[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    lam = params[0]
    p = params[1]
    spline_degree = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mpspline(y_input, lam=lam, p=p, spline_degree=spline_degree)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_jbcd(item: tuple[str, np.ndarray], params: list[int, float]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    max_iter = params[0]
    tol = params[1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.jbcd(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mixture_model(item: tuple[str, np.ndarray], params: list[int, float, int, int, float]) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    lam = params[0]
    p = params[1]
    spline_degree = params[2]
    max_iter = params[3]
    tol = params[4]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mixture_model(y_input, lam, p, spline_degree=spline_degree, max_iter=max_iter,
                                                    tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_irsqr(item: tuple[str, np.ndarray], params: list[int, float, int, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    lam = params[0]
    quantile = params[1]
    spline_degree = params[2]
    max_iter = params[3]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.irsqr(y_input, lam, quantile, spline_degree=spline_degree, max_iter=max_iter)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_corner_cutting(item: tuple[str, np.ndarray], params: int) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    max_iter = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.corner_cutting(y_input, max_iter)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_noise_median(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.noise_median(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_snip(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.snip(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_swima(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.swima(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_ipsa(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.ipsa(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_ria(item: tuple[str, np.ndarray], params: float) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    tol = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.ria(y_input, tol=tol)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_dietrich(item: tuple[str, np.ndarray], params: list[float, int, float, int, int, int]) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    num_std = params[0]
    poly_order = params[1]
    tol = params[2]
    max_iter = params[3]
    interp_half_window = params[4]
    min_length = params[5]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.dietrich(y_input, num_std=num_std, poly_order=poly_order, tol=tol,
                                               max_iter=max_iter, interp_half_window=interp_half_window,
                                               min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_golotvin(item: tuple[str, np.ndarray], params: list[float, int, int, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    num_std = params[0]
    interp_half_window = params[1]
    min_length = params[2]
    sections = params[3]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.golotvin(y_input, num_std=num_std, interp_half_window=interp_half_window,
                                               sections=sections, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_std_distribution(item: tuple[str, np.ndarray], params: list[float, int, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    num_std = params[0]
    interp_half_window = params[1]
    fill_half_window = params[2]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.std_distribution(y_input, num_std=num_std,
                                                       interp_half_window=interp_half_window,
                                                       fill_half_window=fill_half_window)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_fastchrom(item: tuple[str, np.ndarray], params: list[int, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    interp_half_window = params[0]
    max_iter = params[1]
    min_length = params[2]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.fastchrom(y_input, interp_half_window=interp_half_window,
                                                max_iter=max_iter, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_fabc(item: tuple[str, np.ndarray], params: list[int, float, int]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    lam = params[0]
    num_std = params[1]
    min_length = params[2]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.fabc(y_input, lam, num_std=num_std, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_optimize_extended_range(item: tuple[str, np.ndarray], params: str) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    method = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.optimize_extended_range(y_input, method=method)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_adaptive_minmax(item: tuple[str, np.ndarray], params: str) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    method = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.adaptive_minmax(y_input, method=method)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_beads(item: tuple[str, np.ndarray], _) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.beads(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T
