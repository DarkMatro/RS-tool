import numpy as np
from numpy.linalg import norm, lstsq
from numpy.polynomial.polyutils import getdomain, mapdomain
from numpy.polynomial.polynomial import polyvander
from pybaselines import Baseline
from scipy.sparse import csc_matrix, spdiags, diags, eye as sparse_eye
from scipy.sparse.linalg import spsolve
from scipy.signal import detrend
from modules.mutual_functions.work_with_arrays import extend_bool_mask_two_sided
from numba import njit


def baseline_poly(item: tuple[str, np.ndarray], poly_order: int = 5) -> tuple[str, np.ndarray, np.ndarray]:
    """Computes a polynomial that fits the baseline of the data. """
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.poly(y_input, poly_order)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_modpoly(item: tuple[str, np.ndarray], poly_order: int = 5, tol: float = 1e-3, max_iter: int = 250) \
        -> tuple[str, np.ndarray, np.ndarray]:
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
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.modpoly(y_input, poly_order, tol, max_iter)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_imodpoly(item: tuple[str, np.ndarray], poly_order: int = 6, tol: float = 1e-3, max_iter: int = 250,
                      num_std: float = 0.) -> tuple[str, np.ndarray, np.ndarray]:
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
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.imodpoly(y_input, poly_order, tol, max_iter, num_std=num_std)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def ex_mod_poly(item: tuple[str, np.ndarray], poly_order: int = 5, tol: float = 1e-6, max_iter: int = 100,
                quantile: float = 1e-5, w_scale_factor: float = .5, recalc_y: bool = False, num_std=3.,
                window_size=3) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Ex-ModPoly is extension of ModPoly method from paper [1] and Quantile regression using iteratively
    reweighed least squares (IRLS) [2]

    Returns baseline and raman spectrum as 2D ndarray.

    Main differences from ModPoly, I-ModPoly and Quantile regression are:
        1) The presence of absorption lines in the spectrum is taken into account
        2) The accuracy of polynomial approximation for a descending baseline as in the spectra of biomaterials has
            been increased by scaling the IRLS weights defined as in the article [3] and rebuilding y signal at each
            iteration.
        3) Increased performance compared to other methods.

    Metrics for 1000 synthesized Raman spectra.
    EMSC normalized spectra
                   MAPE, %       R^2      time, sec
    I-ModPoly |  12.468    | 0.9997914 |  40
    Quant_reg |   0.163    | 0.9999796 |   6
    Ex-ModPoly|   0.132    | 0.9999883 |   4

    SNV normalized spectra
                   MAPE, %       R^2      time, sec
    I-ModPoly |  14.187    | 0.9997348 |  40
    Quant_reg |   4.231    | 0.9929289 |  70
    Ex-ModPoly|   0.135    | 0.9999882 |   5

    For results like using Quantile regression make w_scale_factor=0.5, recalc_y=False.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        key: str
            filename
        input_array: ndarray
            2D array of smoothed spectrum with baseline + raman signal
    poly_order: float, optional
        The polynomial order for fitting the baseline. Recommended value from 5 to 11.
    tol: float, optional
        The exit criteria. Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations. Default is 100.
    quantile : float, optional
        The quantile at which to fit the baseline. Default is 1e-4.
    w_scale_factor: float, optional
        Scales quantile regression weights by formula: q = q**w_scale_factor. Where q calculated like in quantile
        regression [3]
        Low values like 1e-10 makes weights like rectangular shape (0 for negative residual, 1 for positive).
        0.5 is standard value makes result equal to quantile regression.
        Values lower than 0.5 allows make baseline lower
        For SNV normalization recommended range [0.01 - 0.1] with recalc_y = True
        For EMSC normalization recommended range [0.01 - 0.5] with recalc_y = False
    recalc_y: bool, optional
        Rebuild y in at every iteration by formula: y = np.minimum(y, baseline)
        True makes result like using ModPoly-IModPoly
        False makes result like using Quantile regression. Default is False.
    num_std : float, optional
        For absorption removing. Consider signal lower than (baseline - num_std * np.std(detrended(y)))
        as strong absorption.
        Higher values for stronger absorption lines. Lower value for weak absorption lines.
        If num_std = 0.0 absorption signal will not be removed.
        Default is 3.
    window_size : int, optional
        For absorption removing.
        How many points around found absorption signal will be considered as absorption too.
        window_size = 2, makes mask of absorption indexes like [0, 0, 1, 0, 0] --> [0, 1, 1, 1, 0]
        Higher values for wide absorption lines, lower values for narrow.

    Returns
    -------
    out : tuple[str, np.ndarray, np.ndarray]
        key: str
            filename as given without changes.
        baseline: ndarray
            2D array of baseline fitted by polynome of some degree.
        y_raman: ndarray
            2D array of Raman spectrum = input_array - baseline.

    Notes
    ----
        Modpoly algorithm originally developed in [1]_.

        I-ModPoly algorithm originally developed in [2]_.

        Performs quantile regression using iteratively reweighted least squares (IRLS)
        as described in [3]_.

     References
    ----------
    .. [1] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
    .. [2] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.
    .. [3] Schnabel, S., et al. Simultaneous estimation of quantile curves using
            quantile sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

    """
    key, input_array = item
    x_input, y_input = input_array[:, 0], input_array[:, 1]
    x, y = x_input.copy(), y_input.copy()
    w = np.ones_like(y)
    x_domain = getdomain(x)
    mapped_x = mapdomain(x, x_domain, np.array([-1., 1.]))
    vandermonde = polyvander(mapped_x, poly_order)
    # 1st baseline fit without weights
    coef = lstsq(vandermonde * w[:, None], y * w, None)[0]
    baseline = vandermonde @ coef
    # Find indexes of strong absorption lines to remove
    indexes = absorption_indexes(y, baseline, num_std, window_size) if num_std != 0.0 else None
    for i in range(max_iter):
        baseline_old = baseline
        coef = lstsq(vandermonde * w[:, None], y * w, None)[0]
        baseline = vandermonde @ coef
        w = _quantile(y, baseline, quantile, None, w_scale_factor, indexes)
        if i == 0:
            continue
        if recalc_y:
            y = np.minimum(y, baseline)
        if relative_difference(baseline, baseline_old) < tol:
            break
    y_raman = y_input - baseline  # Corrected spectrum.
    return key, np.vstack((x_input, baseline)).T, np.vstack((x_input, y_raman)).T


def absorption_indexes(y: np.ndarray, baseline: np.ndarray, sd_factor: float = 3., window_size=2) -> np.ndarray:
    """
    Find indexes of strong absorption lines. Consider signal lower than (baseline - sd_factor * np.std(detrended(y)))
        as strong absorption.

    Parameters
    ----------
    y : numpy.ndarray
       The values of the raw data.
    baseline : numpy.ndarray
       1st fitted baseline without weights.
    sd_factor : float, optional
        For absorption removing. Consider signal lower than (baseline - num_std * np.std(detrended(y)))
        as strong absorption.
        Higher values for stronger absorption lines. Lower value for weak absorption lines.
        If num_std = 0.0 absorption signal will not be removed.
        Default is 3.
    window_size : int, optional
        For absorption removing.
        How many points around found absorption signal will be considered as absorption too.
        window_size = 2, makes mask of absorption indexes like [0, 0, 1, 0, 0] --> [0, 1, 1, 1, 0]
        Higher values for wide absorption lines, lower values for narrow.

    Returns
    -------
    numpy.ndarray
        Indexes
    """
    detrended = detrend(y)
    sd = np.std(detrended)
    cond = y < (baseline - sd_factor * sd)
    cond = extend_bool_mask_two_sided(cond.tolist(), window_size)
    idx = np.argwhere(cond).T[0]
    return idx


@njit(cache=True, fastmath=True)
def _quantile(y: np.ndarray, fit: np.ndarray, quantile: float = 1e-4, eps=None, w_scale_factor: float = .5,
              absorption_idx=None) -> np.ndarray:
    r"""
    An approximation of quantile loss.

    The loss is defined as :math:`\rho(r) / |r|`, where r is the residual, `y - fit`,
    and the function :math:`\rho(r)` is `quantile` for `r` > 0 and 1 - `quantile`
    for `r` < 0. Rather than using `|r|` as the denominator, which is non-differentiable
    and causes issues when `r` = 0, the denominator is approximated as
    :math:`\sqrt{r^2 + eps}` where `eps` is a small number.

    Parameters
    ----------
    y : numpy.ndarray
        The values of the raw data.
    fit : numpy.ndarray
        The fit values.
    quantile : float
        The quantile value.
    eps : float, optional
        A small value added to the square of `residual` to prevent dividing by 0.
        Default is None, which uses `(1e-6 * max(abs(fit)))**2`.
    w_scale_factor : float
        scaling coefficient
    absorption_idx: numpy.ndarray
        indexes of absorption lines to be zero.

    Returns
    -------
    numpy.ndarray
        The calculated loss, which can be used as weighting when performing iteratively
        reweighted least squares (IRLS)

    References
    ----------
    Schnabel, S., et al. Simultaneous estimation of quantile curves using quantile
    sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

    """
    if eps is None:
        # 1e-6 seems to work better than the 1e-4 in Schnabel, et al
        eps = (np.abs(fit).max() * 1e-6) ** 2
        eps = min(eps, 2.22e-16)
    residual = y - fit
    numerator = np.where(residual > 0, quantile, 1 - quantile)
    # use max(eps, _MIN_FLOAT) to ensure that eps + 0 > 0
    denominator = np.sqrt(residual ** 2 + eps)  # approximates abs(residual)
    q = (numerator / denominator) ** w_scale_factor
    if absorption_idx is not None:
        q[absorption_idx] = 0.
    return q


@njit(cache=True, fastmath=True)
def relative_difference(old: np.ndarray, new: np.ndarray, norm_order=None) -> float:
    """
    Calculates the relative difference, ``(norm(new-old) / norm(old))``, of two values.

    Used as an exit criteria in many baseline algorithms.

    Parameters
    ----------
    old : numpy.ndarray or float
        The array or single value from the previous iteration.
    new : numpy.ndarray or float
        The array or single value from the current iteration.
    norm_order : int, optional
        The type of norm to calculate. Default is None, which is l2
        norm for arrays, abs for scalars.

    Returns
    -------
    float
        The relative difference between the old and new values.

    """
    numerator = np.linalg.norm(new - old, None)
    denominator = np.maximum(np.linalg.norm(old, norm_order), 1e-20)
    return numerator / denominator


def baseline_penalized_poly(item: tuple[str, np.ndarray], poly_order: int = 6, tol: float = 1e-3, max_iter: int = 250,
                            alpha_factor: float = 0.99999, cost_function: str = 'asymmetric_truncated_quadratic') \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.penalized_poly(y_input, poly_order, tol, max_iter, cost_function=cost_function,
                                                     alpha_factor=alpha_factor, threshold=0.001)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_quant_reg(item: tuple[str, np.ndarray], poly_order: int = 5, tol: float = 1e-6, max_iter: int = 100,
                       quantile: float = 1e-5) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.quant_reg(y_input, poly_order, quantile, tol, max_iter)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_goldindec(item: tuple[str, np.ndarray], poly_order: int = 5, tol: float = 1e-6, max_iter: int = 100,
                       alpha_factor: float = 0.99999, cost_function: str = 'asymmetric_truncated_quadratic',
                       peak_ratio: float = 0.5) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.goldindec(y_input, poly_order, tol, max_iter, cost_function=cost_function,
                                                peak_ratio=peak_ratio, alpha_factor=alpha_factor)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_asls(item: tuple[str, np.ndarray], lam: int = 1e6, p: float = 1e-3, max_iter: int = 50) \
        -> tuple[str, np.ndarray, np.ndarray]:
    # Matlab code in Eilers et Boelens 2005
    # Python adaptation found on stackoverflow:
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    z = np.zeros_like(y_input)
    # starting the algorithm
    y_shape = y_input.shape[0]
    d = csc_matrix(np.diff(np.eye(y_shape), 2))
    w = np.ones(y_shape)
    for i in range(max_iter):
        w_w = spdiags(w, 0, y_shape, y_shape)
        z_z = w_w + lam * d.dot(d.transpose())
        z = spsolve(z_z, w * y_input)
        w = p * (y_input > z) + (1 - p) * (y_input < z)

    baseline_fitted = z
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_iasls(item: tuple[str, np.ndarray], lam: int = 1e6, p: float = 1e-3, max_iter: int = 50) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.iasls(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_arpls(item: tuple[str, np.ndarray], lam: int = 1e5, p: float = 1e-6, max_iter: int = 50) \
        -> tuple[str, np.ndarray, np.ndarray]:
    # Adaptation of the Matlab code in Baek et al. 2015 DOI: 10.1039/c4an01061b

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

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
        if norm(w - wt) / norm(w) < p:
            break
        if i >= max_iter:
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


def baseline_airpls(item: tuple[str, np.ndarray], lam: int = 1e6, p: float = 1e-6, max_iter: int = 50) \
        -> tuple[str, np.ndarray, np.ndarray]:
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


def baseline_drpls(item: tuple[str, np.ndarray], lam: int = 1e5, p: float = 1e-6, max_iter: int = 50, eta: float = .5) \
        -> tuple[
            str, np.ndarray, np.ndarray]:
    # according to Applied Optics, 2019, 58, 3913-3920. https://doi.org/10.1364/AO.58.003913
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters

    niter = 100 if max_iter > 100 else max_iter
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
        if np.linalg.norm(z_z - z_prev) > p:
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


def baseline_iarpls(item: tuple[str, np.ndarray], lam: int = 100, p: float = 1e-6, max_iter: int = 50) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.iarpls(y_input, lam, max_iter=max_iter, tol=p)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_aspls(item: tuple[str, np.ndarray], lam: int = 1e5, p: float = 1e-6, max_iter: int = 100) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.aspls(y_input, lam, max_iter=max_iter, tol=p)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_psalsa(item: tuple[str, np.ndarray], lam: int = 1e5, p: float = .5, max_iter: int = 50) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.psalsa(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_derpsalsa(item: tuple[str, np.ndarray], lam: int = 1e3, p: float = .01, max_iter: int = 50) -> tuple[
    str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.derpsalsa(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mpls(item: tuple[str, np.ndarray], lam: int = 1e6, p: float = 0.0, max_iter: int = 50) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mpls(y_input, lam=lam, p=p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mor(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mor(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_imor(item: tuple[str, np.ndarray], tol: float = 1e-3, max_iter: int = 200) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.imor(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mormol(item: tuple[str, np.ndarray], tol: float = 1e-3, max_iter: int = 250) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mormol(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_amormol(item: tuple[str, np.ndarray], tol: float = 1e-3, max_iter: int = 250) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.amormol(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_rolling_ball(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.rolling_ball(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mwmv(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mwmv(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_tophat(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.tophat(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mpspline(item: tuple[str, np.ndarray], lam: int = 1e4, p: float = 0.0, spline_degree: int = 3) -> tuple[
    str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mpspline(y_input, lam=lam, p=p, spline_degree=spline_degree)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_jbcd(item: tuple[str, np.ndarray], tol: float = 1e-2, max_iter: int = 20) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.jbcd(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_mixture_model(item: tuple[str, np.ndarray], lam: int = 1e5, p: float = 1e-2, spline_degree: int = 3,
                           tol: float = 1e-3, max_iter: int = 50) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mixture_model(y_input, lam, p, spline_degree=spline_degree, max_iter=max_iter,
                                                    tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_irsqr(item: tuple[str, np.ndarray], lam: int = 100, quantile: float = .01, spline_degree: int = 3,
                   max_iter: int = 100) -> tuple[
    str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.irsqr(y_input, lam, quantile, spline_degree=spline_degree, max_iter=max_iter)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_corner_cutting(item: tuple[str, np.ndarray], max_iter: int = 100) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.corner_cutting(y_input, max_iter)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_noise_median(item: tuple[str, np.ndarray], half_window: int) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.noise_median(y_input, half_window)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_snip(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.snip(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_swima(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.swima(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_ipsa(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.ipsa(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_ria(item: tuple[str, np.ndarray], tol: float = 1e-6) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.ria(y_input, tol=tol)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_dietrich(item: tuple[str, np.ndarray], num_std: float = 3., poly_order: int = 5, tol: float = 1e-3,
                      max_iter: int = 50, interp_half_window: int = 5, min_length: int = 2) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.dietrich(y_input, num_std=num_std, poly_order=poly_order, tol=tol,
                                               max_iter=max_iter, interp_half_window=interp_half_window,
                                               min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_golotvin(item: tuple[str, np.ndarray], num_std: float = 3., interp_half_window: int = 5,
                      min_length: int = 2, sections: int = 32) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.golotvin(y_input, num_std=num_std, interp_half_window=interp_half_window,
                                               sections=sections, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_std_distribution(item: tuple[str, np.ndarray], num_std: float = 3., interp_half_window: int = 5,
                              fill_half_window: int = 3) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.std_distribution(y_input, num_std=num_std,
                                                       interp_half_window=interp_half_window,
                                                       fill_half_window=fill_half_window)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_fastchrom(item: tuple[str, np.ndarray], interp_half_window: int = 5, max_iter: int = 100,
                       min_length: int = 2) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.fastchrom(y_input, interp_half_window=interp_half_window,
                                                max_iter=max_iter, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_fabc(item: tuple[str, np.ndarray], num_std: float = 3., lam: int = 1e6, min_length: int = 2) \
        -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.fabc(y_input, lam, num_std=num_std, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_optimize_extended_range(item: tuple[str, np.ndarray], method: str) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.optimize_extended_range(y_input, method=method)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_adaptive_minmax(item: tuple[str, np.ndarray], method: str) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.adaptive_minmax(y_input, method=method)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_beads(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.beads(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T
