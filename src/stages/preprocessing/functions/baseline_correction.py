# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides various functions for performing baseline correction on spectral data using
different polynomial fitting techniques. The primary goal is to accurately remove the baseline from
the Raman spectra, enhancing the quality of the resulting data for further analysis.

Each function returns the corrected baseline and the baseline-corrected spectrum.
"""

import numpy as np
from numba import njit
from numpy.linalg import norm, lstsq
from numpy.polynomial.polynomial import polyvander
from numpy.polynomial.polyutils import getdomain, mapdomain
from pybaselines import Baseline
from scipy.signal import detrend
from scipy.sparse import csc_matrix, spdiags, diags, eye as sparse_eye
from scipy.sparse.linalg import spsolve

from src.data.work_with_arrays import extend_bool_mask_two_sided


def baseline_correct(item: tuple[str, np.ndarray], **kwargs) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Computes baseline corrected spectrum using Baseline library.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input array of the spectrum.
    kwargs : dict
        Additional parameters for baseline correction.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
        The key, the fitted baseline, and the baseline-corrected spectrum.
    """
    method = kwargs.pop('method')
    x_input = item[1][:, 0]
    y_input = item[1][:, 1]
    bf = Baseline(x_data=x_input)
    method_func = {'Poly': bf.poly, 'ModPoly': bf.modpoly, 'iModPoly': bf.imodpoly,
                   'Penalized poly': bf.penalized_poly, 'Quantile regression': bf.quant_reg,
                   'Goldindec': bf.goldindec, 'iAsLS': bf.iasls, 'iarPLS': bf.iarpls,
                   'asPLS': bf.aspls, 'psaLSA': bf.psalsa, 'DerPSALSA': bf.derpsalsa,
                   'MPLS': bf.mpls, 'Morphological': bf.mor, 'iMor': bf.imor, 'MorMol': bf.mormol,
                   'AMorMol': bf.amormol, 'Rolling Ball': bf.rolling_ball, 'MWMV': bf.mwmv,
                   'Top-hat': bf.tophat, 'MPSpline': bf.mpspline, 'JBCD': bf.jbcd,
                   'Mixture Model': bf.mixture_model, 'IRSQR': bf.irsqr,
                   'Corner-Cutting': bf.corner_cutting, 'Noise Median': bf.noise_median,
                   'SNIP': bf.snip, 'IPSA': bf.ipsa, 'SWiMA': bf.swima, 'RIA': bf.ria,
                   'Dietrich': bf.dietrich, 'Golotvin': bf.golotvin,
                   'Std Distribution': bf.std_distribution, 'FastChrom': bf.fastchrom,
                   'FABC': bf.fabc, 'BEaDS': bf.beads}
    func = method_func[method]
    baseline_fitted = func(y_input, **kwargs)[0]
    return (item[0], np.vstack((x_input, baseline_fitted)).T,
            np.vstack((x_input, y_input - baseline_fitted)).T)


def ex_mod_poly(item: tuple[str, np.ndarray], **kwargs) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Ex-ModPoly is extension of ModPoly method from paper [1] and Quantile regression using
    iteratively reweighed the least squares (IRLS) [2]

    Returns baseline and raman spectrum as 2D ndarray.

    Main differences from ModPoly, I-ModPoly and Quantile regression are:
        1) The presence of absorption lines in the spectrum is taken into account
        2) The accuracy of polynomial approximation for a descending baseline as in the spectra of
            biomaterials has
            been increased by scaling the IRLS weights defined as in the article [3] and rebuilding
            y signal at each iteration.
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
    kwargs:
        poly_order: float, optional
            The polynomial order for fitting the baseline. Recommended value from 5 to 11.
        tol: float, optional
            The exit criteria. Default is 1e-6.
        max_iter : int, optional
            The maximum number of iterations. Default is 100.
        quantile : float, optional
            The quantile at which to fit the baseline. Default is 1e-4.
        w_scale_factor: float, optional
            Scales quantile regression weights by formula: q = q**w_scale_factor. Where q
            calculated like in quantile regression [3]
            Low values like 1e-10 makes weights like rectangular shape (0 for negative residual, 1
            for positive).
            0.5 is standard value makes result equal to quantile regression.
            Values lower than 0.5 allows make baseline lower
            For SNV normalization recommended range [0.01 - 0.1] with recalc_y = True
            For EMSC normalization recommended range [0.01 - 0.5] with recalc_y = False
        recalc_y: bool, optional
            Rebuild y in at every iteration by formula: y = np.minimum(y, baseline)
            True makes result like using ModPoly-IModPoly
            False makes result like using Quantile regression. Default is False.
        num_std : float, optional
            For absorption removing. Consider signal lower than
            (baseline - num_std * np.std(detrended(y))) as strong absorption.
            Higher values for stronger absorption lines. Lower value for weak absorption lines.
            If num_std = 0.0 absorption signal will not be removed.
            Default is 3.
        window_size : int, optional
            For absorption removing.
            How many points around found absorption signal will be considered as absorption too.
            window_size = 2, makes mask of absorption indexes like
            [0, 0, 1, 0, 0] --> [0, 1, 1, 1, 0]
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

        Performs quantile regression using iteratively reweighted the least squares (IRLS)
        as described in [3]_.

     References
    ----------
    . [1] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
    . [2] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.
    . [3] Schnabel, S., et al. Simultaneous estimation of quantile curves using
            quantile sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.
    """
    x, y = item[1][:, 0].copy(), item[1][:, 1].copy()
    w = np.ones_like(y)
    mapped_x = mapdomain(x, getdomain(x), np.array([-1., 1.]))
    vandermonde = polyvander(mapped_x, kwargs['poly_order'])
    # 1st baseline fit without weights
    baseline = vandermonde @ lstsq(vandermonde * w[:, None], y * w, None)[0]
    # Find indexes of strong absorption lines to remove
    indexes = absorption_indexes(y, baseline, kwargs['num_std'], kwargs['window_size']) \
        if kwargs['num_std'] != 0.0 else None
    for i in range(kwargs['max_iter']):
        baseline_old = baseline
        baseline = vandermonde @ lstsq(vandermonde * w[:, None], y * w, None)[0]
        w = _quantile(y, baseline, kwargs['quantile'], kwargs['w_scale_factor'], indexes)
        if i == 0:
            continue
        if kwargs['recalc_y']:
            y = np.minimum(y, baseline)
        if relative_difference(baseline, baseline_old) < kwargs['tol']:
            break
    return (item[0], np.vstack((item[1][:, 0], baseline)).T,
            np.vstack((item[1][:, 0], item[1][:, 1] - baseline)).T)


def absorption_indexes(y: np.ndarray, baseline: np.ndarray, sd_factor: float = 3., window_size=2) \
        -> np.ndarray:
    """
    Find indexes of strong absorption lines. Consider signal lower than
    (baseline - sd_factor * np.std(detrended(y))) as strong absorption.

    Parameters
    ----------
    y : numpy.ndarray
       The values of the raw data.
    baseline : numpy.ndarray
       1st fitted baseline without weights.
    sd_factor : float, optional
        For absorption removing. Consider signal lower than
        (baseline - num_std * np.std(detrended(y))) as strong absorption.
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
        Indices of strong absorption lines.
    """
    detrended = detrend(y)
    sd = np.std(detrended)
    cond = y < (baseline - sd_factor * sd)
    cond = extend_bool_mask_two_sided(cond.tolist(), window_size)
    idx = np.argwhere(cond).T[0]
    return idx


@njit(fastmath=True)
def _quantile(y: np.ndarray, fit: np.ndarray, quantile: float = 1e-4,
              w_scale_factor: float = .5, absorption_idx=None) -> np.ndarray:
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
    w_scale_factor : float
        scaling coefficient
    absorption_idx: numpy.ndarray
        indexes of absorption lines to be zero.

    Returns
    -------
    numpy.ndarray
        The calculated loss, which can be used as weighting when performing iteratively
        reweighted the least squares (IRLS)

    References
    ----------
    Schnabel, S., et al. Simultaneous estimation of quantile curves using quantile
    sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

    """
    # eps is a small value added to the square of `residual` to prevent dividing by 0.
    # 1e-6 seems to work better than the 1e-4 in Schnabel, et al.
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


@njit(fastmath=True)
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
    numerator = np.linalg.norm(new - old)
    denominator = np.maximum(np.linalg.norm(old, norm_order), 1e-20)
    return numerator / denominator


def baseline_asls(item: tuple[str, np.ndarray], lam: int = 1e6, p: float = 1e-3,
                  max_iter: int = 50) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Applies Asymmetric Least Squares (ALS) baseline correction.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input array of the spectrum.
    lam : float, optional
        Smoothing parameter. Default is 1e6.
    p : float, optional
        Asymmetry parameter. Default is 0.001.
    max_iter : int, optional
        Number of iterations. Default is 50.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
        The key, the fitted baseline, and the baseline-corrected spectrum.

    References
    ----------
    Eilers, P.H.C., et al. Baseline correction with asymmetric least squares smoothing. Leiden
    University Medical Centre Report, 2005.
    """
    # Matlab code in Eilers et Boelens 2005
    # Python adaptation found on stackoverflow:
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    x_input = item[1][:, 0]
    y_input = item[1][:, 1]
    z = np.zeros_like(y_input)
    # starting the algorithm
    y_shape = y_input.shape[0]
    d = csc_matrix(np.diff(np.eye(y_shape), 2))
    w = np.ones(y_shape)
    for _ in range(max_iter):
        w_w = spdiags(w, 0, y_shape, y_shape)
        z_z = w_w + lam * d.dot(d.transpose())
        z = spsolve(z_z, w * y_input)
        w = p * (y_input > z) + (1 - p) * (y_input < z)

    baseline_fitted = z
    y_new = y_input - baseline_fitted
    return item[0], np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_arpls(item: tuple[str, np.ndarray], lam: int = 1e5, p: float = 1e-6,
                   max_iter: int = 50) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Adaptive and Robust Polynomial Smoothing (arPLS) baseline correction.
    Adaptation of the Matlab code in Baek et al. 2015 DOI: 10.1039/c4an01061b

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input array of the spectrum.
    lam : float, optional
        Smoothing parameter. Default is 1e5.
    p : float, optional
        Convergence ratio. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 50.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
        The key, the fitted baseline, and the baseline-corrected spectrum.

    Notes
    -----
    The arPLS method adapts and robustly fits a polynomial to the baseline,
    iteratively refining the baseline to improve accuracy and handle outliers.
    """
    n = item[1][:, 1].shape[0]
    d_d = csc_matrix(np.diff(np.eye(n), 2))
    w = np.ones(n)
    i = 0
    while True:
        i += 1
        w_w = spdiags(w, 0, n, n)
        z_z = w_w + lam * d_d.dot(d_d.transpose())
        z = spsolve(z_z, w * item[1][:, 1])
        d = item[1][:, 1] - z
        # make d- and get w^t with m and s
        dn = d[d < 0]
        s = np.std(dn)
        wt = 1.0 / (1 + np.exp(2 * (d - (2 * s - np.mean(dn))) / s))
        # check exit condition and backup
        if norm(w - wt) / norm(w) < p:
            break
        if i >= max_iter:
            break
        w = wt
    return item[0], np.vstack((item[1][:, 0], z)).T, np.vstack((item[1][:, 0], item[1][:, 1] - z)).T


def _whittaker_smooth(x: np.ndarray, w: np.ndarray, lambda_: int) -> np.ndarray:
    """
    Penalized the least squares algorithm for background fitting

    Parameters
    ----------
    x: np.ndarray
        input data (i.e. chromatogram of spectrum)
    w: np.ndarray
        binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
    lambda_: int
        parameter that can be adjusted by user. The larger lambda is,  the smoother the
        resulting background

    Returns
    -------
    np.ndarray
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


def baseline_airpls(item: tuple[str, np.ndarray], lam: int = 1e6, p: float = 1e-6,
                    max_iter: int = 50) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Asymmetric Least Squares Smoothing with AirPLS baseline correction.
    Implementation of Zhang fit for Adaptive iteratively reweighed penalized the least squares for
    baseline fitting.
    Modified from Original implementation by Professor Zhimin Zhang at
    https://github.com/zmzhang/airPLS/
    https://doi.org/10.1039/B922045C

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input array of the spectrum.
    lam : float, optional
        Smoothing parameter. Default is 1e6.
    p : int, optional
        Order of the difference penalty. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 50.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
        The key, the fitted baseline, and the baseline-corrected spectrum.

    Notes
    -----
    The airPLS method automatically finds the baseline of a given spectrum by iteratively
    refining the fit, optimizing the baseline correction for asymmetric peaks and noise reduction.
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
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in
        # order to ignore it
        w[d < 0] = np.exp(i * abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]

    y_new = y_input - baseline_fitted
    return key, np.vstack((x_input, baseline_fitted)).T, np.vstack((x_input, y_new)).T


def baseline_drpls(item: tuple[str, np.ndarray], **kwargs) \
        -> tuple[str, np.ndarray, np.ndarray]:
    """
    According to Applied Optics, 2019, 58, 3913-3920. https://doi.org/10.1364/AO.58.003913

    Parameters
    ----------
    item : tuple[str, np.ndarray]
      A tuple containing the key (filename) and the input array of the spectrum.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
      The key, the fitted baseline, and the baseline-corrected spectrum.
    """
    y_input = item[1][:, 1]

    d_d = diags([1, -2, 1], [0, -1, -2], shape=(y_input.shape[0], y_input.shape[0] - 2),
                format='csr')
    d_d = d_d.dot(d_d.transpose())
    d_1 = diags([-1, 1], [0, -1], shape=(y_input.shape[0], y_input.shape[0] - 1), format='csr')
    d_1 = d_1.dot(d_1.transpose())

    w = w_0 = z_z = np.ones(y_input.shape[0])
    i_n = diags(w_0, format='csr')

    # this is the code for the fitting procedure
    w_w = diags(w, format='csr')

    for jj in range(int(100 if kwargs['max_iter'] > 100 else kwargs['max_iter'])):
        w_w.setdiag(w)
        z_prev = z_z
        z_z = spsolve(w_w + d_1 + kwargs['lam'] * (i_n - kwargs['eta'] * w_w) * d_d, w_w * y_input,
                      permc_spec='NATURAL')
        if np.linalg.norm(z_z - z_prev) > kwargs['p']:
            d = y_input - z_z
            sigma_negative = np.std(d[d < 0])
            mean_negative = np.mean(d[d < 0])
            w = 0.5 * (1 - np.exp(jj) * (d - (-mean_negative + 2 *
                                              sigma_negative)) / sigma_negative / (
                    1 + abs(np.exp(jj) *
                            (d - (-mean_negative + 2 * sigma_negative)) / sigma_negative)))
        else:
            break
    return item[0], np.vstack((item[1][:, 0], z_z)).T, np.vstack((item[1][:, 0], y_input - z_z)).T
