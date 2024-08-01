# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functions for Maximum Likelihood Estimation of Signal (MLESG)
in Raman spectroscopy, including core computation and utility functions.

Functions
---------
mlesg(y_axis, x_axis, peaks_idx, snr=None, g_sigma=5)
    Performs MLESG to estimate the signal.

_get_minmax(snr)
    Determines the minimum and maximum values of m based on SNR.

_calculate_m(y_axis, x_axis, peaks_idx, g_sigma, snr=None)
    Calculates the m parameter for MLESG.

_mlesg_core(y_axis, m)
    Core function for MLESG calculation.

_main_job(x, n, m, j, xe, x_dash, sigma, p, lmbd)
    Performs the main iterative computation for MLESG.
"""

import numpy as np
from numba import njit
from scipy.signal import savgol_filter, detrend

const1 = np.array([-0.00000000000020511637148964194179,
                   0.00000000017456738606819003079,
                   -0.00000006108370350381461555,
                   0.000011326343084825261969,
                   -0.001196766893248055099,
                   0.072278932710701945807,
                   -2.3738144356931862866,
                   36.543962848162728108])

const2 = np.array([0.000000049662053296461794872,
                   -0.000045735657830153646274,
                   0.013867582814278392803,
                   -1.7581457324626106331,
                   88.680082559339525284])


def mlesg(y_axis: np.ndarray, x_axis: np.ndarray, peaks_idx: list[int], snr: float = None,
          g_sigma: int = 5) -> np.ndarray:
    """
    Performs Maximum Likelihood Estimation of Signal (MLES) to estimate the signal from noisy data.

    Parameters
    ----------
    y_axis : np.ndarray
        The intensity values of the spectrum.
    x_axis : np.ndarray
        The wave number values of the spectrum.
    peaks_idx : list[int]
        Indices of the peaks in the spectrum.
    snr : float, optional
        Signal-to-noise ratio. If not provided, it is estimated from the data. Default is None.
    g_sigma : int, optional
        Gaussian sigma value for the smoothing function. Default is 5.

    Returns
    -------
    np.ndarray
        The estimated signal.
    """
    m = _calculate_m(y_axis, x_axis, peaks_idx, g_sigma, snr)
    xe = _mlesg_core(y_axis, m)
    return xe


def _get_minmax(snr: float) -> tuple[int, int]:
    """
    Determines the minimum and maximum values of m based on the signal-to-noise ratio (SNR).

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio.

    Returns
    -------
    tuple[int, int]
        Minimum and maximum values of m.
    """
    if snr <= 200:
        min_m = np.round(np.polyval(const1, snr))
        max_m = np.round(np.polyval(const2, snr))
    else:
        min_m = 2
        max_m = 5
    return min_m, max_m


def _calculate_m(y_axis: np.ndarray, x_axis: np.ndarray, peaks_idx: list[int], g_sigma: int,
                 snr=None) -> np.ndarray:
    """
    Calculates the m parameter for MLES based on the provided spectrum data and peaks.

    Parameters
    ----------
    y_axis : np.ndarray
        The intensity values of the spectrum.
    x_axis : np.ndarray
        The wavenumber values of the spectrum.
    peaks_idx : list[int]
        Indices of the peaks in the spectrum.
    g_sigma : int
        Gaussian sigma value for the smoothing function.
    snr : float, optional
        Signal-to-noise ratio. If not provided, it is estimated from the data. Default is None.

    Returns
    -------
    np.ndarray
        The calculated m parameter.
    """
    g_sigma *= x_axis[1] - x_axis[0]
    mu = 0
    y_axis = detrend(y_axis)
    if snr is not None:
        min_m, max_m = _get_minmax(snr)
    else:
        noise = y_axis - mu - savgol_filter(y_axis - mu, 9, 3)
        snr = (y_axis - mu).max() / noise.std()
        min_m, max_m = _get_minmax(snr)

    g = np.zeros((len(peaks_idx), x_axis.shape[0]))

    for i, peak_idx in enumerate(peaks_idx):
        nom = -(x_axis - x_axis[peak_idx]) ** 2
        g[i, :] = 1 - np.exp(nom / 2 * g_sigma * g_sigma)
    result = g.min(axis=0) * (max_m - min_m) + min_m
    return np.round(result).astype(int)


def _mlesg_core(y_axis: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Core function for MLES calculation, iteratively refining the estimated signal.

    Parameters
    ----------
    y_axis : np.ndarray
        The intensity values of the spectrum.
    m : np.ndarray
        The m parameter used in the MLES calculation.

    Returns
    -------
    np.ndarray
        The refined estimated signal.
    """
    mu, p, lmbd, q, v = 0, 0.4, 1.8, 7, 5
    x = y_axis - mu
    sigma = (x - savgol_filter(x, 9, 3)).std()
    xe = x.copy()
    for j in range(m.max()):
        if j < min(m):
            v, q = 5, 7
        elif j >= m.max() - round(m.max() / 5):
            q += 4
            lmbd *= 10
        else:
            v, q = 3, 5
        x_dash = savgol_filter(xe, q, v)
        xe = _main_job(x, x.size, m, j, xe, x_dash, sigma, p, lmbd)
    return xe


# pylint: disable=too-many-arguments
# because using njit

@njit(cache=True, fastmath=True)
def _main_job(x: np.ndarray, n: int, m: np.ndarray, j: int, xe: list[float], x_dash: np.ndarray,
              sigma: float, p: float, lmbd: float) -> list[float]:
    """
    Performs the main iterative computation for MLES.

    Parameters
    ----------
    x : np.ndarray
        The initial signal values.
    n : int
        The length of the signal.
    m : np.ndarray
        The m parameter used in the MLES calculation.
    j : int
        The current iteration number.
    xe : list[float]
        The estimated signal values.
    x_dash : np.ndarray
        The smoothed signal values.
    sigma : float
        Standard deviation of the noise.
    p : float
        Exponent parameter for the cost function.
    lmbd : float
        Regularization parameter for the cost function.

    Returns
    -------
    list[float]
        The updated estimated signal values.
    """
    for i in range(n):
        if j >= m[i]:
            continue
        mle_range = np.arange(xe[i] - 3 * sigma, xe[i] + 3 * sigma, 0.06 * sigma)
        le_mle_range = np.zeros(mle_range.size)
        for k in range(mle_range.size):
            tmp1 = np.abs(mle_range[k] - x_dash[i]) ** p
            limit2 = (x[i] - mle_range[k]) ** 2 / (2 * sigma * sigma)
            le_mle_range[k] = lmbd * tmp1 + limit2
        xe[i] = mle_range[le_mle_range.argmin()]
    return xe
