# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides various signal smoothing functions for spectral data, including Savitzky-Golay,
Whittaker,
median filtering, Wiener filtering, and Empirical Mode Decomposition methods.

Functions
---------
smooth_savgol(item, window_length, polyorder)
    Applies Savitzky-Golay smoothing to the input spectrum.

whittaker(item, lam)
    Applies Whittaker smoothing to the input spectrum.

smooth_flat(item, window_len)
    Applies a flat window smoothing to the input spectrum.

smooth_window(item, window_len, method)
    Applies a window-based smoothing (Hanning, Hamming, Bartlett, or Blackman) to the input spectrum

smooth_window_kaiser(item, window_len, beta)
    Applies Kaiser window smoothing to the input spectrum.

smooth_med_filt(item, window_len)
    Applies median filtering to the input spectrum.

smooth_wiener(item, window_len)
    Applies Wiener filtering to the input spectrum.

smooth_emd(item, noise_first_imfs)
    Applies Empirical Mode Decomposition (EMD) smoothing to the input spectrum.

smooth_eemd(item, noise_first_imfs, trials)
    Applies Ensemble Empirical Mode Decomposition (EEMD) smoothing to the input spectrum.

smooth_ceemdan(item, noise_first_imfs, trials)
    Applies Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) smoothing
    to the input spectrum.

smooth_mlesg(item, distance, sigma)
    Applies Maximum Likelihood Estimation Signal Smoothing (MLESG) to the input spectrum.
"""

import numpy as np
from PyEMD.CEEMDAN import CEEMDAN
from PyEMD.EEMD import EEMD
from PyEMD.EMD import EMD
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyval
from scipy.signal import savgol_filter, medfilt, wiener, detrend, find_peaks
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import spsolve

from src.stages.preprocessing.functions.smoothing.mlesg import mlesg
from src.data.work_with_arrays import nearest_idx


def smooth_savgol(item: tuple[str, np.ndarray], window_length: int, polyorder: int) \
        -> tuple[str, np.ndarray]:
    """
    Applies Savitzky-Golay smoothing to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    window_length : int
        The length of the filter window (number of coefficients).
    polyorder : int
        The order of the polynomial used to fit the samples.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.
    """
    array = item[1]
    y_axis_smooth = savgol_filter(array[:, 1], window_length, polyorder)
    return item[0], np.vstack((array[:, 0], y_axis_smooth)).T


def whittaker(item: tuple[str, np.ndarray], lam: int):
    """
    Applies Whittaker smoothing to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    lam : int
        The smoothing coefficient lambda.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
    P. H. C. Eilers, A Perfect Smoother. Anal. Chem. 75, 3631–3636 (2003).
    """
    array = item[1]
    y_axis = array[:, 1]
    # starting the algorithm
    y_shape = y_axis.shape[0]
    eye = np.eye(y_shape)
    diff_eye = np.diff(eye, 2)
    d = csc_matrix(diff_eye)
    w = np.ones(y_shape)
    w = spdiags(w, 0, y_shape, y_shape)
    dt = d.transpose()
    ddot = d.dot(dt)
    z = w + lam * ddot
    z = spsolve(z, y_axis)
    return item[0], np.vstack((array[:, 0], z)).T


def smooth_flat(item: tuple[str, np.ndarray], window_len: int) -> tuple[str, np.ndarray]:
    """
    Applies a flat window smoothing to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    window_len : int
        The length of the smoothing window.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=smooth
    """
    array = item[1]
    w = np.ones(window_len, 'd')
    y_filt = np.convolve(w / w.sum(), array[:, 1], mode='same')
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_window(item: tuple[str, np.ndarray], window_len: int, method: str) \
        -> tuple[str, np.ndarray]:
    """
    Applies a window-based smoothing (Hanning, Hamming, Bartlett, or Blackman) to the input
    spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    window_len : int
        The length of the smoothing window.
    method : str
        The window method to use ('hanning', 'hamming', 'bartlett', or 'blackman').

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=smooth
    """
    array = item[1]
    func = None
    match method:
        case 'hanning':
            func = np.hanning
        case 'hamming':
            func = np.hamming
        case 'bartlett':
            func = np.bartlett
        case 'blackman':
            func = np.blackman
    w = func(window_len)
    y_filt = np.convolve(w / w.sum(), array[:, 1], mode='same')
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_window_kaiser(item: tuple[str, np.ndarray], window_len: int, beta: float) \
        -> tuple[str, np.ndarray]:
    """
    Applies Kaiser window smoothing to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    window_len : int
        The length of the smoothing window.
    beta : float
        The beta parameter for the Kaiser window.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.
    """
    array = item[1]
    w = np.kaiser(window_len, beta)
    y_filt = np.convolve(w / w.sum(), array[:, 1], mode='same')
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_med_filt(item: tuple[str, np.ndarray], window_len: int) -> tuple[str, np.ndarray]:
    """
    Applies median filtering to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    window_len : int
        The length of the smoothing window.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
    docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html#scipy.signal.medfilt
    """
    # scipy median filter, from
    #
    array = item[1]
    if window_len % 2 == 0:
        window_len += 1
    y_filt = medfilt(array[:, 1], kernel_size=window_len)
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_wiener(item: tuple[str, np.ndarray], window_len: int) -> tuple[str, np.ndarray]:
    """
    Applies Wiener filtering to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    window_len : int
        The length of the smoothing window.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
    docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html#scipy.signal.wiener
    """
    array = item[1]
    y_filt = wiener(array[:, 1], mysize=window_len)
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_emd(item: tuple[str, np.ndarray], noise_first_imfs: int) -> tuple[str, np.ndarray]:
    """
    Applies Empirical Mode Decomposition (EMD) smoothing to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    noise_first_imfs : int
        Number of the first Intrinsic Mode Functions (IMFs) to treat as noise.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
        https://pyemd.readthedocs.io/en/latest/emd.html
    """
    x_axis, y_axis = item[1][:, 0], item[1][:, 1]
    emd = EMD(spline_kind='akima', DTYPE=np.float16)
    emd.emd(y_axis, x_axis, max_imf=noise_first_imfs)
    imfs, _ = emd.get_imfs_and_residue()

    for i in range(noise_first_imfs):
        y_axis = np.subtract(y_axis, imfs[i])
    return item[0], np.vstack((x_axis, y_axis)).T


def smooth_eemd(item: tuple[str, np.ndarray], noise_first_imfs: int, trials: int) \
        -> tuple[str, np.ndarray]:
    """
    Applies Ensemble Empirical Mode Decomposition (EEMD) smoothing to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    noise_first_imfs : int
        Number of the first Intrinsic Mode Functions (IMFs) to treat as noise.
    trials : int
        Number of EEMD trials to perform.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
        https://pyemd.readthedocs.io/en/latest/eemd.html
    """
    x_axis, y_axis = item[1][:, 0], item[1][:, 1]
    x_max = abs(max(x_axis))
    x_range = x_max - abs(min(x_axis))
    noise_width = x_range * 3 / (x_axis.shape[0] - 1) ** 2
    eemd = EEMD(trials=trials, DTYPE=np.float16, spline_kind='akima', noise_width=noise_width,
                noise_kind='uniform')
    eemd.noise_seed(481516234)
    e_imfs = eemd.eemd(y_axis, x_axis, max_imf=noise_first_imfs)
    for i in range(noise_first_imfs):
        y_axis = np.subtract(y_axis, e_imfs[i])
    return item[0], np.vstack((x_axis, y_axis)).T


def smooth_ceemdan(item: tuple[str, np.ndarray], noise_first_imfs: int, trials: int) \
        -> tuple[str, np.ndarray]:
    """
    Applies Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) smoothing
    to the input spectrum.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the key (filename) and the input spectrum array.
    noise_first_imfs : int
        Number of the first Intrinsic Mode Functions (IMFs) to treat as noise.
    trials : int
        Number of CEEMDAN trials to perform.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
        https://pyemd.readthedocs.io/en/latest/ceemdan.html
    """
    x_axis, y_axis = item[1][:, 0], item[1][:, 1]
    # Prepare and run CEEMDAN
    x_range = abs(max(x_axis)) - abs(min(x_axis))
    epsilon = x_range * 10 / (x_axis.shape[0] - 1) ** 2

    ceemdan = CEEMDAN(trials=trials, spline_kind='akima', DTYPE=np.float16, noise_kind='uniform',
                      epsilon=epsilon)
    ceemdan.noise_seed(481516234)
    c_imfs = ceemdan(y_axis, x_axis, max_imf=noise_first_imfs)

    for i in range(noise_first_imfs):
        y_axis = np.subtract(y_axis, c_imfs[i])
    return item[0], np.vstack((x_axis, y_axis)).T


def smooth_mlesg(item: tuple[str, tuple[np.ndarray, float]], distance: float, sigma: int) \
        -> tuple[str, np.ndarray]:
    """
    Applies Maximum Likelihood Estimation Signal Smoothing (MLESG) to the input spectrum.

    Parameters
    ----------
    item : tuple[str, tuple[np.ndarray, float]]
        A tuple containing the key (filename) and a tuple with the input spectrum array and SNR.
    distance : float
        The distance parameter for peak finding.
    sigma : int
        The sigma parameter for Gaussian smoothing.

    Returns
    -------
    tuple[str, np.ndarray]
        The key and the smoothed spectrum.

    References
    ----------
        https://github.com/Trel725/RamanDenoising
    """
    x_axis, y_axis = item[1][0][:, 0], item[1][0][:, 1]
    snr = item[1][1]
    distance /= (max(x_axis) - min(x_axis)) / x_axis.shape[0]
    y_d = detrend(y_axis)
    polynome = polynomial.polyfit(x_axis, y_d, 9)
    pv = polyval(x_axis, polynome)
    y_d = np.subtract(y_d, pv)
    y_min = min(y_d)
    y_max = max(y_d)
    y_d = np.divide(np.subtract(y_d, y_min), y_max - y_min)
    peaks, _ = find_peaks(y_d, distance=1.5 * distance, width=(distance, 5 * distance),
                                   height=(1 - np.std(y_d) * 2, 1))
    if not peaks.any():
        peaks = [nearest_idx(y_d, y_max)]
    filtered = mlesg(y_axis, x_axis, peaks, snr=snr, g_sigma=sigma)
    return item[0], np.vstack((x_axis, filtered)).T
