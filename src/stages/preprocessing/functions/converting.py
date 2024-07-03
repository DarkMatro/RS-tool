"""
Module for converting spectroscopic data.

This module provides functions for converting and processing spectroscopic data.
It includes functions to convert wavelengths from nm to cm, compute the full width
at half maximum (FWHM) of laser peaks, and calculate the signal-to-noise ratio (SNR)
of the processed data.

Functions:
    convert(item, near_idx, laser_nm, max_ccd_value): Converts and processes spectroscopic data.
    get_laser_peak_fwhm(array, laser_wl, min_nm, max_nm): Computes the FWHM of laser peaks.
    _convert_nm_to_cm(x_axis, base_wave_length, laser_nm): Converts wavelengths from nm to cm.
    _get_args_by_value(array, value): Helper function to get indices of elements equal to a specific
        value.
"""

import numpy as np
from numba import njit
from numpy import ndarray
from scipy.signal import peak_widths, peak_prominences, savgol_filter

from src.data.work_with_arrays import nearest_idx


def convert(item: tuple[str, np.ndarray], near_idx: int, laser_nm: float = 784.5,
            max_ccd_value: float = 65536.0) -> tuple[str, ndarray, float, float, float]:
    """
    Converts and processes spectroscopic data.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing a string identifier and a 2D NumPy array with wavelength and intensity
        data.
    near_idx : int
        The index to cut off the data array.
    laser_nm : float, optional
        The laser wavelength in nm. Default is 784.5 nm.
    max_ccd_value : float, optional
        The maximum CCD value to consider for peak detection. Default is 65536.0.

    Returns
    -------
    tuple
        A tuple containing the string identifier, the processed 2D array with converted wavelengths
        and intensities, the base wavelength, the FWHM of the laser peak, and the signal-to-noise
        ratio.
    """
    array = item[1]
    y_axis = array[:, 1][:near_idx]
    if np.amax(y_axis) >= max_ccd_value:
        args = _get_args_by_value(y_axis, max_ccd_value)
        local_maxima = int(np.mean(args)) if args else np.argmax(y_axis)
    else:
        local_maxima = np.argmax(y_axis)
    base_wave_length: float = array[local_maxima][0]
    new_x_axis = _convert_nm_to_cm(array[:, 0], base_wave_length, laser_nm)
    n_array = np.vstack((new_x_axis, array[:, 1])).T
    fwhm_cm = get_laser_peak_fwhm(n_array, 0, np.min(new_x_axis), np.max(new_x_axis))
    y_axis = n_array[:, 1]
    filtered = savgol_filter(y_axis, 9, 3)
    noise = y_axis - filtered
    snr: float = y_axis.max() / noise.std()
    return item[0], n_array, base_wave_length, fwhm_cm, round(snr, 2)


def get_laser_peak_fwhm(array: np.ndarray, laser_wl: float, min_nm: float,
                        max_nm: float) -> float:
    """
    Computes the full width at half maximum (FWHM) of laser peaks.

    Parameters
    ----------
    array : np.ndarray
        A 2D NumPy array with wavelength and intensity data.
    laser_wl : float
        The wavelength of the laser peak.
    min_nm : float
        The minimum wavelength in nm to consider in the array.
    max_nm : float
        The maximum wavelength in nm to consider in the array.

    Returns
    -------
    float
        The FWHM of the laser peak.
    """
    # get +-5 nm range of array from laser peak
    diff = 40 if laser_wl == 0 else 5
    left_nm = laser_wl - diff
    right_nm = laser_wl + diff
    dx = (max_nm - min_nm) / array.shape[0]
    left_idx = nearest_idx(array[:, 0], left_nm)
    right_idx = nearest_idx(array[:, 0], right_nm)
    # fit peak and find fwhm
    y_axis = array[left_idx:right_idx][:, 1]
    peaks = [nearest_idx(y_axis, np.max(y_axis))]
    prominences = peak_prominences(y_axis, peaks)
    results_half = peak_widths(y_axis, peaks, prominence_data=prominences)
    fwhm: float = results_half[0][0] * dx
    return np.round(fwhm, 5)


@njit(fastmath=True)
def _convert_nm_to_cm(x_axis: np.ndarray, base_wave_length: float, laser_nm: float) \
        -> np.ndarray:
    """
    Converts wavelengths from nm to cm.

    Parameters
    ----------
    x_axis : np.ndarray
        The array of wavelengths in nm.
    base_wave_length : np.ndarray
        The base wavelength in nm.
    laser_nm : float
        The laser wavelength in nm.

    Returns
    -------
    np.ndarray
        The array of wavelengths converted to cm.
    """
    return (1e7 / laser_nm) - (1e7 / (x_axis + laser_nm - base_wave_length))


@njit(fastmath=True)
def _get_args_by_value(array: np.ndarray, value: float) -> list[float]:
    """
    Helper function to get indices of elements equal to a specific value.

    Parameters
    ----------
    array : np.ndarray
        The array to search through.
    value : float
        The value to find in the array.

    Returns
    -------
    list of float
        A list of indices where the array elements are equal to the specified value.
    """
    return_list = []
    prev_value = float(0)
    for i, v in enumerate(array):
        if v == value:
            return_list.append(i)
            prev_value = v
        elif prev_value == value:
            break
    return return_list
