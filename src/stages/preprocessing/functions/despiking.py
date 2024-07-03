"""
Module for removing strong spikes in spectral data.

This module provides functionality for detecting and removing strong spikes, known as cosmic spikes,
from spectral data. It includes classes and functions for processing spectral data, fitting peaks,
and subtracting identified spikes.

Classes:
    ExpSpec: Class representing an experimental spectrum.
    SpectralFeature: Abstract class for defining a spectral feature.
    CalcPeak: Class for calculating an asymmetric peak on a grid of wave-numbers.
    DespikeParams: Named tuple to hold parameters for the despiking process.

Functions:
    * despike: Removes strong spikes from a given spectrum. Call this function outside.
    voigt_asym: Computes a pseudo-Voigt profile with asymmetry.
    moving_average_molification: Smoothens a spectrum using a moving average filter.
    _func2min: Function to minimize during peak fitting.
    _fit_single_peak: Fits a single peak in the given spectrum.
    _get_molli_width: Calculates the mollification width.
    _split_spectrum: Splits the spectrum into two parts, excluding the laser peak area.
    _check_stop_criteria: Checks criteria to stop the despiking loop.
    _concatenated_spectrum: Concatenates spectrum arrays for the result.
    _subtract_spectrum_and_peak: Subtracts the identified cosmic spikes from the spectrum.
    _get_cosmic_spikes: Identifies and fits cosmic spikes in the spectrum.
"""

from math import ceil as ceil
from os import environ

import numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.signal import detrend

from src.data.work_with_arrays import nearest_idx


class ExpSpec:
    """
    Class representing an experimental spectrum.

    Attributes
    ----------
    full_x : np.ndarray | list
        Full x-axis data.
    full_y : np.ndarray
        Full y-axis data.
    xrange : tuple[float, float]
        Range of x-axis data.
    x : np.ndarray
        Current x-axis data.
    y : np.ndarray
        Current y-axis data.
    """

    def __init__(self, full_x: np.ndarray | list, full_y: np.ndarray) -> None:
        self.full_x = full_x
        self.full_y = full_y
        self.xrange = (np.min(full_x), np.max(full_x))
        self.x = full_x
        self.y = full_y

    @property
    def working_range(self) -> tuple[float, float]:
        """Get or set the working range of the spectrum."""
        return self.xrange

    @working_range.setter
    def working_range(self, xrange: np.ndarray) -> None:
        self.xrange = (np.maximum(np.min(xrange), np.min(self.full_x)),
                       np.minimum(np.max(xrange), np.max(self.full_x)))
        self.x = self.full_x[np.where(
            np.logical_and(self.full_x >= np.amin(xrange), self.full_x <= np.amax(xrange)))]
        self.y = self.full_y[np.where(
            np.logical_and(self.full_x >= np.amin(xrange), self.full_x <= np.amax(xrange)))]


class SpectralFeature:
    """
    Abstract spectral feature class with no defined x-axis.

    Attributes
    ----------
    specs_array : np.ndarray
        Array of spectral feature parameters:
        0: x0 (default 0)
        1: FWHM (default 1)
        2: asymmetry (default 0)
        3: Gaussian share (default 0, i.e., Lorentzian peak)
        4: voigt amplitude (~area, not height)
        5: Baseline slope (k) for linear baseline
        6: Baseline offset (b) for linear baseline
    """

    def __init__(self) -> None:
        self.specs_array = np.zeros(7, dtype=np.float32)
        self.specs_array[1] = 1  # set default FWHM to 1. Otherwise, we can get division by 0

    @property
    def position(self) -> float:
        """Get or set the position of the spectral feature."""
        return self.specs_array[0]

    @position.setter
    def position(self, position: float) -> None:
        self.specs_array[0] = position

    @property
    def fwhm(self) -> float:
        """Get or set the full width at half maximum (FWHM) of the spectral feature."""
        return self.specs_array[1]

    @fwhm.setter
    def fwhm(self, fwhm: float) -> None:
        self.specs_array[1] = fwhm

    @property
    def asymmetry(self) -> float:
        """Get or set the asymmetry parameter of the spectral feature."""
        return self.specs_array[2]

    @asymmetry.setter
    def asymmetry(self, asymmetry: float) -> None:
        """Get or set the asymmetry of the spectral feature."""
        self.specs_array[2] = asymmetry

    @property
    def gaussian_share(self) -> float:
        """Get or set the Gaussian share of the spectral feature."""
        return self.specs_array[3]

    @gaussian_share.setter
    def gaussian_share(self, gaussian_share: float) -> None:
        self.specs_array[3] = gaussian_share

    @property
    def voigt_amplitude(self) -> float:
        """Get or set the voigt amplitude of the spectral feature."""
        return self.specs_array[4]

    @voigt_amplitude.setter
    def voigt_amplitude(self, voigt_amplitude: float) -> None:
        self.specs_array[4] = voigt_amplitude

    @property
    def bl_slope(self) -> float:
        """Get or set the baseline slope (k) of the spectral feature."""
        return self.specs_array[5]

    @bl_slope.setter
    def bl_slope(self, BL_slope: float) -> None:
        self.specs_array[5] = BL_slope

    @property
    def bl_offset(self) -> float:
        """Get or set the baseline offset (b) of the spectral feature."""
        return self.specs_array[6]

    @bl_offset.setter
    def bl_offset(self, BL_offset: float) -> None:
        self.specs_array[6] = BL_offset


class CalcPeak(SpectralFeature):
    """
    Class for calculating an asymmetric peak on a grid of wave-numbers.

    Attributes
    ----------
    specs_array : np.ndarray
        Array of spectral feature parameters:
        0: x0
        1: FWHM
        2: asymmetry
        3: Gaussian share
        4: voigt amplitude (~area, not height)
        5: Baseline slope (k) for linear baseline
        6: Baseline offset (b) for linear baseline
    """

    def __init__(self, wn=np.linspace(0, 1, 129)):
        super().__init__()
        self.wn = wn
        self.specs_array[0] = (wn[-1] - wn[0]) / 2

    @property
    def peak_area(self):
        """
        Returns Voigt peak area.
        """
        a = 1 - self.specs_array[3]
        b = 1 + 0.69 * self.specs_array[2] ** 2 + 1.35 * self.specs_array[2] ** 4
        c = 1 + 0.67 * self.specs_array[2] ** 2 + 3.43 * self.specs_array[2] ** 4
        peak_area = a * self.specs_array[4] * b + self.specs_array[3] * self.specs_array[4] * c
        return peak_area

    @property
    def peak_height(self):
        """
        Returns Voigt peak height.
        """
        amplitudes_l = self.specs_array[4] * 2 / (np.pi * self.specs_array[0])
        amplitudes_g = self.specs_array[4] * (4 * np.log(2) / np.pi) ** 0.5 / self.specs_array[0]
        peak_height = self.specs_array[3] * amplitudes_g + (1 - self.specs_array[3]) * amplitudes_l
        return peak_height

    @peak_height.setter
    def peak_height(self, new_height):
        a = (4 * np.log(2) / np.pi) ** 0.5
        b = 1 - self.specs_array[3]
        c = 2 / (np.pi * self.specs_array[1])
        self.specs_array[4] = new_height / (self.specs_array[3] * a / self.specs_array[1] + b * c)

    @property
    def fwhm_asym(self):
        """ real fwhm of an asymmetric peak"""
        fwhm_asym = self.fwhm * (1 + 0.4 * self.asymmetry ** 2 + 1.35 * self.asymmetry ** 4)
        return fwhm_asym

    @property
    def curve(self):
        """ Asymmetric pseudo-Voigt function as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = self.specs_array[4] * _voigt_asym(self.wn - self.specs_array[0],
                                                  self.specs_array[1],
                                                  self.specs_array[2],
                                                  self.specs_array[3])
        return curve


class DespikeParams:
    """
    Named tuple to hold parameters for the despiking process.

    Attributes
    ----------
    subtracted_peaks : list
        of subtracted peaks.
    subtracted_peaks_idx : list
        of indices of subtracted peaks.
    y_moll : np.ndarray | None
        Mollified spectrum.
    y_mod_score : np.ndarray | None
        Modulation score of the spectrum.
    stop_loop : bool
        Flag to indicate whether to stop the despiking loop.
    top_peak_idx : list
        of indices of the top peaks.
    cosmic_positions : list
        of positions of cosmic spikes.
    mollification_width: int
    width: float
        corrected width
    """

    def __init__(self):
        self.subtracted_peaks = []
        self.subtracted_peaks_idx = []
        self.y_moll = None
        self.y_mod_score = None
        self.stop_loop = False
        self.top_peak_idx = []
        self.cosmic_positions = []
        self.mollification_width = 0
        self.width = 0.0


def despike(item: tuple[str, np.ndarray, float], laser_wl: float = 785., maxima_count: int = 2,
            width: float = 0.0) -> tuple[str, np.ndarray, list] | None:
    """
    Remove strong spikes in spectrum.

    Parameters
    -------
    item: tuple[str, np.ndarray, float]
        filename, spectrum array, laser_fwhm in nm.

    laser_wl: float, default = 785.
        Wavelength at laser peak

    maxima_count: int, default = 2
        How many cosmic spikes are looked for per loop. Smaller number - less precision.
        Higher number - more false positives.

    width: float, default = 0.0
        FWHM of cosmic spikes, in units of x-axis, not in pixels!
        By default, width = distance be two pixels of CCD;
            if the data were automatically processed (like, say, in Bruker Senterra),
            then the width should be set to ~2
        The script assumes that x are sorted in the ascending order.

    Returns
    -------
    items: tuple[str, np.ndarray, list]
        filename: str
            as input
        return_array: np.ndarray
            fixed spectrum
        subtracted_peaks: list
            Wavelength array where peaks were fixed

        None if nothing was changed
    """
    params = DespikeParams()
    sp = _split_spectrum(item[1], laser_wl)
    params.mollification_width, params.width = _get_molli_width(sp['x'], width, item[2])
    spectrum = ExpSpec(sp['x'], sp['y'])
    for _ in range(5):
        if environ['CANCEL'] == '1':
            break
        params.y_moll = _moving_average_molification(spectrum.y,
                                                     struct_el=params.mollification_width)
        params.y_mod_score = np.abs(spectrum.y - params.y_moll)
        params.stop_loop, params.top_peak_idx = _check_stop_criteria(maxima_count,
                                                                     params.y_mod_score,
                                                                     params.subtracted_peaks_idx,
                                                                     params.mollification_width)
        if params.stop_loop:
            break
        params.cosmic_positions = []
        for i in range(maxima_count):
            params.cosmic_positions.append(sp['x'][params.top_peak_idx[i]])
        peak_signs = []
        for i in range(maxima_count):
            y_i = spectrum.y[params.top_peak_idx[i]] - params.y_moll[params.top_peak_idx[i]]
            peak_signs.append(np.sign(y_i))
        # fit:
        cosmic_spikes = _get_cosmic_spikes(maxima_count, peak_signs, spectrum, params)
        # subtract:
        result_of_subtract = _subtract_spectrum_and_peak(sp['x'], maxima_count, cosmic_spikes,
                                                         spectrum, params)
        if len(params.subtracted_peaks) == 0 or result_of_subtract[0]:
            break
        spectrum = result_of_subtract[1]
    result_array = _concatenated_spectrum(sp, spectrum.y)

    return (item[0], result_array, params.subtracted_peaks) if len(params.subtracted_peaks) > 0 \
        else None


def _get_molli_width(x: np.ndarray, width: float, fwhm_nm: float) -> tuple[int, float]:
    """
    Calculate the mollification_width.

    Parameters
    -------
    x: np.ndarray
        x_axis

    width: float
        target FWHM of cosmic spikes

    fwhm_nm: float
        laser_fwhm in nm

    Returns
    -------
    mollification_width: int

    width: float
        corrected width
    """
    spectrum_nm_range = np.abs(x[-1] - x[0])
    if width < 0.05:
        width = spectrum_nm_range / (len(x) - 1) * 1.6
    width = min(width, fwhm_nm)
    width_in_pixels = width * len(x) / spectrum_nm_range
    mollification_width = int(2 * ceil(width_in_pixels) + 1)
    return mollification_width, width


def _split_spectrum(spectrum: np.ndarray, laser_wl: float) -> dict:
    """
    # We do not study the area near the laser peak so split spectrum into two parts.

    Parameters
    -------
    spectrum: np.ndarray

    laser_wl: float, default = 785.
        Wavelength at laser peak

    Returns
    -------
    result: dict with
        x: np.ndarray
            x_axis data
        y: np.ndarray
            y_axis data

        _x: np.ndarray
            x_axis data near laser peak
        _y: np.ndarray
            y_axis data near laser peak

    """
    x = spectrum[:, 0]
    y = spectrum[:, 1]
    idx = nearest_idx(x, laser_wl + 5)
    _x, x = x[0:idx], x[idx:]
    _y, y = y[0:idx], y[idx:]
    result = {'x': x, 'y': y, '_x': _x, '_y': _y}
    return result


def _check_stop_criteria(maxima_count: int, y_mod_score: np.ndarray,
                         subtracted_peaks_idx: list[int], mollification_width: int) \
        -> tuple[bool, list]:
    """
    Find the largest peaks and store it.
    """
    stop_loop = False
    top_peak_indexes = []  # Store peaks so that they don't repeat themselves.
    for _ in range(maxima_count):
        top_peak_index = (np.abs(y_mod_score)).argmax()
        stop_loop = top_peak_index in subtracted_peaks_idx or stop_loop
        if stop_loop:
            break
        top_peak_indexes.append(top_peak_index)
        subtracted_peaks_idx.append(top_peak_index)
        for k in range(-mollification_width, mollification_width + 1):
            idx = top_peak_index + k
            if idx < 0 or idx >= len(y_mod_score):
                continue
            y_mod_score[idx] = 0.
    return stop_loop, top_peak_indexes


def _concatenated_spectrum(sp: dict, y: np.ndarray) -> np.ndarray:
    """
    Make spectrum array for result.

    Parameters
    -------
    sp: dict
        with x, y - axes

    y: np.ndarray
        final y-axis data

    Returns
    -------
    arr: np.ndarray
        spectrum as 2D array
    """
    x = np.concatenate((sp['_x'], sp['x']))
    y = np.concatenate((sp['_y'], y))
    return np.vstack((x, y)).T


def _subtract_spectrum_and_peak(x: np.ndarray, maxima_count: int,
                                cosmic_spikes: list[CalcPeak],
                                spectrum: ExpSpec, params: DespikeParams) \
        -> tuple[bool, ExpSpec]:
    """
    Subtracts spectral peaks from the given spectrum and identifies broad peaks.

    Parameters
    ----------
    x : np.ndarray
        The input data used for peak calculations.
    maxima_count : int
        The number of peaks to process.
    cosmic_spikes : list[CalcPeak]
        List containing cosmic spike data or calculated peaks.
    spectrum : ExpSpec
        The experimental spectrum from which peaks are subtracted.
    params : DespikeParams

    Returns
    -------
    tuple
        A tuple containing:
        - all_peaks_very_broad (bool): Indicator of whether all peaks are broader than the
            specified width.
        - spectrum (ExpSpec): The spectrum after peak subtraction.
    """
    broad_peaks = np.zeros(maxima_count)
    all_peaks_very_broad = False
    for i in range(maxima_count):
        peak2subtract = CalcPeak(x)
        peak2subtract.specs_array = cosmic_spikes[i].specs_array
        if peak2subtract.fwhm > params.width:
            broad_peaks[i] = 1
            continue
        spectrum = ExpSpec(x, spectrum.y - peak2subtract.curve)
        params.subtracted_peaks.append(params.cosmic_positions[i])
    if np.max(broad_peaks) == np.min(broad_peaks) and broad_peaks[0] == 1:
        all_peaks_very_broad = True
    return all_peaks_very_broad, spectrum


def _get_cosmic_spikes(maxima_count: int, peak_signs: list, spectrum: ExpSpec,
                       params: DespikeParams) -> list[int | CalcPeak]:
    """
    Identify and fit cosmic spikes in a given spectral data set.

    This function detects cosmic spikes within a spectrum by fitting peaks at specified positions.
    Each peak is fitted within a range defined around the given position. The fitting process
    uses a specified full-width at half-maximum (FWHM) and peak signs.

    Parameters:
    ----------
    maxima_count: int
        The number of peaks (cosmic spikes) to be fitted in the spectrum.
    peak_signs: list
        A list of signs indicating the direction of the peaks. Typically, these are either positive
        or negative values to denote peak or trough.
    spectrum: ExpSpec
        The experimental spectrum data where cosmic spikes need to be identified.
    params: DespikeParams
        Parameters for the despiking process. Includes the positions of cosmic spikes and the width
        used for fitting.

    Returns:
    ----------
    list[int | CalcPeak]:
        A list of fitted peak results for each cosmic spike. Each item in the list
        is either an integer indicating the position of the peak or a `CalcPeak`
        object containing detailed fitting information.
    """
    cosmic_spikes = []
    for i in range(maxima_count):
        range_left = params.cosmic_positions[i] - 8 * params.width
        range_right = params.cosmic_positions[i] + 8 * params.width
        cosmic_spike = _fit_single_peak(spectrum, peak_position=params.cosmic_positions[i],
                                        fit_range=(range_left, range_right),
                                        peak_sign=peak_signs[i], fwhm=params.width)
        cosmic_spikes.append(cosmic_spike)
    return cosmic_spikes


def _fit_single_peak(spectrum: ExpSpec, peak_position: list[int] | None = None,
                     fit_range: list[tuple[float, float] | None] = None, peak_sign: int = 0,
                     fwhm: float | None = None) -> CalcPeak:
    """
    Fits a single peak in the given spectrum data.

    This function fits a peak in the spectrum data by initializing the peak's
    position, fitting range, and other parameters. It uses least squares
    optimization method to find the best fit for the peak.

    Parameters
    ----------
    spectrum : ExpSpec
        The experimental spectrum data containing x and y values.
    peak_position : list[int] or None, optional
        The initial position of the peak. If None, the position is determined
        automatically based on the peak_sign (default is None).
    fit_range : list[tuple[float, float]] or None, optional
        The range (x1, x2) within which the peak fitting is performed. If None,
        the range is set to ±4*FWHM around the peak position (default is None).
    peak_sign : int, optional
        Indicates the direction of the peak. Use 1 for positive peaks and -1 for
        negative peaks (default is 0).
    fwhm : float or None, optional
        The full width at half maximum (FWHM) of the peak. If None, it defaults
        to 4 times the inter-point distance (default is None).

    Returns
    -------
    CalcPeak
        A `CalcPeak` object containing the fitted peak information.

    Notes
    -----
    - The function initializes the peak position and FWHM, and sets the fitting
        range based on the provided parameters or defaults.
    - The fitting process involves setting bounds for the parameters and using
        least squares method to find the optimal fit.
    - The original working range of the spectrum is restored after fitting.
    """
    # step 0: initialize the calc peak:
    peak = CalcPeak(spectrum.x)
    # capture the original working range, which has to be restored later:
    original_working_range = spectrum.working_range

    # step 1: check if we need to set x0 and find the index of x0
    if fit_range is not None:  # set to +-4*fwhm
        spectrum.working_range = fit_range

    if peak_position is None:
        # find index of a maximum y:
        if peak_sign != 0:
            idx0 = (peak_sign * detrend(spectrum.y)).argmax()
        else:
            idx0 = np.abs(spectrum.x - peak_position).argmin()
            peak_sign = np.sign(detrend(spectrum.y)[idx0])
        peak.position = spectrum.x[idx0]
    else:
        peak.position = peak_position
        if peak_sign == 0:
            idx0 = np.abs(spectrum.x - peak_position).argmin()
            peak_sign = np.sign(detrend(spectrum.y)[idx0])

    # step 2: set initial value of fwhm from input or to 5 points (4 inter-point distances)
    inter_point_distance = (spectrum.x[-1] - spectrum.x[0]) / (len(spectrum.x) - 1)
    peak.fwhm = abs(4 * inter_point_distance) if fwhm is None else fwhm

    # step 3: Set initial working range
    if fit_range is None:  # set to +-4*fwhm
        # find point number for the closest to x0 point:
        idx0 = (np.abs(spectrum.x - peak.position)).argmin()
        spectrum.working_range = (spectrum.x[idx0] - 4 * peak.fwhm,
                                  spectrum.x[idx0] + 4 * peak.fwhm)

    # step 4: Set fitting range
    peak.wn = spectrum.x

    # step 5: Set other starting values
    peak.voigt_amplitude, peak.asymmetry = 0, 0
    starting_point = np.zeros(7)
    bounds = {'high': np.full_like(starting_point, np.inf),
              'low': np.full_like(starting_point, -np.inf)}
    # asymmetry and Gaussian share
    bounds['low'][2], bounds['high'][2], bounds['low'][3], bounds['high'][3] = -0.36, 0.36, 0, 1
    while True:
        # 1: find index of a y_max, y_min within the fitting range,
        idx0local = (np.abs(spectrum.x - peak.position)).argmin()
        peak_height = peak_sign * (np.abs((detrend(spectrum.y))[idx0local]))
        y_min_local = spectrum.y[idx0local] - peak_height
        peak.peak_height, starting_point = peak_height, peak.specs_array
        starting_point[5] = 0  # always start next round with the flat baseline
        starting_point[6] = y_min_local

        # 2: set bounds for parameters:
        bounds['low'][0] = peak.position - np.sign(spectrum.x[-1] - spectrum.x[0]) * peak.fwhm * 2
        bounds['high'][0] = peak.position + np.sign(spectrum.x[-1] - spectrum.x[0]) * peak.fwhm * 2
        bounds['low'][1] = 0.25 * peak.fwhm
        bounds['high'][1] = abs((spectrum.x[-1] - spectrum.x[0])) / 2 if fwhm is None \
            else 8 * peak.fwhm

        # amplitude depending on sign:
        if peak_sign > 0:
            bounds['low'][4] = 0
        elif peak_sign < 0:
            bounds['high'][4] = 0
        try:
            solution = least_squares(_func2min, starting_point,
                                     args=(spectrum.x, spectrum.y, peak_sign),
                                     bounds=[bounds['low'], bounds['high']], ftol=1e-5, xtol=1e-5,
                                     gtol=1e-5)
            peak.specs_array = solution.x
            break
        except RuntimeError:
            spectrum.working_range = (spectrum.working_range[0] - inter_point_distance,
                                      spectrum.working_range[1] + inter_point_distance)
            peak.wn = spectrum.x
            continue
    # restoring the working range of the ExpSpec class:
    spectrum.working_range = original_working_range
    return peak


@njit(cache=True, fastmath=True)
def _voigt_asym(x: list[float], fwhm: float, asymmetry: float, gaussian_share: float) \
        -> list[float]:
    """
    Compute the pseudo-Voigt profile composed of Gaussian and Lorentzian components.

    This function calculates the pseudo-Voigt profile, which is a combination of
    Gaussian and Lorentzian profiles, and accounts for asymmetry in the peaks. The
    profile is normalized to have a unit area if it is symmetric.

    Parameters
    ----------
    x : list of float
        The x-values at which to calculate the profile.
    fwhm : float
        The full width at half maximum (FWHM) of the profile.
    asymmetry : float
        The asymmetry parameter, which distorts the profile.
    gaussian_share : float
        The fraction of the Gaussian component in the pseudo-Voigt profile.

    Returns
    -------
    list of float
        The pseudo-Voigt profile values at the given x-values.

    References
    ----------
    - Analyst: 10.1039/C8AN00710A
    """
    x_distorted = x * (1 - np.exp(-x ** 2 / (2 * (2 * fwhm) ** 2)) * asymmetry * x / fwhm)
    lor_asym = fwhm / (x_distorted ** 2 + fwhm ** 2 / 4) / 6.2831853
    gauss_asym = 0.9394373 / fwhm * np.exp(-(x_distorted ** 2 * 2.7725887) / fwhm ** 2)
    result = (1 - gaussian_share) * lor_asym + gaussian_share * gauss_asym
    return result


def _moving_average_molification(raw_spectrum: np.ndarray, struct_el: int = 7) -> np.ndarray:
    """
    Apply moving average molification to a raw spectrum.

    This function smooths the raw spectrum using a moving average with a specified
    structuring element size.

    Parameters
    ----------
    raw_spectrum : np.ndarray
        The raw spectrum data to be smoothed.
    struct_el : int, optional
        The size of the structuring element for the moving average. Default is 7.

    Returns
    -------
    np.ndarray
        The smoothed spectrum.
    """
    mol_kernel = np.ones(struct_el) / struct_el
    denominator = np.convolve(np.ones_like(raw_spectrum), mol_kernel, 'same')
    smooth_line = np.convolve(raw_spectrum, mol_kernel, 'same') / denominator
    return smooth_line


@njit
def _func2min(peak_params: list[float], *args):
    """
    Objective function to minimize during peak fitting.

    This function calculates the difference between the observed data and the model,
    which consists of a Voigt profile and a linear baseline. The function also applies
    a weighting to the residuals based on their sign and magnitude.

    Parameters
    ----------
    peak_params : list of float
        The parameters of the peak model, which include:
        0 - x0 (peak position)
        1 - fwhm (full width at half maximum)
        2 - asymmetry
        3 - Gaussian share
        4 - Voigt amplitude
        5 - baseline slope
        6 - baseline intercept
    args : tuple
        Additional arguments containing:
        0 - x-values of the data
        1 - y-values of the data
        2 - sign of the peak

    Returns
    -------
    np.ndarray
        The weighted differences between the observed data and the model.
    """
    x_diff = args[0] - peak_params[0]
    voigt = _voigt_asym(x_diff, peak_params[1], peak_params[2], peak_params[3])
    the_diff = args[1] - (peak_params[4] * voigt + peak_params[5] * x_diff + peak_params[6])
    der_func = the_diff * np.exp(0.1 * (1 - args[2] * np.sign(the_diff)) ** 2)
    return der_func
