# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functions for generating various peak shapes commonly used in
spectroscopic analysis. The peak shapes include Gaussian, Lorentzian, Voigt, and Pearson
distributions, along with their skewed and split variants.

Functions
---------
gaussian : Compute a Gaussian peak.
split_gaussian : Compute a split Gaussian peak with different widths for left and right slopes.
skewed_gaussian : Compute a skewed Gaussian peak.
lorentzian : Compute a Lorentzian peak.
split_lorentzian : Compute a split Lorentzian peak with different widths for left and right slopes.
voigt : Compute a Voigt peak.
voigt_sigma : Compute the sigma parameter for the Voigt peak.
voigt_z_norm : Compute the normalized z parameter for the Voigt peak.
voigt_z : Compute the z parameter for the Voigt peak.
split_voigt : Compute a split Voigt peak with different widths for left and right slopes.
skewed_voigt : Compute a skewed Voigt peak.
pseudovoigt : Compute a pseudo-Voigt peak.
split_pseudovoigt : Compute a split pseudo-Voigt peak with different widths for left and right
slopes.
pearson4 : Compute a Pearson IV peak.
split_pearson4 : Compute a split Pearson IV peak with different widths for left and right slopes.
pearson7 : Compute a Pearson VII peak.
split_pearson7 : Compute a split Pearson VII peak with different widths for left and right slopes.
"""

import numpy as np
import scipy.special as sc
from numba import njit
from scipy.special import wofz

from src.data.work_with_arrays import nearest_idx


@njit(cache=True, fastmath=True)
def gaussian(x: np.ndarray, a: float, x0: float, dx: float) -> np.ndarray:
    """
    Compute a Gaussian peak.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Gaussian component.
    dx : float or ndarray
        Half-width at half-maximum.

    Returns
    -------
    ndarray
        The signal.
    """
    return a * np.exp(-np.log(2) * ((x - x0) / dx) ** 2)


@njit(cache=True, fastmath=True)
def split_gaussian(x: np.ndarray, a: float, x0: float, dx: float, dx_left: float) -> np.ndarray:
    """
    Compute a split Gaussian peak with different widths for left and right slopes.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Gaussian component.
    dx : float or ndarray
        Half-width at half-maximum of the right slope.
    dx_left : float or ndarray
        Half-width at half-maximum of the left slope.

    Returns
    -------
    ndarray
        The signal.
    """
    x0_idx = nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = gaussian(x_left, a, x0, dx_left)
    y_right = gaussian(x_right, a, x0, dx)
    return np.concatenate((y_left, y_right), axis=0)


def skewed_gaussian(x: np.ndarray, a: float, x0: float, dx: float, gamma: float) -> np.ndarray:
    """Return a Gaussian line shape, skewed with error function.

    Equal to: gaussian(x, center, sigma)*(1+erf(beta*(x-center)))

    where ``beta = gamma/(sigma*sqrt(2))``

    with ``gamma < 0``: tail to low value of centroid
         ``gamma > 0``: tail to high value of centroid

    For more information, see:
    https://en.wikipedia.org/wiki/Skew_normal_distribution

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Gaussian component.
    dx : float or ndarray
        Half-width at half-maximum.
    gamma : float
        Skewness parameter.

    Returns
    -------
    ndarray
        The signal.
    """
    sigma = dx / np.sqrt(2 * np.log(2))
    asym = 1 + sc.erf(gamma * (x - x0) / (np.sqrt(2.0) * sigma))
    return asym * gaussian(x, a, x0, dx)


@njit(cache=True, fastmath=True)
def lorentzian(x: np.ndarray, a: float, x0: float, dx: float) -> np.ndarray:
    """
    Compute a Lorentzian peak.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Lorentzian component.
    dx : float or ndarray
        Half-width at half-maximum.

    Returns
    -------
    ndarray
        The signal.
    """
    return a / (1 + ((x - x0) / dx) ** 2)


@njit(cache=True, fastmath=True)
def split_lorentzian(x: np.ndarray, a: float, x0: float, dx: float, dx_left: float) ->np.ndarray:
    """
    Compute a split Lorentzian peak with different widths for left and right slopes.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Lorentzian component.
    dx : float or ndarray
        Half-width at half-maximum of the right slope.
    dx_left : float or ndarray
        Half-width at half-maximum of the left slope.

    Returns
    -------
    ndarray
        The signal.
    """
    x0_idx = nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = lorentzian(x_left, a, x0, dx_left)
    y_right = lorentzian(x_right, a, x0, dx)
    return np.concatenate((y_left, y_right), axis=0)


def voigt(x: np.ndarray, a: float = 1., x0: float = 0., dx: float = 1., gamma: float = 0.) \
        -> np.ndarray:
    """
    Compute a Voigt peak.
    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float
        Amplitude.
    x0 : float
        Frequency/position of the Voigt component.
    dx : float
        Half-width at half-maximum.
    gamma : float
        Lorentzian width.

    Returns
    -------
    ndarray
        The signal.
    """
    sigma = voigt_sigma(dx, gamma)
    z_norm = voigt_z_norm(gamma, sigma)
    _norm_factor = wofz(z_norm).real / (sigma * np.sqrt(2 * np.pi))
    z = voigt_z(x - x0, gamma, sigma)
    y = wofz(z).real / (sigma * np.sqrt(2 * np.pi))
    return a * y / _norm_factor


@njit(cache=True, fastmath=True)
def voigt_sigma(dx, gamma) -> float:
    """
    Compute the sigma parameter for the Voigt peak.

    Parameters
    ----------
    dx : float
        Half-width at half-maximum.
    gamma : float
        Lorentzian width.

    Returns
    -------
    float
        Sigma parameter.
    """
    sigma = dx**2 - 1.0692 * dx * gamma + .06919716 * gamma**2
    sigma = np.sqrt(sigma) / np.sqrt(2 * np.log(2))
    return max(1.e-15, sigma)


@njit(cache=True, fastmath=True)
def voigt_z_norm(gamma, sigma) -> np.ndarray:
    """
    Compute the normalized z parameter for the Voigt peak.

    Parameters
    ----------
    gamma : float
        Lorentzian width.
    sigma : float
        Sigma parameter.

    Returns
    -------
    ndarray
        Normalized z parameter.
    """
    return (1j * gamma) / (sigma * np.sqrt(2.0))


@njit(cache=True, fastmath=True)
def voigt_z(x, gamma, sigma) -> np.ndarray:
    """
    Compute the z parameter for the Voigt peak.

    Parameters
    ----------
    x : float or ndarray
        Frequency/position.
    gamma : float
        Lorentzian width.
    sigma : float
        Sigma parameter.

    Returns
    -------
    ndarray
        z parameter.
    """
    return (x + 1j * gamma) / (sigma * np.sqrt(2.0))


def split_voigt(x: np.ndarray, a: float, x0: float, dx: float, dx_left: float, gamma: float) \
        -> np.ndarray:
    """
    Compute a split Voigt peak with different widths for left and right slopes.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Voigt component.
    dx : float or ndarray
        Half-width at half-maximum of the right slope.
    dx_left : float or ndarray
        Half-width at half-maximum of the left slope.
    gamma : float
        Lorentzian width.

    Returns
    -------
    ndarray
        The signal.
    """
    x0_idx = nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = voigt(x_left, a, x0, dx_left, gamma)
    y_right = voigt(x_right, a, x0, dx, gamma)
    return np.concatenate((y_left, y_right), axis=0)


def skewed_voigt(x: np.ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0,
                 gamma: float = 0.0, skew: float = 0.0) -> np.ndarray:
    """Return a Voigt line shape, skewed with error function.

    Equal to: voigt(x, center, sigma, gamma)*(1+erf(beta*(x-center)))

    where ``beta = skew/(sigma*sqrt(2))``

    with ``skew < 0``: tail to low value of centroid
         ``skew > 0``: tail to high value of centroid

    Useful, for example, for ad-hoc Compton scatter profile. For more
    information, see: https://en.wikipedia.org/wiki/Skew_normal_distribution

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Voigt component.
    dx : float or ndarray
        Half-width at half-maximum.
    gamma : float
        Lorentzian width.
    skew : float
        Skewness parameter.

    Returns
    -------
    ndarray
        The signal.
    """
    sigma = dx * np.sqrt(2 * np.log(2))
    beta = skew / sigma
    asym = 1 + sc.erf(beta * (x - x0))
    return asym * voigt(x, a, x0, dx, gamma)


@njit(cache=True, fastmath=True)
def pseudovoigt(x: np.ndarray, a: float, x0: float, dx: float, l_ratio: float) -> np.ndarray:
    """
    compute a pseudo-Voigt peak

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the pseudo-Voigt component.
    dx : float or ndarray
        Half-width at half-maximum.
    l_ratio : float or ndarray
        Ratio of the Lorentzian component, should be between 0 and 1.

    Returns
    -------
    ndarray
        The signal.
    """
    if (l_ratio > 1) or (l_ratio < 0):  # if entries are floats
        raise ValueError("L_ratio should be comprised between 0 and 1")
    return l_ratio * lorentzian(x, a, x0, dx) + (1 - l_ratio) * gaussian(x, a, x0, dx)


@njit(cache=True, fastmath=True)
def split_pseudovoigt(x: np.ndarray, a: float, x0: float, dx: float, dx_left: float,
                      l_ratio: float) -> np.ndarray:
    """
    Compute a split pseudo-Voigt peak with different widths for left and right slopes.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the pseudo-Voigt component.
    dx : float or ndarray
        Half-width at half-maximum of the right slope.
    dx_left : float or ndarray
        Half-width at half-maximum of the left slope.
    l_ratio : float or ndarray
        Ratio of the Lorentzian component, should be between 0 and 1.

    Returns
    -------
    ndarray
        The signal.
    """
    x0_idx = nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = pseudovoigt(x_left, a, x0, dx_left, l_ratio)
    y_right = pseudovoigt(x_right, a, x0, dx, l_ratio)
    return np.concatenate((y_left, y_right), axis=0)


def pearson4(x: np.ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0,
             expon: float = 1.0, skew: float = 0.0) -> np.ndarray:
    """Return a Pearson4 line shape.

    Using the Wikipedia definition:

    pearson4(x, amplitude, center, sigma, expon, skew) =
        * amplitude*|gamma(expon + I skew/2)/gamma(m)|**2/(w*beta(expon-0.5, 0.5)) *
        * (1+arg**2)**(-expon) * exp(-skew * arc-tan(arg))

    where ``arg = (x-center)/sigma``, `gamma` is the gamma function and `beta` is the beta
    function.

    For more information,
    see: https://en.wikipedia.org/wiki/Pearson_distribution#The_Pearson_type_IV_distribution

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float
        Amplitude.
    x0 : float
        Frequency/position of the Pearson IV component.
    dx : float
        Half-width at half-maximum.
    expon : float
        Exponent parameter.
    skew : float
        Skewness parameter.

    Returns
    -------
    ndarray
        The signal.
    """
    arg = (x - x0) / dx
    log_pre_factor = (2 * (np.real(sc.loggamma(expon + skew * 0.5j)) - sc.loggamma(expon))
                      - sc.betaln(expon - 0.5, 0.5))
    return (a * np.pi / expon) * np.exp(log_pre_factor - expon * np.log1p(arg * arg) - skew
                                        * np.arctan(arg))


def split_pearson4(x: np.ndarray, a: float, x0: float, dx: float, dx_left: float, expon: float,
                   skew: float = 0.0) -> np.ndarray:
    """
    Return a 1-dimensional piecewise Pearson7 function.

    Split means that width of the function is different between
    left and right slope of the function.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Pearson IV component.
    dx : float or ndarray
        Half-width at half-maximum of the right slope.
    dx_left : float or ndarray
        Half-width at half-maximum of the left slope.
    expon : float
        Exponent parameter.
    skew : float
        Skewness parameter.

    Returns
    -------
    ndarray
        The signal.
    """
    x0_idx = nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = pearson4(x_left, a, x0, dx_left, expon, skew)
    y_right = pearson4(x_right, a, x0, dx, expon, skew)
    y = np.concatenate((y_left, y_right), axis=0)
    return y


@njit(cache=True, fastmath=True)
def pearson7(x: np.ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0,
             expon: float = 1.0) -> np.ndarray:
    """Return a Pearson7 line shape.

    Using the Wikipedia definition:

    pearson7(x, center, sigma, expon) =
        amplitude*(1+arg**2)**(-expon)/(sigma*beta(expon-0.5, 0.5))

    where ``arg = (x-center)/sigma`` and `beta` is the beta function.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Pearson VII component.
    dx : float or ndarray
        Half-width at half-maximum.
    expon : float or ndarray
        Exponent parameter.

    Returns
    -------
    ndarray
        The signal.
    """
    if expon == 0.0:
        expon = 0.1
    arg = ((x - x0) / dx) ** 2
    arg2 = 2 ** (1 / expon) - 1
    arg3 = (1 + arg * arg2) ** expon
    return a / arg3


@njit(cache=True, fastmath=True)
def split_pearson7(x: np.ndarray, a: float, x0: float, dx: float, dx_left: float, expon: float) \
        -> np.ndarray:
    """
    Return a 1-dimensional piecewise Pearson7 function.

    Split means that width of the function is different between
    left and right slope of the function.

    Parameters
    ----------
    x : ndarray
        The positions at which the signal should be sampled.
    a : float or ndarray
        Amplitude.
    x0 : float or ndarray
        Frequency/position of the Pearson VII component.
    dx : float or ndarray
        Half-width at half-maximum of the right slope.
    dx_left : float or ndarray
        Half-width at half-maximum of the left slope.
    expon : float or ndarray
        Exponent parameter.

    Returns
    -------
    ndarray
        The signal.
    """
    x0_idx = nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = pearson7(x_left, a, x0, dx_left, expon)
    y_right = pearson7(x_right, a, x0, dx, expon)
    return np.concatenate((y_left, y_right), axis=0)
