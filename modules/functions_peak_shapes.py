import scipy.special as sc
from numba import njit, float64, int64
from numpy import ndarray, exp, sqrt, log, pi, concatenate, real, log1p, arctan, cos

from modules.static_functions import find_nearest_idx

s2 = sqrt(2.0)
ln2 = log(2)
s2ln2 = sqrt(2 * ln2)
s2pi = sqrt(2 * pi)


@njit(float64[::](float64[::], float64, float64, float64), fastmath=True)
def gaussian(x: ndarray, a: float, x0: float, dx: float) -> ndarray:
    """compute a Gaussian peak

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx : float or ndarray with size equal to x.shape
        half-width at half-maximum

    Returns
    -------
    out : ndarray
        the signal

    Remarks
    -------
    Formula is a*np.exp(-np.log(2)*((x-x0)/dx)**2)
    """
    return a * exp(-ln2 * ((x - x0) / dx) ** 2)


@njit(float64[::](float64[::], float64, float64, float64, float64),
      locals={'x0_idx': int64, 'x_left': float64[::], 'x_right': float64[::], 'y_left': float64[::],
              'y_right': float64[::]}, fastmath=True)
def split_gaussian(x: ndarray, a: float, x0: float, dx: float, dx_left: float) -> ndarray:
    """Return a 1-dimensional piecewise gaussian function.

    Split means that width of the function is different between
    left and right slope of the function.

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx_left, dx_right : float or ndarray with size equal to x.shape
        half-width at half-maximum of left/right slope

    Returns
    -------
    out : ndarray
        the signal
    """
    x0_idx = find_nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = gaussian(x_left, a, x0, dx_left)
    y_right = gaussian(x_right, a, x0, dx)
    return concatenate((y_left, y_right), axis=0)


def skewed_gaussian(x: ndarray, a: float, x0: float, dx: float, gamma: float) -> ndarray:
    """Return a Gaussian line shape, skewed with error function.

    Equal to: gaussian(x, center, sigma)*(1+erf(beta*(x-center)))

    where ``beta = gamma/(sigma*sqrt(2))``

    with ``gamma < 0``: tail to low value of centroid
         ``gamma > 0``: tail to high value of centroid

    For more information, see:
    https://en.wikipedia.org/wiki/Skew_normal_distribution

    """
    sigma = dx / s2ln2
    asym = 1 + sc.erf(gamma * (x - x0) / (s2 * sigma))
    return asym * gaussian(x, a, x0, dx)


@njit(float64[::](float64[::], float64, float64, float64), fastmath=True)
def lorentzian(x: ndarray, a: float, x0: float, dx: float) -> ndarray:
    """compute a Lorentzian peak

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx : float or ndarray with size equal to x.shape
        half-width at half-maximum

    Returns
    -------
    out : ndarray
        the signal
    """
    return a / (1 + ((x - x0) / dx) ** 2)


@njit(float64[::](float64[::], float64, float64, float64, float64),
      locals={'x0_idx': int64, 'x_left': float64[::], 'x_right': float64[::], 'y_left': float64[::],
              'y_right': float64[::]}, fastmath=True)
def split_lorentzian(x: ndarray, a: float, x0: float, dx: float, dx_left: float) -> ndarray:
    """Return a 1-dimensional piecewise Lorentzian function.

    Split means that width of the function is different between
    left and right slope of the function.

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx_left, dx_right : float or ndarray with size equal to x.shape
        half-width at half-maximum of left/right slope

    Returns
    -------
    out : ndarray
        the signal
    """
    x0_idx = find_nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = lorentzian(x_left, a, x0, dx_left)
    y_right = lorentzian(x_right, a, x0, dx)
    return concatenate((y_left, y_right), axis=0)


def voigt(x: ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0, gamma: float = 0.0) -> ndarray:
    """Return a 1-dimensional Voigt function.

    voigt(x, amplitude, center, sigma, gamma) =
        amplitude*wofz(z).real / (sigma*s2pi)

    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile

    """
    sigma = dx * s2ln2
    if gamma is None:
        gamma = sigma
    z = (x - x0 + 1j * gamma) / sigma
    return a * sc.wofz(z).real


def split_voigt(x: ndarray, a: float, x0: float, dx: float, dx_left: float, gamma: float) -> ndarray:
    """Return a 1-dimensional piecewise Lorentzian function.

    Split means that width of the function is different between
    left and right slope of the function.

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx_left, dx_right : float or ndarray with size equal to x.shape
        half-width at half-maximum of left/right slope
    gam : float

    Returns
    -------
    out : ndarray
        the signal
    """
    x0_idx = find_nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = voigt(x_left, a, x0, dx_left, gamma)
    y_right = voigt(x_right, a, x0, dx, gamma)
    return concatenate((y_left, y_right), axis=0)


def skewed_voigt(x: ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0, gamma: float = 0.0, skew: float = 0.0) \
        -> ndarray:
    """Return a Voigt line shape, skewed with error function.

    Equal to: voigt(x, center, sigma, gamma)*(1+erf(beta*(x-center)))

    where ``beta = skew/(sigma*sqrt(2))``

    with ``skew < 0``: tail to low value of centroid
         ``skew > 0``: tail to high value of centroid

    Useful, for example, for ad-hoc Compton scatter profile. For more
    information, see: https://en.wikipedia.org/wiki/Skew_normal_distribution

    """
    sigma = dx * s2ln2
    beta = skew / sigma
    asym = 1 + sc.erf(beta * (x - x0))
    return asym * voigt(x, a, x0, dx, gamma)


@njit(float64[::](float64[::], float64, float64, float64, float64), fastmath=True)
def pseudovoigt(x: ndarray, a: float, x0: float, dx: float, l_ratio: float) -> ndarray:
    """compute a pseudo-Voigt peak
    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled. Can be provided as vector, nx1 or nxm array.
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx : float or ndarray with size equal to x.shape
        half-width at half-maximum
    L_ratio : float or ndarray with size equal to x.shape
        ratio pf the Lorentzian component, should be between 0 and 1 (included)

    Returns
    -------
    out : ndarray of size equal to x.shape
        the signal
    """
    if (l_ratio > 1) or (l_ratio < 0):  # if entries are floats
        raise ValueError("L_ratio should be comprised between 0 and 1")
    return l_ratio * lorentzian(x, a, x0, dx) + (1 - l_ratio) * gaussian(x, a, x0, dx)


@njit(float64[::](float64[::], float64, float64, float64, float64, float64), fastmath=True)
def split_pseudovoigt(x: ndarray, a: float, x0: float, dx: float, dx_left: float, l_ratio: float) -> ndarray:
    """Return a 1-dimensional piecewise Pseudo-Voigt function.

    Split means that width of the function is different between
    left and right slope of the function.

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx_left, dx_right : float or ndarray with size equal to x.shape
        half-width at half-maximum of left/right slope
    l_ratio : float [0.0 - 1.0]

    Returns
    -------
    out : ndarray
        the signal
    """
    x0_idx = find_nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = pseudovoigt(x_left, a, x0, dx_left, l_ratio)
    y_right = pseudovoigt(x_right, a, x0, dx, l_ratio)
    return concatenate((y_left, y_right), axis=0)


def pearson4(x: ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0, expon: float = 1.0, skew: float = 0.0) \
        -> ndarray:
    """Return a Pearson4 line shape.

    Using the Wikipedia definition:

    pearson4(x, amplitude, center, sigma, expon, skew) =
        amplitude*|gamma(expon + I skew/2)/gamma(m)|**2/(w*beta(expon-0.5, 0.5)) * (1+arg**2)**(-expon)
        * exp(-skew * arc-tan(arg))

    where ``arg = (x-center)/sigma``, `gamma` is the gamma function and `beta` is the beta function.

    For more information, see: https://en.wikipedia.org/wiki/Pearson_distribution#The_Pearson_type_IV_distribution

    """
    arg = (x - x0) / dx
    log_pre_factor = 2 * (real(sc.loggamma(expon + skew * 0.5j)) - sc.loggamma(expon)) - sc.betaln(expon - 0.5, 0.5)
    return (a * pi / expon) * exp(log_pre_factor - expon * log1p(arg * arg) - skew * arctan(arg))


def split_pearson4(x: ndarray, a: float, x0: float, dx: float, dx_left: float, expon: float, skew: float = 0.0) \
        -> ndarray:
    """Return a 1-dimensional piecewise Pearson7 function.

    Split means that width of the function is different between
    left and right slope of the function.

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx_left, dx_right : float or ndarray with size equal to x.shape
        half-width at half-maximum of left/right slope
    expon : float > 0.0
    skew: float

    Returns
    -------
    out : ndarray
        the signal
    """
    x0_idx = find_nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = pearson4(x_left, a, x0, dx_left, expon, skew)
    y_right = pearson4(x_right, a, x0, dx, expon, skew)
    y = concatenate((y_left, y_right), axis=0)
    return y


@njit(float64[::](float64[::], float64, float64, float64, float64), fastmath=True)
def pearson7(x: ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0, expon: float = 1.0) -> ndarray:
    """Return a Pearson7 line shape.

    Using the Wikipedia definition:

    pearson7(x, center, sigma, expon) =
        amplitude*(1+arg**2)**(-expon)/(sigma*beta(expon-0.5, 0.5))

    where ``arg = (x-center)/sigma`` and `beta` is the beta function.

    """
    if expon == 0.0:
        expon = 0.1
    arg = ((x - x0) / dx) ** 2
    arg2 = (2 ** (1 / expon) - 1)
    arg3 = (1 + arg * arg2) ** expon
    return a / arg3


@njit(float64[::](float64[::], float64, float64, float64, float64, float64), fastmath=True)
def split_pearson7(x: ndarray, a: float, x0: float, dx: float, dx_left: float, expon: float) -> ndarray:
    """Return a 1-dimensional piecewise Pearson7 function.

    Split means that width of the function is different between
    left and right slope of the function.

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx_left, dx_right : float or ndarray with size equal to x.shape
        half-width at half-maximum of left/right slope
    expon : float > 0.0

    Returns
    -------
    out : ndarray
        the signal
    """
    x0_idx = find_nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = pearson7(x_left, a, x0, dx_left, expon)
    y_right = pearson7(x_right, a, x0, dx, expon)
    return concatenate((y_left, y_right), axis=0)


@njit(float64[::](float64[::], float64, float64, float64, float64), fastmath=True)
def moffat(x: ndarray, a: float = 1.0, x0: float = 0., dx: float = 1.0, beta: float = 1.):
    """Return a 1-dimensional Moffat function.

    moffat(x, amplitude, center, sigma, beta) =
        amplitude / (((x - center)/sigma)**2 + 1)**beta

    """

    return a / (((x - x0) / dx) ** 2 + 1) ** beta


@njit(float64[::](float64[::], float64, float64, float64, float64, float64), fastmath=True)
def split_moffat(x: ndarray, a: float, x0: float, dx: float, dx_left: float, beta: float) -> ndarray:
    """Return a 1-dimensional Moffat function.

    Split means that width of the function is different between
    left and right slope of the function.

    Inputs
    ------
    x : ndarray
        the positions at which the signal should be sampled
    a : float or ndarray with size equal to x.shape
        amplitude
    x0 : float or ndarray with size equal to x.shape
        frequency/position of the Gaussian component
    dx_left, dx_right : float or ndarray with size equal to x.shape
        half-width at half-maximum of left/right slope
    beta : float > 0.0

    Returns
    -------
    out : ndarray
        the signal
    """
    x0_idx = find_nearest_idx(x, x0)
    x_left = x[:x0_idx]
    x_right = x[x0_idx:]
    y_left = moffat(x_left, a, x0, dx_left, beta)
    y_right = moffat(x_right, a, x0, dx, beta)
    y = concatenate((y_left, y_right), axis=0)
    return y


@njit(float64[::](float64[::], float64, float64, float64, float64), fastmath=True)
def doniach(x: ndarray, a: float = 1.0, x0: float = 0.0, dx: float = 1.0, alpha: float = 0.0) -> ndarray:
    """Return a Doniach Sunjic asymmetric line shape.

    doniach(x, amplitude, center, sigma, gamma) =
        amplitude / sigma^(1-gamma) *
        cos(pi*gamma/2 + (1-gamma) arc-tan((x-center)/sigma) /
        (sigma**2 + (x-center)**2)**[(1-gamma)/2]
     -1.0 > gam < 1.0

    For example used in photo-emission; see
    http://www.casaxps.com/help_manual/line_shapes.htm for more information.

    """
    arg = (x - x0) / dx
    gm1 = (1.0 - alpha)
    return a * cos(pi * alpha / 2 + gm1 * arctan(arg)) / (1 + arg ** 2) ** (gm1 / 2)


@njit(float64[::](float64[::], float64, float64, float64, float64), fastmath=True)
def bwf(x: ndarray, a: float, x0: float, dx: float, q: float) -> ndarray:
    # Breit-Wigner-Fano line shape
    temp = (x - x0) / dx
    return a * (1 + temp * q)**2 / (1 + temp**2)
