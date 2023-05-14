import numpy as np
from numba import njit, float64
from scipy.optimize import least_squares
from scipy.signal import detrend


class ExpSpec:
    def __init__(self, full_x: np.ndarray | list, full_y: np.ndarray) -> None:
        self.full_x = full_x
        self.full_y = full_y
        self.xrange = (np.min(full_x), np.max(full_x))
        self.x = full_x
        self.y = full_y

    @property
    def working_range(self) -> tuple[float, float]:
        return self.xrange

    @working_range.setter
    def working_range(self, xrange: np.ndarray) -> None:
        self.xrange = (np.maximum(np.min(xrange), np.min(self.full_x)), np.minimum(np.max(xrange), np.max(self.full_x)))
        self.x = self.full_x[np.where(np.logical_and(self.full_x >= np.amin(xrange), self.full_x <= np.amax(xrange)))]
        self.y = self.full_y[np.where(np.logical_and(self.full_x >= np.amin(xrange), self.full_x <= np.amax(xrange)))]


class SpectralFeature:
    """ Abstract spectral feature, with no x-axis defined
     Order of parameters in array:
     0:    x0 (default 0)
     1:    FWHM (default 1)
     2:    asymmetry (default 0)
     3:    Gaussian_share (default 0, i.e. Lorentzian peak)
     4:    voigt_amplitude (~area, not height)
     5:    Baseline slope (k) for linear BL
     6:    Baseline offset (b) for linear BL
    """

    def __init__(self) -> None:
        self.specs_array = np.zeros(7)
        self.specs_array[1] = 1  # set default FWHM to 1. Otherwise, we can get division by 0

    @property
    def position(self) -> float:
        return self.specs_array[0]

    @position.setter
    def position(self, position: float) -> None:
        self.specs_array[0] = position

    @property
    def fwhm(self) -> float:
        return self.specs_array[1]

    @fwhm.setter
    def fwhm(self, fwhm: float) -> None:
        self.specs_array[1] = fwhm

    @property
    def asymmetry(self) -> float:
        return self.specs_array[2]

    @asymmetry.setter
    def asymmetry(self, asymmetry: float) -> None:
        self.specs_array[2] = asymmetry

    @property
    def Gaussian_share(self) -> float:
        return self.specs_array[3]

    @Gaussian_share.setter
    def Gaussian_share(self, Gaussian_share: float) -> None:
        self.specs_array[3] = Gaussian_share

    @property
    def voigt_amplitude(self) -> float:
        return self.specs_array[4]

    @voigt_amplitude.setter
    def voigt_amplitude(self, voigt_amplitude: float) -> None:
        self.specs_array[4] = voigt_amplitude

    @property
    def BL_slope(self) -> float:
        return self.specs_array[5]

    @BL_slope.setter
    def BL_slope(self, BL_slope: float) -> None:
        self.specs_array[5] = BL_slope

    @property
    def BL_offset(self) -> float:
        return self.specs_array[6]

    @BL_offset.setter
    def BL_offset(self, BL_offset: float) -> None:
        self.specs_array[6] = BL_offset


class CalcPeak(SpectralFeature):
    """ Asymmetric peak calculated on x-asis (a grid of wave-numbers).
    It is possible to set a peak height,
        Changing fwhm keeps area same, while changes height.
        Changing height changes area while keeps fwhm.
    """

    def __init__(self, wn=np.linspace(0, 1, 129)):
        super().__init__()
        self.wn = wn
        self.specs_array[0] = (wn[-1] - wn[0]) / 2

    @property
    def peak_area(self):
        peak_area = (1 - self.specs_array[3]) * self.specs_array[4] * (
                1 + 0.69 * self.specs_array[2] ** 2 + 1.35 * self.specs_array[2] ** 4) + self.specs_array[3] * \
                    self.specs_array[4] * (1 + 0.67 * self.specs_array[2] ** 2 + 3.43 * self.specs_array[2] ** 4)
        return peak_area

    @property
    def peak_height(self):
        amplitudes_l = self.specs_array[4] * 2 / (np.pi * self.specs_array[0])
        amplitudes_g = self.specs_array[4] * (4 * np.log(2) / np.pi) ** 0.5 / self.specs_array[0]
        peak_height = self.specs_array[3] * amplitudes_g + (1 - self.specs_array[3]) * amplitudes_l
        return peak_height

    @peak_height.setter
    def peak_height(self, new_height):
        self.specs_array[4] = new_height / (
                self.specs_array[3] * (4 * np.log(2) / np.pi) ** 0.5 / self.specs_array[1] + (
                    1 - self.specs_array[3]) * 2 / (np.pi * self.specs_array[1])
        )

    @property
    def fwhm_asym(self):
        """ real fwhm of an asymmetric peak"""
        fwhm_asym = self.fwhm * (1 + 0.4 * self.asymmetry ** 2 + 1.35 * self.asymmetry ** 4)
        return fwhm_asym

    @property
    def curve(self):
        """ Asymmetric pseudo-Voigt function as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = self.specs_array[4] * voigt_asym(self.wn - self.specs_array[0], self.specs_array[1],
                                                 self.specs_array[2], self.specs_array[3])
        return curve

    @property
    def curve_with_BL(self):
        curve_with_bl = self.curve + self.specs_array[5] * (self.wn - self.specs_array[0]) + self.specs_array[6]
        return curve_with_bl


def fit_single_peak(spectrum: ExpSpec, peak_position: list[int | None] = None,
                    fit_range: list[tuple[float, float] | None] = None, peak_sign: int = 0,
                    fwhm: float | None = None, print_messages: bool = False) -> CalcPeak:
    """ Returns class calc peak
     peak_position: optional starting value.
     fit-range: (x1, x2) is the fitting range
     peak-sign = 1 or -1 for positive or negative peaks
     fwhm by default is 4 inter-point distances (starting value)
     """

    # step 0: initialize the calc peak:
    peak = CalcPeak(spectrum.x)
    # capture the original working range, which has to be restored later:
    original_working_range = spectrum.working_range

    # step 1: check if we need to set x0 and find the index of x0
    if fit_range is not None:  # set to +-4*fwhm
        spectrum.working_range = fit_range
        # logging.info('fitting range from input: %s', spectrum.working_range)

    if peak_position is None:
        # find index of a maximum y:
        if peak_sign != 0:
            idx0 = (peak_sign * detrend(spectrum.y)).argmax()
        else:
            idx0 = np.abs(spectrum.x - peak_position).argmin()
            peak_sign = np.sign(detrend(spectrum.y)[idx0])
            # logging.info('peak sign detected as %s', peak_sign.working_range)
        peak.position = spectrum.x[idx0]
        # logging.info('x0 set by default to %s', peak.position)
    else:
        peak.position = peak_position
        if peak_sign == 0:
            idx0 = np.abs(spectrum.x - peak_position).argmin()
            peak_sign = np.sign(detrend(spectrum.y)[idx0])
            # logging.info('peak sign detected as %s', peak_sign)

    # step 2: set initial value of fwhm from input or to 5 points (4 inter-point distances)
    inter_point_distance = (spectrum.x[-1] - spectrum.x[0]) / (len(spectrum.x) - 1)
    if fwhm is None:
        peak.fwhm = abs(4 * inter_point_distance)
        # logging.info('fwhm is set to 5 points: {} cm\N{superscript minus}\N{superscript one} %s', peak.fwhm)
    else:
        peak.fwhm = fwhm
        # logging.info('fwhm is set from input: {} cm\N{superscript minus}\N{superscript one} %s', fwhm)

    # step 3: Set initial working range
    if fit_range is None:  # set to +-4*fwhm
        idx0 = (np.abs(spectrum.x - peak.position)).argmin()  # find point number for the closest to x0 point:
        spectrum.working_range = (spectrum.x[idx0] - 4 * peak.fwhm, spectrum.x[idx0] + 4 * peak.fwhm)
        # logging.info('fitting range by default: %s', (np.min(spectrum.x), np.max(spectrum.x)))
        # logging.info('fitting range calculated: %s', spectrum.working_range)

    # step 4: Set fitting range
    peak.wn = spectrum.x

    # step 5: Set other starting values
    peak.voigt_amplitude = 0
    peak.asymmetry = 0
    starting_point = np.zeros(7)
    bounds_high = np.full_like(starting_point, np.inf)
    bounds_low = np.full_like(starting_point, -np.inf)
    bounds_low[2] = -0.36
    bounds_high[2] = 0.36  # asymmetry
    bounds_low[3] = 0
    bounds_high[3] = 1  # Gaussian share
    while True:
        # 1: find index of a y_max, y_min within the fitting range,
        idx0local = (np.abs(spectrum.x - peak.position)).argmin()
        peak_height = peak_sign * (np.abs((detrend(spectrum.y))[idx0local]))
        y_min_local = spectrum.y[idx0local] - peak_height
        # logging.info('starting peak position = %s, peak_height = %g', spectrum.x[idx0local], peak_height)
        # logging.info('starting FWHM = ' + str(peak.fwhm) + ', fitting range = ' + str(spectrum.working_range))
        peak.peak_height = peak_height
        starting_point = peak.specs_array
        starting_point[5] = 0  # always start next round with the flat baseline
        starting_point[6] = y_min_local

        # 2: set bounds for parameters:
        # position: x0 +- 2 * fwhm (0)
        bounds_low[0] = peak.position - np.sign(spectrum.x[-1] - spectrum.x[0]) * peak.fwhm * 2
        bounds_high[0] = peak.position + np.sign(spectrum.x[-1] - spectrum.x[0]) * peak.fwhm * 2
        # fwhm: 0.25-8x fwhm (0)
        bounds_low[1] = 0.25 * peak.fwhm
        if fwhm is None:
            bounds_high[1] = abs((spectrum.x[-1] - spectrum.x[0])) / 2
        else:
            bounds_high[1] = 8 * peak.fwhm
        # amplitude depending on sign:
        if peak_sign > 0:
            bounds_low[4] = 0
        elif peak_sign < 0:
            bounds_high[4] = 0
        try:

            solution = least_squares(func2min, starting_point, args=(spectrum.x, spectrum.y, peak_sign),
                                     bounds=[bounds_low, bounds_high], ftol=1e-5, xtol=1e-5, gtol=1e-5)
            peak.specs_array = solution.x
            # logging.info('least_squares converged')
            break
        except RuntimeError:
            # logging.info('least_squares optimization error, expanding the fitting range')
            spectrum.working_range = (
                spectrum.working_range[0] - inter_point_distance, spectrum.working_range[1] + inter_point_distance)
            peak.wn = spectrum.x
            continue
    # if print_messages:
    #     the_baseline = (peak.BL_offset + peak.BL_slope * (spectrum.x - peak.position))
    #     matplotlib.pyplot.plt.plot(spectrum.x, spectrum.y, 'ko',
    #              spectrum.x, peak.curve_with_BL, 'r:',
    #              spectrum.x, the_baseline, 'k-', mfc='none')
    #     matplotlib.pyplot.plt.title('fit single peak')
    #     matplotlib.pyplot.plt.show()
    # logging.info('position: %s', peak.position)
    # logging.info('fwhm: %s', peak.fwhm)
    # logging.info('peak area: %s', peak.peak_area)
    # logging.info('peak asymmetry: %s', peak.asymmetry)
    # logging.info('peak Gaussian share: %s', peak.Gaussian_share)
    # logging.info('peak slope: %s', peak.BL_slope)
    # logging.info('peak offset: %s', peak.BL_offset)
    # logging.info('peak height: %s', peak.peak_height)
    # logging.info('peak Voigt_amplitude: %s', peak.voigt_amplitude)

    # restoring the working range of the ExpSpec class:
    spectrum.working_range = original_working_range
    return peak


@njit(float64[:](float64[:], float64, float64, float64), fastmath=True)
def voigt_asym(x: list[float], fwhm: float, asymmetry: float, gaussian_share: float) -> list[float]:
    """ returns pseudo-voigt profile composed of Gaussian and Lorentzian,
            which would be normalized by unit area if symmetric
        The function as defined in Analyst: 10.1039/C8AN00710A
        """
    x_distorted = x * (1 - np.exp(-x ** 2 / (2 * (2 * fwhm) ** 2)) * asymmetry * x / fwhm)
    lor_asym = fwhm / (x_distorted ** 2 + fwhm ** 2 / 4) / 6.2831853
    gauss_asym = 0.9394373 / fwhm * np.exp(-(x_distorted ** 2 * 2.7725887) / fwhm ** 2)
    result = (1 - gaussian_share) * lor_asym + gaussian_share * gauss_asym
    return result


def moving_average_molification(raw_spectrum: np.ndarray, struct_el: int = 7) -> np.ndarray:
    molifier_kernel = np.ones(struct_el) / struct_el
    denominator = np.convolve(np.ones_like(raw_spectrum), molifier_kernel, 'same')
    smooth_line = np.convolve(raw_spectrum, molifier_kernel, 'same') / denominator
    return smooth_line


@njit
def func2min(peak_params: list[float], *args):
    # 0-x0, 1-fwhm, 2-asymmetry, 3-Gaussian_share, 4-voigt_amplitude, 5-baseline_k, 6-baseline_y0):
    x_diff = args[0] - peak_params[0]
    voigt = voigt_asym(x_diff, peak_params[1], peak_params[2], peak_params[3])
    # print(peak_params[0])
    the_diff = args[1] - (peak_params[4] * voigt + peak_params[5] * x_diff + peak_params[6])
    der_func = the_diff * np.exp(0.1 * (1 - args[2] * np.sign(the_diff)) ** 2)
    return der_func
