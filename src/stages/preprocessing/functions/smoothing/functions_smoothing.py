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
    array = item[1]
    y_axis_smooth = savgol_filter(array[:, 1], window_length, polyorder)
    return item[0], np.vstack((array[:, 0], y_axis_smooth)).T


def whittaker(item: tuple[str, np.ndarray], lam: int):
    """smooth a signal with the Whittaker smoother

    Inputs
    ------
    y : ndarray
        An array with the values to smooth (equally spaced).

    kwargs
    ------
    Lambda : float
        The smoothing coefficient, the higher the smoother. Default = 10^5.

    Outputs
    -------
    z : ndarray
        An array containing the smoothed values.

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
    # various window filters, from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=smooth
    array = item[1]
    w = np.ones(window_len, 'd')
    y_filt = np.convolve(w / w.sum(), array[:, 1], mode='same')
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_window(item: tuple[str, np.ndarray], window_len: int, method: str) -> tuple[str, np.ndarray]:
    # various window filters, from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=smooth
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
    array = item[1]
    w = np.kaiser(window_len, beta)
    y_filt = np.convolve(w / w.sum(), array[:, 1], mode='same')
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_med_filt(item: tuple[str, np.ndarray], window_len: int) -> tuple[str, np.ndarray]:
    # scipy median filter, from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html#scipy.signal.medfilt
    array = item[1]
    if window_len % 2 == 0:
        window_len += 1
    y_filt = medfilt(array[:, 1], kernel_size=window_len)
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_wiener(item: tuple[str, np.ndarray], window_len: int) -> tuple[str, np.ndarray]:
    # scipy wiener filter, from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html#scipy.signal.wiener
    array = item[1]
    y_filt = wiener(array[:, 1], mysize=window_len)
    return item[0], np.vstack((array[:, 0], y_filt)).T


def smooth_emd(item: tuple[str, np.ndarray], noise_first_imfs: int) -> tuple[str, np.ndarray]:
    # PyEMD, from
    # https://pyemd.readthedocs.io/en/latest/emd.html
    x_axis, y_axis = item[1][:, 0], item[1][:, 1]
    emd = EMD(spline_kind='akima', DTYPE=np.float16)
    emd.emd(y_axis, x_axis, max_imf=noise_first_imfs)
    imfs, _ = emd.get_imfs_and_residue()

    for i in range(noise_first_imfs):
        y_axis = np.subtract(y_axis, imfs[i])
    return item[0], np.vstack((x_axis, y_axis)).T


def smooth_eemd(item: tuple[str, np.ndarray], noise_first_imfs: int, trials: int) -> tuple[str, np.ndarray]:
    # PyEMD, from
    # https://pyemd.readthedocs.io/en/latest/eemd.html
    x_axis, y_axis = item[1][:, 0], item[1][:, 1]
    # Prepare and run EEMD
    x_max = abs(max(x_axis))
    x_range = x_max - abs(min(x_axis))
    noise_width = x_range * 3 / (x_axis.shape[0] - 1) ** 2
    eemd = EEMD(trials=trials, DTYPE=np.float16, spline_kind='akima', noise_width=noise_width, noise_kind='uniform')
    eemd.noise_seed(481516234)
    e_imfs = eemd.eemd(y_axis, x_axis, max_imf=noise_first_imfs)
    for i in range(noise_first_imfs):
        y_axis = np.subtract(y_axis, e_imfs[i])
    return item[0], np.vstack((x_axis, y_axis)).T


def smooth_ceemdan(item: tuple[str, np.ndarray], noise_first_imfs: int, trials: int) \
        -> tuple[str, np.ndarray]:
    # PyEMD, from
    # https://pyemd.readthedocs.io/en/latest/ceemdan.html
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
    # MLESG, from
    # https://github.com/Trel725/RamanDenoising
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

#
# def mlesg_cuda_test(array: ndarray, max_block_dim_x, func_arrange, func_le_mle, temp, x_dash):
#     # not used cause in this case cuda code costs 100ms vs 29ms of CPU njit
#     cols = int(100)
#     y_axis = array[:, 1]
#     noise = y_axis - temp
#     sigma_noise = noise.std()
#     sigma = sigma_noise.astype(float32)
#     y_axis_repeated = np.repeat(y_axis[:, np.newaxis], cols, axis=1)
#     xe = y_axis.astype(np.float32)
#     mle_range = np.zeros_like(y_axis_repeated).astype(np.float32)
#     bdim = (max_block_dim_x, 1, 1)
#     rows = y_axis_repeated.shape[0]
#     x_mul = divmod(rows, bdim[0])[0] + 1
#     gdim = (x_mul * cols, 1, 1)
#
#     func_arrange(drv.Out(mle_range), drv.In(xe), np.float32(sigma), np.int32(cols),
#                  block=bdim, grid=gdim)
#
#     le_mle_range = np.zeros_like(mle_range).astype(np.float32)
#     p = 0.4
#     lmbd = 1.8
#
#     tmp2_denom = 2 * sigma ** 2
#
#     func_le_mle(drv.Out(le_mle_range), drv.In(mle_range), drv.In(x_dash), np.int32(cols), np.float32(p),
#                 np.float32(lmbd), drv.In(y_axis.astype(np.float32)), np.float32(tmp2_denom), block=bdim, grid=gdim)
