import numpy as np
from numba import njit, float64, int64, int32
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


def get_minmax(snr: float) -> tuple[int, int]:
    if snr <= 200:
        min_m = np.round(np.polyval(const1, snr))
        max_m = np.round(np.polyval(const2, snr))
    else:
        min_m = 2
        max_m = 5
    return min_m, max_m


def calculate_m(y_axis: np.ndarray, x_axis: np.ndarray, peaks_idx: list[int], g_sigma: int,
                snr=None, min_m=None, max_m=None, mu=None) -> np.ndarray:
    g_sigma = g_sigma * (x_axis[1] - x_axis[0])

    if mu is None:
        mu = 0
    y_axis = detrend(y_axis)
    if snr is not None:
        min_m, max_m = get_minmax(snr)
    else:
        filtered = savgol_filter(y_axis - mu, 9, 3)
        noise = y_axis - mu - filtered
        snr = (y_axis - mu).max() / noise.std()
        min_m, max_m = get_minmax(snr)

    n = x_axis.shape[0]
    g = np.zeros((len(peaks_idx), n))

    for i in range(len(peaks_idx)):
        nom = -(x_axis - x_axis[peaks_idx[i]]) ** 2
        denom = 2 * g_sigma * g_sigma
        g[i, :] = 1 - np.exp(nom / denom)
    m = g.min(axis=0)
    result = m * (max_m - min_m) + min_m
    result_rounded = np.round(result).astype(int)
    return result_rounded


# @njit(float64[::1](float64[::1], int64, int32[::1], int64, float64[::1], float64[::1], float64, float64, float64),
#       fastmath=True)
@njit(cache=True, fastmath=True)
def main_job(x: list[float], n: int, m: list[int], j: int, xe: list[float], x_dash: list[float], sigma: float,
             p: float, lmbd: float) -> list[float]:

    for i in range(n):
        if j >= m[i]:
            continue
        mle_range = np.arange(xe[i] - 3 * sigma,
                              xe[i] + 3 * sigma,
                              0.06 * sigma)
        le_mle_range = np.zeros(mle_range.size)
        for k in range(mle_range.size):
            tmp1 = np.abs(mle_range[k] - x_dash[i]) ** p
            limit1 = lmbd * tmp1
            tmp2_nom = (x[i] - mle_range[k]) ** 2
            tmp2_denom = 2 * sigma * sigma
            limit2 = tmp2_nom / tmp2_denom
            le_mle_range[k] = limit1 + limit2

        b = le_mle_range.argmin()
        xe[i] = mle_range[b]
    return xe


def MLESG_core(y_axis: np.ndarray, m: np.ndarray, v: int = 5, q: int = 7, lmbd: float = 1.8, p: float = 0.4,
               mu: int = None) -> np.ndarray:
    if mu is None:
        mu = 0

    x = y_axis - mu
    n = x.size
    temp = savgol_filter(x, 9, 3)
    noise = x - temp
    sigma = noise.std()
    xe = x

    for j in range(m.max()):
        if j < min(m):
            v = 5
            q = 7
        elif j >= m.max() - round(m.max() / 5):
            q += 4
            lmbd *= 10
        else:
            v = 3
            q = 5
        x_dash = savgol_filter(xe, q, v)
        xe = main_job(x, n, m, j, xe, x_dash, sigma, p, lmbd)

    return xe


def mlesg(y_axis: np.ndarray, x_axis: np.ndarray, peaks_idx: list[int], snr: float = None,
          g_sigma: int = 5) -> np.ndarray:
    m = calculate_m(y_axis, x_axis, peaks_idx, g_sigma, snr=snr)
    xe = MLESG_core(y_axis, m)
    return xe

