import numpy as np
from numba import njit
from src.stages.preprocessing.functions.normalization.EMSC import Kohler


def get_emsc_average_spectrum(item: tuple[str, np.ndarray]) -> np.ndarray:
    y_axes = []
    for arr in item:
        y_axes.append(arr[:, 1])
    np_y_axes = np.array(y_axes)
    np_y_axis = np.mean(np_y_axes, axis=0)
    return np_y_axis


def normalize_emsc(item: tuple[str, np.ndarray], y_axis_mean: np.ndarray, n_pca: int = 8) -> tuple[str, np.ndarray]:
    key = item[0]
    arr = item[1]
    y_axis_new = Kohler(arr[:, 0], arr[:, 1], y_axis_mean, n_components=n_pca)
    return key, np.vstack((arr[:, 0], y_axis_new)).T


@njit(fastmath=True, cache=True)
def normalize_snv(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    key = item[0]
    arr = item[1]
    y_axis = arr[:, 1]
    y_mean = np.mean(y_axis)
    sd = np.std(y_axis)
    y_axis_s = np.subtract(y_axis, y_mean)
    y_axis_sd = np.divide(y_axis_s, sd)
    return key, np.vstack((arr[:, 0], y_axis_sd)).T


@njit(fastmath=True, cache=True)
def normalize_area(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    key = item[0]
    arr = item[1]
    y_axis = arr[:, 1]
    y_axis_pow = np.power(y_axis, 2)
    y_axis_sum = np.sum(y_axis_pow)
    norm = np.sqrt(y_axis_sum)
    y_axis_norm = np.divide(y_axis, norm)
    return key, np.vstack((arr[:, 0], y_axis_norm)).T


@njit(fastmath=True, cache=True)
def normalize_trapz_area(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    key = item[0]
    arr = item[1]
    y_axis = arr[:, 1]
    y_norm = np.trapz(y_axis)
    y_axis_norm = np.divide(y_axis, y_norm)
    return key, np.vstack((arr[:, 0], y_axis_norm)).T


@njit(fastmath=True, cache=True)
def normalize_max(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    key = item[0]
    arr = item[1]
    y_axis = arr[:, 1]
    norm = max(y_axis)
    y_axis_norm = np.divide(y_axis, norm)
    return key, np.vstack((arr[:, 0], y_axis_norm)).T


@njit(fastmath=True, cache=True)
def normalize_minmax(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    key = item[0]
    arr = item[1]
    y_axis = arr[:, 1]
    y_min = np.min(y_axis)
    y_max = max(y_axis)
    norm = y_max - y_min
    y_axis_pre_norm = np.subtract(y_axis, y_min)
    y_axis_norm = np.divide(y_axis_pre_norm, norm)
    return key, np.vstack((arr[:, 0], y_axis_norm)).T
