import numpy as np
from numba import njit
from modules.functions_for_arrays import nearest_idx


@njit(cache=True, fastmath=True)
def cut_spectrum(i: tuple[str, np.ndarray], value_start: float, value_end: float) -> tuple[str, np.ndarray]:
    """
    The function returns name and cutted spectrum in new range from value_start cm-1 to value_end cm-1

    Parameters
    ----------
    i: tuple[str, np.ndarray]
        filename of spectrum and data

    value_start: float
        new range start value (cm-1)

    value_end: float
        new range end value (cm-1)

    Returns
    -------
    tuple with name and cutted array
    """
    return i[0], cut_full_spectrum(i[1], value_start, value_end)


@njit(cache=True, fastmath=True)
def cut_full_spectrum(array: np.ndarray, value_start: float, value_end: float) -> np.ndarray:
    """
    The function returns cutted spectrum in new range from value_start cm-1 to value_end cm-1

    Parameters
    ----------
    array: np.ndarray
        original array to cut

    value_start: float
        new range start value (cm-1)

    value_end: float
        new range end value (cm-1)

    Returns
    -------
    np.ndarray in new range
    """
    x_axis = array[:, 0]
    idx_start = nearest_idx(x_axis, value_start)
    idx_end = nearest_idx(x_axis, value_end)
    result = array[idx_start: idx_end + 1]
    return result


def cut_axis(axis: np.ndarray, value_start: float, value_end: float) -> tuple[np.ndarray, int, int]:
    idx_start = nearest_idx(axis, value_start)
    idx_end = nearest_idx(axis, value_end)
    return axis[idx_start: idx_end + 1], idx_start, idx_end
