# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functions for cutting and trimming spectra data.

Functions:
- cut_spectrum: Cuts the spectrum to a specified range.
- cut_full_spectrum: Cuts a 2D array to a specified range.
- cut_axis: Cuts the x-axis to a specified range and returns the cut axis along with start and end
    indices.
- find_fluorescence_beginning: Finds the beginning of fluorescence in the spectrum data.
- find_first_right_local_minimum: Finds the first local minimum to the right in the spectrum data.
- find_first_left_local_minimum: Finds the first local minimum to the left in the spectrum data.
"""

import numpy as np
from numba import njit
from scipy.signal import argrelmin
from src.data.work_with_arrays import nearest_idx


@njit(cache=True, fastmath=True)
def cut_spectrum(i: tuple[str, np.ndarray], value_start: float, value_end: float) \
        -> tuple[str, np.ndarray]:
    """
    Cuts the spectrum to a specified range.

    Parameters
    ----------
    i : tuple[str, np.ndarray]
        A tuple containing the filename of the spectrum and the data.
    value_start : float
        The new range start value (in cm-1).
    value_end : float
        The new range end value (in cm-1).

    Returns
    -------
    tuple[str, np.ndarray]
        A tuple containing the filename and the cut array.

    Examples
    --------
    >>> filename = "spectrum1"
    >>> data = np.array([[100, 1.2], [200, 2.3], [300, 3.4]])
    >>> cut_spectrum((filename, data), 150, 250)
    ('spectrum1', array([[100, 1.2]]))
    """
    return i[0], cut_full_spectrum(i[1], value_start, value_end)


@njit(cache=True, fastmath=True)
def cut_full_spectrum(array: np.ndarray, value_start: float, value_end: float) -> np.ndarray:
    """
    Cuts a 2D array to a specified range.

    Parameters
    ----------
    array : np.ndarray
        The original 2D array to cut.
    value_start : float
        The new range start value (in cm-1).
    value_end : float
        The new range end value (in cm-1).

    Returns
    -------
    np.ndarray
        A 2D array in the new range.

    Examples
    --------
    >>> array = np.array([[100, 1.2], [200, 2.3], [300, 3.4]])
    >>> cut_full_spectrum(array, 150, 250)
    array([[100. , 1.2])
    """
    x_axis = array[:, 0]
    idx_start = nearest_idx(x_axis, value_start)
    idx_end = nearest_idx(x_axis, value_end)
    result = array[idx_start: idx_end + 1]
    return result


def cut_axis(axis: np.ndarray, value_start: float, value_end: float) -> tuple[np.ndarray, int, int]:
    """
    Cuts the x-axis to a specified range and returns the cut axis along with start and end indices.

    Parameters
    ----------
    axis : np.ndarray
        The original x-axis array.
    value_start : float
        The new range start value (in cm-1).
    value_end : float
        The new range end value (in cm-1).

    Returns
    -------
    tuple[np.ndarray, int, int]
        A tuple containing the cut x-axis array, the start index, and the end index.

    Examples
    --------
    >>> axis = np.array([100, 150, 200, 250, 300])
    >>> cut_axis(axis, 140, 260)
    (array([100, 150, 200]), 0, 2)
    """
    idx_start = nearest_idx(axis, value_start)
    idx_end = nearest_idx(axis, value_end)
    return axis[idx_start: idx_end + 1], idx_start, idx_end


def find_fluorescence_beginning(item: tuple[str, np.ndarray], factor: int = 1) -> int:
    """
    Finds the beginning of fluorescence in the spectrum data.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A tuple containing the filename and the data array.
    factor : int, optional
        The factor for determining the end of the gradient increase (default is 1).

    Returns
    -------
    int
        The index where fluorescence begins.

    Examples
    --------
    >>> filename = "spectrum1"
    >>> data = np.array([[0, 1], [50, 2], [100, 1], [150, 5], [200, 7], [250, 4], [300, 3]])
    >>> find_fluorescence_beginning((filename, data))
    4
    """
    idx_100cm = nearest_idx(item[1][:, 0], 100.)
    part_of_y_axis = item[1][:, 1][idx_100cm:]
    grad = np.gradient(part_of_y_axis)
    grad_std = np.std(grad)
    start_of_grow = np.argmax(grad > grad_std)
    negative_grads = np.argwhere(grad[start_of_grow:] < 0)
    f = 0
    end_of_grow_idx = None
    for j, v in enumerate(negative_grads):
        current_idx = v[0]
        prev_idx = negative_grads[j - 1][0]
        if current_idx - prev_idx == 1:
            f += 1
        else:
            f = 0
        if f == factor - 1:
            end_of_grow_idx = current_idx
            break
    return idx_100cm + start_of_grow + end_of_grow_idx


def find_first_right_local_minimum(i: tuple[str, np.ndarray]) -> int:
    """
    Finds the first local minimum to the right in the spectrum data.

    Parameters
    ----------
    i : tuple[str, np.ndarray]
        A tuple containing the filename and the data array.

    Returns
    -------
    int
        The index of the first local minimum to the right.

    Examples
    --------
    >>> filename = "spectrum1"
    >>> data = np.array([[0, 5], [50, 3], [100, 1], [150, 4], [200, 2]])
    >>> find_first_right_local_minimum((filename, data))
    2
    """
    y = i[1][:, 1]
    y_min = argrelmin(y)[0][-1]
    return y_min


def find_first_left_local_minimum(i: tuple[str, np.ndarray]) -> int:
    """
    Finds the first local minimum to the left in the spectrum data.

    Parameters
    ----------
    i : tuple[str, np.ndarray]
        A tuple containing the filename and the data array.

    Returns
    -------
    int
        The index of the first local minimum to the left.

    Examples
    --------
    >>> filename = "spectrum1"
    >>> data = np.array([[0, 5], [50, 3], [100, 1], [150, 4], [200, 2]])
    >>> find_first_left_local_minimum((filename, data))
    2
    """
    y = i[1][:, 1]
    y_min = argrelmin(y)[0][0]
    return y_min
