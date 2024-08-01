"""
This module contains various utility functions for numerical computations and data manipulation
using NumPy and Numba.

Functions
---------
- nearest_idx(array: np.ndarray, value: float) -> int
  Find the index of the value in the array that is nearest to the given value.

- find_nearest(array: np.ndarray, value: float, take_left_value: bool = False) -> float
  Find the value in the array that is nearest to the given value, with an option to select the
    left-side value.

- normalize_between_0_1(x: np.ndarray) -> np.ndarray
  Normalize the values in the input array to be within the range [0, 1].

- diff(x: np.ndarray) -> np.ndarray
  Calculate the first discrete difference of the input array.

- extend_bool_mask(x: list[bool], window_size: int = 2) -> list[bool]
  Extend False values to True if they are within a specified window size of True values in the list.

- extend_bool_mask_two_sided(x: list[int], window_size: int = 2) -> list[int]
  Extend False values to True if they are within a specified window size of True values in the list,
    considering both sides.

- find_nearest_by_idx(array: np.ndarray, idx: int, take_left_value: bool = False) -> float
  Find the nearest value in the array to the value at a specified index, with an option to select
    the left-side value.

Notes
-----
- The functions utilize NumPy arrays for efficient computation.
- The `nearest_idx`, `normalize_between_0_1`, and `diff` functions are optimized with Numba's
    Just-In-Time (JIT) compilation for performance improvements.
- Functions `extend_bool_mask` and `extend_bool_mask_two_sided` operate on lists of boolean or
    integer values to modify the data based on surrounding values.
"""

import numpy as np
from numba import njit


@njit(fastmath=True)
def nearest_idx(array: np.ndarray, value: float) -> int:
    """
    nearest_idx(array: np.ndarray, value: float)

    Return an index of value nearest to input value in array

    Parameters
    ---------
    array : 1D ndarray
        Search the nearest value in this array
    value : float
        Input value to compare with values in array

    Returns
    -------
    out : int
        Index of nearest value

    Examples
    --------
    >>> ar = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> nearest_idx(ar, 2.9)
    2
    """
    idx = np.abs(array - value).argmin()
    array_v = np.round(array[idx], 5)
    value_r = np.round(value, 5)
    if np.abs(array_v - value_r) > 1 and idx != 0:
        return idx - 1
    else:
        return idx


def find_nearest(array: np.ndarray, value: float, take_left_value: bool = False) -> float:
    """
    find_nearest(array: np.ndarray, value: float)

    Return value nearest to input value in array

    Parameters
    ---------
    array : 1D ndarray
        Search the nearest value in this array
    value : float
        Input value to compare with values in array
    take_left_value : bool
        If true take the nearest value from left side, else from right.

    Returns
    -------
    out : float
        Nearest value

    Examples
    --------
    >>> ar = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> find_nearest(ar, 2.9)
    3.0
    >>> find_nearest(ar, 2.9, take_left_value=True)
    2.0
    >>> find_nearest(ar, 2.5, take_left_value=True)
    2.0
    >>> find_nearest(ar, 2.5)
    2.0
    >>> find_nearest(ar, 2.51)
    3.0
    """
    idx = np.abs(array - value).argmin()
    if take_left_value and array[idx] > value and idx != 0:
        return array[idx - 1]
    else:
        return array[idx]


@njit(cache=True, fastmath=True)
def normalize_between_0_1(x: np.ndarray) -> np.ndarray:
    """
    normalize data between 0 and 1

    Parameters
    ---------
    x : 1D ndarray

    Returns
    -------
    out : 1D ndarray

    Examples
    --------
    >>> ar = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> normalize_between_0_1(ar)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    min_x = np.min(x)
    return (x - min_x) / (np.max(x) - min_x)


@njit(cache=True, fastmath=True)
def diff(x: np.ndarray) -> np.ndarray:
    """
    Calculate the 1-th discrete difference along the given axis x.

    The first difference is given by ``out[i] = x[i+1] - x[i]`` along the given axis

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    d : ndarray
        The 1-th differences. The shape of the output is the same as `x`.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> diff(x)
    array([ 0.,  1.,  2.,  3., -7.])
    """
    d = np.zeros(x.shape)
    d[1:] = x[1:] - x[:-1]
    return d


def extend_bool_mask(x: list[bool], window_size: int = 2) -> list[bool]:
    """
    Makes False value with True if near this value +- window_size in the list True stored.

    Parameters
    ----------
    x : array_like of bool
        Input array
    window_size: int
        distance from current cell to find False values

    Returns
    -------
    out : list[bool]

    Examples
    --------
    >>> x = [False, False, False, True, True, True, False, False, False]
    >>> extend_bool_mask(x)
    [False, False, True, True, True, True, False, False, False]

    >>> x = [False, True, False, False, False, True, False, False, False]
    >>> extend_bool_mask(x)
    [True, True, False, False, True, True, False, False, False]
    """
    out = []
    for i in range(len(x) - window_size + 1):
        wind = x[i:i + window_size]
        count_of_true = wind.count(True)
        out.append(count_of_true > 0)
    out.append(x[-1])
    return out


def extend_bool_mask_two_sided(x: list[int], window_size: int = 2) -> list[int]:
    """
    Makes False value with True if near this value +- window_size in the list True stored for both
    sides.

    Parameters
    ----------
    x : array_like of bool
        Input array
    window_size: int
        distance from current cell to find False values

    Returns
    -------
    out : list[bool]

    Examples
    --------
    >>> x = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    >>> extend_bool_mask_two_sided(x, 3)
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

    >>> x = [0, 1, 0, 0, 0, 1, 0, 0, 0]
    >>> extend_bool_mask_two_sided(x)
    [0, 1, 1, 0, 1, 1, 1, 0, 0]
    """
    ans = [0] * len(x)
    for i in range(len(x)):
        if any(x[i - window_size + 1: i + window_size]):
            ans[i] = 1
    return ans


def find_nearest_by_idx(array: np.ndarray, idx: int, take_left_value: bool = False) -> float:
    """
    Find the nearest value in the array to the value at a specified index.

    Parameters
    ----------
    array : np.ndarray
        The input array in which to find the nearest value.
    idx : int
        The index of the value in the array to use as the reference point.
    take_left_value : bool, optional
        If True and the nearest value is greater than the reference value, return the value to the
        left of the nearest value.
        If False, or if the nearest value is less than or equal to the reference value, return the
        nearest value. Default is False.

    Returns
    -------
    float
        The nearest value in the array to the value at the specified index, adjusted based on the
        `take_left_value` flag.
    """
    value = array[idx]
    idx = np.abs(array - value).argmin()
    if take_left_value and array[idx] > value and idx != 0:
        return array[idx - 1]
    return array[idx]
