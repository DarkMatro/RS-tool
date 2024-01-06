import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
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
    return (x - np.min(x)) / (np.max(x) - np.min(x))


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


@njit(cache=True, fastmath=True)
def extend_bool_false_mask(x: list[bool], window_size: int = 2) -> list[bool]:
    """
    Makes True value False if near this value +- window_size in the list False stored.

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
    >>> x = [True, True, True, False, False, False, True, True, True]
    >>> extend_bool_false_mask(x)
    [True, True, False, False, False, False, False, True, True]
    """
    out = []
    out.extend(x[:window_size])
    for i in range(window_size, len(x) - window_size):
        wind = x[i - window_size:i + window_size + 1]
        count_of_false = wind.count(False)
        out.append(count_of_false == 0)
    out.extend(x[-window_size:])
    return out
