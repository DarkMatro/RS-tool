# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides a function to interpolate data points to match a given reference x-axis.

The primary function in this module is:
- interpolate: Interpolates the y-values of a given set of data points to a new set of x-values
    provided in a reference file.
"""

import numpy as np


def interpolate(item: tuple[str, np.ndarray], ref_file: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Interpolates the given item's data to match the reference file's x-axis values.

    Parameters
    ----------
    item : tuple[str, np.ndarray]
        A key-value pair where the key is a string identifier and the value is a 2D numpy array.
        The numpy array's first column represents the x-axis values and the second column
        represents the y-axis values.
    ref_file : np.ndarray
        A 1D numpy array representing the new x-axis values to interpolate to.

    Returns
    -------
    tuple[str, np.ndarray]
        A tuple containing the item's key and a 2D numpy array. The 2D numpy array has the
        new x-axis values (from ref_file) in the first column and the interpolated y-axis
        values in the second column.

    Examples
    --------
    >>> item = ("spectrum1", np.array([[1, 2], [2, 4], [3, 6]]))
    >>> ref_file = np.array([1.5, 2.5, 3.5])
    >>> interpolate(item, ref_file)
    ('spectrum1', array([[1.5, 3. ],
           [2.5, 5. ],
           [3.5, 6. ]]))
    """
    x_axis_old, y_axis_old = item[1][:, 0], item[1][:, 1]
    y_axis_new = np.interp(ref_file, x_axis_old, y_axis_old)
    new_spectrum = np.vstack((ref_file, y_axis_new)).T
    return item[0], new_spectrum
