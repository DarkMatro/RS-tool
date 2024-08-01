# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functionality to compute the average spectrum from a list of Raman spectra
using either the mean or median method.

Functions
---------
get_average_spectrum(spectra, method='Mean')
    Computes and returns the average spectrum from a list of spectra based on the specified method.

Dependencies
------------
- numpy : Provides support for array operations and mathematical functions.
"""

import numpy as np


def get_average_spectrum(spectra: list[np.ndarray], method: str = 'Mean') -> np.ndarray:
    """
    Returns mean / median spectrum for all spectra

    Parameters
    ----------
    spectra: list[np.ndarray]
       Contains lists of Raman spectra with 2 columns: x - cm-1, y - Intensity
    method: str
        'Mean' or 'Median'

    Returns
    -------
    np.ndarray
       averaged spectrum 2D (x, y)

    Usage Example
    -------------
    >>> import numpy as np
    >>> spectra = [np.array([[100, 1.1], [200, 2.2]]), np.array([[100, 1.2], [200, 2.3]])]
    >>> average_spectrum = get_average_spectrum(spectra, method='Mean')
    >>> print(average_spectrum)
    [[100.     1.15]
     [200.     2.25]]
    """
    assert spectra
    assert method in ['Mean', 'Median']
    x_axis = spectra[0][:, 0]
    y_axes = [spectrum[:, 1] for spectrum in spectra]
    y_axes = np.array(y_axes)
    np_y_axis = np.mean(y_axes, axis=0) if method == 'Mean' else np.median(y_axes, axis=0)
    return np.vstack((x_axis, np_y_axis)).T
