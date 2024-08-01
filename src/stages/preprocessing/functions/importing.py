# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functions for importing and processing spectral data from files.

Functions:
- import_spectrum: Imports spectral data from a file and extracts relevant information.
- get_group_number_from_filename: Extracts the group number from a filename based on specific
    patterns.
- get_result_square_brackets: Helper function to find group numbers enclosed in square brackets.
"""

from pathlib import Path
from re import findall

import numpy as np

from src.stages.preprocessing.functions.converting import get_laser_peak_fwhm


def import_spectrum(file: str, laser_wl: float) \
        -> tuple[str, np.ndarray, int, float, float, float]:
    """
    Imports spectral data from a file and extracts relevant information.

    Parameters
    ----------
    file : str
        The path to the file containing the spectral data.
    laser_wl : float
        The wavelength of the laser in nanometers.

    Returns
    -------
    tuple[str, np.ndarray, int, float, float, float]
        A tuple containing:
        - The basename of the file (str).
        - A 2D numpy array of the spectral data (np.ndarray).
        - The group number extracted from the filename (int).
        - The minimum x-axis value in nanometers (float).
        - The maximum x-axis value in nanometers (float).
        - The full width at half maximum (FWHM) of the laser peak (float).

    Examples
    --------
    >>> file = "1_spectrum.txt"
    >>> laser_wl = 532.0
    >>> import_spectrum(file, laser_wl)
    ('1_spectrum.txt', array([[400.0, 0.1], [450.0, 0.5], [500.0, 0.9]]), 1, 400.0, 500.0, 50.0)
    """
    n_array = np.loadtxt(file)
    basename_of_file = Path(file).name
    group_number = get_group_number_from_filename(basename_of_file)
    x_axis = n_array[:, 0]
    min_nm = np.min(x_axis)

    max_nm = np.max(x_axis)
    if min_nm < laser_wl:
        fwhm = get_laser_peak_fwhm(n_array, laser_wl, min_nm, max_nm)
    else:
        fwhm = 0.
    return basename_of_file, n_array, group_number, min_nm, max_nm, fwhm


def get_group_number_from_filename(basename_with_group_number: str) -> int:
    """
    Extracts the group number from a filename based on specific patterns.

    Parameters
    ----------
    basename_with_group_number : str
        The basename of the file containing the group number.

    Returns
    -------
    int
        The extracted group number.

    Examples
    --------
    >>> get_group_number_from_filename('1_spectrum.txt')
    1
    >>> get_group_number_from_filename('[2]_spectrum.txt')
    2
    """
    result_just_number = findall(r'^\d+', basename_with_group_number)
    result_square_brackets = get_result_square_brackets(basename_with_group_number)
    result_round_brackets = findall(r'^\(\d+\)', basename_with_group_number)
    if result_round_brackets:
        result_round_brackets = findall(r'\d', result_round_brackets[0])

    if result_just_number and result_just_number[0] != '':
        group_number_str = result_just_number[0]
    elif result_square_brackets and result_square_brackets[0] != '':
        group_number_str = result_square_brackets[0]
    elif result_round_brackets and result_round_brackets[0] != '':
        group_number_str = result_round_brackets[0]
    else:
        group_number_str = '0'

    return int(group_number_str)


def get_result_square_brackets(basename_with_group_number: str) -> list:
    """
    Finds group numbers enclosed in square brackets in a filename.

    Parameters
    ----------
    basename_with_group_number : str
        The basename of the file containing the group number in square brackets.

    Returns
    -------
    list
        A list containing the extracted group number as a string.

    Examples
    --------
    >>> get_result_square_brackets('[2]_spectrum.txt')
    ['2']
    >>> get_result_square_brackets('spectrum.txt')
    []
    """
    result_square_brackets = findall(r'^\[\d]', basename_with_group_number)
    if result_square_brackets:
        result_square_brackets = findall(r'\d', result_square_brackets[0])
    return result_square_brackets
