# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functions for Extended Multiplicative Scatter Correction (EMSC) of spectral
data, including Kohler's algorithm and the calculation of scattering extinction values.

Functions
---------
q_ext_kohler(wn, alpha)
    Compute the scattering extinction values for a given alpha and a range of wavenumbers.

apparent_spectrum_fit_function(z_ref, p, b, c, g)
    Function used to fit the apparent spectrum.

kohler(x_axis, y_axis, ref_array, n_components=8)
    Correct scattered spectra using Kohler's algorithm.
"""

import numpy as np
import scipy.optimize
from numba import njit
from sklearn.decomposition import _incremental_pca


@njit(fastmath=True)
def q_ext_kohler(wn: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the scattering extinction values for a given alpha and a range of wavenumbers.

    Parameters
    ----------
    wn : np.ndarray
        Array of wavenumbers.
    alpha : float
        Scalar alpha.

    Returns
    -------
    np.ndarray
        Array of scattering extinctions calculated for alpha in the given wavenumbers.
    """
    rho = alpha * wn
    q = 2.0 - (4.0 / rho) * np.sin(rho) + (2.0 / rho) ** 2.0 * (1.0 - np.cos(rho))
    return q


def apparent_spectrum_fit_function(z_ref: np.ndarray, p: np.ndarray, b: float, c: float,
                                   g: np.ndarray) -> np.ndarray:
    """
    Function used to fit the apparent spectrum.

    Parameters
    ----------
    z_ref : np.ndarray
        Reference spectrum.
    p : np.ndarray
        Principal components of the extinction matrix.
    b : float
        Reference's linear factor.
    c : float
        Offset.
    g : np.ndarray
        Extinction matrix's PCA scores (to be fitted).

    Returns
    -------
    np.ndarray
        Fitting of the apparent spectrum.
    """
    a = b * z_ref + c + np.dot(g, p)  # Extended multiplicative scattering correction formula
    return a


def kohler(x_axis: np.ndarray, y_axis: np.ndarray, ref_array: np.ndarray, n_components: int = 8) \
        -> np.ndarray:
    """
   Correct scattered spectra using Kohler's algorithm.

   Parameters
   ----------
   x_axis : np.ndarray
       Array of wavenumbers.
   y_axis : np.ndarray
       Apparent spectrum.
   ref_array : np.ndarray
       Reference spectrum.
   n_components : int, optional
       Number of principal components to be calculated (default is 8).

   Returns
   -------
   np.ndarray
       Corrected data.
   """
    # Initialize the alpha parameter:
    alpha = np.linspace(3.14, 49.95, 150) * 1.0e-4  # alpha = 2 * pi * d * (n - 1) * wave_number
    p0 = np.ones(2 + n_components)  # Initialize the initial guess for the fitting

    # # Initialize the extinction matrix:
    q_ext = np.zeros((np.size(alpha), np.size(x_axis)))
    for i in range(np.size(alpha)):
        q_ext[i][:] = q_ext_kohler(x_axis, alpha=alpha[i])

    # Perform PCA of Q_ext:
    pca = _incremental_pca.IncrementalPCA(n_components=n_components)
    pca.fit(q_ext)

    def min_fun(x):
        """
        Function to be minimized by the fitting
        :param x: array containing the reference linear factor, the offset, and the PCA scores
        :return: function to be minimized
        """
        bb, cc, g = x[0], x[1], x[2:]
        # Return the squared norm of the difference between the apparent spectrum and the fit
        return np.linalg.norm(y_axis - apparent_spectrum_fit_function(ref_array,
                                                                      pca.components_,
                                                                      bb, cc, g)) ** 2.0

    # Minimize the function using Powell method
    res = scipy.optimize.minimize(min_fun, p0, method='Powell')

    # Apply the correction to the apparent spectrum
    z_corr = np.zeros(np.shape(ref_array))
    for i in range(len(x_axis)):
        sum1 = 0
        for j, g in enumerate(res.x[2:]):
            sum1 += g * pca.components_[j][i]
        z_corr[i] = (y_axis[i] - res.x[1] - sum1) / res.x[0]

    return z_corr  # Return the correction in reverse order for compatibility
