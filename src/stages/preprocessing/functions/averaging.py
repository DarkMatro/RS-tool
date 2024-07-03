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
    """
    assert spectra
    assert method in ['Mean', 'Median']
    x_axis = spectra[0][:, 0]
    y_axes = [spectrum[:, 1] for spectrum in spectra]
    y_axes = np.array(y_axes)
    np_y_axis = np.mean(y_axes, axis=0) if method == 'Mean' else np.median(y_axes, axis=0)
    return np.vstack((x_axis, np_y_axis)).T
