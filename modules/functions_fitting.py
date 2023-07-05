import numpy as np
from lmfit import Model, Parameters
from lmfit.model import ModelResult


def fitting_model(func_legend: list[tuple]) -> Model:
    """
    Prepare fitting model. Final model is a sum of different line models.

    Parameters
    ---------
    func_legend : list[tuple[func, str]]
        using to create fit Model. func is a line function from spec_functions/ peak_shapes, str is curve legend

    Returns
    -------
    out :
        Model()
    """
    model = None
    for i, f_l in enumerate(func_legend):
        if i == 0:
            model = Model(f_l[0], prefix=f_l[1])
            continue
        model += Model(f_l[0], prefix=f_l[1])
    return model


def split_by_borders(arr: np.ndarray, intervals: list[tuple[int, int]]) -> list[np.ndarray]:
    """
    The function returns sliced array by given interval range.

    Parameters
    ----------
    arr: np.ndarray
        input array to slice

    intervals: list[tuple[int, int]]
        ranges (start, end) indexes

    Returns
    -------
    out : list[np.ndarray]
        arr sliced

    Examples
    --------
    >>> data = [[0., 1.], [1., 1.1], [2., 1.2], [3., 1.3], [4., 1.4], [5., 1.5], [6., 1.4], [7., 1.3], [8., 1.2]]
    >>> arr = np.array(data)
    >>> intervals = [(0, 2), (2, 5), (5, 8)]
    >>> split_by_borders(arr, intervals)
    [array([[0. , 1. ],
           [1. , 1.1]]), array([[2. , 1.2],
           [3. , 1.3],
           [4. , 1.4]]), array([[5. , 1.5],
           [6. , 1.4],
           [7. , 1.3]])]
    """
    x_axis = arr[:, 0]
    y_axis = arr[:, 1]
    arrays_sliced = []
    for idx_start, idx_end in intervals:
        arr = np.vstack((x_axis[idx_start:idx_end], y_axis[idx_start:idx_end])).T
        arrays_sliced.append(arr)
    return arrays_sliced


def fit_model(model: Model, y: np.ndarray, params: Parameters, x, method: str) -> ModelResult:
    """
    Fit model using lmfit

    Parameters
    ---------
    model : Model
        lmfit Model
    y : np.ndarray
         Array of data to be fit.
    params : Parameters,
        Parameters to use in fit (default is None).
    x: np.ndarray
         Array of x_axis.
    method : str
        Name of fitting method to use (default is `'leastsq'`).

    Returns
    -------
    ModelResult
    """
    result = model.fit(y, params, x=x, method=method, n_jobs=-1, nan_policy='omit')
    return result
