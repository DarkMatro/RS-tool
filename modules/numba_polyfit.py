import numpy as np
from numba import njit, float64, int64
import matplotlib.pyplot as plt


@njit(float64[:](float64[:], float64[:], int64), fastmath=True)
def polyfit(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    """
    GOOD FOR polynomial with LOW DEGREE
    Least-squares fit of a polynomial to data.

    Return the coefficients of a polynomial of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit.

    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.

    Examples
    --------

    >>> x_t = np.linspace(0, 2, 20)
    >>> y_t = np.cos(x_t) + 0.3 * 1
    >>> polyfit(x_t, y_t, 3)
    array([ 1.29604443,  0.05561102, -0.64554113,  0.13196941])
    """
    mat = np.zeros(shape=(x.shape[0], deg + 1))
    mat[:, 0] = np.ones_like(x)
    for n in range(1, deg + 1):
        mat[:, n] = x**n
    return np.linalg.lstsq(mat, y)[0]


@njit(float64[:](float64[:], float64[:]), fastmath=True)
def polyval(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial at points x.

    If `c` is of length `n + 1`, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to a ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two-dimensional case the coefficients may be thought
         of as stored in the columns of `c`.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    Notes
    -----
    The evaluation uses Horner's method.

    Examples
    --------
    >>> x_t = np.linspace(0, 2, 20)
    >>> c_t = np.array([ 1.29604443,  0.05561102, -0.64554113,  0.13196941])
    >>> polyval(x_t, c_t)
    array([ 1.29604443,  1.29489933,  1.28037215,  1.25338642,  1.21486568,
            1.16573346,  1.1069133 ,  1.03932873,  0.96390329,  0.88156051,
            0.79322394,  0.6998171 ,  0.60226354,  0.50148678,  0.39841037,
            0.29395784,  0.18905273,  0.08461856, -0.01842112, -0.11914277])
    """
    result = np.zeros_like(x)
    for coef in c[::-1]:
        result = x * result + coef
    return result


@njit('Tuple((float64[:, ::1], float64[:]))(float64[:], int64)', fastmath=True)
def _coef_mat(x: np.ndarray, deg: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create "Coefficient" matrix and scale vector.
    Idea here is to solve ax = b, using the least squares, where 'a' represents our coefficients e.g. x**2, x, constants

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`th term are included in the
        fit.

    Returns
    -------
    mat_ : (M, N) array_like
        "Coefficient" matrix.
    scale_vect : (N, ) array_like

    Examples
    --------

    >>> x_t = np.linspace(0, 2, 5)
    >>> _coef_mat(x_t, 3)
    (array([[0.4472136 , 0.        , 0.        , 0.        ],
           [0.4472136 , 0.18257419, 0.0531494 , 0.01430031],
           [0.4472136 , 0.36514837, 0.2125976 , 0.11440251],
           [0.4472136 , 0.54772256, 0.4783446 , 0.38610848],
           [0.4472136 , 0.73029674, 0.85039041, 0.91522009]]), array([2.23606798, 2.73861279, 4.70372193, 8.74106687]))
    """
    mat_ = np.ones(shape=(x.shape[0], deg + 1))
    mat_[:, 1] = x

    if deg > 1:
        for n in range(2, deg + 1):
            # here, the pow()-function was turned into multiplication, which gave some speedup for me
            # (up to factor 2 for small degrees, which is the normal application case)
            mat_[:, n] = mat_[:, n - 1] * x

    # evaluation of the norm of each column by means of a loop
    scale_vect = np.empty((deg + 1,))
    for n in range(0, deg + 1):
        # evaluation of the column's norm (stored for later processing)
        col_norm = np.linalg.norm(mat_[:, n])
        scale_vect[n] = col_norm
        # scaling the column to unit-length
        mat_[:, n] /= col_norm
    return mat_, scale_vect


@njit(fastmath=True, cache=True)
def _fit_x(a: np.ndarray, b: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Idea here is to solve ax = b, using the least squares, where 'a' represents our coefficients e.g. x**2, x, constants
    Return the least-squares solution to a linear matrix equation.

    Computes the vector `x` that approximately solves the equation
    ``a @ x = b``. The equation may be under-, well-, or over-determined
    (i.e., the number of linearly independent rows of `a` can be less than,
    equal to, or greater than its number of linearly independent columns).


    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    scales : (N, ) array_like
        using for correcting coefficients

    Returns
    -------
    det_ : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.

    Examples
    --------

    >>> x_t = np.linspace(0, 2, 5)
    >>> a_t, scales_t = _coef_mat(x_t, 3)
    >>> y_t = np.cos(x_t) + 0.3
    >>> _fit_x(a_t, y_t, scales_t)
    array([ 1.29953732,  0.04744303, -0.64115007,  0.13169592])
    """
    det_ = np.linalg.lstsq(a, b)[0]
    det_ /= scales  # due to the stabilization, the coefficients have the wrong scale, which is corrected now
    return det_


@njit('float64[:](float64[:], float64[:], int64, float64[:])', fastmath=True)
def fit_poly(x: np.ndarray, y: np.ndarray, deg: int, w: np.ndarray) -> np.ndarray:
    """
    Least-squares fit of a polynomial to data.

    Return the coefficients of a polynomial of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit.
    w : array_like, shape (`M`,), optional
        Weights.

    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.

    Examples
    --------

    >>> x_t = np.linspace(0, 2, 5)
    >>> y_t = np.cos(x_t) + 0.3
    >>> w_t = np.ones_like(x_t)
    >>> fit_poly(x_t, y_t, 3, w_t)
    array([ 1.29953732,  0.04744303, -0.64115007,  0.13169592])
    """
    w = np.asarray(w) + 0.0
    a, scales = _coef_mat(x, deg)
    a = a.T * w
    y = y.T * w
    c = _fit_x(a.T, y.T, scales)
    return c


# Create Dummy Data and use existing numpy polyfit as test
if __name__ == "__main__":
    x_test = np.linspace(0, 2, 20)
    y_test = np.cos(x_test) + 0.3 * np.random.rand(20)
    p = np.poly1d(np.polyfit(x_test, y_test, 3))
    plt.plot(x_test, y_test, 'o')
    t = np.linspace(0, 2, 200)
    plt.plot(t, p(t), '-', color='red', dashes=[2, 2])

    # Now plot using the Numba functions
    p_coef = polyfit(x_test, y_test, deg=3)
    plt.plot(t, polyval(t, p_coef), '-', color='black', dashes=[4, 4])
    plt.show()
