from numpy import ndarray, vstack, inf, minimum, abs, std, power, zeros_like, diff, eye, ones, mean, exp, matrix, \
    array, sum as np_sum, linalg
from numpy.linalg import norm
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyval
from pybaselines import Baseline
from scipy.sparse import csc_matrix, spdiags, diags, eye as sparse_eye
from scipy.sparse.linalg import spsolve


def baseline_poly(item: tuple[str, ndarray], params: int) -> tuple[str, ndarray, ndarray]:
    """Computes a polynomial that fits the baseline of the data. """
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    poly_order = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.poly(y_input, poly_order)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_modpoly(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    """Implementation of Modified polyfit method from paper: Automated Method for Subtraction of Fluorescence from
    Biological Raman Spectra, by Lieber & Mahadevan-Jansen (2003)

    degree: Polynomial degree, default is 6

    max_iter: How many iterations to run. Default is 100

    gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration.
    If gain in any iteration is less than this, further improvement will stop
    """
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]

    # initial improvement criteria is set as positive infinity, to be replaced later on with actual value
    criteria = inf
    baseline_fitted = y_input
    y_old = y_input
    n_iter = 0

    while (criteria >= gradient) and (n_iter <= max_iter):
        poly_coef = polynomial.polyfit(x_input, y_old, degree)
        baseline_fitted = polyval(x_input, poly_coef)
        y_work = minimum(y_input, baseline_fitted)
        criteria = sum(abs((y_work - y_old) / y_old))
        y_old = y_work
        n_iter += 1
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_imodpoly(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    """IModPoly from paper: Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman
     Spectroscopy, by Zhao, Jianhua, Lui, Harvey, McLean, David I., Zeng, Haishan (2007)

    degree: Polynomial degree, default is 2

    max_iter: How many iterations to run. Default is 100

    gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration. If gain
     in any iteration is less than this, further improvement will stop
    """
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    y = y_input
    x = x_input
    n_iter = 0
    dev = 0
    while n_iter < max_iter:
        n_iter += 1
        previous_dev = dev
        poly_coef = polynomial.polyfit(x, y, degree)
        baseline_fitted = polyval(x, poly_coef)
        di = y - baseline_fitted
        dev = std(di)
        if n_iter == 1:
            y = y[y_input <= (baseline_fitted + dev)]
            x = x[y_input <= (baseline_fitted + dev)]
            baseline_fitted = baseline_fitted[y_input <= (baseline_fitted + dev)]
        stop_criteria = abs((dev - previous_dev) / dev)
        print(stop_criteria, dev, n_iter)
        if stop_criteria < gradient:
            break
        y = minimum(y, baseline_fitted + dev)
    poly_coef = polynomial.polyfit(x, y, degree)
    baseline_fitted = polyval(x_input, poly_coef)
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_imodpoly_plus(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    """IModPoly from paper: Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman
    Spectroscopy, by Zhao, Jianhua, Lui, Harvey, McLean, David I., Zeng, Haishan (2007)

    degree: Polynomial degree, default is 2

    max_iter: How many iterations to run. Default is 100

    gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration.
    If gain in any iteration is less than this, further improvement will stop
    """

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    y = y_input
    x = x_input
    n_iter = 0
    dev = 0
    while n_iter < max_iter:
        n_iter += 1
        previous_dev = dev
        poly_coef = polynomial.polyfit(x, y, degree)
        baseline_fitted = polyval(x, poly_coef)
        di = y - baseline_fitted
        residual = di * power(std(di) / 1000, (1 - (abs(di) / di)))
        dev = std(residual)
        if n_iter == 1:
            y = y[y_input <= (baseline_fitted + dev)]
            x = x[y_input <= (baseline_fitted + dev)]
            baseline_fitted = baseline_fitted[y_input <= (baseline_fitted + dev)]
        stop_criteria = abs((dev - previous_dev) / dev)
        if stop_criteria < gradient or dev < 1e-06:
            break
        y = minimum(y, baseline_fitted + dev)
    poly_coef = polynomial.polyfit(x, y, degree)
    baseline_fitted = polyval(x_input, poly_coef)
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_penalized_poly(item: tuple[str, ndarray], params: list[int, float, int, float, str]) \
        -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    alpha_factor = params[3]
    cost_function = params[4]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.penalized_poly(y_input, degree, gradient, max_iter, cost_function=cost_function,
                                                     alpha_factor=alpha_factor)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_loess(item: tuple[str, ndarray], params: list[int, float, int, float, int]) \
        -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    fraction = params[3]
    scale = params[4]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.loess(y_input, fraction, poly_order=degree, tol=gradient, max_iter=max_iter,
                                            scale=scale)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_quant_reg(item: tuple[str, ndarray], params: list[int, float, int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    quantile = params[3]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.quant_reg(y_input, degree, quantile, gradient, max_iter)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_goldindec(item: tuple[str, ndarray], params: list[int, float, int, str, float, float]) \
        -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    degree = params[0]
    gradient = params[1]
    max_iter = params[2]
    cost_func = params[3]
    peak_ratio = params[4]
    alpha_factor = params[5]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.goldindec(y_input, degree, gradient, max_iter, cost_function=cost_func,
                                                peak_ratio=peak_ratio, alpha_factor=alpha_factor)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_asls(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    # Matlab code in Eilers et Boelens 2005
    # Python adaptation found on stackoverflow:
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    niter = params[2]
    z = zeros_like(y_input)
    # starting the algorithm
    y_shape = y_input.shape[0]
    d = csc_matrix(diff(eye(y_shape), 2))
    w = ones(y_shape)
    for i in range(niter):
        w_w = spdiags(w, 0, y_shape, y_shape)
        z_z = w_w + lam * d.dot(d.transpose())
        z = spsolve(z_z, w * y_input)
        w = p * (y_input > z) + (1 - p) * (y_input < z)

    baseline_fitted = z
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_iasls(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.iasls(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_arpls(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    # Adaptation of the Matlab code in Baek et al. 2015 DOI: 10.1039/c4an01061b

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    # optional parameters
    lam = params[0]
    ratio = params[1]
    n_iter = params[2]

    n = y_input.shape[0]
    d_d = csc_matrix(diff(eye(n), 2))
    w = ones(n)
    i = 0
    while True:
        i += 1
        w_w = spdiags(w, 0, n, n)
        z_z = w_w + lam * d_d.dot(d_d.transpose())
        z = spsolve(z_z, w * y_input)
        d = y_input - z
        # make d- and get w^t with m and s
        dn = d[d < 0]
        m = mean(dn)
        s = std(dn)
        wt = 1.0 / (1 + exp(2 * (d - (2 * s - m)) / s))
        # check exit condition and backup
        if norm(w - wt) / norm(w) < ratio:
            break
        if i >= n_iter:
            break
        w = wt
    baseline_fitted = z
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def _whittaker_smooth(x, w, lambda_):
    """
        Penalized the least squares algorithm for background fitting

        input
            x: input data (i.e. chromatogram of spectrum)
            w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
            lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting
             background

        output
            the fitted background vector
    """
    x = matrix(x)
    m = x.size
    e = sparse_eye(m, format='csc')
    d = e[1:] - e[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    w = diags(w, shape=(m, m))
    a = csc_matrix(w + (lambda_ * d.T * d))
    b = csc_matrix(w * x.T)
    background = spsolve(a, b)
    return array(background)


def baseline_airpls(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    """
    Implementation of Zhang fit for Adaptive iteratively reweighted penalized the least squares for baseline fitting.
    Modified from Original implementation by Professor Zhimin Zhang at https://github.com/zmzhang/airPLS/
    https://doi.org/10.1039/B922045C

    lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z

    max_iter: how many iterations to run, and default value is 15.
    """

    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitted = []
    y_len = y_input.shape[0]
    w = ones(y_len)
    for i in range(1, max_iter + 1):
        baseline_fitted = _whittaker_smooth(y_input, w, lam)
        d = y_input - baseline_fitted
        dssn = abs(d[d < 0].sum())
        if dssn < p * (np_sum(abs(y_input))):
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = exp(i * abs(d[d < 0]) / dssn)
        w[0] = exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]

    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_drpls(item: tuple[str, ndarray], params: list[int, float, int, float]) -> tuple[str, ndarray, ndarray]:
    # according to Applied Optics, 2019, 58, 3913-3920. https://doi.org/10.1364/AO.58.003913
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    ratio = params[1]
    niter = params[2]
    eta = params[3]
    niter = 100 if niter > 100 else niter
    # optional smoothing in the next line, currently commented out
    # y = np.around(savgol_filter(raw_data,19,2,deriv=0,axis=1),decimals=6)

    y_shape = y_input.shape[0]

    d_d = diags([1, -2, 1], [0, -1, -2], shape=(y_shape, y_shape - 2), format='csr')
    d_d = d_d.dot(d_d.transpose())
    d_1 = diags([-1, 1], [0, -1], shape=(y_shape, y_shape - 1), format='csr')
    d_1 = d_1.dot(d_1.transpose())

    w_0 = ones(y_shape)
    i_n = diags(w_0, format='csr')

    # this is the code for the fitting procedure
    w = w_0
    w_w = diags(w, format='csr')
    z_z = w_0

    for jj in range(int(niter)):
        w_w.setdiag(w)
        z_prev = z_z
        z_z = spsolve(w_w + d_1 + lam * (i_n - eta * w_w) * d_d, w_w * y_input, permc_spec='NATURAL')
        if linalg.norm(z_z - z_prev) > ratio:
            d = y_input - z_z
            d_negative = d[d < 0]
            sigma_negative = std(d_negative)
            mean_negative = mean(d_negative)
            w = 0.5 * (1 - exp(jj) * (d - (-mean_negative + 2 * sigma_negative)) / sigma_negative / (
                    1 + abs(exp(jj) * (d - (-mean_negative + 2 * sigma_negative)) / sigma_negative)))
        else:
            break
    # end of fitting procedure

    baseline_fitted = z_z
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_iarpls(item: tuple[str, ndarray], params: list[int, int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    tol = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.iarpls(y_input, lam, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_aspls(item: tuple[str, ndarray], params: list[int, int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    tol = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.aspls(y_input, lam, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_psalsa(item: tuple[str, ndarray], params: list[int, int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.psalsa(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_derpsalsa(item: tuple[str, ndarray], params: list[int, int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.derpsalsa(y_input, lam, p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_mpls(item: tuple[str, ndarray], params: tuple[int, float, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    lam = params[0]
    p = params[1]
    max_iter = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mpls(y_input, lam=lam, p=p, max_iter=max_iter, tol=1e-6)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_mor(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mor(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_imor(item: tuple[str, ndarray], params: list[int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    max_iter = params[0]
    tol = params[1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.imor(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_mormol(item: tuple[str, ndarray], params: list[int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    max_iter = params[0]
    tol = params[1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mormol(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_amormol(item: tuple[str, ndarray], params: list[int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    # optional parameters
    max_iter = params[0]
    tol = params[1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.amormol(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_rolling_ball(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.rolling_ball(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_mwmv(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mwmv(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_tophat(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.tophat(y_input)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_mpspline(item: tuple[str, ndarray], params: tuple[int, float, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    lam = params[0]
    p = params[1]
    spline_degree = params[2]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mpspline(y_input, lam=lam, p=p, spline_degree=spline_degree)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_jbcd(item: tuple[str, ndarray], params: list[int, float]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    max_iter = params[0]
    tol = params[1]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.jbcd(y_input, max_iter=max_iter, tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_mixture_model(item: tuple[str, ndarray], params: list[int, float, int, int, float]) \
        -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    lam = params[0]
    p = params[1]
    spline_degree = params[2]
    max_iter = params[3]
    tol = params[4]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.mixture_model(y_input, lam, p, spline_degree=spline_degree, max_iter=max_iter,
                                                    tol=tol)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_irsqr(item: tuple[str, ndarray], params: list[int, float, int, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]
    lam = params[0]
    quantile = params[1]
    spline_degree = params[2]
    max_iter = params[3]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.irsqr(y_input, lam, quantile, spline_degree=spline_degree, max_iter=max_iter)[0]
    # end of fitting procedure
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_corner_cutting(item: tuple[str, ndarray], params: int) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    max_iter = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.corner_cutting(y_input, max_iter)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_noise_median(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.noise_median(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_snip(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.snip(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_swima(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.swima(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_ipsa(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.ipsa(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_ria(item: tuple[str, ndarray], params: float) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    tol = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.ria(y_input, tol=tol)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_dietrich(item: tuple[str, ndarray], params: list[float, int, float, int, int, int]) \
        -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    num_std = params[0]
    poly_order = params[1]
    tol = params[2]
    max_iter = params[3]
    interp_half_window = params[4]
    min_length = params[5]
    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.dietrich(y_input, num_std=num_std, poly_order=poly_order, tol=tol,
                                               max_iter=max_iter, interp_half_window=interp_half_window,
                                               min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_golotvin(item: tuple[str, ndarray], params: list[float, int, int, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    num_std = params[0]
    interp_half_window = params[1]
    min_length = params[2]
    sections = params[3]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.golotvin(y_input, num_std=num_std, interp_half_window=interp_half_window,
                                               sections=sections, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_std_distribution(item: tuple[str, ndarray], params: list[float, int, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    num_std = params[0]
    interp_half_window = params[1]
    fill_half_window = params[2]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.std_distribution(y_input, num_std=num_std,
                                                       interp_half_window=interp_half_window,
                                                       fill_half_window=fill_half_window)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_fastchrom(item: tuple[str, ndarray], params: list[int, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    interp_half_window = params[0]
    max_iter = params[1]
    min_length = params[2]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.fastchrom(y_input, interp_half_window=interp_half_window,
                                                max_iter=max_iter, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_fabc(item: tuple[str, ndarray], params: list[int, float, int]) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    lam = params[0]
    num_std = params[1]
    min_length = params[2]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.fabc(y_input, lam, num_std=num_std, min_length=min_length)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_optimize_extended_range(item: tuple[str, ndarray], params: str) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    method = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.optimize_extended_range(y_input, method=method)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_adaptive_minmax(item: tuple[str, ndarray], params: str) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    method = params

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.adaptive_minmax(y_input, method=method)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T


def baseline_beads(item: tuple[str, ndarray], _) -> tuple[str, ndarray, ndarray]:
    key = item[0]
    input_array = item[1]
    x_input = input_array[:, 0]
    y_input = input_array[:, 1]

    baseline_fitter = Baseline(x_data=x_input)
    baseline_fitted = baseline_fitter.beads(y_input)[0]
    y_new = y_input - baseline_fitted
    return key, vstack((x_input, baseline_fitted)).T, vstack((x_input, y_new)).T
